// RpcMemMapper 的实现。
// validate(): 在执行算子前确保其所有张量 buffer 已 mmap 到 DSP，必要时驱逐旧的映射。
// get_tensor_mapping(): 将 ggml_tensor 指针解析为 (fd, offset_in_buffer) 对供 DSP 端寻址。
// unmap_all_pending_buffers(): 批量执行所有延迟的 fastrpc_munmap 请求。

#include "rpcmem_mapper.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <unistd.h>
#include <unordered_set>
#include <vector>

#include "dsprpc_interface.h"
#include "ggml-backend-impl.h"
#include "ggml-htp-impl.h"
#include "ggml-htp.h"
#include "message.h"

namespace {

// 在 hybrid 路径里 view 张量可能还没把 buffer 挂回自己身上，这里统一沿着 view_src 找到底层 backing buffer。
ggml_backend_buffer * get_backing_buffer(const ggml_tensor * tensor) {
    while (tensor != nullptr) {
        if (tensor->buffer != nullptr) {
            return tensor->buffer;
        }
        tensor = tensor->view_src;
    }
    return nullptr;
}

// pending 列表里只有 host 尚未真正 munmap 的条目会继续占用 DSP 映射窗口。
size_t get_pending_host_mapped_size(const std::list<RpcMemMapper::UnmapRequest> & pending_unmap_reqs) {
    size_t total = 0;
    for (const auto & req : pending_unmap_reqs) {
        if (!req.host_unmapped) {
            total += req.buf_size;
        }
    }
    return total;
}

template <typename T>
void write_buf(uint8_t *& p, const T & v) {
    *reinterpret_cast<T *>(p) = v;
    p += sizeof(v);
}

void send_rpcmem_puts_to_dsp(const std::list<RpcMemMapper::UnmapRequest> & pending_unmap_reqs) {
    if (pending_unmap_reqs.empty()) {
        return;
    }

    auto * ctx = ggml_backend_htp_context::instance();
    auto * msg_hdr = reinterpret_cast<MessageHeader *>(ctx->ops_msg_chan);
    GGML_ASSERT(msg_hdr != nullptr);

    const int n_unmap_fds = static_cast<int>(pending_unmap_reqs.size());
    const size_t map_req_size = sizeof(RequestHeader) + sizeof(RpcmemMapRequest) + n_unmap_fds * sizeof(int32_t);

    auto * d_ptr = reinterpret_cast<volatile std::atomic<uint64_t> *>(&(msg_hdr->state.d));
    std::atomic_store(d_ptr, uint64_t{0});

    msg_hdr->n_reqs         = 1;
    msg_hdr->req_offsets[0] = message_header_size(msg_hdr);
    msg_hdr->req_offsets[1] = msg_hdr->req_offsets[0] + map_req_size;

    RequestHeader req_hdr{
        .state = 0,
        .type  = REQUEST_TYPE_RPCMEM_MAP,
    };
    RpcmemMapRequest map_req{
        .n_puts = n_unmap_fds,
        .n_gets = 0,
    };

    auto * p = reinterpret_cast<uint8_t *>(message_header_get_request_ptr(msg_hdr, 0));
    write_buf(p, req_hdr);
    write_buf(p, map_req);
    for (const auto & req : pending_unmap_reqs) {
        write_buf(p, req.fd);
    }

    uint32_t sum = 0;
    uint32_t * begin = reinterpret_cast<uint32_t *>(msg_hdr) + 3;
    uint32_t * end   = reinterpret_cast<uint32_t *>(msg_hdr) + ggml_backend_htp_context::MAX_MSG_SIZE / 4;
    for (auto * cur = begin; cur < end; ++cur) {
        sum += *cur;
    }
    sum += 0x00000001 + 0x00000000;
    msg_hdr->checksum = -sum;

    auto * v0_ptr = reinterpret_cast<volatile std::atomic<uint8_t> *>(&(msg_hdr->state.v[0]));
    auto * v1_ptr = reinterpret_cast<volatile std::atomic<uint8_t> *>(&(msg_hdr->state.v[1]));
    std::atomic_store_explicit(v0_ptr, uint8_t{1}, std::memory_order_release);
    while (std::atomic_load_explicit(v1_ptr, std::memory_order_acquire) == 0) {
        usleep(1);
    }
    d_ptr->store(0, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_acquire);
}

} // namespace

void RpcMemMapper::validate(const ggml_tensor * dst, const ggml_tensor * extra) {
    std::vector<ggml_backend_buffer *> buffers;
    std::unordered_set<ggml_backend_buffer *> seen_buffers;

    auto add_buffer = [&](ggml_backend_buffer * buf) {
        // 同一个 RPCMEM buffer 可能同时作为 src / dst / view backing 出现，这里先按 buffer 去重，避免预算和映射重试被重复放大。
        if (buf != nullptr && ggml_backend_buft_is_rpcmem(buf->buft) && seen_buffers.insert(buf).second) {
            buffers.push_back(buf);
        }
    };

    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        auto * src = dst->src[i];
        if (src != nullptr) {
            add_buffer(get_backing_buffer(src));
        }
    }
    add_buffer(get_backing_buffer(dst));
    add_buffer(get_backing_buffer(extra));

    // 记录当前算子真正需要保留的 buffer，mmap 失败后的紧急驱逐不能把这些 src/dst 本身提前释放掉。
    std::unordered_set<void *> required_buf_bases;
    size_t required_size = 0;
    for (auto * buf : buffers) {
        void * buf_base = ggml_backend_buffer_get_base(buf);
        required_buf_bases.insert(buf_base);
        if (buf_mapping.count(buf_base)) {
            accessed_bufs.erase(buf_iters.at(buf_base));
            accessed_bufs.push_front(buf_base);
            buf_iters[buf_base] = accessed_bufs.begin();
        } else {
            required_size += buf->size;
        }
    }

    // 预算驱逐阶段仍按原有 LRU 逻辑工作；只有真正 hit 到 fastrpc_mmap 失败时，才额外强制释放非当前算子所需的旧映射。
    auto force_release_one_lru_mapping = [&]() -> bool {
        for (auto it = accessed_bufs.end(); it != accessed_bufs.begin();) {
            --it;

            void * buf_base = *it;
            if (required_buf_bases.count(buf_base) != 0) {
                continue;
            }

            auto [fd, buf_size] = buf_mapping.at(buf_base);
            pending_unmap_reqs.push_back({fd, buf_base, buf_size, false});
            accessed_bufs.erase(it);
            buf_iters.erase(buf_base);
            buf_mapping.erase(buf_base);
            active_map_size -= buf_size;
            flush_pending_unmaps();
            return true;
        }

        return false;
    };

    size_t pending_host_mapped_size = get_pending_host_mapped_size(pending_unmap_reqs);
    GGML_ASSERT(required_size <= max_active_map_size);
    while (active_map_size + pending_host_mapped_size + required_size > max_active_map_size) {
        // 如果仅靠继续驱逐 active 映射还是不够，就先同步通知 DSP put_map，再执行 host munmap，释放真实地址空间。
        if (pending_host_mapped_size > 0) {
            flush_pending_unmaps();
            pending_host_mapped_size = 0;
            continue;
        }

        void * buf_base     = accessed_bufs.back();
        auto [fd, buf_size] = buf_mapping.at(buf_base);

        if (defer_unmap) {
            pending_unmap_reqs.push_back({fd, buf_base, buf_size, false});
            pending_host_mapped_size += buf_size;
        } else {
            int err = fastrpc_munmap(CDSP_DOMAIN_ID, fd, buf_base, buf_size);
            if (err != 0) {
                std::fprintf(stderr, "fastrpc_munmap failed with return code: %x\n", err);
            }
            ++n_map_ops;
        }

        accessed_bufs.pop_back();
        buf_iters.erase(buf_base);
        buf_mapping.erase(buf_base);
        active_map_size -= buf_size;
    }

    for (auto * buf : buffers) {
        void * buf_base = ggml_backend_buffer_get_base(buf);
        size_t buf_size = buf->size;
        if (buf_mapping.count(buf_base)) {
            continue;
        }

        int fd = rpcmem_to_fd(buf_base);
        if (fd < 0) {
            GGML_ABORT("rpcmem_to_fd returns %d, ptr %p, for dst tensor %s, n_ops %d\n",
                       fd, buf_base, dst->name, n_map_ops);
        }

        auto it = std::find_if(
            pending_unmap_reqs.begin(), pending_unmap_reqs.end(),
            [fd](const auto & v) { return v.fd == fd; });

        if (it == pending_unmap_reqs.end()) {
            int err = fastrpc_mmap(CDSP_DOMAIN_ID, fd, buf_base, 0, buf_size, FASTRPC_MAP_FD);
            while (err != 0) {
                // 部分真机的真实 FastRPC 映射窗口会比静态预算更小，这里在 mmap 失败后继续强制释放旧映射并重试，直到当前 buffer 真正映射成功或已经无可释放对象。
                const size_t pending_size = get_pending_host_mapped_size(pending_unmap_reqs);
                if (pending_size > 0) {
                    flush_pending_unmaps();
                } else if (!force_release_one_lru_mapping()) {
                    dump_state();

                    for (int i = 0; i < GGML_MAX_SRC; ++i) {
                        auto * src = dst->src[i];
                        if (src == nullptr) {
                            continue;
                        }

                        std::fprintf(stderr, "  src index %d name %s", i, src->name);
                        ggml_backend_buffer * src_buf = get_backing_buffer(src);
                        if (src_buf != nullptr && ggml_backend_buft_is_rpcmem(src_buf->buft)) {
                            auto * base = ggml_backend_buffer_get_base(src_buf);
                            std::fprintf(stderr, " buf %p\n", base);
                        } else {
                            std::fprintf(stderr, " (non rpcmem)\n");
                        }
                    }

                    std::fprintf(stderr,
                                 "dst name: %s, op: %s, n buffers to map in this step: %ld, active size: %.2f MiB\n",
                                 dst->name, ggml_op_name(dst->op), buffers.size(), active_map_size / 1048576.0);

                    GGML_ABORT("fastrpc_mmap failed with return code: 0x%x fd: %d buf_base: %p buf_size: %ld buf usage: %d\n",
                               err, fd, buf_base, buf_size, buf->usage);
                }

                err = fastrpc_mmap(CDSP_DOMAIN_ID, fd, buf_base, 0, buf_size, FASTRPC_MAP_FD);
            }

            ++n_map_ops;
        } else {
            if (it->host_unmapped) {
                pending_unmap_reqs.erase(it);
            } else {
                // 如果 buffer 还没真正 host-unmap 就又被需要了，直接撤销这条 pending 释放请求并恢复为 active。
                pending_unmap_reqs.erase(it);
                accessed_bufs.push_front(buf_base);
                buf_iters[buf_base]   = accessed_bufs.begin();
                buf_mapping[buf_base] = { fd, buf_size };
                active_map_size += buf_size;
                continue;
            }
        }

        accessed_bufs.push_front(buf_base);
        buf_iters[buf_base]   = accessed_bufs.begin();
        buf_mapping[buf_base] = { fd, buf_size };
        active_map_size += buf_size;
    }
}

std::pair<int, ssize_t> RpcMemMapper::get_tensor_mapping(const ggml_tensor * tensor) const {
    ggml_backend_buffer * buf = get_backing_buffer(tensor);
    GGML_ASSERT(buf != nullptr);
    GGML_ASSERT(ggml_backend_buft_is_rpcmem(buf->buft));

    void * buf_base = ggml_backend_buffer_get_base(buf);
    auto [fd, _]    = buf_mapping.at(buf_base);
    auto offset     = (intptr_t) tensor->data - (intptr_t) buf_base;
    return { fd, offset };
}

void RpcMemMapper::release_buffer(void * buf_base) {
    auto pending_it = std::find_if(
        pending_unmap_reqs.begin(), pending_unmap_reqs.end(),
        [buf_base](const auto & req) { return req.buf_base == buf_base; });
    if (pending_it != pending_unmap_reqs.end()) {
        flush_pending_unmaps();
        return;
    }

    auto mapping_it = buf_mapping.find(buf_base);
    if (mapping_it == buf_mapping.end()) {
        return;
    }

    const auto [fd, buf_size] = mapping_it->second;
    pending_unmap_reqs.push_back({fd, buf_base, buf_size, false});
    accessed_bufs.erase(buf_iters.at(buf_base));
    buf_iters.erase(buf_base);
    buf_mapping.erase(mapping_it);
    active_map_size -= buf_size;

    // The rpcmem allocation is about to be freed, so both DSP and host mappings
    // must be gone before the fd or virtual address can be reused.
    flush_pending_unmaps();
}

void RpcMemMapper::flush_pending_unmaps() {
    send_rpcmem_puts_to_dsp(pending_unmap_reqs);
    unmap_all_pending_buffers();
}

void RpcMemMapper::unmap_all_pending_buffers() {
    for (auto it = pending_unmap_reqs.begin(); it != pending_unmap_reqs.end();) {
        if (!it->host_unmapped) {
            int err = fastrpc_munmap(CDSP_DOMAIN_ID, it->fd, it->buf_base, it->buf_size);
            if (err != 0) {
                std::fprintf(stderr, "fastrpc_munmap failed with return code: 0x%x\n", err);
            }
            ++n_map_ops;
        }
        it = pending_unmap_reqs.erase(it);
    }
}

void RpcMemMapper::dump_state() const {
    std::fprintf(stderr, "total %d fastrpc_mmap + fastrpc_munmap ops\n", n_map_ops);

    if (!buf_mapping.empty()) {
        std::fprintf(stderr, "active mappings:\n");
    }
    for (const auto & [addr, pair] : buf_mapping) {
        std::fprintf(stderr, "    addr %p -> fd %d, size %.2f MiB\n", addr, pair.first, pair.second / 1048576.0);
    }

    if (!pending_unmap_reqs.empty()) {
        std::fprintf(stderr, "pending unmap requests:\n");
    }
    for (const auto & req : pending_unmap_reqs) {
        std::fprintf(stderr, "    fd %d, addr %p, size %.2f MiB, host_unmapped=%d\n",
                     req.fd, req.buf_base, req.buf_size / 1048576.0, req.host_unmapped ? 1 : 0);
    }
}

extern "C" {

int prepare_tensor_rpcmem_mapping(const struct ggml_tensor * dst) {
    ggml_backend_htp_context::instance()->mapper.validate(dst);
    return 0;
}

int prepare_tensor_rpcmem_mapping_with_extra(const struct ggml_tensor * dst, const struct ggml_tensor * extra) {
    ggml_backend_htp_context::instance()->mapper.validate(dst, extra);
    return 0;
}

}
