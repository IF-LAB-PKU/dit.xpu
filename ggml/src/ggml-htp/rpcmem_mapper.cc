// RpcMemMapper 的实现。
// validate(): 在执行算子前确保其所有张量 buffer 已 mmap 到 DSP，必要时驱逐旧的映射。
// get_tensor_mapping(): 将 ggml_tensor 指针解析为 (fd, offset_in_buffer) 对供 DSP 端寻址。
// unmap_all_pending_buffers(): 在 DSP put 完成后清理 FastRPC session 映射。

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
#include "rpcmem_session_route.h"

namespace {

// FastRPC normally receives the host RPCMEM address as the requested DSP
// mapping address.  Some devices reject a reused address after a domain
// switch even though the fd was unmapped successfully.  Keep the legacy
// fixed-address behavior by default, but allow a diagnostic run to let
// FastRPC choose a fresh remote address.
void * mapping_addr(void * host_addr) {
    const char * env = std::getenv("SD_HTP_RPCMEM_MAP_NULL_ADDR");
    return env != nullptr && env[0] != '\0' && std::strcmp(env, "0") != 0
        ? nullptr : host_addr;
}

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

// pending 列表里的 DSP put 尚未发送前仍占用映射窗口。
size_t get_pending_host_mapped_size(const std::list<RpcMemMapper::UnmapRequest> & pending_unmap_reqs) {
    size_t total = 0;
    for (const auto & req : pending_unmap_reqs) {
        total += req.buf_size;
    }
    return total;
}

template <typename T>
void write_buf(uint8_t *& p, const T & v) {
    *reinterpret_cast<T *>(p) = v;
    p += sizeof(v);
}

bool send_rpcmem_puts_to_dsp(
        const std::list<RpcMemMapper::UnmapRequest> & pending_unmap_reqs,
        int mapper_domain) {
    if (pending_unmap_reqs.empty()) {
        return true;
    }

    auto * ctx = ggml_backend_htp_context::instance();
    const int session_index = htp_rpcmem_session_index_for_domain(ctx->sessions, mapper_domain);
    auto * runtime = ctx->session_runtime(session_index);
    void * channel = runtime != nullptr ? runtime->message_channel : nullptr;
    auto * msg_hdr = reinterpret_cast<MessageHeader *>(channel);
    GGML_ASSERT(msg_hdr != nullptr);

    const int n_unmap_fds = static_cast<int>(pending_unmap_reqs.size());
    const bool use_v2 = ctx->multi_session_enabled && runtime != nullptr;
    const size_t map_req_size = sizeof(RequestHeader) +
        (use_v2 ? sizeof(RpcmemMapRequestV2) + n_unmap_fds * sizeof(RpcmemMapEntryV2)
                : sizeof(RpcmemMapRequest) + n_unmap_fds * sizeof(int32_t));

    auto * d_ptr = reinterpret_cast<volatile std::atomic<uint64_t> *>(&(msg_hdr->state.d));
    std::atomic_store(d_ptr, uint64_t{0});

    msg_hdr->n_reqs         = 1;
    msg_hdr->req_offsets[0] = message_header_size(msg_hdr);
    msg_hdr->req_offsets[1] = msg_hdr->req_offsets[0] + map_req_size;

    RequestHeader req_hdr{
        .state = 0,
        .type  = REQUEST_TYPE_RPCMEM_MAP,
    };
    auto * p = reinterpret_cast<uint8_t *>(message_header_get_request_ptr(msg_hdr, 0));
    write_buf(p, req_hdr);
    if (use_v2) {
        RpcmemMapRequestV2 map_req{
            .magic = RPCMEM_MAP_REQUEST_SESSION_MAGIC,
            .version = RPCMEM_MAP_REQUEST_VERSION_SESSION,
            .header_size = sizeof(RpcmemMapRequestV2),
            .session_token = runtime->id == 0 ? 0 :
                ((static_cast<uint64_t>(runtime->id) << 32) | runtime->domain),
            .n_puts = n_unmap_fds,
            .n_gets = 0,
        };
        write_buf(p, map_req);
        for (const auto & req : pending_unmap_reqs) {
            RpcmemMapEntryV2 entry{req.fd, 0, 0, req.buf_size, 0};
            write_buf(p, entry);
        }
    } else {
        RpcmemMapRequest map_req{n_unmap_fds, 0};
        write_buf(p, map_req);
        for (const auto & req : pending_unmap_reqs) write_buf(p, req.fd);
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
    auto * response = message_header_get_request_ptr(msg_hdr, 0);
    const int response_state = response->state;
    if (std::getenv("SD_HTP_RPCMEM_TRACE") != nullptr) {
        std::fprintf(stderr, "[rpcmem-trace] dsp_put n_puts=%d state=%d\n",
                     n_unmap_fds, response_state);
    }
    d_ptr->store(0, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_acquire);
    return response_state == 0;
}

} // namespace

bool RpcMemMapper::validate(const ggml_tensor * dst, const ggml_tensor * extra) {
    std::vector<ggml_backend_buffer *> buffers;
    std::unordered_set<ggml_backend_buffer *> seen_buffers;

    auto add_buffer = [&](ggml_backend_buffer * buf) {
        // 同一个 RPCMEM buffer 可能同时作为 src / dst / view backing 出现，这里先按 buffer 去重，避免预算和映射重试被重复放大。
        if (buf != nullptr && ggml_backend_buft_is_rpcmem(buf->buft) && seen_buffers.insert(buf).second) {
            buffers.push_back(buf);
            void * base = ggml_backend_buffer_get_base(buf);
            if (ggml_backend_htp_buffer_is_persistent_staging(buf)) {
                persistent_bufs.insert(base);
            }
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
            if (persistent_bufs.count(buf_base) == 0) {
                accessed_bufs.erase(buf_iters.at(buf_base));
                accessed_bufs.push_front(buf_base);
                buf_iters[buf_base] = accessed_bufs.begin();
            }
        } else if (persistent_bufs.count(buf_base) == 0) {
            required_size += buf->size;
        }
    }

    size_t pending_host_mapped_size = get_pending_host_mapped_size(pending_unmap_reqs);
    const auto * htp_ctx = ggml_backend_htp_context::instance();
    const bool multi_session = htp_ctx != nullptr && htp_ctx->multi_session_enabled &&
        htp_ctx->session_count() > 1;
    if (!multi_session && (persistent_map_size > max_active_map_size ||
        required_size > max_active_map_size - persistent_map_size)) {
        return false;
    }
    if (!multi_session && !htp_rpcmem_mapping_fits(
            max_active_map_size,
            persistent_map_size,
            active_map_size,
            pending_host_mapped_size,
            required_size)) {
        // A live allocation retains its cumulative FastRPC VA consumption even
        // after fastrpc_munmap (FastRPC issue #137). Removing only mapper
        // bookkeeping would hide that consumption without making the VA reusable.
        return false;
    }

    for (auto * buf : buffers) {
        void * buf_base = ggml_backend_buffer_get_base(buf);
        size_t buf_size = buf->size;
        if (buf_mapping.count(buf_base)) {
            continue;
        }

        int fd = rpcmem_to_fd(buf_base);
        if (diagnostic_freed_bufs.count(buf_base) != 0) {
            GGML_ABORT("diagnostic free-on-mmap-failure requires reload for ptr %p, tensor %s\n",
                       buf_base, dst->name);
        }
        if (fd < 0) {
            GGML_ABORT("rpcmem_to_fd returns %d, ptr %p, for dst tensor %s, n_ops %d\n",
                       fd, buf_base, dst->name, n_map_ops);
        }

        auto it = std::find_if(
            pending_unmap_reqs.begin(), pending_unmap_reqs.end(),
            [fd](const auto & v) { return v.fd == fd; });

        if (it == pending_unmap_reqs.end()) {
            int err = fastrpc_mmap(domain_id, fd, mapping_addr(buf_base), 0, buf_size, FASTRPC_MAP_FD);
            while (err != 0) {
                dump_state();
                return false;
            }

            ++n_map_ops;
        } else {
            // The DSP put has not been sent yet, so cancel it and restore the
            // mapper entry when this buffer is needed again.
            pending_unmap_reqs.erase(it);
            accessed_bufs.push_front(buf_base);
            buf_iters[buf_base]   = accessed_bufs.begin();
            buf_mapping[buf_base] = { fd, buf_size };
            active_map_size += buf_size;
            continue;
        }

        if (persistent_bufs.count(buf_base) == 0) {
            accessed_bufs.push_front(buf_base);
            buf_iters[buf_base]   = accessed_bufs.begin();
        }
        buf_mapping[buf_base] = { fd, buf_size };
        if (persistent_bufs.count(buf_base) == 0) {
            active_map_size += buf_size;
        } else {
            persistent_map_size += buf_size;
        }
    }
    return true;
}

bool RpcMemMapper::was_freed_by_diagnostic(void * buf_base) const {
    return diagnostic_freed_bufs.count(buf_base) != 0;
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

bool RpcMemMapper::release_buffer(void * buf_base) {
    auto pending_it = std::find_if(
        pending_unmap_reqs.begin(), pending_unmap_reqs.end(),
        [buf_base](const auto & req) { return req.buf_base == buf_base; });
    if (pending_it != pending_unmap_reqs.end()) {
        return flush_pending_unmaps();
    }

    auto mapping_it = buf_mapping.find(buf_base);
    if (mapping_it == buf_mapping.end()) {
        return true;
    }

    const auto [fd, buf_size] = mapping_it->second;
    if (persistent_bufs.count(buf_base) != 0) {
        // fastrpc_munmap alone does not reclaim cumulative DSP VA (issue #137),
        // but it must remove libcdsprpc's per-session static_map entry before
        // rpcmem_free closes the fd. Otherwise immediate fd reuse returns
        // AEE_EALREADY from the next fastrpc_mmap.
        pending_unmap_reqs.push_back({fd, buf_base, buf_size});
        if (!flush_pending_unmaps()) {
            return false;
        }
        buf_mapping.erase(mapping_it);
        persistent_bufs.erase(buf_base);
        GGML_ASSERT(persistent_map_size >= buf_size);
        persistent_map_size -= buf_size;
        return true;
    }
    pending_unmap_reqs.push_back({fd, buf_base, buf_size});
    if (!flush_pending_unmaps()) {
        return false;
    }
    accessed_bufs.erase(buf_iters.at(buf_base));
    buf_iters.erase(buf_base);
    buf_mapping.erase(mapping_it);
    active_map_size -= buf_size;
    return true;
}

bool RpcMemMapper::flush_pending_unmaps() {
    if (!send_rpcmem_puts_to_dsp(pending_unmap_reqs, domain_id)) {
        return false;
    }
    return unmap_all_pending_buffers();
}

void RpcMemMapper::release_all_mappings() {
    for (const auto & [buf_base, mapping] : buf_mapping) {
        if (persistent_bufs.count(buf_base) != 0) {
            continue;
        }
        pending_unmap_reqs.push_back({mapping.first, buf_base, mapping.second});
    }
    for (auto it = buf_mapping.begin(); it != buf_mapping.end();) {
        if (persistent_bufs.count(it->first) == 0) {
            it = buf_mapping.erase(it);
        } else {
            ++it;
        }
    }
    for (auto it = buf_iters.begin(); it != buf_iters.end();) {
        if (persistent_bufs.count(it->first) == 0) {
            it = buf_iters.erase(it);
        } else {
            ++it;
        }
    }
    accessed_bufs.remove_if([this](void * buf_base) {
        return persistent_bufs.count(buf_base) == 0;
    });
    active_map_size = 0;
    if (!flush_pending_unmaps()) {
        GGML_ABORT("failed to release FastRPC mappings for domain %d\n", domain_id);
    }
}

bool RpcMemMapper::unmap_all_pending_buffers() {
    bool success = true;
    for (auto it = pending_unmap_reqs.begin(); it != pending_unmap_reqs.end();) {
        const int err = fastrpc_munmap(domain_id, it->fd, it->buf_base, it->buf_size);
        ++n_map_ops;
        if (err != 0) {
            std::fprintf(stderr,
                         "fastrpc_munmap failed domain=%d fd=%d addr=%p size=%zu ret=0x%x\n",
                         domain_id, it->fd, it->buf_base, it->buf_size, err);
            success = false;
            ++it;
            continue;
        }
        it = pending_unmap_reqs.erase(it);
    }
    return success;
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
        std::fprintf(stderr, "    fd %d, addr %p, size %.2f MiB\n",
                     req.fd, req.buf_base, req.buf_size / 1048576.0);
    }
}

extern "C" {

int prepare_tensor_rpcmem_mapping(const struct ggml_tensor * dst) {
    auto * ctx = ggml_backend_htp_context::instance();
    if (ctx == nullptr) return -1;
    if (ctx->dispatch_mapper().validate(dst)) return 0;
    if (ctx->multi_session_enabled && ctx->session_count() > 1) {
        const int current = ctx->active_session_index;
        for (size_t offset = 1; offset < ctx->session_count(); ++offset) {
            const int candidate = (current + static_cast<int>(offset)) %
                static_cast<int>(ctx->session_count());
            if (!ggml_backend_htp_dispatch_acquire(candidate) &&
                !ggml_backend_htp_select_session_for_compute(candidate)) continue;
            if (ctx->dispatch_mapper().validate(dst)) {
                ggml_backend_htp_dispatch_force_owner(candidate);
                return 0;
            }
        }
    }
    GGML_ABORT("HTP RPCMEM mapping budget exhausted in all session-local views for tensor %s\n",
               dst != nullptr ? dst->name : "(null)");
}

int prepare_tensor_rpcmem_mapping_with_extra(const struct ggml_tensor * dst, const struct ggml_tensor * extra) {
    auto * ctx = ggml_backend_htp_context::instance();
    if (ctx == nullptr) return -1;
    if (ctx->dispatch_mapper().validate(dst, extra)) return 0;
    if (ctx->multi_session_enabled && ctx->session_count() > 1) {
        const int current = ctx->active_session_index;
        for (size_t offset = 1; offset < ctx->session_count(); ++offset) {
            const int candidate = (current + static_cast<int>(offset)) %
                static_cast<int>(ctx->session_count());
            if (!ggml_backend_htp_dispatch_acquire(candidate) &&
                !ggml_backend_htp_select_session_for_compute(candidate)) continue;
            if (ctx->dispatch_mapper().validate(dst, extra)) {
                ggml_backend_htp_dispatch_force_owner(candidate);
                return 0;
            }
        }
    }
    GGML_ABORT("HTP RPCMEM mapping budget exhausted in all session-local views for tensor %s\n",
               dst != nullptr ? dst->name : "(null)");
}

}
