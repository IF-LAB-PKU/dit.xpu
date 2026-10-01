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
    std::unordered_set<void *> seen;
    for (const auto & req : pending_unmap_reqs) {
        if (seen.insert(req.host_map_addr).second) {
            total += req.host_map_size;
        }
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

    const bool use_v2 = ctx->multi_session_enabled && runtime != nullptr;
    const size_t entry_size = use_v2 ? sizeof(RpcmemMapEntryV2) : sizeof(int32_t);
    const size_t fixed_size = sizeof(MessageHeader) + 2 * sizeof(int32_t) +
        sizeof(RequestHeader) + (use_v2 ? sizeof(RpcmemMapRequestV2) : sizeof(RpcmemMapRequest));
    const size_t max_entries = fixed_size < ggml_backend_htp_context::MAX_MSG_SIZE
        ? (ggml_backend_htp_context::MAX_MSG_SIZE - fixed_size) / entry_size : 0;
    if (max_entries == 0) return false;

    auto begin = pending_unmap_reqs.begin();
    while (begin != pending_unmap_reqs.end()) {
        auto end = begin;
        size_t count = 0;
        while (end != pending_unmap_reqs.end() && count < max_entries) {
            ++end;
            ++count;
        }
        const int n_unmap_fds = static_cast<int>(count);
        const size_t map_req_size = sizeof(RequestHeader) +
            (use_v2 ? sizeof(RpcmemMapRequestV2) + count * sizeof(RpcmemMapEntryV2)
                    : sizeof(RpcmemMapRequest) + count * sizeof(int32_t));
        auto * d_ptr = reinterpret_cast<volatile std::atomic<uint64_t> *>(&(msg_hdr->state.d));
        std::atomic_store(d_ptr, uint64_t{0});
        msg_hdr->n_reqs = 1;
        msg_hdr->req_offsets[0] = message_header_size(msg_hdr);
        msg_hdr->req_offsets[1] = msg_hdr->req_offsets[0] + map_req_size;
        RequestHeader req_hdr{.state = 0, .type = REQUEST_TYPE_RPCMEM_MAP};
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
            for (auto it = begin; it != end; ++it) {
                RpcmemMapEntryV2 entry{it->fd, it->role, it->offset, it->buf_size, 0};
                write_buf(p, entry);
            }
        } else {
            RpcmemMapRequest map_req{n_unmap_fds, 0};
            write_buf(p, map_req);
            for (auto it = begin; it != end; ++it) write_buf(p, it->fd);
        }
        uint32_t sum = 0;
        uint32_t * sum_begin = reinterpret_cast<uint32_t *>(msg_hdr) + 3;
        uint32_t * sum_end = reinterpret_cast<uint32_t *>(msg_hdr) + ggml_backend_htp_context::MAX_MSG_SIZE / 4;
        for (auto * cur = sum_begin; cur < sum_end; ++cur) sum += *cur;
        msg_hdr->checksum = -(sum + 0x00000001 + 0x00000000);
        auto * v0_ptr = reinterpret_cast<volatile std::atomic<uint8_t> *>(&(msg_hdr->state.v[0]));
        auto * v1_ptr = reinterpret_cast<volatile std::atomic<uint8_t> *>(&(msg_hdr->state.v[1]));
        std::atomic_store_explicit(v0_ptr, uint8_t{1}, std::memory_order_release);
        while (std::atomic_load_explicit(v1_ptr, std::memory_order_acquire) == 0) usleep(1);
        const int response_state = message_header_get_request_ptr(msg_hdr, 0)->state;
        if (std::getenv("SD_HTP_RPCMEM_TRACE") != nullptr) {
            std::fprintf(stderr, "[rpcmem-trace] dsp_put n_puts=%d state=%d\n", n_unmap_fds, response_state);
        }
        d_ptr->store(0, std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_acquire);
        if (response_state != 0) return false;
        begin = end;
    }
    return true;
}

} // namespace

bool RpcMemMapper::validate(const ggml_tensor * dst, const ggml_tensor * extra) {
    if (htp_rpcmem_plain_mapping_path()) {
        return validate_plain(dst, extra);
    }

    const char * staging_env = std::getenv("SD_HTP_PACKED_STAGING");
    const bool staging = staging_env != nullptr && staging_env[0] != '\0' &&
        std::strcmp(staging_env, "0") != 0;
    if (staging) {
        // Staging has intentionally stable whole-buffer semantics.  Keep this
        // branch isolated so range views cannot alter graph-cut ownership.
        std::vector<ggml_backend_buffer *> buffers;
        std::unordered_set<ggml_backend_buffer *> seen;
        auto add = [&](ggml_backend_buffer * buf) {
            if (buf != nullptr && ggml_backend_buft_is_rpcmem(buf->buft) && seen.insert(buf).second) {
                buffers.push_back(buf);
                void * base = ggml_backend_buffer_get_base(buf);
                if (ggml_backend_htp_buffer_is_persistent_staging(buf)) persistent_bufs.insert(base);
            }
        };
        if (dst != nullptr) for (int i = 0; i < GGML_MAX_SRC; ++i) add(get_backing_buffer(dst->src[i]));
        add(get_backing_buffer(dst));
        add(get_backing_buffer(extra));
        size_t required_size = 0;
        for (auto * buf : buffers) {
            void * base = ggml_backend_buffer_get_base(buf);
            if (buf_mapping.count(base) == 0) required_size += buf->size;
        }
        const auto * htp_ctx = ggml_backend_htp_context::instance();
        const bool multi_session = htp_ctx != nullptr && htp_ctx->multi_session_enabled &&
            htp_ctx->session_count() > 1;
        if (!multi_session && !htp_rpcmem_mapping_fits(
                max_active_map_size, persistent_map_size, active_map_size,
                get_pending_host_mapped_size(pending_unmap_reqs), required_size)) return false;
        for (auto * buf : buffers) {
            void * base = ggml_backend_buffer_get_base(buf);
            if (buf_mapping.count(base) != 0) continue;
            const int fd = rpcmem_to_fd(base);
            if (fd < 0) return false;
            if (mapped_fds.insert(fd).second) {
                const int err = fastrpc_mmap(domain_id, fd, mapping_addr(base), 0, buf->size, FASTRPC_MAP_FD);
                if (err != 0) return false;
                ++n_map_ops;
                cumulative_map_size += buf->size;
            }
            buf_mapping[base] = {fd, buf->size};
            const bool persistent = persistent_bufs.count(base) != 0;
            const ViewKey key{base, fd, 0, buf->size};
            view_mapping.emplace(key, ViewRecord{fd, base, base, 0, buf->size, 0, persistent});
            if (persistent) persistent_map_size += buf->size;
            else {
                active_map_size += buf->size;
                accessed_bufs.push_front(base);
                buf_iters[base] = accessed_bufs.begin();
            }
        }
        return true;
    }

    struct RequiredRange {
        ggml_backend_buffer * buffer;
        void * base;
        int fd;
        size_t offset;
        size_t size;
        uint32_t role;
        bool persistent;
    };
    std::vector<RequiredRange> required;
    std::unordered_map<ggml_backend_buffer *, size_t> range_index;

    auto add_tensor = [&](const ggml_tensor * tensor, uint32_t role) {
        ggml_backend_buffer * buffer = get_backing_buffer(tensor);
        if (tensor == nullptr || buffer == nullptr || !ggml_backend_buft_is_rpcmem(buffer->buft)) {
            return;
        }
        void * base = ggml_backend_buffer_get_base(buffer);
        const uintptr_t data = reinterpret_cast<uintptr_t>(tensor->data);
        const uintptr_t base_addr = reinterpret_cast<uintptr_t>(base);
        if (data < base_addr || data - base_addr >= buffer->size) {
            return;
        }
        const size_t raw_offset = static_cast<size_t>(data - base_addr);
        const size_t raw_size = std::min(ggml_nbytes(tensor), buffer->size - raw_offset);
        if (raw_size == 0) {
            return;
        }
        constexpr size_t page = 4096;
        const size_t offset = raw_offset / page * page;
        const size_t end = std::min(buffer->size, (raw_offset + raw_size + page - 1) / page * page);
        const size_t size = end - offset;
        auto it = range_index.find(buffer);
        if (it == range_index.end()) {
            const int fd = rpcmem_to_fd(base);
            if (fd < 0) {
                GGML_ABORT("rpcmem_to_fd returns %d, ptr %p, for dst tensor %s, n_ops %d\n",
                           fd, base, dst != nullptr ? dst->name : "(null)", n_map_ops);
            }
            const bool persistent = ggml_backend_htp_buffer_is_persistent_staging(buffer);
            if (persistent) persistent_bufs.insert(base);
            range_index.emplace(buffer, required.size());
            required.push_back({buffer, base, fd, offset, size, role, persistent});
        } else {
            RequiredRange & current = required[it->second];
            const size_t current_end = current.offset + current.size;
            const size_t new_end = std::max(current_end, end);
            current.offset = std::min(current.offset, offset);
            current.size = new_end - current.offset;
            current.role |= role;
        }
    };

    if (dst != nullptr) {
        for (int i = 0; i < GGML_MAX_SRC; ++i) add_tensor(dst->src[i], 1u);
    }
    add_tensor(dst, 2u);
    add_tensor(extra, 1u);

    size_t required_size = 0;
    for (const RequiredRange & range : required) {
        if (accounted_bufs.find(range.base) == accounted_bufs.end()) {
            required_size += range.buffer->size;
        }
    }
    const size_t pending_host_mapped_size = get_pending_host_mapped_size(pending_unmap_reqs);
    if (std::getenv("SD_HTP_MULTI_SESSION_TRACE") != nullptr) {
        std::fprintf(stderr, "[htp-map] validate domain=%d ranges=%zu required=%zu max=%zu active=%zu persistent=%zu pending=%zu views=%zu fds=%zu tensor=%s\n",
                     domain_id, required.size(), required_size, max_active_map_size,
                     active_map_size, persistent_map_size, pending_host_mapped_size,
                     view_mapping.size(), mapped_fds.size(), dst != nullptr ? dst->name : "(null)");
    }
    if (!htp_rpcmem_mapping_fits(max_active_map_size, persistent_map_size,
                                 active_map_size, pending_host_mapped_size, required_size)) {
        if (std::getenv("SD_HTP_MULTI_SESSION_TRACE") != nullptr) {
            std::fprintf(stderr,
                         "[htp-map] budget-reject domain=%d max=%zu persistent=%zu active=%zu "
                         "pending=%zu required=%zu tensor=%s\n",
                         domain_id, max_active_map_size, persistent_map_size,
                         active_map_size, pending_host_mapped_size, required_size,
                         dst != nullptr ? dst->name : "(null)");
        }
        return false;
    }

    for (const RequiredRange & range : required) {
        const ViewKey key{range.base, range.fd, range.offset, range.size};
        if (view_mapping.find(key) != view_mapping.end()) {
            continue;
        }
        void * map_addr = static_cast<uint8_t *>(range.base) + range.offset;
        auto pending_it = std::find_if(pending_unmap_reqs.begin(), pending_unmap_reqs.end(),
            [&](const auto & request) { return request.fd == range.fd; });
        if (pending_it != pending_unmap_reqs.end()) {
            pending_unmap_reqs.erase(pending_it);
        } else if (mapped_fds.insert(range.fd).second) {
            // FastRPC's fd mapping is allocation-granular on the target
            // devices.  Map the backing allocation once; tensor offset/size
            // remain logical view metadata used by the DSP op payload.
            const int err = fastrpc_mmap(domain_id, range.fd, mapping_addr(range.base),
                                         0, range.buffer->size, FASTRPC_MAP_FD);
            if (err != 0) {
                mapped_fds.erase(range.fd);
                std::fprintf(stderr,
                             "fastrpc_mmap failed domain=%d fd=%d alloc_size=%zu "
                             "logical_offset=%zu logical_size=%zu active=%zu "
                             "persistent=%zu pending=%zu max=%zu tensor=%s ret=0x%x\n",
                             domain_id, range.fd, range.buffer->size, range.offset,
                             range.size, active_map_size, persistent_map_size,
                             get_pending_host_mapped_size(pending_unmap_reqs),
                             max_active_map_size,
                             dst != nullptr ? dst->name : "(null)", err);
                dump_state();
                return false;
            }
            ++n_map_ops;
            cumulative_map_size += range.buffer->size;
        }
        view_mapping.emplace(key, ViewRecord{range.fd, range.base, map_addr, range.offset,
                                             range.size, range.role, range.persistent});
        if (buf_mapping.find(range.base) == buf_mapping.end()) {
            buf_mapping.emplace(range.base, std::make_pair(range.fd, range.buffer->size));
        }
        if (accounted_bufs.insert(range.base).second) {
            if (!range.persistent) {
                active_map_size += range.buffer->size;
            } else {
                persistent_map_size += range.buffer->size;
            }
        }
        if (!range.persistent) {
            if (buf_iters.find(range.base) == buf_iters.end()) {
                accessed_bufs.push_front(range.base);
                buf_iters[range.base] = accessed_bufs.begin();
            }
        }
        if (std::getenv("SD_HTP_MULTI_SESSION_TRACE") != nullptr) {
            std::fprintf(stderr,
                         "[htp-map] domain=%d fd=%d offset=%zu size=%zu role=0x%x "
                         "active=%zu persistent=%zu cumulative=%zu tensor=%s\n",
                         domain_id, range.fd, range.offset, range.size, range.role,
                         active_map_size, persistent_map_size, cumulative_map_size,
                         dst != nullptr ? dst->name : "(null)");
        }
    }
    return true;
}

bool RpcMemMapper::was_freed_by_diagnostic(void * buf_base) const {
    return diagnostic_freed_bufs.count(buf_base) != 0;
}

// final 版逻辑：buffer 整块映射，活跃集超过预算时按 LRU 整块解除映射，
// mmap 失败时也强制再驱逐一条重试，因此真机窗口比预算小时同样能收敛。
bool RpcMemMapper::validate_plain(const ggml_tensor * dst, const ggml_tensor * extra) {
    std::vector<ggml_backend_buffer *> buffers;
    std::unordered_set<ggml_backend_buffer *> seen_buffers;
    auto add_buffer = [&](ggml_backend_buffer * buf) {
        if (buf != nullptr && ggml_backend_buft_is_rpcmem(buf->buft) && seen_buffers.insert(buf).second) {
            buffers.push_back(buf);
        }
    };
    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        const ggml_tensor * src = dst != nullptr ? dst->src[i] : nullptr;
        if (src != nullptr) add_buffer(get_backing_buffer(src));
    }
    add_buffer(get_backing_buffer(dst));
    add_buffer(get_backing_buffer(extra));

    // 本次算子自身需要的 buffer 不能在驱逐里被放掉，否则会一边释放一边又要用。
    std::unordered_set<void *> required_buf_bases;
    size_t required_size = 0;
    for (auto * buf : buffers) {
        void * buf_base = ggml_backend_buffer_get_base(buf);
        required_buf_bases.insert(buf_base);
        if (accounted_bufs.count(buf_base) != 0) {
            if (auto iter = buf_iters.find(buf_base); iter != buf_iters.end()) {
                accessed_bufs.erase(iter->second);
                accessed_bufs.push_front(buf_base);
                buf_iters[buf_base] = accessed_bufs.begin();
            }
        } else {
            required_size += buf->size;
        }
    }

    size_t pending_host_mapped_size = get_pending_host_mapped_size(pending_unmap_reqs);
    while (!htp_rpcmem_mapping_fits(max_active_map_size, persistent_map_size, active_map_size,
                                    pending_host_mapped_size, required_size)) {
        // 先让步于未完成的 DSP put，再继续驱逐更旧的映射。
        if (pending_host_mapped_size > 0) {
            flush_pending_unmaps();
            pending_host_mapped_size = 0;
            continue;
        }
        if (!evict_lru_buffer(required_buf_bases)) return false;
    }

    for (auto * buf : buffers) {
        void * buf_base = ggml_backend_buffer_get_base(buf);
        if (accounted_bufs.count(buf_base) != 0) continue;

        const size_t buf_size = buf->size;
        const int fd = rpcmem_to_fd(buf_base);
        if (fd < 0) {
            GGML_ABORT("rpcmem_to_fd returns %d, ptr %p, for dst tensor %s, n_ops %d\n",
                       fd, buf_base, dst != nullptr ? dst->name : "(null)", n_map_ops);
        }

        int err = fastrpc_mmap(domain_id, fd, mapping_addr(buf_base), 0, buf_size, FASTRPC_MAP_FD);
        while (err != 0) {
            // 静态预算可能比真机的映射窗口大，映射失败时强制驱逐更旧的映射再重试。
            if (!evict_lru_buffer(required_buf_bases)) {
                std::fprintf(stderr,
                             "fastrpc_mmap failed domain=%d fd=%d buf_size=%zu active=%zu max=%zu tensor=%s ret=0x%x\n",
                             domain_id, fd, buf_size, active_map_size, max_active_map_size,
                             dst != nullptr ? dst->name : "(null)", err);
                dump_state();
                GGML_ABORT("fastrpc_mmap failed with return code: 0x%x fd: %d buf_base: %p buf_size: %zu\n",
                           err, fd, buf_base, buf_size);
            }
            err = fastrpc_mmap(domain_id, fd, mapping_addr(buf_base), 0, buf_size, FASTRPC_MAP_FD);
        }

        ++n_map_ops;
        cumulative_map_size += buf_size;
        buf_mapping[buf_base] = { fd, buf_size };
        const ViewKey key{ buf_base, fd, 0, buf_size };
        view_mapping.emplace(key, ViewRecord{ fd, buf_base, buf_base, 0, buf_size, 0, false });
        accounted_bufs.insert(buf_base);
        active_map_size += buf_size;
        accessed_bufs.push_front(buf_base);
        buf_iters[buf_base] = accessed_bufs.begin();
    }
    return true;
}

bool RpcMemMapper::evict_lru_buffer(const std::unordered_set<void *> & required_buf_bases) {
    for (auto it = accessed_bufs.end(); it != accessed_bufs.begin();) {
        --it;
        void * buf_base = *it;
        if (required_buf_bases.count(buf_base) != 0 || persistent_bufs.count(buf_base) != 0) {
            continue;
        }
        return release_buffer(buf_base);
    }
    return false;
}

std::pair<int, ssize_t> RpcMemMapper::get_tensor_mapping(const ggml_tensor * tensor) const {
    ggml_backend_buffer * buf = get_backing_buffer(tensor);
    GGML_ASSERT(buf != nullptr);
    GGML_ASSERT(ggml_backend_buft_is_rpcmem(buf->buft));

    void * buf_base = ggml_backend_buffer_get_base(buf);
    auto offset     = (intptr_t) tensor->data - (intptr_t) buf_base;
    const size_t tensor_size = ggml_nbytes(tensor);
    for (const auto & [key, view] : view_mapping) {
        if (key.buf_base != buf_base || offset < static_cast<ssize_t>(key.offset)) {
            continue;
        }
        const size_t relative = static_cast<size_t>(offset) - key.offset;
        if (relative <= key.size && tensor_size <= key.size - relative) {
            return { view.fd, offset };
        }
    }
    // Preserve the legacy diagnostic behavior for callers that inspect a
    // tensor before its range is validated.
    auto mapping = buf_mapping.find(buf_base);
    if (std::getenv("SD_HTP_MULTI_SESSION_TRACE") != nullptr) {
        std::fprintf(stderr, "[htp-map] missing-view domain=%d base=%p offset=%zd bytes=%zu views=%zu\n",
                     domain_id, buf_base, offset, tensor_size, view_mapping.size());
    }
    GGML_ASSERT(mapping != buf_mapping.end());
    return { mapping->second.first, offset };
}

bool RpcMemMapper::release_buffer(void * buf_base) {
    for (auto it = view_mapping.begin(); it != view_mapping.end();) {
        if (it->second.buf_base != buf_base) {
            ++it;
            continue;
        }
        const ViewRecord view = it->second;
        queue_unmap_request(view, buf_mapping.at(buf_base).second);
        it = view_mapping.erase(it);
    }
    if (accounted_bufs.erase(buf_base) != 0) {
        const size_t allocation_size = buf_mapping.at(buf_base).second;
        if (persistent_bufs.count(buf_base) != 0) {
            GGML_ASSERT(persistent_map_size >= allocation_size);
            persistent_map_size -= allocation_size;
        } else {
            GGML_ASSERT(active_map_size >= allocation_size);
            active_map_size -= allocation_size;
        }
    }
    if (!flush_pending_unmaps()) return false;
    if (auto iter = buf_iters.find(buf_base); iter != buf_iters.end()) {
        accessed_bufs.erase(iter->second);
        buf_iters.erase(iter);
    }
    buf_mapping.erase(buf_base);
    persistent_bufs.erase(buf_base);
    return true;
}

void RpcMemMapper::queue_unmap_request(const ViewRecord & view, size_t allocation_size) {
    auto existing = std::find_if(pending_unmap_reqs.begin(), pending_unmap_reqs.end(),
        [&](const UnmapRequest & request) { return request.fd == view.fd; });
    if (existing != pending_unmap_reqs.end()) {
        existing->role |= view.role;
        return;
    }
    pending_unmap_reqs.push_back({view.fd, view.buf_base, view.buf_base,
                                  0, allocation_size, view.role,
                                  view.buf_base, allocation_size});
}

bool RpcMemMapper::flush_pending_unmaps() {
    if (!send_rpcmem_puts_to_dsp(pending_unmap_reqs, domain_id)) {
        return false;
    }
    return unmap_all_pending_buffers();
}

void RpcMemMapper::release_all_mappings() {
    for (auto it = view_mapping.begin(); it != view_mapping.end();) {
        const ViewRecord view = it->second;
        if (view.persistent) {
            ++it;
            continue;
        }
        queue_unmap_request(view, buf_mapping.at(view.buf_base).second);
        it = view_mapping.erase(it);
    }
    for (auto it = accounted_bufs.begin(); it != accounted_bufs.end();) {
        void * base = *it;
        if (persistent_bufs.count(base) == 0) {
            auto mapping = buf_mapping.find(base);
            if (mapping != buf_mapping.end()) {
                GGML_ASSERT(active_map_size >= mapping->second.second);
                active_map_size -= mapping->second.second;
            }
            it = accounted_bufs.erase(it);
        } else {
            ++it;
        }
    }
    for (auto it = buf_mapping.begin(); it != buf_mapping.end();) {
        if (persistent_bufs.count(it->first) == 0) it = buf_mapping.erase(it);
        else ++it;
    }
    for (auto it = buf_iters.begin(); it != buf_iters.end();) {
        if (persistent_bufs.count(it->first) == 0) it = buf_iters.erase(it);
        else ++it;
    }
    accessed_bufs.remove_if([this](void * base) { return persistent_bufs.count(base) == 0; });
    if (!flush_pending_unmaps()) {
        GGML_ABORT("failed to release FastRPC mappings for domain %d\n", domain_id);
    }
}

bool RpcMemMapper::unmap_all_pending_buffers() {
    bool success = true;
    std::unordered_set<int> unmapped_fds;
    for (auto it = pending_unmap_reqs.begin(); it != pending_unmap_reqs.end();) {
        int err = 0;
        if (unmapped_fds.insert(it->fd).second) {
            err = fastrpc_munmap(domain_id, it->fd, it->host_map_addr, it->host_map_size);
        }
        ++n_map_ops;
        if (err != 0) {
            std::fprintf(stderr,
                         "fastrpc_munmap failed domain=%d fd=%d addr=%p size=%zu ret=0x%x\n",
                         domain_id, it->fd, it->host_map_addr, it->host_map_size, err);
            success = false;
            ++it;
            continue;
        }
        mapped_fds.erase(it->fd);
        it = pending_unmap_reqs.erase(it);
    }
    return success;
}

void RpcMemMapper::dump_state() const {
    std::fprintf(stderr, "total %d fastrpc_mmap + fastrpc_munmap ops\n", n_map_ops);
    std::fprintf(stderr, "domain %d map bytes: active=%zu persistent=%zu cumulative=%zu pending=%zu\n",
                 domain_id, active_map_size, persistent_map_size, cumulative_map_size,
                 get_pending_host_mapped_size(pending_unmap_reqs));

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
    // Non-staging execution has an immutable graph owner.  Only the legacy
    // staging path may retry another session after a mapping failure.
    const bool allow_session_fallback = std::getenv("SD_HTP_PACKED_STAGING") != nullptr &&
        std::strcmp(std::getenv("SD_HTP_PACKED_STAGING"), "0") != 0;
    if (allow_session_fallback && ctx->multi_session_enabled && ctx->session_count() > 1) {
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
    const bool allow_session_fallback = std::getenv("SD_HTP_PACKED_STAGING") != nullptr &&
        std::strcmp(std::getenv("SD_HTP_PACKED_STAGING"), "0") != 0;
    if (allow_session_fallback && ctx->multi_session_enabled && ctx->session_count() > 1) {
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
