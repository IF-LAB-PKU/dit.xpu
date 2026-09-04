// RPC 内存映射管理器。
// 管理哪些 GGML 缓冲区已通过 fastrpc_mmap 映射到 DSP 地址空间，维护活跃映射的查找表。
// 当总映射大小超过预算时采用 LRU 驱逐策略；
// 支持延迟 DSP put（defer_unmap），将释放通知与算子计算合并为一次批量请求。
// Allocation teardown first removes the FastRPC session map and then lets the
// owning backend buffer call rpcmem_free to release the DMA buffer.

#pragma once

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

int prepare_tensor_rpcmem_mapping(const struct ggml_tensor * dst);
int prepare_tensor_rpcmem_mapping_with_extra(const struct ggml_tensor * dst, const struct ggml_tensor * extra);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
#include <list>
#include <tuple>
#include <cstdint>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// Session-local logical view accounting.  A released view lowers active bytes,
// but cumulative bytes remain monotonic for the lifetime of the backing fd;
// FastRPC VA is not assumed to become reusable after munmap.
class RpcMemRangeLedger {
public:
    explicit RpcMemRangeLedger(size_t budget) : budget_(budget) {}

    bool reserve(int fd, size_t offset, size_t size, size_t allocation_size = 0) {
        if (fd < 0 || size == 0) return false;
        if (allocation_size == 0) allocation_size = size;
        const Key key{fd, offset, size};
        if (views_.find(key) != views_.end()) return true;
        const bool new_allocation = allocations_.find(fd) == allocations_.end();
        if (new_allocation && (active_bytes_ > budget_ || allocation_size > budget_ - active_bytes_)) {
            return false;
        }
        views_.insert(key);
        if (new_allocation) {
            allocations_.insert(fd);
            allocation_sizes_[fd] = allocation_size;
            active_bytes_ += allocation_size;
            cumulative_bytes_ += allocation_size;
        }
        return true;
    }

    void release(int fd, size_t offset, size_t size) {
        const Key key{fd, offset, size};
        auto it = views_.find(key);
        if (it == views_.end()) return;
        views_.erase(it);
        bool allocation_still_viewed = false;
        for (const Key & view : views_) {
            if (view.fd == fd) {
                allocation_still_viewed = true;
                break;
            }
        }
        if (!allocation_still_viewed) {
            allocations_.erase(fd);
            const size_t allocation_size = allocation_sizes_[fd];
            allocation_sizes_.erase(fd);
            if (active_bytes_ >= allocation_size) active_bytes_ -= allocation_size;
        }
    }

    bool contains(int fd, size_t offset, size_t size) const {
        return views_.find(Key{fd, offset, size}) != views_.end();
    }
    size_t active_bytes() const { return active_bytes_; }
    size_t cumulative_bytes() const { return cumulative_bytes_; }

private:
    struct Key {
        int fd;
        size_t offset;
        size_t size;
        bool operator==(const Key & other) const {
            return fd == other.fd && offset == other.offset && size == other.size;
        }
    };
    struct KeyHash {
        size_t operator()(const Key & key) const {
            size_t hash = std::hash<int>{}(key.fd);
            hash ^= std::hash<size_t>{}(key.offset) + 0x9e3779b9u + (hash << 6) + (hash >> 2);
            hash ^= std::hash<size_t>{}(key.size) + 0x9e3779b9u + (hash << 6) + (hash >> 2);
            return hash;
        }
    };

    size_t budget_;
    size_t active_bytes_ = 0;
    size_t cumulative_bytes_ = 0;
    std::unordered_set<Key, KeyHash> views_;
    std::unordered_set<int> allocations_;
    std::unordered_map<int, size_t> allocation_sizes_;
};

struct RpcMemMapper {
    RpcMemMapper() : max_active_map_size{ (size_t) -1 }, active_map_size{ 0 }, persistent_map_size{ 0 }, defer_unmap{ false }, domain_id{ 3 } {}

    RpcMemMapper(size_t max_size, bool defer_unmap_ops, int mapping_domain = 3) :
        max_active_map_size{ max_size },
        active_map_size{ 0 },
        persistent_map_size{ 0 },
        defer_unmap{ defer_unmap_ops },
        domain_id{ mapping_domain } {}

    int domain() const { return domain_id; }
    void set_domain(int mapping_domain) { domain_id = mapping_domain; }
    size_t active_bytes() const { return active_map_size; }
    size_t persistent_bytes() const { return persistent_map_size; }
    size_t cumulative_bytes() const { return cumulative_map_size; }

    bool validate(const struct ggml_tensor * dst, const struct ggml_tensor * extra = nullptr);
    std::pair<int, ssize_t> get_tensor_mapping(const struct ggml_tensor * tensor) const;
    bool release_buffer(void * buf_base);
    bool was_freed_by_diagnostic(void * buf_base) const;

    struct UnmapRequest {
        int fd;
        void * buf_base;
        void * map_addr;
        size_t offset;
        size_t buf_size;
        uint32_t role;
        void * host_map_addr;
        size_t host_map_size;
    };

    const std::list<UnmapRequest> & get_pending_unmap_reqs() const { return pending_unmap_reqs; }

    // Complete DSP put_map before removing the corresponding FastRPC session
    // mapping. rpcmem_free is performed later by the allocation owner.
    bool flush_pending_unmaps();
    bool unmap_all_pending_buffers();
    // Drop every mapping owned by this session. This is diagnostic-only for
    // multi-session switching, where the same RPCMEM fd must not remain mapped
    // in both FastRPC PDs at once.
    void release_all_mappings();
    void dump_state() const;

  private:
    int n_map_ops = 0;

    size_t max_active_map_size;
    size_t active_map_size;
    size_t persistent_map_size;
    size_t cumulative_map_size = 0;

    bool defer_unmap;
    int domain_id;
    std::list<UnmapRequest> pending_unmap_reqs;

    std::unordered_map<void *, std::pair<int, size_t>>      buf_mapping;
    struct ViewKey {
        void * buf_base;
        int fd;
        size_t offset;
        size_t size;
        bool operator==(const ViewKey & other) const {
            return buf_base == other.buf_base && fd == other.fd &&
                offset == other.offset && size == other.size;
        }
    };
    struct ViewKeyHash {
        size_t operator()(const ViewKey & key) const {
            size_t hash = std::hash<void *>{}(key.buf_base);
            hash ^= std::hash<int>{}(key.fd) + 0x9e3779b9u + (hash << 6) + (hash >> 2);
            hash ^= std::hash<size_t>{}(key.offset) + 0x9e3779b9u + (hash << 6) + (hash >> 2);
            hash ^= std::hash<size_t>{}(key.size) + 0x9e3779b9u + (hash << 6) + (hash >> 2);
            return hash;
        }
    };
    struct ViewRecord {
        int fd;
        void * buf_base;
        void * map_addr;
        size_t offset;
        size_t size;
        uint32_t role;
        bool persistent;
    };
    void queue_unmap_request(const ViewRecord & view, size_t allocation_size);
    std::unordered_map<ViewKey, ViewRecord, ViewKeyHash> view_mapping;
    // FastRPC maps a complete backing allocation. Keep budget accounting at
    // that granularity even when one allocation has multiple logical views.
    std::unordered_set<void *> accounted_bufs;
    // FastRPC/libcdsprpc keeps one live VA mapping per fd in a session.  Keep
    // the physical mapping stable and represent subsequent tensor subranges as
    // logical views instead of issuing a second mmap for the same fd.
    std::unordered_set<int> mapped_fds;
    std::unordered_map<void *, std::list<void *>::iterator> buf_iters;
    std::list<void *>                                       accessed_bufs;
    std::unordered_set<void *>                              diagnostic_freed_bufs;
    std::unordered_set<void *>                              persistent_bufs;
};
#endif
