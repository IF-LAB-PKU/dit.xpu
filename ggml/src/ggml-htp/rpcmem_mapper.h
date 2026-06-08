// RPC 内存映射管理器。
// 管理哪些 GGML 缓冲区已通过 fastrpc_mmap 映射到 DSP 地址空间，维护活跃映射的查找表。
// 当总映射大小超过预算时采用 LRU 驱逐策略；
// 支持延迟解除映射（defer_unmap），将 munmap 与算子计算合并为一次批量请求以降低 RPC 往返开销。

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
#include <unordered_map>

struct RpcMemMapper {
    RpcMemMapper() : max_active_map_size{ (size_t) -1 }, active_map_size{ 0 }, defer_unmap{ false } {}

    RpcMemMapper(size_t max_size, bool defer_unmap_ops) :
        max_active_map_size{ max_size },
        active_map_size{ 0 },
        defer_unmap{ defer_unmap_ops } {}

    void validate(const struct ggml_tensor * dst, const struct ggml_tensor * extra = nullptr);
    std::pair<int, ssize_t> get_tensor_mapping(const struct ggml_tensor * tensor) const;

    struct UnmapRequest {
        int fd;
        void * buf_base;
        size_t buf_size;
        bool host_unmapped;
    };

    const std::list<UnmapRequest> & get_pending_unmap_reqs() const { return pending_unmap_reqs; }

    // 当 host 侧映射窗口不够时，先同步通知 DSP 释放 HAP_mmap，再执行 host 侧 fastrpc_munmap。
    void flush_pending_unmaps();
    void unmap_all_pending_buffers();
    void dump_state() const;

  private:
    int n_map_ops = 0;

    size_t max_active_map_size;
    size_t active_map_size;

    bool defer_unmap;
    std::list<UnmapRequest> pending_unmap_reqs;

    std::unordered_map<void *, std::pair<int, size_t>>      buf_mapping;
    std::unordered_map<void *, std::list<void *>::iterator> buf_iters;
    std::list<void *>                                       accessed_bufs;
};
#endif
