#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-htp-lifetime.h"

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_htp_fused_anchor_metadata {
    int32_t qo_len;
    int32_t kv_len;
    int32_t n_heads;
    int32_t n_kv_heads;
    int32_t head_dim;
    int32_t block_q;
    int32_t block_k;
    int32_t n_q_blocks;
    int32_t n_k_blocks;
    int32_t max_kept_blocks;
    int32_t mask_format;
    int32_t sparse_policy;
    int32_t image_tokens;
    int32_t text_tokens;
    int32_t image_block_rows;
    int32_t image_block_cols;
};

enum ggml_htp_route_mode {
    GGML_HTP_ROUTE_DISABLED = 0,
    GGML_HTP_ROUTE_FUSED_OPENCL_RESIDUAL,
    GGML_HTP_ROUTE_OPENCL_ANCHOR,
    GGML_HTP_ROUTE_HTP_CPU,
    GGML_HTP_ROUTE_INVALID,
};

struct ggml_htp_route_summary {
    enum ggml_htp_route_mode mode;
    int32_t anchor;
    int32_t fused_anchor;
    int32_t htp_dense;
    int32_t htp_sparse;
    int32_t opencl_sparse;
    int32_t opencl_residual;
    int32_t cpu_residual;
    int32_t cache_pairs;
    int32_t cpu_attention_fallback;
    int32_t status_ok;
};

GGML_BACKEND_API bool ggml_backend_htp_route_summary_validate(
    const struct ggml_htp_route_summary * summary,
    char * reason,
    size_t reason_capacity);

GGML_BACKEND_API bool ggml_backend_htp_get_last_route_summary(
    ggml_backend_t backend,
    struct ggml_htp_route_summary * summary);

GGML_BACKEND_API bool ggml_backend_htp_collect_route_summary(
    ggml_backend_t backend,
    const struct ggml_cgraph * cgraph,
    struct ggml_htp_route_summary * summary);

GGML_BACKEND_API ggml_backend_t ggml_backend_htp_init(void);

GGML_BACKEND_API bool ggml_backend_is_htp(ggml_backend_t backend);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_htp_reg(void);

GGML_BACKEND_API bool ggml_backend_htp_runtime_available(void);

GGML_BACKEND_API const char * ggml_backend_htp_runtime_error(void);

struct ggml_backend_htp_rpcmem_info {
    void * base;
    // Borrowed from RPCMEM. Valid only while the backing buffer is alive; do not close it.
    int fd;
    size_t logical_size;
    size_t allocation_size;
};

GGML_BACKEND_API bool ggml_backend_htp_get_rpcmem_info(
    ggml_backend_buffer_t buffer,
    struct ggml_backend_htp_rpcmem_info * info);

typedef struct ggml_backend_htp_async_cache_slot ggml_backend_htp_async_cache_slot;

GGML_BACKEND_API ggml_backend_htp_async_cache_slot *
ggml_backend_htp_async_cache_slot_create(
    ggml_backend_t backend,
    const char * cache_key,
    uint64_t generation,
    enum ggml_type type,
    const int64_t ne[GGML_MAX_DIMS],
    char * reason,
    size_t reason_capacity);

GGML_BACKEND_API struct ggml_tensor * ggml_backend_htp_async_cache_slot_tensor(
    ggml_backend_htp_async_cache_slot * slot);

GGML_BACKEND_API bool ggml_backend_htp_async_cache_slot_rpcmem_info(
    const ggml_backend_htp_async_cache_slot * slot,
    struct ggml_backend_htp_rpcmem_info * info);

GGML_BACKEND_API uint64_t ggml_backend_htp_async_cache_slot_opencl_import_identity(
    const ggml_backend_htp_async_cache_slot * slot);

GGML_BACKEND_API void ggml_backend_htp_async_cache_slot_free(
    ggml_backend_htp_async_cache_slot * slot);

GGML_BACKEND_API bool ggml_backend_buft_is_rpcmem(ggml_backend_buffer_type_t buft);
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_htp_qkv_buffer_type(void);
GGML_BACKEND_API void ggml_backend_htp_set_n_threads(ggml_backend_t backend, int n_threads);
GGML_BACKEND_API void ggml_backend_htp_reset_attention_profile(ggml_backend_t backend);
GGML_BACKEND_API int64_t ggml_backend_htp_get_last_attention_elapsed_us(ggml_backend_t backend);
GGML_BACKEND_API void ggml_backend_htp_pack_f16_weight(ggml_fp16_t * dst, const ggml_fp16_t * src, int64_t k, int64_t n);
GGML_BACKEND_API void ggml_backend_htp_register_packed_f16_weight(const struct ggml_tensor * original,
                                                                  struct ggml_tensor * packed);
GGML_BACKEND_API void ggml_backend_htp_unregister_packed_f16_weight(const struct ggml_tensor * original);
GGML_BACKEND_API struct ggml_tensor * ggml_backend_htp_get_packed_f16_weight(const struct ggml_tensor * original);
// 暴露 hybrid 开关给上层推理线路，避免只能依赖全局环境变量控制 HTP 混合执行。
GGML_BACKEND_API void ggml_backend_htp_set_hybrid_enabled(bool enabled);
GGML_BACKEND_API bool ggml_backend_htp_get_hybrid_enabled(void);

// Mark packed dense attention tensors used by deferred OpenCL residual jobs as
// graph outputs before gallocr plans the arena, so their spans stay live until
// graph completion.
GGML_BACKEND_API size_t ggml_backend_htp_opencl_retain_deferred_residual_inputs(
    struct ggml_cgraph * graph);

GGML_BACKEND_API void ggml_backend_htp_opencl_set_compute_arena_generation(
    ggml_backend_t backend,
    uint64_t generation);

GGML_BACKEND_API bool ggml_backend_htp_opencl_arena_generation_pending(
    ggml_backend_t backend,
    uint64_t generation);

GGML_BACKEND_API bool ggml_backend_htp_opencl_release_qkv_generation(
    ggml_backend_t backend,
    uint64_t generation,
    bool wait_for_sparse,
    bool * released,
    char * reason,
    size_t reason_capacity);

GGML_BACKEND_API bool ggml_backend_htp_opencl_release_arena_generation(
    ggml_backend_t backend,
    uint64_t generation,
    bool wait,
    bool * released,
    char * reason,
    size_t reason_capacity);

GGML_BACKEND_API size_t ggml_backend_htp_opencl_deferred_arena_count(
    ggml_backend_t backend);

GGML_BACKEND_API bool ggml_backend_htp_opencl_bind_cache_input(
    ggml_backend_t backend,
    const struct ggml_tensor * input,
    const char * cache_key,
    uint64_t generation,
    char * reason,
    size_t reason_capacity);

GGML_BACKEND_API bool ggml_backend_htp_opencl_cache_query(
    ggml_backend_t backend,
    const char * cache_key,
    uint64_t generation,
    bool * ready,
    char * reason,
    size_t reason_capacity);

GGML_BACKEND_API bool ggml_backend_htp_opencl_cache_wait(
    ggml_backend_t backend,
    const char * cache_key,
    uint64_t generation,
    bool * waited,
    char * reason,
    size_t reason_capacity);

GGML_BACKEND_API void ggml_backend_htp_opencl_test_fail_next_cache_wait(
    ggml_backend_t backend);

GGML_BACKEND_API size_t ggml_backend_htp_opencl_cache_input_binding_count(
    ggml_backend_t backend);

GGML_BACKEND_API size_t ggml_backend_htp_opencl_live_cache_import_count(
    ggml_backend_t backend);

GGML_BACKEND_API size_t ggml_backend_htp_opencl_live_cache_event_count(
    ggml_backend_t backend);

#ifdef __cplusplus
}
#endif
