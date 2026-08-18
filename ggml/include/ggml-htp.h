#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-htp-lifetime.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

#define GGML_HTP_MEMORY_LEDGER_MAX_CACHE_SLOTS 256
#define GGML_HTP_MEMORY_LEDGER_CACHE_KEY_CAPACITY 128

struct ggml_htp_cache_slot_snapshot {
    char cache_key[GGML_HTP_MEMORY_LEDGER_CACHE_KEY_CAPACITY];
    int32_t branch;
    uint64_t generation;
    int fd;
    size_t logical_size;
    size_t allocation_size;
    uint64_t opencl_import_identity;
};

struct ggml_htp_lifetime_snapshot {
    size_t live_imports;
    size_t live_events;
    size_t live_leases;
    size_t pending_jobs;
    size_t retired_arenas;
};

struct ggml_htp_memory_ledger_snapshot {
    enum ggml_htp_route_mode requested_route;
    int32_t preflight_result;
    size_t routed_sparse_count;
    size_t routed_rho_count;
    size_t routed_residual_count;
    size_t reported_opencl_residual;
    size_t runtime_buffer_bytes;
    size_t htp_buffer_bytes;
    size_t qkv_buffer_bytes;
    size_t host_cache_bytes;
    size_t rpcmem_cache_logical_bytes;
    size_t rpcmem_cache_allocation_bytes;
    size_t cache_slot_count;
    struct ggml_htp_cache_slot_snapshot
        cache_slots[GGML_HTP_MEMORY_LEDGER_MAX_CACHE_SLOTS];
    size_t live_imports;
    size_t live_events;
    size_t live_leases;
    size_t pending_jobs;
    size_t retired_arenas;
    size_t spilled_host_bytes;
    size_t hydrated_rpcmem_bytes;
};

static inline const char * ggml_htp_memory_ledger_route_name(
        enum ggml_htp_route_mode mode) {
    switch (mode) {
        case GGML_HTP_ROUTE_FUSED_OPENCL_RESIDUAL: return "fused";
        case GGML_HTP_ROUTE_OPENCL_ANCHOR:         return "opencl";
        case GGML_HTP_ROUTE_HTP_CPU:               return "htp-cpu";
        case GGML_HTP_ROUTE_DISABLED:              return "disabled";
        case GGML_HTP_ROUTE_INVALID:               return "invalid";
    }
    return "invalid";
}

static inline bool ggml_backend_htp_memory_ledger_recalculate(
        struct ggml_htp_memory_ledger_snapshot * snapshot) {
    if (snapshot == NULL ||
        snapshot->cache_slot_count > GGML_HTP_MEMORY_LEDGER_MAX_CACHE_SLOTS) {
        return false;
    }
    size_t logical = 0;
    size_t allocation = 0;
    for (size_t i = 0; i < snapshot->cache_slot_count; ++i) {
        const struct ggml_htp_cache_slot_snapshot * slot =
            &snapshot->cache_slots[i];
        if (SIZE_MAX - logical < slot->logical_size ||
            SIZE_MAX - allocation < slot->allocation_size) {
            return false;
        }
        logical += slot->logical_size;
        allocation += slot->allocation_size;
    }
    snapshot->rpcmem_cache_logical_bytes = logical;
    snapshot->rpcmem_cache_allocation_bytes = allocation;
    return true;
}

static inline int ggml_backend_htp_format_memory_ledger(
        char * buffer,
        size_t capacity,
        const char * phase,
        const struct ggml_htp_memory_ledger_snapshot * snapshot) {
    if (buffer == NULL || capacity == 0 || phase == NULL || snapshot == NULL) {
        return -1;
    }
    return snprintf(
        buffer, capacity,
        "htp_memory_ledger phase=%s requested=%s preflight=%d "
        "routed_sparse=%zu routed_rho=%zu routed_residual=%zu "
        "reported_opencl_residual=%zu runtime_buffer_bytes=%zu "
        "htp_buffer_bytes=%zu qkv_buffer_bytes=%zu host_cache_bytes=%zu "
        "rpcmem_cache_logical_bytes=%zu rpcmem_cache_allocation_bytes=%zu "
        "rpcmem_cache_slot_count=%zu live_imports=%zu live_events=%zu "
        "live_leases=%zu pending_jobs=%zu retired_arenas=%zu "
        "spilled_host_bytes=%zu hydrated_rpcmem_bytes=%zu",
        phase, ggml_htp_memory_ledger_route_name(snapshot->requested_route),
        snapshot->preflight_result, snapshot->routed_sparse_count,
        snapshot->routed_rho_count, snapshot->routed_residual_count,
        snapshot->reported_opencl_residual, snapshot->runtime_buffer_bytes,
        snapshot->htp_buffer_bytes, snapshot->qkv_buffer_bytes,
        snapshot->host_cache_bytes, snapshot->rpcmem_cache_logical_bytes,
        snapshot->rpcmem_cache_allocation_bytes, snapshot->cache_slot_count,
        snapshot->live_imports, snapshot->live_events, snapshot->live_leases,
        snapshot->pending_jobs, snapshot->retired_arenas,
        snapshot->spilled_host_bytes, snapshot->hydrated_rpcmem_bytes);
}

static inline int ggml_backend_htp_format_memory_ledger_slot(
        char * buffer,
        size_t capacity,
        const struct ggml_htp_cache_slot_snapshot * slot) {
    if (buffer == NULL || capacity == 0 || slot == NULL) {
        return -1;
    }
    return snprintf(
        buffer, capacity,
        "htp_memory_ledger_slot key=%s branch=%d generation=%llu fd=%d "
        "logical_bytes=%zu allocation_bytes=%zu import_identity=%llu",
        slot->cache_key, slot->branch,
        (unsigned long long) slot->generation, slot->fd, slot->logical_size,
        slot->allocation_size,
        (unsigned long long) slot->opencl_import_identity);
}

static inline bool ggml_backend_htp_memory_ledger_trace_enabled(void) {
    const char * value = getenv("SD_HTP_SPARSE_OPENCL_TRACE");
    return value != NULL && value[0] != '\0' && strcmp(value, "0") != 0;
}

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

GGML_BACKEND_API bool ggml_backend_htp_async_cache_slot_snapshot(
    const ggml_backend_htp_async_cache_slot * slot,
    struct ggml_htp_cache_slot_snapshot * snapshot);

GGML_BACKEND_API bool ggml_backend_htp_set_graph_memory_ledger(
    ggml_backend_t backend,
    const struct ggml_htp_memory_ledger_snapshot * snapshot);

GGML_BACKEND_API bool ggml_backend_htp_get_last_memory_ledger(
    ggml_backend_t backend,
    struct ggml_htp_memory_ledger_snapshot * snapshot);

GGML_BACKEND_API bool ggml_backend_htp_get_lifetime_snapshot(
    ggml_backend_t backend,
    struct ggml_htp_lifetime_snapshot * snapshot);

// Persistent OpenCL aliases are needed only by the OpenCL sparse producer
// route. Fused and HTP+CPU routes consume the same RPCMEM cache directly.
GGML_BACKEND_API bool ggml_backend_htp_opencl_persistent_import_required(void);

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
