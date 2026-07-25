#ifndef GGML_OPENCL_HTP_H
#define GGML_OPENCL_HTP_H

#include "ggml-backend.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ggml_opencl_htp_sidecar ggml_opencl_htp_sidecar;
typedef struct ggml_opencl_htp_graph ggml_opencl_htp_graph;

struct ggml_opencl_qcom_requirements {
    size_t ext_mem_padding;
    size_t page_size;
};

struct ggml_opencl_htp_import_descriptor {
    ggml_backend_buffer_t backing;
    void * base;
    int fd;
    size_t logical_size;
    size_t allocation_size;
};

enum ggml_opencl_htp_result {
    GGML_OPENCL_HTP_OK          = 0,
    GGML_OPENCL_HTP_UNSUPPORTED = 1,
    GGML_OPENCL_HTP_ERROR       = 2,
    GGML_OPENCL_HTP_NO_WORK     = 3,
};

// A persistent import is owned by the sidecar rather than a graph. Graphs that
// reference the same backing retain this cl_mem without importing the dmabuf again.
GGML_BACKEND_API enum ggml_opencl_htp_result
ggml_backend_opencl_htp_persistent_import_acquire(
    ggml_opencl_htp_sidecar * sidecar,
    const struct ggml_opencl_htp_import_descriptor * descriptor,
    const char * cache_key,
    uint64_t generation,
    uint64_t * import_identity,
    char * reason,
    size_t reason_capacity);

GGML_BACKEND_API enum ggml_opencl_htp_result
ggml_backend_opencl_htp_persistent_import_release(
    ggml_opencl_htp_sidecar * sidecar,
    const char * cache_key,
    uint64_t generation,
    char * reason,
    size_t reason_capacity);

enum ggml_opencl_htp_cache_state {
    GGML_OPENCL_HTP_CACHE_EMPTY = 0,
    GGML_OPENCL_HTP_CACHE_ALLOCATED,
    GGML_OPENCL_HTP_CACHE_QUEUED,
    GGML_OPENCL_HTP_CACHE_SUBMITTED,
    GGML_OPENCL_HTP_CACHE_READY,
    GGML_OPENCL_HTP_CACHE_FAILED,
};

struct ggml_opencl_htp_residual_dispatch {
    const struct ggml_tensor * dense_packed;
    const struct ggml_tensor * sparse_packed_producer;
    const struct ggml_tensor * residual_destination;
    const struct ggml_tensor * rho_destination;
    const char * residual_cache_key;
    const char * rho_cache_key;
    uint64_t generation;
};

GGML_BACKEND_API ggml_opencl_htp_sidecar * ggml_backend_opencl_htp_sidecar_create(
    struct ggml_opencl_qcom_requirements * requirements,
    char * reason,
    size_t reason_capacity);

GGML_BACKEND_API void ggml_backend_opencl_htp_sidecar_free(
    ggml_opencl_htp_sidecar * sidecar);

GGML_BACKEND_API const char * ggml_backend_opencl_htp_sidecar_platform_vendor(
    const ggml_opencl_htp_sidecar * sidecar);

GGML_BACKEND_API const char * ggml_backend_opencl_htp_sidecar_device_name(
    const ggml_opencl_htp_sidecar * sidecar);

GGML_BACKEND_API const char * ggml_backend_opencl_htp_sidecar_device_extensions(
    const ggml_opencl_htp_sidecar * sidecar);

GGML_BACKEND_API bool ggml_backend_opencl_htp_sidecar_has_extension(
    const ggml_opencl_htp_sidecar * sidecar,
    const char * extension);

GGML_BACKEND_API bool ggml_backend_opencl_htp_sidecar_queue_is_in_order(
    const ggml_opencl_htp_sidecar * sidecar);

GGML_BACKEND_API size_t ggml_backend_opencl_htp_sidecar_live_persistent_import_count(
    const ggml_opencl_htp_sidecar * sidecar);

GGML_BACKEND_API size_t ggml_backend_opencl_htp_sidecar_live_cache_event_count(
    const ggml_opencl_htp_sidecar * sidecar);

GGML_BACKEND_API enum ggml_opencl_htp_result ggml_backend_opencl_htp_graph_create(
    ggml_opencl_htp_sidecar * sidecar,
    ggml_opencl_htp_graph ** graph,
    char * reason,
    size_t reason_capacity);

// Preflight freezes the sparse nodes routed to the sidecar and imports each distinct backing.
// Descriptors and graph tensors are borrowed. Their backing storage and fd/base metadata must
// remain valid until graph finish/free. On any non-OK result, *graph is null and partial imports
// have been released; no OpenCL command has been enqueued.
GGML_BACKEND_API enum ggml_opencl_htp_result ggml_backend_opencl_htp_graph_preflight(
    ggml_opencl_htp_sidecar * sidecar,
    const struct ggml_cgraph * cgraph,
    const struct ggml_opencl_htp_import_descriptor * descriptors,
    size_t descriptor_count,
    ggml_opencl_htp_graph ** graph,
    char * reason,
    size_t reason_capacity);

GGML_BACKEND_API size_t ggml_backend_opencl_htp_graph_routed_node_count(
    const ggml_opencl_htp_graph * graph);

GGML_BACKEND_API bool ggml_backend_opencl_htp_graph_routes_node(
    const ggml_opencl_htp_graph * graph,
    const struct ggml_tensor * node);

// Validates and freezes an exact-rho residual job. No OpenCL work is enqueued.
// Tensor pointers are borrowed by the graph-local prepared job only. Persistent
// cache lookup uses the copied key and generation, never a graph tensor pointer.
GGML_BACKEND_API enum ggml_opencl_htp_result
ggml_backend_opencl_htp_graph_prepare_exact_rho_residual(
    ggml_opencl_htp_graph * graph,
    const struct ggml_opencl_htp_residual_dispatch * dispatch,
    char * reason,
    size_t reason_capacity);

// Marks a prepared job queued. Kernel submission is implemented separately.
GGML_BACKEND_API enum ggml_opencl_htp_result
ggml_backend_opencl_htp_graph_queue_exact_rho_residual(
    ggml_opencl_htp_graph * graph,
    const char * residual_cache_key,
    uint64_t generation,
    char * reason,
    size_t reason_capacity);

GGML_BACKEND_API enum ggml_opencl_htp_result
ggml_backend_opencl_htp_graph_queue_exact_rho_residual_after_dense(
    ggml_opencl_htp_graph * graph,
    const struct ggml_tensor * dense_producer,
    char * reason,
    size_t reason_capacity);

GGML_BACKEND_API enum ggml_opencl_htp_result
ggml_backend_opencl_htp_graph_compute_exact_rho_residual_sync(
    ggml_opencl_htp_graph * graph,
    const struct ggml_tensor * residual_node,
    char * reason,
    size_t reason_capacity);

GGML_BACKEND_API enum ggml_opencl_htp_result ggml_backend_opencl_htp_cache_wait(
    ggml_opencl_htp_sidecar * sidecar,
    const char * cache_key,
    uint64_t generation,
    bool * waited,
    char * reason,
    size_t reason_capacity);

GGML_BACKEND_API enum ggml_opencl_htp_result ggml_backend_opencl_htp_cache_query(
    const ggml_opencl_htp_sidecar * sidecar,
    const char * cache_key,
    uint64_t generation,
    enum ggml_opencl_htp_cache_state * state,
    char * reason,
    size_t reason_capacity);

// The sidecar must outlive the graph. Imported backing storage and its borrowed fd/base must
// remain valid until graph finish or free; the descriptor itself is copied during import.
GGML_BACKEND_API enum ggml_opencl_htp_result ggml_backend_opencl_htp_graph_import(
    ggml_opencl_htp_graph * graph,
    const struct ggml_opencl_htp_import_descriptor * descriptor,
    char * reason,
    size_t reason_capacity);

GGML_BACKEND_API enum ggml_opencl_htp_result ggml_backend_opencl_htp_graph_fill(
    ggml_opencl_htp_graph * graph,
    ggml_backend_buffer_t backing,
    size_t offset,
    size_t size,
    uint32_t pattern,
    char * reason,
    size_t reason_capacity);

GGML_BACKEND_API enum ggml_opencl_htp_result ggml_backend_opencl_htp_graph_fill_tensor(
    ggml_opencl_htp_graph * graph,
    const struct ggml_tensor * tensor,
    uint32_t pattern,
    char * reason,
    size_t reason_capacity);

GGML_BACKEND_API enum ggml_opencl_htp_result
ggml_backend_opencl_htp_graph_compute_block_sparse_attn_async(
    ggml_opencl_htp_graph * graph,
    const struct ggml_tensor * node,
    char * reason,
    size_t reason_capacity);

GGML_BACKEND_API enum ggml_opencl_htp_result
ggml_backend_opencl_htp_graph_compute_block_sparse_attn_early_async(
    ggml_opencl_htp_graph * graph,
    const struct ggml_tensor * node,
    char * reason,
    size_t reason_capacity);

GGML_BACKEND_API enum ggml_opencl_htp_result
ggml_backend_opencl_htp_graph_stage_block_sparse_attn_mask(
    ggml_opencl_htp_graph * graph,
    const struct ggml_tensor * node,
    char * reason,
    size_t reason_capacity);

GGML_BACKEND_API enum ggml_opencl_htp_result
ggml_backend_opencl_htp_graph_start_submission_thread(
    ggml_opencl_htp_graph * graph,
    char * reason,
    size_t reason_capacity);

GGML_BACKEND_API enum ggml_opencl_htp_result
ggml_backend_opencl_htp_graph_queue_block_sparse_attn(
    ggml_opencl_htp_graph * graph,
    const struct ggml_tensor * node,
    char * reason,
    size_t reason_capacity);

GGML_BACKEND_API enum ggml_opencl_htp_result
ggml_backend_opencl_htp_graph_stop_submission_thread(
    ggml_opencl_htp_graph * graph,
    char * reason,
    size_t reason_capacity);

GGML_BACKEND_API enum ggml_opencl_htp_result
ggml_backend_opencl_htp_graph_cancel_submission_thread(
    ggml_opencl_htp_graph * graph,
    char * reason,
    size_t reason_capacity);

GGML_BACKEND_API enum ggml_opencl_htp_result
ggml_backend_opencl_htp_graph_wait_for_tensor(
    ggml_opencl_htp_graph * graph,
    const struct ggml_tensor * producer,
    bool * waited,
    char * reason,
    size_t reason_capacity);

GGML_BACKEND_API bool
ggml_backend_opencl_htp_graph_sparse_output_is_residual_owned(
    const ggml_opencl_htp_graph * graph,
    const struct ggml_tensor * sparse_producer);

GGML_BACKEND_API enum ggml_opencl_htp_result
ggml_backend_opencl_htp_graph_wait_for_sparse_input_lifetime(
    ggml_opencl_htp_graph * graph,
    const struct ggml_tensor * sparse_producer,
    bool * waited,
    char * reason,
    size_t reason_capacity);

GGML_BACKEND_API enum ggml_opencl_htp_result
ggml_backend_opencl_htp_graph_compute_block_sparse_attn_sync(
    ggml_opencl_htp_graph * graph,
    const struct ggml_tensor * node,
    char * reason,
    size_t reason_capacity);

GGML_BACKEND_API enum ggml_opencl_htp_result ggml_backend_opencl_htp_graph_finish(
    ggml_opencl_htp_graph * graph,
    char * reason,
    size_t reason_capacity);

GGML_BACKEND_API enum ggml_opencl_htp_result ggml_backend_opencl_htp_graph_defer(
    ggml_opencl_htp_graph * graph,
    bool * pending,
    char * reason,
    size_t reason_capacity);

GGML_BACKEND_API enum ggml_opencl_htp_result
ggml_backend_opencl_htp_graph_deferred_ready(
    ggml_opencl_htp_graph * graph,
    bool * ready,
    char * reason,
    size_t reason_capacity);

GGML_BACKEND_API enum ggml_opencl_htp_result
ggml_backend_opencl_htp_graph_wait_deferred(
    ggml_opencl_htp_graph * graph,
    char * reason,
    size_t reason_capacity);

GGML_BACKEND_API enum ggml_opencl_htp_result
ggml_backend_opencl_htp_graph_reap_deferred(
    ggml_opencl_htp_graph * graph,
    char * reason,
    size_t reason_capacity);

GGML_BACKEND_API void ggml_backend_opencl_htp_graph_free(
    ggml_opencl_htp_graph ** graph);

GGML_BACKEND_API size_t ggml_backend_opencl_htp_graph_live_alias_count(
    const ggml_opencl_htp_graph * graph);

GGML_BACKEND_API size_t ggml_backend_opencl_htp_graph_live_event_count(
    const ggml_opencl_htp_graph * graph);

GGML_BACKEND_API size_t ggml_backend_opencl_htp_graph_pending_sparse_event_count(
    const ggml_opencl_htp_graph * graph);

GGML_BACKEND_API size_t ggml_backend_opencl_htp_graph_pending_sparse_submission_count(
    const ggml_opencl_htp_graph * graph);

GGML_BACKEND_API size_t ggml_backend_opencl_htp_graph_submitted_sparse_count(
    const ggml_opencl_htp_graph * graph);

GGML_BACKEND_API size_t ggml_backend_opencl_htp_graph_pending_residual_submission_count(
    const ggml_opencl_htp_graph * graph);

GGML_BACKEND_API size_t ggml_backend_opencl_htp_graph_pending_residual_event_count(
    const ggml_opencl_htp_graph * graph);

// Diagnostic fault injection used by the shared-memory lifecycle test.
GGML_BACKEND_API void ggml_backend_opencl_htp_graph_test_fail_next_event_cache(
    ggml_opencl_htp_graph * graph);

GGML_BACKEND_API void ggml_backend_opencl_htp_graph_test_fail_next_sparse_enqueue(
    ggml_opencl_htp_graph * graph);

GGML_BACKEND_API void ggml_backend_opencl_htp_graph_test_fail_next_sparse_wait(
    ggml_opencl_htp_graph * graph);

GGML_BACKEND_API void ggml_backend_opencl_htp_graph_test_fail_next_rho_enqueue(
    ggml_opencl_htp_graph * graph);

GGML_BACKEND_API void ggml_backend_opencl_htp_graph_test_fail_next_residual_enqueue(
    ggml_opencl_htp_graph * graph);

GGML_BACKEND_API void ggml_backend_opencl_htp_graph_test_fail_next_residual_wait(
    ggml_opencl_htp_graph * graph);

GGML_BACKEND_API void ggml_backend_opencl_htp_sidecar_test_fail_next_residual_wait(
    ggml_opencl_htp_sidecar * sidecar);

#ifdef __cplusplus
}
#endif

#endif // GGML_OPENCL_HTP_H
