#ifndef GGML_OPENCL_H
#define GGML_OPENCL_H

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

//
// backend API
//
GGML_BACKEND_API ggml_backend_t ggml_backend_opencl_init(void);
GGML_BACKEND_API bool ggml_backend_is_opencl(ggml_backend_t backend);
GGML_BACKEND_API void ggml_backend_opencl_set_attention_profiling_enabled(bool enabled);
GGML_BACKEND_API bool ggml_backend_opencl_attention_profiling_enabled(void);
GGML_BACKEND_API void ggml_backend_opencl_reset_attention_profile(void);
GGML_BACKEND_API void ggml_backend_opencl_record_attention_elapsed_us(int64_t elapsed_us);
GGML_BACKEND_API int64_t ggml_backend_opencl_get_last_attention_elapsed_us(void);

struct ggml_opencl_attention_profile {
    int64_t host_total_us;
    int64_t validation_us;
    int64_t binding_us;
    int64_t args_us;
    int64_t enqueue_us;
    int64_t flush_us;
    int64_t wait_us;
    uint64_t event_queued_ns;
    uint64_t event_submit_ns;
    uint64_t event_start_ns;
    uint64_t event_end_ns;
};

GGML_BACKEND_API void ggml_backend_opencl_record_attention_profile(
    const struct ggml_opencl_attention_profile * profile);
GGML_BACKEND_API bool ggml_backend_opencl_get_last_attention_profile(
    struct ggml_opencl_attention_profile * profile);

GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_opencl_buffer_type(void);
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_opencl_host_buffer_type(void);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_opencl_reg(void);

#ifdef  __cplusplus
}
#endif

#endif // GGML_OPENCL_H
