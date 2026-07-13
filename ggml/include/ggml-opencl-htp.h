#ifndef GGML_OPENCL_HTP_H
#define GGML_OPENCL_HTP_H

#include "ggml-backend.h"

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ggml_opencl_htp_sidecar ggml_opencl_htp_sidecar;
typedef struct ggml_opencl_htp_graph ggml_opencl_htp_graph;

struct ggml_opencl_qcom_requirements {
    size_t ext_mem_padding;
    size_t page_size;
};

enum ggml_opencl_htp_result {
    GGML_OPENCL_HTP_OK          = 0,
    GGML_OPENCL_HTP_UNSUPPORTED = 1,
    GGML_OPENCL_HTP_ERROR       = 2,
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

#ifdef __cplusplus
}
#endif

#endif // GGML_OPENCL_HTP_H
