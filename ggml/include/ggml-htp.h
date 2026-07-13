#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

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

GGML_BACKEND_API bool ggml_backend_buft_is_rpcmem(ggml_backend_buffer_type_t buft);
GGML_BACKEND_API void ggml_backend_htp_set_n_threads(ggml_backend_t backend, int n_threads);
GGML_BACKEND_API void ggml_backend_htp_pack_f16_weight(ggml_fp16_t * dst, const ggml_fp16_t * src, int64_t k, int64_t n);
GGML_BACKEND_API void ggml_backend_htp_register_packed_f16_weight(const struct ggml_tensor * original,
                                                                  struct ggml_tensor * packed);
GGML_BACKEND_API void ggml_backend_htp_unregister_packed_f16_weight(const struct ggml_tensor * original);
GGML_BACKEND_API struct ggml_tensor * ggml_backend_htp_get_packed_f16_weight(const struct ggml_tensor * original);
// 暴露 hybrid 开关给上层推理线路，避免只能依赖全局环境变量控制 HTP 混合执行。
GGML_BACKEND_API void ggml_backend_htp_set_hybrid_enabled(bool enabled);
GGML_BACKEND_API bool ggml_backend_htp_get_hybrid_enabled(void);

#ifdef __cplusplus
}
#endif
