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

GGML_BACKEND_API bool ggml_backend_buft_is_rpcmem(ggml_backend_buffer_type_t buft);
// 暴露 hybrid 开关给上层推理线路，避免只能依赖全局环境变量控制 HTP 混合执行。
GGML_BACKEND_API void ggml_backend_htp_set_hybrid_enabled(bool enabled);
GGML_BACKEND_API bool ggml_backend_htp_get_hybrid_enabled(void);

#ifdef __cplusplus
}
#endif
