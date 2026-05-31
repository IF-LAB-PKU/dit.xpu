// HTP 算子分发的公共头文件。
// 声明两个核心函数：
// - htp_ops_support_op: 判断某个 GGML 张量算子能否在 HTP 上执行
// - htp_ops_compute_op:  通过共享内存消息将算子分发到 DSP 侧执行

#pragma once

#include <stdbool.h>

#include "ggml-cpu/ggml-cpu-impl.h"
#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

bool htp_ops_support_op(const struct ggml_tensor * dst);
int  htp_ops_compute_op(struct ggml_compute_params * params, struct ggml_tensor * dst);

#ifdef __cplusplus
}
#endif
