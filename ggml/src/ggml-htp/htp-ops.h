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
bool htp_ops_support_op_runtime(const struct ggml_tensor * dst);
int  htp_ops_compute_op(struct ggml_compute_params * params, struct ggml_tensor * dst);
int  htp_ops_resolve_halo_kv_load_mode(void);
bool htp_ops_opencl_routes_node(const struct ggml_tensor * dst);
bool htp_ops_opencl_async_enabled(void);
bool htp_ops_opencl_node_needs_wait(const struct ggml_tensor * dst);
int  htp_ops_opencl_wait_node(const struct ggml_tensor * dst);
void htp_ops_opencl_wait_node_completed(const struct ggml_tensor * dst);
int  htp_ops_opencl_wait_producer(const struct ggml_tensor * producer);
bool htp_ops_opencl_sparse_is_residual_owned(const struct ggml_tensor * producer);
int  htp_ops_opencl_wait_sparse_input_lifetime(const struct ggml_tensor * producer);
int  htp_ops_opencl_compute_node(struct ggml_tensor * dst);
int  htp_ops_opencl_compute_residual_node(struct ggml_tensor * dst);
int  htp_ops_opencl_queue_residual_after_dense(struct ggml_tensor * dense);
void htp_ops_opencl_trace_next_node_begin(const struct ggml_tensor * node);
int  htp_ops_opencl_early_submit_begin(const struct ggml_cgraph * graph);
void htp_ops_opencl_early_submit_end(void);
bool htp_ops_opencl_early_submit_tracks_node(const struct ggml_tensor * node);
bool htp_ops_opencl_early_submit_is_prerequisite(const struct ggml_tensor * node);
int  htp_ops_opencl_early_submit_node_completed(const struct ggml_tensor * node);
bool htp_ops_opencl_early_submit_was_submitted(const struct ggml_tensor * node);
void htp_ops_opencl_early_submit_mark_submitted(const struct ggml_tensor * node);

#ifdef __cplusplus
}
#endif
