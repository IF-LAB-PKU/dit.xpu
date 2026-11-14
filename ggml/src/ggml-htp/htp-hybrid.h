#pragma once

#include "ggml-cpu.h"

#ifdef __cplusplus
extern "C" {
#endif

// HTP owns a different worker implementation from ggml-cpu (which may use
// OpenMP). Never pass these pools to the generic CPU lifecycle functions.
struct ggml_threadpool * ggml_htp_threadpool_new(struct ggml_threadpool_params * params);
void ggml_htp_threadpool_free(struct ggml_threadpool * pool);
struct ggml_cplan ggml_htp_graph_plan(const struct ggml_cgraph * graph, int n_threads,
                                    struct ggml_threadpool * pool);
enum ggml_status ggml_graph_compute_htp_hybrid(struct ggml_cgraph * graph, struct ggml_cplan * plan);

#ifdef __cplusplus
}
#endif
