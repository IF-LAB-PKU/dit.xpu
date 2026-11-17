#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-htp.h"
#include <cstdio>
#include <vector>

// A packed matrix with an unmappable CPU activation must report an error.
// Ordinary row-major matrices still have a valid CPU fallback.
int main() {
    auto* htp = ggml_backend_htp_init();
    auto* cpu = ggml_backend_init_by_name("CPU", nullptr);
    GGML_ASSERT(htp && cpu && ggml_backend_htp_runtime_available());
    const ggml_init_params p{ggml_tensor_overhead()*8 + ggml_graph_overhead(), nullptr, true};
    auto* wc = ggml_init(p);
    auto* ac = ggml_init(p);
    auto* gc = ggml_init(p);
    auto* w = ggml_new_tensor_2d(wc, GGML_TYPE_F16, 32, 32);
    auto* a = ggml_new_tensor_2d(ac, GGML_TYPE_F32, 32, 1);
    auto* y = ggml_mul_mat(gc, w, a);
    auto* wb = ggml_backend_alloc_ctx_tensors(wc, htp);
    auto* ab = ggml_backend_alloc_ctx_tensors(ac, cpu);
    auto* gb = ggml_backend_alloc_ctx_tensors(gc, htp);
    GGML_ASSERT(wb && ab && gb);
    std::vector<ggml_fp16_t> weights(1024, ggml_fp32_to_fp16(1.f));
    std::vector<float> input(32, 1.f), output(32);
    ggml_backend_tensor_set(w, weights.data(), 0, ggml_nbytes(w));
    ggml_backend_tensor_set(a, input.data(), 0, ggml_nbytes(a));
    auto* graph = ggml_new_graph(gc);
    ggml_build_forward_expand(graph, y);
    ggml_backend_htp_register_packed_f16_weight(w, w);
    ggml_backend_htp_set_hybrid_enabled(true);
    GGML_ASSERT(ggml_backend_graph_compute(htp, graph) != GGML_STATUS_SUCCESS);
    ggml_backend_htp_unregister_packed_f16_weight(w);
    GGML_ASSERT(ggml_backend_graph_compute(htp, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_tensor_get(y, output.data(), 0, ggml_nbytes(y));
    for (float value : output) GGML_ASSERT(value == 32.f);
    ggml_backend_htp_set_hybrid_enabled(false);
    ggml_backend_buffer_free(gb);
    ggml_backend_buffer_free(ab);
    ggml_backend_buffer_free(wb);
    ggml_free(gc); ggml_free(ac); ggml_free(wc);
    ggml_backend_free(cpu); ggml_backend_free(htp);
    std::puts("Offline packed CPU fallback rejection and row-major recovery passed");
}
