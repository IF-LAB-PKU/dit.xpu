#include "ggml_extend.hpp"
#include <cmath>

struct ViewCutRunner : GGMLRunner {
    ggml_tensor* weight;
    ViewCutRunner(ggml_backend_t cpu) : GGMLRunner(cpu, cpu) {
        weight = ggml_new_tensor_2d(params_ctx, GGML_TYPE_F32, 32, 32);
        GGML_ASSERT(alloc_params_buffer());
        std::vector<float> data(1024);
        for (int i = 0; i < 1024; ++i) data[i] = std::sin(i * .17f) * .04f;
        ggml_backend_tensor_set(weight, data.data(), 0, data.size() * sizeof(float));
    }
    std::string get_desc() override { return "view-cut-test"; }
    sd::Tensor<float> run() {
        sd::Tensor<float> input({32, 7, 2}), bias({64});
        for (int i = 0; i < input.numel(); ++i) input[i] = std::sin(i * .1f);
        for (int i = 0; i < bias.numel(); ++i) bias[i] = std::cos(i * .1f);
        auto graph = [&]() {
            auto* source = make_input(input);
            auto* x = ggml_view_2d(compute_ctx, source, 32, 7, source->nb[1], 32 * 7 * sizeof(float));
            auto* b = ggml_view_1d(compute_ctx, make_input(bias), 32, 32 * sizeof(float));
            sd::ggml_graph_cut::mark_graph_cut(x, "prelude", "x");
            sd::ggml_graph_cut::mark_graph_cut(b, "bias", "shared");
            for (int i = 0; i < 3; ++i) {
                x = ggml_silu(compute_ctx, ggml_add(compute_ctx, ggml_mul_mat(compute_ctx, weight, x), b));
                sd::ggml_graph_cut::mark_graph_cut(x, "layer." + std::to_string(i), "x");
            }
            auto* out = ggml_scale(compute_ctx, x, .5f);
            auto* gf = ggml_new_graph(compute_ctx);
            ggml_build_forward_expand(gf, out);
            return gf;
        };
        auto result = compute<float>(graph, 2, true);
        GGML_ASSERT(result && !result->empty());
        return std::move(*result);
    }
};
int main() {
    auto* cpu = ggml_backend_cpu_init();
    GGML_ASSERT(cpu);
    {
        unsetenv("SD_HTP_PACKED_STAGING");
        ViewCutRunner runner(cpu);
        const auto expected = runner.run();
        setenv("SD_HTP_PACKED_STAGING", "1", 1);
        runner.set_max_graph_vram_bytes(1); // Keep every cut in its own segment.
        for (int repeat = 0; repeat < 2; ++repeat) {
            const auto actual = runner.run();
            GGML_ASSERT(actual.shape() == expected.shape());
            for (int64_t i = 0; i < actual.numel(); ++i) {
                GGML_ASSERT(std::isfinite(actual[i]) && std::abs(actual[i] - expected[i]) < 1e-6f);
            }
        }
    }
    ggml_backend_free(cpu);
    std::puts("Graph-cut views, offsets, shared inputs and repeated execution passed");
    return 0;
}
