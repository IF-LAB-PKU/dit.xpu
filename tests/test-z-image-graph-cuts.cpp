#include "z_image.hpp"
#include <cmath>

// Exercise the actual model builder with small widths. Refiner weights must
// never be pulled into the embedding segment by a reused group name.
struct ZImageCutRunner : GGMLRunner {
    ZImage::ZImageModel model;
    explicit ZImageCutRunner(ggml_backend_t cpu) : GGMLRunner(cpu, cpu) {
        ZImage::ZImageParams p;
        // AdaLN consumes the model's fixed 256-wide timestep embedding.
        p.hidden_size = 256;
        p.head_dim = 128;
        p.num_heads = p.num_kv_heads = 2;
        p.cap_feat_dim = p.multiple_of = 32;
        p.num_layers = 2;
        model = ZImage::ZImageModel(p);
        model.init(params_ctx, {}, "model.diffusion_model");
        GGML_ASSERT(alloc_params_buffer());
        for (auto* t = ggml_get_first_tensor(params_ctx); t; t = ggml_get_next_tensor(params_ctx, t)) {
            std::vector<float> values(ggml_nelements(t));
            for (size_t i = 0; i < values.size(); ++i)
                values[i] = std::strstr(t->name, "norm") ? 1.f : .02f * std::sin(float(i)*.13f);
            if (t->type == GGML_TYPE_F32) {
                ggml_backend_tensor_set(t, values.data(), 0, ggml_nbytes(t));
            } else {
                GGML_ASSERT(t->type == GGML_TYPE_F16);
                std::vector<ggml_fp16_t> half(values.size());
                ggml_fp32_to_fp16_row(values.data(), half.data(), half.size());
                ggml_backend_tensor_set(t, half.data(), 0, ggml_nbytes(t));
            }
        }
    }
    std::string get_desc() override { return "z-image-cut-test"; }
    sd::Tensor<float> run() {
        sd::Tensor<float> image({64, 32, 1}), text({32, 4, 1}), time({1}), pe({2, 2, 64, 64});
        for (int64_t i = 0; i < image.numel(); ++i) image[i] = std::sin(float(i)*.1f);
        for (int64_t i = 0; i < text.numel(); ++i) text[i] = std::cos(float(i)*.07f);
        time[0] = 500.f;
        for (int64_t i = 0; i < pe.numel(); ++i) pe[i] = i%4 == 0 || i%4 == 3 ? 1.f : 0.f;
        auto graph = [&]() {
            auto ctx = get_context();
            auto* out = model.forward_core(&ctx, make_input(image), make_input(time), make_input(text), make_input(pe));
            auto* gf = ggml_new_graph_custom(compute_ctx, ZImage::Z_IMAGE_GRAPH_SIZE, false);
            ggml_build_forward_expand(gf, out);
            std::unordered_set<const ggml_tensor*> weights;
            for (auto* t = ggml_get_first_tensor(params_ctx); t; t = ggml_get_next_tensor(params_ctx, t)) weights.insert(t);
            auto plan = sd::ggml_graph_cut::build_plan(runtime_backend, gf, weights, "z-image-cut-test");
            GGML_ASSERT(plan.valid && plan.has_cuts);
            int refiners = 0;
            bool join = false;
            for (const auto& segment : plan.segments) {
                if (segment.group_name.find("z_image.context_refiner.") == 0 ||
                    segment.group_name.find("z_image.noise_refiner.") == 0) ++refiners;
                if (segment.group_name == "z_image.prelude") {
                    for (auto* t : sd::ggml_graph_cut::param_tensors(gf, segment))
                        GGML_ASSERT(!std::strstr(t->name, "refiner."));
                }
                if (segment.group_name == "z_image.joint_input") {
                    GGML_ASSERT(refiners == 4);
                    GGML_ASSERT(sd::ggml_graph_cut::param_tensors(gf, segment).empty());
                    join = true;
                }
            }
            GGML_ASSERT(join);
            return gf;
        };
        auto out = compute<float>(graph, 2, true);
        GGML_ASSERT(out && !out->empty());
        return std::move(*out);
    }
};

int main() {
    auto* cpu = ggml_backend_cpu_init();
    GGML_ASSERT(cpu);
    {
        unsetenv("SD_HTP_PACKED_STAGING");
        ZImageCutRunner runner(cpu);
        auto expected = runner.run();
        setenv("SD_HTP_PACKED_STAGING", "1", 1);
        runner.set_max_graph_vram_bytes(1);
        for (int repeat = 0; repeat < 2; ++repeat) {
            auto actual = runner.run();
            GGML_ASSERT(actual.shape() == expected.shape());
            for (int64_t i = 0; i < actual.numel(); ++i)
                GGML_ASSERT(std::isfinite(actual[i]) && std::abs(actual[i]-expected[i]) < 1e-6f);
        }
    }
    ggml_backend_free(cpu);
    std::puts("Z-Image prelude/refiner separation and segmented output parity passed");
}
