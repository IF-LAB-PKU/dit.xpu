// Explicit real-model comparison: run CPU/raw and HTP/offline separately,
// then compare the unmodified F32 predictions. Not an ordinary CTest target.
#include "z_image.hpp"
#include <chrono>

static void log_line(sd_log_level_t, const char* text, void*) { std::fputs(text, stderr); }

struct ModulationRunner : GGMLRunner {
    Linear linear{256, 15360, true};
    ModulationRunner(ggml_backend_t cpu, const String2TensorStorage& storage) : GGMLRunner(cpu, cpu) {
        linear.init(params_ctx, storage, "model.diffusion_model.noise_refiner.0.adaLN_modulation.0");
    }
    std::string get_desc() override { return "z-image-modulation-test"; }
    sd::Tensor<float> run() {
        sd::Tensor<float> input({256, 1});
        for (int64_t i = 0; i < input.numel(); ++i) input[i] = .1f*std::sin(float(i)*.13f);
        auto build = [&]() {
            auto ctx = get_context();
            auto* out = linear.forward(&ctx, make_input(input));
            auto* gf = ggml_new_graph(compute_ctx);
            ggml_build_forward_expand(gf, out);
            return gf;
        };
        auto out = compute<float>(build, 4, true);
        GGML_ASSERT(out && !out->empty());
        return std::move(*out);
    }
};

struct ComparisonRunner : ZImage::ZImageRunner {
    using ZImage::ZImageRunner::ZImageRunner;
    sd::Tensor<float> diagnostic(const sd::Tensor<float>& x, const sd::Tensor<float>& time, const sd::Tensor<float>& text) {
        auto build = [&]() {
            auto* full = build_graph(x, time, text);
            auto* graph = ggml_new_graph_custom(compute_ctx, ZImage::Z_IMAGE_GRAPH_SIZE, false);
            ggml_tensor* joined = nullptr;
            for (int i = 0; i < ggml_graph_n_nodes(full); ++i) {
                auto* t = ggml_graph_node(full, i);
                std::string name = t->name;
                const bool projection = t->op == GGML_OP_MUL_MAT && t->src[0] &&
                    std::string(t->src[0]->name).find("model.diffusion_model.noise_refiner.0.") == 0;
                if (projection) name = t->src[0]->name;
                if (!projection && name.find(" (reshaped)") != std::string::npos) continue;
                if (!projection && name.find("ggml_runner_cut:z_image.prelude|") != 0 &&
                    name.find("ggml_runner_cut:z_image.context_refiner.") != 0 &&
                    name.find("ggml_runner_cut:z_image.noise_refiner.") != 0) continue;
                std::printf("diagnostic tensor=%s elements=%lld\n", name.c_str(), (long long)ggml_nelements(t));
                auto* flat = ggml_reshape_1d(compute_ctx, ggml_cont(compute_ctx, t), ggml_nelements(t));
                joined = joined ? ggml_concat(compute_ctx, joined, flat, 0) : flat;
                if (projection && name.find("adaLN_modulation") != std::string::npos) {
                    auto* input = t->src[1];
                    std::printf("diagnostic tensor=modulation-input elements=%lld\n", (long long)ggml_nelements(input));
                    auto* input_flat = ggml_reshape_1d(compute_ctx, ggml_cont(compute_ctx, input), ggml_nelements(input));
                    joined = ggml_concat(compute_ctx, joined, input_flat, 0);
                }
            }
            GGML_ASSERT(joined);
            ggml_build_forward_expand(graph, joined);
            return graph;
        };
        auto out = GGMLRunner::compute<float>(build, 4, true);
        GGML_ASSERT(out && !out->empty());
        return std::move(*out);
    }
};

int main(int argc, char** argv) {
    if ((argc != 4 && argc != 5) ||
        (argc == 5 && std::string(argv[4]) != "--projection" && std::string(argv[4]) != "--diagnostic") ||
        (std::string(argv[2]) != "cpu" && std::string(argv[2]) != "cpu-cut" && std::string(argv[2]) != "htp")) {
        std::fprintf(stderr, "usage: %s MODEL cpu|cpu-cut|htp OUTPUT.f32 [--diagnostic|--projection]\n", argv[0]);
        return 2;
    }
    sd_set_log_callback(log_line, nullptr);
    auto* cpu = ggml_backend_init_by_name("CPU", nullptr);
    const bool accelerated = std::string(argv[2]) == "htp";
    auto* htp = accelerated ? ggml_backend_init_by_name("HTP", nullptr) : nullptr;
    GGML_ASSERT(cpu && (!accelerated || htp));
    if (htp) {
        // Match production CLI layout construction as well as runner routing.
        setenv("SD_HTP_OPS", "1", 1);
        setenv("SD_HTP_ATTENTION", "1", 0);
    }
    {
        ModelLoader loader;
        GGML_ASSERT(loader.init_from_file_and_convert_name(argv[1], "model.diffusion_model.", VERSION_Z_IMAGE));
        if (argc == 5 && std::string(argv[4]) == "--projection") {
            ModulationRunner runner(cpu, loader.get_tensor_storage_map());
            if (htp) { setenv("SD_HTP_COMPUTE", "1", 1); runner.set_htp_backend(htp); }
            GGML_ASSERT(runner.alloc_params_buffer());
            std::map<std::string, ggml_tensor*> params;
            runner.linear.get_param_tensors(params, "model.diffusion_model.noise_refiner.0.adaLN_modulation.0");
            GGML_ASSERT(loader.load_tensor_subset(params, 4));
            auto out = runner.run();
            std::ofstream file(argv[3], std::ios::binary);
            GGML_ASSERT(file.write(reinterpret_cast<const char*>(out.data()), out.numel()*sizeof(float)));
        } else {
            ComparisonRunner runner(cpu, cpu, loader.get_tensor_storage_map(), "model.diffusion_model");
            if (std::string(argv[2]) == "cpu-cut") runner.set_max_graph_vram_bytes(512 * 1024 * 1024);
            if (htp) {
                setenv("SD_HTP_COMPUTE", "1", 1);
                runner.set_htp_backend(htp);
                runner.set_htp_attention_enabled(sd_env_flag_enabled_default("SD_HTP_ATTENTION", true));
                runner.set_htp_packed_staging_loader(&loader);
                runner.set_max_graph_vram_bytes(512 * 1024 * 1024);
            }
            runner.set_flash_attention_enabled(true);
            GGML_ASSERT(runner.alloc_params_buffer());
            std::map<std::string, ggml_tensor*> params;
            runner.get_param_tensors(params, "model.diffusion_model");
            GGML_ASSERT(loader.load_tensor_subset(params, 4));
            sd::Tensor<float> x({8, 8, 16, 1}), text({2560, 15, 1}), time({1});
            for (int64_t i = 0; i < x.numel(); ++i) x[i] = std::sin(float(i)*.13f);
            for (int64_t i = 0; i < text.numel(); ++i) text[i] = std::cos(float(i)*.017f);
            time[0] = 500.f;
            auto start = std::chrono::steady_clock::now();
            auto out = argc == 5 ? runner.diagnostic(x, time, text) : runner.compute(4, x, time, text);
            GGML_ASSERT(!out.empty());
            for (int64_t i = 0; i < out.numel(); ++i) GGML_ASSERT(std::isfinite(out[i]));
            std::ofstream file(argv[3], std::ios::binary);
            GGML_ASSERT(file.write(reinterpret_cast<const char*>(out.data()), out.numel()*sizeof(float)));
            std::printf("mode=%s elements=%lld seconds=%.6f\n", argv[2], (long long)out.numel(),
                        std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count());
        }
    }
    ggml_backend_free(htp);
    ggml_backend_free(cpu);
}
