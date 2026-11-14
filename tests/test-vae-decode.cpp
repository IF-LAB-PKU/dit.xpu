// Device correctness checks and isolated SD3 VAE decoder timing.
#include "auto_encoder_kl.hpp"
#include <chrono>

static void log_line(sd_log_level_t, const char* text, void*) { std::fputs(text, stderr); }
static double seconds() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct ConvRunner : GGMLRunner {
    ggml_tensor* weight;
    int stride, pad, dilation;
    ConvRunner(ggml_backend_t cpu, ggml_backend_t htp, int ic, int oc, int kernel,
               int stride, int pad, int dilation)
        : GGMLRunner(cpu, cpu), stride(stride), pad(pad), dilation(dilation) {
        weight = ggml_new_tensor_4d(params_ctx, GGML_TYPE_F16, kernel, kernel, ic, oc);
        ggml_set_name(weight, "conv_test.weight");
        GGML_ASSERT(alloc_params_buffer());
        std::vector<ggml_fp16_t> w(ggml_nelements(weight));
        for (size_t i = 0; i < w.size(); ++i) w[i] = ggml_fp32_to_fp16(std::sin(float(i) * .17f) * .04f);
        ggml_backend_tensor_set(weight, w.data(), 0, ggml_nbytes(weight));
        if (htp) { set_htp_backend(htp); set_htp_vae_conv_enabled(true); }
    }
    std::string get_desc() override { return "vae-conv-test"; }
    sd::Tensor<float> run(const sd::Tensor<float>& input, bool htp) {
        auto build = [&]() {
            auto ctx = get_context();
            auto* x = make_input(input);
            auto* packed = ctx.htp_conv_weight ? ctx.htp_conv_weight(weight) : nullptr;
            auto* out = packed
                ? ggml_htp_vae_conv_2d(compute_ctx, x, weight, packed,
                                        stride, stride, pad, pad, dilation, dilation, 16 * 1024)
                : ggml_ext_conv_2d(compute_ctx, x, weight, nullptr,
                                    stride, stride, pad, pad, dilation, dilation);
            auto* graph = ggml_new_graph(compute_ctx);
            ggml_build_forward_expand(graph, out);
            if (htp) GGML_ASSERT(!collect_planned_htp_target_ops(graph).empty());
            return graph;
        };
        auto output = compute<float>(build, 4, true);
        GGML_ASSERT(output.has_value() && !output->empty());
        return std::move(*output);
    }
};

static void self_test(ggml_backend_t cpu, ggml_backend_t htp) {
    // Unequal spatial dimensions and batch=2 expose axis/order mistakes;
    // non-multiple channels exercise K/N padding (including RGB output).
    const int cases[][8] = {
        {11, 9, 3, 5, 3, 1, 1, 1}, {9, 7, 16, 32, 1, 1, 0, 1},
        {13, 11, 32, 3, 3, 2, 1, 1}, {11, 9, 32, 64, 3, 1, 2, 2},
    };
    for (const auto& c : cases) {
        sd::Tensor<float> input({c[0], c[1], c[2], 2});
        for (int64_t i = 0; i < input.numel(); ++i) input[i] = std::sin(float(i) * .071f);
        ConvRunner reference(cpu, nullptr, c[2], c[3], c[4], c[5], c[6], c[7]);
        auto expected = reference.run(input, false);
        ConvRunner accelerated(cpu, htp, c[2], c[3], c[4], c[5], c[6], c[7]);
        for (int repeat = 0; repeat < 2; ++repeat) {
            auto actual = accelerated.run(input, true);
            GGML_ASSERT(actual.shape() == expected.shape());
            double max_error = 0, squared = 0;
            for (int64_t i = 0; i < actual.numel(); ++i) {
                GGML_ASSERT(std::isfinite(actual[i]));
                const double error = actual[i] - expected[i];
                max_error = std::max(max_error, std::abs(error));
                squared += error * error;
            }
            std::printf("conv %dx%d ic=%d oc=%d kernel=%d stride=%d dilation=%d repeat=%d max_abs=%.6g rmse=%.6g\n",
                        c[0], c[1], c[2], c[3], c[4], c[5], c[7], repeat,
                        max_error, std::sqrt(squared / actual.numel()));
            GGML_ASSERT(max_error < .03);
            GGML_ASSERT(accelerated.get_htp_vae_packed_count() == 1);
        }
        accelerated.free_params_buffer();
        GGML_ASSERT(accelerated.get_htp_vae_packed_count() == 0);
    }
    std::puts("VAE convolution HTP correctness PASSED");
}

struct RawVae : AutoEncoderKL {
    using AutoEncoderKL::AutoEncoderKL;
    void raw_output() { scale_input = false; } // Preserve errors before display clipping.
};

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "Usage: %s --self-test | <vae.safetensors> <cpu|htp> <image-size> <repeats> <output.f32> [latent.f32] [threads]\n", argv[0]);
        return 2;
    }
    sd_set_log_callback(log_line, nullptr);
    const bool test = std::string(argv[1]) == "--self-test";
    if (!test && (argc < 6 || (std::string(argv[2]) != "cpu" && std::string(argv[2]) != "htp"))) return 2;
    const bool use_htp = test || std::string(argv[2]) == "htp";
    auto* cpu = ggml_backend_init_by_name("CPU", nullptr);
    GGML_ASSERT(cpu);
    auto* htp = use_htp ? ggml_backend_init_by_name("HTP", nullptr) : nullptr;
    if (use_htp) {
#ifdef GGML_USE_HTP
        GGML_ASSERT(htp && ggml_backend_htp_runtime_available());
#else
        GGML_ABORT("Build with GGML_HTP=ON to test HTP");
#endif
        setenv("SD_HTP_COMPUTE", "1", 1);
    }
    if (test) {
        self_test(cpu, htp);
    } else {
        const int size = std::atoi(argv[3]), repeats = std::atoi(argv[4]);
        const int threads = argc > 7 ? std::atoi(argv[7]) : 4;
        GGML_ASSERT(size > 0 && size <= 2048 && size % 8 == 0 && repeats > 0 && threads > 0);
        const double load_start = seconds();
        ModelLoader loader;
        GGML_ASSERT(loader.init_from_file_and_convert_name(argv[1], "vae.", VERSION_SD3));
        RawVae vae(cpu, cpu, loader.get_tensor_storage_map(), "first_stage_model", true, VERSION_SD3);
        if (htp) { vae.set_htp_backend(htp); vae.set_htp_vae_conv_enabled(true); }
        GGML_ASSERT(vae.alloc_params_buffer());
        std::map<std::string, ggml_tensor*> tensors;
        vae.get_param_tensors(tensors, "first_stage_model");
        GGML_ASSERT(loader.load_tensor_subset(tensors, threads));
        vae.raw_output();
        if (htp) { vae.set_htp_backend(htp); vae.set_htp_vae_conv_enabled(true); }
        std::printf("load_seconds=%.6f mode=%s size=%d threads=%d\n", seconds() - load_start, argv[2], size, threads);
        sd::Tensor<float> latent({size / 8, size / 8, 16, 1});
        if (argc > 6 && std::string(argv[6]) != "-") {
            std::ifstream file(argv[6], std::ios::binary);
            GGML_ASSERT(file.read(reinterpret_cast<char*>(latent.data()), latent.numel() * sizeof(float)));
        } else {
            for (int64_t i = 0; i < latent.numel(); ++i) latent[i] = std::sin(float(i) * .017f) * .5f;
        }
        sd_tiling_params_t tiling{};
        for (int repeat = 0; repeat < repeats; ++repeat) {
            const double begin = seconds();
            auto output = vae.decode(threads, latent, tiling);
            GGML_ASSERT(!output.empty());
            for (int64_t i = 0; i < output.numel(); ++i) GGML_ASSERT(std::isfinite(output[i]));
            std::printf("decode repeat=%d seconds=%.6f packed_count=%zu packed_bytes=%zu offline_count=%zu params_bytes=%zu elements=%lld\n",
                        repeat, seconds() - begin, vae.get_htp_vae_packed_count(),
                        vae.get_htp_vae_packed_bytes(), vae.get_htp_vae_offline_count(), vae.get_params_buffer_size(), (long long) output.numel());
            if (vae.get_htp_vae_offline_count() != 0) {
                GGML_ASSERT(vae.get_htp_vae_packed_count() == 0 && vae.get_htp_vae_packed_bytes() == 0);
            }
            std::fflush(stdout);
            if (repeat == repeats - 1) {
                std::ofstream file(argv[5], std::ios::binary);
                GGML_ASSERT(file.write(reinterpret_cast<const char*>(output.data()), output.numel() * sizeof(float)));
            }
        }
    }
    ggml_backend_free(htp);
    ggml_backend_free(cpu);
    return 0;
}
