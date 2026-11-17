// Explicit device benchmark: actual SD3 tokenization and conditioning path.
#include "conditioner.hpp"
#include <chrono>

static double seconds() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
static void log_line(sd_log_level_t, const char* text, void*) { std::fputs(text, stderr); }
static void save(const std::string& path, const sd::Tensor<float>& tensor) {
    GGML_ASSERT(!tensor.empty());
    for (int64_t i = 0; i < tensor.numel(); ++i) GGML_ASSERT(std::isfinite(tensor[i]));
    std::ofstream file(path, std::ios::binary);
    GGML_ASSERT(file.write(reinterpret_cast<const char*>(tensor.data()), tensor.numel() * sizeof(float)));
}
static int benchmark_qwen3(int argc, char** argv, ggml_backend_t cpu, ggml_backend_t htp) {
    const auto version = std::string(argv[1]) == "qwen3-z-image" ? VERSION_Z_IMAGE : VERSION_FLUX2_KLEIN;
    ModelLoader loader;
    GGML_ASSERT(loader.init_from_file_and_convert_name(argv[2], "text_encoders.llm.", version));
    LLMEmbedder encoder(cpu, cpu, loader.get_tensor_storage_map(), version);
    if (htp) GGML_ASSERT(encoder.configure_htp_text_encoder(htp, &loader));
    encoder.alloc_params_buffer();
    std::map<std::string, ggml_tensor*> params;
    encoder.get_param_tensors(params);
    GGML_ASSERT(loader.load_tensor_subset(params, 8));
    std::vector<float> reference;
    for (int r = 0; r < std::atoi(argv[4]); ++r) {
        const bool prefetch_ab = htp && std::getenv("SD_HTP_PREFETCH_AB");
        if (prefetch_ab) setenv("SD_HTP_PACKED_PREFETCH", r % 4 == 1 || r % 4 == 2 ? "1" : "0", 1);
        ConditionerParams p;
        p.text = argc > 6 ? argv[6] : "a cat sitting on a wooden table";
        const double start = seconds();
        auto condition = encoder.get_learned_condition(8, p);
        const double elapsed = seconds() - start;
        const auto& out = condition.c_crossattn;
        save(std::string(argv[5]) + "." + std::to_string(r) + ".cross.f32", out);
        if (prefetch_ab) {
            if (!r) reference.assign(out.data(), out.data() + out.numel());
            else GGML_ASSERT(reference.size() == size_t(out.numel()) &&
                             std::memcmp(reference.data(), out.data(), reference.size() * sizeof(float)) == 0);
        }
        std::printf("encoder=%s mode=%s repeat=%d prefetch=%d seconds=%.6f cross_elements=%lld\n",
                    argv[1], argv[3], r, sd_env_flag_enabled("SD_HTP_PACKED_PREFETCH"), elapsed, (long long)out.numel());
    }
    return 0;
}
int main(int argc, char** argv) {
    if (argc < 6) {
        std::fprintf(stderr, "usage: %s clip_l|clip_g|t5xxl|qwen3|qwen3-z-image MODEL cpu|htp REPEATS OUTPUT_PREFIX [PROMPT]\n", argv[0]);
        return 2;
    }
    const std::string kind = argv[1];
    if (kind != "clip_l" && kind != "clip_g" && kind != "t5xxl" && kind != "qwen3" && kind != "qwen3-z-image") return 2;
    const bool use_htp = std::string(argv[3]) == "htp";
    if (!use_htp && std::string(argv[3]) != "cpu") return 2;
    const int repeats = std::atoi(argv[4]);
    GGML_ASSERT(repeats > 0);
    sd_set_log_callback(log_line, nullptr);
    auto* cpu = ggml_backend_init_by_name("CPU", nullptr);
    auto* htp = use_htp ? ggml_backend_init_by_name("HTP", nullptr) : nullptr;
    GGML_ASSERT(cpu && (!use_htp || htp));
    if (use_htp) setenv("SD_HTP_COMPUTE", "1", 1);
    if (kind == "qwen3" || kind == "qwen3-z-image") {
        const int result = benchmark_qwen3(argc, argv, cpu, htp);
        ggml_backend_free(htp);
        ggml_backend_free(cpu);
        return result;
    }
    {
        const double t0 = seconds();
        ModelLoader loader;
        GGML_ASSERT(loader.init_from_file_and_convert_name(argv[2], "text_encoders." + kind + ".transformer.", VERSION_SD3));
        SD3CLIPEmbedder encoder(cpu, cpu, loader.get_tensor_storage_map());
        if (htp) GGML_ASSERT(encoder.configure_htp_text_encoder(htp, &loader));
        encoder.alloc_params_buffer();
        std::map<std::string, ggml_tensor*> params;
        encoder.get_param_tensors(params);
        GGML_ASSERT(loader.load_tensor_subset(params, 4));
        size_t packed = 0, deferred = 0;
        for (const auto& item : params) {
            if (item.second->flags & SD_GGML_TENSOR_FLAG_HTP_PACKED_W16A32) ++packed;
            if (item.second->flags & SD_GGML_TENSOR_FLAG_HTP_DEFERRED_PACKED_PARAM) ++deferred;
        }
        std::printf("encoder=%s mode=%s load_seconds=%.6f packed=%zu deferred=%zu params_bytes=%zu\n",
                    kind.c_str(), argv[3], seconds() - t0, packed, deferred, encoder.get_params_buffer_size());
        std::vector<float> shared_reference_cross, shared_reference_vector;
        for (int r = 0; r < repeats; ++r) {
            const bool prefetch_ab = encoder.t5 && std::getenv("SD_HTP_PREFETCH_AB");
            if (prefetch_ab) setenv("SD_HTP_PACKED_PREFETCH", r % 4 == 1 || r % 4 == 2 ? "1" : "0", 1);
            if (std::getenv("SD_HTP_FUSION_AB")) setenv("SD_HTP_TEXT_FUSION",r%2 ? "1" : "0",1);
            ConditionerParams p;
            p.text = argc > 6 ? argv[6] : "a cat sitting on a wooden table";
            const double start = seconds();
            auto tokens = encoder.tokenize(p.text, 77, 77, true);
            const auto index = kind == "clip_l" ? 0 : kind == "clip_g" ? 1 : 2;
            auto ids = sd::Tensor<int32_t>::from_vector(tokens[index].first);
            SDCondition condition;
            if (encoder.t5) {
                condition.c_crossattn = encoder.t5->compute(4, ids, {});
                condition.c_vector = sd::Tensor<float>::zeros({1});
            } else {
                auto model = encoder.clip_l ? encoder.clip_l : encoder.clip_g;
                const auto eos = std::find(tokens[index].first.begin(), tokens[index].first.end(), 49407);
                const auto eos_index = std::min<size_t>(eos - tokens[index].first.begin(), tokens[index].first.size() - 1);
                int clip_skip = sd_env_int_value("SD_TEST_CLIP_SKIP", 2);
                const bool skip_cycle = std::getenv("SD_TEST_CLIP_SKIP_CYCLE") != nullptr;
                if (skip_cycle) {
                    const int skips[] = {2, 1, model->model.n_layer};
                    clip_skip = skips[(r / 2) % 3];
                }
                const bool shared = skip_cycle ? r % 2 != 0 : std::getenv("SD_CLIP_SHARED_PASS_AB")
                    ? (r % 4 == 1 || r % 4 == 2)
                    : sd_env_flag_enabled_default("SD_CLIP_SHARED_PASS", true);
                if (shared) {
                    auto both = model->compute_hidden_and_pooled(4, ids, eos_index, clip_skip);
                    condition.c_crossattn = std::move(both.first);
                    condition.c_vector = std::move(both.second);
                } else {
                    condition.c_crossattn = model->compute(4, ids, 0, nullptr, eos_index, false, clip_skip);
                    condition.c_vector = model->compute(4, ids, 0, nullptr, eos_index, true, clip_skip);
                }
                std::printf("shared_pass=%d clip_skip=%d ", shared ? 1 : 0, clip_skip);
            }
            const double elapsed = seconds() - start;
            if ((prefetch_ab || (std::getenv("SD_CLIP_SHARED_PASS_AB") && !encoder.t5)) &&
                !std::getenv("SD_HTP_FUSION_AB")) {
                if (r == 0 || (std::getenv("SD_TEST_CLIP_SKIP_CYCLE") && r % 2 == 0)) {
                    shared_reference_cross.assign(condition.c_crossattn.data(),
                        condition.c_crossattn.data() + condition.c_crossattn.numel());
                    shared_reference_vector.assign(condition.c_vector.data(),
                        condition.c_vector.data() + condition.c_vector.numel());
                } else {
                    GGML_ASSERT(size_t(condition.c_crossattn.numel()) == shared_reference_cross.size());
                    GGML_ASSERT(size_t(condition.c_vector.numel()) == shared_reference_vector.size());
                    GGML_ASSERT(std::memcmp(condition.c_crossattn.data(), shared_reference_cross.data(),
                        shared_reference_cross.size() * sizeof(float)) == 0);
                    GGML_ASSERT(std::memcmp(condition.c_vector.data(), shared_reference_vector.data(),
                        shared_reference_vector.size() * sizeof(float)) == 0);
                    std::printf("output_check=exact ");
                }
            }
            const std::string prefix = std::string(argv[5]) + "." + std::to_string(r);
            save(prefix + ".cross.f32", condition.c_crossattn);
            save(prefix + ".vector.f32", condition.c_vector);
            std::printf("fusion=%d ",sd_env_flag_enabled("SD_HTP_TEXT_FUSION") ? 1 : 0);
            if (prefetch_ab) std::printf("prefetch=%d ", sd_env_flag_enabled("SD_HTP_PACKED_PREFETCH") ? 1 : 0);
            std::printf("encoder=%s mode=%s repeat=%d seconds=%.6f params_and_staging_bytes=%zu cross_elements=%lld vector_elements=%lld\n",
                        kind.c_str(), argv[3], r, elapsed, encoder.get_params_buffer_size(),
                        (long long)condition.c_crossattn.numel(), (long long)condition.c_vector.numel());
            std::fflush(stdout);
        }
        encoder.free_params_buffer();
    }
    if (htp) ggml_backend_free(htp);
    ggml_backend_free(cpu);
    return 0;
}
