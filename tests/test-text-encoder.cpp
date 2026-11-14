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
int main(int argc, char** argv) {
    if (argc < 6) {
        std::fprintf(stderr, "usage: %s clip_l|clip_g|t5xxl MODEL cpu|htp REPEATS OUTPUT_PREFIX [PROMPT]\n", argv[0]);
        return 2;
    }
    const std::string kind = argv[1];
    if (kind != "clip_l" && kind != "clip_g" && kind != "t5xxl") return 2;
    const bool use_htp = std::string(argv[3]) == "htp";
    if (!use_htp && std::string(argv[3]) != "cpu") return 2;
    const int repeats = std::atoi(argv[4]);
    GGML_ASSERT(repeats > 0);
    sd_set_log_callback(log_line, nullptr);
    auto* cpu = ggml_backend_init_by_name("CPU", nullptr);
    auto* htp = use_htp ? ggml_backend_init_by_name("HTP", nullptr) : nullptr;
    GGML_ASSERT(cpu && (!use_htp || htp));
    if (use_htp) setenv("SD_HTP_COMPUTE", "1", 1);
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
        for (int r = 0; r < repeats; ++r) {
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
                condition.c_crossattn = model->compute(4, ids, 0, nullptr, eos_index, false, 2);
                condition.c_vector = model->compute(4, ids, 0, nullptr, eos_index, true, 2);
            }
            const double elapsed = seconds() - start;
            const std::string prefix = std::string(argv[5]) + "." + std::to_string(r);
            save(prefix + ".cross.f32", condition.c_crossattn);
            save(prefix + ".vector.f32", condition.c_vector);
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
