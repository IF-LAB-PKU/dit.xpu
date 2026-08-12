#include "model.h"
#include "model_io/safetensors_io.h"
#include "stable-diffusion.h"
#include "ggml-htp.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

#define CHECK(condition)                                                                 \
    do {                                                                                 \
        if (!(condition)) {                                                              \
            fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #condition); \
            std::abort();                                                                \
        }                                                                                \
    } while (0)

#if !defined(GGML_USE_HTP)
extern "C" ggml_backend_buffer_type_t ggml_backend_htp_qkv_buffer_type(void) {
    return nullptr;
}

extern "C" bool ggml_backend_htp_opencl_release_qkv_generation(ggml_backend_t,
                                                                uint64_t,
                                                                bool,
                                                                bool* released,
                                                                char*,
                                                                size_t) {
    if (released != nullptr) {
        *released = true;
    }
    return true;
}
#endif

static TensorWriteInfo tensor_write_info(ggml_tensor* tensor) {
    TensorWriteInfo info;
    info.n_dims = ggml_n_dims(tensor);
    info.tensor = tensor;
    for (int i = 0; i < info.n_dims; ++i) {
        info.ne[i] = tensor->ne[i];
    }
    return info;
}

static void assert_values(const ggml_tensor* tensor, const std::vector<float>& expected) {
    CHECK(static_cast<size_t>(ggml_nelements(tensor)) == expected.size());
    const float* values = static_cast<const float*>(tensor->data);
    for (size_t i = 0; i < expected.size(); ++i) {
        CHECK(values[i] == expected[i]);
    }
}

int main() {
    sd_ctx_params_t context_params;
    sd_ctx_params_init(&context_params);
    CHECK(!context_params.staged_params_loading);

    ggml_init_params params = {
        1024 * 1024,
        nullptr,
        false,
    };
    ggml_context* ctx = ggml_init(params);
    CHECK(ctx != nullptr);

    ggml_tensor* encoder_source   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    ggml_tensor* diffusion_source = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    ggml_set_name(encoder_source, "encoder.weight");
    ggml_set_name(diffusion_source, "diffusion.weight");

    const std::vector<float> encoder_values   = {1.f, 2.f, 3.f, 4.f};
    const std::vector<float> diffusion_values = {5.f, 6.f, 7.f, 8.f};
    memcpy(encoder_source->data, encoder_values.data(), ggml_nbytes(encoder_source));
    memcpy(diffusion_source->data, diffusion_values.data(), ggml_nbytes(diffusion_source));

    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    const std::filesystem::path model_path =
        std::filesystem::temp_directory_path() /
        ("sd-model-loader-subset-" + std::to_string(nonce) + ".safetensors");

    std::string error;
    CHECK(write_safetensors_file(model_path.string(),
                                 {tensor_write_info(encoder_source), tensor_write_info(diffusion_source)},
                                 &error));

    ggml_tensor* encoder_destination   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    ggml_tensor* diffusion_destination = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    ggml_tensor* missing_destination   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    ggml_tensor* optional_clip_destination = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    const std::vector<float> untouched(4, -1.f);
    memcpy(encoder_destination->data, untouched.data(), ggml_nbytes(encoder_destination));
    memcpy(diffusion_destination->data, untouched.data(), ggml_nbytes(diffusion_destination));
    memcpy(optional_clip_destination->data, untouched.data(), ggml_nbytes(optional_clip_destination));

    ModelLoader loader;
    CHECK(loader.init_from_file(model_path.string()));

    std::map<std::string, ggml_tensor*> encoder_only = {
        {"encoder.weight", encoder_destination},
    };
    CHECK(loader.load_tensor_subset(encoder_only, 1));
    assert_values(encoder_destination, encoder_values);
    assert_values(diffusion_destination, untouched);

    std::map<std::string, ggml_tensor*> diffusion_only = {
        {"diffusion.weight", diffusion_destination},
    };
    CHECK(loader.load_tensor_subset(diffusion_only, 1));
    assert_values(diffusion_destination, diffusion_values);

    std::map<std::string, ggml_tensor*> missing = {
        {"missing.weight", missing_destination},
    };
    CHECK(!loader.load_tensor_subset(missing, 1));

    std::map<std::string, ggml_tensor*> optional_clip_layer = {
        {"cond_stage_model.transformer.text_model.encoder.layers.23.self_attn.q_proj.weight",
         optional_clip_destination},
    };
    CHECK(loader.load_tensor_subset(optional_clip_layer, 1));
    assert_values(optional_clip_destination, untouched);

    std::remove(model_path.string().c_str());
    ggml_free(ctx);
    return 0;
}
