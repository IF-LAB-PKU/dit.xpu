#include "model.h"
#include "model_io/safetensors_io.h"
#include "stable-diffusion.h"

#include <chrono>
#include <cstdio>
#include <filesystem>

#define CHECK(condition) \
    do { \
        if (!(condition)) { \
            std::fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #condition); \
            return 1; \
        } \
    } while (0)

static TensorWriteInfo tensor_write_info(ggml_tensor* tensor) {
    TensorWriteInfo info;
    info.tensor = tensor;
    info.n_dims = ggml_n_dims(tensor);
    for (int i = 0; i < info.n_dims; ++i) {
        info.ne[i] = tensor->ne[i];
    }
    return info;
}

int main() {
    ggml_context* ctx = ggml_init({1024 * 1024, nullptr, false});
    CHECK(ctx != nullptr);
    ggml_tensor* source = ggml_new_tensor_2d(ctx, GGML_TYPE_BF16, 4, 2);
    CHECK(source != nullptr);
    ggml_set_name(source, "model.diffusion_model.x_embedder.weight");
    for (int i = 0; i < ggml_nelements(source); ++i) {
        static_cast<ggml_bf16_t*>(source->data)[i] = ggml_fp32_to_bf16(static_cast<float>(i));
    }

    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto input = std::filesystem::temp_directory_path() / ("sd-force-f16-input-" + std::to_string(nonce) + ".safetensors");
    const auto output = std::filesystem::temp_directory_path() / ("sd-force-f16-output-" + std::to_string(nonce) + ".safetensors");
    std::string error;
    CHECK(write_safetensors_file(input.string(), {tensor_write_info(source)}, &error));
    CHECK(convert(input.string().c_str(), "", output.string().c_str(), SD_TYPE_F16, "", false, false, true));

    ModelLoader loader;
    CHECK(loader.init_from_file(output.string()));
    const auto& tensors = loader.get_tensor_storage_map();
    auto it = tensors.find("model.diffusion_model.x_embedder.weight");
    CHECK(it != tensors.end());
    CHECK(it->second.type == GGML_TYPE_F16);

    std::filesystem::remove(input);
    std::filesystem::remove(output);
    ggml_free(ctx);
    return 0;
}
