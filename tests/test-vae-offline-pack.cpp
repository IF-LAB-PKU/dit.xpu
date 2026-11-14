#include "ggml_extend.hpp"
#include "model_io/gguf_io.h"
#include "model_io/safetensors_io.h"
#include "stable-diffusion.h"

#include <chrono>
#include <filesystem>

#define CHECK(condition) do { if (!(condition)) { \
    std::fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #condition); return 1; \
} } while (0)

static TensorWriteInfo info(ggml_tensor* tensor) {
    TensorWriteInfo result;
    result.tensor = tensor;
    result.n_dims = ggml_n_dims(tensor);
    std::copy_n(tensor->ne, 4, result.ne);
    return result;
}

int main() {
    auto* ctx = ggml_init({1024 * 1024, nullptr, false});
    CHECK(ctx);
    auto* source = ggml_new_tensor_4d(ctx, GGML_TYPE_BF16, 3, 3, 16, 5);
    ggml_set_name(source, "decoder.conv_in.weight");
    auto* data = static_cast<ggml_bf16_t*>(source->data);
    for (int64_t i = 0; i < ggml_nelements(source); ++i) data[i] = ggml_fp32_to_bf16(std::sin(float(i) * .071f));
    auto* norm = ggml_new_tensor_1d(ctx, GGML_TYPE_BF16, 5);
    ggml_set_name(norm, "decoder.conv_norm_out.weight");
    for (int i = 0; i < 5; ++i) static_cast<ggml_bf16_t*>(norm->data)[i] = ggml_fp32_to_bf16(float(i) + .5f);
    auto* probe = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 3);
    ggml_set_name(probe, "decoder.probe.weight");
    for (int i = 0; i < 12; ++i) static_cast<float*>(probe->data)[i] = 1e-9f * (i + 1);

    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto root = std::filesystem::temp_directory_path() / ("sd-vae-offline-" + std::to_string(nonce));
    std::filesystem::create_directory(root);
    const auto input = (root / "original.safetensors").string();
    const auto output = (root / "packed.gguf").string();
    const auto bad = (root / "bad.gguf").string();
    std::string error;
    CHECK(write_safetensors_file(input, {info(source), info(norm), info(probe)}, &error));
    CHECK(convert_vae_htp(input.c_str(), output.c_str()));
    CHECK(!convert_vae_htp(input.c_str(), (root / "packed.safetensors").string().c_str()));

    std::vector<TensorStorage> tensors;
    GGUFHTPMetadata metadata;
    CHECK(read_gguf_file(output, tensors, &error, &metadata));
    CHECK(metadata.htp_packed_vae && !metadata.htp_packed_dit);
    CHECK(metadata.htp_layout == "permuted_w16a32");
    CHECK(metadata.htp_vae_conv_tensors.size() == 1);
    CHECK(metadata.htp_vae_conv_shapes == std::vector<int64_t>({3, 3, 16, 5}));
    ModelLoader loader;
    // Canonical names must survive the same vae. prefix used by inference.
    CHECK(loader.init_from_file_and_convert_name(output, "vae.", VERSION_SD3));
    const auto& stored = loader.get_tensor_storage_map();
    const std::string name = "first_stage_model.decoder.conv_in.weight";
    const auto& packed = stored.at(name);
    CHECK(packed.layout == TensorDataLayout::HTP_CONV2D_W16A32);
    CHECK(packed.ne[0] == 160 && packed.ne[1] == 32 && packed.n_dims == 2);
    CHECK((packed.htp_conv_shape == std::array<int64_t, 4>{3, 3, 16, 5}));
    CHECK(stored.at("first_stage_model.decoder.norm_out.weight").type == GGML_TYPE_BF16);
    CHECK(stored.at("first_stage_model.decoder.probe.weight").type == GGML_TYPE_F32);
    Conv2d conv(16, 5, {3, 3});
    conv.init(ctx, stored, "first_stage_model.decoder.conv_in");
    auto* weight = conv.get_param("weight");
    CHECK(weight->ne[0] == 160 && weight->ne[1] == 32);
    CHECK((weight->flags & SD_GGML_TENSOR_FLAG_HTP_VAE_CONV_WEIGHT) != 0);
    ggml_set_name(weight, name.c_str());
    CHECK(loader.load_tensor_subset({{name, weight}}, 1));
    const auto* bytes = static_cast<const ggml_fp16_t*>(weight->data);
    for (int64_t row = 0; row < 32; ++row) {
        for (int64_t col = 0; col < 160; ++col) {
            const int64_t tile = (row / 32) * 5 + col / 32;
            const int64_t index = tile * 1024 + ((col % 32) & ~int64_t{1}) * 32 + (row % 32) * 2 + (col & 1);
            const auto expected = row < 5 && col < 144
                ? ggml_fp32_to_fp16(ggml_bf16_to_fp32(data[row * 144 + col])) : ggml_fp32_to_fp16(0.f);
            CHECK(bytes[index] == expected);
        }
    }
    CHECK(!convert_vae_htp(output.c_str(), bad.c_str())); // Never double-pack tiled input.
    CHECK(!std::filesystem::exists(bad));

    GGUFWriteOptions options;
    options.htp = metadata;
    options.htp.htp_vae_conv_shapes[2] = 12; // Physical K no longer matches the logical shape.
    CHECK(write_gguf_file(bad, {info(weight)}, &error, &options));
    CHECK(!read_gguf_file(bad, tensors, &error));
    options.htp = metadata;
    options.htp.htp_vae_conv_shapes.pop_back();
    CHECK(write_gguf_file(bad, {info(weight)}, &error, &options));
    CHECK(!read_gguf_file(bad, tensors, &error));
    options.htp = metadata;
    options.htp.htp_layout = "unknown_layout";
    CHECK(write_gguf_file(bad, {info(weight)}, &error, &options));
    CHECK(!read_gguf_file(bad, tensors, &error));

    // The existing DIT export/read path shares the GGUF writer and must retain its layout.
    auto* dit = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 64, 32);
    ggml_set_name(dit, "model.diffusion_model.proj.weight");
    for (int i = 0; i < 64 * 32; ++i) static_cast<ggml_fp16_t*>(dit->data)[i] = ggml_fp32_to_fp16(i * .01f);
    const auto dit_in = (root / "dit.safetensors").string();
    const auto dit_out = (root / "dit.gguf").string();
    CHECK(write_safetensors_file(dit_in, {info(dit)}, &error));
    CHECK(convert(dit_in.c_str(), "", dit_out.c_str(), SD_TYPE_F16, "", false, true, false));
    CHECK(read_gguf_file(dit_out, tensors, &error, &metadata));
    CHECK(metadata.htp_packed_dit && !metadata.htp_packed_vae);
    CHECK(tensors.size() == 1 && tensors[0].layout == TensorDataLayout::HTP_PERMUTED_W16A32);
    std::filesystem::remove_all(root);
    ggml_free(ctx);
    std::puts("VAE offline packing/metadata/namespace and DIT compatibility tests passed");
    return 0;
}
