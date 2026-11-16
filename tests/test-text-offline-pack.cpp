#include "model.h"
#include "model_io/gguf_io.h"
#include "model_io/safetensors_io.h"
#include "stable-diffusion.h"
#include "htp_text_encoder.h"
#include <algorithm>
#include <cstring>
#include <chrono>
#include <cmath>
#include <filesystem>

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "CHECK failed line %d: %s\n", __LINE__, #x); return 1; } } while (0)
static TensorWriteInfo info(ggml_tensor* tensor) {
    TensorWriteInfo value;
    value.tensor = tensor;
    value.n_dims = ggml_n_dims(tensor);
    std::copy_n(tensor->ne, 4, value.ne);
    return value;
}
int main() {
    const auto root = std::filesystem::temp_directory_path() / ("sd-text-pack-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directory(root);
    auto* ctx = ggml_init({1024 * 1024, nullptr, false});
    CHECK(ctx);
    for (const std::string kind : {"clip_l", "clip_g", "t5xxl", "qwen3"}) {
        const auto prefix = sd_htp_text_prefix(kind);
        const auto local = kind == "qwen3" ? "model.layers.0.self_attn.q_proj.weight" : kind == "t5xxl" ? "encoder.block.0.layer.0.SelfAttention.q.weight" : "text_model.encoder.layers.0.mlp.fc1.weight";
        auto* linear = ggml_new_tensor_2d(ctx, GGML_TYPE_BF16, 32, 64);
        ggml_set_name(linear, local);
        for (int i = 0; i < 2048; ++i) static_cast<ggml_bf16_t*>(linear->data)[i] = ggml_fp32_to_bf16(std::sin(i * .01f));
        auto* table = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 32, 64);
        const auto table_local = kind == "qwen3" ? "model.embed_tokens.weight" : kind == "t5xxl" ? "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight" : "text_model.embeddings.token_embedding.weight";
        ggml_set_name(table, table_local);
        for (int i = 0; i < 2048; ++i) static_cast<ggml_fp16_t*>(table->data)[i] = ggml_fp32_to_fp16(i * .01f);
        const auto input = (root / (kind + ".safetensors")).string();
        const auto output = (root / (kind + ".gguf")).string();
        std::string error;
        CHECK(write_safetensors_file(input, {info(linear), info(table)}, &error));
        CHECK(convert_text_encoder_htp(input.c_str(), output.c_str(), kind.c_str()));
        std::vector<TensorStorage> tensors;
        GGUFHTPMetadata metadata;
        CHECK(read_gguf_file(output, tensors, &error, &metadata));
        CHECK(metadata.htp_packed_text && metadata.htp_text_encoder == kind && !metadata.htp_packed_dit);
        CHECK(metadata.htp_packed_tensors == std::vector<std::string>{prefix + local});
        ModelLoader loader;
        CHECK(loader.init_from_file_and_convert_name(output, prefix, kind == "qwen3" ? VERSION_FLUX2_KLEIN : VERSION_SD3));
        CHECK(loader.get_tensor_storage_map().at(prefix + local).layout == TensorDataLayout::HTP_PERMUTED_W16A32);
        CHECK(loader.get_tensor_storage_map().at(prefix + table_local).layout == TensorDataLayout::REFERENCE);
        auto* packed = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 32, 64);
        auto* unchanged = ggml_dup_tensor(ctx, table);
        CHECK(loader.load_tensor_subset({{prefix + local, packed}, {prefix + table_local, unchanged}}, 1));
        const auto* src = static_cast<ggml_bf16_t*>(linear->data);
        const auto* data = static_cast<ggml_fp16_t*>(packed->data);
        for (int row = 0; row < 64; ++row) for (int col = 0; col < 32; ++col) {
            const int index = (row / 32) * 1024 + (col & ~1) * 32 + (row % 32) * 2 + (col & 1);
            CHECK(data[index] == ggml_fp32_to_fp16(ggml_bf16_to_fp32(src[row * 32 + col])));
        }
        CHECK(std::memcmp(table->data, unchanged->data, ggml_nbytes(table)) == 0);
        CHECK(!convert_text_encoder_htp(output.c_str(), (root / "repacked.gguf").string().c_str(), kind.c_str()));
        CHECK(!convert_text_encoder_htp(input.c_str(), (root / "bad.safetensors").string().c_str(), kind.c_str()));
        CHECK(!convert_text_encoder_htp(input.c_str(), output.c_str(), "bad_kind"));
        ModelLoader wrong;
        CHECK(!wrong.init_from_file(output, "text_encoders.wrong.transformer."));
        // Embedding tables must never be declared packed, even if tile-aligned.
        ggml_set_name(packed, (prefix + table_local).c_str());
        GGUFWriteOptions options;
        options.htp = metadata;
        options.htp.htp_packed_tensors = {prefix + table_local};
        const auto bad = (root / "bad.gguf").string();
        CHECK(write_gguf_file(bad, {info(packed)}, &error, &options));
        CHECK(!read_gguf_file(bad, tensors, &error));
    }
    ggml_free(ctx);
    std::filesystem::remove_all(root);
    std::puts("Text offline packing/namespace/embedding preservation tests passed");
    return 0;
}
