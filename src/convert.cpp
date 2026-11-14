#include <algorithm>
#include <cstring>
#include <mutex>
#include <regex>
#include <vector>

#include "model.h"
#include "model_io/gguf_io.h"
#include "model_io/safetensors_io.h"
#include "util.h"

#include "ggml-cpu.h"
#include "htp_text_encoder.h"

static constexpr const char* SD_GGUF_HTP_LAYOUT_W16A32 = "permuted_w16a32";

static ggml_type get_export_tensor_type(ModelLoader& model_loader,
                                        const TensorStorage& tensor_storage,
                                        ggml_type type,
                                        const TensorTypeRules& tensor_type_rules,
                                        bool force_f16) {
    const std::string& name = tensor_storage.name;
    ggml_type tensor_type   = tensor_storage.type;
    ggml_type dst_type      = type;

    for (const auto& tensor_type_rule : tensor_type_rules) {
        std::regex pattern(tensor_type_rule.first);
        if (std::regex_search(name, pattern)) {
            dst_type = tensor_type_rule.second;
            break;
        }
    }

    if (force_f16 && tensor_storage.type == GGML_TYPE_BF16) {
        tensor_type = GGML_TYPE_F16;
    } else if (model_loader.tensor_should_be_converted(tensor_storage, dst_type)) {
        tensor_type = dst_type;
    }

    return tensor_type;
}

static bool load_tensors_for_export(ModelLoader& model_loader,
                                    ggml_context* ggml_ctx,
                                    ggml_type type,
                                    const TensorTypeRules& tensor_type_rules,
                                    bool force_f16,
                                    std::vector<TensorWriteInfo>& tensors,
                                    bool htp_pack_vae = false, const std::string& text_kind = "") {
    std::mutex tensor_mutex;
    auto on_new_tensor_cb = [&](const TensorStorage& tensor_storage, ggml_tensor** dst_tensor) -> bool {
        const std::string& name = tensor_storage.name;
        ggml_type tensor_type   = (htp_pack_vae || !text_kind.empty()) ? tensor_storage.type
            : get_export_tensor_type(model_loader, tensor_storage, type, tensor_type_rules, force_f16);
        if (htp_pack_vae && starts_with(name, "first_stage_model.") &&
            ends_with(name, ".weight") && tensor_storage.n_dims == 4) {
            tensor_type = GGML_TYPE_F16;
        }
        if (!text_kind.empty() && sd_htp_text_linear_name(name, text_kind)) {
            tensor_type = GGML_TYPE_F16;
        }
        if (tensor_storage.layout != TensorDataLayout::REFERENCE) {
            LOG_ERROR("re-exporting packed input is unsupported; use the original model: %s", name.c_str());
            return false;
        }

        std::lock_guard<std::mutex> lock(tensor_mutex);
        ggml_tensor* tensor = ggml_new_tensor(ggml_ctx, tensor_type, tensor_storage.n_dims, tensor_storage.ne);
        if (tensor == nullptr) {
            LOG_ERROR("ggml_new_tensor failed");
            return false;
        }
        ggml_set_name(tensor, name.c_str());

        if (!tensor->data) {
            GGML_ASSERT(ggml_nelements(tensor) == 0);
            // Avoid crashing writers by setting a dummy pointer for zero-sized tensors.
            LOG_DEBUG("setting dummy pointer for zero-sized tensor %s", name.c_str());
            tensor->data = ggml_get_mem_buffer(ggml_ctx);
        }

        TensorWriteInfo write_info;
        write_info.tensor = tensor;
        write_info.n_dims = tensor_storage.n_dims;
        for (int i = 0; i < tensor_storage.n_dims; ++i) {
            write_info.ne[i] = tensor_storage.ne[i];
        }

        *dst_tensor = tensor;
        tensors.push_back(std::move(write_info));

        return true;
    };

    bool success = model_loader.load_tensors(on_new_tensor_cb);
    LOG_INFO("load tensors done");
    return success;
}

static bool is_htp_packable_dit_weight(const ggml_tensor* tensor) {
    if (tensor == nullptr) {
        return false;
    }

    const std::string name = tensor->name;
    // These MMDiT edge projections are constructed with force_f32=true at runtime.
    // Keeping them row-major lets CPU fallback load them as F32; packing them would make fallback consume W16A32 bytes.
    const bool force_f32_runtime_weight = ends_with(name, "context_embedder.weight") ||
                                          contains(name, "t_embedder.mlp.") ||
                                          contains(name, "y_embedder.mlp.") ||
                                          ends_with(name, "final_layer.linear.weight");
    if (force_f32_runtime_weight) {
        return false;
    }

    return starts_with(name, "model.diffusion_model.") &&
           ends_with(name, ".weight") &&
           tensor->type == GGML_TYPE_F16 &&
           ggml_n_dims(tensor) == 2 &&
           tensor->ne[0] % 32 == 0 &&
           tensor->ne[1] % 32 == 0 &&
           tensor->data != nullptr;
}

static void pack_htp_f16_weight_w16a32(ggml_fp16_t* dst, const ggml_fp16_t* src, int64_t k, int64_t n) {
    GGML_ASSERT(dst != nullptr);
    GGML_ASSERT(src != nullptr);
    GGML_ASSERT(k % 32 == 0);
    GGML_ASSERT(n % 32 == 0);

    // Tile-sized source working set also keeps large T5 exports cache-friendly.
    for (int64_t j0 = 0; j0 < n; j0 += 32) {
        for (int64_t i0 = 0; i0 < k; i0 += 32) {
            auto* tile = dst + ((j0 / 32) * (k / 32) + i0 / 32) * 1024;
            for (int64_t i1 = 0; i1 < 32; i1 += 2) {
                for (int64_t j1 = 0; j1 < 32; ++j1) {
                    tile[i1 * 32 + j1 * 2] = src[(j0 + j1) * k + i0 + i1];
                    tile[i1 * 32 + j1 * 2 + 1] = src[(j0 + j1) * k + i0 + i1 + 1];
                }
            }
        }
    }
}

static bool pack_htp_dit_weights_for_export(std::vector<TensorWriteInfo>& tensors,
                                            std::vector<std::string>& packed_tensor_names) {
    packed_tensor_names.clear();

    // 离线转换阶段直接原地改写 eligible DIT F16 weight 的数据布局。
    // 运行时不会保留 row-major 副本，因此这里必须同步记录 tensor 名单到 GGUF 元数据。
    size_t packed_bytes = 0;
    for (TensorWriteInfo& write_info : tensors) {
        ggml_tensor* tensor = write_info.tensor;
        if (!is_htp_packable_dit_weight(tensor)) {
            continue;
        }

        const size_t nbytes = ggml_nbytes(tensor);
        std::vector<ggml_fp16_t> packed(nbytes / sizeof(ggml_fp16_t));
        pack_htp_f16_weight_w16a32(packed.data(),
                                   static_cast<const ggml_fp16_t*>(tensor->data),
                                   tensor->ne[0],
                                   tensor->ne[1]);
        std::memcpy(tensor->data, packed.data(), nbytes);
        packed_tensor_names.push_back(tensor->name);
        packed_bytes += nbytes;
    }

    LOG_INFO("HTP packed DIT GGUF conversion selected %zu tensors (%.2f MB)",
             packed_tensor_names.size(),
             packed_bytes / 1024.f / 1024.f);
    if (packed_tensor_names.empty()) {
        LOG_ERROR("--htp-pack-dit did not find any eligible model.diffusion_model.*.weight F16 2D tensors; check the model and --convert-name");
        return false;
    }
    return true;
}

static bool pack_htp_vae_weights_for_export(ggml_context* packed_ctx,
                                             std::vector<TensorWriteInfo>& tensors,
                                             std::vector<std::vector<ggml_fp16_t>>& storage,
                                             GGUFHTPMetadata& metadata) {
    metadata.htp_packed_vae = true;
    metadata.htp_layout = SD_GGUF_HTP_LAYOUT_W16A32;
    size_t bytes = 0;
    for (auto& info : tensors) {
        auto* source = info.tensor;
        const std::string name = source->name;
        if (!starts_with(name, "first_stage_model.") || !ends_with(name, ".weight") || info.n_dims != 4) continue;
        if (source->type != GGML_TYPE_F16) return false;
        const int64_t k = source->ne[0] * source->ne[1] * source->ne[2], n = source->ne[3];
        const int64_t kp = GGML_PAD(k, 32), np = GGML_PAD(n, 32);
        std::vector<ggml_fp16_t> padded(size_t(kp * np), 0);
        const auto* data = static_cast<const ggml_fp16_t*>(source->data);
        for (int64_t row = 0; row < n; ++row) std::copy_n(data + row * k, k, padded.data() + row * kp);
        storage.emplace_back(size_t(kp * np));
        pack_htp_f16_weight_w16a32(storage.back().data(), padded.data(), kp, np);
        auto* packed = ggml_new_tensor_2d(packed_ctx, GGML_TYPE_F16, kp, np);
        ggml_set_name(packed, name.c_str());
        packed->data = storage.back().data();
        metadata.htp_vae_conv_tensors.push_back(name);
        metadata.htp_vae_conv_shapes.insert(metadata.htp_vae_conv_shapes.end(), source->ne, source->ne + 4);
        info = TensorWriteInfo{};
        info.tensor = packed;
        info.n_dims = 2;
        info.ne[0] = kp;
        info.ne[1] = np;
        bytes += ggml_nbytes(packed);
    }
    LOG_INFO("HTP packed VAE GGUF: %zu convolutions, %.2f MiB; non-convolution tensors preserve source precision",
             metadata.htp_vae_conv_tensors.size(), bytes / 1024.f / 1024.f);
    return !metadata.htp_vae_conv_tensors.empty();
}

static bool pack_htp_text_weights_for_export(std::vector<TensorWriteInfo>& tensors,
                                               const std::string& kind, GGUFHTPMetadata& metadata) {
    metadata.htp_packed_text = true;
    metadata.htp_text_encoder = kind;
    metadata.htp_layout = SD_GGUF_HTP_LAYOUT_W16A32;
    size_t bytes = 0;
    for (auto& info : tensors) {
        auto* t = info.tensor;
        if (!sd_htp_text_linear_name(t->name, kind)) continue;
        if (t->type != GGML_TYPE_F16 || info.n_dims != 2 || t->ne[0] % 32 || t->ne[1] % 32) {
            LOG_ERROR("unsupported HTP text Linear shape/type: %s", t->name);
            return false;
        }
        std::vector<ggml_fp16_t> packed(ggml_nelements(t));
        pack_htp_f16_weight_w16a32(packed.data(), static_cast<const ggml_fp16_t*>(t->data), t->ne[0], t->ne[1]);
        std::memcpy(t->data, packed.data(), ggml_nbytes(t));
        metadata.htp_packed_tensors.emplace_back(t->name);
        bytes += ggml_nbytes(t);
    }
    LOG_INFO("HTP packed %s: %zu Linear weights, %.2f MiB; embeddings/norms preserve source precision",
             kind.c_str(), metadata.htp_packed_tensors.size(), bytes / 1024.f / 1024.f);
    return !metadata.htp_packed_tensors.empty();
}

static bool convert_impl(const char* input_path,
             const char* vae_path,
             const char* output_path,
             sd_type_t output_type,
             const char* tensor_type_rules,
             bool convert_name,
             bool htp_pack_dit,
             bool force_f16,
             bool htp_pack_vae, const std::string& text_kind = "") {
    const bool output_is_safetensors = ends_with(output_path, ".safetensors");
    if ((htp_pack_dit || htp_pack_vae || !text_kind.empty()) && output_is_safetensors) {
        LOG_ERROR("HTP offline packing only supports GGUF output");
        return false;
    }
    if (htp_pack_dit) {
        LOG_WARN("HTP packed DIT GGUF stores selected weights in HTP-only layout. Use it only with SD_HTP_COMPUTE=1 and HTP backend available.");
    }

    ModelLoader model_loader;

    // HTP packing accepts standalone DiT files whose tensor names omit the runtime namespace.
    const std::string input_prefix = htp_pack_dit ? "model.diffusion_model." : "";
    const bool initialized = !text_kind.empty()
        ? model_loader.init_from_file_and_convert_name(input_path, sd_htp_text_prefix(text_kind), VERSION_SD3)
        : htp_pack_vae
        ? model_loader.init_from_file_and_convert_name(input_path, "vae.", VERSION_SD3)
        : model_loader.init_from_file(input_path, input_prefix);
    if (!initialized) {
        LOG_ERROR("init model loader from file failed: '%s'", input_path);
        return false;
    }

    if (vae_path != nullptr && strlen(vae_path) > 0) {
        if (!model_loader.init_from_file(vae_path, "vae.")) {
            LOG_ERROR("init model loader from file failed: '%s'", vae_path);
            return false;
        }
    }
    if (convert_name && !htp_pack_vae && text_kind.empty()) {
        model_loader.convert_tensors_name();
    }

    ggml_type type             = (ggml_type)output_type;
    TensorTypeRules type_rules = parse_tensor_type_rules(tensor_type_rules);

    auto backend    = ggml_backend_cpu_init();
    size_t mem_size = 1 * 1024 * 1024;  // for padding
    mem_size += model_loader.get_tensor_storage_map().size() * ggml_tensor_overhead();
    if (!text_kind.empty()) {
        for (const auto& item : model_loader.get_tensor_storage_map()) {
            auto storage = item.second;
            if (sd_htp_text_linear_name(storage.name, text_kind)) storage.type = GGML_TYPE_F16;
            mem_size += storage.nbytes() + 128;
        }
    } else {
        mem_size += model_loader.get_params_mem_size(backend, htp_pack_vae ? GGML_TYPE_F32 : type);
    }
    LOG_INFO("model tensors mem size: %.2fMB", mem_size / 1024.f / 1024.f);
    ggml_context* ggml_ctx = ggml_init({mem_size, nullptr, false});

    if (ggml_ctx == nullptr) {
        LOG_ERROR("ggml_init failed for converter");
        ggml_backend_free(backend);
        return false;
    }

    std::vector<TensorWriteInfo> tensors;
    bool success = load_tensors_for_export(model_loader, ggml_ctx, type, type_rules, force_f16, tensors, htp_pack_vae, text_kind);
    std::vector<std::string> packed_tensor_names;
    if (success && htp_pack_dit) {
        // --htp-pack-dit 只在导出 GGUF 时执行一次重排；推理阶段只消费这个 packed GGUF。
        success = pack_htp_dit_weights_for_export(tensors, packed_tensor_names);
    }
    ggml_context* packed_ctx = nullptr;
    std::vector<std::vector<ggml_fp16_t>> packed_storage;
    GGUFWriteOptions options;
    if (success && htp_pack_vae) {
        packed_ctx = ggml_init({(tensors.size() + 1) * ggml_tensor_overhead() + 1024, nullptr, true});
        success = packed_ctx != nullptr && pack_htp_vae_weights_for_export(packed_ctx, tensors, packed_storage, options.htp);
    }
    if (success && !text_kind.empty()) {
        success = pack_htp_text_weights_for_export(tensors, text_kind, options.htp);
    }
    ggml_backend_free(backend);

    std::string error;
    if (success) {
        if (output_is_safetensors) {
            success = write_safetensors_file(output_path, tensors, &error);
        } else {
            if (htp_pack_dit) {
                // 把离线 packed 布局写入 GGUF 元数据，加载端用它恢复 HTP-only tensor flag。
                options.htp.htp_packed_dit     = true;
                options.htp.htp_layout         = SD_GGUF_HTP_LAYOUT_W16A32;
                options.htp.htp_packed_tensors = packed_tensor_names;
            }
            success = write_gguf_file(output_path, tensors, &error, (htp_pack_dit || htp_pack_vae || !text_kind.empty()) ? &options : nullptr);
        }
    }

    if (!success && !error.empty()) {
        LOG_ERROR("%s", error.c_str());
    }

    ggml_free(packed_ctx);
    ggml_free(ggml_ctx);
    return success;
}

// Keep the existing conversion ABI intact.
bool convert(const char* input_path, const char* vae_path, const char* output_path,
             sd_type_t type, const char* rules, bool convert_name, bool pack_dit, bool force_f16) {
    return convert_impl(input_path, vae_path, output_path, type, rules, convert_name, pack_dit, force_f16, false);
}

bool convert_vae_htp(const char* input_path, const char* output_path) {
    return convert_impl(input_path, "", output_path, SD_TYPE_F16, "", true, false, false, true);
}

bool convert_text_encoder_htp(const char* input_path, const char* output_path, const char* kind) {
    const std::string encoder = kind ? kind : "";
    if (sd_htp_text_prefix(encoder).empty()) {
        LOG_ERROR("HTP text encoder kind must be clip_l, clip_g or t5xxl");
        return false;
    }
    return convert_impl(input_path, "", output_path, SD_TYPE_F16, "", true, false, false, false, encoder);
}
