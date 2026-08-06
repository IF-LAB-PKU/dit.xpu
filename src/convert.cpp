#include <cstring>
#include <mutex>
#include <regex>
#include <vector>

#include "model.h"
#include "model_io/gguf_io.h"
#include "model_io/safetensors_io.h"
#include "util.h"

#include "ggml-cpu.h"

static constexpr const char* SD_GGUF_HTP_LAYOUT_W16A32 = "permuted_w16a32";

static ggml_type get_export_tensor_type(ModelLoader& model_loader,
                                        const TensorStorage& tensor_storage,
                                        ggml_type type,
                                        const TensorTypeRules& tensor_type_rules) {
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

    if (model_loader.tensor_should_be_converted(tensor_storage, dst_type)) {
        tensor_type = dst_type;
    }

    return tensor_type;
}

static bool load_tensors_for_export(ModelLoader& model_loader,
                                    ggml_context* ggml_ctx,
                                    ggml_type type,
                                    const TensorTypeRules& tensor_type_rules,
                                    std::vector<TensorWriteInfo>& tensors) {
    std::mutex tensor_mutex;
    auto on_new_tensor_cb = [&](const TensorStorage& tensor_storage, ggml_tensor** dst_tensor) -> bool {
        const std::string& name = tensor_storage.name;
        ggml_type tensor_type   = get_export_tensor_type(model_loader, tensor_storage, type, tensor_type_rules);

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

    for (int64_t i = 0; i < k; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            const int64_t i0 = i / 32;
            const int64_t i1 = i % 32;
            const int64_t j0 = j / 32;
            const int64_t j1 = j % 32;

            const int64_t tile_idx = j0 * (k / 32) + i0;
            ggml_fp16_t* tile = dst + tile_idx * 32 * 32;
            tile[(i1 & ~int64_t{1}) * 32 + j1 * 2 + (i1 & int64_t{1})] = src[j * k + i];
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

bool convert(const char* input_path,
             const char* vae_path,
             const char* output_path,
             sd_type_t output_type,
             const char* tensor_type_rules,
             bool convert_name,
             bool htp_pack_dit) {
    const bool output_is_safetensors = ends_with(output_path, ".safetensors");
    if (htp_pack_dit && output_is_safetensors) {
        LOG_ERROR("--htp-pack-dit only supports GGUF output");
        return false;
    }
    if (htp_pack_dit) {
        LOG_WARN("HTP packed DIT GGUF stores selected weights in HTP-only layout. Use it only with SD_HTP_COMPUTE=1 and HTP backend available.");
    }

    ModelLoader model_loader;

    // HTP packing accepts standalone DiT files whose tensor names omit the runtime namespace.
    const std::string input_prefix = htp_pack_dit ? "model.diffusion_model." : "";
    if (!model_loader.init_from_file(input_path, input_prefix)) {
        LOG_ERROR("init model loader from file failed: '%s'", input_path);
        return false;
    }

    if (vae_path != nullptr && strlen(vae_path) > 0) {
        if (!model_loader.init_from_file(vae_path, "vae.")) {
            LOG_ERROR("init model loader from file failed: '%s'", vae_path);
            return false;
        }
    }
    if (convert_name) {
        model_loader.convert_tensors_name();
    }

    ggml_type type             = (ggml_type)output_type;
    TensorTypeRules type_rules = parse_tensor_type_rules(tensor_type_rules);

    auto backend    = ggml_backend_cpu_init();
    size_t mem_size = 1 * 1024 * 1024;  // for padding
    mem_size += model_loader.get_tensor_storage_map().size() * ggml_tensor_overhead();
    mem_size += model_loader.get_params_mem_size(backend, type);
    LOG_INFO("model tensors mem size: %.2fMB", mem_size / 1024.f / 1024.f);
    ggml_context* ggml_ctx = ggml_init({mem_size, nullptr, false});

    if (ggml_ctx == nullptr) {
        LOG_ERROR("ggml_init failed for converter");
        ggml_backend_free(backend);
        return false;
    }

    std::vector<TensorWriteInfo> tensors;
    bool success = load_tensors_for_export(model_loader, ggml_ctx, type, type_rules, tensors);
    std::vector<std::string> packed_tensor_names;
    if (success && htp_pack_dit) {
        // --htp-pack-dit 只在导出 GGUF 时执行一次重排；推理阶段只消费这个 packed GGUF。
        success = pack_htp_dit_weights_for_export(tensors, packed_tensor_names);
    }
    ggml_backend_free(backend);

    std::string error;
    if (success) {
        if (output_is_safetensors) {
            success = write_safetensors_file(output_path, tensors, &error);
        } else {
            GGUFWriteOptions options;
            if (htp_pack_dit) {
                // 把离线 packed 布局写入 GGUF 元数据，加载端用它恢复 HTP-only tensor flag。
                options.htp.htp_packed_dit     = true;
                options.htp.htp_layout         = SD_GGUF_HTP_LAYOUT_W16A32;
                options.htp.htp_packed_tensors = packed_tensor_names;
            }
            success = write_gguf_file(output_path, tensors, &error, htp_pack_dit ? &options : nullptr);
        }
    }

    if (!success && !error.empty()) {
        LOG_ERROR("%s", error.c_str());
    }

    ggml_free(ggml_ctx);
    return success;
}
