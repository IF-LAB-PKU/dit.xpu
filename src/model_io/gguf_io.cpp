#include "gguf_io.h"

#include <cstdint>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "gguf.h"
#include "gguf_reader_ext.h"
#include "util.h"
#include "htp_text_encoder.h"

// HTP packed DIT GGUF 通过这些私有元数据声明：文件里的部分权重已经不是普通 row-major，
// 而是只能给 HTP permuted W16A32 matmul 使用的离线重排布局。
static constexpr const char* SD_GGUF_HTP_PACKED_DIT_KEY     = "sd.cpp.htp.packed_dit";
static constexpr const char* SD_GGUF_HTP_LAYOUT_KEY         = "sd.cpp.htp.layout";
static constexpr const char* SD_GGUF_HTP_PACKED_TENSORS_KEY = "sd.cpp.htp.packed_tensors";
static constexpr const char* SD_GGUF_HTP_PACKED_TEXT_KEY = "sd.cpp.htp.packed_text";
static constexpr const char* SD_GGUF_HTP_TEXT_ENCODER_KEY = "sd.cpp.htp.text_encoder";

static constexpr const char* SD_GGUF_HTP_PACKED_VAE_KEY = "sd.cpp.htp.packed_vae";
static constexpr const char* SD_GGUF_HTP_VAE_TENSORS_KEY = "sd.cpp.htp.vae_conv_tensors";
static constexpr const char* SD_GGUF_HTP_VAE_SHAPES_KEY = "sd.cpp.htp.vae_conv_shapes";

static void set_error(std::string* error, const std::string& message) {
    if (error != nullptr) {
        *error = message;
    }
}

static void read_newline_delimited_strings(const char* value, std::vector<std::string>& out) {
    if (value == nullptr) {
        return;
    }
    std::stringstream ss(value);
    std::string item;
    while (std::getline(ss, item)) {
        if (!item.empty()) {
            out.push_back(item);
        }
    }
}

static void read_htp_metadata_from_gguf(const gguf_context* ctx, GGUFHTPMetadata* metadata) {
    if (ctx == nullptr || metadata == nullptr) {
        return;
    }

    // 读取 packed 标记、布局名和 tensor 名单；后续只按名单给 TensorStorage 打 layout 标记，
    // 不尝试从 shape/type 推断，避免普通 F16 weight 被误当作 HTP-only packed 权重。
    const int64_t packed_dit_key = gguf_find_key(ctx, SD_GGUF_HTP_PACKED_DIT_KEY);
    if (packed_dit_key >= 0 && gguf_get_kv_type(ctx, packed_dit_key) == GGUF_TYPE_BOOL) {
        metadata->htp_packed_dit = gguf_get_val_bool(ctx, packed_dit_key);
    }

    const int64_t layout_key = gguf_find_key(ctx, SD_GGUF_HTP_LAYOUT_KEY);
    if (layout_key >= 0 && gguf_get_kv_type(ctx, layout_key) == GGUF_TYPE_STRING) {
        const char* layout = gguf_get_val_str(ctx, layout_key);
        metadata->htp_layout = layout != nullptr ? layout : "";
    }

    const auto text_key = gguf_find_key(ctx, SD_GGUF_HTP_PACKED_TEXT_KEY);
    if (text_key >= 0 && gguf_get_kv_type(ctx, text_key) == GGUF_TYPE_BOOL) {
        metadata->htp_packed_text = gguf_get_val_bool(ctx, text_key);
    }
    const auto encoder_key = gguf_find_key(ctx, SD_GGUF_HTP_TEXT_ENCODER_KEY);
    if (encoder_key >= 0 && gguf_get_kv_type(ctx, encoder_key) == GGUF_TYPE_STRING) {
        metadata->htp_text_encoder = gguf_get_val_str(ctx, encoder_key);
    }

    const int64_t vae_key = gguf_find_key(ctx, SD_GGUF_HTP_PACKED_VAE_KEY);
    if (vae_key >= 0 && gguf_get_kv_type(ctx, vae_key) == GGUF_TYPE_BOOL) {
        metadata->htp_packed_vae = gguf_get_val_bool(ctx, vae_key);
    }
    const int64_t vae_names = gguf_find_key(ctx, SD_GGUF_HTP_VAE_TENSORS_KEY);
    if (vae_names >= 0 && gguf_get_kv_type(ctx, vae_names) == GGUF_TYPE_ARRAY &&
        gguf_get_arr_type(ctx, vae_names) == GGUF_TYPE_STRING) {
        for (size_t i = 0; i < gguf_get_arr_n(ctx, vae_names); ++i) {
            metadata->htp_vae_conv_tensors.emplace_back(gguf_get_arr_str(ctx, vae_names, i));
        }
    }
    const int64_t vae_shapes = gguf_find_key(ctx, SD_GGUF_HTP_VAE_SHAPES_KEY);
    if (vae_shapes >= 0 && gguf_get_kv_type(ctx, vae_shapes) == GGUF_TYPE_ARRAY &&
        gguf_get_arr_type(ctx, vae_shapes) == GGUF_TYPE_INT64) {
        const auto* shapes = static_cast<const int64_t*>(gguf_get_arr_data(ctx, vae_shapes));
        metadata->htp_vae_conv_shapes.assign(shapes, shapes + gguf_get_arr_n(ctx, vae_shapes));
    }

    const int64_t packed_tensors_key = gguf_find_key(ctx, SD_GGUF_HTP_PACKED_TENSORS_KEY);
    if (packed_tensors_key < 0) {
        return;
    }

    const gguf_type kv_type = gguf_get_kv_type(ctx, packed_tensors_key);
    if (kv_type == GGUF_TYPE_ARRAY && gguf_get_arr_type(ctx, packed_tensors_key) == GGUF_TYPE_STRING) {
        const size_t n = gguf_get_arr_n(ctx, packed_tensors_key);
        metadata->htp_packed_tensors.reserve(metadata->htp_packed_tensors.size() + n);
        for (size_t i = 0; i < n; ++i) {
            const char* name = gguf_get_arr_str(ctx, packed_tensors_key, i);
            if (name != nullptr) {
                metadata->htp_packed_tensors.push_back(name);
            }
        }
    } else if (kv_type == GGUF_TYPE_STRING) {
        read_newline_delimited_strings(gguf_get_val_str(ctx, packed_tensors_key), metadata->htp_packed_tensors);
    }
}

static bool apply_text_metadata(const GGUFHTPMetadata& metadata, std::vector<TensorStorage>& tensors, std::string* error) {
    if (!metadata.htp_packed_text) return true;
    if (metadata.htp_packed_dit || metadata.htp_packed_vae ||
        sd_htp_text_prefix(metadata.htp_text_encoder).empty() ||
        metadata.htp_layout != "permuted_w16a32" || metadata.htp_packed_tensors.empty()) {
        set_error(error, "invalid HTP text encoder metadata");
        return false;
    }
    std::map<std::string, TensorStorage*> by_name;
    for (auto& t : tensors) by_name.emplace(t.name, &t);
    std::unordered_set<std::string> seen;
    for (const auto& name : metadata.htp_packed_tensors) {
        const auto found = by_name.find(name);
        if (found == by_name.end() || !seen.insert(name).second ||
            !sd_htp_text_linear_name(name, metadata.htp_text_encoder)) {
            set_error(error, "invalid/missing/duplicate packed text Linear: " + name);
            return false;
        }
        auto& t = *found->second;
        if (t.type != GGML_TYPE_F16 || t.n_dims != 2 || t.ne[0] % 32 || t.ne[1] % 32) {
            set_error(error, "invalid packed text Linear shape/type: " + name);
            return false;
        }
        t.layout = TensorDataLayout::HTP_PERMUTED_W16A32;
    }
    return true;
}

static void apply_htp_metadata_to_tensors(const GGUFHTPMetadata& metadata, std::vector<TensorStorage>& tensor_storages) {
    if (!metadata.htp_packed_dit || metadata.htp_packed_tensors.empty()) {
        return;
    }

    // 将 GGUF 元数据转换成运行时可见的 tensor layout；真正的 ggml tensor flag 会在模型加载阶段设置。
    std::unordered_set<std::string> packed_names(metadata.htp_packed_tensors.begin(), metadata.htp_packed_tensors.end());
    for (TensorStorage& tensor_storage : tensor_storages) {
        if (packed_names.count(tensor_storage.name) > 0) {
            tensor_storage.layout = TensorDataLayout::HTP_PERMUTED_W16A32;
        }
    }
}

static bool apply_vae_metadata(const GGUFHTPMetadata& metadata,
                                std::vector<TensorStorage>& tensors, std::string* error) {
    if (!metadata.htp_packed_vae) {
        if (!metadata.htp_vae_conv_tensors.empty() || !metadata.htp_vae_conv_shapes.empty()) {
            set_error(error, "VAE packed tensors require sd.cpp.htp.packed_vae=true");
            return false;
        }
        return true;
    }
    const auto& names = metadata.htp_vae_conv_tensors;
    const auto& shapes = metadata.htp_vae_conv_shapes;
    if (metadata.htp_layout != "permuted_w16a32" || names.empty() || shapes.size() != names.size() * 4) {
        set_error(error, "invalid HTP VAE layout/shape metadata");
        return false;
    }
    std::map<std::string, TensorStorage*> by_name;
    for (auto& tensor : tensors) by_name.emplace(tensor.name, &tensor);
    std::unordered_set<std::string> seen;
    for (size_t i = 0; i < names.size(); ++i) {
        auto found = by_name.find(names[i]);
        if (found == by_name.end() || !seen.insert(names[i]).second) {
            set_error(error, "missing or duplicate packed VAE tensor: " + names[i]);
            return false;
        }
        auto& tensor = *found->second;
        int64_t k = 1;
        for (int axis = 0; axis < 4; ++axis) {
            const int64_t dim = shapes[4 * i + axis];
            if (dim <= 0 || dim > std::numeric_limits<int32_t>::max() ||
                (axis < 3 && k > (std::numeric_limits<int64_t>::max() - 31) / dim)) {
                set_error(error, "invalid packed VAE convolution shape: " + names[i]);
                return false;
            }
            tensor.htp_conv_shape[axis] = dim;
            if (axis < 3) k *= dim;
        }
        const int64_t n = tensor.htp_conv_shape[3];
        if (tensor.type != GGML_TYPE_F16 || tensor.n_dims != 2 ||
            tensor.ne[0] != GGML_PAD(k, 32) || tensor.ne[1] != GGML_PAD(n, 32) ||
            tensor.layout != TensorDataLayout::REFERENCE) {
            set_error(error, "packed VAE tensor has incompatible type, size or layout: " + names[i]);
            return false;
        }
        tensor.layout = TensorDataLayout::HTP_CONV2D_W16A32;
    }
    return true;
}

bool is_gguf_file(const std::string& file_path) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    char magic[4];

    file.read(magic, sizeof(magic));
    if (!file) {
        return false;
    }
    for (uint32_t i = 0; i < sizeof(magic); i++) {
        if (magic[i] != GGUF_MAGIC[i]) {
            return false;
        }
    }

    return true;
}

bool read_gguf_file(const std::string& file_path,
                    std::vector<TensorStorage>& tensor_storages,
                    std::string* error,
                    GGUFHTPMetadata* htp_metadata) {
    if (htp_metadata != nullptr) {
        *htp_metadata = GGUFHTPMetadata{};
    }
    tensor_storages.clear();

    gguf_context* ctx_gguf_ = nullptr;
    ggml_context* ctx_meta_ = nullptr;

    ctx_gguf_ = gguf_init_from_file(file_path.c_str(), {true, &ctx_meta_});
    if (!ctx_gguf_) {
        GGUFReader gguf_reader;
        if (!gguf_reader.load(file_path)) {
            set_error(error, "failed to open '" + file_path + "' with GGUFReader");
            return false;
        }

        size_t data_offset = gguf_reader.data_offset();
        for (const auto& gguf_tensor_info : gguf_reader.tensors()) {
            TensorStorage tensor_storage(
                gguf_tensor_info.name,
                gguf_tensor_info.type,
                gguf_tensor_info.shape.data(),
                static_cast<int>(gguf_tensor_info.shape.size()),
                0,
                data_offset + gguf_tensor_info.offset);

            tensor_storages.push_back(tensor_storage);
        }

        return true;
    }

    GGUFHTPMetadata metadata;
    read_htp_metadata_from_gguf(ctx_gguf_, &metadata);
    const auto text_key = gguf_find_key(ctx_gguf_, SD_GGUF_HTP_PACKED_TEXT_KEY);
    const auto encoder_key = gguf_find_key(ctx_gguf_, SD_GGUF_HTP_TEXT_ENCODER_KEY);
    if ((text_key >= 0 && gguf_get_kv_type(ctx_gguf_, text_key) != GGUF_TYPE_BOOL) ||
        (encoder_key >= 0 && !metadata.htp_packed_text)) {
        set_error(error, "invalid HTP text encoder flag");
        gguf_free(ctx_gguf_);
        ggml_free(ctx_meta_);
        return false;
    }
    const int64_t vae_key = gguf_find_key(ctx_gguf_, SD_GGUF_HTP_PACKED_VAE_KEY);
    const int64_t vae_names = gguf_find_key(ctx_gguf_, SD_GGUF_HTP_VAE_TENSORS_KEY);
    const int64_t vae_shapes = gguf_find_key(ctx_gguf_, SD_GGUF_HTP_VAE_SHAPES_KEY);
    if ((vae_key >= 0 && gguf_get_kv_type(ctx_gguf_, vae_key) != GGUF_TYPE_BOOL) ||
        ((vae_names >= 0 || vae_shapes >= 0) && !metadata.htp_packed_vae)) {
        set_error(error, "invalid HTP VAE metadata flag");
        gguf_free(ctx_gguf_);
        ggml_free(ctx_meta_);
        return false;
    }

    int n_tensors = static_cast<int>(gguf_get_n_tensors(ctx_gguf_));

    size_t data_offset = gguf_get_data_offset(ctx_gguf_);
    for (int i = 0; i < n_tensors; i++) {
        std::string name   = gguf_get_tensor_name(ctx_gguf_, i);
        ggml_tensor* dummy = ggml_get_tensor(ctx_meta_, name.c_str());
        size_t offset      = data_offset + gguf_get_tensor_offset(ctx_gguf_, i);

        TensorStorage tensor_storage(name, dummy->type, dummy->ne, ggml_n_dims(dummy), 0, offset);

        if (ggml_nbytes(dummy) != tensor_storage.nbytes()) {
            gguf_free(ctx_gguf_);
            ggml_free(ctx_meta_);
            set_error(error, "size mismatch for tensor '" + name + "'");
            return false;
        }

        tensor_storages.push_back(tensor_storage);
    }

    apply_htp_metadata_to_tensors(metadata, tensor_storages);
    if (!apply_text_metadata(metadata, tensor_storages, error) ||
        !apply_vae_metadata(metadata, tensor_storages, error)) {
        gguf_free(ctx_gguf_);
        ggml_free(ctx_meta_);
        return false;
    }
    if (htp_metadata != nullptr) {
        *htp_metadata = metadata;
    }

    gguf_free(ctx_gguf_);
    ggml_free(ctx_meta_);

    return true;
}

bool write_gguf_file(const std::string& file_path,
                     const std::vector<TensorWriteInfo>& tensors,
                     std::string* error,
                     const GGUFWriteOptions* options) {
    gguf_context* gguf_ctx = gguf_init_empty();
    if (gguf_ctx == nullptr) {
        set_error(error, "gguf_init_empty failed");
        return false;
    }

    for (const TensorWriteInfo& write_tensor : tensors) {
        ggml_tensor* tensor = write_tensor.tensor;
        if (tensor == nullptr) {
            set_error(error, "null tensor cannot be written to GGUF");
            gguf_free(gguf_ctx);
            return false;
        }
        gguf_add_tensor(gguf_ctx, tensor);
    }

    if (options != nullptr && (options->htp.htp_packed_dit || options->htp.htp_packed_text)) {
        // 写出离线重排 tensor 名单，加载端依赖这个名单禁止 packed 权重走 CPU fallback。
        if (options->htp.htp_packed_text) {
            gguf_set_val_bool(gguf_ctx, SD_GGUF_HTP_PACKED_TEXT_KEY, true);
            gguf_set_val_str(gguf_ctx, SD_GGUF_HTP_TEXT_ENCODER_KEY, options->htp.htp_text_encoder.c_str());
        } else {
            gguf_set_val_bool(gguf_ctx, SD_GGUF_HTP_PACKED_DIT_KEY, true);
        }
        gguf_set_val_str(gguf_ctx, SD_GGUF_HTP_LAYOUT_KEY, options->htp.htp_layout.c_str());

        std::vector<const char*> names;
        names.reserve(options->htp.htp_packed_tensors.size());
        for (const std::string& name : options->htp.htp_packed_tensors) {
            names.push_back(name.c_str());
        }
        gguf_set_arr_str(gguf_ctx, SD_GGUF_HTP_PACKED_TENSORS_KEY, names.data(), names.size());
    }

    if (options != nullptr && options->htp.htp_packed_vae) {
        const auto& metadata = options->htp;
        gguf_set_val_bool(gguf_ctx, SD_GGUF_HTP_PACKED_VAE_KEY, true);
        gguf_set_val_str(gguf_ctx, SD_GGUF_HTP_LAYOUT_KEY, metadata.htp_layout.c_str());
        std::vector<const char*> names;
        for (const auto& name : metadata.htp_vae_conv_tensors) names.push_back(name.c_str());
        gguf_set_arr_str(gguf_ctx, SD_GGUF_HTP_VAE_TENSORS_KEY, names.data(), names.size());
        gguf_set_arr_data(gguf_ctx, SD_GGUF_HTP_VAE_SHAPES_KEY, GGUF_TYPE_INT64,
                          metadata.htp_vae_conv_shapes.data(), metadata.htp_vae_conv_shapes.size());
    }

    LOG_INFO("trying to save tensors to %s", file_path.c_str());
    bool success = gguf_write_to_file(gguf_ctx, file_path.c_str(), false);
    if (!success) {
        set_error(error, "failed to write GGUF file '" + file_path + "'");
    }
    gguf_free(gguf_ctx);
    return success;
}
