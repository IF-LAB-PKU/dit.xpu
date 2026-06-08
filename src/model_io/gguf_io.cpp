#include "gguf_io.h"

#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "gguf.h"
#include "gguf_reader_ext.h"
#include "util.h"

// HTP packed DIT GGUF 通过这些私有元数据声明：文件里的部分权重已经不是普通 row-major，
// 而是只能给 HTP permuted W16A32 matmul 使用的离线重排布局。
static constexpr const char* SD_GGUF_HTP_PACKED_DIT_KEY     = "sd.cpp.htp.packed_dit";
static constexpr const char* SD_GGUF_HTP_LAYOUT_KEY         = "sd.cpp.htp.layout";
static constexpr const char* SD_GGUF_HTP_PACKED_TENSORS_KEY = "sd.cpp.htp.packed_tensors";

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

    if (options != nullptr && options->htp.htp_packed_dit) {
        // 写出离线重排 tensor 名单，加载端依赖这个名单禁止 packed 权重走 CPU fallback。
        gguf_set_val_bool(gguf_ctx, SD_GGUF_HTP_PACKED_DIT_KEY, true);
        gguf_set_val_str(gguf_ctx, SD_GGUF_HTP_LAYOUT_KEY, options->htp.htp_layout.c_str());

        std::vector<const char*> names;
        names.reserve(options->htp.htp_packed_tensors.size());
        for (const std::string& name : options->htp.htp_packed_tensors) {
            names.push_back(name.c_str());
        }
        gguf_set_arr_str(gguf_ctx, SD_GGUF_HTP_PACKED_TENSORS_KEY, names.data(), names.size());
    }

    LOG_INFO("trying to save tensors to %s", file_path.c_str());
    bool success = gguf_write_to_file(gguf_ctx, file_path.c_str(), false);
    if (!success) {
        set_error(error, "failed to write GGUF file '" + file_path + "'");
    }
    gguf_free(gguf_ctx);
    return success;
}
