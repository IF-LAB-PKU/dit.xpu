#ifndef __SD_MODEL_IO_GGUF_IO_H__
#define __SD_MODEL_IO_GGUF_IO_H__

#include <string>
#include <vector>

#include "tensor_storage.h"

struct GGUFHTPMetadata {
    bool htp_packed_dit = false;
    std::string htp_layout;
    std::vector<std::string> htp_packed_tensors;
};

struct GGUFWriteOptions {
    GGUFHTPMetadata htp;
};

bool is_gguf_file(const std::string& file_path);
bool read_gguf_file(const std::string& file_path,
                    std::vector<TensorStorage>& tensor_storages,
                    std::string* error = nullptr,
                    GGUFHTPMetadata* htp_metadata = nullptr);
bool write_gguf_file(const std::string& file_path,
                     const std::vector<TensorWriteInfo>& tensors,
                     std::string* error = nullptr,
                     const GGUFWriteOptions* options = nullptr);

#endif  // __SD_MODEL_IO_GGUF_IO_H__
