#pragma once

#include <cstddef>
#include <vector>

// Per-tensor placement deliberately does not aggregate adjacent parameters.
// The configured limit is only a validation ceiling for each allocation.
static inline std::vector<size_t> htp_per_tensor_chunk_sizes(
        const std::vector<size_t> & tensor_sizes,
        size_t max_chunk_size) {
    std::vector<size_t> chunks;
    chunks.reserve(tensor_sizes.size());
    for (size_t size : tensor_sizes) {
        if (size == 0 || size > max_chunk_size) {
            return {};
        }
        chunks.push_back(size);
    }
    return chunks;
}
