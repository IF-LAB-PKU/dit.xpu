#ifndef SD_HTP_DENSE_MASK_H
#define SD_HTP_DENSE_MASK_H

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <tuple>

// Immutable masks own their backing independently of per-forward graph arenas.
class HtpDenseMaskCache {
    struct Entry {
        ggml_context* ctx = nullptr;
        ggml_backend_buffer_t buffer = nullptr;
        ggml_tensor* tensor = nullptr;
        bool used = false;

        ~Entry() {
            ggml_backend_buffer_free(buffer);
            ggml_free(ctx);
        }
    };
    using Key = std::tuple<int64_t, int64_t, int64_t>;
    std::map<Key, std::unique_ptr<Entry>> entries_;
    ggml_backend_buffer_type_t buft_ = nullptr;

public:
    void begin_graph() {
        for (auto& item : entries_) item.second->used = false;
    }

    void end_graph() {
        // Dense HTP dispatch is synchronous; old graph metadata has been freed.
        for (auto it = entries_.begin(); it != entries_.end();) {
            if (!it->second->used) it = entries_.erase(it);
            else ++it;
        }
    }

    void clear() {
        entries_.clear();
        buft_ = nullptr;
    }

    size_t size() const { return entries_.size(); }

    ggml_tensor* get(ggml_context* graph_ctx, ggml_backend_buffer_type_t buft,
                     int64_t queries, int64_t keys, int64_t stride) {
        if (graph_ctx == nullptr || buft == nullptr || !ggml_backend_buft_is_host(buft) ||
            queries <= 0 || keys <= 0 || stride < keys || stride % 64 != 0 ||
            static_cast<uint64_t>(stride) > SIZE_MAX / sizeof(ggml_fp16_t) /
                static_cast<uint64_t>(queries)) {
            return nullptr;
        }
        // The owning runner clears the cache before changing its HTP backend.
        if (buft_ != nullptr && buft_ != buft) return nullptr;
        const Key key{queries, keys, stride};
        auto it = entries_.find(key);
        if (it == entries_.end()) {
            auto entry = std::make_unique<Entry>();
            entry->ctx = ggml_init({ggml_tensor_overhead() + 1024, nullptr, true});
            if (entry->ctx == nullptr) return nullptr;
            entry->tensor = ggml_new_tensor_2d(entry->ctx, GGML_TYPE_F16, stride, queries);
            ggml_format_name(entry->tensor, "htp_dense_mask_%lld_%lld_%lld",
                             (long long) queries, (long long) keys, (long long) stride);
            entry->buffer = ggml_backend_alloc_ctx_tensors_from_buft(entry->ctx, buft);
            if (entry->buffer == nullptr) return nullptr;
            auto* data = static_cast<ggml_fp16_t*>(entry->tensor->data);
            const ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity());
            std::memset(data, 0, ggml_nbytes(entry->tensor));
            if (stride != keys) {
                for (int64_t row = 0; row < queries; ++row) {
                    std::fill(data + row * stride + keys, data + (row + 1) * stride, neg_inf);
                }
            }
            buft_ = buft;
            it = entries_.emplace(key, std::move(entry)).first;
        }
        it->second->used = true;
        ggml_tensor* view = ggml_view_tensor(graph_ctx, it->second->tensor);
        if (ggml_backend_view_init(view) != GGML_STATUS_SUCCESS) return nullptr;
        return view;
    }
};

#endif
