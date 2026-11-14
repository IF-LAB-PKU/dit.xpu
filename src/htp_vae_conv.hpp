#ifndef SD_HTP_VAE_CONV_HPP
#define SD_HTP_VAE_CONV_HPP

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-htp.h"
#include "htp_tensor_flags.h"

#include <algorithm>
#include <map>
#include <memory>
#include <vector>

// Original VAE files retain row-major weights and cache a separate packed RPCMEM
// copy. Offline GGUF parameters already use that layout and are returned directly.
class HtpVaeConvWeights {
    struct Entry {
        ggml_context* ctx = nullptr;
        ggml_backend_buffer_t buffer = nullptr;
        ggml_tensor* tensor = nullptr;
        ~Entry() {
#ifdef GGML_USE_HTP
            if (tensor) ggml_backend_htp_unregister_packed_f16_weight(tensor);
#endif
            ggml_backend_buffer_free(buffer);
            ggml_free(ctx);
        }
    };
    std::map<const ggml_tensor*, std::unique_ptr<Entry>> entries_;

public:
    void clear() { entries_.clear(); }
    size_t size() const { return entries_.size(); }
    size_t bytes() const {
        size_t result = 0;
        for (const auto& item : entries_) result += ggml_backend_buffer_get_size(item.second->buffer);
        return result;
    }

    ggml_tensor* get(ggml_backend_t backend, ggml_tensor* weight) {
#ifdef GGML_USE_HTP
        if (!backend || !weight || weight->type != GGML_TYPE_F16 ||
            !ggml_is_contiguous(weight) || !weight->data) return nullptr;
        if ((weight->flags & SD_GGML_TENSOR_FLAG_HTP_VAE_CONV_WEIGHT) != 0) {
            // Already padded and tiled offline; registration is owned by the runner.
            GGML_ASSERT(ggml_n_dims(weight) == 2 && weight->ne[0] % 32 == 0 && weight->ne[1] % 32 == 0);
            return weight;
        }
        auto found = entries_.find(weight);
        if (found != entries_.end()) return found->second->tensor;
        const int64_t k = weight->ne[0] * weight->ne[1] * weight->ne[2];
        const int64_t n = weight->ne[3];
        const int64_t kp = GGML_PAD(k, 32), np = GGML_PAD(n, 32);
        auto entry = std::make_unique<Entry>();
        entry->ctx = ggml_init({ggml_tensor_overhead() + 1024, nullptr, true});
        GGML_ASSERT(entry->ctx);
        entry->tensor = ggml_new_tensor_2d(entry->ctx, GGML_TYPE_F16, kp, np);
        ggml_format_name(entry->tensor, "htp_vae_%s", weight->name);
        entry->tensor->flags |= SD_GGML_TENSOR_FLAG_HTP_PACKED_W16A32;
        auto buft = ggml_backend_get_default_buffer_type(backend);
        GGML_ASSERT(ggml_backend_buft_is_host(buft));
        entry->buffer = ggml_backend_alloc_ctx_tensors_from_buft(entry->ctx, buft);
        GGML_ASSERT(entry->buffer);
        ggml_backend_buffer_set_usage(entry->buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        std::vector<ggml_fp16_t> source(size_t(k * n));
        ggml_backend_tensor_get(weight, source.data(), 0, ggml_nbytes(weight));
        std::vector<ggml_fp16_t> padded(size_t(kp * np), 0);
        for (int64_t row = 0; row < n; ++row) {
            std::copy_n(source.data() + row * k, k, padded.data() + row * kp);
        }
        ggml_backend_htp_pack_f16_weight(static_cast<ggml_fp16_t*>(entry->tensor->data),
                                         padded.data(), kp, np);
        ggml_backend_htp_register_packed_f16_weight(entry->tensor, entry->tensor);
        auto* result = entry->tensor;
        entries_.emplace(weight, std::move(entry));
        return result;
#else
        (void) backend;
        (void) weight;
        return nullptr;
#endif
    }
};

// CPU im2col -> HTP F16 weight/F32 activation GEMM -> CPU channel-major output.
// Spatial positions and batches are flattened together, as required by HTP's
// two-dimensional GEMM. Padding supports the 16-channel input and RGB output.
inline ggml_tensor* ggml_htp_vae_conv_2d(ggml_context* ctx, ggml_tensor* x,
                                        ggml_tensor* weight, ggml_tensor* packed,
                                        int s0, int s1, int p0, int p1, int d0, int d1,
                                        size_t chunk_bytes = 128 * 1024 * 1024,
                                        ggml_tensor* bias = nullptr, bool fused = false) {
    GGML_ASSERT(x->type == GGML_TYPE_F32 && packed != nullptr);
    const char *native=getenv("SD_HTP_VAE_NATIVE_CONV");
    if (!native) native=getenv("SD_HTP_VAE_NATIVE");
    if (native && strcmp(native,"0")!=0) return ggml_htp_native_conv(ctx,packed,x,bias,
        weight->ne[0],weight->ne[1],weight->ne[3],s0,s1,p0,p1,d0,d1);
    auto* cols = ggml_im2col(ctx, weight, x, s0, s1, p0, p1, d0, d1, true, GGML_TYPE_F32);
    const int64_t ow = cols->ne[1], oh = cols->ne[2], batch = cols->ne[3];
    cols = ggml_reshape_2d(ctx, cols, cols->ne[0], ow * oh * batch);
    if (cols->ne[0] != packed->ne[0]) {
        cols = ggml_pad(ctx, cols, packed->ne[0] - cols->ne[0], 0, 0, 0);
    }
    const int64_t positions = cols->ne[1];
    const int64_t chunk_positions = std::max<int64_t>(1, chunk_bytes / (packed->ne[0] * sizeof(float)));
    ggml_tensor* result = nullptr;
    for (int64_t first = 0; first < positions; first += chunk_positions) {
        const int64_t count = std::min(chunk_positions, positions - first);
        auto* activation = cols;
        if (count != positions) {
            // Keep the full im2col in CPU RAM. Only a bounded contiguous copy
            // enters RPCMEM; otherwise a 512px decoder requests >1 GiB tensors
            // and can exhaust the RPCMEM allocator/mapping window.
            activation = ggml_cont(ctx, ggml_view_2d(ctx, cols, cols->ne[0], count,
                                                     cols->nb[1], first * cols->nb[1]));
        }
        auto* tile = fused ? ggml_htp_f16_epilogue(ctx,packed,activation,bias,3,weight->ne[3])
                           : ggml_mul_mat(ctx, packed, activation);
        if (!fused) ggml_format_name(tile, "htp_vae_conv_%s_%lld", weight->name, (long long) first);
        result = result ? ggml_concat(ctx, result, tile, fused ? 0 : 1) : tile;
    }
    if (fused) {
        result=ggml_reshape_4d(ctx,result,ow,oh,batch,weight->ne[3]);
        result=ggml_permute(ctx,result,0,1,3,2);
        return ggml_is_contiguous(result) ? result : ggml_cont(ctx,result);
    }
    result = ggml_reshape_4d(ctx, result, packed->ne[1], ow, oh, batch);
    if (weight->ne[3] != packed->ne[1]) {
        result = ggml_view_4d(ctx, result, weight->ne[3], ow, oh, batch,
                              result->nb[1], result->nb[2], result->nb[3], 0);
    }
    return ggml_cont(ctx, ggml_permute(ctx, result, 2, 0, 1, 3));
}

#endif
