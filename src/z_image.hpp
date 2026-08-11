#ifndef __Z_IMAGE_HPP__
#define __Z_IMAGE_HPP__

#include <algorithm>

#include "flux.hpp"
#include "ggml_extend.hpp"
#include "mmdit.hpp"

// Ref: https://github.com/Alpha-VLLM/Lumina-Image-2.0/blob/main/models/model.py
// Ref: https://github.com/huggingface/diffusers/pull/12703

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

namespace ZImage {
    constexpr int Z_IMAGE_GRAPH_SIZE = 20480;
    constexpr int ADALN_EMBED_DIM    = 256;
    constexpr int SEQ_MULTI_OF       = 32;

    // Static-halo eligibility predicate for Z-Image dimensions.
    __STATIC_INLINE__ bool z_image_static_halo_eligible(int image_tokens,
                                                        int token_h,
                                                        int token_w,
                                                        int text_tokens) {
        return image_tokens % 32 == 0 &&
               token_h % 4 == 0 &&
               token_w % 8 == 0 &&
               text_tokens >= 0;
    }

    // Check whether a Z-Image graph can use the sparse layout.
    __STATIC_INLINE__ bool z_image_sparse_layout_eligible(int image_tokens,
                                                           int token_h,
                                                           int token_w,
                                                           int text_tokens,
                                                           bool has_ref_latents) {
        return !has_ref_latents &&
               z_image_static_halo_eligible(image_tokens, token_h, token_w, text_tokens);
    }

    // Packed sparse attention can represent a single stream as an image region
    // with no text suffix. Text-only refiners use a synthetic grid and a full
    // keep mask; image-only refiners use the real image grid and halo pruning.
    __STATIC_INLINE__ bool z_image_packed_stream_eligible(int image_tokens,
                                                           int text_tokens,
                                                           int token_h,
                                                           int token_w,
                                                           bool full_keep) {
        return image_tokens > 0 && image_tokens % 32 == 0 &&
               text_tokens >= 0 && token_h > 0 && token_w > 0 &&
               token_h % 4 == 0 && token_w % 8 == 0 &&
               token_h * token_w == image_tokens &&
               (text_tokens == 0 || !full_keep);
    }

    enum class ZImageAttentionRoute {
        DENSE,
        REORDERED,
        PACKED,
    };

    struct ZImageAttentionRouteDecision {
        ZImageAttentionRoute route;
        const char* fallback_reason;
    };

    // Keep route selection independent of graph construction so the fallback
    // matrix can be tested without requiring a backend or model weights.
    __STATIC_INLINE__ ZImageAttentionRouteDecision z_image_select_attention_route(
        bool htp_enabled,
        bool sparse_enabled,
        bool packed_requested,
        bool geometry_supported,
        bool packed_shape_supported,
        bool packed_weights_available,
        bool has_ref_latents,
        bool residual_anchor) {
        if (!htp_enabled) {
            return {ZImageAttentionRoute::DENSE, "htp_disabled"};
        }
        if (has_ref_latents) {
            return {ZImageAttentionRoute::DENSE, "ref_latents"};
        }
        if (residual_anchor) {
            return {ZImageAttentionRoute::DENSE, "residual_anchor"};
        }
        if (!sparse_enabled) {
            return {ZImageAttentionRoute::DENSE, "sparse_disabled"};
        }
        if (!geometry_supported) {
            return {ZImageAttentionRoute::DENSE, "unsupported_geometry"};
        }
        if (!packed_requested) {
            return {ZImageAttentionRoute::REORDERED, "packed_disabled"};
        }
        if (!packed_shape_supported) {
            return {ZImageAttentionRoute::REORDERED, "packed_unsupported_shape"};
        }
        if (!packed_weights_available) {
            return {ZImageAttentionRoute::REORDERED, "packed_weights_missing"};
        }
        return {ZImageAttentionRoute::PACKED, "none"};
    }

    __STATIC_INLINE__ const char* z_image_attention_route_name(ZImageAttentionRoute route) {
        switch (route) {
            case ZImageAttentionRoute::DENSE:     return "dense";
            case ZImageAttentionRoute::REORDERED: return "reordered";
            case ZImageAttentionRoute::PACKED:    return "packed";
        }
        return "dense";
    }

    // Generate a patch-major reorder index for an image token grid.
    __STATIC_INLINE__ std::vector<int> z_image_patch_major_reorder_map(int token_h,
                                                                        int token_w,
                                                                        int patch) {
        int image_tokens = token_h * token_w;
        std::vector<int> map(image_tokens);
        for (int i = 0; i < image_tokens; ++i) {
            map[i] = sd_patch_major_index_from_row_major(i, token_h, token_w, patch);
        }
        return map;
    }

    // Generate an inverse patch-major reorder index.
    __STATIC_INLINE__ std::vector<int> z_image_row_major_reorder_map(int token_h,
                                                                      int token_w,
                                                                      int patch) {
        int image_tokens = token_h * token_w;
        std::vector<int> map(image_tokens);
        for (int i = 0; i < image_tokens; ++i) {
            map[i] = sd_row_major_index_from_patch_major(i, token_h, token_w, patch);
        }
        return map;
    }

    // Packed QKV helpers for Z-Image 30-head attention.
    struct PackedRopeQkv {
        std::vector<ggml_tensor*> qkv;
        ggml_tensor* storage = nullptr;
    };

    __STATIC_INLINE__ bool packed_z_image_eligible(const GGMLRunnerContext* ctx,
                                                    int image_tokens,
                                                    int text_tokens,
                                                    ggml_tensor* pe) {
        return ctx != nullptr &&
               sd_env_flag_enabled("SD_HTP_PACKED_Z_IMAGE") &&
               sd_env_flag_enabled_default("SD_HTP_PACKED_ATTENTION", true) &&
               sd_env_flag_enabled("SD_HTP_COMPUTE") &&
               ctx->block_sparse_attn_enabled &&
               ctx->block_sparse_residual_mode != SD_HTP_BLOCK_SPARSE_RESIDUAL_ANCHOR &&
               z_image_packed_stream_eligible(image_tokens, text_tokens,
                                              ctx->block_sparse_token_h,
                                              ctx->block_sparse_token_w,
                                              false) &&
               pe != nullptr && pe->type == GGML_TYPE_F32 &&
               pe->ne[0] == 2 && pe->ne[1] == 2 &&
               pe->ne[2] == 64 && pe->ne[3] >= image_tokens + text_tokens;
    }

    __STATIC_INLINE__ bool packed_z_image_qkv_branch_valid(ggml_tensor* activation,
                                                            ggml_tensor* weight,
                                                            ggml_tensor* q_norm,
                                                            ggml_tensor* k_norm) {
        constexpr int n_heads  = 30;
        constexpr int head_dim = 128;
        constexpr int qkv_dim  = 3 * n_heads * head_dim;  // 11520
        // Z-Image has no QKV bias, so bias is nullptr is valid
        return activation != nullptr && activation->type == GGML_TYPE_F32 && activation->ne[2] == 1 &&
               weight != nullptr && weight->type == GGML_TYPE_F16 &&
               weight->ne[0] == activation->ne[0] && weight->ne[1] == qkv_dim &&
               (weight->flags & SD_GGML_TENSOR_FLAG_HTP_PACKED_W16A32) != 0 &&
               q_norm != nullptr && q_norm->type == GGML_TYPE_F32 && q_norm->ne[0] == head_dim &&
               k_norm != nullptr && k_norm->type == GGML_TYPE_F32 && k_norm->ne[0] == head_dim;
    }

    __STATIC_INLINE__ PackedRopeQkv make_packed_z_image_qkv_views(GGMLRunnerContext* ctx,
                                                                    ggml_tensor* storage,
                                                                    int tokens) {
        PackedRopeQkv result;
        result.storage = storage;
        constexpr int n_heads  = 30;
        constexpr int head_dim = 128;
        const int token_tiles  = (tokens + 31) / 32;
        const size_t qkv_bytes = (size_t)n_heads * (size_t)token_tiles *
                                 (size_t)(head_dim / 32) * (size_t)1024 * sizeof(ggml_fp16_t);
        const size_t head_stride = (size_t)token_tiles * (size_t)(head_dim / 32) *
                                    (size_t)1024 * sizeof(ggml_fp16_t);
        result.qkv = {
            ggml_view_4d(ctx->ggml_ctx, storage, head_dim, tokens, n_heads, 1,
                         sizeof(ggml_fp16_t), head_dim * sizeof(ggml_fp16_t), head_stride, 0),
            ggml_view_4d(ctx->ggml_ctx, storage, head_dim, tokens, n_heads, 1,
                         sizeof(ggml_fp16_t), head_dim * sizeof(ggml_fp16_t), head_stride, qkv_bytes),
            ggml_view_4d(ctx->ggml_ctx, storage, head_dim, tokens, n_heads, 1,
                         sizeof(ggml_fp16_t), head_dim * sizeof(ggml_fp16_t), head_stride, 2 * qkv_bytes),
        };
        for (ggml_tensor* t : result.qkv) {
            t->op   = GGML_OP_NONE;
            t->type = GGML_TYPE_F16;
        }
        return result;
    }

    __STATIC_INLINE__ PackedRopeQkv make_packed_z_image_qkv(
        GGMLRunnerContext* ctx,
        ggml_tensor* activation,
        ggml_tensor* weight,
        ggml_tensor* q_norm,
        ggml_tensor* k_norm,
        ggml_tensor* pe,
        int image_tokens) {
        PackedRopeQkv result;
        if (!sd_env_flag_enabled("SD_HTP_PACKED_Z_IMAGE")) {
            return result;
        }
        const int tokens = activation != nullptr ? static_cast<int>(activation->ne[1]) : 0;
        const int text_tokens = tokens - image_tokens;
        if (!packed_z_image_eligible(ctx, image_tokens, text_tokens, pe)) {
            return result;
        }
        if (!packed_z_image_qkv_branch_valid(activation, weight, q_norm, k_norm)) {
            if (sd_env_flag_enabled("SD_HTP_PACKED_Z_IMAGE_TRACE")) {
                LOG_INFO("z_image_packed_qkv skip=invalid_branch weight_flags=%u heads=30",
                         weight != nullptr ? weight->flags : 0);
            }
            return result;
        }
        const int token_tiles = (tokens + 31) / 32;
        constexpr int n_heads  = 30;
        constexpr int head_dim = 128;
        const size_t region_bytes = (size_t)n_heads * (size_t)token_tiles *
                                     (size_t)(head_dim / 32) * (size_t)1024 * sizeof(ggml_fp16_t);
        // Z-Image has no bias: pass nullptr bias and null bias tensor
        ggml_tensor* null_bias = nullptr;
        ggml_tensor* args[] = {activation, weight, null_bias, q_norm, k_norm, pe};
        ggml_tensor* storage = ggml_custom_4d(ctx->ggml_ctx, GGML_TYPE_F16,
                                               static_cast<int64_t>(3 * region_bytes / sizeof(ggml_fp16_t)),
                                               1, 1, 1, args, 6,
                                               [](ggml_tensor*, int, int, void*) {}, 1, nullptr);
        ggml_set_name(storage, "qkv_mul_rms_packed_rope");
        if (sd_env_flag_enabled("SD_HTP_PACKED_Z_IMAGE_TRACE")) {
            LOG_INFO("z_image_packed_qkv kind=single heads=30 image_tokens=%d text_tokens=%d head_dim=%d",
                     image_tokens, text_tokens, head_dim);
        }
        return make_packed_z_image_qkv_views(ctx, storage, tokens);
    }

    struct JointAttention : public GGMLBlock {
    protected:
        int64_t head_dim;
        int64_t num_heads;
        int64_t num_kv_heads;
        bool qk_norm;

    public:
        JointAttention(int64_t hidden_size, int64_t head_dim, int64_t num_heads, int64_t num_kv_heads, bool qk_norm)
            : head_dim(head_dim), num_heads(num_heads), num_kv_heads(num_kv_heads), qk_norm(qk_norm) {
            blocks["qkv"] = std::make_shared<Linear>(hidden_size, (num_heads + num_kv_heads * 2) * head_dim, false);
            float scale   = 1.f;
            blocks["out"] = std::make_shared<Linear>(num_heads * head_dim, hidden_size, false, false, false, scale);
            if (qk_norm) {
                blocks["q_norm"] = std::make_shared<RMSNorm>(head_dim);
                blocks["k_norm"] = std::make_shared<RMSNorm>(head_dim);
            }
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* pe,
                             ggml_tensor* mask            = nullptr,
                             bool block_sparse_attn       = false,
                             int protected_start          = 0,
                             int protected_len            = 0,
                             bool block_sparse_force_full_keep = false) {
            // x: [N, n_token, hidden_size]
            int64_t n_token = x->ne[1];
            int64_t N       = x->ne[2];
            auto qkv_proj   = std::dynamic_pointer_cast<Linear>(blocks["qkv"]);
            auto out_proj   = std::dynamic_pointer_cast<Linear>(blocks["out"]);

            if (sd_backend_is(ctx->backend, "ROCm")) {
                out_proj->set_scale(1.f / 16.f);
            }

            const bool single_stream_sparse = block_sparse_attn && protected_start == 0 && protected_len == 0;
            const int image_tokens = block_sparse_attn && !single_stream_sparse ? protected_start : static_cast<int>(n_token);
            const int text_tokens  = block_sparse_attn && !single_stream_sparse ? protected_len : 0;
            const bool htp_enabled = ctx != nullptr && ctx->htp_attention_enabled &&
                                     sd_env_flag_enabled("SD_HTP_COMPUTE");
            const bool geometry_supported = z_image_static_halo_eligible(
                image_tokens, ctx != nullptr ? ctx->block_sparse_token_h : 0,
                ctx != nullptr ? ctx->block_sparse_token_w : 0, text_tokens);
            const bool packed_requested = sd_env_flag_enabled("SD_HTP_PACKED_Z_IMAGE") &&
                                          sd_env_flag_enabled_default("SD_HTP_PACKED_ATTENTION", true);
            auto qkv_proj_weight = qkv_proj->get_param("weight");
            auto q_norm_block    = std::dynamic_pointer_cast<RMSNorm>(blocks["q_norm"]);
            auto k_norm_block    = std::dynamic_pointer_cast<RMSNorm>(blocks["k_norm"]);
            ggml_tensor* q_gamma = q_norm_block ? q_norm_block->get_param("weight") : nullptr;
            ggml_tensor* k_gamma = k_norm_block ? k_norm_block->get_param("weight") : nullptr;
            const bool packed_shape_supported = qk_norm && num_heads == 30 && num_kv_heads == 30 && head_dim == 128;
            const bool packed_weights_available = packed_shape_supported &&
                                                  packed_z_image_qkv_branch_valid(x, qkv_proj_weight,
                                                                                   q_gamma, k_gamma);
            auto decision = z_image_select_attention_route(
                htp_enabled,
                block_sparse_attn && ctx != nullptr && ctx->block_sparse_attn_enabled,
                packed_requested,
                geometry_supported,
                packed_shape_supported,
                packed_weights_available,
                ctx != nullptr && ctx->z_image_has_ref_latents,
                ctx != nullptr && ctx->block_sparse_residual_mode == SD_HTP_BLOCK_SPARSE_RESIDUAL_ANCHOR);
            if (sd_env_flag_enabled("SD_HTP_Z_IMAGE_TRACE") ||
                sd_env_flag_enabled("SD_HTP_PACKED_Z_IMAGE_TRACE")) {
                LOG_INFO("z_image_attention route=%s fallback=%s heads=%lld kv_heads=%lld head_dim=%lld image_tokens=%d text_tokens=%d token_h=%d token_w=%d",
                         z_image_attention_route_name(decision.route), decision.fallback_reason,
                         (long long)num_heads, (long long)num_kv_heads, (long long)head_dim,
                         image_tokens, text_tokens,
                         ctx != nullptr ? ctx->block_sparse_token_h : 0,
                         ctx != nullptr ? ctx->block_sparse_token_w : 0);
            }

            // Try packed Z-Image QKV path when the route matrix selects it.
            ggml_tensor* q = nullptr;
            ggml_tensor* k = nullptr;
            ggml_tensor* v = nullptr;
            bool use_packed = false;

            if (decision.route == ZImageAttentionRoute::PACKED) {
                auto packed = make_packed_z_image_qkv(ctx, x, qkv_proj_weight, q_gamma, k_gamma,
                                                       pe, image_tokens);
                if (!packed.qkv.empty() && packed.storage != nullptr) {
                    q = packed.qkv[0];
                    k = packed.qkv[1];
                    v = packed.qkv[2];
                    use_packed = true;
                } else {
                    decision = {ZImageAttentionRoute::REORDERED, "packed_qkv_unavailable"};
                    if (sd_env_flag_enabled("SD_HTP_Z_IMAGE_TRACE") ||
                        sd_env_flag_enabled("SD_HTP_PACKED_Z_IMAGE_TRACE")) {
                        LOG_INFO("z_image_attention route=%s fallback=%s heads=%lld kv_heads=%lld head_dim=%lld image_tokens=%d text_tokens=%d token_h=%d token_w=%d",
                                 z_image_attention_route_name(decision.route), decision.fallback_reason,
                                 (long long)num_heads, (long long)num_kv_heads, (long long)head_dim,
                                 image_tokens, text_tokens,
                                 ctx != nullptr ? ctx->block_sparse_token_h : 0,
                                 ctx != nullptr ? ctx->block_sparse_token_w : 0);
                    }
                }
            }

            if (!use_packed) {
                // Standard QKV projection path
                auto qkv = qkv_proj->forward(ctx, x);                                                                            // [N, n_token, (num_heads + num_kv_heads*2)*head_dim]
                qkv      = ggml_reshape_4d(ctx->ggml_ctx, qkv, head_dim, num_heads + num_kv_heads * 2, qkv->ne[1], qkv->ne[2]);  // [N, n_token, num_heads + num_kv_heads*2, head_dim]

                q = ggml_view_4d(ctx->ggml_ctx,
                                      qkv,
                                      qkv->ne[0],
                                      num_heads,
                                      qkv->ne[2],
                                      qkv->ne[3],
                                      qkv->nb[1],
                                      qkv->nb[2],
                                      qkv->nb[3],
                                      0);  // [N, n_token, num_heads, head_dim]
                k = ggml_view_4d(ctx->ggml_ctx,
                                      qkv,
                                      qkv->ne[0],
                                      num_kv_heads,
                                      qkv->ne[2],
                                      qkv->ne[3],
                                      qkv->nb[1],
                                      qkv->nb[2],
                                      qkv->nb[3],
                                      num_heads * qkv->nb[1]);  // [N, n_token, num_kv_heads, head_dim]
                v = ggml_view_4d(ctx->ggml_ctx,
                                      qkv,
                                      qkv->ne[0],
                                      num_kv_heads,
                                      qkv->ne[2],
                                      qkv->ne[3],
                                      qkv->nb[1],
                                      qkv->nb[2],
                                      qkv->nb[3],
                                      (num_heads + num_kv_heads) * qkv->nb[1]);  // [N, n_token, num_kv_heads, head_dim]

                if (qk_norm) {
                    auto q_norm = std::dynamic_pointer_cast<RMSNorm>(blocks["q_norm"]);
                    auto k_norm = std::dynamic_pointer_cast<RMSNorm>(blocks["k_norm"]);

                    q = q_norm->forward(ctx, q);
                    k = k_norm->forward(ctx, k);
                }

                if (decision.route == ZImageAttentionRoute::REORDERED && ctx->block_sparse_attn_enabled) {
                    x = Rope::attention_ext(ctx, q, k, v, pe, mask,
                                            1.f / 128.f,              // kv_scale
                                            true,                      // rope_interleaved
                                            true,                      // block_sparse_attn
                                            protected_start,
                                            protected_len);
                } else {
                    x = Rope::attention(ctx, q, k, v, pe, mask, 1.f / 128.f);
                }
            } else {
                // Packed path: RoPE already applied in fused op, go directly to attention
                x = ggml_ext_attention_ext(ctx->ggml_ctx,
                                            ctx->backend,
                                            q, k, v,
                                            num_kv_heads,  // n_head for GQA=1
                                            mask,
                                            true,          // skip_reshape (packed views are pre-shaped)
                                            ctx->flash_attn_enabled,
                                            1.f / 128.f,   // kv_scale
                                            true,          // block_sparse_attn
                                            protected_start,
                                            protected_len,
                                            ctx->block_sparse_residual_mode,
                                            ctx->block_sparse_residual_scale,
                                            ctx->block_sparse_residual_rho,
                                            ctx->block_sparse_residual_exact_rho,
                                            nullptr, nullptr, nullptr, nullptr, nullptr,
                                            ctx->block_sparse_token_h,
                                            ctx->block_sparse_token_w,
                                            block_sparse_force_full_keep);
            }

            x = out_proj->forward(ctx, x);  // [N, n_token, hidden_size]
            return x;
        }
    };

    class FeedForward : public GGMLBlock {
    public:
        FeedForward(int64_t dim,
                    int64_t hidden_dim,
                    int64_t multiple_of,
                    float ffn_dim_multiplier = 0.f) {
            if (ffn_dim_multiplier > 0.f) {
                hidden_dim = static_cast<int64_t>(ffn_dim_multiplier * hidden_dim);
            }
            hidden_dim   = multiple_of * ((hidden_dim + multiple_of - 1) / multiple_of);
            blocks["w1"] = std::make_shared<Linear>(dim, hidden_dim, false);

            bool force_prec_f32 = false;
            float scale         = 1.f / 128.f;

            // The purpose of the scale here is to prevent NaN issues in certain situations.
            // For example, when using CUDA but the weights are k-quants.
            blocks["w2"] = std::make_shared<Linear>(hidden_dim, dim, false, false, force_prec_f32, scale);
            blocks["w3"] = std::make_shared<Linear>(dim, hidden_dim, false);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto w1 = std::dynamic_pointer_cast<Linear>(blocks["w1"]);
            auto w2 = std::dynamic_pointer_cast<Linear>(blocks["w2"]);
            auto w3 = std::dynamic_pointer_cast<Linear>(blocks["w3"]);

            if (sd_backend_is(ctx->backend, "Vulkan")) {
                w2->set_force_prec_f32(true);
            }

            auto x1 = w1->forward(ctx, x);
            auto x3 = w3->forward(ctx, x);
            x       = ggml_swiglu_split(ctx->ggml_ctx, x1, x3);
            x       = w2->forward(ctx, x);

            return x;
        }
    };

    __STATIC_INLINE__ ggml_tensor* modulate(ggml_context* ctx,
                                            ggml_tensor* x,
                                            ggml_tensor* scale) {
        // x: [N, L, C]
        // scale: [N, C]
        scale = ggml_reshape_3d(ctx, scale, scale->ne[0], 1, scale->ne[1]);  // [N, 1, C]
        x     = ggml_add(ctx, x, ggml_mul(ctx, x, scale));
        return x;
    }

    struct JointTransformerBlock : public GGMLBlock {
    protected:
        bool modulation;

    public:
        JointTransformerBlock(int layer_id,
                              int64_t hidden_size,
                              int64_t head_dim,
                              int64_t num_heads,
                              int64_t num_kv_heads,
                              int64_t multiple_of,
                              float ffn_dim_multiplier,
                              float norm_eps,
                              bool qk_norm,
                              bool modulation = true)
            : modulation(modulation) {
            blocks["attention"]       = std::make_shared<JointAttention>(hidden_size, head_dim, num_heads, num_kv_heads, qk_norm);
            blocks["feed_forward"]    = std::make_shared<FeedForward>(hidden_size, hidden_size, multiple_of, ffn_dim_multiplier);
            blocks["attention_norm1"] = std::make_shared<RMSNorm>(hidden_size, norm_eps);
            blocks["ffn_norm1"]       = std::make_shared<RMSNorm>(hidden_size, norm_eps);
            blocks["attention_norm2"] = std::make_shared<RMSNorm>(hidden_size, norm_eps);
            blocks["ffn_norm2"]       = std::make_shared<RMSNorm>(hidden_size, norm_eps);
            if (modulation) {
                blocks["adaLN_modulation.0"] = std::make_shared<Linear>(MIN(hidden_size, ADALN_EMBED_DIM), 4 * hidden_size);
            }
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* pe,
                             ggml_tensor* mask             = nullptr,
                             ggml_tensor* adaln_input      = nullptr,
                             bool block_sparse_attn        = false,
                             int protected_start           = 0,
                             int protected_len             = 0,
                             bool block_sparse_force_full_keep = false) {
            auto attention       = std::dynamic_pointer_cast<JointAttention>(blocks["attention"]);
            auto feed_forward    = std::dynamic_pointer_cast<FeedForward>(blocks["feed_forward"]);
            auto attention_norm1 = std::dynamic_pointer_cast<RMSNorm>(blocks["attention_norm1"]);
            auto ffn_norm1       = std::dynamic_pointer_cast<RMSNorm>(blocks["ffn_norm1"]);
            auto attention_norm2 = std::dynamic_pointer_cast<RMSNorm>(blocks["attention_norm2"]);
            auto ffn_norm2       = std::dynamic_pointer_cast<RMSNorm>(blocks["ffn_norm2"]);

            if (modulation) {
                GGML_ASSERT(adaln_input != nullptr);
                auto adaLN_modulation_0 = std::dynamic_pointer_cast<Linear>(blocks["adaLN_modulation.0"]);

                auto m         = adaLN_modulation_0->forward(ctx, adaln_input);  // [N, 4 * hidden_size]
                auto mods      = ggml_ext_chunk(ctx->ggml_ctx, m, 4, 0);
                auto scale_msa = mods[0];
                auto gate_msa  = mods[1];
                auto scale_mlp = mods[2];
                auto gate_mlp  = mods[3];

                auto residual = x;
                x             = modulate(ctx->ggml_ctx, attention_norm1->forward(ctx, x), scale_msa);
                x             = attention->forward(ctx, x, pe, mask, block_sparse_attn, protected_start, protected_len,
                                                   block_sparse_force_full_keep);
                x             = attention_norm2->forward(ctx, x);
                x             = ggml_mul(ctx->ggml_ctx, x, ggml_tanh(ctx->ggml_ctx, gate_msa));
                x             = ggml_add(ctx->ggml_ctx, x, residual);

                residual = x;
                x        = modulate(ctx->ggml_ctx, ffn_norm1->forward(ctx, x), scale_mlp);
                x        = feed_forward->forward(ctx, x);
                x        = ffn_norm2->forward(ctx, x);
                x        = ggml_mul(ctx->ggml_ctx, x, ggml_tanh(ctx->ggml_ctx, gate_mlp));
                x        = ggml_add(ctx->ggml_ctx, x, residual);
            } else {
                GGML_ASSERT(adaln_input == nullptr);

                auto residual = x;
                x             = attention_norm1->forward(ctx, x);
                x             = attention->forward(ctx, x, pe, mask, block_sparse_attn, protected_start, protected_len,
                                                   block_sparse_force_full_keep);
                x             = attention_norm2->forward(ctx, x);
                x             = ggml_add(ctx->ggml_ctx, x, residual);

                residual = x;
                x        = ffn_norm1->forward(ctx, x);
                x        = feed_forward->forward(ctx, x);
                x        = ffn_norm2->forward(ctx, x);
                x        = ggml_add(ctx->ggml_ctx, x, residual);
            }

            return x;
        }
    };

    struct FinalLayer : public GGMLBlock {
    public:
        FinalLayer(int64_t hidden_size,
                   int64_t patch_size,
                   int64_t out_channels) {
            blocks["norm_final"]         = std::make_shared<LayerNorm>(hidden_size, 1e-06f, false);
            blocks["linear"]             = std::make_shared<Linear>(hidden_size, patch_size * patch_size * out_channels, true, true);
            blocks["adaLN_modulation.1"] = std::make_shared<Linear>(MIN(hidden_size, ADALN_EMBED_DIM), hidden_size);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* c) {
            // x: [N, n_token, hidden_size]
            // c: [N, hidden_size]
            // return: [N, n_token, patch_size * patch_size * out_channels]
            auto norm_final         = std::dynamic_pointer_cast<LayerNorm>(blocks["norm_final"]);
            auto linear             = std::dynamic_pointer_cast<Linear>(blocks["linear"]);
            auto adaLN_modulation_1 = std::dynamic_pointer_cast<Linear>(blocks["adaLN_modulation.1"]);

            auto scale = adaLN_modulation_1->forward(ctx, ggml_silu(ctx->ggml_ctx, c));  // [N, hidden_size]
            x          = norm_final->forward(ctx, x);
            x          = modulate(ctx->ggml_ctx, x, scale);
            x          = linear->forward(ctx, x);

            return x;
        }
    };

    struct ZImageParams {
        int patch_size             = 2;
        int64_t hidden_size        = 3840;
        int64_t in_channels        = 16;
        int64_t out_channels       = 16;
        int64_t num_layers         = 30;
        int64_t num_refiner_layers = 2;
        int64_t head_dim           = 128;
        int64_t num_heads          = 30;
        int64_t num_kv_heads       = 30;
        int64_t multiple_of        = 256;
        float ffn_dim_multiplier   = 8.0f / 3.0f;
        float norm_eps             = 1e-5f;
        bool qk_norm               = true;
        int64_t cap_feat_dim       = 2560;
        int theta                  = 256;
        std::vector<int> axes_dim  = {32, 48, 48};
        int64_t axes_dim_sum       = 128;
    };

    class ZImageModel : public GGMLBlock {
    protected:
        ZImageParams z_image_params;

        void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") override {
            params["cap_pad_token"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, z_image_params.hidden_size);
            params["x_pad_token"]   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, z_image_params.hidden_size);
        }

    public:
        ZImageModel() = default;
        ZImageModel(ZImageParams z_image_params)
            : z_image_params(z_image_params) {
            blocks["x_embedder"]     = std::make_shared<Linear>(z_image_params.patch_size * z_image_params.patch_size * z_image_params.in_channels, z_image_params.hidden_size);
            blocks["t_embedder"]     = std::make_shared<TimestepEmbedder>(MIN(z_image_params.hidden_size, 1024), 256, 256);
            blocks["cap_embedder.0"] = std::make_shared<RMSNorm>(z_image_params.cap_feat_dim, z_image_params.norm_eps);
            blocks["cap_embedder.1"] = std::make_shared<Linear>(z_image_params.cap_feat_dim, z_image_params.hidden_size);

            for (int i = 0; i < z_image_params.num_refiner_layers; i++) {
                auto block = std::make_shared<JointTransformerBlock>(i,
                                                                     z_image_params.hidden_size,
                                                                     z_image_params.head_dim,
                                                                     z_image_params.num_heads,
                                                                     z_image_params.num_kv_heads,
                                                                     z_image_params.multiple_of,
                                                                     z_image_params.ffn_dim_multiplier,
                                                                     z_image_params.norm_eps,
                                                                     z_image_params.qk_norm,
                                                                     true);

                blocks["noise_refiner." + std::to_string(i)] = block;
            }

            for (int i = 0; i < z_image_params.num_refiner_layers; i++) {
                auto block = std::make_shared<JointTransformerBlock>(i,
                                                                     z_image_params.hidden_size,
                                                                     z_image_params.head_dim,
                                                                     z_image_params.num_heads,
                                                                     z_image_params.num_kv_heads,
                                                                     z_image_params.multiple_of,
                                                                     z_image_params.ffn_dim_multiplier,
                                                                     z_image_params.norm_eps,
                                                                     z_image_params.qk_norm,
                                                                     false);

                blocks["context_refiner." + std::to_string(i)] = block;
            }

            for (int i = 0; i < z_image_params.num_layers; i++) {
                auto block = std::make_shared<JointTransformerBlock>(i,
                                                                     z_image_params.hidden_size,
                                                                     z_image_params.head_dim,
                                                                     z_image_params.num_heads,
                                                                     z_image_params.num_kv_heads,
                                                                     z_image_params.multiple_of,
                                                                     z_image_params.ffn_dim_multiplier,
                                                                     z_image_params.norm_eps,
                                                                     z_image_params.qk_norm,
                                                                     true);

                blocks["layers." + std::to_string(i)] = block;
            }

            blocks["final_layer"] = std::make_shared<FinalLayer>(z_image_params.hidden_size, z_image_params.patch_size, z_image_params.out_channels);
        }

        ggml_tensor* forward_core(GGMLRunnerContext* ctx,
                                  ggml_tensor* x,
                                  ggml_tensor* timestep,
                                  ggml_tensor* context,
                                  ggml_tensor* pe,
                                  bool has_ref_latents = false) {
            auto x_embedder     = std::dynamic_pointer_cast<Linear>(blocks["x_embedder"]);
            auto t_embedder     = std::dynamic_pointer_cast<TimestepEmbedder>(blocks["t_embedder"]);
            auto cap_embedder_0 = std::dynamic_pointer_cast<RMSNorm>(blocks["cap_embedder.0"]);
            auto cap_embedder_1 = std::dynamic_pointer_cast<Linear>(blocks["cap_embedder.1"]);
            auto final_layer    = std::dynamic_pointer_cast<FinalLayer>(blocks["final_layer"]);

            auto txt_pad_token = params["cap_pad_token"];
            auto img_pad_token = params["x_pad_token"];

            int64_t N           = x->ne[2];
            int64_t n_img_token = x->ne[1];
            int64_t n_txt_token = context->ne[1];

            auto t_emb = t_embedder->forward(ctx, timestep);

            auto txt = cap_embedder_1->forward(ctx, cap_embedder_0->forward(ctx, context));  // [N, n_txt_token, hidden_size]
            auto img = x_embedder->forward(ctx, x);                                          // [N, n_img_token, hidden_size]
            sd::ggml_graph_cut::mark_graph_cut(txt, "z_image.prelude", "txt");
            sd::ggml_graph_cut::mark_graph_cut(img, "z_image.prelude", "img");
            sd::ggml_graph_cut::mark_graph_cut(t_emb, "z_image.prelude", "t_emb");

            int64_t n_txt_pad_token = Rope::bound_mod(static_cast<int>(n_txt_token), SEQ_MULTI_OF);
            if (n_txt_pad_token > 0) {
                auto txt_pad_tokens = ggml_repeat_4d(ctx->ggml_ctx, txt_pad_token, txt_pad_token->ne[0], n_txt_pad_token, N, 1);
                txt                 = ggml_concat(ctx->ggml_ctx, txt, txt_pad_tokens, 1);  // [N, n_txt_token + n_txt_pad_token, hidden_size]
            }

            int64_t n_img_pad_token = Rope::bound_mod(static_cast<int>(n_img_token), SEQ_MULTI_OF);
            if (n_img_pad_token > 0) {
                auto img_pad_tokens = ggml_repeat_4d(ctx->ggml_ctx, img_pad_token, img_pad_token->ne[0], n_img_pad_token, N, 1);
                img                 = ggml_concat(ctx->ggml_ctx, img, img_pad_tokens, 1);  // [N, n_img_token + n_img_pad_token, hidden_size]
            }

            int64_t total_img_tokens = img->ne[1];  // padded image tokens
            int64_t total_txt_tokens = txt->ne[1];  // padded text tokens

            GGML_ASSERT(total_txt_tokens + total_img_tokens == pe->ne[3]);

            // Determine sparse layout eligibility
            int token_h           = ctx->block_sparse_token_h;
            int token_w           = ctx->block_sparse_token_w;
            int image_tokens      = token_h * token_w;
            bool htp_enabled      = ctx->htp_attention_enabled && sd_env_flag_enabled("SD_HTP_COMPUTE");
            bool sparse_eligible  = htp_enabled && ctx->block_sparse_attn_enabled &&
                                    z_image_sparse_layout_eligible(image_tokens, token_h, token_w,
                                                                    static_cast<int>(n_txt_token), has_ref_latents);

            // Trace layout mode
            const bool packed_requested = sparse_eligible &&
                                          sd_env_flag_enabled("SD_HTP_PACKED_Z_IMAGE") &&
                                          sd_env_flag_enabled_default("SD_HTP_PACKED_ATTENTION", true);
            const char* layout_mode = sparse_eligible ? (packed_requested ? "packed" : "reordered") : "dense";
            LOG_INFO("z_image_layout mode=%s heads=%lld kv_heads=%lld head_dim=%lld image_tokens=%d text_tokens=%lld token_h=%d token_w=%d",
                     layout_mode,
                     (long long)z_image_params.num_heads,
                     (long long)z_image_params.num_kv_heads,
                     (long long)z_image_params.head_dim,
                     image_tokens,
                     (long long)n_txt_token,
                     token_h, token_w);

            // PE slices for refiners: adjust to PE layout (text-first or image-first)
            ggml_tensor* txt_pe;
            ggml_tensor* img_pe;
            if (sparse_eligible) {
                // PE is image-first: [img, txt]
                img_pe = ggml_ext_slice(ctx->ggml_ctx, pe, 3, 0, total_img_tokens);
                txt_pe = ggml_ext_slice(ctx->ggml_ctx, pe, 3, total_img_tokens, pe->ne[3]);
            } else {
                // PE is text-first: [txt, img]
                txt_pe = ggml_ext_slice(ctx->ggml_ctx, pe, 3, 0, total_txt_tokens);
                img_pe = ggml_ext_slice(ctx->ggml_ctx, pe, 3, total_txt_tokens, pe->ne[3]);
            }

            // Context refiner is text-only full attention. Keep it on the dense
            // HTP route; there is no block pruning to exploit here.
            const bool single_stream_htp = ctx->htp_attention_enabled &&
                                            ctx->block_sparse_attn_enabled &&
                                            sd_env_flag_enabled("SD_HTP_COMPUTE");
            for (int i = 0; i < z_image_params.num_refiner_layers; i++) {
                auto block = std::dynamic_pointer_cast<JointTransformerBlock>(blocks["context_refiner." + std::to_string(i)]);

                txt = block->forward(ctx, txt, txt_pe, nullptr, nullptr,
                                     false, 0, 0, false);
                sd::ggml_graph_cut::mark_graph_cut(txt, "z_image.context_refiner." + std::to_string(i), "txt");
            }

            // Noise refiner: image-only, using the real image grid and halo.
            const bool noise_stream_sparse = single_stream_htp &&
                                              z_image_static_halo_eligible(
                                                  static_cast<int>(total_img_tokens),
                                                  token_h, token_w, 0);
            for (int i = 0; i < z_image_params.num_refiner_layers; i++) {
                auto block = std::dynamic_pointer_cast<JointTransformerBlock>(blocks["noise_refiner." + std::to_string(i)]);

                img = block->forward(ctx, img, img_pe, nullptr, t_emb,
                                     noise_stream_sparse, 0, 0, false);
                sd::ggml_graph_cut::mark_graph_cut(img, "z_image.noise_refiner." + std::to_string(i), "img");
            }

            ggml_tensor* txt_img;
            ggml_tensor* joint_pe;
            int64_t img_slice_start;
            int64_t img_slice_len;

            if (sparse_eligible) {
                // Sparse layout: image-first with patch-major reorder
                int patch = 4;
                sd_patch_major_reorder_params* reorder_params = new sd_patch_major_reorder_params();
                reorder_params->token_h       = token_h;
                reorder_params->token_w       = token_w;
                reorder_params->patch         = patch;
                reorder_params->inverse       = 0;
                reorder_params->shift_y       = 0;
                reorder_params->shift_x       = 0;
                reorder_params->src_shift_y   = 0;
                reorder_params->src_shift_x   = 0;
                reorder_params->src_patch_major = 0;

                ggml_tensor* args_reorder[1] = { img };
                ggml_tensor* img_reordered = ggml_custom_4d(ctx->ggml_ctx, GGML_TYPE_F32,
                                                             img->ne[0], total_img_tokens, N, 1,
                                                             args_reorder, 1,
                                                             sd_patch_major_reorder_custom,
                                                             GGML_N_TASKS_MAX,
                                                             (void*)reorder_params);
                ggml_set_name(img_reordered, "z_image_img_reordered");
                sd::ggml_graph_cut::mark_graph_cut(img_reordered, "z_image.reorder", "img_reordered");

                // Image-first concatenation: [img_reordered, txt]
                txt_img = ggml_concat(ctx->ggml_ctx, img_reordered, txt, 1);
                joint_pe = pe;  // PE is image-first for sparse layout
                img_slice_start = 0;
                img_slice_len   = total_img_tokens;
            } else {
                // Dense layout: text-first with row-major image tokens (current behavior)
                txt_img = ggml_concat(ctx->ggml_ctx, txt, img, 1);
                joint_pe = pe;
                img_slice_start = total_txt_tokens;
                img_slice_len   = total_img_tokens;
            }
            sd::ggml_graph_cut::mark_graph_cut(txt_img, "z_image.prelude", "txt_img");

            // Joint layers: sparse uses image-first with protected text suffix
            int block_sparse_protected_start = sparse_eligible ? static_cast<int>(total_img_tokens) : 0;
            int block_sparse_protected_len   = sparse_eligible ? static_cast<int>(total_txt_tokens) : 0;

            for (int i = 0; i < z_image_params.num_layers; i++) {
                auto block = std::dynamic_pointer_cast<JointTransformerBlock>(blocks["layers." + std::to_string(i)]);

                txt_img = block->forward(ctx, txt_img, joint_pe, nullptr, t_emb,
                                         /*block_sparse_attn=*/ctx->block_sparse_attn_enabled,
                                         block_sparse_protected_start,
                                         block_sparse_protected_len);
                sd::ggml_graph_cut::mark_graph_cut(txt_img, "z_image.layers." + std::to_string(i), "txt_img");
            }

            txt_img = final_layer->forward(ctx, txt_img, t_emb);  // [N, seq_len, ph*pw*C]

            // Slice out the image region
            ggml_tensor* img_out = ggml_ext_slice(ctx->ggml_ctx, txt_img, 1, img_slice_start, img_slice_start + img_slice_len);

            if (sparse_eligible) {
                // Inverse-reorder image tokens from patch-major back to row-major
                int patch = 4;
                sd_patch_major_reorder_params* inv_params = new sd_patch_major_reorder_params();
                inv_params->token_h        = token_h;
                inv_params->token_w        = token_w;
                inv_params->patch          = patch;
                inv_params->inverse        = 1;
                inv_params->shift_y        = 0;
                inv_params->shift_x        = 0;
                inv_params->src_shift_y    = 0;
                inv_params->src_shift_x    = 0;
                inv_params->src_patch_major = 0;

                ggml_tensor* args_inv[1] = { img_out };
                img_out = ggml_custom_4d(ctx->ggml_ctx, GGML_TYPE_F32,
                                         img_out->ne[0], img_slice_len, N, 1,
                                         args_inv, 1,
                                         sd_patch_major_reorder_custom,
                                         GGML_N_TASKS_MAX,
                                         (void*)inv_params);
                ggml_set_name(img_out, "z_image_img_inverse_reordered");
                sd::ggml_graph_cut::mark_graph_cut(img_out, "z_image.inverse_reorder", "img_inv");
            }

            // Retain only the original unpadded image tokens
            img_out = ggml_ext_slice(ctx->ggml_ctx, img_out, 1, 0, n_img_token);  // [N, n_img_token, ph*pw*C]

            return img_out;
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* timestep,
                             ggml_tensor* context,
                             ggml_tensor* pe,
                             std::vector<ggml_tensor*> ref_latents = {},
                             bool increase_ref_index = false) {
            // Forward pass of DiT.
            // x: [N, C, H, W]
            // timestep: [N,]
            // context: [N, L, D]
            // pe: [L, d_head/2, 2, 2]
            // return: [N, C, H, W]

            int64_t W = x->ne[0];
            int64_t H = x->ne[1];
            int64_t C = x->ne[2];
            int64_t N = x->ne[3];

            int patch_size = z_image_params.patch_size;

            auto img             = DiT::pad_and_patchify(ctx, x, patch_size, patch_size, false);
            uint64_t n_img_token = img->ne[1];

            if (ref_latents.size() > 0) {
                for (ggml_tensor* ref : ref_latents) {
                    ref = DiT::pad_and_patchify(ctx, ref, patch_size, patch_size, false);
                    img = ggml_concat(ctx->ggml_ctx, img, ref, 1);
                }
            }

            auto out = forward_core(ctx, img, timestep, context, pe,
                                    !ref_latents.empty() || increase_ref_index);

            out = ggml_ext_slice(ctx->ggml_ctx, out, 1, 0, n_img_token);                              // [N, n_img_token, ph*pw*C]
            out = DiT::unpatchify_and_crop(ctx->ggml_ctx, out, H, W, patch_size, patch_size, false);  // [N, C, H, W]

            out = ggml_ext_scale(ctx->ggml_ctx, out, -1.f);

            return out;
        }
    };

    struct ZImageRunner : public GGMLRunner {
    public:
        ZImageParams z_image_params;
        ZImageModel z_image;
        std::vector<float> pe_vec;
        std::vector<float> timestep_vec;
        SDVersion version;

        ZImageRunner(ggml_backend_t backend,
                     ggml_backend_t params_backend,
                     const String2TensorStorage& tensor_storage_map = {},
                     const std::string prefix                       = "",
                     SDVersion version                              = VERSION_Z_IMAGE)
            : GGMLRunner(backend, params_backend) {
            z_image = ZImageModel(z_image_params);
            z_image.init(params_ctx, tensor_storage_map, prefix);
        }

        std::string get_desc() override {
            return "z_image";
        }

        void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors, const std::string prefix) {
            z_image.get_param_tensors(tensors, prefix);
        }

        ggml_cgraph* build_graph(const sd::Tensor<float>& x_tensor,
                                 const sd::Tensor<float>& timesteps_tensor,
                                 const sd::Tensor<float>& context_tensor,
                                 const std::vector<sd::Tensor<float>>& ref_latents_tensor = {},
                                 bool increase_ref_index                                  = false,
                                 int block_sparse_step_index                              = 0,
                                 int block_sparse_residual_mode                           = SD_HTP_BLOCK_SPARSE_RESIDUAL_OFF,
                                 float block_sparse_residual_scale                        = 1.0f,
                                 float block_sparse_residual_rho                          = 1.0f,
                                 bool block_sparse_residual_exact_rho                     = false,
                                 int block_sparse_residual_branch_index                   = 0) {
            ggml_cgraph* gf        = new_graph_custom(Z_IMAGE_GRAPH_SIZE);
            ggml_tensor* x         = make_input(x_tensor);
            ggml_tensor* timesteps = make_input(timesteps_tensor);
            GGML_ASSERT(x->ne[3] == 1);
            GGML_ASSERT(!context_tensor.empty());
            ggml_tensor* context = make_input(context_tensor);
            std::vector<ggml_tensor*> ref_latents;
            ref_latents.reserve(ref_latents_tensor.size());
            for (const auto& ref_latent_tensor : ref_latents_tensor) {
                ref_latents.push_back(make_input(ref_latent_tensor));
            }

            // Compute image geometry for token grid
            int64_t W = x->ne[0];
            int64_t H = x->ne[1];
            int patch_size = z_image_params.patch_size;
            int pad_h      = (patch_size - (H % patch_size)) % patch_size;
            int pad_w      = (patch_size - (W % patch_size)) % patch_size;
            int token_h    = (H + pad_h) / patch_size;
            int token_w    = (W + pad_w) / patch_size;
            int image_tokens = token_h * token_w;

            // Determine sparse eligibility for PE generation
            bool sparse_pe_image_first = is_block_sparse_attention_enabled() &&
                                         z_image_sparse_layout_eligible(image_tokens, token_h, token_w,
                                                                         static_cast<int>(context->ne[1]),
                                                                         ref_latents.size() > 0 || increase_ref_index);

            pe_vec      = Rope::gen_z_image_pe(static_cast<int>(x->ne[1]),
                                               static_cast<int>(x->ne[0]),
                                               z_image_params.patch_size,
                                               static_cast<int>(x->ne[3]),
                                               static_cast<int>(context->ne[1]),
                                               SEQ_MULTI_OF,
                                               ref_latents,
                                               increase_ref_index,
                                               z_image_params.theta,
                                               circular_y_enabled,
                                               circular_x_enabled,
                                               z_image_params.axes_dim,
                                               sparse_pe_image_first);
            int pos_len = static_cast<int>(pe_vec.size() / z_image_params.axes_dim_sum / 2);
            auto pe = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32, 2, 2, z_image_params.axes_dim_sum / 2, pos_len);
            set_backend_tensor_data(pe, pe_vec.data());
            auto runner_ctx = get_context();

            runner_ctx.block_sparse_residual_mode            = block_sparse_residual_mode;
            runner_ctx.block_sparse_residual_scale           = block_sparse_residual_scale;
            runner_ctx.block_sparse_residual_rho             = block_sparse_residual_rho;
            runner_ctx.block_sparse_residual_exact_rho       = block_sparse_residual_exact_rho;
            runner_ctx.block_sparse_residual_branch_index    = block_sparse_residual_branch_index;
            runner_ctx.block_sparse_residual_attention_index = 0;
            runner_ctx.z_image_has_ref_latents                = !ref_latents.empty() || increase_ref_index;

            // Set and restore image geometry for block-sparse attention
            const int previous_token_h = runner_ctx.block_sparse_token_h;
            const int previous_token_w = runner_ctx.block_sparse_token_w;
            runner_ctx.block_sparse_token_h = token_h;
            runner_ctx.block_sparse_token_w = token_w;

            ggml_tensor* out = z_image.forward(&runner_ctx,
                                               x,
                                               timesteps,
                                               context,
                                               pe,
                                               ref_latents,
                                               increase_ref_index);

            runner_ctx.block_sparse_token_h = previous_token_h;
            runner_ctx.block_sparse_token_w = previous_token_w;

            ggml_build_forward_expand(gf, out);

            return gf;
        }

        sd::Tensor<float> compute(int n_threads,
                                  const sd::Tensor<float>& x,
                                  const sd::Tensor<float>& timesteps,
                                  const sd::Tensor<float>& context,
                                  const std::vector<sd::Tensor<float>>& ref_latents = {},
                                  bool increase_ref_index                           = false,
                                  int block_sparse_step_index                       = 0,
                                  int block_sparse_residual_mode                    = SD_HTP_BLOCK_SPARSE_RESIDUAL_OFF,
                                  float block_sparse_residual_scale                 = 1.0f,
                                  float block_sparse_residual_rho                   = 1.0f,
                                  bool block_sparse_residual_exact_rho              = false,
                                  int block_sparse_residual_branch_index            = 0) {
            // x: [N, in_channels, h, w]
            // timesteps: [N, ]
            // context: [N, max_position, hidden_size]
            auto get_graph = [&]() -> ggml_cgraph* {
                return build_graph(x, timesteps, context, ref_latents, increase_ref_index,
                                   block_sparse_step_index,
                                   block_sparse_residual_mode,
                                   block_sparse_residual_scale,
                                   block_sparse_residual_rho,
                                   block_sparse_residual_exact_rho,
                                   block_sparse_residual_branch_index);
            };

            return restore_trailing_singleton_dims(GGMLRunner::compute<float>(get_graph, n_threads, false), x.dim());
        }

        void test() {
            ggml_init_params params;
            params.mem_size   = static_cast<size_t>(1024 * 1024) * 1024;  // 1GB
            params.mem_buffer = nullptr;
            params.no_alloc   = false;

            ggml_context* ctx = ggml_init(params);
            GGML_ASSERT(ctx != nullptr);

            {
                // auto x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 16, 16, 16, 1);
                // ggml_set_f32(x, 0.01f);
                auto x = sd::load_tensor_from_file_as_tensor<float>("./z_image_x.bin");
                print_sd_tensor(x);

                std::vector<float> timesteps_vec(1, 0.f);
                auto timesteps = sd::Tensor<float>::from_vector(timesteps_vec);

                // auto context = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2560, 256, 1);
                // ggml_set_f32(context, 0.01f);
                auto context = sd::load_tensor_from_file_as_tensor<float>("./z_image_context.bin");
                print_sd_tensor(context);

                sd::Tensor<float> out;

                int64_t t0   = ggml_time_ms();
                auto out_opt = compute(8,
                                       x,
                                       timesteps,
                                       context,
                                       {},
                                       false);
                int64_t t1   = ggml_time_ms();

                GGML_ASSERT(!out_opt.empty());
                out = std::move(out_opt);
                print_sd_tensor(out);
                LOG_DEBUG("z_image test done in %lldms", t1 - t0);
            }
        }

        static void load_from_file_and_test(const std::string& file_path) {
            // cuda q8: pass
            // cuda q8 fa: pass
            // ggml_backend_t backend = ggml_backend_cuda_init(0);
            ggml_backend_t backend    = ggml_backend_cpu_init();
            ggml_type model_data_type = GGML_TYPE_Q8_0;

            ModelLoader model_loader;
            if (!model_loader.init_from_file_and_convert_name(file_path, "model.diffusion_model.")) {
                LOG_ERROR("init model loader from file failed: '%s'", file_path.c_str());
                return;
            }

            auto& tensor_storage_map = model_loader.get_tensor_storage_map();
            if (model_data_type != GGML_TYPE_COUNT) {
                for (auto& [name, tensor_storage] : tensor_storage_map) {
                    if (ends_with(name, "weight")) {
                        tensor_storage.expected_type = model_data_type;
                    }
                }
            }

            std::shared_ptr<ZImageRunner> z_image = std::make_shared<ZImageRunner>(backend,
                                                                                   backend,
                                                                                   tensor_storage_map,
                                                                                   "model.diffusion_model",
                                                                                   VERSION_QWEN_IMAGE);

            z_image->alloc_params_buffer();
            std::map<std::string, ggml_tensor*> tensors;
            z_image->get_param_tensors(tensors, "model.diffusion_model");

            bool success = model_loader.load_tensors(tensors);

            if (!success) {
                LOG_ERROR("load tensors from model loader failed");
                return;
            }

            LOG_INFO("z_image model loaded");
            z_image->test();
        }
    };

}  // namespace ZImage

#endif  // __Z_IMAGE_HPP__
