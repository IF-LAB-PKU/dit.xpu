#ifndef __FLUX_HPP__
#define __FLUX_HPP__

#include <memory>
#include <vector>

#include "common_dit.hpp"
#include "model.h"
#include "rope.hpp"

#define FLUX_GRAPH_SIZE 10240

namespace Flux {

    struct MLPEmbedder : public UnaryBlock {
    public:
        MLPEmbedder(int64_t in_dim, int64_t hidden_dim, bool bias = true) {
            blocks["in_layer"]  = std::shared_ptr<GGMLBlock>(new Linear(in_dim, hidden_dim, bias));
            blocks["out_layer"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_dim, hidden_dim, bias));
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
            // x: [..., in_dim]
            // return: [..., hidden_dim]
            auto in_layer  = std::dynamic_pointer_cast<Linear>(blocks["in_layer"]);
            auto out_layer = std::dynamic_pointer_cast<Linear>(blocks["out_layer"]);

            x = in_layer->forward(ctx, x);
            x = ggml_silu_inplace(ctx->ggml_ctx, x);
            x = out_layer->forward(ctx, x);
            return x;
        }
    };

    class RMSNorm : public UnaryBlock {
    protected:
        int64_t hidden_size;
        float eps;

        void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") override {
            ggml_type wtype = GGML_TYPE_F32;
            params["scale"] = ggml_new_tensor_1d(ctx, wtype, hidden_size);
        }

    public:
        RMSNorm(int64_t hidden_size,
                float eps = 1e-06f)
            : hidden_size(hidden_size),
              eps(eps) {}

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
            ggml_tensor* w = params["scale"];
            x              = ggml_rms_norm(ctx->ggml_ctx, x, eps);
            x              = ggml_mul(ctx->ggml_ctx, x, w);
            return x;
        }
    };

    struct QKNorm : public GGMLBlock {
    public:
        QKNorm(int64_t dim) {
            blocks["query_norm"] = std::shared_ptr<GGMLBlock>(new RMSNorm(dim));
            blocks["key_norm"]   = std::shared_ptr<GGMLBlock>(new RMSNorm(dim));
        }

        ggml_tensor* query_norm(GGMLRunnerContext* ctx, ggml_tensor* x) {
            // x: [..., dim]
            // return: [..., dim]
            auto norm = std::dynamic_pointer_cast<RMSNorm>(blocks["query_norm"]);

            x = norm->forward(ctx, x);
            return x;
        }

        ggml_tensor* key_norm(GGMLRunnerContext* ctx, ggml_tensor* x) {
            // x: [..., dim]
            // return: [..., dim]
            auto norm = std::dynamic_pointer_cast<RMSNorm>(blocks["key_norm"]);

            x = norm->forward(ctx, x);
            return x;
        }

        ggml_tensor* param(const std::string& name) const {
            const char* block_name = name == "q_norm" ? "query_norm" :
                                     (name == "k_norm" ? "key_norm" : nullptr);
            if (block_name == nullptr) {
                return nullptr;
            }
            auto norm = std::dynamic_pointer_cast<RMSNorm>(blocks.at(block_name));
            return norm->get_param("scale");
        }
    };

    struct SelfAttention : public GGMLBlock {
    public:
        int64_t num_heads;

    public:
        SelfAttention(int64_t dim,
                      int64_t num_heads = 8,
                      bool qkv_bias     = false,
                      bool proj_bias    = true)
            : num_heads(num_heads) {
            int64_t head_dim = dim / num_heads;
            blocks["qkv"]    = std::shared_ptr<GGMLBlock>(new Linear(dim, dim * 3, qkv_bias));
            blocks["norm"]   = std::shared_ptr<GGMLBlock>(new QKNorm(head_dim));
            blocks["proj"]   = std::shared_ptr<GGMLBlock>(new Linear(dim, dim, proj_bias));
        }

        std::vector<ggml_tensor*> pre_attention(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto qkv_proj = std::dynamic_pointer_cast<Linear>(blocks["qkv"]);
            auto norm     = std::dynamic_pointer_cast<QKNorm>(blocks["norm"]);

            auto qkv         = qkv_proj->forward(ctx, x);
            int64_t head_dim = qkv->ne[0] / 3 / num_heads;
            auto q           = ggml_view_4d(ctx->ggml_ctx, qkv, head_dim, num_heads, qkv->ne[1], qkv->ne[2],
                                            qkv->nb[0] * head_dim, qkv->nb[1], qkv->nb[2], 0);
            auto k           = ggml_view_4d(ctx->ggml_ctx, qkv, head_dim, num_heads, qkv->ne[1], qkv->ne[2],
                                            qkv->nb[0] * head_dim, qkv->nb[1], qkv->nb[2], (qkv->nb[0]) * qkv->ne[0] / 3);
            auto v           = ggml_view_4d(ctx->ggml_ctx, qkv, head_dim, num_heads, qkv->ne[1], qkv->ne[2],
                                            qkv->nb[0] * head_dim, qkv->nb[1], qkv->nb[2], (qkv->nb[0]) * 2 * qkv->ne[0] / 3);
            q                = norm->query_norm(ctx, q);
            k                = norm->key_norm(ctx, k);
            return {q, k, v};
        }

        ggml_tensor* param(const std::string& name) const {
            if (name == "weight" || name == "bias") {
                auto qkv = std::dynamic_pointer_cast<Linear>(blocks.at("qkv"));
                return qkv->get_param(name);
            }
            auto norm = std::dynamic_pointer_cast<QKNorm>(blocks.at("norm"));
            if (name == "q_norm") {
                return norm->param(name);
            }
            if (name == "k_norm") {
                return norm->param(name);
            }
            return nullptr;
        }

        ggml_tensor* post_attention(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto proj = std::dynamic_pointer_cast<Linear>(blocks["proj"]);

            x = proj->forward(ctx, x);  // [N, n_token, dim]
            return x;
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* pe,
                             ggml_tensor* mask) {
            // x: [N, n_token, dim]
            // pe: [n_token, d_head/2, 2, 2]
            // return [N, n_token, dim]
            auto qkv = pre_attention(ctx, x);                                   // q,k,v: [N, n_token, n_head, d_head]
            x        = Rope::attention(ctx, qkv[0], qkv[1], qkv[2], pe, mask);  // [N, n_token, dim]
            x        = post_attention(ctx, x);                                  // [N, n_token, dim]
            return x;
        }
    };

    struct PackedRopeQkv {
        std::vector<ggml_tensor*> qkv;
        ggml_tensor* storage = nullptr;
    };

    // Flux2 anchor attention still needs the packed position encoding as the
    // marker that selects the residual-attention path. The anchor deliberately
    // keeps ordinary row-major QKV, so packed QKV itself remains disabled.
    __STATIC_INLINE__ bool flux2_sparse_pe_required(bool is_flux2,
                                                     bool sparse_graph,
                                                     bool residual_anchor,
                                                     bool packed_qkv_requested,
                                                     bool has_refs,
                                                     bool circular) {
        return is_flux2 && sparse_graph && !has_refs && !circular &&
               (residual_anchor || packed_qkv_requested);
    }

    __STATIC_INLINE__ bool packed_flux2_context_enabled(const GGMLRunnerContext* ctx,
                                                         int image_tokens,
                                                         int text_tokens,
                                                         ggml_tensor* pe) {
        return ctx != nullptr && sd_env_flag_enabled("SD_HTP_PACKED_FLUX2") &&
               sd_env_flag_enabled_default("SD_HTP_PACKED_ATTENTION", true) &&
               sd_env_flag_enabled("SD_HTP_COMPUTE") && ctx->block_sparse_attn_enabled &&
               ctx->block_sparse_residual_mode != SD_HTP_BLOCK_SPARSE_RESIDUAL_ANCHOR &&
               image_tokens > 0 && image_tokens % 32 == 0 && text_tokens > 0 &&
               ctx->block_sparse_token_h > 0 && ctx->block_sparse_token_w > 0 &&
               ctx->block_sparse_token_h % 4 == 0 && ctx->block_sparse_token_w % 8 == 0 &&
               ctx->block_sparse_token_h * ctx->block_sparse_token_w == image_tokens &&
               pe != nullptr && pe->type == GGML_TYPE_F32 && pe->ne[0] == 2 && pe->ne[1] == 2 &&
               pe->ne[2] == 64 && pe->ne[3] >= image_tokens + text_tokens;
    }

    __STATIC_INLINE__ bool packed_flux2_qkv_branch_valid(ggml_tensor* activation,
                                                          ggml_tensor* weight,
                                                          ggml_tensor* bias,
                                                          ggml_tensor* q_norm,
                                                          ggml_tensor* k_norm) {
        constexpr int n_heads = 24;
        constexpr int head_dim = 128;
        constexpr int qkv_dim = 3 * n_heads * head_dim;
        return activation != nullptr && activation->type == GGML_TYPE_F32 && activation->ne[2] == 1 &&
               weight != nullptr && weight->type == GGML_TYPE_F16 && weight->ne[0] == activation->ne[0] &&
               weight->ne[1] == qkv_dim &&
               (weight->flags & SD_GGML_TENSOR_FLAG_HTP_PACKED_W16A32) != 0 &&
               (bias == nullptr || (bias->type == GGML_TYPE_F32 && bias->ne[0] == qkv_dim)) &&
               q_norm != nullptr && q_norm->type == GGML_TYPE_F32 && q_norm->ne[0] == head_dim &&
               k_norm != nullptr && k_norm->type == GGML_TYPE_F32 && k_norm->ne[0] == head_dim;
    }

    __STATIC_INLINE__ PackedRopeQkv make_packed_flux2_qkv_views(GGMLRunnerContext* ctx,
                                                                 ggml_tensor* storage,
                                                                 int tokens) {
        PackedRopeQkv result;
        result.storage = storage;
        constexpr int n_heads = 24;
        constexpr int head_dim = 128;
        const int token_tiles = (tokens + 31) / 32;
        const size_t qkv_bytes = size_t(n_heads) * size_t(token_tiles) * size_t(head_dim / 32) *
                                 size_t(1024) * sizeof(ggml_fp16_t);
        const size_t head_stride = size_t(token_tiles) * size_t(head_dim / 32) *
                                   size_t(1024) * sizeof(ggml_fp16_t);
        result.qkv = {
            ggml_view_4d(ctx->ggml_ctx, storage, head_dim, tokens, n_heads, 1,
                         sizeof(ggml_fp16_t), head_dim * sizeof(ggml_fp16_t), head_stride, 0),
            ggml_view_4d(ctx->ggml_ctx, storage, head_dim, tokens, n_heads, 1,
                         sizeof(ggml_fp16_t), head_dim * sizeof(ggml_fp16_t), head_stride, qkv_bytes),
            ggml_view_4d(ctx->ggml_ctx, storage, head_dim, tokens, n_heads, 1,
                         sizeof(ggml_fp16_t), head_dim * sizeof(ggml_fp16_t), head_stride, 2 * qkv_bytes),
        };
        // Keep these tensors as real graph views.  Marking them as NONE turns
        // them into leaves, so gallocr misses the corresponding view counts
        // during its graph prepass and never releases the backing storage.
        return result;
    }

    __STATIC_INLINE__ PackedRopeQkv make_packed_flux2_dual_qkv(
        GGMLRunnerContext* ctx,
        const std::shared_ptr<SelfAttention>& image_attn,
        const std::shared_ptr<SelfAttention>& text_attn,
        ggml_tensor* image_activation,
        ggml_tensor* text_activation,
        ggml_tensor* pe) {
        PackedRopeQkv result;
        if (!sd_env_flag_enabled_default("SD_HTP_PACKED_FLUX2_DOUBLE", true)) {
            return result;
        }
        const int image_tokens = image_activation != nullptr ? static_cast<int>(image_activation->ne[1]) : 0;
        const int text_tokens = text_activation != nullptr ? static_cast<int>(text_activation->ne[1]) : 0;
        const bool context_ok = packed_flux2_context_enabled(ctx, image_tokens, text_tokens, pe);
        const bool heads_ok = image_attn != nullptr && text_attn != nullptr &&
                              image_attn->num_heads == 24 && text_attn->num_heads == 24;
        if (!context_ok || !heads_ok) {
            if (sd_env_flag_enabled("SD_HTP_PACKED_FLUX2_TRACE")) {
                LOG_INFO("flux2_packed_qkv skip=dual_context context_ok=%d heads_ok=%d image=%d text=%d grid=%dx%d sparse=%d residual=%d pe=[%lld,%lld,%lld,%lld]",
                         context_ok ? 1 : 0, heads_ok ? 1 : 0, image_tokens, text_tokens,
                         ctx != nullptr ? ctx->block_sparse_token_h : 0,
                         ctx != nullptr ? ctx->block_sparse_token_w : 0,
                         ctx != nullptr && ctx->block_sparse_attn_enabled ? 1 : 0,
                         ctx != nullptr ? ctx->block_sparse_residual_mode : -1,
                         pe != nullptr ? (long long) pe->ne[0] : 0,
                         pe != nullptr ? (long long) pe->ne[1] : 0,
                         pe != nullptr ? (long long) pe->ne[2] : 0,
                         pe != nullptr ? (long long) pe->ne[3] : 0);
            }
            return result;
        }
        ggml_tensor* image_weight = image_attn->param("weight");
        ggml_tensor* image_bias = image_attn->param("bias");
        ggml_tensor* image_q_norm = image_attn->param("q_norm");
        ggml_tensor* image_k_norm = image_attn->param("k_norm");
        ggml_tensor* text_weight = text_attn->param("weight");
        ggml_tensor* text_bias = text_attn->param("bias");
        ggml_tensor* text_q_norm = text_attn->param("q_norm");
        ggml_tensor* text_k_norm = text_attn->param("k_norm");
        if (!packed_flux2_qkv_branch_valid(image_activation, image_weight, image_bias, image_q_norm, image_k_norm) ||
            !packed_flux2_qkv_branch_valid(text_activation, text_weight, text_bias, text_q_norm, text_k_norm)) {
            if (sd_env_flag_enabled("SD_HTP_PACKED_FLUX2_TRACE")) {
                LOG_INFO("flux2_packed_qkv skip=dual_params image_weight=[%lld,%lld] type=%s flags=%u text_weight=[%lld,%lld] type=%s flags=%u",
                         image_weight != nullptr ? (long long) image_weight->ne[0] : 0,
                         image_weight != nullptr ? (long long) image_weight->ne[1] : 0,
                         image_weight != nullptr ? ggml_type_name(image_weight->type) : "null",
                         image_weight != nullptr ? image_weight->flags : 0,
                         text_weight != nullptr ? (long long) text_weight->ne[0] : 0,
                         text_weight != nullptr ? (long long) text_weight->ne[1] : 0,
                         text_weight != nullptr ? ggml_type_name(text_weight->type) : "null",
                         text_weight != nullptr ? text_weight->flags : 0);
            }
            return result;
        }
        const int tokens = image_tokens + text_tokens;
        const int token_tiles = (tokens + 31) / 32;
        const size_t region_bytes = size_t(24) * size_t(token_tiles) * 4 * size_t(1024) * sizeof(ggml_fp16_t);
        ggml_tensor* args[] = {image_activation, image_weight, image_bias, image_q_norm, image_k_norm,
                               text_activation, text_weight, text_bias, text_q_norm, text_k_norm, pe};
        ggml_tensor* storage = ggml_custom_4d(ctx->ggml_ctx, GGML_TYPE_F16,
                                              static_cast<int64_t>(3 * region_bytes / sizeof(ggml_fp16_t)),
                                              1, 1, 1, args, 11,
                                              [](ggml_tensor*, int, int, void*) {}, 1, nullptr);
        ggml_set_name(storage, "qkv_mul_rms_packed_dual_rope");
        if (sd_env_flag_enabled("SD_HTP_PACKED_FLUX2_TRACE")) {
            LOG_INFO("flux2_packed_qkv kind=dual image_tokens=%d text_tokens=%d head_dim=128", image_tokens, text_tokens);
        }
        return make_packed_flux2_qkv_views(ctx, storage, tokens);
    }

    __STATIC_INLINE__ PackedRopeQkv make_packed_flux2_single_qkv(
        GGMLRunnerContext* ctx,
        ggml_tensor* activation,
        ggml_tensor* weight,
        ggml_tensor* bias,
        ggml_tensor* q_norm,
        ggml_tensor* k_norm,
        ggml_tensor* pe,
        int image_tokens) {
        PackedRopeQkv result;
        if (!sd_env_flag_enabled_default("SD_HTP_PACKED_FLUX2_SINGLE", true)) {
            return result;
        }
        const int tokens = activation != nullptr ? static_cast<int>(activation->ne[1]) : 0;
        const int text_tokens = tokens - image_tokens;
        if (!packed_flux2_context_enabled(ctx, image_tokens, text_tokens, pe) ||
            !packed_flux2_qkv_branch_valid(activation, weight, bias, q_norm, k_norm)) {
            return result;
        }
        const int token_tiles = (tokens + 31) / 32;
        const size_t region_bytes = size_t(24) * size_t(token_tiles) * 4 * size_t(1024) * sizeof(ggml_fp16_t);
        ggml_tensor* args[] = {activation, weight, bias, q_norm, k_norm, pe};
        ggml_tensor* storage = ggml_custom_4d(ctx->ggml_ctx, GGML_TYPE_F16,
                                              static_cast<int64_t>(3 * region_bytes / sizeof(ggml_fp16_t)),
                                              1, 1, 1, args, 6,
                                              [](ggml_tensor*, int, int, void*) {}, 1, nullptr);
        ggml_set_name(storage, "qkv_mul_rms_packed_rope");
        if (sd_env_flag_enabled("SD_HTP_PACKED_FLUX2_TRACE")) {
            LOG_INFO("flux2_packed_qkv kind=single image_tokens=%d text_tokens=%d head_dim=128", image_tokens, text_tokens);
        }
        return make_packed_flux2_qkv_views(ctx, storage, tokens);
    }

    __STATIC_INLINE__ ggml_tensor* flux2_residual_attention(GGMLRunnerContext* ctx,
                                                             ggml_tensor* q,
                                                             ggml_tensor* k,
                                                             ggml_tensor* v,
                                                             int64_t num_heads,
                                                             ggml_tensor* mask,
                                                             bool skip_reshape,
                                                             float kv_scale,
                                                             bool block_sparse_attn,
                                                             int image_tokens,
                                                             int text_tokens,
                                                             const char* residual_kind) {
        const auto binding = sd_htp_prepare_residual_binding(ctx, residual_kind, true);

        return ggml_ext_attention_ext(ctx->ggml_ctx,
                                      ctx->backend,
                                      q,
                                      k,
                                      v,
                                      num_heads,
                                      mask,
                                      skip_reshape,
                                      ctx->flash_attn_enabled,
                                      kv_scale,
                                      block_sparse_attn ||
                                          ctx->block_sparse_residual_mode == SD_HTP_BLOCK_SPARSE_RESIDUAL_ANCHOR,
                                      image_tokens,
                                      text_tokens,
                                      ctx->block_sparse_residual_mode,
                                      ctx->block_sparse_residual_scale,
                                      ctx->block_sparse_residual_rho,
                                      ctx->block_sparse_residual_exact_rho,
                                      binding.residual_key.empty() ? nullptr : binding.residual_key.c_str(),
                                      binding.cached_residual,
                                      binding.rho_key.empty() ? nullptr : binding.rho_key.c_str(),
                                      binding.cached_rho,
                                      &ctx->cache_tensor,
                                      ctx->block_sparse_token_h,
                                      ctx->block_sparse_token_w);
    }

    __STATIC_INLINE__ ggml_tensor* flux2_rope_residual_attention(
        GGMLRunnerContext* ctx,
        ggml_tensor* q,
        ggml_tensor* k,
        ggml_tensor* v,
        ggml_tensor* pe,
        ggml_tensor* mask,
        int image_tokens,
        int text_tokens,
        const char* residual_kind) {
        q = Rope::apply_rope(ctx->ggml_ctx, q, pe);
        k = Rope::apply_rope(ctx->ggml_ctx, k, pe);
        return flux2_residual_attention(ctx,
                                        q, k, v,
                                        v->ne[1],
                                        mask,
                                        true,
                                        1.0f,
                                        ctx->block_sparse_attn_enabled,
                                        image_tokens,
                                        text_tokens,
                                        residual_kind);
    }

    __STATIC_INLINE__ ggml_tensor* packed_flux2_attention(GGMLRunnerContext* ctx,
                                                           const PackedRopeQkv& packed,
                                                           int image_tokens,
                                                           int text_tokens,
                                                           const char* residual_kind) {
        return flux2_residual_attention(ctx,
                                        packed.qkv[0],
                                        packed.qkv[1],
                                        packed.qkv[2],
                                        24,
                                        nullptr,
                                        false,
                                        1.0f,
                                        true,
                                        image_tokens,
                                        text_tokens,
                                        residual_kind);
    }

    struct MLP : public UnaryBlock {
        bool use_mlp_silu_act;

    public:
        MLP(int64_t hidden_size, int64_t intermediate_size, bool use_mlp_silu_act = false, bool bias = false)
            : use_mlp_silu_act(use_mlp_silu_act) {
            int64_t mlp_mult_factor = use_mlp_silu_act ? 2 : 1;
            blocks["0"]             = std::make_shared<Linear>(hidden_size, intermediate_size * mlp_mult_factor, bias);
            blocks["2"]             = std::make_shared<Linear>(intermediate_size, hidden_size, bias);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto mlp_0 = std::dynamic_pointer_cast<Linear>(blocks["0"]);
            auto mlp_2 = std::dynamic_pointer_cast<Linear>(blocks["2"]);

            x = mlp_0->forward(ctx, x);
            if (use_mlp_silu_act) {
                x = ggml_ext_silu_act(ctx->ggml_ctx, x);
            } else {
                x = ggml_ext_gelu(ctx->ggml_ctx, x, true);
            }
            x = mlp_2->forward(ctx, x);
            return x;
        }
    };

    struct YakMLP : public UnaryBlock {
    public:
        YakMLP(int64_t hidden_size, int64_t intermediate_size, bool bias = true) {
            blocks["gate_proj"] = std::make_shared<Linear>(hidden_size, intermediate_size, bias);
            blocks["up_proj"]   = std::make_shared<Linear>(hidden_size, intermediate_size, bias);
            blocks["down_proj"] = std::make_shared<Linear>(intermediate_size, hidden_size, bias);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto gate_proj = std::dynamic_pointer_cast<Linear>(blocks["gate_proj"]);
            auto up_proj   = std::dynamic_pointer_cast<Linear>(blocks["up_proj"]);
            auto down_proj = std::dynamic_pointer_cast<Linear>(blocks["down_proj"]);

            auto gate = gate_proj->forward(ctx, x);
            gate      = ggml_silu_inplace(ctx->ggml_ctx, gate);
            x         = up_proj->forward(ctx, x);
            x         = ggml_mul(ctx->ggml_ctx, x, gate);
            x         = down_proj->forward(ctx, x);
            return x;
        }
    };

    struct ModulationOut {
        ggml_tensor* shift = nullptr;
        ggml_tensor* scale = nullptr;
        ggml_tensor* gate  = nullptr;

        ModulationOut(ggml_tensor* shift = nullptr, ggml_tensor* scale = nullptr, ggml_tensor* gate = nullptr)
            : shift(shift), scale(scale), gate(gate) {}

        ModulationOut(GGMLRunnerContext* ctx, ggml_tensor* vec, int64_t offset) {
            int64_t stride = vec->nb[1] * vec->ne[1];
            shift          = ggml_view_2d(ctx->ggml_ctx, vec, vec->ne[0], vec->ne[1], vec->nb[1], stride * (offset + 0));  // [N, dim]
            scale          = ggml_view_2d(ctx->ggml_ctx, vec, vec->ne[0], vec->ne[1], vec->nb[1], stride * (offset + 1));  // [N, dim]
            gate           = ggml_view_2d(ctx->ggml_ctx, vec, vec->ne[0], vec->ne[1], vec->nb[1], stride * (offset + 2));  // [N, dim]
        }
    };

    struct Modulation : public GGMLBlock {
    public:
        bool is_double;
        int multiplier;

    public:
        Modulation(int64_t dim, bool is_double, bool bias = true)
            : is_double(is_double) {
            multiplier    = is_double ? 6 : 3;
            blocks["lin"] = std::shared_ptr<GGMLBlock>(new Linear(dim, dim * multiplier, bias));
        }

        std::vector<ModulationOut> forward(GGMLRunnerContext* ctx, ggml_tensor* vec) {
            // x: [N, dim]
            // return: [ModulationOut, ModulationOut]
            auto lin = std::dynamic_pointer_cast<Linear>(blocks["lin"]);

            auto out = ggml_silu(ctx->ggml_ctx, vec);
            out      = lin->forward(ctx, out);  // [N, multiplier*dim]

            auto m = ggml_reshape_3d(ctx->ggml_ctx, out, vec->ne[0], multiplier, vec->ne[1]);  // [N, multiplier, dim]
            m      = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, m, 0, 2, 1, 3));     // [multiplier, N, dim]

            ModulationOut m_0 = ModulationOut(ctx, m, 0);
            if (is_double) {
                return {m_0, ModulationOut(ctx, m, 3)};
            }

            return {m_0, ModulationOut()};
        }
    };

    __STATIC_INLINE__ ggml_tensor* modulate(ggml_context* ctx,
                                            ggml_tensor* x,
                                            ggml_tensor* shift,
                                            ggml_tensor* scale,
                                            bool skip_reshape = false) {
        // x: [N, L, C]
        // scale: [N, C]
        // shift: [N, C]
        if (!skip_reshape) {
            scale = ggml_reshape_3d(ctx, scale, scale->ne[0], 1, scale->ne[1]);  // [N, 1, C]
            shift = ggml_reshape_3d(ctx, shift, shift->ne[0], 1, shift->ne[1]);  // [N, 1, C]
        }
        x = ggml_add(ctx, x, ggml_mul(ctx, x, scale));
        x = ggml_add(ctx, x, shift);
        return x;
    }

    struct DoubleStreamBlock : public GGMLBlock {
        bool prune_mod;
        int idx = 0;

    public:
        DoubleStreamBlock(int64_t hidden_size,
                          int64_t num_heads,
                          float mlp_ratio,
                          int idx               = 0,
                          bool qkv_bias         = false,
                          bool prune_mod        = false,
                          bool share_modulation = false,
                          bool mlp_proj_bias    = true,
                          bool use_yak_mlp      = false,
                          bool use_mlp_silu_act = false)
            : idx(idx), prune_mod(prune_mod) {
            int64_t mlp_hidden_dim = static_cast<int64_t>(hidden_size * mlp_ratio);

            if (!prune_mod && !share_modulation) {
                blocks["img_mod"] = std::shared_ptr<GGMLBlock>(new Modulation(hidden_size, true));
            }
            blocks["img_norm1"] = std::shared_ptr<GGMLBlock>(new LayerNorm(hidden_size, 1e-6f, false));
            blocks["img_attn"]  = std::shared_ptr<GGMLBlock>(new SelfAttention(hidden_size, num_heads, qkv_bias, mlp_proj_bias));

            blocks["img_norm2"] = std::shared_ptr<GGMLBlock>(new LayerNorm(hidden_size, 1e-6f, false));
            if (use_yak_mlp) {
                blocks["img_mlp"] = std::shared_ptr<GGMLBlock>(new YakMLP(hidden_size, mlp_hidden_dim, mlp_proj_bias));
            } else {
                blocks["img_mlp"] = std::shared_ptr<GGMLBlock>(new MLP(hidden_size, mlp_hidden_dim, use_mlp_silu_act, mlp_proj_bias));
            }

            if (!prune_mod && !share_modulation) {
                blocks["txt_mod"] = std::shared_ptr<GGMLBlock>(new Modulation(hidden_size, true));
            }
            blocks["txt_norm1"] = std::shared_ptr<GGMLBlock>(new LayerNorm(hidden_size, 1e-6f, false));
            blocks["txt_attn"]  = std::shared_ptr<GGMLBlock>(new SelfAttention(hidden_size, num_heads, qkv_bias, mlp_proj_bias));

            blocks["txt_norm2"] = std::shared_ptr<GGMLBlock>(new LayerNorm(hidden_size, 1e-6f, false));
            if (use_yak_mlp) {
                blocks["txt_mlp"] = std::shared_ptr<GGMLBlock>(new YakMLP(hidden_size, mlp_hidden_dim, mlp_proj_bias));
            } else {
                blocks["txt_mlp"] = std::shared_ptr<GGMLBlock>(new MLP(hidden_size, mlp_hidden_dim, use_mlp_silu_act, mlp_proj_bias));
            }
        }

        std::vector<ModulationOut> get_distil_img_mod(GGMLRunnerContext* ctx, ggml_tensor* vec) {
            // TODO: not hardcoded?
            const int single_blocks_count = 38;
            const int double_blocks_count = 19;

            int64_t offset = 6 * idx + 3 * single_blocks_count;
            return {ModulationOut(ctx, vec, offset), ModulationOut(ctx, vec, offset + 3)};
        }

        std::vector<ModulationOut> get_distil_txt_mod(GGMLRunnerContext* ctx, ggml_tensor* vec) {
            // TODO: not hardcoded?
            const int single_blocks_count = 38;
            const int double_blocks_count = 19;

            int64_t offset = 6 * idx + 6 * double_blocks_count + 3 * single_blocks_count;
            return {ModulationOut(ctx, vec, offset), ModulationOut(ctx, vec, offset + 3)};
        }

        std::pair<ggml_tensor*, ggml_tensor*> forward(GGMLRunnerContext* ctx,
                                                      ggml_tensor* img,
                                                      ggml_tensor* txt,
                                                      ggml_tensor* vec,
                                                      ggml_tensor* pe,
                                                      ggml_tensor* mask                   = nullptr,
                                                      std::vector<ModulationOut> img_mods = {},
                                                      std::vector<ModulationOut> txt_mods = {},
                                                      ggml_tensor* packed_pe              = nullptr) {
            // img: [N, n_img_token, hidden_size]
            // txt: [N, n_txt_token, hidden_size]
            // pe: [n_img_token + n_txt_token, d_head/2, 2, 2]
            // return: ([N, n_img_token, hidden_size], [N, n_txt_token, hidden_size])
            auto img_norm1 = std::dynamic_pointer_cast<LayerNorm>(blocks["img_norm1"]);
            auto img_attn  = std::dynamic_pointer_cast<SelfAttention>(blocks["img_attn"]);

            auto img_norm2 = std::dynamic_pointer_cast<LayerNorm>(blocks["img_norm2"]);
            auto img_mlp   = std::dynamic_pointer_cast<UnaryBlock>(blocks["img_mlp"]);

            auto txt_norm1 = std::dynamic_pointer_cast<LayerNorm>(blocks["txt_norm1"]);
            auto txt_attn  = std::dynamic_pointer_cast<SelfAttention>(blocks["txt_attn"]);

            auto txt_norm2 = std::dynamic_pointer_cast<LayerNorm>(blocks["txt_norm2"]);
            auto txt_mlp   = std::dynamic_pointer_cast<UnaryBlock>(blocks["txt_mlp"]);

            if (img_mods.empty()) {
                if (prune_mod) {
                    img_mods = get_distil_img_mod(ctx, vec);
                } else {
                    auto img_mod = std::dynamic_pointer_cast<Modulation>(blocks["img_mod"]);
                    img_mods     = img_mod->forward(ctx, vec);
                }
            }
            ModulationOut img_mod1 = img_mods[0];
            ModulationOut img_mod2 = img_mods[1];
            if (txt_mods.empty()) {
                if (prune_mod) {
                    txt_mods = get_distil_txt_mod(ctx, vec);
                } else {
                    auto txt_mod = std::dynamic_pointer_cast<Modulation>(blocks["txt_mod"]);
                    txt_mods     = txt_mod->forward(ctx, vec);
                }
            }
            ModulationOut txt_mod1 = txt_mods[0];
            ModulationOut txt_mod2 = txt_mods[1];

            // prepare image for attention
            auto img_modulated = img_norm1->forward(ctx, img);
            img_modulated      = Flux::modulate(ctx->ggml_ctx, img_modulated, img_mod1.shift, img_mod1.scale);

            // prepare txt for attention
            auto txt_modulated = txt_norm1->forward(ctx, txt);
            txt_modulated      = Flux::modulate(ctx->ggml_ctx, txt_modulated, txt_mod1.shift, txt_mod1.scale);
            ggml_tensor* attn = nullptr;
            ggml_tensor* img_attn_out = nullptr;
            ggml_tensor* txt_attn_out = nullptr;
            auto packed = make_packed_flux2_dual_qkv(ctx, img_attn, txt_attn,
                                                      img_modulated, txt_modulated, packed_pe);
            if (packed.storage != nullptr && mask == nullptr) {
                attn = packed_flux2_attention(ctx, packed, static_cast<int>(img->ne[1]),
                                              static_cast<int>(txt->ne[1]), "double");
                img_attn_out = ggml_view_3d(ctx->ggml_ctx, attn, attn->ne[0], img->ne[1], attn->ne[2],
                                            attn->nb[1], attn->nb[2], 0);
                txt_attn_out = ggml_view_3d(ctx->ggml_ctx, attn, attn->ne[0], txt->ne[1], attn->ne[2],
                                            attn->nb[1], attn->nb[2], img->ne[1] * attn->nb[1]);
            } else {
                auto img_qkv = img_attn->pre_attention(ctx, img_modulated);
                auto txt_qkv = txt_attn->pre_attention(ctx, txt_modulated);
                const bool image_first = packed_pe != nullptr;
                auto q = image_first ? ggml_concat(ctx->ggml_ctx, img_qkv[0], txt_qkv[0], 2) :
                                       ggml_concat(ctx->ggml_ctx, txt_qkv[0], img_qkv[0], 2);
                auto k = image_first ? ggml_concat(ctx->ggml_ctx, img_qkv[1], txt_qkv[1], 2) :
                                       ggml_concat(ctx->ggml_ctx, txt_qkv[1], img_qkv[1], 2);
                auto v = image_first ? ggml_concat(ctx->ggml_ctx, img_qkv[2], txt_qkv[2], 2) :
                                       ggml_concat(ctx->ggml_ctx, txt_qkv[2], img_qkv[2], 2);
                if (image_first) {
                    attn = flux2_rope_residual_attention(
                        ctx, q, k, v, packed_pe, mask,
                        static_cast<int>(img->ne[1]), static_cast<int>(txt->ne[1]), "double");
                    img_attn_out = ggml_view_3d(ctx->ggml_ctx, attn, attn->ne[0], img->ne[1], attn->ne[2],
                                                attn->nb[1], attn->nb[2], 0);
                    txt_attn_out = ggml_view_3d(ctx->ggml_ctx, attn, attn->ne[0], txt->ne[1], attn->ne[2],
                                                attn->nb[1], attn->nb[2], img->ne[1] * attn->nb[1]);
                } else {
                    attn = Rope::attention(ctx, q, k, v, pe, mask);
                    txt_attn_out = ggml_view_3d(ctx->ggml_ctx, attn, attn->ne[0], txt->ne[1], attn->ne[2],
                                                attn->nb[1], attn->nb[2], 0);
                    img_attn_out = ggml_view_3d(ctx->ggml_ctx, attn, attn->ne[0], img->ne[1], attn->ne[2],
                                                attn->nb[1], attn->nb[2], txt->ne[1] * attn->nb[1]);
                }
            }

            // calculate the img bloks
            img = ggml_add(ctx->ggml_ctx, img, ggml_mul(ctx->ggml_ctx, img_attn->post_attention(ctx, img_attn_out), img_mod1.gate));

            auto img_mlp_out = img_mlp->forward(ctx, Flux::modulate(ctx->ggml_ctx, img_norm2->forward(ctx, img), img_mod2.shift, img_mod2.scale));

            img = ggml_add(ctx->ggml_ctx, img, ggml_mul(ctx->ggml_ctx, img_mlp_out, img_mod2.gate));

            // calculate the txt bloks
            txt = ggml_add(ctx->ggml_ctx, txt, ggml_mul(ctx->ggml_ctx, txt_attn->post_attention(ctx, txt_attn_out), txt_mod1.gate));

            auto txt_mlp_out = txt_mlp->forward(ctx, Flux::modulate(ctx->ggml_ctx, txt_norm2->forward(ctx, txt), txt_mod2.shift, txt_mod2.scale));
            txt              = ggml_add(ctx->ggml_ctx, txt, ggml_mul(ctx->ggml_ctx, txt_mlp_out, txt_mod2.gate));

            return {img, txt};
        }
    };

    struct SingleStreamBlock : public GGMLBlock {
    public:
        int64_t num_heads;
        int64_t hidden_size;
        int64_t mlp_hidden_dim;
        bool prune_mod;
        int idx = 0;
        bool use_yak_mlp;
        bool use_mlp_silu_act;
        int64_t mlp_mult_factor;

    public:
        SingleStreamBlock(int64_t hidden_size,
                          int64_t num_heads,
                          float mlp_ratio       = 4.0f,
                          int idx               = 0,
                          float qk_scale        = 0.f,
                          bool prune_mod        = false,
                          bool share_modulation = false,
                          bool mlp_proj_bias    = true,
                          bool use_yak_mlp      = false,
                          bool use_mlp_silu_act = false)
            : hidden_size(hidden_size), num_heads(num_heads), idx(idx), prune_mod(prune_mod), use_yak_mlp(use_yak_mlp), use_mlp_silu_act(use_mlp_silu_act) {
            int64_t head_dim = hidden_size / num_heads;
            float scale      = qk_scale;
            if (scale <= 0.f) {
                scale = 1 / sqrt((float)head_dim);
            }
            mlp_hidden_dim  = static_cast<int64_t>(hidden_size * mlp_ratio);
            mlp_mult_factor = 1;
            if (use_yak_mlp || use_mlp_silu_act) {
                mlp_mult_factor = 2;
            }
            blocks["linear1"]  = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, hidden_size * 3 + mlp_hidden_dim * mlp_mult_factor, mlp_proj_bias));
            blocks["linear2"]  = std::shared_ptr<GGMLBlock>(new Linear(hidden_size + mlp_hidden_dim, hidden_size, mlp_proj_bias));
            blocks["norm"]     = std::shared_ptr<GGMLBlock>(new QKNorm(head_dim));
            blocks["pre_norm"] = std::shared_ptr<GGMLBlock>(new LayerNorm(hidden_size, 1e-6f, false));
            // mlp_act is nn.GELU(approximate="tanh")
            if (!prune_mod && !share_modulation) {
                blocks["modulation"] = std::shared_ptr<GGMLBlock>(new Modulation(hidden_size, false));
            }
        }

        ModulationOut get_distil_mod(GGMLRunnerContext* ctx, ggml_tensor* vec) {
            int64_t offset = 3 * idx;
            return ModulationOut(ctx, vec, offset);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* vec,
                             ggml_tensor* pe,
                             ggml_tensor* mask               = nullptr,
                             std::vector<ModulationOut> mods = {},
                             ggml_tensor* packed_pe          = nullptr) {
            // x: [N, n_token, hidden_size]
            // pe: [n_token, d_head/2, 2, 2]
            // return: [N, n_token, hidden_size]

            auto linear1  = std::dynamic_pointer_cast<Linear>(blocks["linear1"]);
            auto linear2  = std::dynamic_pointer_cast<Linear>(blocks["linear2"]);
            auto norm     = std::dynamic_pointer_cast<QKNorm>(blocks["norm"]);
            auto pre_norm = std::dynamic_pointer_cast<LayerNorm>(blocks["pre_norm"]);

            ModulationOut mod;
            if (!mods.empty()) {
                mod = mods[0];
            } else {
                if (prune_mod) {
                    mod = get_distil_mod(ctx, vec);
                } else {
                    auto modulation = std::dynamic_pointer_cast<Modulation>(blocks["modulation"]);

                    mod = modulation->forward(ctx, vec)[0];
                }
            }

            auto x_mod = Flux::modulate(ctx->ggml_ctx, pre_norm->forward(ctx, x), mod.shift, mod.scale);
            int64_t head_dim = hidden_size / num_heads;
            ggml_tensor* attn = nullptr;
            ggml_tensor* mlp = nullptr;
            ggml_tensor* weight = linear1->get_param("weight");
            ggml_tensor* bias = linear1->get_param("bias");
            ggml_tensor* q_norm = norm->param("q_norm");
            ggml_tensor* k_norm = norm->param("k_norm");
            const int image_tokens = ctx->block_sparse_token_h * ctx->block_sparse_token_w;
            PackedRopeQkv packed;
            if (packed_pe != nullptr && weight != nullptr && weight->type == GGML_TYPE_F16 &&
                weight->ne[0] == hidden_size && weight->ne[1] >= 3 * hidden_size) {
                ggml_tensor* qkv_weight = ggml_view_2d(ctx->ggml_ctx, weight, hidden_size, 3 * hidden_size,
                                                       weight->nb[1], 0);
                qkv_weight->flags |= weight->flags & (SD_GGML_TENSOR_FLAG_HTP_PACKED_W16A32 |
                                                      SD_GGML_TENSOR_FLAG_HTP_MUL_MAT_WEIGHT);
                ggml_tensor* qkv_bias = bias != nullptr ?
                    ggml_view_1d(ctx->ggml_ctx, bias, 3 * hidden_size, 0) : nullptr;
                packed = make_packed_flux2_single_qkv(ctx, x_mod, qkv_weight, qkv_bias,
                                                      q_norm, k_norm, packed_pe, image_tokens);
            }
            if (packed.storage != nullptr && mask == nullptr) {
                attn = packed_flux2_attention(ctx, packed, image_tokens,
                                              static_cast<int>(x->ne[1]) - image_tokens,
                                              "single");
                const int64_t mlp_cols = mlp_hidden_dim * mlp_mult_factor;
                ggml_tensor* full_qkv_mlp = nullptr;
                if (sd_env_flag_enabled("SD_HTP_PACKED_FLUX2_SINGLE_FULL_MLP")) {
                    ggml_tensor* qkv_mlp = linear1->forward(ctx, x_mod);
                    full_qkv_mlp = qkv_mlp;
                    mlp = ggml_view_3d(ctx->ggml_ctx, qkv_mlp, mlp_cols, qkv_mlp->ne[1], qkv_mlp->ne[2],
                                       qkv_mlp->nb[1], qkv_mlp->nb[2], 3 * hidden_size * qkv_mlp->nb[0]);
                } else {
                    ggml_tensor* mlp_weight = ggml_view_2d(ctx->ggml_ctx, weight, hidden_size, mlp_cols,
                                                           weight->nb[1], 3 * hidden_size * weight->nb[1]);
                    mlp_weight->flags |= weight->flags & (SD_GGML_TENSOR_FLAG_HTP_PACKED_W16A32 |
                                                          SD_GGML_TENSOR_FLAG_HTP_MUL_MAT_WEIGHT);
                    ggml_tensor* mlp_bias = bias != nullptr ?
                        ggml_view_1d(ctx->ggml_ctx, bias, mlp_cols, 3 * hidden_size * bias->nb[0]) : nullptr;
                    mlp = ggml_ext_linear(ctx->ggml_ctx, x_mod, mlp_weight, mlp_bias, false, 1.0f);
                }
                if (sd_env_flag_enabled("SD_HTP_PACKED_FLUX2_TRACE") && idx == 0) {
                    LOG_INFO("flux2_single_weight root=[%lld,%lld] nb=[%zu,%zu,%zu,%zu] nbytes=%zu qkv_offset=%zu mlp_offset=%zu",
                             (long long) weight->ne[0], (long long) weight->ne[1],
                             weight->nb[0], weight->nb[1], weight->nb[2], weight->nb[3],
                             ggml_nbytes(weight),
                             (size_t) 0,
                             (size_t) 3 * (size_t) hidden_size * weight->nb[1]);
                    LOG_INFO("flux2_single_layout packed attn=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu] mlp=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu] xmod=[%lld,%lld,%lld,%lld] linear1=[%lld,%lld,%lld,%lld] linear2w=[%lld,%lld,%lld,%lld]",
                             (long long) attn->ne[0], (long long) attn->ne[1], (long long) attn->ne[2], (long long) attn->ne[3],
                             attn->nb[0], attn->nb[1], attn->nb[2], attn->nb[3],
                             (long long) mlp->ne[0], (long long) mlp->ne[1], (long long) mlp->ne[2], (long long) mlp->ne[3],
                             mlp->nb[0], mlp->nb[1], mlp->nb[2], mlp->nb[3],
                             (long long) x_mod->ne[0], (long long) x_mod->ne[1], (long long) x_mod->ne[2], (long long) x_mod->ne[3],
                             full_qkv_mlp != nullptr ? (long long) full_qkv_mlp->ne[0] : -1,
                             full_qkv_mlp != nullptr ? (long long) full_qkv_mlp->ne[1] : -1,
                             full_qkv_mlp != nullptr ? (long long) full_qkv_mlp->ne[2] : -1,
                             full_qkv_mlp != nullptr ? (long long) full_qkv_mlp->ne[3] : -1,
                             (long long) linear2->get_param("weight")->ne[0], (long long) linear2->get_param("weight")->ne[1],
                             (long long) linear2->get_param("weight")->ne[2], (long long) linear2->get_param("weight")->ne[3]);
                }
            } else {
                auto qkv_mlp = linear1->forward(ctx, x_mod);
                if (sd_env_flag_enabled("SD_HTP_PACKED_FLUX2_TRACE") &&
                    packed_pe != nullptr && idx == 0) {
                    ggml_set_name(qkv_mlp, "flux2_single_dense_qkv_debug");
                    if (qkv_mlp->data != nullptr && q_norm->data != nullptr && packed_pe->data != nullptr) {
                        const float * raw = static_cast<const float *>(qkv_mlp->data);
                        const float * gamma = static_cast<const float *>(q_norm->data);
                        const float * pe_data = static_cast<const float *>(packed_pe->data);
                        const int64_t output_dim = qkv_mlp->ne[0];
                        LOG_INFO("flux2_dense_qkv_values raw_t0=[%.8g,%.8g,%.8g,%.8g] raw_t1=[%.8g,%.8g,%.8g,%.8g] qgamma=[%.8g,%.8g,%.8g,%.8g] pe=[%.8g,%.8g,%.8g,%.8g]",
                                 raw[0], raw[1], raw[2], raw[3],
                                 raw[output_dim], raw[output_dim + 1], raw[output_dim + 2], raw[output_dim + 3],
                                 gamma[0], gamma[1], gamma[2], gamma[3],
                                 pe_data[0], pe_data[1], pe_data[2], pe_data[3]);
                    }
                }
                auto q = ggml_view_4d(ctx->ggml_ctx, qkv_mlp, head_dim, num_heads, qkv_mlp->ne[1], qkv_mlp->ne[2],
                                      qkv_mlp->nb[0] * head_dim, qkv_mlp->nb[1], qkv_mlp->nb[2], 0);
                auto k = ggml_view_4d(ctx->ggml_ctx, qkv_mlp, head_dim, num_heads, qkv_mlp->ne[1], qkv_mlp->ne[2],
                                      qkv_mlp->nb[0] * head_dim, qkv_mlp->nb[1], qkv_mlp->nb[2], qkv_mlp->nb[0] * hidden_size);
                auto v = ggml_view_4d(ctx->ggml_ctx, qkv_mlp, head_dim, num_heads, qkv_mlp->ne[1], qkv_mlp->ne[2],
                                      qkv_mlp->nb[0] * head_dim, qkv_mlp->nb[1], qkv_mlp->nb[2], qkv_mlp->nb[0] * 2 * hidden_size);
                q = norm->query_norm(ctx, q);
                k = norm->key_norm(ctx, k);
                if (packed_pe != nullptr) {
                    attn = flux2_rope_residual_attention(
                        ctx, q, k, v, packed_pe, mask, image_tokens,
                        static_cast<int>(x->ne[1]) - image_tokens, "single");
                } else {
                    attn = Rope::attention(ctx, q, k, v, pe, mask);
                }
                mlp = ggml_view_3d(ctx->ggml_ctx, qkv_mlp, mlp_hidden_dim * mlp_mult_factor,
                                   qkv_mlp->ne[1], qkv_mlp->ne[2], qkv_mlp->nb[1], qkv_mlp->nb[2],
                                   hidden_size * 3 * qkv_mlp->nb[0]);
            }
            if (use_yak_mlp) {
                mlp = ggml_ext_silu_act(ctx->ggml_ctx, mlp, false);
            } else if (use_mlp_silu_act) {
                mlp = ggml_ext_silu_act(ctx->ggml_ctx, mlp);
            } else {
                mlp = ggml_ext_gelu(ctx->ggml_ctx, mlp, true);
            }
            auto attn_mlp = ggml_concat(ctx->ggml_ctx, attn, mlp, 0);  // [N, n_token, hidden_size + mlp_hidden_dim]
            if (packed.storage != nullptr && mask == nullptr &&
                sd_env_flag_enabled("SD_HTP_PACKED_FLUX2_TRACE") && idx == 0) {
                LOG_INFO("flux2_single_layout post_mlp=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu] attn_mlp=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu]",
                         (long long) mlp->ne[0], (long long) mlp->ne[1], (long long) mlp->ne[2], (long long) mlp->ne[3],
                         mlp->nb[0], mlp->nb[1], mlp->nb[2], mlp->nb[3],
                         (long long) attn_mlp->ne[0], (long long) attn_mlp->ne[1], (long long) attn_mlp->ne[2], (long long) attn_mlp->ne[3],
                         attn_mlp->nb[0], attn_mlp->nb[1], attn_mlp->nb[2], attn_mlp->nb[3]);
            }
            auto output   = linear2->forward(ctx, attn_mlp);           // [N, n_token, hidden_size]

            output = ggml_add(ctx->ggml_ctx, x, ggml_mul(ctx->ggml_ctx, output, mod.gate));
            return output;
        }
    };

    struct LastLayer : public GGMLBlock {
        bool prune_mod;

    public:
        LastLayer(int64_t hidden_size,
                  int64_t patch_size,
                  int64_t out_channels,
                  bool prune_mod = false,
                  bool bias      = true)
            : prune_mod(prune_mod) {
            blocks["norm_final"] = std::shared_ptr<GGMLBlock>(new LayerNorm(hidden_size, 1e-06f, false));
            blocks["linear"]     = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, patch_size * patch_size * out_channels, bias));
            if (!prune_mod) {
                blocks["adaLN_modulation.1"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, 2 * hidden_size, bias));
            }
        }

        ModulationOut get_distil_mod(GGMLRunnerContext* ctx, ggml_tensor* vec) {
            int64_t offset = vec->ne[2] - 2;
            int64_t stride = vec->nb[1] * vec->ne[1];
            auto shift     = ggml_view_2d(ctx->ggml_ctx, vec, vec->ne[0], vec->ne[1], vec->nb[1], stride * (offset + 0));  // [N, dim]
            auto scale     = ggml_view_2d(ctx->ggml_ctx, vec, vec->ne[0], vec->ne[1], vec->nb[1], stride * (offset + 1));  // [N, dim]
            // No gate
            return {shift, scale, nullptr};
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* c) {
            // x: [N, n_token, hidden_size]
            // c: [N, hidden_size]
            // return: [N, n_token, patch_size * patch_size * out_channels]
            auto norm_final = std::dynamic_pointer_cast<LayerNorm>(blocks["norm_final"]);
            auto linear     = std::dynamic_pointer_cast<Linear>(blocks["linear"]);
            ggml_tensor *shift, *scale;
            if (prune_mod) {
                auto mod = get_distil_mod(ctx, c);
                shift    = mod.shift;
                scale    = mod.scale;
            } else {
                auto adaLN_modulation_1 = std::dynamic_pointer_cast<Linear>(blocks["adaLN_modulation.1"]);

                auto m     = adaLN_modulation_1->forward(ctx, ggml_silu(ctx->ggml_ctx, c));  // [N, 2 * hidden_size]
                auto m_vec = ggml_ext_chunk(ctx->ggml_ctx, m, 2, 0);
                shift      = m_vec[0];  // [N, hidden_size]
                scale      = m_vec[1];  // [N, hidden_size]
            }

            x = Flux::modulate(ctx->ggml_ctx, norm_final->forward(ctx, x), shift, scale);
            x = linear->forward(ctx, x);

            return x;
        }
    };

    struct ChromaApproximator : public GGMLBlock {
        int64_t inner_size = 5120;
        int64_t n_layers   = 5;
        ChromaApproximator(int64_t in_channels = 64, int64_t hidden_size = 3072) {
            blocks["in_proj"] = std::shared_ptr<GGMLBlock>(new Linear(in_channels, inner_size, true));
            for (int i = 0; i < n_layers; i++) {
                blocks["norms." + std::to_string(i)]  = std::shared_ptr<GGMLBlock>(new RMSNorm(inner_size));
                blocks["layers." + std::to_string(i)] = std::shared_ptr<GGMLBlock>(new MLPEmbedder(inner_size, inner_size));
            }
            blocks["out_proj"] = std::shared_ptr<GGMLBlock>(new Linear(inner_size, hidden_size, true));
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto in_proj  = std::dynamic_pointer_cast<Linear>(blocks["in_proj"]);
            auto out_proj = std::dynamic_pointer_cast<Linear>(blocks["out_proj"]);

            x = in_proj->forward(ctx, x);
            for (int i = 0; i < n_layers; i++) {
                auto norm  = std::dynamic_pointer_cast<RMSNorm>(blocks["norms." + std::to_string(i)]);
                auto embed = std::dynamic_pointer_cast<MLPEmbedder>(blocks["layers." + std::to_string(i)]);
                x          = ggml_add_inplace(ctx->ggml_ctx, x, embed->forward(ctx, norm->forward(ctx, x)));
            }
            x = out_proj->forward(ctx, x);

            return x;
        }
    };

    struct NerfEmbedder : public GGMLBlock {
        NerfEmbedder(int64_t in_channels,
                     int64_t hidden_size_input,
                     int64_t max_freqs) {
            blocks["embedder.0"] = std::make_shared<Linear>(in_channels + max_freqs * max_freqs, hidden_size_input);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* dct) {
            // x: (B, P^2, C)
            // dct: (1, P^2, max_freqs^2)
            // return: (B, P^2, hidden_size_input)
            auto embedder = std::dynamic_pointer_cast<Linear>(blocks["embedder.0"]);

            dct = ggml_repeat_4d(ctx->ggml_ctx, dct, dct->ne[0], dct->ne[1], x->ne[2], x->ne[3]);
            x   = ggml_concat(ctx->ggml_ctx, x, dct, 0);
            x   = embedder->forward(ctx, x);

            return x;
        }
    };

    struct NerfGLUBlock : public GGMLBlock {
        int64_t mlp_ratio;
        NerfGLUBlock(int64_t hidden_size_s,
                     int64_t hidden_size_x,
                     int64_t mlp_ratio)
            : mlp_ratio(mlp_ratio) {
            int64_t total_params      = 3 * hidden_size_x * hidden_size_x * mlp_ratio;
            blocks["param_generator"] = std::make_shared<Linear>(hidden_size_s, total_params);
            blocks["norm"]            = std::make_shared<RMSNorm>(hidden_size_x);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* s) {
            // x: (batch_size, n_token, hidden_size_x)
            // s: (batch_size, hidden_size_s)
            // return: (batch_size, n_token, hidden_size_x)
            auto param_generator = std::dynamic_pointer_cast<Linear>(blocks["param_generator"]);
            auto norm            = std::dynamic_pointer_cast<RMSNorm>(blocks["norm"]);

            int64_t batch_size    = x->ne[2];
            int64_t hidden_size_x = x->ne[0];

            auto mlp_params = param_generator->forward(ctx, s);
            auto fc_params  = ggml_ext_chunk(ctx->ggml_ctx, mlp_params, 3, 0);
            auto fc1_gate   = ggml_reshape_3d(ctx->ggml_ctx, fc_params[0], hidden_size_x * mlp_ratio, hidden_size_x, batch_size);
            auto fc1_value  = ggml_reshape_3d(ctx->ggml_ctx, fc_params[1], hidden_size_x * mlp_ratio, hidden_size_x, batch_size);
            auto fc2        = ggml_reshape_3d(ctx->ggml_ctx, fc_params[2], hidden_size_x, mlp_ratio * hidden_size_x, batch_size);

            fc1_gate  = ggml_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, fc1_gate, 1, 0, 2, 3));  // [batch_size, hidden_size_x*mlp_ratio, hidden_size_x]
            fc1_gate  = ggml_l2_norm(ctx->ggml_ctx, fc1_gate, 1e-12f);
            fc1_value = ggml_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, fc1_value, 1, 0, 2, 3));  // [batch_size, hidden_size_x*mlp_ratio, hidden_size_x]
            fc1_value = ggml_l2_norm(ctx->ggml_ctx, fc1_value, 1e-12f);
            fc2       = ggml_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, fc2, 1, 0, 2, 3));  // [batch_size, hidden_size_x, hidden_size_x*mlp_ratio]
            fc2       = ggml_l2_norm(ctx->ggml_ctx, fc2, 1e-12f);

            auto res_x = x;
            x          = norm->forward(ctx, x);  // [batch_size, n_token, hidden_size_x]

            auto x1 = ggml_mul_mat(ctx->ggml_ctx, fc1_gate, x);  // [batch_size, n_token, hidden_size_x*mlp_ratio]
            x1      = ggml_silu_inplace(ctx->ggml_ctx, x1);

            auto x2 = ggml_mul_mat(ctx->ggml_ctx, fc1_value, x);  // [batch_size, n_token, hidden_size_x*mlp_ratio]

            x = ggml_mul_inplace(ctx->ggml_ctx, x1, x2);  // [batch_size, n_token, hidden_size_x*mlp_ratio]

            x = ggml_mul_mat(ctx->ggml_ctx, fc2, x);  // [batch_size, n_token, hidden_size_x]

            x = ggml_add_inplace(ctx->ggml_ctx, x, res_x);

            return x;
        }
    };

    struct NerfFinalLayer : public GGMLBlock {
        NerfFinalLayer(int64_t hidden_size,
                       int64_t out_channels) {
            blocks["norm"]   = std::make_shared<RMSNorm>(hidden_size);
            blocks["linear"] = std::make_shared<Linear>(hidden_size, out_channels);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x) {
            auto norm   = std::dynamic_pointer_cast<RMSNorm>(blocks["norm"]);
            auto linear = std::dynamic_pointer_cast<Linear>(blocks["linear"]);

            x = norm->forward(ctx, x);
            x = linear->forward(ctx, x);

            return x;
        }
    };

    struct NerfFinalLayerConv : public GGMLBlock {
        NerfFinalLayerConv(int64_t hidden_size,
                           int64_t out_channels) {
            blocks["norm"] = std::make_shared<RMSNorm>(hidden_size);
            blocks["conv"] = std::make_shared<Conv2d>(hidden_size, out_channels, std::pair{3, 3}, std::pair{1, 1}, std::pair{1, 1});
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x) {
            // x: [N, C, H, W]
            auto norm = std::dynamic_pointer_cast<RMSNorm>(blocks["norm"]);
            auto conv = std::dynamic_pointer_cast<Conv2d>(blocks["conv"]);

            x = ggml_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, x, 2, 0, 1, 3));  // [N, H, W, C]
            x = norm->forward(ctx, x);
            x = ggml_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, x, 1, 2, 0, 3));  // [N, C, H, W]
            x = conv->forward(ctx, x);

            return x;
        }
    };

    struct ChromaRadianceParams {
        int64_t nerf_hidden_size = 64;
        int nerf_mlp_ratio       = 4;
        int nerf_depth           = 4;
        int nerf_max_freqs       = 8;
        bool use_x0              = false;
        bool fake_patch_size_x2  = false;
    };

    struct FluxParams {
        SDVersion version         = VERSION_FLUX;
        bool is_chroma            = false;
        int patch_size            = 2;
        int64_t in_channels       = 64;
        int64_t out_channels      = 64;
        int64_t vec_in_dim        = 768;
        int64_t context_in_dim    = 4096;
        int64_t hidden_size       = 3072;
        float mlp_ratio           = 4.0f;
        int num_heads             = 24;
        int depth                 = 19;
        int depth_single_blocks   = 38;
        std::vector<int> axes_dim = {16, 56, 56};
        int axes_dim_sum          = 128;
        int theta                 = 10000;
        bool qkv_bias             = true;
        bool guidance_embed       = true;
        int64_t in_dim            = 64;
        bool disable_bias         = false;
        bool share_modulation     = false;
        bool semantic_txt_norm    = false;
        bool use_yak_mlp          = false;
        bool use_mlp_silu_act     = false;
        float ref_index_scale     = 1.f;
        ChromaRadianceParams chroma_radiance_params;
    };

    struct Flux : public GGMLBlock {
    public:
        FluxParams params;
        Flux() {}
        Flux(FluxParams params)
            : params(params) {
            if (params.version == VERSION_CHROMA_RADIANCE) {
                std::pair<int, int> kernel_size = {params.patch_size, params.patch_size};
                if (params.chroma_radiance_params.fake_patch_size_x2) {
                    kernel_size = {params.patch_size / 2, params.patch_size / 2};
                }
                std::pair<int, int> stride = kernel_size;

                blocks["img_in_patch"] = std::make_shared<Conv2d>(params.in_channels,
                                                                  params.hidden_size,
                                                                  kernel_size,
                                                                  stride);
            } else {
                blocks["img_in"] = std::make_shared<Linear>(params.in_channels, params.hidden_size, !params.disable_bias);
            }
            if (params.is_chroma) {
                blocks["distilled_guidance_layer"] = std::make_shared<ChromaApproximator>(params.in_dim, params.hidden_size);
            } else {
                blocks["time_in"] = std::make_shared<MLPEmbedder>(256, params.hidden_size, !params.disable_bias);
                if (params.vec_in_dim > 0) {
                    blocks["vector_in"] = std::make_shared<MLPEmbedder>(params.vec_in_dim, params.hidden_size, !params.disable_bias);
                }
                if (params.guidance_embed) {
                    blocks["guidance_in"] = std::make_shared<MLPEmbedder>(256, params.hidden_size, !params.disable_bias);
                }
            }
            if (params.semantic_txt_norm) {
                blocks["txt_norm"] = std::make_shared<RMSNorm>(params.context_in_dim);
            }
            blocks["txt_in"] = std::make_shared<Linear>(params.context_in_dim, params.hidden_size, !params.disable_bias);

            for (int i = 0; i < params.depth; i++) {
                blocks["double_blocks." + std::to_string(i)] = std::make_shared<DoubleStreamBlock>(params.hidden_size,
                                                                                                   params.num_heads,
                                                                                                   params.mlp_ratio,
                                                                                                   i,
                                                                                                   params.qkv_bias,
                                                                                                   params.is_chroma,
                                                                                                   params.share_modulation,
                                                                                                   !params.disable_bias,
                                                                                                   params.use_yak_mlp,
                                                                                                   params.use_mlp_silu_act);
            }

            for (int i = 0; i < params.depth_single_blocks; i++) {
                blocks["single_blocks." + std::to_string(i)] = std::make_shared<SingleStreamBlock>(params.hidden_size,
                                                                                                   params.num_heads,
                                                                                                   params.mlp_ratio,
                                                                                                   i,
                                                                                                   0.f,
                                                                                                   params.is_chroma,
                                                                                                   params.share_modulation,
                                                                                                   !params.disable_bias,
                                                                                                   params.use_yak_mlp,
                                                                                                   params.use_mlp_silu_act);
            }

            if (params.version == VERSION_CHROMA_RADIANCE) {
                blocks["nerf_image_embedder"] = std::make_shared<NerfEmbedder>(params.in_channels,
                                                                               params.chroma_radiance_params.nerf_hidden_size,
                                                                               params.chroma_radiance_params.nerf_max_freqs);

                for (int i = 0; i < params.chroma_radiance_params.nerf_depth; i++) {
                    blocks["nerf_blocks." + std::to_string(i)] = std::make_shared<NerfGLUBlock>(params.hidden_size,
                                                                                                params.chroma_radiance_params.nerf_hidden_size,
                                                                                                params.chroma_radiance_params.nerf_mlp_ratio);
                }

                blocks["nerf_final_layer_conv"] = std::make_shared<NerfFinalLayerConv>(params.chroma_radiance_params.nerf_hidden_size,
                                                                                       params.in_channels);

            } else {
                blocks["final_layer"] = std::make_shared<LastLayer>(params.hidden_size, 1, params.out_channels, params.is_chroma, !params.disable_bias);
            }

            if (params.share_modulation) {
                blocks["double_stream_modulation_img"] = std::make_shared<Modulation>(params.hidden_size, true, !params.disable_bias);
                blocks["double_stream_modulation_txt"] = std::make_shared<Modulation>(params.hidden_size, true, !params.disable_bias);
                blocks["single_stream_modulation"]     = std::make_shared<Modulation>(params.hidden_size, false, !params.disable_bias);
            }
        }

        ggml_tensor* forward_orig(GGMLRunnerContext* ctx,
                                  ggml_tensor* img,
                                  ggml_tensor* txt,
                                  ggml_tensor* timesteps,
                                  ggml_tensor* y,
                                  ggml_tensor* guidance,
                                  ggml_tensor* pe,
                                  ggml_tensor* mod_index_arange = nullptr,
                                  std::vector<int> skip_layers  = {},
                                  ggml_tensor* packed_pe        = nullptr) {
            auto img_in      = std::dynamic_pointer_cast<Linear>(blocks["img_in"]);
            auto txt_in      = std::dynamic_pointer_cast<Linear>(blocks["txt_in"]);
            auto final_layer = std::dynamic_pointer_cast<LastLayer>(blocks["final_layer"]);

            if (img_in) {
                img = img_in->forward(ctx, img);
            }

            ggml_tensor* vec;
            ggml_tensor* txt_img_mask = nullptr;
            if (params.is_chroma) {
                int64_t mod_index_length = 344;
                auto approx              = std::dynamic_pointer_cast<ChromaApproximator>(blocks["distilled_guidance_layer"]);
                auto distill_timestep    = ggml_ext_timestep_embedding(ctx->ggml_ctx, timesteps, 16, 10000, 1000.f);
                auto distill_guidance    = ggml_ext_timestep_embedding(ctx->ggml_ctx, guidance, 16, 10000, 1000.f);

                // auto mod_index_arange  = ggml_arange(ctx, 0, (float)mod_index_length, 1);
                // ggml_arange tot working on a lot of backends, precomputing it on CPU instead
                GGML_ASSERT(mod_index_arange != nullptr);
                auto modulation_index = ggml_ext_timestep_embedding(ctx->ggml_ctx, mod_index_arange, 32, 10000, 1000.f);  // [1, 344, 32]

                // Batch broadcast (will it ever be useful)
                modulation_index = ggml_repeat(ctx->ggml_ctx, modulation_index, ggml_new_tensor_3d(ctx->ggml_ctx, GGML_TYPE_F32, modulation_index->ne[0], modulation_index->ne[1], img->ne[2]));  // [N, 344, 32]

                auto timestep_guidance = ggml_concat(ctx->ggml_ctx, distill_timestep, distill_guidance, 0);  // [N, 1, 32]
                timestep_guidance      = ggml_repeat(ctx->ggml_ctx, timestep_guidance, modulation_index);    // [N, 344, 32]

                vec = ggml_concat(ctx->ggml_ctx, timestep_guidance, modulation_index, 0);  // [N, 344, 64]
                // Permute for consistency with non-distilled modulation implementation
                vec = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, vec, 0, 2, 1, 3));  // [344, N, 64]
                vec = approx->forward(ctx, vec);                                               // [344, N, hidden_size]

                if (y != nullptr) {
                    txt_img_mask = ggml_pad(ctx->ggml_ctx, y, static_cast<int>(img->ne[1]), 0, 0, 0);
                }
            } else {
                auto time_in = std::dynamic_pointer_cast<MLPEmbedder>(blocks["time_in"]);
                vec          = time_in->forward(ctx, ggml_ext_timestep_embedding(ctx->ggml_ctx, timesteps, 256, 10000, 1000.f));
                if (params.guidance_embed) {
                    GGML_ASSERT(guidance != nullptr);
                    auto guidance_in = std::dynamic_pointer_cast<MLPEmbedder>(blocks["guidance_in"]);
                    // bf16 and fp16 result is different
                    auto g_in = ggml_ext_timestep_embedding(ctx->ggml_ctx, guidance, 256, 10000, 1000.f);
                    vec       = ggml_add(ctx->ggml_ctx, vec, guidance_in->forward(ctx, g_in));
                }

                if (params.vec_in_dim > 0) {
                    auto vector_in = std::dynamic_pointer_cast<MLPEmbedder>(blocks["vector_in"]);
                    vec            = ggml_add(ctx->ggml_ctx, vec, vector_in->forward(ctx, y));
                }
            }

            std::vector<ModulationOut> ds_img_mods;
            std::vector<ModulationOut> ds_txt_mods;
            std::vector<ModulationOut> ss_mods;
            if (params.share_modulation) {
                auto double_stream_modulation_img = std::dynamic_pointer_cast<Modulation>(blocks["double_stream_modulation_img"]);
                auto double_stream_modulation_txt = std::dynamic_pointer_cast<Modulation>(blocks["double_stream_modulation_txt"]);
                auto single_stream_modulation     = std::dynamic_pointer_cast<Modulation>(blocks["single_stream_modulation"]);

                ds_img_mods = double_stream_modulation_img->forward(ctx, vec);
                ds_txt_mods = double_stream_modulation_txt->forward(ctx, vec);
                ss_mods     = single_stream_modulation->forward(ctx, vec);
            }

            if (params.semantic_txt_norm) {
                auto semantic_txt_norm = std::dynamic_pointer_cast<RMSNorm>(blocks["txt_norm"]);

                txt = semantic_txt_norm->forward(ctx, txt);
            }

            txt = txt_in->forward(ctx, txt);
            sd::ggml_graph_cut::mark_graph_cut(img, "flux.prelude", "img");
            sd::ggml_graph_cut::mark_graph_cut(txt, "flux.prelude", "txt");
            sd::ggml_graph_cut::mark_graph_cut(vec, "flux.prelude", "vec");

            for (int i = 0; i < params.depth; i++) {
                if (skip_layers.size() > 0 && std::find(skip_layers.begin(), skip_layers.end(), i) != skip_layers.end()) {
                    continue;
                }

                auto block = std::dynamic_pointer_cast<DoubleStreamBlock>(blocks["double_blocks." + std::to_string(i)]);

                auto img_txt = block->forward(ctx, img, txt, vec, pe, txt_img_mask,
                                              ds_img_mods, ds_txt_mods, packed_pe);
                img          = img_txt.first;   // [N, n_img_token, hidden_size]
                txt          = img_txt.second;  // [N, n_txt_token, hidden_size]
                sd::ggml_graph_cut::mark_graph_cut(img, "flux.double_blocks." + std::to_string(i), "img");
                sd::ggml_graph_cut::mark_graph_cut(txt, "flux.double_blocks." + std::to_string(i), "txt");
            }

            const bool image_first = packed_pe != nullptr;
            auto txt_img = image_first ? ggml_concat(ctx->ggml_ctx, img, txt, 1) :
                                         ggml_concat(ctx->ggml_ctx, txt, img, 1);
            for (int i = 0; i < params.depth_single_blocks; i++) {
                if (skip_layers.size() > 0 && std::find(skip_layers.begin(), skip_layers.end(), i + params.depth) != skip_layers.end()) {
                    continue;
                }
                auto block = std::dynamic_pointer_cast<SingleStreamBlock>(blocks["single_blocks." + std::to_string(i)]);

                txt_img = block->forward(ctx, txt_img, vec, pe, txt_img_mask, ss_mods, packed_pe);
                sd::ggml_graph_cut::mark_graph_cut(txt_img, "flux.single_blocks." + std::to_string(i), "txt_img");
            }

            img = ggml_view_3d(ctx->ggml_ctx,
                               txt_img,
                               txt_img->ne[0],
                               img->ne[1],
                               txt_img->ne[2],
                               txt_img->nb[1],
                               txt_img->nb[2],
                               image_first ? 0 : txt->ne[1] * txt_img->nb[1]);  // [N, n_img_token, hidden_size]

            if (final_layer) {
                img = final_layer->forward(ctx, img, vec);  // (N, T, patch_size ** 2 * out_channels)
            }

            return img;
        }

        ggml_tensor* _apply_x0_residual(GGMLRunnerContext* ctx,
                                        ggml_tensor* predicted,
                                        ggml_tensor* noisy,
                                        ggml_tensor* timesteps) {
            auto x = ggml_sub(ctx->ggml_ctx, noisy, predicted);
            x      = ggml_div(ctx->ggml_ctx, x, timesteps);
            return x;
        }

        ggml_tensor* forward_chroma_radiance(GGMLRunnerContext* ctx,
                                             ggml_tensor* x,
                                             ggml_tensor* timestep,
                                             ggml_tensor* context,
                                             ggml_tensor* c_concat,
                                             ggml_tensor* y,
                                             ggml_tensor* guidance,
                                             ggml_tensor* pe,
                                             ggml_tensor* mod_index_arange         = nullptr,
                                             ggml_tensor* dct                      = nullptr,
                                             std::vector<ggml_tensor*> ref_latents = {},
                                             std::vector<int> skip_layers          = {}) {
            GGML_ASSERT(x->ne[3] == 1);

            int64_t W      = x->ne[0];
            int64_t H      = x->ne[1];
            int64_t C      = x->ne[2];
            int patch_size = params.patch_size;
            int pad_h      = (patch_size - H % patch_size) % patch_size;
            int pad_w      = (patch_size - W % patch_size) % patch_size;

            auto img      = DiT::pad_to_patch_size(ctx, x, params.patch_size, params.patch_size);
            auto orig_img = img;

            if (params.chroma_radiance_params.fake_patch_size_x2) {
                // It's supposed to be using GGML_SCALE_MODE_NEAREST, but this seems more stable
                // Maybe the implementation of nearest-neighbor interpolation in ggml behaves differently than the one in PyTorch?
                // img = F.interpolate(img, size=(H//2, W//2), mode="nearest")
                img = ggml_interpolate(ctx->ggml_ctx, img, W / 2, H / 2, C, x->ne[3], GGML_SCALE_MODE_BILINEAR);
            }

            auto img_in_patch = std::dynamic_pointer_cast<Conv2d>(blocks["img_in_patch"]);

            img = img_in_patch->forward(ctx, img);                                                       // [N, hidden_size, H/patch_size, W/patch_size]
            img = ggml_reshape_3d(ctx->ggml_ctx, img, img->ne[0] * img->ne[1], img->ne[2], img->ne[3]);  // [N, hidden_size, H/patch_size*W/patch_size]
            img = ggml_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, img, 1, 0, 2, 3));      // [N, H/patch_size*W/patch_size, hidden_size]

            auto out = forward_orig(ctx, img, context, timestep, y, guidance, pe, mod_index_arange, skip_layers);  // [N, n_img_token, hidden_size]

            // nerf decode
            auto nerf_image_embedder   = std::dynamic_pointer_cast<NerfEmbedder>(blocks["nerf_image_embedder"]);
            auto nerf_final_layer_conv = std::dynamic_pointer_cast<NerfFinalLayerConv>(blocks["nerf_final_layer_conv"]);

            auto nerf_pixels    = DiT::patchify(ctx->ggml_ctx, orig_img, patch_size, patch_size);  // [N, num_patches, C * patch_size * patch_size]
            int64_t num_patches = nerf_pixels->ne[1];
            nerf_pixels         = ggml_reshape_3d(ctx->ggml_ctx,
                                                  nerf_pixels,
                                                  nerf_pixels->ne[0] / C,
                                                  C,
                                                  nerf_pixels->ne[1] * nerf_pixels->ne[2]);                                  // [N*num_patches, C, patch_size*patch_size]
            nerf_pixels         = ggml_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, nerf_pixels, 1, 0, 2, 3));  // [N*num_patches, patch_size*patch_size, C]

            auto nerf_hidden = ggml_reshape_2d(ctx->ggml_ctx, out, out->ne[0], out->ne[1] * out->ne[2]);  // [N*num_patches, hidden_size]
            auto img_dct     = nerf_image_embedder->forward(ctx, nerf_pixels, dct);                       // [N*num_patches, patch_size*patch_size, nerf_hidden_size]

            for (int i = 0; i < params.chroma_radiance_params.nerf_depth; i++) {
                auto block = std::dynamic_pointer_cast<NerfGLUBlock>(blocks["nerf_blocks." + std::to_string(i)]);

                img_dct = block->forward(ctx, img_dct, nerf_hidden);
            }

            img_dct = ggml_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, img_dct, 1, 0, 2, 3));                                 // [N*num_patches, nerf_hidden_size, patch_size*patch_size]
            img_dct = ggml_reshape_3d(ctx->ggml_ctx, img_dct, img_dct->ne[0] * img_dct->ne[1], num_patches, img_dct->ne[2] / num_patches);  // [N, num_patches, nerf_hidden_size*patch_size*patch_size]
            img_dct = DiT::unpatchify(ctx->ggml_ctx, img_dct, (H + pad_h) / patch_size, (W + pad_w) / patch_size, patch_size, patch_size);  // [N, nerf_hidden_size, H, W]

            out = nerf_final_layer_conv->forward(ctx, img_dct);  // [N, C, H, W]

            if (params.chroma_radiance_params.use_x0) {
                out = _apply_x0_residual(ctx, out, orig_img, timestep);
            }

            return out;
        }

        ggml_tensor* forward_flux_chroma(GGMLRunnerContext* ctx,
                                         ggml_tensor* x,
                                         ggml_tensor* timestep,
                                         ggml_tensor* context,
                                         ggml_tensor* c_concat,
                                         ggml_tensor* y,
                                         ggml_tensor* guidance,
                                         ggml_tensor* pe,
                                         ggml_tensor* mod_index_arange         = nullptr,
                                         ggml_tensor* dct                      = nullptr,
                                         std::vector<ggml_tensor*> ref_latents = {},
                                         std::vector<int> skip_layers          = {},
                                         ggml_tensor* packed_pe                = nullptr) {
            GGML_ASSERT(x->ne[3] == 1);

            int64_t W      = x->ne[0];
            int64_t H      = x->ne[1];
            int64_t C      = x->ne[2];
            int patch_size = params.patch_size;
            int pad_h      = (patch_size - H % patch_size) % patch_size;
            int pad_w      = (patch_size - W % patch_size) % patch_size;

            auto img           = DiT::pad_and_patchify(ctx, x, patch_size, patch_size);
            int64_t img_tokens = img->ne[1];

            if (params.version == VERSION_FLUX_FILL) {
                GGML_ASSERT(c_concat != nullptr);
                ggml_tensor* masked = ggml_view_4d(ctx->ggml_ctx, c_concat, c_concat->ne[0], c_concat->ne[1], C, 1, c_concat->nb[1], c_concat->nb[2], c_concat->nb[3], 0);
                ggml_tensor* mask   = ggml_view_4d(ctx->ggml_ctx, c_concat, c_concat->ne[0], c_concat->ne[1], 8 * 8, 1, c_concat->nb[1], c_concat->nb[2], c_concat->nb[3], c_concat->nb[2] * C);

                masked = DiT::pad_and_patchify(ctx, masked, patch_size, patch_size);
                mask   = DiT::pad_and_patchify(ctx, mask, patch_size, patch_size);

                img = ggml_concat(ctx->ggml_ctx, img, ggml_concat(ctx->ggml_ctx, masked, mask, 0), 0);
            } else if (params.version == VERSION_FLEX_2) {
                GGML_ASSERT(c_concat != nullptr);
                ggml_tensor* masked  = ggml_view_4d(ctx->ggml_ctx, c_concat, c_concat->ne[0], c_concat->ne[1], C, 1, c_concat->nb[1], c_concat->nb[2], c_concat->nb[3], 0);
                ggml_tensor* mask    = ggml_view_4d(ctx->ggml_ctx, c_concat, c_concat->ne[0], c_concat->ne[1], 1, 1, c_concat->nb[1], c_concat->nb[2], c_concat->nb[3], c_concat->nb[2] * C);
                ggml_tensor* control = ggml_view_4d(ctx->ggml_ctx, c_concat, c_concat->ne[0], c_concat->ne[1], C, 1, c_concat->nb[1], c_concat->nb[2], c_concat->nb[3], c_concat->nb[2] * (C + 1));

                masked  = DiT::pad_and_patchify(ctx, masked, patch_size, patch_size);
                mask    = DiT::pad_and_patchify(ctx, mask, patch_size, patch_size);
                control = DiT::pad_and_patchify(ctx, control, patch_size, patch_size);

                img = ggml_concat(ctx->ggml_ctx, img, ggml_concat(ctx->ggml_ctx, ggml_concat(ctx->ggml_ctx, masked, mask, 0), control, 0), 0);
            } else if (params.version == VERSION_FLUX_CONTROLS) {
                GGML_ASSERT(c_concat != nullptr);

                auto control = DiT::pad_and_patchify(ctx, c_concat, patch_size, patch_size);
                img          = ggml_concat(ctx->ggml_ctx, img, control, 0);
            }

            if (ref_latents.size() > 0) {
                for (ggml_tensor* ref : ref_latents) {
                    ref = DiT::pad_and_patchify(ctx, ref, patch_size, patch_size);
                    img = ggml_concat(ctx->ggml_ctx, img, ref, 1);
                }
            }

            const int previous_token_h = ctx->block_sparse_token_h;
            const int previous_token_w = ctx->block_sparse_token_w;
            ctx->block_sparse_token_h = (H + pad_h) / patch_size;
            ctx->block_sparse_token_w = (W + pad_w) / patch_size;
            auto out = forward_orig(ctx, img, context, timestep, y, guidance, pe,
                                    mod_index_arange, skip_layers, packed_pe);
            ctx->block_sparse_token_h = previous_token_h;
            ctx->block_sparse_token_w = previous_token_w;

            if (out->ne[1] > img_tokens) {
                out = ggml_view_3d(ctx->ggml_ctx, out, out->ne[0], img_tokens, out->ne[2], out->nb[1], out->nb[2], 0);
                out = ggml_cont(ctx->ggml_ctx, out);
            }

            out = DiT::unpatchify_and_crop(ctx->ggml_ctx, out, H, W, patch_size, patch_size);  // [N, C, H, W]
            return out;
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* timestep,
                             ggml_tensor* context,
                             ggml_tensor* c_concat,
                             ggml_tensor* y,
                             ggml_tensor* guidance,
                             ggml_tensor* pe,
                             ggml_tensor* mod_index_arange         = nullptr,
                             ggml_tensor* dct                      = nullptr,
                             std::vector<ggml_tensor*> ref_latents = {},
                             std::vector<int> skip_layers          = {},
                             ggml_tensor* packed_pe                = nullptr) {
            // Forward pass of DiT.
            // x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
            // timestep: (N,) tensor of diffusion timesteps
            // context: (N, L, D)
            // c_concat: nullptr, or for (N,C+M, H, W) for Fill
            // y: (N, adm_in_channels) tensor of class labels
            // guidance: (N,)
            // pe: (L, d_head/2, 2, 2)
            // return: (N, C, H, W)

            if (params.version == VERSION_CHROMA_RADIANCE) {
                return forward_chroma_radiance(ctx,
                                               x,
                                               timestep,
                                               context,
                                               c_concat,
                                               y,
                                               guidance,
                                               pe,
                                               mod_index_arange,
                                               dct,
                                               ref_latents,
                                               skip_layers);
            } else {
                return forward_flux_chroma(ctx,
                                           x,
                                           timestep,
                                           context,
                                           c_concat,
                                           y,
                                           guidance,
                                           pe,
                                           mod_index_arange,
                                           dct,
                                           ref_latents,
                                           skip_layers,
                                           packed_pe);
            }
        }
    };

    struct FluxRunner : public GGMLRunner {
    public:
        FluxParams flux_params;
        Flux flux;
        std::vector<float> pe_vec;
        std::vector<float> packed_pe_vec;
        std::vector<float> mod_index_arange_vec;
        std::vector<float> dct_vec;
        sd::Tensor<float> guidance_tensor;
        SDVersion version;
        bool use_mask = false;

        FluxRunner(ggml_backend_t backend,
                   ggml_backend_t params_backend,
                   const String2TensorStorage& tensor_storage_map = {},
                   const std::string prefix                       = "",
                   SDVersion version                              = VERSION_FLUX,
                   bool use_mask                                  = false)
            : GGMLRunner(backend, params_backend), version(version), use_mask(use_mask) {
            flux_params.version             = version;
            flux_params.guidance_embed      = false;
            flux_params.depth               = 0;
            flux_params.depth_single_blocks = 0;
            if (version == VERSION_FLUX_FILL) {
                flux_params.in_channels = 384;
            } else if (version == VERSION_FLUX_CONTROLS) {
                flux_params.in_channels = 128;
            } else if (version == VERSION_FLEX_2) {
                flux_params.in_channels = 196;
            } else if (version == VERSION_CHROMA_RADIANCE) {
                flux_params.in_channels = 3;
                flux_params.patch_size  = 16;
            } else if (version == VERSION_OVIS_IMAGE) {
                flux_params.semantic_txt_norm = true;
                flux_params.use_yak_mlp       = true;
                flux_params.vec_in_dim        = 0;
            } else if (sd_version_is_flux2(version)) {
                flux_params.in_channels      = 128;
                flux_params.patch_size       = 1;
                flux_params.out_channels     = 128;
                flux_params.mlp_ratio        = 3.f;
                flux_params.theta            = 2000;
                flux_params.axes_dim         = {32, 32, 32, 32};
                flux_params.vec_in_dim       = 0;
                flux_params.qkv_bias         = false;
                flux_params.disable_bias     = true;
                flux_params.share_modulation = true;
                flux_params.ref_index_scale  = 10.f;
                flux_params.use_mlp_silu_act = true;
            } else if (sd_version_is_longcat(version)) {
                flux_params.context_in_dim = 3584;
                flux_params.vec_in_dim     = 0;
            }
            int64_t head_dim                   = 0;
            int64_t actual_radiance_patch_size = -1;
            for (auto pair : tensor_storage_map) {
                std::string tensor_name = pair.first;
                if (!starts_with(tensor_name, prefix))
                    continue;
                if (tensor_name.find("guidance_in.in_layer.weight") != std::string::npos) {
                    flux_params.guidance_embed = true;
                }
                if (tensor_name.find("__x0__") != std::string::npos) {
                    LOG_DEBUG("using x0 prediction");
                    flux_params.chroma_radiance_params.use_x0 = true;
                }
                if (tensor_name.find("__32x32__") != std::string::npos) {
                    LOG_DEBUG("using patch size 32");
                    flux_params.patch_size = 32;
                }
                if (tensor_name.find("img_in_patch.weight") != std::string::npos) {
                    actual_radiance_patch_size = pair.second.ne[0];
                    LOG_DEBUG("actual radiance patch size: %d", actual_radiance_patch_size);
                }
                if (tensor_name.find("distilled_guidance_layer.in_proj.weight") != std::string::npos) {
                    // Chroma
                    flux_params.is_chroma = true;
                }
                size_t db = tensor_name.find("double_blocks.");
                if (db != std::string::npos) {
                    tensor_name     = tensor_name.substr(db);  // remove prefix
                    int block_depth = atoi(tensor_name.substr(14, tensor_name.find(".", 14)).c_str());
                    if (block_depth + 1 > flux_params.depth) {
                        flux_params.depth = block_depth + 1;
                    }
                }
                size_t sb = tensor_name.find("single_blocks.");
                if (sb != std::string::npos) {
                    tensor_name     = tensor_name.substr(sb);  // remove prefix
                    int block_depth = atoi(tensor_name.substr(14, tensor_name.find(".", 14)).c_str());
                    if (block_depth + 1 > flux_params.depth_single_blocks) {
                        flux_params.depth_single_blocks = block_depth + 1;
                    }
                }
                if (ends_with(tensor_name, "txt_in.weight")) {
                    flux_params.context_in_dim = pair.second.ne[0];
                    flux_params.hidden_size    = pair.second.ne[1];
                }
                if (ends_with(tensor_name, "single_blocks.0.norm.key_norm.scale")) {
                    head_dim = pair.second.ne[0];
                }
                if (ends_with(tensor_name, "double_blocks.0.txt_attn.norm.key_norm.scale")) {
                    head_dim = pair.second.ne[0];
                }
            }
            if (actual_radiance_patch_size > 0 && actual_radiance_patch_size != flux_params.patch_size) {
                GGML_ASSERT(flux_params.patch_size == 2 * actual_radiance_patch_size);
                LOG_DEBUG("using fake x2 patch size");
                flux_params.chroma_radiance_params.fake_patch_size_x2 = true;
            }

            flux_params.num_heads = static_cast<int>(flux_params.hidden_size / head_dim);

            LOG_INFO("flux: depth = %d, depth_single_blocks = %d, guidance_embed = %s, context_in_dim = %" PRId64
                     ", hidden_size = %" PRId64 ", num_heads = %d",
                     flux_params.depth,
                     flux_params.depth_single_blocks,
                     flux_params.guidance_embed ? "true" : "false",
                     flux_params.context_in_dim,
                     flux_params.hidden_size,
                     flux_params.num_heads);
            if (flux_params.is_chroma) {
                LOG_INFO("Using pruned modulation (Chroma)");
            }

            flux = Flux(flux_params);
            flux.init(params_ctx, tensor_storage_map, prefix);
        }

        std::string get_desc() override {
            return "flux";
        }

        void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors, const std::string prefix) {
            flux.get_param_tensors(tensors, prefix);
        }

        std::vector<float> fetch_dct_pos(int patch_size, int max_freqs) {
            const float PI = 3.14159265358979323846f;

            std::vector<float> pos(patch_size);
            for (int i = 0; i < patch_size; ++i) {
                pos[i] = static_cast<float>(i) / static_cast<float>(patch_size - 1);
            }

            std::vector<float> pos_x(patch_size * patch_size);
            std::vector<float> pos_y(patch_size * patch_size);
            for (int i = 0; i < patch_size; ++i) {
                for (int j = 0; j < patch_size; ++j) {
                    pos_x[i * patch_size + j] = pos[j];
                    pos_y[i * patch_size + j] = pos[i];
                }
            }

            std::vector<float> freqs(max_freqs);
            for (int i = 0; i < max_freqs; ++i) {
                freqs[i] = static_cast<float>(i);
            }

            std::vector<float> coeffs(max_freqs * max_freqs);
            for (int fx = 0; fx < max_freqs; ++fx) {
                for (int fy = 0; fy < max_freqs; ++fy) {
                    coeffs[fx * max_freqs + fy] = 1.0f / (1.0f + freqs[fx] * freqs[fy]);
                }
            }

            int num_positions = patch_size * patch_size;
            int num_features  = max_freqs * max_freqs;
            std::vector<float> dct(num_positions * num_features);

            for (int p = 0; p < num_positions; ++p) {
                float px = pos_x[p];
                float py = pos_y[p];

                for (int fx = 0; fx < max_freqs; ++fx) {
                    float cx = std::cos(px * freqs[fx] * PI);
                    for (int fy = 0; fy < max_freqs; ++fy) {
                        float cy                                      = std::cos(py * freqs[fy] * PI);
                        float val                                     = cx * cy * coeffs[fx * max_freqs + fy];
                        dct[p * num_features + (fx * max_freqs + fy)] = val;
                    }
                }
            }

            return dct;
        }

        ggml_cgraph* build_graph(const sd::Tensor<float>& x_tensor,
                                 const sd::Tensor<float>& timesteps_tensor,
                                 const sd::Tensor<float>& context_tensor                  = {},
                                 const sd::Tensor<float>& c_concat_tensor                 = {},
                                 const sd::Tensor<float>& y_tensor                        = {},
                                 const sd::Tensor<float>& guidance_tensor                 = {},
                                 const std::vector<sd::Tensor<float>>& ref_latents_tensor = {},
                                 bool increase_ref_index                                  = false,
                                 std::vector<int> skip_layers                             = {},
                                 int block_sparse_residual_mode = SD_HTP_BLOCK_SPARSE_RESIDUAL_OFF,
                                 float block_sparse_residual_scale = 1.0f,
                                 float block_sparse_residual_rho = 1.0f,
                                 bool block_sparse_residual_exact_rho = false,
                                 int block_sparse_residual_branch_index = 0) {
            ggml_tensor* x         = make_input(x_tensor);
            ggml_tensor* timesteps = make_input(timesteps_tensor);
            ggml_tensor* context   = make_optional_input(context_tensor);
            ggml_tensor* c_concat  = make_optional_input(c_concat_tensor);
            ggml_tensor* y         = make_optional_input(y_tensor);
            if (flux_params.guidance_embed || flux_params.is_chroma) {
                if (!guidance_tensor.empty()) {
                    this->guidance_tensor = guidance_tensor;
                    if (flux_params.is_chroma) {
                        this->guidance_tensor.fill_(0.f);
                    }
                }
            }
            ggml_tensor* guidance = make_optional_input(this->guidance_tensor);
            std::vector<ggml_tensor*> ref_latents;
            ref_latents.reserve(ref_latents_tensor.size());
            for (const auto& ref_latent_tensor : ref_latents_tensor) {
                ref_latents.push_back(make_input(ref_latent_tensor));
            }

            GGML_ASSERT(x->ne[3] == 1);
            ggml_cgraph* gf = new_graph_custom(FLUX_GRAPH_SIZE);

            ggml_tensor* mod_index_arange = nullptr;
            ggml_tensor* dct              = nullptr;  // for chroma radiance

            if (flux_params.is_chroma) {
                if (!use_mask) {
                    y = nullptr;
                }

                // ggml_arange is not working on some backends, precompute it
                mod_index_arange_vec = arange(0, 344);
                mod_index_arange     = ggml_new_tensor_1d(compute_ctx, GGML_TYPE_F32, mod_index_arange_vec.size());
                set_backend_tensor_data(mod_index_arange, mod_index_arange_vec.data());
            }
            std::set<int> txt_arange_dims;
            if (sd_version_is_flux2(version)) {
                txt_arange_dims    = {3};
                increase_ref_index = true;
            } else if (version == VERSION_OVIS_IMAGE) {
                txt_arange_dims = {1, 2};
            }
            pe_vec      = Rope::gen_flux_pe(static_cast<int>(x->ne[1]),
                                            static_cast<int>(x->ne[0]),
                                            flux_params.patch_size,
                                            static_cast<int>(x->ne[3]),
                                            static_cast<int>(context->ne[1]),
                                            txt_arange_dims,
                                            ref_latents,
                                            increase_ref_index,
                                            flux_params.ref_index_scale,
                                            flux_params.theta,
                                            circular_y_enabled,
                                            circular_x_enabled,
                                            flux_params.axes_dim,
                                            sd_version_is_longcat(version));
            int pos_len = static_cast<int>(pe_vec.size() / flux_params.axes_dim_sum / 2);
            // LOG_DEBUG("pos_len %d", pos_len);
            auto pe = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32, 2, 2, flux_params.axes_dim_sum / 2, pos_len);
            // pe->data = pe_vec.data();
            // print_ggml_tensor(pe);
            // pe->data = nullptr;
            set_backend_tensor_data(pe, pe_vec.data());

            ggml_tensor* packed_pe = nullptr;
            const bool flux2_sparse_graph = sd_htp_patch_major_layout_enabled(
                block_sparse_attn_enabled, block_sparse_residual_mode);
            const bool packed_flux2_requested = sd_version_is_flux2(version) &&
                                                sd_env_flag_enabled("SD_HTP_PACKED_FLUX2") &&
                                                flux2_sparse_graph && ref_latents.empty() &&
                                                block_sparse_residual_mode != SD_HTP_BLOCK_SPARSE_RESIDUAL_ANCHOR &&
                                                !circular_y_enabled && !circular_x_enabled;
            const bool flux2_sparse_pe_requested = flux2_sparse_pe_required(
                sd_version_is_flux2(version),
                flux2_sparse_graph,
                block_sparse_residual_mode == SD_HTP_BLOCK_SPARSE_RESIDUAL_ANCHOR,
                packed_flux2_requested,
                !ref_latents.empty(),
                circular_y_enabled || circular_x_enabled);
            if (sd_env_flag_enabled("SD_HTP_PACKED_FLUX2_TRACE")) {
                LOG_INFO("flux2_packed_request enabled=%d pe=%d version=%d sparse=%d refs=%zu circular=%d/%d use_mask=%d",
                         packed_flux2_requested ? 1 : 0,
                         flux2_sparse_pe_requested ? 1 : 0,
                         static_cast<int>(version),
                         block_sparse_attn_enabled ? 1 : 0, ref_latents.size(),
                         circular_y_enabled ? 1 : 0, circular_x_enabled ? 1 : 0, use_mask ? 1 : 0);
            }
            if (flux2_sparse_pe_requested) {
                packed_pe_vec = Rope::gen_flux_pe(static_cast<int>(x->ne[1]),
                                                   static_cast<int>(x->ne[0]),
                                                   flux_params.patch_size,
                                                   static_cast<int>(x->ne[3]),
                                                   static_cast<int>(context->ne[1]),
                                                   txt_arange_dims,
                                                   ref_latents,
                                                   increase_ref_index,
                                                   flux_params.ref_index_scale,
                                                   flux_params.theta,
                                                   false,
                                                   false,
                                                   flux_params.axes_dim,
                                                   false,
                                                   true);
                int packed_pos_len = static_cast<int>(packed_pe_vec.size() / flux_params.axes_dim_sum / 2);
                packed_pe = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32, 2, 2,
                                               flux_params.axes_dim_sum / 2, packed_pos_len);
                set_backend_tensor_data(packed_pe, packed_pe_vec.data());
            } else {
                packed_pe_vec.clear();
            }

            if (version == VERSION_CHROMA_RADIANCE) {
                int patch_size     = flux_params.patch_size;
                int nerf_max_freqs = flux_params.chroma_radiance_params.nerf_max_freqs;
                dct_vec            = fetch_dct_pos(patch_size, nerf_max_freqs);
                dct                = ggml_new_tensor_2d(compute_ctx, GGML_TYPE_F32, nerf_max_freqs * nerf_max_freqs, patch_size * patch_size);
                // dct->data = dct_vec.data();
                // print_ggml_tensor(dct);
                // dct->data = nullptr;
                set_backend_tensor_data(dct, dct_vec.data());
            }

            auto runner_ctx = get_context();
            sd_install_local_residual_graph_expansion(
                gf, block_sparse_residual_mode, &runner_ctx);
            runner_ctx.block_sparse_residual_mode = block_sparse_residual_mode;
            runner_ctx.block_sparse_residual_scale = block_sparse_residual_scale;
            runner_ctx.block_sparse_residual_rho = block_sparse_residual_rho;
            runner_ctx.block_sparse_residual_exact_rho = block_sparse_residual_exact_rho;
            runner_ctx.block_sparse_residual_branch_index = block_sparse_residual_branch_index;
            runner_ctx.block_sparse_residual_attention_index = 0;

            ggml_tensor* out = flux.forward(&runner_ctx,
                                            x,
                                            timesteps,
                                            context,
                                            c_concat,
                                            y,
                                            guidance,
                                            pe,
                                            mod_index_arange,
                                            dct,
                                            ref_latents,
                                            skip_layers,
                                            packed_pe);

            ggml_build_forward_expand(gf, out);

            return gf;
        }

        sd::Tensor<float> compute(int n_threads,
                                  const sd::Tensor<float>& x,
                                  const sd::Tensor<float>& timesteps,
                                  const sd::Tensor<float>& context                  = {},
                                  const sd::Tensor<float>& c_concat                 = {},
                                  const sd::Tensor<float>& y                        = {},
                                  const sd::Tensor<float>& guidance                 = {},
                                  const std::vector<sd::Tensor<float>>& ref_latents = {},
                                  bool increase_ref_index                           = false,
                                  std::vector<int> skip_layers                      = std::vector<int>(),
                                  int block_sparse_residual_mode = SD_HTP_BLOCK_SPARSE_RESIDUAL_OFF,
                                  float block_sparse_residual_scale = 1.0f,
                                  float block_sparse_residual_rho = 1.0f,
                                  bool block_sparse_residual_exact_rho = false,
                                  int block_sparse_residual_branch_index = 0) {
            // x: [N, in_channels, h, w]
            // timesteps: [N, ]
            // context: [N, max_position, hidden_size]
            // y: [N, adm_in_channels] or [1, adm_in_channels]
            // guidance: [N, ]
            auto get_graph = [&]() -> ggml_cgraph* {
                return build_graph(x, timesteps, context, c_concat, y, guidance, ref_latents,
                                   increase_ref_index, skip_layers,
                                   block_sparse_residual_mode,
                                   block_sparse_residual_scale,
                                   block_sparse_residual_rho,
                                   block_sparse_residual_exact_rho,
                                   block_sparse_residual_branch_index);
            };

            auto result = restore_trailing_singleton_dims(GGMLRunner::compute<float>(get_graph, n_threads, false), x.dim());
            return result;
        }

        void test() {
            ggml_init_params params;
            params.mem_size   = static_cast<size_t>(1024 * 1024) * 1024;  // 1GB
            params.mem_buffer = nullptr;
            params.no_alloc   = false;

            ggml_context* ctx = ggml_init(params);
            GGML_ASSERT(ctx != nullptr);

            {
                // cpu f16:
                // cuda f16: nan
                // cuda q8_0: pass
                sd::Tensor<float> x({16, 16, 128, 1});
                // ggml_set_f32(x, 0.01f);
                // auto x = load_tensor_from_file(ctx, "chroma_x.bin");
                // print_ggml_tensor(x);

                std::vector<float> timesteps_vec(1, 1.f);
                auto timesteps = sd::Tensor<float>::from_vector(timesteps_vec);

                std::vector<float> guidance_vec(1, 0.f);
                auto guidance = sd::Tensor<float>::from_vector(guidance_vec);

                sd::Tensor<float> context({15360, 256, 1});
                // ggml_set_f32(context, 0.01f);
                // auto context = load_tensor_from_file(ctx, "chroma_context.bin");
                // print_ggml_tensor(context);

                // auto y = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 768, 1);
                // ggml_set_f32(y, 0.01f);
                auto y = nullptr;
                // print_ggml_tensor(y);

                sd::Tensor<float> out;

                int64_t t0   = ggml_time_ms();
                auto out_opt = compute(8,
                                       x,
                                       timesteps,
                                       context,
                                       {},
                                       {},
                                       guidance,
                                       {},
                                       false);
                int64_t t1   = ggml_time_ms();

                GGML_ASSERT(!out_opt.empty());
                out = std::move(out_opt);
                print_sd_tensor(out);
                LOG_DEBUG("flux test done in %lldms", t1 - t0);
            }
        }

        static void load_from_file_and_test(const std::string& file_path) {
            // ggml_backend_t backend = ggml_backend_cuda_init(0);
            ggml_backend_t backend    = ggml_backend_cpu_init();
            ggml_type model_data_type = GGML_TYPE_COUNT;

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

            std::shared_ptr<FluxRunner> flux = std::make_shared<FluxRunner>(backend,
                                                                            backend,
                                                                            tensor_storage_map,
                                                                            "model.diffusion_model",
                                                                            VERSION_FLUX2,
                                                                            false);

            flux->alloc_params_buffer();
            std::map<std::string, ggml_tensor*> tensors;
            flux->get_param_tensors(tensors, "model.diffusion_model");

            bool success = model_loader.load_tensors(tensors);

            if (!success) {
                LOG_ERROR("load tensors from model loader failed");
                return;
            }

            LOG_INFO("flux model loaded");
            flux->test();
        }
    };

}  // namespace Flux

#endif  // __FLUX_HPP__
