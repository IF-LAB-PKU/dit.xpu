#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <iostream>
#include <map>
#include <string>

#include "ggml-htp.h"
#include "ggml-cpu.h"
#include "z_image.hpp"

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

void test_packed_rope_reference_convention() {
    ggml_context* ctx = ggml_init({1024 * 1024, nullptr, false});
    require(ctx != nullptr, "RoPE reference context must initialize");
    constexpr int dim = 4, heads = 2, tokens = 3;
    ggml_tensor* x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, dim, heads, tokens, 1);
    ggml_tensor* pe = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 2, 2, dim / 2, tokens);
    float* input = static_cast<float*>(x->data);
    float* matrices = static_cast<float*>(pe->data);
    const auto embeddings = Rope::rope({1.0f, 3.0f, 7.0f}, dim, 10000.0f);
    for (int t = 0; t < tokens; ++t) {
        std::memcpy(matrices + t * dim * 2, embeddings[t].data(), dim * 2 * sizeof(float));
    }
    for (int i = 0; i < dim * heads * tokens; ++i) {
        input[i] = 0.25f * (i + 1);
    }
    ggml_tensor* out = Rope::apply_rope(ctx, x, pe, true);
    ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    require(ggml_graph_compute_with_ctx(ctx, graph, 1) == GGML_STATUS_SUCCESS,
            "RoPE reference graph must compute");
    const float* actual = static_cast<const float*>(out->data);
    for (int h = 0; h < heads; ++h) {
        for (int t = 0; t < tokens; ++t) {
            for (int d = 0; d < dim / 2; ++d) {
                const float* pair = input + (t * heads + h) * dim + 2 * d;
                const float* m = matrices + t * dim * 2 + d * 4;
                const int i = (h * tokens + t) * dim + 2 * d;
                require(std::fabs(actual[i] - (pair[0] * m[0] + pair[1] * m[1])) < 1e-5f &&
                            std::fabs(actual[i + 1] - (pair[0] * m[2] + pair[1] * m[3])) < 1e-5f,
                        "packed RoPE reference must match the unfused graph");
            }
        }
    }
    ggml_free(ctx);
}

void test_z_image_packed_qkv_param_placement() {
    ggml_context* ctx = ggml_init({1024 * 1024, nullptr, true});
    require(ctx != nullptr, "Z-Image parameter context must initialize");
    ZImage::JointAttention attention(3840, 128, 30, 30, true);
    attention.init(ctx, {}, "layers.0.attention");
    std::map<std::string, ggml_tensor*> parameters;
    attention.get_param_tensors(parameters, "layers.0.attention");
    for (const char* name : {"layers.0.attention.q_norm.weight",
                             "layers.0.attention.k_norm.weight"}) {
        ggml_tensor* gamma = parameters.at(name);
        require(gamma != nullptr && gamma->type == GGML_TYPE_F32 &&
                    (gamma->flags & SD_GGML_TENSOR_FLAG_HTP_PACKED_QKV_PARAM) != 0,
                "Z-Image Q/K norm scales must be marked for HTP RPCMEM placement");
    }
    GGMLRunnerContext runner = {};
    runner.ggml_ctx = ctx;
    constexpr int tokens = 288;
    constexpr size_t region_bytes = 30 * tokens * 128 * sizeof(ggml_fp16_t);
    ggml_tensor* storage = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, 3 * region_bytes / sizeof(ggml_fp16_t));
    const auto packed = ZImage::make_packed_z_image_qkv_views(&runner, storage, tokens);
    for (size_t i = 0; i < packed.qkv.size(); ++i) {
        require(ggml_nbytes(packed.qkv[i]) == region_bytes &&
                    packed.qkv[i]->view_offs == i * region_bytes,
                "packed Q/K/V views must cover separate complete regions");
    }
    ggml_free(ctx);
}

void test_z_image_anchor_scale_contract() {
    const char* enabled[] = {"SD_HTP_OPS", "SD_HTP_COMPUTE", "SD_HTP_ATTENTION"};
    const char* disabled[] = {"SD_HTP_SPARSE_OPENCL", "SD_HTP_SPARSE_OPENCL_RESIDUAL",
                              "SD_HTP_BLOCK_SPARSE_FUSED_ANCHOR"};
    std::map<std::string, std::string> saved;
    for (const auto& names : {std::vector<const char*>(std::begin(enabled), std::end(enabled)),
                              std::vector<const char*>(std::begin(disabled), std::end(disabled))}) {
        for (const char* name : names) {
            if (const char* value = std::getenv(name)) saved[name] = value;
        }
    }
    for (const char* name : enabled) setenv(name, "1", 1);
    for (const char* name : disabled) setenv(name, "0", 1);
    ggml_context* ctx = ggml_init({4 * 1024 * 1024, nullptr, true});
    require(ctx != nullptr, "anchor scale context must initialize");
    ggml_tensor* one = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_set_name(one, "ggml_runner_build_in_tensor:one");
    ggml_tensor* q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 128, 64, 30);
    ggml_tensor* k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 128, 64, 30);
    ggml_tensor* v = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 128, 30, 64);
    ggml_cgraph* graph = ggml_new_graph(ctx);
    const std::function<ggml_tensor*(const std::string&, ggml_tensor*)> cache =
        [&](const std::string&, ggml_tensor* tensor) {
            ggml_build_forward_expand(graph, tensor);
            return tensor;
        };
    ggml_tensor* out = ggml_ext_attention_ext(
        ctx, nullptr, q, k, v, 30, nullptr, true, true, 1.0f / 128.0f,
        true, 0, 0, SD_HTP_BLOCK_SPARSE_RESIDUAL_ANCHOR, 1.0f, 1.0f, true,
        "residual", nullptr, "rho", nullptr, &cache, 8, 8, true);
    ggml_build_forward_expand(graph, out);
    int attention_count = 0;
    ggml_tensor* shared_v = nullptr;
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        ggml_tensor* node = ggml_graph_node(graph, i);
        if (node->op != GGML_OP_FLASH_ATTN_EXT) continue;
        ++attention_count;
        float scale;
        std::memcpy(&scale, node->op_params, sizeof(scale));
        require(std::fabs(scale - 1.0f / std::sqrt(128.0f)) < 1e-6f,
                "anchor Q/K compensation must not be applied twice by OpenCL");
        if (shared_v == nullptr) shared_v = node->src[2];
        require(node->src[2] == shared_v, "anchor pair must share its V input");
        ggml_tensor* source = node->src[2];
        while (source != v && source != nullptr) {
            require(source->op != GGML_OP_SCALE,
                    "LSE anchor must expose unscaled V for residual cache and packed apply");
            source = source->src[0];
        }
        require(source == v, "anchor V must originate from the original values");
    }
    require(attention_count == 2, "exact-rho anchor must construct dense and sparse attention");
    ggml_free(ctx);
    for (const auto& names : {std::vector<const char*>(std::begin(enabled), std::end(enabled)),
                              std::vector<const char*>(std::begin(disabled), std::end(disabled))}) {
        for (const char* name : names) {
            const auto previous = saved.find(name);
            if (previous == saved.end()) unsetenv(name);
            else setenv(name, previous->second.c_str(), 1);
        }
    }
}

void test_flash_attention_halo_loader_metadata() {
    const ggml_init_params params = {
        /*.mem_size   =*/ 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context* ctx = ggml_init(params);
    require(ctx != nullptr, "flash attention metadata context must initialize");

    ggml_tensor* q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 16, 4, 2, 1);
    ggml_tensor* k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 16, 4, 2, 1);
    ggml_tensor* v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 16, 2, 4, 1);
    ggml_tensor* attn = ggml_flash_attn_ext_with_lse(ctx, q, k, v, nullptr, 1.0f, 0.0f, 0.0f);
    require(attn != nullptr, "flash attention metadata node must initialize");
    require(ggml_flash_attn_ext_get_halo_kv_load_mode(attn) ==
                GGML_FLASH_ATTN_EXT_HALO_KV_LOAD_INHERIT,
            "new flash attention nodes must inherit the global halo loader");

    ggml_flash_attn_ext_set_halo_kv_load_mode(
        attn, GGML_FLASH_ATTN_EXT_HALO_KV_LOAD_CHUNKED);
    require(ggml_flash_attn_ext_get_halo_kv_load_mode(attn) ==
                GGML_FLASH_ATTN_EXT_HALO_KV_LOAD_CHUNKED,
            "per-node chunked override must round-trip");

    ggml_tensor* q_sparse = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 16, 32, 1, 1);
    ggml_tensor* k_sparse = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 16, 32, 1, 1);
    ggml_tensor* v_sparse = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 16, 1, 32, 1);
    ggml_tensor* mask_sparse = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, 1);
    ggml_tensor* sparse = ggml_flash_attn_ext(ctx, q_sparse, k_sparse, v_sparse,
                                              mask_sparse, 1.0f, 0.0f, 0.0f);
    require(sparse != nullptr, "sparse metadata node must initialize");
    ggml_flash_attn_ext_set_block_sparse(
        sparse, 32, 32, GGML_FLASH_ATTN_EXT_MASK_BLOCK_INDEX);
    ggml_flash_attn_ext_set_static_halo_2d(sparse, 32, 0, 1, 1);
    ggml_flash_attn_ext_set_halo_kv_load_mode(
        sparse, GGML_FLASH_ATTN_EXT_HALO_KV_LOAD_INHERIT);
    sd_htp_apply_sparse_halo_kv_load_override(
        sparse, SD_HTP_ROUTE_HTP_CPU, SD_HTP_BLOCK_SPARSE_RESIDUAL_ANCHOR);
    require(ggml_flash_attn_ext_get_halo_kv_load_mode(sparse) ==
                GGML_FLASH_ATTN_EXT_HALO_KV_LOAD_CHUNKED,
            "HTP CPU static-halo anchor node must receive chunked override");
    ggml_flash_attn_ext_set_halo_kv_load_mode(
        sparse, GGML_FLASH_ATTN_EXT_HALO_KV_LOAD_INHERIT);
    sd_htp_apply_sparse_halo_kv_load_override(
        sparse, SD_HTP_ROUTE_HTP_CPU, SD_HTP_BLOCK_SPARSE_RESIDUAL_APPLY);
    require(ggml_flash_attn_ext_get_halo_kv_load_mode(sparse) ==
                GGML_FLASH_ATTN_EXT_HALO_KV_LOAD_INHERIT,
            "HTP CPU apply node must inherit the global halo loader");
    sd_htp_apply_sparse_halo_kv_load_override(
        sparse, SD_HTP_ROUTE_OPENCL_ANCHOR, SD_HTP_BLOCK_SPARSE_RESIDUAL_ANCHOR);
    require(ggml_flash_attn_ext_get_halo_kv_load_mode(sparse) ==
                GGML_FLASH_ATTN_EXT_HALO_KV_LOAD_INHERIT,
            "OpenCL anchor node must inherit the global halo loader");

    ggml_free(ctx);
}

void test_route_mode_from_flags() {
    struct RouteCase {
        bool fused;
        bool opencl_sparse;
        bool opencl_residual;
        sd_htp_route_mode expected;
    };

    const RouteCase valid_cases[] = {
        {true, false, true, SD_HTP_ROUTE_FUSED_OPENCL_RESIDUAL},
        {false, true, true, SD_HTP_ROUTE_OPENCL_ANCHOR},
        {false, false, false, SD_HTP_ROUTE_HTP_CPU},
    };
    for (const auto& test_case : valid_cases) {
        require(sd_htp_route_mode_from_flags(test_case.fused,
                                             test_case.opencl_sparse,
                                             test_case.opencl_residual) == test_case.expected,
                "valid route flags must select their exact route");
    }

    const RouteCase invalid_cases[] = {
        {true, true, true, SD_HTP_ROUTE_INVALID},
        {true, false, false, SD_HTP_ROUTE_INVALID},
        {false, true, false, SD_HTP_ROUTE_INVALID},
        {false, false, true, SD_HTP_ROUTE_INVALID},
    };
    for (const auto& test_case : invalid_cases) {
        require(sd_htp_route_mode_from_flags(test_case.fused,
                                             test_case.opencl_sparse,
                                             test_case.opencl_residual) == test_case.expected,
                "mixed route ownership flags must be rejected");
    }
}

void test_attention_policy() {
    const auto anchor = sd_htp_attention_policy_for(
        false, SD_HTP_BLOCK_SPARSE_RESIDUAL_ANCHOR, true);
    require(anchor.sparse_policy_active,
            "an anchor must activate sparse geometry without a sparse step");
    require(!anchor.packed_qkv_allowed,
            "an anchor must never use packed QKV");
    require(anchor.residual_eligible,
            "image attention must participate in anchor residual accounting");

    const auto apply = sd_htp_attention_policy_for(
        true, SD_HTP_BLOCK_SPARSE_RESIDUAL_APPLY, true);
    require(apply.sparse_policy_active,
            "a sparse apply step must activate sparse geometry");
    require(apply.packed_qkv_allowed,
            "packed QKV is allowed on a sparse apply step");
    require(apply.residual_eligible,
            "image attention must participate in apply residual accounting");

    const auto sparse_without_residual_cache = sd_htp_attention_policy_for(
        true, SD_HTP_BLOCK_SPARSE_RESIDUAL_OFF, true);
    require(sparse_without_residual_cache.sparse_policy_active,
            "a sparse step without residual cache must activate sparse geometry");
    require(sparse_without_residual_cache.packed_qkv_allowed,
            "packed QKV must be allowed without residual cache on non-anchor sparse steps");
    require(!sparse_without_residual_cache.residual_eligible,
            "residual accounting must remain disabled without residual cache");

    const auto text_only = sd_htp_attention_policy_for(
        true, SD_HTP_BLOCK_SPARSE_RESIDUAL_APPLY, false);
    require(!text_only.residual_eligible,
            "text-only attention must be excluded from residual accounting");
    require(!text_only.packed_qkv_allowed,
            "text-only attention must not enter the residual packed route");
}

void test_packed_residual_apply_layout_policy() {
    require(sd_htp_packed_residual_apply_needs_contiguous(
                true, SD_HTP_BLOCK_SPARSE_RESIDUAL_APPLY),
            "packed residual apply must materialize a contiguous output");
    require(!sd_htp_packed_residual_apply_needs_contiguous(
                 false, SD_HTP_BLOCK_SPARSE_RESIDUAL_APPLY),
            "reordered residual apply must not add a copy");
    require(!sd_htp_packed_residual_apply_needs_contiguous(
                 true, SD_HTP_BLOCK_SPARSE_RESIDUAL_ANCHOR),
            "packed anchor must not use the apply-only copy policy");
    require(sd_htp_packed_residual_apply_uses_lse(
                 true, SD_HTP_BLOCK_SPARSE_RESIDUAL_APPLY),
            "packed residual apply must request the stable LSE writeback path");
    require(sd_htp_packed_residual_apply_uses_lse(
                true, SD_HTP_BLOCK_SPARSE_RESIDUAL_ANCHOR),
            "packed residual anchor must retain its LSE slab");
}

void test_packed_residual_apply_lse_admission() {
    require(sd_htp_sparse_attention_needs_lse(
                true, SD_HTP_BLOCK_SPARSE_RESIDUAL_APPLY, 1.0f, false),
            "packed residual apply must need LSE at unit scale");
    require(sd_htp_sparse_attention_needs_lse(
                true, SD_HTP_BLOCK_SPARSE_RESIDUAL_APPLY, 1.0f / 128.0f, false),
            "packed residual apply must need LSE for kv scaling");
    require(sd_htp_sparse_attention_needs_lse(
                true, SD_HTP_BLOCK_SPARSE_RESIDUAL_ANCHOR, 1.0f / 128.0f, false),
            "packed residual anchor must need LSE");
    require(sd_htp_sparse_attention_needs_lse(
                false, SD_HTP_BLOCK_SPARSE_RESIDUAL_APPLY, 1.0f / 128.0f, false),
            "reordered attention still needs LSE for kv scaling");
    require(sd_htp_sparse_attention_needs_lse(
                true, SD_HTP_BLOCK_SPARSE_RESIDUAL_APPLY, 1.0f, true),
            "exact rho anchor cache must need LSE");
}

void test_anchor_halo_loader_policy() {
    struct PolicyCase {
        sd_htp_route_mode route;
        int residual_mode;
        ggml_flash_attn_ext_halo_kv_load_mode expected;
    };
    const PolicyCase cases[] = {
        {SD_HTP_ROUTE_HTP_CPU, SD_HTP_BLOCK_SPARSE_RESIDUAL_ANCHOR,
         GGML_FLASH_ATTN_EXT_HALO_KV_LOAD_CHUNKED},
        {SD_HTP_ROUTE_HTP_CPU, SD_HTP_BLOCK_SPARSE_RESIDUAL_APPLY,
         GGML_FLASH_ATTN_EXT_HALO_KV_LOAD_INHERIT},
        {SD_HTP_ROUTE_HTP_CPU, SD_HTP_BLOCK_SPARSE_RESIDUAL_OFF,
         GGML_FLASH_ATTN_EXT_HALO_KV_LOAD_INHERIT},
        {SD_HTP_ROUTE_OPENCL_ANCHOR, SD_HTP_BLOCK_SPARSE_RESIDUAL_ANCHOR,
         GGML_FLASH_ATTN_EXT_HALO_KV_LOAD_INHERIT},
        {SD_HTP_ROUTE_FUSED_OPENCL_RESIDUAL, SD_HTP_BLOCK_SPARSE_RESIDUAL_ANCHOR,
         GGML_FLASH_ATTN_EXT_HALO_KV_LOAD_INHERIT},
        {SD_HTP_ROUTE_INVALID, SD_HTP_BLOCK_SPARSE_RESIDUAL_ANCHOR,
         GGML_FLASH_ATTN_EXT_HALO_KV_LOAD_INHERIT},
    };
    for (const auto& test_case : cases) {
        require(sd_htp_sparse_halo_kv_load_override(
                    test_case.route, test_case.residual_mode) == test_case.expected,
                "only HTP CPU residual anchors may force chunked sparse loading");
    }
}

void test_residual_binding_keys() {
    GGMLRunnerContext anchor_ctx;
    anchor_ctx.block_sparse_residual_mode = SD_HTP_BLOCK_SPARSE_RESIDUAL_ANCHOR;
    anchor_ctx.block_sparse_residual_exact_rho = true;
    anchor_ctx.block_sparse_residual_branch_index = 4;

    const auto first = sd_htp_prepare_residual_binding(&anchor_ctx, "joint", true);
    const auto skipped = sd_htp_prepare_residual_binding(&anchor_ctx, "text", false);
    const auto second = sd_htp_prepare_residual_binding(&anchor_ctx, "joint", true);

    require(first.slot == 0, "first eligible attention must use slot zero");
    require(first.residual_key == "htp_block_sparse_residual_b4_joint_s0",
            "residual key must include branch, kind, and slot");
    require(first.rho_key == "htp_block_sparse_residual_rho_b4_joint_s0",
            "rho key must include branch, kind, and slot");
    require(skipped.slot == -1 && skipped.residual_key.empty() && skipped.rho_key.empty(),
            "ineligible attention must not allocate cache keys");
    require(second.slot == 1,
            "ineligible attention must not consume a residual slot");
    require(anchor_ctx.block_sparse_residual_attention_index == 2,
            "only eligible attention must advance the residual slot index");
}

void test_stable_binding_sequence(const char* const* kinds, int kind_count) {
    GGMLRunnerContext anchor_ctx;
    anchor_ctx.block_sparse_residual_mode = SD_HTP_BLOCK_SPARSE_RESIDUAL_ANCHOR;
    anchor_ctx.block_sparse_residual_exact_rho = true;
    anchor_ctx.block_sparse_residual_branch_index = 9;

    GGMLRunnerContext apply_ctx;
    apply_ctx.block_sparse_residual_mode = SD_HTP_BLOCK_SPARSE_RESIDUAL_APPLY;
    apply_ctx.block_sparse_residual_exact_rho = true;
    apply_ctx.block_sparse_residual_branch_index = 9;

    for (int i = 0; i < kind_count; ++i) {
        const auto anchor = sd_htp_prepare_residual_binding(&anchor_ctx, kinds[i], true);
        const auto apply = sd_htp_prepare_residual_binding(&apply_ctx, kinds[i], true);
        require(anchor.slot == i && apply.slot == i,
                "anchor and apply must allocate identical residual slots");
        require(anchor.residual_key == apply.residual_key,
                "anchor and apply residual keys must match");
        require(anchor.rho_key == apply.rho_key,
                "anchor and apply rho keys must match");
    }
}

void test_model_binding_sequences() {
    const char* mmdit_kinds[] = {"self", "self2", "joint"};
    test_stable_binding_sequence(mmdit_kinds, 3);

    const char* flux2_kinds[] = {"dual", "single"};
    test_stable_binding_sequence(flux2_kinds, 2);
}

void test_z_image_route_selection() {
    using ZImage::ZImageAttentionRoute;
    using ZImage::z_image_select_attention_route;

    const auto anchor = z_image_select_attention_route(
        true, true, true, true, true, true, true, false, true);
    require(anchor.route == ZImageAttentionRoute::REORDERED,
            "Z-Image anchor attention must use the non-packed reordered route");

    const auto packed_disabled = z_image_select_attention_route(
        true, true, true, false, true, true, true, false, false);
    require(packed_disabled.route == ZImageAttentionRoute::REORDERED &&
                std::string(packed_disabled.fallback_reason) == "packed_integration_unverified",
            "Z-Image packed request must fall back until full-graph integration is verified");

    const auto packed_apply = z_image_select_attention_route(
        true, true, true, true, true, true, true, false, false);
    require(packed_apply.route == ZImageAttentionRoute::PACKED,
            "Z-Image experimental packed route may use QKV when available");

    const auto reordered_apply = z_image_select_attention_route(
        true, true, true, true, true, true, false, false, false);
    require(reordered_apply.route == ZImageAttentionRoute::REORDERED,
            "Z-Image sparse apply must fall back to reordered non-packed QKV");

    const auto text_only = z_image_select_attention_route(
        true, false, false, false, false, false, false, false, false);
    require(text_only.route == ZImageAttentionRoute::DENSE,
            "Z-Image text-only context refiner must remain dense");

    require(ZImage::z_image_sparse_layout_eligible(256, 16, 16, 11, false),
            "position-index offsets without ref tensors must keep sparse geometry eligible");
    require(!ZImage::z_image_sparse_layout_eligible(256, 16, 16, 11, true),
            "actual appended ref latent tokens must remain excluded");
}

void test_z_image_pe_ordering() {
    const std::vector<ggml_tensor*> no_refs;
    constexpr int h = 32;
    constexpr int w = 32;
    constexpr int patch_size = 2;
    constexpr int token_h = 16;
    constexpr int token_w = 16;
    constexpr int context_len = 11;
    constexpr int seq_multi_of = 32;
    constexpr int text_padded = 32;
    constexpr int image_tokens = token_h * token_w;
    constexpr int sparse_patch = ZImage::Z_IMAGE_SPARSE_PATCH_SIZE;

    const auto text_first = Rope::gen_z_image_ids(
        h, w, patch_size, 1, context_len, seq_multi_of, no_refs, false,
        false, false);
    const auto image_first_row_major = Rope::gen_z_image_ids(
        h, w, patch_size, 1, context_len, seq_multi_of, no_refs, false,
        true, false);
    const auto image_first_patch_major = Rope::gen_z_image_ids(
        h, w, patch_size, 1, context_len, seq_multi_of, no_refs, false,
        true, true);

    require(text_first.size() == image_first_row_major.size() &&
                image_first_row_major.size() == image_first_patch_major.size(),
            "Z-Image PE layouts must have the same padded sequence length");
    for (int i = 0; i < text_padded; ++i) {
        require(text_first[i] == image_first_patch_major[image_tokens + i],
                "text suffix must remain unchanged when image tokens are reordered");
        require(text_first[i] == image_first_row_major[image_tokens + i],
                "image-first row-major text suffix must match text-first IDs");
    }
    for (int i = 0; i < image_tokens; ++i) {
        const int row_major = sd_row_major_index_from_patch_major(
            i, token_h, token_w, sparse_patch);
        require(image_first_patch_major[i] == image_first_row_major[row_major],
                "patch-major image IDs must follow the shared inverse mapping");
    }
    require(image_first_patch_major[0] == image_first_row_major[0] &&
                image_first_patch_major[1] == image_first_row_major[1] &&
                image_first_patch_major[2] == image_first_row_major[2] &&
                image_first_patch_major[3] == image_first_row_major[3] &&
                image_first_patch_major[4] == image_first_row_major[16],
            "patch-major order must differ at the first patch boundary");

    for (int i = 0; i < image_tokens; ++i) {
        const int patch_major = sd_patch_major_index_from_row_major(
            i, token_h, token_w, sparse_patch);
        require(sd_row_major_index_from_patch_major(
                    patch_major, token_h, token_w, sparse_patch) == i,
                "patch-major and row-major mappings must be inverses");
    }

    const auto dense_ids = Rope::gen_z_image_ids(
        h, w, patch_size, 1, context_len, seq_multi_of, no_refs, false,
        false, false);
    require(dense_ids == text_first,
            "explicit dense PE ordering must preserve the existing row-major layout");

    const std::vector<int> axes_dim = {32, 48, 48};
    const int pe_stride = 256;
    const auto row_major_pe = Rope::gen_z_image_pe(
        h, w, patch_size, 1, context_len, seq_multi_of, no_refs, false,
        10000, false, false, axes_dim, true, false);
    const auto patch_major_pe = Rope::gen_z_image_pe(
        h, w, patch_size, 1, context_len, seq_multi_of, no_refs, false,
        10000, false, false, axes_dim, true, true);
    require(row_major_pe.size() == patch_major_pe.size(),
            "patch-major and row-major PE tensors must have the same shape");
    for (int i = 0; i < image_tokens; ++i) {
        const int row_major = sd_row_major_index_from_patch_major(
            i, token_h, token_w, sparse_patch);
        for (int j = 0; j < pe_stride; ++j) {
            require(patch_major_pe[i * pe_stride + j] ==
                        row_major_pe[row_major * pe_stride + j],
                    "patch-major PE values must follow the image ID mapping");
        }
    }
    for (int i = 0; i < text_padded; ++i) {
        for (int j = 0; j < pe_stride; ++j) {
            require(patch_major_pe[(image_tokens + i) * pe_stride + j] ==
                        row_major_pe[(image_tokens + i) * pe_stride + j],
                    "patch-major PE text suffix must remain unchanged");
        }
    }

    constexpr int padded_h = 5;
    constexpr int padded_w = 5;
    constexpr int valid_image_tokens = padded_h * padded_w;
    constexpr int padded_image_tokens = 32;
    const auto padded_row_major = Rope::gen_z_image_ids(
        10, 10, patch_size, 1, context_len, seq_multi_of, no_refs, false,
        true, false);
    const auto padded_patch_major = Rope::gen_z_image_ids(
        10, 10, patch_size, 1, context_len, seq_multi_of, no_refs, false,
        true, true);
    for (int i = 0; i < valid_image_tokens; ++i) {
        const int row_major = sd_row_major_index_from_patch_major(
            i, padded_h, padded_w, sparse_patch);
        require(padded_patch_major[i] == padded_row_major[row_major],
                "valid image IDs must be reordered before image padding");
    }
    for (int i = valid_image_tokens; i < padded_image_tokens; ++i) {
        require(padded_patch_major[i] == std::vector<float>({0.f, 0.f, 0.f}),
                "image padding IDs must remain at the end of the image segment");
    }
}

void test_z_image_binding_sequences() {
    const char* z_image_kinds[] = {"noise", "joint", "joint"};
    test_stable_binding_sequence(z_image_kinds, 3);
}

void set_opencl_lifetime_route_env(const char* mode) {
    setenv("SD_HTP_BLOCK_SPARSE_FUSED_ANCHOR", "0", 1);
    setenv("SD_HTP_BLOCK_SPARSE_RESIDUAL_CACHE", "1", 1);
    setenv("SD_HTP_BLOCK_SPARSE_RESIDUAL_EXACT_RHO", "1", 1);
    setenv("SD_HTP_SPARSE_OPENCL", "1", 1);
    setenv("SD_HTP_SPARSE_OPENCL_RESIDUAL", "1", 1);
    setenv("SD_HTP_SPARSE_OPENCL_ASYNC", "1", 1);
    setenv("SD_HTP_SPARSE_OPENCL_EARLY_SUBMIT", "1", 1);
    setenv("SD_HTP_SPARSE_OPENCL_SUBMIT_THREAD", "1", 1);
    setenv("SD_HTP_SPARSE_OPENCL_DEFER_RESIDUAL", "1", 1);
    if (mode != nullptr) {
        setenv("SD_HTP_SPARSE_OPENCL_LIFETIME_MODE", mode, 1);
    } else {
        unsetenv("SD_HTP_SPARSE_OPENCL_LIFETIME_MODE");
    }
}

void test_opencl_lifetime_mode_parser() {
    struct ModeCase {
        const char* value;
        ggml_htp_opencl_lifetime_mode expected;
    };
    const ModeCase valid[] = {
        {nullptr, GGML_HTP_OPENCL_LIFETIME_DEFERRED},
        {"deferred", GGML_HTP_OPENCL_LIFETIME_DEFERRED},
        {"reuse", GGML_HTP_OPENCL_LIFETIME_REUSE},
        {"strict", GGML_HTP_OPENCL_LIFETIME_STRICT},
    };
    for (const auto& test_case : valid) {
        const auto mode = ggml_htp_opencl_lifetime_mode_parse(test_case.value);
        require(mode == test_case.expected,
                "valid OpenCL lifetime modes must parse exactly");
        require(std::string(ggml_htp_opencl_lifetime_mode_name(mode)) ==
                    (test_case.value == nullptr ? "deferred" : test_case.value),
                "OpenCL lifetime mode names must round-trip");
    }
    require(ggml_htp_opencl_lifetime_uses_global_mask_sweep(
                GGML_HTP_OPENCL_LIFETIME_DEFERRED),
            "legacy deferred mode must keep the global mask sweep");
    require(!ggml_htp_opencl_lifetime_uses_global_mask_sweep(
                 GGML_HTP_OPENCL_LIFETIME_REUSE) &&
                !ggml_htp_opencl_lifetime_uses_global_mask_sweep(
                    GGML_HTP_OPENCL_LIFETIME_STRICT),
            "local lifetime modes must execute masks in graph order");

    const char* invalid[] = {"", "auto", "sync", "REUSE", "Strict", " reuse"};
    for (const char* value : invalid) {
        require(ggml_htp_opencl_lifetime_mode_parse(value) ==
                    GGML_HTP_OPENCL_LIFETIME_INVALID,
                "unknown OpenCL lifetime modes must fail closed");
    }
}

void test_opencl_staging_slot_parser() {
    int slots = 0;
    require(ggml_htp_opencl_staging_slots_parse(nullptr, &slots) && slots == 2,
            "unset staging slots must default to two");
    for (int expected = 1; expected <= 8; ++expected) {
        const std::string value = std::to_string(expected);
        slots = 0;
        require(ggml_htp_opencl_staging_slots_parse(value.c_str(), &slots) &&
                    slots == expected,
                "staging slots in the supported range must parse");
    }
    const char* invalid[] = {"", "0", "9", "-1", "+1", "1x", " 2"};
    for (const char* value : invalid) {
        slots = 99;
        require(!ggml_htp_opencl_staging_slots_parse(value, &slots),
                "invalid staging slot values must fail closed");
    }
}

void test_opencl_lifetime_route_contract() {
    set_opencl_lifetime_route_env(nullptr);
    require(sd_htp_opencl_background_residual_job_enabled(),
            "legacy deferred mode must keep the background residual job");
    require(sd_htp_opencl_whole_arena_deferred_ownership(),
            "legacy deferred mode must keep whole-arena retirement");
    require(sd_htp_opencl_persistent_cache_outputs_enabled(),
            "legacy deferred mode must still use persistent residual outputs");
    require(!sd_htp_opencl_bind_cache_outputs_before_reserve(),
            "legacy deferred mode must preserve post-allocation cache binding");

    for (const char* mode : {"reuse", "strict"}) {
        set_opencl_lifetime_route_env(mode);
        require(sd_htp_opencl_local_lifetime_prerequisites(),
                "local lifetime modes must accept the complete OpenCL route");
        require(sd_htp_opencl_background_residual_job_enabled(),
                "local lifetime modes must keep the background residual job");
        require(!sd_htp_opencl_whole_arena_deferred_ownership(),
                "local lifetime modes must disable whole-arena retirement");
        require(sd_htp_opencl_persistent_cache_outputs_enabled(),
                "local lifetime modes must use persistent residual outputs");
        require(sd_htp_opencl_bind_cache_outputs_before_reserve(),
                "local lifetime modes must bind persistent outputs before reserve");
    }

    set_opencl_lifetime_route_env("reuse");
    const char* required[] = {
        "SD_HTP_SPARSE_OPENCL",
        "SD_HTP_SPARSE_OPENCL_RESIDUAL",
        "SD_HTP_SPARSE_OPENCL_ASYNC",
        "SD_HTP_SPARSE_OPENCL_EARLY_SUBMIT",
        "SD_HTP_SPARSE_OPENCL_SUBMIT_THREAD",
        "SD_HTP_SPARSE_OPENCL_DEFER_RESIDUAL",
    };
    for (const char* name : required) {
        setenv(name, "0", 1);
        require(!sd_htp_opencl_local_lifetime_prerequisites(),
                "local lifetime modes must reject missing route prerequisites");
        setenv(name, "1", 1);
    }

    setenv("SD_HTP_SPARSE_OPENCL_LIFETIME_MODE", "auto", 1);
    require(!sd_htp_opencl_local_lifetime_prerequisites(),
            "an invalid local lifetime mode must fail closed");
    require(!sd_htp_opencl_persistent_cache_outputs_enabled(),
            "an invalid lifetime mode must not allocate persistent outputs");
}

void set_local_cache_route_env(bool fused, bool opencl_sparse,
                               bool opencl_residual, const char* mode) {
    setenv("SD_HTP_BLOCK_SPARSE_RESIDUAL_CACHE", "1", 1);
    setenv("SD_HTP_BLOCK_SPARSE_RESIDUAL_EXACT_RHO", "1", 1);
    setenv("SD_HTP_BLOCK_SPARSE_FUSED_ANCHOR", fused ? "1" : "0", 1);
    setenv("SD_HTP_SPARSE_OPENCL", opencl_sparse ? "1" : "0", 1);
    setenv("SD_HTP_SPARSE_OPENCL_RESIDUAL", opencl_residual ? "1" : "0", 1);
    setenv("SD_HTP_SPARSE_OPENCL_ASYNC", opencl_sparse ? "1" : "0", 1);
    setenv("SD_HTP_SPARSE_OPENCL_EARLY_SUBMIT", opencl_sparse ? "1" : "0", 1);
    setenv("SD_HTP_SPARSE_OPENCL_SUBMIT_THREAD", opencl_sparse ? "1" : "0", 1);
    setenv("SD_HTP_SPARSE_OPENCL_DEFER_RESIDUAL", opencl_sparse ? "1" : "0", 1);
    if (mode != nullptr) {
        setenv("SD_HTP_SPARSE_OPENCL_LIFETIME_MODE", mode, 1);
    } else {
        unsetenv("SD_HTP_SPARSE_OPENCL_LIFETIME_MODE");
    }
}

void test_local_cache_lifetime_route_matrix() {
    set_local_cache_route_env(true, false, true, "reuse");
    require(sd_htp_opencl_local_lifetime_prerequisites(),
            "fused residual-only route must use local cache lifetime");
    require(!sd_htp_opencl_sparse_local_lifetime_prerequisites(),
            "fused route must keep fused HTP output allocated");

    set_local_cache_route_env(false, false, false, "strict");
    require(sd_htp_opencl_local_lifetime_prerequisites(),
            "HTP/CPU route must use local cache lifetime");
    require(!sd_htp_opencl_sparse_local_lifetime_prerequisites(),
            "HTP/CPU sparse output must not become metadata-only");

    set_opencl_lifetime_route_env("reuse");
    require(sd_htp_opencl_local_lifetime_prerequisites(),
            "OpenCL route must use local cache lifetime");
    require(sd_htp_opencl_sparse_local_lifetime_prerequisites(),
            "only OpenCL sparse route may use sparse metadata-only outputs");

    set_local_cache_route_env(false, false, false, nullptr);
    require(sd_htp_opencl_local_lifetime_prerequisites(),
            "unset HTP/CPU route must enable local cache lifetime by default");
}

void test_opencl_lifetime_effective_defaults() {
    set_local_cache_route_env(true, false, true, nullptr);
    require(ggml_htp_opencl_lifetime_mode_effective() ==
                GGML_HTP_OPENCL_LIFETIME_REUSE,
            "unset fused route must default to reuse");

    set_local_cache_route_env(false, false, false, nullptr);
    require(ggml_htp_opencl_lifetime_mode_effective() ==
                GGML_HTP_OPENCL_LIFETIME_REUSE,
            "unset HTP/CPU route must default to reuse");

    set_opencl_lifetime_route_env(nullptr);
    require(ggml_htp_opencl_lifetime_mode_effective() ==
                GGML_HTP_OPENCL_LIFETIME_DEFERRED,
            "unset OpenCL sparse route must preserve deferred default");

    for (const char* mode : {"deferred", "reuse", "strict"}) {
        set_local_cache_route_env(true, false, true, mode);
        const auto expected = ggml_htp_opencl_lifetime_mode_parse(mode);
        require(ggml_htp_opencl_lifetime_mode_effective() == expected,
                "explicit lifetime mode must override fused default");

        set_local_cache_route_env(false, false, false, mode);
        require(ggml_htp_opencl_lifetime_mode_effective() == expected,
                "explicit lifetime mode must override HTP/CPU default");

        set_opencl_lifetime_route_env(mode);
        require(ggml_htp_opencl_lifetime_mode_effective() == expected,
                "explicit lifetime mode must override OpenCL default");
    }

    set_local_cache_route_env(true, false, true, "auto");
    require(ggml_htp_opencl_lifetime_mode_effective() ==
                GGML_HTP_OPENCL_LIFETIME_INVALID,
            "invalid explicit lifetime mode must fail closed");
}

#if defined(GGML_USE_HTP)
void test_opencl_persistent_import_route_policy() {
    setenv("SD_HTP_BLOCK_SPARSE_RESIDUAL_CACHE", "1", 1);
    setenv("SD_HTP_BLOCK_SPARSE_RESIDUAL_EXACT_RHO", "1", 1);

    setenv("SD_HTP_BLOCK_SPARSE_FUSED_ANCHOR", "1", 1);
    setenv("SD_HTP_SPARSE_OPENCL", "0", 1);
    setenv("SD_HTP_SPARSE_OPENCL_RESIDUAL", "1", 1);
    require(ggml_backend_htp_opencl_persistent_import_required(),
            "fused residual-only cache must create an OpenCL persistent import");

    setenv("SD_HTP_BLOCK_SPARSE_FUSED_ANCHOR", "0", 1);
    setenv("SD_HTP_SPARSE_OPENCL", "0", 1);
    setenv("SD_HTP_SPARSE_OPENCL_RESIDUAL", "0", 1);
    require(!ggml_backend_htp_opencl_persistent_import_required(),
            "HTP+CPU cache must not create an OpenCL persistent import");

    setenv("SD_HTP_SPARSE_OPENCL", "1", 1);
    setenv("SD_HTP_SPARSE_OPENCL_RESIDUAL", "1", 1);
    require(ggml_backend_htp_opencl_persistent_import_required(),
            "OpenCL sparse cache must create an OpenCL persistent import");
}
#endif

void test_post_compute_transient_buffer_release_policy() {
    require(ggml_htp_opencl_can_release_transient_buffers_post_compute(
                GGML_HTP_OPENCL_LIFETIME_REUSE,
                true,
                true,
                true,
                false),
            "local fused graphs may release transient HTP buffers after compute");
    require(ggml_htp_opencl_can_release_transient_buffers_post_compute(
                GGML_HTP_OPENCL_LIFETIME_STRICT,
                true,
                true,
                true,
                false),
            "strict local graphs use the same post-compute release policy");

    require(!ggml_htp_opencl_can_release_transient_buffers_post_compute(
                 GGML_HTP_OPENCL_LIFETIME_DEFERRED,
                 true,
                 true,
                 true,
                 false),
            "deferred ownership must retain its transient buffers");
    require(!ggml_htp_opencl_can_release_transient_buffers_post_compute(
                 GGML_HTP_OPENCL_LIFETIME_REUSE,
                 true,
                 false,
                 true,
                 false),
            "background residual work must retain transient buffers");
    require(!ggml_htp_opencl_can_release_transient_buffers_post_compute(
                 GGML_HTP_OPENCL_LIFETIME_REUSE,
                 true,
                 true,
                 false,
                 false),
            "ordinary cache copies may still need transient HTP tensors");
    require(!ggml_htp_opencl_can_release_transient_buffers_post_compute(
                 GGML_HTP_OPENCL_LIFETIME_REUSE,
                 true,
                 true,
                 true,
                 true),
            "debug tensor reads must retain transient HTP tensors");
    require(!ggml_htp_opencl_can_release_transient_buffers_post_compute(
                 GGML_HTP_OPENCL_LIFETIME_REUSE,
                 false,
                 true,
                 true,
                 false),
            "non-HTP graphs have no HTP transient buffer to release");
}

#if defined(GGML_USE_HTP)
void require_valid_summary(ggml_htp_route_summary summary, const char* message) {
    char reason[256] = {};
    require(ggml_backend_htp_route_summary_validate(
                &summary, reason, sizeof(reason)), message);
}

void require_invalid_summary(ggml_htp_route_summary summary, const char* message) {
    char reason[256] = {};
    require(!ggml_backend_htp_route_summary_validate(
                 &summary, reason, sizeof(reason)), message);
}

void test_route_summaries() {
    ggml_htp_route_summary fused = {};
    fused.mode = GGML_HTP_ROUTE_FUSED_OPENCL_RESIDUAL;
    fused.anchor = 1;
    fused.fused_anchor = 4;
    fused.opencl_residual = 4;
    fused.cache_pairs = 4;
    require_valid_summary(fused, "fused anchor summary must pass");

    ggml_htp_route_summary opencl = {};
    opencl.mode = GGML_HTP_ROUTE_OPENCL_ANCHOR;
    opencl.anchor = 1;
    opencl.htp_dense = 4;
    opencl.opencl_sparse = 4;
    opencl.opencl_residual = 4;
    opencl.cache_pairs = 4;
    require_valid_summary(opencl, "OpenCL anchor summary must pass");

    ggml_htp_route_summary htp_cpu = {};
    htp_cpu.mode = GGML_HTP_ROUTE_HTP_CPU;
    htp_cpu.anchor = 1;
    htp_cpu.htp_dense = 4;
    htp_cpu.htp_sparse = 4;
    htp_cpu.cpu_residual = 4;
    htp_cpu.cache_pairs = 4;
    require_valid_summary(htp_cpu, "HTP/CPU anchor summary must pass");

    for (const auto mode : {GGML_HTP_ROUTE_FUSED_OPENCL_RESIDUAL,
                            GGML_HTP_ROUTE_OPENCL_ANCHOR,
                            GGML_HTP_ROUTE_HTP_CPU}) {
        ggml_htp_route_summary apply = {};
        apply.mode = mode;
        apply.htp_sparse = 4;
        require_valid_summary(apply, "HTP sparse apply summary must pass");
    }

    ggml_htp_route_summary zero_work = {};
    zero_work.mode = GGML_HTP_ROUTE_OPENCL_ANCHOR;
    zero_work.anchor = 1;
    require_invalid_summary(zero_work, "zero-work anchor must fail");

    ggml_htp_route_summary warmup_dense = {};
    warmup_dense.mode = GGML_HTP_ROUTE_OPENCL_ANCHOR;
    warmup_dense.anchor = 1;
    warmup_dense.htp_dense = 4;
    require_valid_summary(
        warmup_dense,
        "dense-only warmup graph without residual cache pairs must pass");

    ggml_htp_route_summary mixed = opencl;
    mixed.htp_sparse = 1;
    require_invalid_summary(mixed, "mixed anchor ownership must fail");

    ggml_htp_route_summary cpu_fallback = htp_cpu;
    cpu_fallback.cpu_attention_fallback = 1;
    require_invalid_summary(cpu_fallback, "CPU attention fallback must fail");
}
#endif

void test_memory_ledger_snapshot() {
    ggml_htp_memory_ledger_snapshot snapshot = {};
    snapshot.requested_route = GGML_HTP_ROUTE_FUSED_OPENCL_RESIDUAL;
    snapshot.preflight_result = 1;
    snapshot.routed_sparse_count = 0;
    snapshot.routed_rho_count = 0;
    snapshot.routed_residual_count = 0;
    snapshot.reported_opencl_residual = 2;
    snapshot.runtime_buffer_bytes = 1024;
    snapshot.htp_buffer_bytes = 2048;
    snapshot.qkv_buffer_bytes = 4096;
    snapshot.host_cache_bytes = 8192;
    snapshot.live_imports = 4;
    snapshot.live_events = 3;
    snapshot.live_leases = 2;
    snapshot.pending_jobs = 1;
    snapshot.retired_arenas = 2;
    snapshot.spilled_host_bytes = 256;
    snapshot.hydrated_rpcmem_bytes = 512;
    snapshot.cache_slot_count = 4;

    const struct SlotCase {
        const char * key;
        int branch;
        uint64_t generation;
        int fd;
        size_t logical_size;
        size_t allocation_size;
        uint64_t import_identity;
    } slots[] = {
        {"htp_block_sparse_residual_b0_self_s0", 0, 11, 101, 100, 128, 1001},
        {"htp_block_sparse_residual_rho_b0_self_s0", 0, 11, 102, 200, 256, 1002},
        {"htp_block_sparse_residual_b1_self_s0", 1, 12, 103, 300, 384, 1003},
        {"htp_block_sparse_residual_rho_b1_self_s0", 1, 12, 104, 400, 512, 1004},
    };
    for (size_t i = 0; i < snapshot.cache_slot_count; ++i) {
        std::snprintf(snapshot.cache_slots[i].cache_key,
                      sizeof(snapshot.cache_slots[i].cache_key), "%s",
                      slots[i].key);
        snapshot.cache_slots[i].branch = slots[i].branch;
        snapshot.cache_slots[i].generation = slots[i].generation;
        snapshot.cache_slots[i].fd = slots[i].fd;
        snapshot.cache_slots[i].logical_size = slots[i].logical_size;
        snapshot.cache_slots[i].allocation_size = slots[i].allocation_size;
        snapshot.cache_slots[i].opencl_import_identity = slots[i].import_identity;
    }

    require(ggml_backend_htp_memory_ledger_recalculate(&snapshot),
            "memory ledger totals must be recalculated from cache slots");
    require(snapshot.rpcmem_cache_logical_bytes == 1000,
            "memory ledger logical bytes must equal the slot sum");
    require(snapshot.rpcmem_cache_allocation_bytes == 1280,
            "memory ledger allocation bytes must equal the slot sum");

    char summary[1024] = {};
    require(ggml_backend_htp_format_memory_ledger(
                summary, sizeof(summary), "pre_compute", &snapshot) > 0,
            "memory ledger summary must format");
    const std::string summary_text(summary);
    require(summary_text.find("requested=fused") != std::string::npos,
            "memory ledger must report the requested route");
    require(summary_text.find("routed_sparse=0 routed_rho=0 routed_residual=0") !=
                std::string::npos,
            "memory ledger must report actual routed counts");
    require(summary_text.find("reported_opencl_residual=2") != std::string::npos,
            "memory ledger must keep the legacy reported route separate");
    require(summary_text.find("rpcmem_cache_logical_bytes=1000") !=
                std::string::npos,
            "memory ledger must report logical cache bytes");
    require(summary_text.find("rpcmem_cache_allocation_bytes=1280") !=
                std::string::npos,
            "memory ledger must report allocation cache bytes");

    char slot_line[512] = {};
    require(ggml_backend_htp_format_memory_ledger_slot(
                slot_line, sizeof(slot_line), &snapshot.cache_slots[2]) > 0,
            "memory ledger slot must format");
    const std::string slot_text(slot_line);
    require(slot_text.find("branch=1") != std::string::npos &&
                slot_text.find("generation=12") != std::string::npos &&
                slot_text.find("fd=103") != std::string::npos &&
                slot_text.find("import_identity=1003") != std::string::npos,
            "memory ledger slot must preserve branch and backing identity");

    unsetenv("SD_HTP_SPARSE_OPENCL_TRACE");
    require(!ggml_backend_htp_memory_ledger_trace_enabled(),
            "memory ledger diagnostics must be silent by default");
    setenv("SD_HTP_SPARSE_OPENCL_TRACE", "1", 1);
    require(ggml_backend_htp_memory_ledger_trace_enabled(),
            "memory ledger diagnostics must use the sparse OpenCL trace gate");
    setenv("SD_HTP_SPARSE_OPENCL_TRACE", "0", 1);
    require(!ggml_backend_htp_memory_ledger_trace_enabled(),
            "memory ledger diagnostics must be silent when trace is zero");
}

#if defined(GGML_USE_HTP)
void test_cpu_residual_graph_summary() {
    setenv("SD_HTP_BLOCK_SPARSE_RESIDUAL_CACHE", "1", 1);
    setenv("SD_HTP_BLOCK_SPARSE_FUSED_ANCHOR", "0", 1);
    setenv("SD_HTP_SPARSE_OPENCL", "0", 1);
    setenv("SD_HTP_SPARSE_OPENCL_RESIDUAL", "0", 1);

    ggml_init_params params = {
        /*.mem_size   =*/ 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context* ctx = ggml_init(params);
    require(ctx != nullptr, "CPU residual graph context must initialize");

    ggml_tensor* q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 16, 4, 2, 1);
    ggml_tensor* k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 16, 4, 2, 1);
    ggml_tensor* v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 16, 2, 4, 1);
    ggml_tensor* block_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, 2);

    ggml_tensor* dense = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 17, 2, 4, 1);
    dense->op = GGML_OP_FLASH_ATTN_EXT;
    dense->src[0] = q;
    dense->src[1] = k;
    dense->src[2] = v;
    ggml_set_name(dense, "route_test_dense_attention");

    ggml_tensor* sparse = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 17, 2, 4, 1);
    sparse->op = GGML_OP_FLASH_ATTN_EXT;
    sparse->src[0] = q;
    sparse->src[1] = k;
    sparse->src[2] = v;
    sparse->src[3] = block_mask;
    ggml_set_name(sparse, "route_test_sparse_attention");

    ggml_tensor* residual = ggml_cast(ctx, ggml_sub(ctx, dense, sparse), GGML_TYPE_F16);
    ggml_set_name(residual, "htp_block_sparse_residual_b0_self_s0");
    ggml_cgraph* graph = ggml_new_graph_custom(ctx, 32, false);
    ggml_build_forward_expand(graph, residual);

    ggml_htp_route_summary summary = {};
    require(ggml_backend_htp_collect_route_summary(nullptr, graph, &summary),
            "CPU residual graph must participate in route accounting");
    require(summary.anchor == 1,
            "a named CPU residual cache output must identify an anchor graph");
    require(summary.cache_pairs == 1,
            "a named CPU residual cache output must count one cache pair");
    require(summary.cpu_residual == 1,
            "a non-OpenCL residual cache output must be CPU owned");

    ggml_tensor* cached_input = ggml_new_tensor_4d(
        ctx, GGML_TYPE_F32, 17, 2, 4, 1);
    ggml_tensor* cached_copy = ggml_dup(ctx, cached_input);
    ggml_set_name(cached_copy, "htp_block_sparse_residual_b0_self_s0");
    ggml_tensor* apply_output = ggml_add(ctx, sparse, cached_copy);
    ggml_cgraph* apply_graph = ggml_new_graph_custom(ctx, 32, false);
    ggml_build_forward_expand(apply_graph, apply_output);

    summary = {};
    require(ggml_backend_htp_collect_route_summary(nullptr, apply_graph, &summary),
            "sparse apply graph must participate in route accounting");
    require(summary.anchor == 0 && summary.cache_pairs == 0 &&
                summary.cpu_residual == 0,
            "a named cache input without attention ancestors must remain an apply graph");

    ggml_free(ctx);
}

void test_fused_route_summary_uses_actual_routed_set() {
    setenv("SD_HTP_BLOCK_SPARSE_RESIDUAL_CACHE", "1", 1);
    setenv("SD_HTP_BLOCK_SPARSE_FUSED_ANCHOR", "1", 1);
    setenv("SD_HTP_SPARSE_OPENCL", "0", 1);
    setenv("SD_HTP_SPARSE_OPENCL_RESIDUAL", "1", 1);

    ggml_init_params params = {
        /*.mem_size   =*/ 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context* ctx = ggml_init(params);
    require(ctx != nullptr, "fused route graph context must initialize");

    ggml_tensor* fused = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 272);
    fused->op = GGML_OP_CUSTOM;
    ggml_set_name(fused, "htp_fused_anchor_attention");
    ggml_tensor* dense = ggml_view_4d(
        ctx, fused, 17, 2, 4, 1, 17 * sizeof(float),
        34 * sizeof(float), 136 * sizeof(float), 0);
    ggml_tensor* sparse = ggml_view_4d(
        ctx, fused, 17, 2, 4, 1, 17 * sizeof(float),
        34 * sizeof(float), 136 * sizeof(float), 136 * sizeof(float));

    ggml_tensor* rho = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, 2, 4, 1);
    rho->op = GGML_OP_CUSTOM;
    rho->src[0] = dense;
    rho->src[1] = sparse;
    ggml_set_name(rho, "htp_block_sparse_residual_rho_b0_route_s0");
    ggml_tensor* residual = ggml_new_tensor_4d(
        ctx, GGML_TYPE_F16, 16, 2, 4, 1);
    residual->op = GGML_OP_CUSTOM;
    residual->src[0] = dense;
    residual->src[1] = sparse;
    residual->src[2] = rho;
    ggml_set_name(residual, "htp_block_sparse_residual_b0_route_s0");

    ggml_cgraph* graph = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(graph, residual);

    ggml_htp_route_summary summary = {};
    require(ggml_backend_htp_collect_route_summary(nullptr, graph, &summary),
            "fused residual graph must participate in route accounting");
    require(summary.mode == GGML_HTP_ROUTE_FUSED_OPENCL_RESIDUAL &&
                summary.cache_pairs == 1,
            "fused residual graph must identify one requested cache pair");
    require(summary.opencl_residual == 0 && summary.cpu_residual == 1,
            "fused summary must not report OpenCL without a frozen routed pair");
    require_invalid_summary(
        summary,
        "requested fused route without actual residual routing must fail closed");

    ggml_free(ctx);
}
#endif

}  // namespace

int main() {
    test_packed_rope_reference_convention();
    test_z_image_packed_qkv_param_placement();
    test_z_image_anchor_scale_contract();
    test_flash_attention_halo_loader_metadata();
    test_route_mode_from_flags();
    test_attention_policy();
    test_packed_residual_apply_layout_policy();
    test_packed_residual_apply_lse_admission();
    test_anchor_halo_loader_policy();
    test_residual_binding_keys();
    test_model_binding_sequences();
    test_z_image_route_selection();
    test_z_image_pe_ordering();
    test_z_image_binding_sequences();
    test_opencl_lifetime_mode_parser();
    test_opencl_staging_slot_parser();
    test_opencl_lifetime_route_contract();
    test_local_cache_lifetime_route_matrix();
    test_opencl_lifetime_effective_defaults();
#if defined(GGML_USE_HTP)
    test_opencl_persistent_import_route_policy();
#endif
    test_post_compute_transient_buffer_release_policy();
#if defined(GGML_USE_HTP)
    test_route_summaries();
    test_cpu_residual_graph_summary();
    test_fused_route_summary_uses_actual_routed_set();
#endif
    test_memory_ledger_snapshot();
    std::cout << "htp route policy tests passed\n";
    return 0;
}
