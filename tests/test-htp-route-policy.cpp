#include <cstdlib>
#include <iostream>
#include <string>

#include "ggml-htp.h"
#include "z_image.hpp"

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
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

    const auto sparse_without_apply = sd_htp_attention_policy_for(
        true, SD_HTP_BLOCK_SPARSE_RESIDUAL_OFF, true);
    require(!sparse_without_apply.packed_qkv_allowed,
            "packed QKV is reserved for sparse residual apply steps");

    const auto text_only = sd_htp_attention_policy_for(
        true, SD_HTP_BLOCK_SPARSE_RESIDUAL_APPLY, false);
    require(!text_only.residual_eligible,
            "text-only attention must be excluded from residual accounting");
    require(!text_only.packed_qkv_allowed,
            "text-only attention must not enter the residual packed route");
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
        true, true, true, true, true, true, false, true);
    require(anchor.route == ZImageAttentionRoute::REORDERED,
            "Z-Image anchor attention must use the non-packed reordered route");

    const auto packed_apply = z_image_select_attention_route(
        true, true, true, true, true, true, false, false);
    require(packed_apply.route == ZImageAttentionRoute::PACKED,
            "Z-Image sparse apply may use packed QKV when available");

    const auto reordered_apply = z_image_select_attention_route(
        true, true, true, true, true, false, false, false);
    require(reordered_apply.route == ZImageAttentionRoute::REORDERED,
            "Z-Image sparse apply must fall back to reordered non-packed QKV");

    const auto text_only = z_image_select_attention_route(
        true, false, false, false, false, false, false, false);
    require(text_only.route == ZImageAttentionRoute::DENSE,
            "Z-Image text-only context refiner must remain dense");

    require(ZImage::z_image_sparse_layout_eligible(256, 16, 16, 11, false),
            "position-index offsets without ref tensors must keep sparse geometry eligible");
    require(!ZImage::z_image_sparse_layout_eligible(256, 16, 16, 11, true),
            "actual appended ref latent tokens must remain excluded");
}

void test_z_image_binding_sequences() {
    const char* z_image_kinds[] = {"noise", "joint", "joint"};
    test_stable_binding_sequence(z_image_kinds, 3);
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

    ggml_htp_route_summary mixed = opencl;
    mixed.htp_sparse = 1;
    require_invalid_summary(mixed, "mixed anchor ownership must fail");

    ggml_htp_route_summary cpu_fallback = htp_cpu;
    cpu_fallback.cpu_attention_fallback = 1;
    require_invalid_summary(cpu_fallback, "CPU attention fallback must fail");
}

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
#endif

}  // namespace

int main() {
    test_route_mode_from_flags();
    test_attention_policy();
    test_residual_binding_keys();
    test_model_binding_sequences();
    test_z_image_route_selection();
    test_z_image_binding_sequences();
#if defined(GGML_USE_HTP)
    test_route_summaries();
    test_cpu_residual_graph_summary();
#endif
    std::cout << "htp route policy tests passed\n";
    return 0;
}
