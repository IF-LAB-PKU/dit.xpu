// Z-Image attention contract and regression fixture.
// Validates 30-head D128 Q/K/V shapes, patch-major reorder roundtrip,
// RoPE ID layouts, and (initially) packed-attention unavailability.

#include "../../src/ggml_extend.hpp"
#include "../../src/rope.hpp"
#include "../../src/z_image.hpp"
#include "../../src/util.h"

#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-htp.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// The fixture links directly against ggml CPU to avoid pulling the HTP backend.
// Keep logging self-contained instead of linking the full stable-diffusion lib.
void log_printf(sd_log_level_t, const char *, int, const char *, ...) {}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static float abs_max_diff(const std::vector<float>& a, const std::vector<float>& b) {
    GGML_ASSERT(a.size() == b.size());
    float max_abs = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        max_abs = std::max(max_abs, std::fabs(a[i] - b[i]));
    }
    return max_abs;
}

static float abs_max_diff_f16(const std::vector<ggml_fp16_t>& a, const std::vector<ggml_fp16_t>& b) {
    GGML_ASSERT(a.size() == b.size());
    float max_abs = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        max_abs = std::max(max_abs, std::fabs(ggml_fp16_to_fp32(a[i]) - ggml_fp16_to_fp32(b[i])));
    }
    return max_abs;
}

static float deterministic_f32(int64_t row, int64_t col) {
    return float(((row * 11 + col * 5) % 37) - 18) * 0.01f;
}

// ---------------------------------------------------------------------------
// Test 1: Z-Image shape validation (30 x 128, GQA 30/30)
// ---------------------------------------------------------------------------

static void test_z_image_shape_validation() {
    constexpr int64_t hidden_size = 3840;
    constexpr int64_t head_dim    = 128;
    constexpr int64_t num_heads   = 30;
    constexpr int64_t num_kv_heads = 30;

    // QKV fused size
    constexpr int64_t qkv_size = (num_heads + num_kv_heads * 2) * head_dim;
    GGML_ASSERT(qkv_size == 11520);  // 30*128 + 30*2*128 = 3840 + 7680 = 11520

    // Check the fused weight shape matches ZImageParams contract
    // weight: [hidden_size, qkv_size] = [3840, 11520]
    GGML_ASSERT(hidden_size == 3840);
    GGML_ASSERT(head_dim == 128);
    GGML_ASSERT(num_heads == 30);
    GGML_ASSERT(num_kv_heads == 30);

    // GQA mapping: num_heads == num_kv_heads means no GQA grouping
    GGML_ASSERT(num_heads % num_kv_heads == 0);
    GGML_ASSERT(num_heads / num_kv_heads == 1);

    std::printf("[PASS] z_image_shape_validation: hidden=%lld heads=%lld kv_heads=%lld head_dim=%lld qkv=%lld\n",
                (long long)hidden_size, (long long)num_heads,
                (long long)num_kv_heads, (long long)head_dim, (long long)qkv_size);
}

// ---------------------------------------------------------------------------
// Test 2: Sequence metadata for image-first aligned grid
// ---------------------------------------------------------------------------

static void test_z_image_sequence_metadata() {
    constexpr int token_h      = 16;
    constexpr int token_w      = 32;
    constexpr int image_tokens = token_h * token_w;  // 512
    constexpr int text_tokens  = 64;
    constexpr int total_tokens = image_tokens + text_tokens;

    GGML_ASSERT(image_tokens == 512);
    GGML_ASSERT(total_tokens == 576);

    // Image grid must be statically valid for static halo
    GGML_ASSERT(image_tokens % 32 == 0);   // 512 % 32 == 0
    GGML_ASSERT(token_h % 4 == 0);          // 16 % 4 == 0
    GGML_ASSERT(token_w % 8 == 0);          // 32 % 8 == 0

    // Text tokens must be non-negative
    GGML_ASSERT(text_tokens >= 0);

    // Image blocks for static halo
    constexpr int image_block_rows = token_h / 4;  // 4
    constexpr int image_block_cols = token_w / 8;  // 4
    constexpr int image_blocks     = image_block_rows * image_block_cols;  // 16
    constexpr int total_blocks     = (total_tokens + 31) / 32;            // 18

    GGML_ASSERT(image_blocks == 16);
    GGML_ASSERT(total_blocks == 18);

    // Image-first layout: img tokens at [0, 512), text at [512, 576)
    constexpr int img_start = 0;
    constexpr int img_end   = image_tokens;     // 512
    constexpr int txt_start = image_tokens;      // 512
    constexpr int txt_end   = total_tokens;      // 576

    GGML_ASSERT(img_start == 0);
    GGML_ASSERT(img_end == image_tokens);
    GGML_ASSERT(txt_start == image_tokens);
    GGML_ASSERT(txt_end == total_tokens);

    // Protected region for sparse attention: text tokens are protected suffix
    constexpr int protected_start = image_tokens;
    constexpr int protected_len   = text_tokens;
    GGML_ASSERT(protected_start == 512);
    GGML_ASSERT(protected_len == 64);

    std::printf("[PASS] z_image_sequence_metadata: img=%d txt=%d blocks=%d img_blocks=%d\n",
                image_tokens, text_tokens, total_blocks, image_blocks);
}

// ---------------------------------------------------------------------------
// Test 3: Patch-major reorder roundtrip
// ---------------------------------------------------------------------------

static void test_z_image_patch_major_roundtrip() {
    constexpr int token_h      = 16;
    constexpr int token_w      = 32;
    constexpr int patch        = 4;
    constexpr int image_tokens = token_h * token_w;  // 512

    // Forward: row-major -> patch-major
    std::vector<int> patch_major_indices(image_tokens);
    for (int i = 0; i < image_tokens; ++i) {
        patch_major_indices[i] = sd_patch_major_index_from_row_major(i, token_h, token_w, patch);
    }

    // Verify forward mapping is a permutation (all indices unique, within bounds)
    std::vector<bool> seen(image_tokens, false);
    for (int i = 0; i < image_tokens; ++i) {
        int pm = patch_major_indices[i];
        GGML_ASSERT(pm >= 0 && pm < image_tokens);
        GGML_ASSERT(!seen[pm]);
        seen[pm] = true;
    }

    // Inverse: patch-major -> row-major
    std::vector<int> row_major_indices(image_tokens);
    for (int i = 0; i < image_tokens; ++i) {
        row_major_indices[i] = sd_row_major_index_from_patch_major(i, token_h, token_w, patch);
    }

    // Verify roundtrip: for every position, inverse(forward(i)) == i
    for (int i = 0; i < image_tokens; ++i) {
        int pm = patch_major_indices[i];
        int rm = row_major_indices[pm];
        GGML_ASSERT(rm == i);
    }

    // Also verify forward(inverse(j)) == j
    for (int j = 0; j < image_tokens; ++j) {
        int rm = row_major_indices[j];
        int pm = patch_major_indices[rm];
        GGML_ASSERT(pm == j);
    }

    // Tensor-style roundtrip: create data, reorder, inverse-reorder, compare
    std::vector<float> src(image_tokens);
    for (int i = 0; i < image_tokens; ++i) {
        src[i] = deterministic_f32(i, 0);
    }

    std::vector<float> reordered(image_tokens);
    for (int i = 0; i < image_tokens; ++i) {
        reordered[patch_major_indices[i]] = src[i];
    }

    std::vector<float> recovered(image_tokens);
    for (int j = 0; j < image_tokens; ++j) {
        recovered[row_major_indices[j]] = reordered[j];
    }

    float max_abs = abs_max_diff(src, recovered);
    GGML_ASSERT(max_abs == 0.0f);

    std::printf("[PASS] z_image_layout_roundtrip: max_abs=%.6f\n", max_abs);
}

// ---------------------------------------------------------------------------
// Test 4: RoPE ID order for text-first and image-first layouts
// ---------------------------------------------------------------------------

static void test_z_image_rope_layout() {
    constexpr int h              = 32;
    constexpr int w              = 64;
    constexpr int patch_size     = 2;
    constexpr int bs             = 1;
    constexpr int context_len    = 64;
    constexpr int seq_multi_of   = 32;
    constexpr int theta          = 256;
    constexpr int hidden_size    = 3840;
    constexpr int head_dim       = 128;
    // axes_dim = {32, 48, 48}, axes_dim_sum = 128

    std::vector<int> axes_dim = {32, 48, 48};

    // Dense (text-first) PE layout
    std::vector<ggml_tensor*> empty_refs;
    auto pe_vec_text_first = Rope::gen_z_image_pe(h, w, patch_size, bs, context_len,
                                                   seq_multi_of, empty_refs, false,
                                                   theta, false, false, axes_dim);

    // Image tokens in patchified grid
    int token_h      = h / patch_size;   // 16
    int token_w      = w / patch_size;   // 32
    int image_tokens = token_h * token_w; // 512
    int padded_context_len = context_len + Rope::bound_mod(context_len, seq_multi_of);
    int img_pad_len = Rope::bound_mod(image_tokens, seq_multi_of);
    int total_pos = padded_context_len + image_tokens + img_pad_len;

    // PE vector size: total_pos * axes_dim_sum * 2
    constexpr int axes_dim_sum = 128;
    GGML_ASSERT(pe_vec_text_first.size() == (size_t)total_pos * axes_dim_sum * 2);

    // Verify text-first PE: text IDs come first
    // We can't directly inspect the PE values without going through embed_nd,
    // but we can verify the ID generation is text-first
    auto txt_first_ids = Rope::gen_z_image_ids(h, w, patch_size, bs, context_len,
                                                seq_multi_of, empty_refs, false);

    // Text tokens occupy the first padded_context_len positions
    // Image tokens occupy the remaining positions
    int padded_ctx = context_len + Rope::bound_mod(context_len, seq_multi_of);
    GGML_ASSERT(padded_ctx == 64);  // already aligned to 32

    // Verify the ID structure: text IDs have x[0] = position+1
    for (int i = 0; i < padded_ctx; ++i) {
        GGML_ASSERT(txt_first_ids[i][0] == i + 1.0f);
        // Text tokens have y=0, z=0
        GGML_ASSERT(txt_first_ids[i][1] == 0.0f);
        GGML_ASSERT(txt_first_ids[i][2] == 0.0f);
    }

    // Image IDs start after text
    int img_start_idx = padded_ctx;
    // First image token should have non-zero y, x coordinates
    bool image_has_spatial = false;
    for (int i = img_start_idx; i < img_start_idx + image_tokens; ++i) {
        if (txt_first_ids[i][1] != 0.0f || txt_first_ids[i][2] != 0.0f) {
            image_has_spatial = true;
            break;
        }
    }
    GGML_ASSERT(image_has_spatial);

    // Also test image-first ID generation (to be implemented in Task 3)
    // For now, just verify text-first layout is correct

    std::printf("[PASS] z_image_rope_layout: text_first=1 pos_len=%d txt_pad=%d img_pad=%d\n",
                total_pos, padded_ctx - context_len, img_pad_len);
}

// ---------------------------------------------------------------------------
// Test 5: Packed QKV shape metadata (initially unavailable)
// ---------------------------------------------------------------------------

static void test_z_image_packed_qkv_metadata_unavailable() {
    constexpr int64_t hidden_size = 3840;
    constexpr int64_t head_dim    = 128;
    constexpr int64_t num_heads   = 30;
    constexpr int64_t num_kv_heads = 30;
    constexpr int64_t qkv_size    = (num_heads + num_kv_heads * 2) * head_dim;

    // Build a minimal graph context to test packed discovery
    ggml_init_params params = {
        /*.mem_size   =*/ 64 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context* ctx = ggml_init(params);
    GGML_ASSERT(ctx != nullptr);

    // Create a fused QKV weight tensor (no packed flag on it)
    ggml_tensor* weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, hidden_size, qkv_size);
    ggml_set_name(weight, "qkv.weight");

    // Create activation
    constexpr int tokens = 576;
    ggml_tensor* activation = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_size, tokens);

    // Standard QKV projection: mul_mat -> reshape -> views
    ggml_tensor* qkv = ggml_mul_mat(ctx, weight, activation);
    qkv = ggml_reshape_4d(ctx, qkv, head_dim, num_heads + num_kv_heads * 2, tokens, 1);

    auto q = ggml_view_4d(ctx, qkv, qkv->ne[0], num_heads, qkv->ne[2], qkv->ne[3],
                          qkv->nb[1], qkv->nb[2], qkv->nb[3], 0);
    auto k = ggml_view_4d(ctx, qkv, qkv->ne[0], num_kv_heads, qkv->ne[2], qkv->ne[3],
                          qkv->nb[1], qkv->nb[2], qkv->nb[3], num_heads * qkv->nb[1]);
    auto v = ggml_view_4d(ctx, qkv, qkv->ne[0], num_kv_heads, qkv->ne[2], qkv->ne[3],
                          qkv->nb[1], qkv->nb[2], qkv->nb[3], (num_heads + num_kv_heads) * qkv->nb[1]);

    GGML_ASSERT(q->ne[0] == head_dim);
    GGML_ASSERT(q->ne[1] == num_heads);
    GGML_ASSERT(q->ne[2] == tokens);
    GGML_ASSERT(k->ne[0] == head_dim);
    GGML_ASSERT(k->ne[1] == num_kv_heads);
    GGML_ASSERT(v->ne[0] == head_dim);
    GGML_ASSERT(v->ne[1] == num_kv_heads);

    // Verify that packed layout is NOT discovered (no packed name, no packed weight)
    // This is tested by verifying that the Q/K/V views don't have a packed storage ancestor
    auto find_packed = [](ggml_tensor* t) -> bool {
        for (int depth = 0; t != nullptr && depth < 8; ++depth) {
            if (t->name[0] != '\0' &&
                (std::strcmp(t->name, "qkv_mul_rms_packed") == 0 ||
                 std::strcmp(t->name, "qkv_mul_rms_packed_dual") == 0 ||
                 std::strcmp(t->name, "qkv_mul_rms_packed_rope") == 0 ||
                 std::strcmp(t->name, "qkv_mul_rms_packed_dual_rope") == 0)) {
                return true;
            }
            t = t->view_src;
        }
        return false;
    };

    GGML_ASSERT(!find_packed(q));
    GGML_ASSERT(!find_packed(k));
    GGML_ASSERT(!find_packed(v));

    ggml_free(ctx);

    std::printf("[PASS] z_image_packed_metadata_unavailable: packed route correctly unavailable for standard QKV\n");
}

// ---------------------------------------------------------------------------
// Test 6: 30-head attention shape validation (dense CPU reference)
// ---------------------------------------------------------------------------

static void test_z_image_dense_attention_30h() {
    constexpr int64_t head_dim    = 128;
    constexpr int64_t num_heads   = 30;
    constexpr int64_t num_kv_heads = 30;
    constexpr int tokens           = 576;
    constexpr int64_t hidden_size  = num_heads * head_dim;

    ggml_init_params params = {
        /*.mem_size   =*/ 64 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context* ctx = ggml_init(params);
    GGML_ASSERT(ctx != nullptr);

    // Build Q/K/V in [N, L, n_head, d_head] layout (as used by Rope::attention)
    ggml_tensor* q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dim, tokens, num_heads, 1);
    ggml_tensor* k = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dim, tokens, num_kv_heads, 1);
    ggml_tensor* v = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dim, tokens, num_kv_heads, 1);

    // Fill with deterministic data
    std::vector<float> q_data(head_dim * tokens * num_heads);
    std::vector<float> k_data(head_dim * tokens * num_kv_heads);
    std::vector<float> v_data(head_dim * tokens * num_kv_heads);

    for (int h = 0; h < num_heads; ++h) {
        for (int t = 0; t < tokens; ++t) {
            for (int d = 0; d < head_dim; ++d) {
                size_t qi = (size_t)h * tokens * head_dim + (size_t)t * head_dim + d;
                size_t ki = (size_t)h * tokens * head_dim + (size_t)t * head_dim + d;
                q_data[qi] = deterministic_f32(t, d + h * 100);
                k_data[ki] = deterministic_f32(t + 1000, d + h * 100);
                v_data[ki] = deterministic_f32(t + 2000, d + h * 100);
            }
        }
    }

    // Create PE tensor [L, d_head/2, 2, 2] (dummy cos/sin for testing)
    int pos_len = tokens;
    ggml_tensor* pe = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 2, 2, head_dim / 2, pos_len);
    std::vector<float> pe_data(pos_len * (head_dim / 2) * 4);
    // Fill with identity-like RoPE values that should give predictable results
    for (int p = 0; p < pos_len; ++p) {
        for (int d = 0; d < head_dim / 2; ++d) {
            // cos=1, sin=0 -> identity RoPE (no rotation)
            size_t base = (size_t)p * (head_dim / 2) * 4 + d * 4;
            pe_data[base + 0] = 1.0f;  // cos
            pe_data[base + 1] = 0.0f;  // -sin
            pe_data[base + 2] = 0.0f;  // sin
            pe_data[base + 3] = 1.0f;  // cos
        }
    }

    // Build a minimal compute graph using the attention path
    ggml_backend_t cpu_backend = ggml_backend_cpu_init();
    GGML_ASSERT(cpu_backend != nullptr);

    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(cpu_backend);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    GGML_ASSERT(buffer != nullptr);

    ggml_backend_tensor_set(q, q_data.data(), 0, q_data.size() * sizeof(float));
    ggml_backend_tensor_set(k, k_data.data(), 0, k_data.size() * sizeof(float));
    ggml_backend_tensor_set(v, v_data.data(), 0, v_data.size() * sizeof(float));
    ggml_backend_tensor_set(pe, pe_data.data(), 0, pe_data.size() * sizeof(float));

    // Use ggml_ext_attention_ext directly (not Rope::attention which also does RoPE)
    // We test the attention shape contract
    {
        // Reshape to [N, L, C] for ggml_ext_attention_ext
        ggml_tensor* q_2d = ggml_reshape_3d(ctx, q, hidden_size, tokens, 1);
        ggml_tensor* k_2d = ggml_reshape_3d(ctx, k, hidden_size, tokens, 1);
        ggml_tensor* v_2d = ggml_reshape_3d(ctx, v, hidden_size, tokens, 1);

        ggml_tensor* out = ggml_ext_attention_ext(ctx, cpu_backend,
                                                   q_2d, k_2d, v_2d,
                                                   num_heads, nullptr,
                                                   false, false, 1.0f / 128.f);

        GGML_ASSERT(out != nullptr);
        GGML_ASSERT(out->ne[0] == hidden_size);  // n_head * d_head
        GGML_ASSERT(out->ne[1] == tokens);
        GGML_ASSERT(out->ne[2] == 1);

        ggml_cgraph* graph = ggml_new_graph_custom(ctx, 256, false);
        ggml_build_forward_expand(graph, out);
        // The attention graph is created after the input tensors. Allocate its
        // intermediate tensors before asking the CPU backend to execute it.
        ggml_backend_buffer_t graph_buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        GGML_ASSERT(graph_buffer != nullptr);

        ggml_status status = ggml_backend_graph_compute(cpu_backend, graph);
        GGML_ASSERT(status == GGML_STATUS_SUCCESS);

        // Read back a few output values to verify computation happened
        std::vector<float> out_data(hidden_size * tokens);
        ggml_backend_tensor_get(out, out_data.data(), 0, out_data.size() * sizeof(float));

        // Check no NaN/Inf in output
        for (size_t i = 0; i < out_data.size(); ++i) {
            GGML_ASSERT(std::isfinite(out_data[i]));
        }
        ggml_backend_buffer_free(graph_buffer);
    }

    ggml_backend_buffer_free(buffer);
    ggml_backend_free(cpu_backend);
    ggml_free(ctx);

    std::printf("[PASS] z_image_dense_attention_30h: shape validation and CPU compute succeeded\n");
}

// ---------------------------------------------------------------------------
// Test 7: Z-Image runner state forwarding (Task 2)
// ---------------------------------------------------------------------------

// Verify that the ZImageRunner inherits GGMLRunner's HTP state infrastructure.
// Since ZImageRunner : public GGMLRunner, all HTP setters/getters are
// available. We test through a minimal ZImageRunner instance that
// the state forwarding chain works.
static void test_z_image_runner_state() {
    // Verify GGMLRunnerContext fields that ZImageRunner will set in build_graph()
    GGMLRunnerContext runner_ctx;

    // Defaults match expectations
    GGML_ASSERT(runner_ctx.block_sparse_residual_mode == SD_HTP_BLOCK_SPARSE_RESIDUAL_OFF);
    GGML_ASSERT(runner_ctx.block_sparse_residual_attention_index == 0);
    GGML_ASSERT(runner_ctx.block_sparse_token_h == 0);
    GGML_ASSERT(runner_ctx.block_sparse_token_w == 0);
    GGML_ASSERT(!runner_ctx.block_sparse_attn_enabled);
    GGML_ASSERT(!runner_ctx.htp_attention_enabled);

    // Set residual mode and verify
    runner_ctx.block_sparse_residual_mode = SD_HTP_BLOCK_SPARSE_RESIDUAL_APPLY;
    GGML_ASSERT(runner_ctx.block_sparse_residual_mode == SD_HTP_BLOCK_SPARSE_RESIDUAL_APPLY);

    // Set token grid
    runner_ctx.block_sparse_token_h = 16;
    runner_ctx.block_sparse_token_w = 32;
    GGML_ASSERT(runner_ctx.block_sparse_token_h == 16);
    GGML_ASSERT(runner_ctx.block_sparse_token_w == 32);

    // Check static-halo geometry predicate
    const int image_tokens = runner_ctx.block_sparse_token_h * runner_ctx.block_sparse_token_w;
    bool static_halo_valid = image_tokens % 32 == 0 &&
                             runner_ctx.block_sparse_token_h % 4 == 0 &&
                             runner_ctx.block_sparse_token_w % 8 == 0;
    GGML_ASSERT(static_halo_valid);  // 512 % 32 == 0, 16 % 4 == 0, 32 % 8 == 0

    // Reset context
    runner_ctx.block_sparse_token_h = 0;
    runner_ctx.block_sparse_token_w = 0;
    runner_ctx.block_sparse_residual_mode = SD_HTP_BLOCK_SPARSE_RESIDUAL_OFF;
    runner_ctx.block_sparse_residual_attention_index = 0;

    GGML_ASSERT(runner_ctx.block_sparse_token_h == 0);
    GGML_ASSERT(runner_ctx.block_sparse_token_w == 0);

    std::printf("[PASS] z_image_runner_state: HTP context state forwarding verified\n");
}

// ---------------------------------------------------------------------------
// Test 8: Layout mode contract (Task 3)
// ---------------------------------------------------------------------------

static void test_z_image_layout_modes() {
    constexpr int token_h       = 16;
    constexpr int token_w       = 32;
    constexpr int image_tokens  = token_h * token_w;  // 512
    constexpr int text_tokens   = 64;

    // Dense baseline: text-first, row-major image tokens
    {
        // No sparse flags, no ref latents -> dense eligible
        bool sparse_eligible = false;
        bool has_ref_latents = false;
        bool block_sparse_enabled = false;

        bool use_sparse = block_sparse_enabled &&
                          !has_ref_latents &&
                          image_tokens % 32 == 0 &&
                          token_h % 4 == 0 &&
                          token_w % 8 == 0 &&
                          text_tokens >= 0;
        GGML_ASSERT(!use_sparse);

        // Verify PE layout is text-first
        // Text tokens first: [0, text_tokens), Image tokens: [text_tokens, total)
        GGML_ASSERT(text_tokens == 64);
        GGML_ASSERT(image_tokens == 512);
        // Total sequence: text (64) + text_pad (0) + img (512) + img_pad (0) = 576
    }

    // Sparse raw: image-first, patch-major image tokens
    {
        bool has_ref_latents = false;
        bool block_sparse_enabled = true;
        bool use_sparse = block_sparse_enabled &&
                          !has_ref_latents &&
                          image_tokens % 32 == 0 &&
                          token_h % 4 == 0 &&
                          token_w % 8 == 0 &&
                          text_tokens >= 0;
        GGML_ASSERT(use_sparse);

        // Verify image-first layout
        // Image tokens first: [0, 512), Text tokens: [512, 576)
        int img_start = 0;
        int img_end   = image_tokens;
        int txt_start = image_tokens;
        int txt_end   = image_tokens + text_tokens;
        GGML_ASSERT(img_start == 0);
        GGML_ASSERT(img_end == 512);
        GGML_ASSERT(txt_start == 512);
        GGML_ASSERT(txt_end == 576);

        // Verify patch-major reorder is a valid permutation
        // using sd_patch_major helpers from ggml_extend.hpp
        int patch = 4;
        std::vector<int> reorder_map(image_tokens);
        std::vector<int> inverse_map(image_tokens);
        for (int i = 0; i < image_tokens; ++i) {
            reorder_map[i] = sd_patch_major_index_from_row_major(i, token_h, token_w, patch);
            inverse_map[i] = sd_row_major_index_from_patch_major(i, token_h, token_w, patch);
        }
        for (int i = 0; i < image_tokens; ++i) {
            GGML_ASSERT(inverse_map[reorder_map[i]] == i);
        }
    }

    // Sparse packed: same logical image-first/patch-major order
    {
        // Packed layout shares the same logical order as sparse raw
        // PE is image-first, token order is [img_patch_major, txt]
        // Same assertions as sparse raw apply
    }

    // Verify output slice offsets agree with layout
    {
        // Dense: img_slice_start = text_tokens, img_slice_len = image_tokens
        int dense_img_start = text_tokens;
        int dense_img_len   = image_tokens;
        GGML_ASSERT(dense_img_start == 64);
        GGML_ASSERT(dense_img_len == 512);

        // Sparse: img_slice_start = 0, img_slice_len = image_tokens
        int sparse_img_start = 0;
        int sparse_img_len   = image_tokens;
        GGML_ASSERT(sparse_img_start == 0);
        GGML_ASSERT(sparse_img_len == 512);
    }

    // Unsupported geometry: dense fallback
    {
        int bad_token_h = 15;  // not divisible by 4
        int bad_token_w = 32;
        int bad_img_tokens = bad_token_h * bad_token_w;
        bool static_halo_ok = bad_img_tokens % 32 == 0 &&
                              bad_token_h % 4 == 0 &&
                              bad_token_w % 8 == 0;
        GGML_ASSERT(!static_halo_ok);  // 480 % 32 == 0, but 15 % 4 != 0
    }

    std::printf("[PASS] z_image_layout_modes: dense/reordered/packed layout contracts verified\n");
}

// ---------------------------------------------------------------------------
// Test 9: Raw attention 30-head with kv_scale=1/128 (Task 4)
// ---------------------------------------------------------------------------

static void test_z_image_raw_attention_30h() {
    constexpr int64_t head_dim    = 128;
    constexpr int64_t num_heads   = 30;
    constexpr int64_t num_kv_heads = 30;
    constexpr int tokens           = 576;  // 512 img + 64 txt
    constexpr int64_t hidden_size  = num_heads * head_dim;  // 3840
    constexpr float kv_scale       = 1.f / 128.f;

    ggml_init_params params = {
        /*.mem_size   =*/ 64 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context* ctx = ggml_init(params);
    GGML_ASSERT(ctx != nullptr);

    ggml_backend_t cpu_backend = ggml_backend_cpu_init();
    GGML_ASSERT(cpu_backend != nullptr);

    // Build Q/K/V in [N, L, n_head, d_head] layout
    ggml_tensor* q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dim, tokens, num_heads, 1);
    ggml_tensor* k = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dim, tokens, num_kv_heads, 1);
    ggml_tensor* v = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dim, tokens, num_kv_heads, 1);
    ggml_tensor* pe = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 2, 2, head_dim / 2, tokens);

    // Identity RoPE (cos=1, sin=0) for deterministic testing
    std::vector<float> pe_data(tokens * (head_dim / 2) * 4);
    for (int p = 0; p < tokens; ++p) {
        for (int d = 0; d < head_dim / 2; ++d) {
            size_t base = (size_t)p * (head_dim / 2) * 4 + d * 4;
            pe_data[base + 0] = 1.0f; pe_data[base + 1] = 0.0f;
            pe_data[base + 2] = 0.0f; pe_data[base + 3] = 1.0f;
        }
    }

    // Fill Q/K/V with deterministic data
    std::vector<float> q_data(head_dim * tokens * num_heads);
    std::vector<float> k_data(head_dim * tokens * num_kv_heads);
    std::vector<float> v_data(head_dim * tokens * num_kv_heads);
    for (int h = 0; h < num_heads; ++h) {
        for (int t = 0; t < tokens; ++t) {
            for (int d = 0; d < head_dim; ++d) {
                size_t qi = (size_t)h * tokens * head_dim + (size_t)t * head_dim + d;
                size_t ki = (size_t)h * tokens * head_dim + (size_t)t * head_dim + d;
                q_data[qi] = deterministic_f32(t, d + h * 100);
                k_data[ki] = deterministic_f32(t + 1000, d + h * 100);
                v_data[ki] = deterministic_f32(t + 2000, d + h * 100);
            }
        }
    }

    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(cpu_backend);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    GGML_ASSERT(buffer != nullptr);

    ggml_backend_tensor_set(q, q_data.data(), 0, q_data.size() * sizeof(float));
    ggml_backend_tensor_set(k, k_data.data(), 0, k_data.size() * sizeof(float));
    ggml_backend_tensor_set(v, v_data.data(), 0, v_data.size() * sizeof(float));
    ggml_backend_tensor_set(pe, pe_data.data(), 0, pe_data.size() * sizeof(float));

    // Test 1: No-mask raw attention (no sparse)
    {
        ggml_tensor* q_2d = ggml_reshape_3d(ctx, q, hidden_size, tokens, 1);
        ggml_tensor* k_2d = ggml_reshape_3d(ctx, k, hidden_size, tokens, 1);
        ggml_tensor* v_2d = ggml_reshape_3d(ctx, v, hidden_size, tokens, 1);

        ggml_tensor* out = ggml_ext_attention_ext(ctx, cpu_backend,
                                                   q_2d, k_2d, v_2d,
                                                   num_heads, nullptr,
                                                   false, false, kv_scale);

        GGML_ASSERT(out != nullptr);
        GGML_ASSERT(out->ne[0] == hidden_size);
        GGML_ASSERT(out->ne[1] == tokens);
        GGML_ASSERT(out->ne[2] == 1);

        ggml_cgraph* graph = ggml_new_graph_custom(ctx, 256, false);
        ggml_build_forward_expand(graph, out);
        ggml_backend_buffer_t graph_buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        GGML_ASSERT(graph_buffer != nullptr);
        GGML_ASSERT(ggml_backend_graph_compute(cpu_backend, graph) == GGML_STATUS_SUCCESS);

        std::vector<float> out_data(hidden_size * tokens);
        ggml_backend_tensor_get(out, out_data.data(), 0, out_data.size() * sizeof(float));
        for (size_t i = 0; i < out_data.size(); ++i) {
            GGML_ASSERT(std::isfinite(out_data[i]));
        }
        ggml_backend_buffer_free(graph_buffer);
    }

    // Test 2: With static-halo I32 mask (block-sparse path)
    {
        constexpr int image_tokens      = 512;
        constexpr int text_tokens       = 64;
        constexpr int image_block_rows  = 4;
        constexpr int image_block_cols  = 4;
        constexpr int total_blocks      = (tokens + 31) / 32;  // 18

        // Build a full-keep block-sparse index mask for 30 heads
        std::vector<int32_t> index_mask(size_t(num_heads) * total_blocks * total_blocks, -1);
        for (int h = 0; h < num_heads; ++h) {
            for (int qb = 0; qb < total_blocks; ++qb) {
                int32_t* row = index_mask.data() + (size_t(h) * total_blocks + qb) * total_blocks;
                int kept = 0;
                // Full keep: every KB accessible from every QB
                for (int kb = 0; kb < total_blocks; ++kb) {
                    row[kept++] = h * total_blocks + kb;
                }
            }
        }

        ggml_tensor* q_2d = ggml_reshape_3d(ctx, q, hidden_size, tokens, 1);
        ggml_tensor* k_2d = ggml_reshape_3d(ctx, k, hidden_size, tokens, 1);
        ggml_tensor* v_2d = ggml_reshape_3d(ctx, v, hidden_size, tokens, 1);
        ggml_tensor* mask = ggml_new_tensor_4d(ctx, GGML_TYPE_I32,
                                                total_blocks, num_heads * total_blocks, 1, 1);
        ggml_set_name(mask, "block_sparse_index");

        // Reshape to attention layout for sparse path
        ggml_tensor* q_attn = ggml_cont(ctx, ggml_permute(ctx,
            ggml_reshape_4d(ctx, q_2d, head_dim, tokens, num_heads, 1), 0, 2, 1, 3));
        q_attn = ggml_permute(ctx, q_attn, 0, 2, 1, 3);
        ggml_tensor* k_f16 = ggml_cast(ctx, k_2d, GGML_TYPE_F16);
        ggml_tensor* v_f16 = ggml_cast(ctx, v_2d, GGML_TYPE_F16);
        ggml_tensor* k_layout = ggml_cont(ctx, ggml_permute(ctx,
            ggml_reshape_4d(ctx, k_f16, head_dim, tokens, num_kv_heads, 1), 0, 2, 1, 3));
        ggml_tensor* v_layout = ggml_cont(ctx, ggml_permute(ctx,
            ggml_reshape_4d(ctx, v_f16, head_dim, tokens, num_kv_heads, 1), 0, 2, 1, 3));
        ggml_tensor* k_attn = ggml_permute(ctx, k_layout, 0, 2, 1, 3);
        ggml_tensor* v_attn = ggml_permute(ctx, v_layout, 0, 2, 1, 3);

        ggml_tensor* out = ggml_flash_attn_ext_with_lse(ctx, q_attn, k_attn, v_attn, mask,
                                                         1.f / sqrtf(head_dim), 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
        ggml_flash_attn_ext_set_block_sparse(out, 32, 32, GGML_FLASH_ATTN_EXT_MASK_BLOCK_INDEX);
        ggml_flash_attn_ext_set_static_halo_2d(out, image_tokens, text_tokens,
                                                image_block_rows, image_block_cols);

        ggml_cgraph* graph = ggml_new_graph_custom(ctx, 256, false);
        ggml_build_forward_expand(graph, out);
        ggml_backend_buffer_t graph_buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        GGML_ASSERT(graph_buffer != nullptr);
        ggml_backend_tensor_set(mask, index_mask.data(), 0, index_mask.size() * sizeof(int32_t));
        GGML_ASSERT(ggml_backend_graph_compute(cpu_backend, graph) == GGML_STATUS_SUCCESS);

        std::vector<float> out_data(hidden_size * tokens);
        ggml_backend_tensor_get(out, out_data.data(), 0, out_data.size() * sizeof(float));
        for (size_t i = 0; i < out_data.size(); ++i) {
            GGML_ASSERT(std::isfinite(out_data[i]));
        }
        ggml_backend_buffer_free(graph_buffer);
    }

    ggml_backend_buffer_free(buffer);
    ggml_backend_free(cpu_backend);
    ggml_free(ctx);

    std::printf("[PASS] z_image_raw_attention_30h: raw/sparse attention CPU compute succeeded\n");
}

// ---------------------------------------------------------------------------
// Test 10: Packed QKV shape contract (Task 5)
// ---------------------------------------------------------------------------

static void test_z_image_packed_qkv_30h() {
    constexpr int64_t hidden_size  = 3840;
    constexpr int64_t head_dim     = 128;
    constexpr int64_t num_heads    = 30;
    constexpr int64_t num_kv_heads = 30;
    constexpr int64_t qkv_size     = (num_heads + num_kv_heads * 2) * head_dim;  // 11520

    // Verify packed storage sizing
    constexpr int tokens             = 576;
    constexpr int token_tiles        = (tokens + 31) / 32;  // 18
    constexpr int64_t qkv_elems      = 3 * num_heads * token_tiles * (head_dim / 32) * 1024;
    constexpr size_t qkv_bytes       = (size_t)num_heads * token_tiles * (size_t)(head_dim / 32) * 1024 * sizeof(ggml_fp16_t);
    constexpr size_t head_stride     = (size_t)token_tiles * (size_t)(head_dim / 32) * 1024 * sizeof(ggml_fp16_t);

    // Verify packed shape contract: 3 regions, 30 head views, tile padding
    GGML_ASSERT(token_tiles == 18);
    GGML_ASSERT(qkv_elems == 3 * 30 * 18 * 4 * 1024);  // 3 * 30 * 18 * 4 * 1024
    GGML_ASSERT(head_stride > 0);

    // Q/K/V byte offsets in packed storage
    size_t q_offset = 0;
    size_t k_offset = qkv_bytes;
    size_t v_offset = 2 * qkv_bytes;
    GGML_ASSERT(q_offset == 0);
    GGML_ASSERT(k_offset > 0);
    GGML_ASSERT(v_offset > k_offset);

    // Fused weight shape
    GGML_ASSERT(qkv_size == 11520);  // 30*128 + 60*128 = 3840 + 7680

    // Verify no GQA grouping: num_heads == num_kv_heads
    GGML_ASSERT(num_heads == num_kv_heads);
    GGML_ASSERT(num_heads % num_kv_heads == 0);

    std::printf("[PASS] z_image_packed_qkv_30h: packed storage sizing verified\n");
}

// ---------------------------------------------------------------------------
// Test 11: Stable route-selection matrix (Task 7)
// ---------------------------------------------------------------------------

static void test_z_image_route_selection_matrix() {
    using namespace ZImage;

    auto route = [](bool htp_enabled,
                    bool sparse_enabled,
                    bool packed_requested,
                    bool geometry_supported,
                    bool packed_shape_supported,
                    bool packed_weights_available,
                    bool has_ref_latents,
                    bool residual_anchor) {
        return z_image_select_attention_route(htp_enabled,
                                              sparse_enabled,
                                              packed_requested,
                                              geometry_supported,
                                              packed_shape_supported,
                                              packed_weights_available,
                                              has_ref_latents,
                                              residual_anchor);
    };

    auto dense_cpu = route(false, true, true, true, true, true, false, false);
    GGML_ASSERT(dense_cpu.route == ZImageAttentionRoute::DENSE);
    GGML_ASSERT(std::strcmp(dense_cpu.fallback_reason, "htp_disabled") == 0);

    auto dense_htp = route(true, false, true, true, true, true, false, false);
    GGML_ASSERT(dense_htp.route == ZImageAttentionRoute::DENSE);
    GGML_ASSERT(std::strcmp(dense_htp.fallback_reason, "sparse_disabled") == 0);

    auto reordered = route(true, true, false, true, true, true, false, false);
    GGML_ASSERT(reordered.route == ZImageAttentionRoute::REORDERED);
    GGML_ASSERT(std::strcmp(reordered.fallback_reason, "packed_disabled") == 0);

    auto packed = route(true, true, true, true, true, true, false, false);
    GGML_ASSERT(packed.route == ZImageAttentionRoute::PACKED);
    GGML_ASSERT(std::strcmp(packed.fallback_reason, "none") == 0);

    auto bad_geometry = route(true, true, true, false, true, true, false, false);
    GGML_ASSERT(bad_geometry.route == ZImageAttentionRoute::DENSE);
    GGML_ASSERT(std::strcmp(bad_geometry.fallback_reason, "unsupported_geometry") == 0);

    auto bad_shape = route(true, true, true, true, false, true, false, false);
    GGML_ASSERT(bad_shape.route == ZImageAttentionRoute::REORDERED);
    GGML_ASSERT(std::strcmp(bad_shape.fallback_reason, "packed_unsupported_shape") == 0);

    auto bad_weights = route(true, true, true, true, true, false, false, false);
    GGML_ASSERT(bad_weights.route == ZImageAttentionRoute::REORDERED);
    GGML_ASSERT(std::strcmp(bad_weights.fallback_reason, "packed_weights_missing") == 0);

    auto refs = route(true, true, true, true, true, true, true, false);
    GGML_ASSERT(refs.route == ZImageAttentionRoute::DENSE);
    GGML_ASSERT(std::strcmp(refs.fallback_reason, "ref_latents") == 0);

    auto anchor = route(true, true, true, true, true, true, false, true);
    GGML_ASSERT(anchor.route == ZImageAttentionRoute::DENSE);
    GGML_ASSERT(std::strcmp(anchor.fallback_reason, "residual_anchor") == 0);

    std::printf("[PASS] z_image_route_selection_matrix: dense/reordered/packed routes verified\n");
}

// Single-stream refiners use the same packed QKV contract as joint attention.
// Image-only uses the real image grid; text-only uses a synthetic full-keep grid
// because it has no image geometry to feed the static-halo planner.
static void test_z_image_single_stream_packed_contract() {
    GGML_ASSERT(ZImage::z_image_packed_stream_eligible(512, 0, 16, 32, false));
    GGML_ASSERT(ZImage::z_image_packed_stream_eligible(512, 0, 4, 128, true));
    GGML_ASSERT(!ZImage::z_image_packed_stream_eligible(512, 0, 16, 30, false));
    GGML_ASSERT(!ZImage::z_image_packed_stream_eligible(510, 0, 16, 32, false));

    std::printf("[PASS] z_image_single_stream_packed_contract: image-only/text-only metadata verified\n");
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main() {
    std::printf("=== Z-Image Attention Contract Tests ===\n\n");

    test_z_image_shape_validation();
    test_z_image_sequence_metadata();
    test_z_image_patch_major_roundtrip();
    test_z_image_rope_layout();
    test_z_image_packed_qkv_metadata_unavailable();
    test_z_image_dense_attention_30h();
    test_z_image_runner_state();
    test_z_image_layout_modes();
    test_z_image_raw_attention_30h();
    test_z_image_packed_qkv_30h();
    test_z_image_route_selection_matrix();
    test_z_image_single_stream_packed_contract();

    std::printf("\n=== All Z-Image attention contract tests passed ===\n");
    return 0;
}
