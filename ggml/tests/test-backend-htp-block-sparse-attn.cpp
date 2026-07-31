// Runner-style HTP dense vs block-sparse full-keep attention regression.
// Dense uses the same HTP kernel contract as inference (Q/O F32, K/V F16, F16 zero mask).
// Sparse uses the real inference block-sparse path with keep_ratio=1.0 generated from score/mask.

#include "../../src/ggml_extend.hpp"

#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-htp.h>

#include "../src/ggml-htp/op_reg.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

extern "C" int htp_ops_resolve_halo_kv_load_mode(void);

struct EnvGuard {
    const char * name;
    std::string  old_value;
    bool         had_old = false;

    EnvGuard(const char * n, const char * value) : name(n) {
        const char * old = std::getenv(name);
        if (old != nullptr) {
            had_old   = true;
            old_value = old;
        }
        if (value != nullptr) {
            setenv(name, value, 1);
        } else {
            unsetenv(name);
        }
    }

    ~EnvGuard() {
        if (had_old) {
            setenv(name, old_value.c_str(), 1);
        } else {
            unsetenv(name);
        }
    }
};

static void test_block_sparse_keep_ratio_policy() {
    {
        EnvGuard keep_ratio("SD_HTP_BLOCK_SPARSE_KEEP_RATIO", nullptr);
        GGML_ASSERT(sd_htp_block_sparse_keep_ratio() == 0.0f);
        GGML_ASSERT(!sd_htp_block_sparse_score_enabled());
    }
    {
        EnvGuard keep_ratio("SD_HTP_BLOCK_SPARSE_KEEP_RATIO", "0.8");
        GGML_ASSERT(sd_htp_block_sparse_keep_ratio() == 0.8f);
        GGML_ASSERT(sd_htp_block_sparse_score_enabled());
    }
    {
        EnvGuard keep_ratio("SD_HTP_BLOCK_SPARSE_KEEP_RATIO", "1");
        GGML_ASSERT(sd_htp_block_sparse_keep_ratio() == 1.0f);
        GGML_ASSERT(!sd_htp_block_sparse_score_enabled());
    }
}

static void test_block_sparse_mask_format_is_explicit_for_ambiguous_shapes() {
    constexpr int head_dim = 4;
    constexpr int seq_len = 64;
    constexpr int block_size = 32;
    constexpr int n_blocks = seq_len / block_size;

    ggml_init_params params = {
        /*.mem_size   =*/ 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    ggml_context * ctx = ggml_init(params);
    GGML_ASSERT(ctx != nullptr);

    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dim, seq_len, 1, 1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, head_dim, seq_len, 1, 1);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, head_dim, seq_len, 1, 1);

    // With one KV head, an index row can have the same width as the full block matrix.
    // Deliberately misleading names prove that format is not inferred from names either.
    ggml_tensor * matrix_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_blocks, n_blocks);
    ggml_tensor * index_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_blocks, n_blocks);
    ggml_set_name(matrix_mask, "block_sparse_index");
    ggml_set_name(index_mask, "matrix_named_mask");

    ggml_tensor * matrix_attn = ggml_flash_attn_ext(ctx, q, k, v, matrix_mask, 1.0f, 0.0f, 0.0f);
    ggml_tensor * index_attn = ggml_flash_attn_ext(ctx, q, k, v, index_mask, 1.0f, 0.0f, 0.0f);

    GGML_ASSERT(ggml_flash_attn_ext_get_mask_format(matrix_attn) == GGML_FLASH_ATTN_EXT_MASK_DEFAULT);
    GGML_ASSERT(ggml_flash_attn_ext_get_mask_format(index_attn) == GGML_FLASH_ATTN_EXT_MASK_DEFAULT);

    ggml_flash_attn_ext_set_block_sparse(
        matrix_attn, block_size, block_size, GGML_FLASH_ATTN_EXT_MASK_BLOCK_MATRIX);
    ggml_flash_attn_ext_set_block_sparse(
        index_attn, block_size, block_size, GGML_FLASH_ATTN_EXT_MASK_BLOCK_INDEX);

    GGML_ASSERT(matrix_attn->op_params[4] == block_size);
    GGML_ASSERT(matrix_attn->op_params[5] == block_size);
    GGML_ASSERT(index_attn->op_params[4] == block_size);
    GGML_ASSERT(index_attn->op_params[5] == block_size);
    GGML_ASSERT(ggml_flash_attn_ext_get_mask_format(matrix_attn) == GGML_FLASH_ATTN_EXT_MASK_BLOCK_MATRIX);
    GGML_ASSERT(ggml_flash_attn_ext_get_mask_format(index_attn) == GGML_FLASH_ATTN_EXT_MASK_BLOCK_INDEX);

    ggml_free(ctx);
}

static void test_flash_attn_static_halo_metadata() {
    constexpr int head_dim = 4;
    constexpr int image_tokens = 256;
    constexpr int text_tokens = 154;
    constexpr int image_block_rows = 2;
    constexpr int image_block_cols = 4;
    constexpr int seq_len = image_tokens + text_tokens;
    constexpr int block_size = 32;
    constexpr int n_blocks = (seq_len + block_size - 1) / block_size;

    ggml_init_params params = {
        /*.mem_size   =*/ 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    ggml_context * ctx = ggml_init(params);
    GGML_ASSERT(ctx != nullptr);

    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dim, seq_len, 1, 1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, head_dim, seq_len, 1, 1);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, head_dim, seq_len, 1, 1);
    ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_blocks, n_blocks);
    ggml_tensor * attn = ggml_flash_attn_ext(ctx, q, k, v, mask, 1.0f, 0.0f, 0.0f);
    ggml_flash_attn_ext_set_block_sparse(
        attn, block_size, block_size, GGML_FLASH_ATTN_EXT_MASK_BLOCK_INDEX);

    GGML_ASSERT(ggml_flash_attn_ext_get_sparse_policy(attn) ==
                GGML_FLASH_ATTN_EXT_SPARSE_POLICY_NONE);
    GGML_ASSERT(ggml_flash_attn_ext_get_image_tokens(attn) == 0);
    GGML_ASSERT(ggml_flash_attn_ext_get_text_tokens(attn) == 0);
    GGML_ASSERT(ggml_flash_attn_ext_get_image_block_rows(attn) == 0);
    GGML_ASSERT(ggml_flash_attn_ext_get_image_block_cols(attn) == 0);

    ggml_flash_attn_ext_set_static_halo(attn, image_tokens, text_tokens);

    GGML_ASSERT(ggml_flash_attn_ext_get_sparse_policy(attn) ==
                GGML_FLASH_ATTN_EXT_SPARSE_POLICY_STATIC_HALO);
    GGML_ASSERT(ggml_flash_attn_ext_get_image_tokens(attn) == image_tokens);
    GGML_ASSERT(ggml_flash_attn_ext_get_text_tokens(attn) == text_tokens);
    GGML_ASSERT(ggml_flash_attn_ext_get_image_block_rows(attn) == 0);
    GGML_ASSERT(ggml_flash_attn_ext_get_image_block_cols(attn) == 0);

    ggml_flash_attn_ext_set_static_halo_2d(
        attn, image_tokens, text_tokens, image_block_rows, image_block_cols);

    GGML_ASSERT(ggml_flash_attn_ext_get_sparse_policy(attn) ==
                GGML_FLASH_ATTN_EXT_SPARSE_POLICY_STATIC_HALO);
    GGML_ASSERT(ggml_flash_attn_ext_get_image_tokens(attn) == image_tokens);
    GGML_ASSERT(ggml_flash_attn_ext_get_text_tokens(attn) == text_tokens);
    GGML_ASSERT(ggml_flash_attn_ext_get_image_block_rows(attn) == image_block_rows);
    GGML_ASSERT(ggml_flash_attn_ext_get_image_block_cols(attn) == image_block_cols);

    ggml_free(ctx);
}

static void test_halo_kv_load_mode_selector() {
    {
        EnvGuard load_mode("SD_HTP_HALO_KV_LOAD", nullptr);
        GGML_ASSERT(htp_ops_resolve_halo_kv_load_mode() == HTP_HALO_KV_LOAD_L2_PREFETCH);
    }
    {
        EnvGuard load_mode("SD_HTP_HALO_KV_LOAD", "baseline");
        GGML_ASSERT(htp_ops_resolve_halo_kv_load_mode() == HTP_HALO_KV_LOAD_BASELINE);
    }
    {
        EnvGuard load_mode("SD_HTP_HALO_KV_LOAD", "l2fetch");
        GGML_ASSERT(htp_ops_resolve_halo_kv_load_mode() == HTP_HALO_KV_LOAD_L2_PREFETCH);
    }
    {
        EnvGuard load_mode("SD_HTP_HALO_KV_LOAD", "userdma");
        GGML_ASSERT(htp_ops_resolve_halo_kv_load_mode() == HTP_HALO_KV_LOAD_USERDMA_AB);
    }
    {
        EnvGuard load_mode("SD_HTP_HALO_KV_LOAD", "invalid");
        GGML_ASSERT(htp_ops_resolve_halo_kv_load_mode() == -EINVAL);
    }
}

static float gen_value(int t, int h, int d, int seed) {
    return 0.01f * float(((t * 17 + h * 37 + d * 11 + seed * 13) % 257) - 128);
}

static std::vector<float> gen_q(int seq_len, int n_heads, int head_dim, int seed) {
    std::vector<float> data(size_t(seq_len) * size_t(n_heads) * size_t(head_dim));
    for (int t = 0; t < seq_len; ++t) {
        for (int h = 0; h < n_heads; ++h) {
            for (int d = 0; d < head_dim; ++d) {
                data[(size_t(t) * size_t(n_heads) + size_t(h)) * size_t(head_dim) + size_t(d)] =
                    gen_value(t, h, d, seed);
            }
        }
    }
    return data;
}

static std::vector<ggml_fp16_t> gen_kv_f16(int seq_len, int n_heads, int head_dim, int seed) {
    std::vector<ggml_fp16_t> data(size_t(seq_len) * size_t(n_heads) * size_t(head_dim));
    for (int t = 0; t < seq_len; ++t) {
        for (int h = 0; h < n_heads; ++h) {
            for (int d = 0; d < head_dim; ++d) {
                data[(size_t(t) * size_t(n_heads) + size_t(h)) * size_t(head_dim) + size_t(d)] =
                    ggml_fp32_to_fp16(gen_value(t, h, d, seed));
            }
        }
    }
    return data;
}

static double element_abs_diff(float a, float b) {
    const double da = double(a);
    const double db = double(b);
    if (da == db) {
        return 0.0;
    }
    if (!std::isfinite(da) || !std::isfinite(db)) {
        return std::numeric_limits<double>::infinity();
    }
    const double diff = std::fabs(da - db);
    return std::isfinite(diff) ? diff : std::numeric_limits<double>::infinity();
}

static double max_abs_diff(const std::vector<float> & a, const std::vector<float> & b, size_t * worst = nullptr) {
    GGML_ASSERT(a.size() == b.size());
    double max_diff = 0.0;
    size_t worst_idx = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double diff = element_abs_diff(a[i], b[i]);
        if (diff > max_diff) {
            max_diff = diff;
            worst_idx = i;
        }
    }
    if (worst != nullptr) {
        *worst = worst_idx;
    }
    return max_diff;
}

struct DiffStats {
    double max_abs = 0.0;
    double mean_abs = 0.0;
    double rms_abs = 0.0;
    double ref_rms = 0.0;
    double rel_rms = 0.0;
    size_t worst = 0;
};

static DiffStats diff_stats(const std::vector<float> & ref, const std::vector<float> & got) {
    GGML_ASSERT(ref.size() == got.size());
    DiffStats stats;
    double sum_abs = 0.0;
    double sum_sq = 0.0;
    double ref_sq = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double abs_diff = element_abs_diff(ref[i], got[i]);
        if (abs_diff > stats.max_abs) {
            stats.max_abs = abs_diff;
            stats.worst = i;
        }
        sum_abs += abs_diff;
        sum_sq += abs_diff * abs_diff;
        const double ref_value = double(ref[i]);
        ref_sq += std::isfinite(ref_value) ? ref_value * ref_value : std::numeric_limits<double>::infinity();
    }
    if (!ref.empty()) {
        stats.mean_abs = sum_abs / double(ref.size());
        stats.rms_abs = std::sqrt(sum_sq / double(ref.size()));
        stats.ref_rms = std::sqrt(ref_sq / double(ref.size()));
        if (stats.rms_abs == 0.0) {
            stats.rel_rms = 0.0;
        } else if (std::isfinite(stats.ref_rms) && stats.ref_rms > 0.0) {
            stats.rel_rms = stats.rms_abs / stats.ref_rms;
        } else {
            stats.rel_rms = std::numeric_limits<double>::infinity();
        }
    }
    return stats;
}

static void test_diff_helpers_handle_non_finite_values() {
    const float inf = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();

    const std::vector<float> equal_infinities = {-inf, inf};
    GGML_ASSERT(max_abs_diff(equal_infinities, equal_infinities) == 0.0);
    const DiffStats equal_stats = diff_stats(equal_infinities, equal_infinities);
    GGML_ASSERT(equal_stats.max_abs == 0.0);
    GGML_ASSERT(equal_stats.rms_abs == 0.0);

    GGML_ASSERT(std::isinf(max_abs_diff({inf}, {-inf})));
    GGML_ASSERT(std::isinf(max_abs_diff({nan}, {nan})));
    const DiffStats nan_stats = diff_stats({nan}, {nan});
    GGML_ASSERT(std::isinf(nan_stats.max_abs));
    GGML_ASSERT(std::isinf(nan_stats.rms_abs));
}

struct PackedWithLseDiffStats {
    double out_max_abs = 0.0;
    double lse_max_abs = 0.0;
    size_t out_worst = 0;
    size_t lse_worst = 0;
};

static PackedWithLseDiffStats packed_with_lse_diff_stats(const std::vector<float> & ref_out,
                                                         const std::vector<float> & ref_lse,
                                                         const std::vector<float> & got_packed,
                                                         int qo_len,
                                                         int n_heads,
                                                         int head_dim) {
    GGML_ASSERT(ref_out.size() == size_t(qo_len) * size_t(n_heads) * size_t(head_dim));
    GGML_ASSERT(ref_lse.size() == size_t(qo_len) * size_t(n_heads));
    GGML_ASSERT(got_packed.size() == size_t(qo_len) * size_t(n_heads) * size_t(head_dim + 1));

    PackedWithLseDiffStats stats;
    for (int qt = 0; qt < qo_len; ++qt) {
        for (int h = 0; h < n_heads; ++h) {
            const size_t packed_base = (size_t(qt) * size_t(n_heads) + size_t(h)) * size_t(head_dim + 1);
            const size_t out_base = (size_t(qt) * size_t(n_heads) + size_t(h)) * size_t(head_dim);
            for (int d = 0; d < head_dim; ++d) {
                const size_t out_idx = out_base + size_t(d);
                const double diff = element_abs_diff(ref_out[out_idx], got_packed[packed_base + size_t(d)]);
                if (diff > stats.out_max_abs) {
                    stats.out_max_abs = diff;
                    stats.out_worst = out_idx;
                }
            }

            const size_t lse_idx = size_t(qt) * size_t(n_heads) + size_t(h);
            const double lse_diff = element_abs_diff(ref_lse[lse_idx], got_packed[packed_base + size_t(head_dim)]);
            if (lse_diff > stats.lse_max_abs) {
                stats.lse_max_abs = lse_diff;
                stats.lse_worst = lse_idx;
            }
        }
    }
    return stats;
}

static double packed_linear_prefix_diff(const std::vector<float> & ref_linear,
                                        const std::vector<float> & got_packed,
                                        size_t * worst = nullptr) {
    GGML_ASSERT(got_packed.size() >= ref_linear.size());
    double max_diff = 0.0;
    size_t worst_idx = 0;
    for (size_t i = 0; i < ref_linear.size(); ++i) {
        const double diff = element_abs_diff(ref_linear[i], got_packed[i]);
        if (diff > max_diff) {
            max_diff = diff;
            worst_idx = i;
        }
    }
    if (worst != nullptr) {
        *worst = worst_idx;
    }
    return max_diff;
}

static void test_packed_diff_helpers_handle_non_finite_values() {
    const float inf = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();

    const std::vector<float> ref_out = {1.0f, -2.0f};
    const std::vector<float> ref_lse = {-inf};
    const std::vector<float> matching_packed = {1.0f, -2.0f, -inf};
    const PackedWithLseDiffStats matching_stats =
        packed_with_lse_diff_stats(ref_out, ref_lse, matching_packed, 1, 1, 2);
    GGML_ASSERT(matching_stats.out_max_abs == 0.0);
    GGML_ASSERT(matching_stats.lse_max_abs == 0.0);

    const PackedWithLseDiffStats out_nan_stats =
        packed_with_lse_diff_stats(ref_out, ref_lse, {1.0f, nan, -inf}, 1, 1, 2);
    GGML_ASSERT(std::isinf(out_nan_stats.out_max_abs));

    const PackedWithLseDiffStats lse_nan_stats =
        packed_with_lse_diff_stats(ref_out, ref_lse, {1.0f, -2.0f, nan}, 1, 1, 2);
    GGML_ASSERT(std::isinf(lse_nan_stats.lse_max_abs));
    GGML_ASSERT(std::isinf(packed_linear_prefix_diff(ref_out, {1.0f, nan, -inf})));
}

struct RefAttentionWithLse {
    std::vector<float> out;
    std::vector<float> lse;
};

static RefAttentionWithLse reference_attention_with_lse(int qo_len,
                                                        int kv_len,
                                                        int n_heads,
                                                        int n_kv_heads,
                                                        int head_dim,
                                                        const std::vector<float> & q,
                                                        const std::vector<ggml_fp16_t> & k,
                                                        const std::vector<ggml_fp16_t> & v,
                                                        const std::vector<float> * dense_mask) {
    GGML_ASSERT(int(q.size()) == qo_len * n_heads * head_dim);
    GGML_ASSERT(int(k.size()) == kv_len * n_kv_heads * head_dim);
    GGML_ASSERT(int(v.size()) == kv_len * n_kv_heads * head_dim);
    if (dense_mask != nullptr) {
        GGML_ASSERT(int(dense_mask->size()) == kv_len * qo_len * n_heads);
    }

    RefAttentionWithLse result;
    result.out.resize(size_t(qo_len) * size_t(n_heads) * size_t(head_dim));
    result.lse.resize(size_t(qo_len) * size_t(n_heads));

    const float scale = 1.0f / std::sqrt(float(head_dim));
    const int gqa = std::max(1, n_heads / n_kv_heads);
    std::vector<float> scores;
    scores.resize(size_t(kv_len));

    for (int qt = 0; qt < qo_len; ++qt) {
        for (int h = 0; h < n_heads; ++h) {
            const int kv_head = std::min(n_kv_heads - 1, h / gqa);
            float max_score = -std::numeric_limits<float>::infinity();
            for (int kt = 0; kt < kv_len; ++kt) {
                float score = 0.0f;
                for (int d = 0; d < head_dim; ++d) {
                    const float qv = q[(size_t(qt) * size_t(n_heads) + size_t(h)) * size_t(head_dim) + size_t(d)];
                    const float kv = ggml_fp16_to_fp32(k[(size_t(kt) * size_t(n_kv_heads) + size_t(kv_head)) * size_t(head_dim) + size_t(d)]);
                    score += qv * kv;
                }
                score *= scale;
                if (dense_mask != nullptr) {
                    score += (*dense_mask)[(size_t(h) * size_t(qo_len) + size_t(qt)) * size_t(kv_len) + size_t(kt)];
                }
                scores[size_t(kt)] = score;
                max_score = std::max(max_score, score);
            }

            float exp_sum = 0.0f;
            if (std::isfinite(max_score)) {
                for (int kt = 0; kt < kv_len; ++kt) {
                    exp_sum += std::exp(scores[size_t(kt)] - max_score);
                }
            }
            const float lse = exp_sum > 0.0f ? max_score + std::log(exp_sum)
                                             : -std::numeric_limits<float>::infinity();
            result.lse[size_t(qt) * size_t(n_heads) + size_t(h)] = lse;

            for (int d = 0; d < head_dim; ++d) {
                double out = 0.0;
                if (exp_sum > 0.0f) {
                    for (int kt = 0; kt < kv_len; ++kt) {
                        const float weight = std::exp(scores[size_t(kt)] - lse);
                        const float vv = ggml_fp16_to_fp32(v[(size_t(kt) * size_t(n_kv_heads) + size_t(kv_head)) * size_t(head_dim) + size_t(d)]);
                        out += double(weight) * double(vv);
                    }
                }
                result.out[(size_t(qt) * size_t(n_heads) + size_t(h)) * size_t(head_dim) + size_t(d)] =
                    float(out);
            }
        }
    }

    return result;
}

struct AttentionRunner : GGMLRunner {
    AttentionRunner(ggml_backend_t runtime_backend, ggml_backend_t htp_backend)
        : GGMLRunner(runtime_backend, runtime_backend) {
        set_htp_backend(htp_backend);
    }

    std::string get_desc() override {
        return "htp-block-sparse-attn";
    }

    std::vector<float> run_cache_output_lifetime_test(int n_tensors, int n_elements) {
        GGML_ASSERT(n_tensors > 1 && n_elements > 0);
        cache_tensor_map.clear();
        cache_backed_tensor_name_map.clear();
        free_cache_ctx_and_buffer();

        std::vector<std::string> names;
        names.reserve(size_t(n_tensors));
        auto get_graph = [&]() -> ggml_cgraph * {
            GGMLRunnerContext runner_ctx = get_context();
            ggml_cgraph * graph = ggml_new_graph(runner_ctx.ggml_ctx);
            for (int i = 0; i < n_tensors; ++i) {
                names.push_back("cache_output_lifetime_" + std::to_string(i));
                ggml_tensor * value = ggml_ext_full(runner_ctx.ggml_ctx,
                                                    float(i + 1),
                                                    n_elements,
                                                    1,
                                                    1,
                                                    1);
                ggml_set_name(value, names.back().c_str());
                runner_ctx.cache_tensor(names.back(), value);
                ggml_tensor * consumer = ggml_ext_scale(runner_ctx.ggml_ctx, value, -1.0f);
                ggml_build_forward_expand(graph, consumer);
            }
            return graph;
        };

        const auto result = GGMLRunner::compute<float>(get_graph, 4, true, true);
        GGML_ASSERT(result.has_value());

        std::vector<float> first_values;
        first_values.reserve(size_t(n_tensors));
        for (const std::string & name : names) {
            ggml_tensor * cached = get_cache_tensor_by_name(name);
            GGML_ASSERT(cached != nullptr && ggml_nelements(cached) == n_elements);
            std::vector<float> values(static_cast<size_t>(n_elements));
            ggml_backend_tensor_get(cached, values.data(), 0, values.size() * sizeof(float));
            first_values.push_back(values.front());
        }

        cache_tensor_map.clear();
        cache_backed_tensor_name_map.clear();
        free_cache_ctx_and_buffer();
        return first_values;
    }

    std::vector<float> run(int n_threads,
                           int qo_len,
                           int kv_len,
                           int n_heads,
                           int n_kv_heads,
                           int head_dim,
                           const std::vector<float> & q_data,
                           const std::vector<ggml_fp16_t> & k_data,
                           const std::vector<ggml_fp16_t> & v_data,
                           bool block_sparse_enabled,
                           bool force_full_keep,
                           float keep_ratio_value = 1.0f) {
        set_flash_attention_enabled(true);
        set_htp_attention_enabled(true);
        set_block_sparse_attention_enabled(block_sparse_enabled);

        char keep_ratio_buf[32];
        std::snprintf(keep_ratio_buf, sizeof(keep_ratio_buf), "%.8g", double(keep_ratio_value));

        EnvGuard htp_ops("SD_HTP_OPS", "1");
        EnvGuard htp_compute("SD_HTP_COMPUTE", "1");
        EnvGuard htp_attention("SD_HTP_ATTENTION", "1");
        EnvGuard block_sparse_attn("SD_HTP_BLOCK_SPARSE_ATTN", block_sparse_enabled ? "1" : "0");
        EnvGuard full_keep_flag("SD_HTP_BLOCK_SPARSE_FORCE_FULL_KEEP", force_full_keep ? "1" : "0");
        EnvGuard keep_ratio("SD_HTP_BLOCK_SPARSE_KEEP_RATIO", block_sparse_enabled ? keep_ratio_buf : "1");

        sd::Tensor<float> q_tensor({n_heads * head_dim, qo_len, 1}, q_data);
        sd::Tensor<ggml_fp16_t> k_tensor({n_kv_heads * head_dim, kv_len, 1}, k_data);
        sd::Tensor<ggml_fp16_t> v_tensor({n_kv_heads * head_dim, kv_len, 1}, v_data);
        std::vector<float> zero_mask_data(size_t(kv_len) * size_t(qo_len), 0.0f);
        sd::Tensor<float> zero_mask({kv_len, qo_len, 1, 1}, zero_mask_data);

        auto get_graph = [&]() -> ggml_cgraph * {
            GGMLRunnerContext runner_ctx = get_context();

            ggml_tensor * q = make_input(q_tensor);
            ggml_tensor * k = make_input(k_tensor);
            ggml_tensor * v = make_input(v_tensor);
            ggml_tensor * m = block_sparse_enabled ? nullptr : make_input(zero_mask);

            ggml_tensor * out = ggml_ext_attention_ext(runner_ctx.ggml_ctx,
                                                       runner_ctx.backend,
                                                       q,
                                                       k,
                                                       v,
                                                       n_heads,
                                                       m,
                                                       false,
                                                       runner_ctx.flash_attn_enabled,
                                                       1.0f,
                                                       runner_ctx.block_sparse_attn_enabled);
            GGML_ASSERT(out != nullptr);

            ggml_cgraph * graph = ggml_new_graph(runner_ctx.ggml_ctx);
            ggml_build_forward_expand(graph, out);
            return graph;
        };

        auto result = GGMLRunner::compute<float>(get_graph, n_threads, true);
        GGML_ASSERT(result.has_value());
        return result->values();
    }

    std::vector<int32_t> make_block_index(int n_threads,
                                          int qo_len,
                                          int kv_len,
                                          int n_heads,
                                          int n_kv_heads,
                                          int head_dim,
                                          const std::vector<float> & q_data,
                                          const std::vector<ggml_fp16_t> & k_data,
                                          float keep_ratio_value,
                                          int protected_start = 0,
                                          int protected_len = 0,
                                          bool production_index = false) {
        set_flash_attention_enabled(true);
        set_htp_attention_enabled(true);
        set_block_sparse_attention_enabled(true);

        char keep_ratio_buf[32];
        std::snprintf(keep_ratio_buf, sizeof(keep_ratio_buf), "%.8g", double(keep_ratio_value));

        EnvGuard htp_ops("SD_HTP_OPS", "1");
        EnvGuard htp_compute("SD_HTP_COMPUTE", "1");
        EnvGuard htp_attention("SD_HTP_ATTENTION", "1");
        EnvGuard keep_ratio("SD_HTP_BLOCK_SPARSE_KEEP_RATIO", keep_ratio_buf);

        sd::Tensor<float> q_tensor({n_heads * head_dim, qo_len, 1}, q_data);
        sd::Tensor<ggml_fp16_t> k_tensor({n_kv_heads * head_dim, kv_len, 1}, k_data);

        auto get_graph = [&]() -> ggml_cgraph * {
            GGMLRunnerContext runner_ctx = get_context();

            ggml_tensor * q_orig = make_input(q_tensor);
            ggml_tensor * k_orig = make_input(k_tensor);

            ggml_tensor * q = ggml_reshape_4d(runner_ctx.ggml_ctx, q_orig, head_dim, n_heads, qo_len, 1);
            q = ggml_ext_cont(runner_ctx.ggml_ctx, ggml_permute(runner_ctx.ggml_ctx, q, 0, 2, 1, 3));
            q = ggml_reshape_3d(runner_ctx.ggml_ctx, q, head_dim, qo_len, n_heads);

            ggml_tensor * k = ggml_reshape_4d(runner_ctx.ggml_ctx, k_orig, head_dim, n_kv_heads, kv_len, 1);
            k = ggml_ext_cont(runner_ctx.ggml_ctx, ggml_permute(runner_ctx.ggml_ctx, k, 0, 2, 1, 3));
            k = ggml_reshape_3d(runner_ctx.ggml_ctx, k, head_dim, kv_len, n_kv_heads);

            const int block_q = production_index ? SD_HTP_BLOCK_SPARSE_PATCH_BLOCK :
                                                   SD_HTP_BLOCK_SPARSE_MASK_BLOCK;
            const int block_k = production_index ? SD_HTP_BLOCK_SPARSE_PATCH_BLOCK :
                                                   SD_HTP_BLOCK_SPARSE_MASK_BLOCK;
            ggml_tensor * q_mean = ggml_ext_block_sparse_q_mean_node(runner_ctx.ggml_ctx, q, block_q);
            ggml_tensor * k_mean = ggml_ext_block_sparse_k_mean_node(runner_ctx.ggml_ctx, k, block_k);
            ggml_tensor * score  = ggml_ext_block_sparse_score_node(runner_ctx.ggml_ctx, q_mean, k_mean, block_q, block_k, false);
            ggml_tensor * index = production_index ?
                ggml_ext_block_sparse_index_32_from_16_score_node(runner_ctx.ggml_ctx,
                                                                   score,
                                                                   false,
                                                                   protected_start,
                                                                   protected_len,
                                                                   n_heads,
                                                                   n_kv_heads) :
                ggml_ext_block_sparse_index_from_score_node(runner_ctx.ggml_ctx,
                                                             score,
                                                             block_q,
                                                             block_k,
                                                             false,
                                                             protected_start,
                                                             protected_len,
                                                             n_heads,
                                                             n_kv_heads);

            ggml_cgraph * graph = ggml_new_graph(runner_ctx.ggml_ctx);
            ggml_build_forward_expand(graph, index);
            return graph;
        };

        auto result = GGMLRunner::compute<int32_t>(get_graph, n_threads, true);
        GGML_ASSERT(result.has_value());
        return result->values();
    }

    std::vector<float> run_cpu_dense_masked(int n_threads,
                                            int qo_len,
                                            int kv_len,
                                            int n_heads,
                                            int n_kv_heads,
                                            int head_dim,
                                            const std::vector<float> & q_data,
                                            const std::vector<ggml_fp16_t> & k_data,
                                            const std::vector<ggml_fp16_t> & v_data,
                                            const std::vector<float> & dense_mask_data) {
        set_flash_attention_enabled(false);
        set_htp_attention_enabled(false);
        set_block_sparse_attention_enabled(false);

        EnvGuard htp_ops("SD_HTP_OPS", "0");
        EnvGuard htp_compute("SD_HTP_COMPUTE", "0");
        EnvGuard htp_attention("SD_HTP_ATTENTION", "0");

        sd::Tensor<float> q_tensor({n_heads * head_dim, qo_len, 1}, q_data);
        sd::Tensor<ggml_fp16_t> k_tensor({n_kv_heads * head_dim, kv_len, 1}, k_data);
        sd::Tensor<ggml_fp16_t> v_tensor({n_kv_heads * head_dim, kv_len, 1}, v_data);
        sd::Tensor<float> mask_tensor({kv_len, qo_len, n_heads, 1}, dense_mask_data);

        auto get_graph = [&]() -> ggml_cgraph * {
            GGMLRunnerContext runner_ctx = get_context();

            ggml_tensor * q = make_input(q_tensor);
            ggml_tensor * k = make_input(k_tensor);
            ggml_tensor * v = make_input(v_tensor);
            ggml_tensor * m = make_input(mask_tensor);

            ggml_tensor * out = ggml_ext_attention_ext(runner_ctx.ggml_ctx,
                                                       runner_ctx.backend,
                                                       q,
                                                       k,
                                                       v,
                                                       n_heads,
                                                       m,
                                                       false,
                                                       false,
                                                       1.0f,
                                                       false);
            GGML_ASSERT(out != nullptr);

            ggml_cgraph * graph = ggml_new_graph(runner_ctx.ggml_ctx);
            ggml_build_forward_expand(graph, out);
            return graph;
        };

        auto result = GGMLRunner::compute<float>(get_graph, n_threads, true);
        GGML_ASSERT(result.has_value());
        return result->values();
    }

    std::vector<float> run_htp_dense_f16_mask(int n_threads,
                                              int qo_len,
                                              int kv_len,
                                              int n_heads,
                                              int n_kv_heads,
                                              int head_dim,
                                              const std::vector<float> & q_data,
                                              const std::vector<ggml_fp16_t> & k_data,
                                              const std::vector<ggml_fp16_t> & v_data,
                                              const std::vector<ggml_fp16_t> & dense_mask_data) {
        set_flash_attention_enabled(true);
        set_htp_attention_enabled(true);
        set_block_sparse_attention_enabled(false);

        EnvGuard htp_ops("SD_HTP_OPS", "1");
        EnvGuard htp_compute("SD_HTP_COMPUTE", "1");
        EnvGuard htp_attention("SD_HTP_ATTENTION", "1");

        const int kv_pad_len = ((kv_len + 63) / 64) * 64;
        GGML_ASSERT(dense_mask_data.size() == size_t(kv_pad_len) * size_t(qo_len));

        sd::Tensor<float> q_tensor({n_heads * head_dim, qo_len, 1}, q_data);
        sd::Tensor<ggml_fp16_t> k_tensor({n_kv_heads * head_dim, kv_len, 1}, k_data);
        sd::Tensor<ggml_fp16_t> v_tensor({n_kv_heads * head_dim, kv_len, 1}, v_data);
        sd::Tensor<ggml_fp16_t> mask_tensor({kv_pad_len, qo_len, 1, 1}, dense_mask_data);

        auto get_graph = [&]() -> ggml_cgraph * {
            GGMLRunnerContext runner_ctx = get_context();

            ggml_tensor * q_orig = make_input(q_tensor);
            ggml_tensor * k_orig = make_input(k_tensor);
            ggml_tensor * v_orig = make_input(v_tensor);
            ggml_tensor * mask   = make_input(mask_tensor);

            ggml_tensor * q3 = ggml_reshape_3d(runner_ctx.ggml_ctx, q_orig, head_dim, n_heads, qo_len);
            ggml_tensor * q  = ggml_permute(runner_ctx.ggml_ctx, q3, 0, 2, 1, 3);
            ggml_tensor * k_base = ggml_reshape_3d(runner_ctx.ggml_ctx, k_orig, head_dim, n_kv_heads, kv_len);
            ggml_tensor * v_base = ggml_reshape_3d(runner_ctx.ggml_ctx, v_orig, head_dim, n_kv_heads, kv_len);
            ggml_tensor * k = ggml_permute(runner_ctx.ggml_ctx, k_base, 0, 2, 1, 3);
            ggml_tensor * v = ggml_permute(runner_ctx.ggml_ctx, v_base, 0, 2, 1, 3);

            ggml_tensor * out = ggml_flash_attn_ext(runner_ctx.ggml_ctx,
                                                    q,
                                                    k,
                                                    v,
                                                    mask,
                                                    1.0f / std::sqrt(float(head_dim)),
                                                    0.0f,
                                                    0.0f);
            ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
            out = ggml_view_3d(runner_ctx.ggml_ctx, out, head_dim, n_heads, qo_len, out->nb[1], out->nb[2], 0);
            out = ggml_ext_cont(runner_ctx.ggml_ctx, out);
            out = ggml_reshape_3d(runner_ctx.ggml_ctx, out, head_dim * n_heads, qo_len, 1);

            ggml_cgraph * graph = ggml_new_graph(runner_ctx.ggml_ctx);
            ggml_build_forward_expand(graph, out);
            return graph;
        };

        auto result = GGMLRunner::compute<float>(get_graph, n_threads, true);
        GGML_ASSERT(result.has_value());
        return result->values();
    }

    std::vector<float> run_htp_sparse_index(int n_threads,
                                            int qo_len,
                                            int kv_len,
                                            int n_heads,
                                            int n_kv_heads,
                                            int head_dim,
                                            const std::vector<float> & q_data,
                                            const std::vector<ggml_fp16_t> & k_data,
                                            const std::vector<ggml_fp16_t> & v_data,
                                            const std::vector<int32_t> & block_index,
                                            int static_image_tokens = 0,
                                            int static_text_tokens = 0,
                                            int static_image_block_rows = 0,
                                            int static_image_block_cols = 0) {
        set_flash_attention_enabled(true);
        set_htp_attention_enabled(true);
        set_block_sparse_attention_enabled(true);

        EnvGuard htp_ops("SD_HTP_OPS", "1");
        EnvGuard htp_compute("SD_HTP_COMPUTE", "1");
        EnvGuard htp_attention("SD_HTP_ATTENTION", "1");

        const int block_q = SD_HTP_BLOCK_SPARSE_MASK_BLOCK;
        const int block_k = SD_HTP_BLOCK_SPARSE_MASK_BLOCK;
        const int n_q_blocks = n_heads * ((qo_len + block_q - 1) / block_q);
        const int max_kept = int(block_index.size()) / n_q_blocks;

        sd::Tensor<float> q_tensor({n_heads * head_dim, qo_len, 1}, q_data);
        sd::Tensor<ggml_fp16_t> k_tensor({n_kv_heads * head_dim, kv_len, 1}, k_data);
        sd::Tensor<ggml_fp16_t> v_tensor({n_kv_heads * head_dim, kv_len, 1}, v_data);
        sd::Tensor<int32_t> index_tensor({max_kept, n_q_blocks, 1, 1}, block_index);

        auto get_graph = [&]() -> ggml_cgraph * {
            GGMLRunnerContext runner_ctx = get_context();

            ggml_tensor * q_orig = make_input(q_tensor);
            ggml_tensor * k_orig = make_input(k_tensor);
            ggml_tensor * v_orig = make_input(v_tensor);
            ggml_tensor * index  = make_input(index_tensor);
            ggml_set_name(index, "block_sparse_index");

            ggml_tensor * q3 = ggml_reshape_3d(runner_ctx.ggml_ctx, q_orig, head_dim, n_heads, qo_len);
            ggml_tensor * q  = ggml_permute(runner_ctx.ggml_ctx, q3, 0, 2, 1, 3);
            ggml_tensor * k_base = ggml_reshape_3d(runner_ctx.ggml_ctx, k_orig, head_dim, n_kv_heads, kv_len);
            ggml_tensor * v_base = ggml_reshape_3d(runner_ctx.ggml_ctx, v_orig, head_dim, n_kv_heads, kv_len);
            ggml_tensor * k = ggml_permute(runner_ctx.ggml_ctx, k_base, 0, 2, 1, 3);
            ggml_tensor * v = ggml_permute(runner_ctx.ggml_ctx, v_base, 0, 2, 1, 3);

            ggml_tensor * out = ggml_flash_attn_ext(runner_ctx.ggml_ctx,
                                                    q,
                                                    k,
                                                    v,
                                                    index,
                                                    1.0f / std::sqrt(float(head_dim)),
                                                    0.0f,
                                                    0.0f);
            ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
            ggml_flash_attn_ext_set_block_sparse(
                out, block_q, block_k, GGML_FLASH_ATTN_EXT_MASK_BLOCK_INDEX);
            if (static_image_tokens > 0) {
                if (static_image_block_rows > 0 && static_image_block_cols > 0) {
                    ggml_flash_attn_ext_set_static_halo_2d(out,
                                                           static_image_tokens,
                                                           static_text_tokens,
                                                           static_image_block_rows,
                                                           static_image_block_cols);
                } else {
                    ggml_flash_attn_ext_set_static_halo(out, static_image_tokens, static_text_tokens);
                }
            }
            out = ggml_view_3d(runner_ctx.ggml_ctx, out, head_dim, n_heads, qo_len, out->nb[1], out->nb[2], 0);
            out = ggml_ext_cont(runner_ctx.ggml_ctx, out);
            out = ggml_reshape_3d(runner_ctx.ggml_ctx, out, head_dim * n_heads, qo_len, 1);

            ggml_cgraph * graph = ggml_new_graph(runner_ctx.ggml_ctx);
            ggml_build_forward_expand(graph, out);
            return graph;
        };

        auto result = GGMLRunner::compute<float>(get_graph, n_threads, true);
        GGML_ASSERT(result.has_value());
        return result->values();
    }

    std::vector<float> run_htp_flash_with_lse_packed(int n_threads,
                                                     int qo_len,
                                                     int kv_len,
                                                     int n_heads,
                                                     int n_kv_heads,
                                                     int head_dim,
                                                     const std::vector<float> & q_data,
                                                     const std::vector<ggml_fp16_t> & k_data,
                                                     const std::vector<ggml_fp16_t> & v_data,
                                                     const std::vector<int32_t> * block_index,
                                                     int static_image_tokens = 0,
                                                     int static_text_tokens = 0,
                                                     int static_image_block_rows = 0,
                                                     int static_image_block_cols = 0) {
        set_flash_attention_enabled(true);
        set_htp_attention_enabled(true);
        set_block_sparse_attention_enabled(block_index != nullptr);

        EnvGuard htp_ops("SD_HTP_OPS", "1");
        EnvGuard htp_compute("SD_HTP_COMPUTE", "1");
        EnvGuard htp_attention("SD_HTP_ATTENTION", "1");

        const int block_q = SD_HTP_BLOCK_SPARSE_MASK_BLOCK;
        const int block_k = SD_HTP_BLOCK_SPARSE_MASK_BLOCK;
        const int n_q_blocks = n_heads * ((qo_len + block_q - 1) / block_q);
        const int max_kept = block_index != nullptr ? int(block_index->size()) / n_q_blocks : 1;

        const int kv_pad_len = ((kv_len + 63) / 64) * 64;
        std::vector<ggml_fp16_t> zero_mask_data(size_t(kv_pad_len) * size_t(qo_len),
                                                ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity()));
        for (int qt = 0; qt < qo_len; ++qt) {
            for (int kt = 0; kt < kv_len; ++kt) {
                zero_mask_data[size_t(qt) * size_t(kv_pad_len) + size_t(kt)] = ggml_fp32_to_fp16(0.0f);
            }
        }
        std::vector<int32_t> dummy_index(size_t(n_q_blocks), 0);
        const std::vector<int32_t> & index_data = block_index != nullptr ? *block_index : dummy_index;

        sd::Tensor<float> q_tensor({n_heads * head_dim, qo_len, 1}, q_data);
        sd::Tensor<ggml_fp16_t> k_tensor({n_kv_heads * head_dim, kv_len, 1}, k_data);
        sd::Tensor<ggml_fp16_t> v_tensor({n_kv_heads * head_dim, kv_len, 1}, v_data);
        sd::Tensor<ggml_fp16_t> zero_mask({kv_pad_len, qo_len, 1, 1}, zero_mask_data);
        sd::Tensor<int32_t> index_tensor({max_kept, n_q_blocks, 1, 1}, index_data);

        auto get_graph = [&]() -> ggml_cgraph * {
            GGMLRunnerContext runner_ctx = get_context();

            ggml_tensor * q_orig = make_input(q_tensor);
            ggml_tensor * k_orig = make_input(k_tensor);
            ggml_tensor * v_orig = make_input(v_tensor);

            ggml_tensor * q3 = ggml_reshape_3d(runner_ctx.ggml_ctx, q_orig, head_dim, n_heads, qo_len);
            ggml_tensor * q  = ggml_permute(runner_ctx.ggml_ctx, q3, 0, 2, 1, 3);
            ggml_tensor * k_base = ggml_reshape_3d(runner_ctx.ggml_ctx, k_orig, head_dim, n_kv_heads, kv_len);
            ggml_tensor * v_base = ggml_reshape_3d(runner_ctx.ggml_ctx, v_orig, head_dim, n_kv_heads, kv_len);
            ggml_tensor * k = ggml_permute(runner_ctx.ggml_ctx, k_base, 0, 2, 1, 3);
            ggml_tensor * v = ggml_permute(runner_ctx.ggml_ctx, v_base, 0, 2, 1, 3);

            ggml_tensor * mask = nullptr;
            if (block_index != nullptr) {
                mask = make_input(index_tensor);
                ggml_set_name(mask, "block_sparse_index");
            } else {
                mask = make_input(zero_mask);
            }

            ggml_tensor * packed = ggml_flash_attn_ext_with_lse(runner_ctx.ggml_ctx,
                                                                q,
                                                                k,
                                                                v,
                                                                mask,
                                                                1.0f / std::sqrt(float(head_dim)),
                                                                0.0f,
                                                                0.0f);
            ggml_flash_attn_ext_set_prec(packed, GGML_PREC_F32);
            if (block_index != nullptr) {
                ggml_flash_attn_ext_set_block_sparse(
                    packed, block_q, block_k, GGML_FLASH_ATTN_EXT_MASK_BLOCK_INDEX);
                if (static_image_tokens > 0) {
                    if (static_image_block_rows > 0 && static_image_block_cols > 0) {
                        ggml_flash_attn_ext_set_static_halo_2d(packed,
                                                              static_image_tokens,
                                                              static_text_tokens,
                                                              static_image_block_rows,
                                                              static_image_block_cols);
                    } else {
                        ggml_flash_attn_ext_set_static_halo(packed, static_image_tokens, static_text_tokens);
                    }
                }
            }
            GGML_ASSERT(packed->ne[0] == head_dim + 1);

            ggml_cgraph * graph = ggml_new_graph(runner_ctx.ggml_ctx);
            ggml_build_forward_expand(graph, packed);
            return graph;
        };

        auto result = GGMLRunner::compute<float>(get_graph, n_threads, true);
        GGML_ASSERT(result.has_value());
        return result->values();
    }

    int count_exact_rho_anchor_flash_nodes_missing_block_params(int qo_len,
                                                                int kv_len,
                                                                int n_heads,
                                                                int n_kv_heads,
                                                                int head_dim,
                                                                const std::vector<float> & q_data,
                                                                const std::vector<ggml_fp16_t> & k_data,
                                                                const std::vector<ggml_fp16_t> & v_data) {
        set_flash_attention_enabled(true);
        set_htp_attention_enabled(true);
        set_block_sparse_attention_enabled(true);

        EnvGuard htp_ops("SD_HTP_OPS", "1");
        EnvGuard htp_compute("SD_HTP_COMPUTE", "1");
        EnvGuard htp_attention("SD_HTP_ATTENTION", "1");
        EnvGuard block_sparse_attn("SD_HTP_BLOCK_SPARSE_ATTN", "1");
        EnvGuard keep_ratio("SD_HTP_BLOCK_SPARSE_KEEP_RATIO", "0.5");

        sd::Tensor<float> q_tensor({n_heads * head_dim, qo_len, 1}, q_data);
        sd::Tensor<ggml_fp16_t> k_tensor({n_kv_heads * head_dim, kv_len, 1}, k_data);
        sd::Tensor<ggml_fp16_t> v_tensor({n_kv_heads * head_dim, kv_len, 1}, v_data);

        auto get_graph = [&]() -> ggml_cgraph * {
            GGMLRunnerContext runner_ctx = get_context();

            ggml_tensor * q = make_input(q_tensor);
            ggml_tensor * k = make_input(k_tensor);
            ggml_tensor * v = make_input(v_tensor);

            ggml_tensor * out = ggml_ext_attention_ext(runner_ctx.ggml_ctx,
                                                       runner_ctx.backend,
                                                       q,
                                                       k,
                                                       v,
                                                       n_heads,
                                                       nullptr,
                                                       false,
                                                       runner_ctx.flash_attn_enabled,
                                                       1.0f,
                                                       true,
                                                       0,
                                                       0,
                                                       SD_HTP_BLOCK_SPARSE_RESIDUAL_ANCHOR,
                                                       1.0f,
                                                       1.0f,
                                                       true,
                                                       "test_exact_rho_residual",
                                                       nullptr,
                                                       "test_exact_rho_rho",
                                                       nullptr,
                                                       &runner_ctx.cache_tensor);
            GGML_ASSERT(out != nullptr);

            ggml_cgraph * graph = ggml_new_graph(runner_ctx.ggml_ctx);
            ggml_build_forward_expand(graph, out);
            return graph;
        };

        ggml_cgraph * graph = nullptr;
        GGML_ASSERT(prepare_compute_graph(get_graph, &graph));
        GGML_ASSERT(graph != nullptr);

        int sparse_flash_nodes = 0;
        int missing_params = 0;
        const int n_nodes = ggml_graph_n_nodes(graph);
        for (int i = 0; i < n_nodes; ++i) {
            ggml_tensor * node = ggml_graph_node(graph, i);
            if (node == nullptr || node->op != GGML_OP_FLASH_ATTN_EXT || node->src[3] == nullptr ||
                node->src[3]->type != GGML_TYPE_I32) {
                continue;
            }
            ++sparse_flash_nodes;
            const int32_t block_q = node->op_params[4];
            const int32_t block_k = node->op_params[5];
            if (block_q <= 0 || block_k <= 0) {
                ++missing_params;
                std::fprintf(stderr,
                             "[FAIL] exact-rho anchor sparse flash node missing block params idx=%d "
                             "block_q=%d block_k=%d with_lse=%d shape=[%lld,%lld,%lld,%lld]\n",
                             i,
                             block_q,
                             block_k,
                             node->op_params[6],
                             (long long) node->ne[0],
                             (long long) node->ne[1],
                             (long long) node->ne[2],
                             (long long) node->ne[3]);
            }
        }
        std::printf("[CHECK] exact-rho anchor sparse flash nodes=%d missing_block_params=%d\n",
                    sparse_flash_nodes,
                    missing_params);
        GGML_ASSERT(sparse_flash_nodes > 0);
        cache_tensor_map.clear();
        cache_backed_tensor_name_map.clear();
        free_compute_ctx();
        return missing_params;
    }

    std::vector<float> run_cpu_lse(int n_threads,
                                   int qo_len,
                                   int kv_len,
                                   int n_heads,
                                   int n_kv_heads,
                                   int head_dim,
                                   const std::vector<float> & q_data,
                                   const std::vector<ggml_fp16_t> & k_data,
                                   const std::vector<float> * dense_mask_data) {
        set_flash_attention_enabled(false);
        set_htp_attention_enabled(false);
        set_block_sparse_attention_enabled(false);

        EnvGuard htp_ops("SD_HTP_OPS", "0");
        EnvGuard htp_compute("SD_HTP_COMPUTE", "0");
        EnvGuard htp_attention("SD_HTP_ATTENTION", "0");

        sd::Tensor<float> q_tensor({n_heads * head_dim, qo_len, 1}, q_data);
        sd::Tensor<ggml_fp16_t> k_tensor({n_kv_heads * head_dim, kv_len, 1}, k_data);
        std::vector<float> zero_mask_data(size_t(kv_len) * size_t(qo_len) * size_t(n_heads), 0.0f);
        sd::Tensor<float> mask_tensor({kv_len, qo_len, n_heads, 1},
                                      dense_mask_data != nullptr ? *dense_mask_data : zero_mask_data);

        auto get_graph = [&]() -> ggml_cgraph * {
            GGMLRunnerContext runner_ctx = get_context();

            ggml_tensor * q_orig = make_input(q_tensor);
            ggml_tensor * k_orig = make_input(k_tensor);
            ggml_tensor * mask = dense_mask_data != nullptr ? make_input(mask_tensor) : nullptr;

            ggml_tensor * q3 = ggml_reshape_3d(runner_ctx.ggml_ctx, q_orig, head_dim, n_heads, qo_len);
            ggml_tensor * q = ggml_permute(runner_ctx.ggml_ctx, q3, 0, 2, 1, 3);
            ggml_tensor * k_base = ggml_reshape_3d(runner_ctx.ggml_ctx, k_orig, head_dim, n_kv_heads, kv_len);
            ggml_tensor * k = ggml_permute(runner_ctx.ggml_ctx, k_base, 0, 2, 1, 3);

            ggml_tensor * lse = ggml_ext_attention_lse_node(runner_ctx.ggml_ctx,
                                                            q,
                                                            k,
                                                            mask,
                                                            n_heads,
                                                            1.0f / std::sqrt(float(head_dim)));
            GGML_ASSERT(lse != nullptr);

            ggml_cgraph * graph = ggml_new_graph(runner_ctx.ggml_ctx);
            ggml_build_forward_expand(graph, lse);
            return graph;
        };

        auto result = GGMLRunner::compute<float>(get_graph, n_threads, true);
        GGML_ASSERT(result.has_value());
        return result->values();
    }

    std::vector<float> run_cpu_lse_index(int n_threads,
                                         int qo_len,
                                         int kv_len,
                                         int n_heads,
                                         int n_kv_heads,
                                         int head_dim,
                                         const std::vector<float> & q_data,
                                         const std::vector<ggml_fp16_t> & k_data,
                                         const std::vector<int32_t> & block_index) {
        set_flash_attention_enabled(false);
        set_htp_attention_enabled(false);
        set_block_sparse_attention_enabled(false);

        EnvGuard htp_ops("SD_HTP_OPS", "0");
        EnvGuard htp_compute("SD_HTP_COMPUTE", "0");
        EnvGuard htp_attention("SD_HTP_ATTENTION", "0");

        const int block_q = SD_HTP_BLOCK_SPARSE_MASK_BLOCK;
        const int n_q_blocks = n_heads * ((qo_len + block_q - 1) / block_q);
        const int max_kept = int(block_index.size()) / n_q_blocks;

        sd::Tensor<float> q_tensor({n_heads * head_dim, qo_len, 1}, q_data);
        sd::Tensor<ggml_fp16_t> k_tensor({n_kv_heads * head_dim, kv_len, 1}, k_data);
        sd::Tensor<int32_t> index_tensor({max_kept, n_q_blocks, 1, 1}, block_index);

        auto get_graph = [&]() -> ggml_cgraph * {
            GGMLRunnerContext runner_ctx = get_context();

            ggml_tensor * q_orig = make_input(q_tensor);
            ggml_tensor * k_orig = make_input(k_tensor);
            ggml_tensor * index = make_input(index_tensor);
            ggml_set_name(index, "block_sparse_index");

            ggml_tensor * q3 = ggml_reshape_3d(runner_ctx.ggml_ctx, q_orig, head_dim, n_heads, qo_len);
            ggml_tensor * q = ggml_permute(runner_ctx.ggml_ctx, q3, 0, 2, 1, 3);
            ggml_tensor * k_base = ggml_reshape_3d(runner_ctx.ggml_ctx, k_orig, head_dim, n_kv_heads, kv_len);
            ggml_tensor * k = ggml_permute(runner_ctx.ggml_ctx, k_base, 0, 2, 1, 3);

            ggml_tensor * lse = ggml_ext_attention_lse_node(runner_ctx.ggml_ctx,
                                                            q,
                                                            k,
                                                            index,
                                                            n_heads,
                                                            1.0f / std::sqrt(float(head_dim)),
                                                            GGML_FLASH_ATTN_EXT_MASK_BLOCK_INDEX);
            GGML_ASSERT(lse != nullptr);

            ggml_cgraph * graph = ggml_new_graph(runner_ctx.ggml_ctx);
            ggml_build_forward_expand(graph, lse);
            return graph;
        };

        auto result = GGMLRunner::compute<float>(get_graph, n_threads, true);
        GGML_ASSERT(result.has_value());
        return result->values();
    }

    std::vector<float> run_exact_rho_recovered_graph(int n_threads,
                                                     int qo_len,
                                                     int kv_len,
                                                     int n_heads,
                                                     int n_kv_heads,
                                                     int head_dim,
                                                     const std::vector<float> & q_data,
                                                     const std::vector<ggml_fp16_t> & k_data,
                                                     const std::vector<ggml_fp16_t> & v_data,
                                                     const std::vector<int32_t> & block_index) {
        set_flash_attention_enabled(true);
        set_htp_attention_enabled(true);
        set_block_sparse_attention_enabled(true);

        EnvGuard htp_ops("SD_HTP_OPS", "1");
        EnvGuard htp_compute("SD_HTP_COMPUTE", "1");
        EnvGuard htp_attention("SD_HTP_ATTENTION", "1");

        const int block_q = SD_HTP_BLOCK_SPARSE_MASK_BLOCK;
        const int block_k = SD_HTP_BLOCK_SPARSE_MASK_BLOCK;
        const int n_q_blocks = n_heads * ((qo_len + block_q - 1) / block_q);
        const int max_kept = int(block_index.size()) / n_q_blocks;

        sd::Tensor<float> q_tensor({n_heads * head_dim, qo_len, 1}, q_data);
        sd::Tensor<ggml_fp16_t> k_tensor({n_kv_heads * head_dim, kv_len, 1}, k_data);
        sd::Tensor<ggml_fp16_t> v_tensor({n_kv_heads * head_dim, kv_len, 1}, v_data);
        sd::Tensor<int32_t> index_tensor({max_kept, n_q_blocks, 1, 1}, block_index);

        auto get_graph = [&]() -> ggml_cgraph * {
            GGMLRunnerContext runner_ctx = get_context();

            ggml_tensor * q_orig = make_input(q_tensor);
            ggml_tensor * k_orig = make_input(k_tensor);
            ggml_tensor * v_orig = make_input(v_tensor);
            ggml_tensor * index = make_input(index_tensor);
            ggml_set_name(index, "block_sparse_index");

            ggml_tensor * q3 = ggml_reshape_3d(runner_ctx.ggml_ctx, q_orig, head_dim, n_heads, qo_len);
            ggml_tensor * q = ggml_permute(runner_ctx.ggml_ctx, q3, 0, 2, 1, 3);
            ggml_tensor * k_base = ggml_reshape_3d(runner_ctx.ggml_ctx, k_orig, head_dim, n_kv_heads, kv_len);
            ggml_tensor * v_base = ggml_reshape_3d(runner_ctx.ggml_ctx, v_orig, head_dim, n_kv_heads, kv_len);
            ggml_tensor * k = ggml_permute(runner_ctx.ggml_ctx, k_base, 0, 2, 1, 3);
            ggml_tensor * v = ggml_permute(runner_ctx.ggml_ctx, v_base, 0, 2, 1, 3);

            ggml_tensor * dense_out = ggml_flash_attn_ext(runner_ctx.ggml_ctx,
                                                          q,
                                                          k,
                                                          v,
                                                          nullptr,
                                                          1.0f / std::sqrt(float(head_dim)),
                                                          0.0f,
                                                          0.0f);
            ggml_flash_attn_ext_set_prec(dense_out, GGML_PREC_F32);
            dense_out = ggml_view_3d(runner_ctx.ggml_ctx,
                                     dense_out,
                                     head_dim,
                                     n_heads,
                                     qo_len,
                                     dense_out->nb[1],
                                     dense_out->nb[2],
                                     0);

            ggml_tensor * sparse_out = ggml_flash_attn_ext(runner_ctx.ggml_ctx,
                                                           q,
                                                           k,
                                                           v,
                                                           index,
                                                           1.0f / std::sqrt(float(head_dim)),
                                                           0.0f,
                                                           0.0f);
            ggml_flash_attn_ext_set_prec(sparse_out, GGML_PREC_F32);
            ggml_flash_attn_ext_set_block_sparse(
                sparse_out, block_q, block_k, GGML_FLASH_ATTN_EXT_MASK_BLOCK_INDEX);
            sparse_out = ggml_view_3d(runner_ctx.ggml_ctx,
                                      sparse_out,
                                      head_dim,
                                      n_heads,
                                      qo_len,
                                      sparse_out->nb[1],
                                      sparse_out->nb[2],
                                      0);

            ggml_tensor * dense_lse = ggml_ext_attention_lse_node(runner_ctx.ggml_ctx,
                                                                  q,
                                                                  k,
                                                                  nullptr,
                                                                  n_heads,
                                                                  1.0f / std::sqrt(float(head_dim)));
            ggml_tensor * sparse_lse = ggml_ext_attention_lse_node(runner_ctx.ggml_ctx,
                                                                   q,
                                                                   k,
                                                                   index,
                                                                   n_heads,
                                                                   1.0f / std::sqrt(float(head_dim)),
                                                                   GGML_FLASH_ATTN_EXT_MASK_BLOCK_INDEX);
            ggml_tensor * rho = ggml_exp(runner_ctx.ggml_ctx, ggml_sub(runner_ctx.ggml_ctx, sparse_lse, dense_lse));
            ggml_tensor * scaled_sparse = ggml_mul(runner_ctx.ggml_ctx, sparse_out, rho);
            ggml_tensor * residual = ggml_sub(runner_ctx.ggml_ctx, dense_out, scaled_sparse);
            ggml_tensor * recovered = ggml_add(runner_ctx.ggml_ctx, ggml_mul(runner_ctx.ggml_ctx, sparse_out, rho), residual);
            recovered = ggml_ext_cont(runner_ctx.ggml_ctx, recovered);
            recovered = ggml_reshape_3d(runner_ctx.ggml_ctx, recovered, head_dim * n_heads, qo_len, 1);

            ggml_cgraph * graph = ggml_new_graph(runner_ctx.ggml_ctx);
            ggml_build_forward_expand(graph, recovered);
            return graph;
        };

        auto result = GGMLRunner::compute<float>(get_graph, n_threads, true);
        GGML_ASSERT(result.has_value());
        return result->values();
    }
};

static std::vector<float> dense_mask_from_block_index(const std::vector<int32_t> & block_index,
                                                      int qo_len,
                                                      int kv_len,
                                                      int n_heads,
                                                      int n_kv_heads) {
    const int block_q = SD_HTP_BLOCK_SPARSE_MASK_BLOCK;
    const int block_k = SD_HTP_BLOCK_SPARSE_MASK_BLOCK;
    const int q_blocks_per_head = (qo_len + block_q - 1) / block_q;
    const int k_blocks_per_head = (kv_len + block_k - 1) / block_k;
    const int n_q_blocks = n_heads * q_blocks_per_head;
    const int n_k_blocks = n_kv_heads * k_blocks_per_head;
    const int max_kept = int(block_index.size()) / n_q_blocks;
    const int gqa = std::max(1, n_heads / n_kv_heads);
    std::vector<uint8_t> keep(size_t(n_q_blocks) * size_t(n_k_blocks), 0);
    for (int qb = 0; qb < n_q_blocks; ++qb) {
        for (int i = 0; i < max_kept; ++i) {
            const int kb = block_index[size_t(qb) * size_t(max_kept) + size_t(i)];
            if (kb >= 0 && kb < n_k_blocks) {
                keep[size_t(qb) * size_t(n_k_blocks) + size_t(kb)] = 1;
            }
        }
    }

    std::vector<float> dense_mask(size_t(kv_len) * size_t(qo_len) * size_t(n_heads));
    const float zero = 0.0f;
    const float neg_inf = -INFINITY;
    for (int h = 0; h < n_heads; ++h) {
        const int kv_head = std::min(n_kv_heads - 1, h / gqa);
        for (int qt = 0; qt < qo_len; ++qt) {
            const int qb = h * q_blocks_per_head + qt / block_q;
            for (int kt = 0; kt < kv_len; ++kt) {
                const int kb = kv_head * k_blocks_per_head + kt / block_k;
                const bool is_kept = keep[size_t(qb) * size_t(n_k_blocks) + size_t(kb)] != 0;
                dense_mask[(size_t(h) * size_t(qo_len) + size_t(qt)) * size_t(kv_len) + size_t(kt)] =
                    is_kept ? zero : neg_inf;
            }
        }
    }
    return dense_mask;
}

static std::vector<int32_t> full_keep_block_index(int qo_len, int kv_len, int n_heads, int n_kv_heads) {
    const int block_q = SD_HTP_BLOCK_SPARSE_MASK_BLOCK;
    const int block_k = SD_HTP_BLOCK_SPARSE_MASK_BLOCK;
    const int n_q_blocks = n_heads * ((qo_len + block_q - 1) / block_q);
    const int n_k_blocks = n_kv_heads * ((kv_len + block_k - 1) / block_k);
    std::vector<int32_t> index(size_t(n_q_blocks) * size_t(n_k_blocks));
    for (int qb = 0; qb < n_q_blocks; ++qb) {
        for (int kb = 0; kb < n_k_blocks; ++kb) {
            index[size_t(qb) * size_t(n_k_blocks) + size_t(kb)] = kb;
        }
    }
    return index;
}

static void test_static_halo_characterization(AttentionRunner & runner,
                                              const char * load_mode,
                                              int n_heads = 1,
                                              int head_dim = 64,
                                              int token_h = 16,
                                              int token_w = 32,
                                              bool pass_static_grid = true,
                                              bool malformed_head1 = false) {
    GGML_ASSERT(token_h > 0 && token_w > 0 && token_h % 4 == 0 && token_w % 8 == 0);
    const int image_tokens = token_h * token_w;
    constexpr int text_tokens = 154;
    const int seq_len = image_tokens + text_tokens;
    const int n_kv_heads = n_heads;
    constexpr int block_size = SD_HTP_BLOCK_SPARSE_MASK_BLOCK;
    const int blocks_per_head = (seq_len + block_size - 1) / block_size;

    std::vector<int32_t> block_index(size_t(n_heads) * size_t(blocks_per_head) * size_t(blocks_per_head), -1);
    GGML_ASSERT(ggml_ext_block_sparse_fill_direct_halo_index_32(block_index.data(),
                                                                blocks_per_head,
                                                                seq_len,
                                                                seq_len,
                                                                image_tokens,
                                                                text_tokens,
                                                                token_h,
                                                                token_w,
                                                                n_heads,
                                                                n_kv_heads,
                                                                false));

    if (malformed_head1) {
        GGML_ASSERT(n_heads == 2 && token_h == 16 && token_w == 32);
        int32_t * row = block_index.data() + size_t(blocks_per_head) * size_t(blocks_per_head);
        const int head_base = blocks_per_head;
        GGML_ASSERT(row[0] == head_base && row[1] == head_base + 1 &&
                    row[2] == head_base + 4 && row[3] == head_base + 5);
        row[0] = head_base + 1;
        row[1] = head_base + 2;
    }

    const std::vector<float> q = gen_q(seq_len, n_heads, head_dim, 301);
    const std::vector<ggml_fp16_t> k = gen_kv_f16(seq_len, n_kv_heads, head_dim, 302);
    const std::vector<ggml_fp16_t> v = gen_kv_f16(seq_len, n_kv_heads, head_dim, 303);
    const std::vector<float> dense_mask =
        dense_mask_from_block_index(block_index, seq_len, seq_len, n_heads, n_kv_heads);
    const RefAttentionWithLse ref =
        reference_attention_with_lse(seq_len, seq_len, n_heads, n_kv_heads, head_dim, q, k, v, &dense_mask);

    EnvGuard halo_load_mode("SD_HTP_HALO_KV_LOAD", load_mode);
    EnvGuard sparse_opencl("SD_HTP_SPARSE_OPENCL", "0");
    const bool require_userdma_trace =
        std::strcmp(load_mode, "userdma") == 0 && pass_static_grid && !malformed_head1;
    const bool collect_halo_trace = require_userdma_trace || malformed_head1;
    EnvGuard halo_trace("SD_HTP_HALO_KV_TRACE", collect_halo_trace ? "1" : nullptr);
    EnvGuard require_halo_trace("SD_HTP_HALO_REQUIRE_USERDMA_TRACE", require_userdma_trace ? "1" : nullptr);
    const std::vector<float> sparse = runner.run_htp_sparse_index(4,
                                                                  seq_len,
                                                                  seq_len,
                                                                  n_heads,
                                                                  n_kv_heads,
                                                                  head_dim,
                                                                  q,
                                                                  k,
                                                                  v,
                                                                  block_index,
                                                                  image_tokens,
                                                                  text_tokens,
                                                                  pass_static_grid ? token_h / 4 : 0,
                                                                  pass_static_grid ? token_w / 8 : 0);
    const DiffStats out_stats = diff_stats(ref.out, sparse);
    const std::vector<float> packed = runner.run_htp_flash_with_lse_packed(4,
                                                                           seq_len,
                                                                           seq_len,
                                                                           n_heads,
                                                                           n_kv_heads,
                                                                           head_dim,
                                                                           q,
                                                                           k,
                                                                           v,
                                                                           &block_index,
                                                                           image_tokens,
                                                                           text_tokens,
                                                                           pass_static_grid ? token_h / 4 : 0,
                                                                           pass_static_grid ? token_w / 8 : 0);
    const PackedWithLseDiffStats packed_stats =
        packed_with_lse_diff_stats(ref.out, ref.lse, packed, seq_len, n_heads, head_dim);
    std::printf("[STATIC_HALO_%s_H%d_D%d] out_max_abs=%.9g packed_out_max_abs=%.9g lse_max_abs=%.9g "
                "out_worst=%zu out_worst_q=%zu lse_worst=%zu lse_worst_q=%zu\n",
                load_mode,
                n_heads,
                head_dim,
                out_stats.max_abs,
                packed_stats.out_max_abs,
                packed_stats.lse_max_abs,
                out_stats.worst,
                out_stats.worst / (size_t(n_heads) * size_t(head_dim)),
                packed_stats.lse_worst,
                packed_stats.lse_worst / size_t(n_heads));
    const bool baseline = std::strcmp(load_mode, "baseline") == 0;
    const double out_tolerance = baseline ? 3e-3 : (head_dim == 128 ? 4.1e-3 : 4e-3);
    GGML_ASSERT(out_stats.max_abs <= out_tolerance);
    GGML_ASSERT(packed_stats.out_max_abs <= out_tolerance);
    GGML_ASSERT(packed_stats.lse_max_abs <= 5e-3);
}

static void test_static_halo_stable_normal_high_logit(AttentionRunner & runner) {
    constexpr int token_h = 16;
    constexpr int token_w = 32;
    constexpr int image_tokens = token_h * token_w;
    constexpr int text_tokens = 154;
    constexpr int seq_len = image_tokens + text_tokens;
    constexpr int n_heads = 1;
    constexpr int n_kv_heads = 1;
    constexpr int head_dim = 64;
    constexpr int block_size = SD_HTP_BLOCK_SPARSE_MASK_BLOCK;
    constexpr int blocks_per_head = (seq_len + block_size - 1) / block_size;

    std::vector<int32_t> block_index(size_t(blocks_per_head) * size_t(blocks_per_head), -1);
    GGML_ASSERT(ggml_ext_block_sparse_fill_direct_halo_index_32(block_index.data(),
                                                                blocks_per_head,
                                                                seq_len,
                                                                seq_len,
                                                                image_tokens,
                                                                text_tokens,
                                                                token_h,
                                                                token_w,
                                                                n_heads,
                                                                n_kv_heads,
                                                                false));

    std::vector<float> q = gen_q(seq_len, n_heads, head_dim, 912);
    for (int token = 0; token < 32; ++token) {
        for (int d = 0; d < head_dim; ++d) {
            q[size_t(token) * head_dim + size_t(d)] = 4.0f;
        }
    }
    std::vector<ggml_fp16_t> k(size_t(seq_len) * head_dim);
    for (int token = 0; token < seq_len; ++token) {
        const float key_value = token < 32 ? 4.0f : (token < 64 ? 3.5f : 0.0f);
        for (int d = 0; d < head_dim; ++d) {
            k[size_t(token) * head_dim + size_t(d)] = ggml_fp32_to_fp16(key_value);
        }
    }
    const std::vector<ggml_fp16_t> v = gen_kv_f16(seq_len, n_kv_heads, head_dim, 911);
    const std::vector<float> dense_mask =
        dense_mask_from_block_index(block_index, seq_len, seq_len, n_heads, n_kv_heads);

    float max_unshifted_logit = -std::numeric_limits<float>::infinity();
    const float scale = 1.0f / std::sqrt(float(head_dim));
    for (int token = 0; token < seq_len; ++token) {
        float score = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            score += q[size_t(d)] * ggml_fp16_to_fp32(k[size_t(token) * head_dim + size_t(d)]);
        }
        max_unshifted_logit = std::max(max_unshifted_logit, score * scale);
    }
    const float naive_exp = std::exp(max_unshifted_logit);
    GGML_ASSERT(!std::isfinite(naive_exp));

    const RefAttentionWithLse ref =
        reference_attention_with_lse(seq_len, seq_len, n_heads, n_kv_heads, head_dim, q, k, v, &dense_mask);
    GGML_ASSERT(std::all_of(ref.out.begin(), ref.out.end(), [](float x) { return std::isfinite(x); }));
    GGML_ASSERT(std::all_of(ref.lse.begin(), ref.lse.end(), [](float x) { return std::isfinite(x); }));

    for (const char * load_mode : {"l2fetch", "userdma"}) {
        EnvGuard halo_load_mode("SD_HTP_HALO_KV_LOAD", load_mode);
        EnvGuard sparse_opencl("SD_HTP_SPARSE_OPENCL", "0");
        EnvGuard halo_trace("SD_HTP_HALO_KV_TRACE", "1");
        EnvGuard require_userdma_trace("SD_HTP_HALO_REQUIRE_USERDMA_TRACE",
                                       std::strcmp(load_mode, "userdma") == 0 ? "1" : nullptr);
        const std::vector<float> packed = runner.run_htp_flash_with_lse_packed(4,
                                                                               seq_len,
                                                                               seq_len,
                                                                               n_heads,
                                                                               n_kv_heads,
                                                                               head_dim,
                                                                               q,
                                                                               k,
                                                                               v,
                                                                               &block_index,
                                                                               image_tokens,
                                                                               text_tokens,
                                                                               token_h / 4,
                                                                               token_w / 8);
        GGML_ASSERT(std::all_of(packed.begin(), packed.end(), [](float x) { return std::isfinite(x); }));
        const PackedWithLseDiffStats stats =
            packed_with_lse_diff_stats(ref.out, ref.lse, packed, seq_len, n_heads, head_dim);
        const size_t worst_lse_base = stats.lse_worst * size_t(head_dim + 1) + size_t(head_dim);
        std::printf("[STATIC_HALO_STABLE_NORMAL_%s] max_logit=%.9g out_max_abs=%.9g lse_max_abs=%.9g "
                    "lse_worst=%zu ref_lse=%.9g got_lse=%.9g\n",
                    load_mode,
                    max_unshifted_logit,
                    stats.out_max_abs,
                    stats.lse_max_abs,
                    stats.lse_worst,
                    ref.lse[stats.lse_worst],
                    packed[worst_lse_base]);
        GGML_ASSERT(stats.out_max_abs <= 4e-3);
        GGML_ASSERT(stats.lse_max_abs <= 5e-3);
    }
}

static std::vector<int32_t> fragmented_per_head_block_index(int qo_len,
                                                            int kv_len,
                                                            int n_heads,
                                                            int n_kv_heads) {
    GGML_ASSERT(n_heads == n_kv_heads);
    const int block_q = SD_HTP_BLOCK_SPARSE_MASK_BLOCK;
    const int block_k = SD_HTP_BLOCK_SPARSE_MASK_BLOCK;
    const int q_blocks_per_head = (qo_len + block_q - 1) / block_q;
    const int k_blocks_per_head = (kv_len + block_k - 1) / block_k;
    constexpr int max_kept = 5;
    GGML_ASSERT(q_blocks_per_head > 0 && k_blocks_per_head > 96);

    std::vector<int32_t> index(size_t(n_heads) * size_t(q_blocks_per_head) * max_kept, -1);
    for (int h = 0; h < n_heads; ++h) {
        const int head_base = h * k_blocks_per_head;
        for (int qb = 0; qb < q_blocks_per_head; ++qb) {
            const int run0 = 1 + (h * 7 + qb * 11) % 24;
            const int run1 = 64 + (h * 13 + qb * 5) % 24;
            const int blocks[max_kept] = {
                head_base + run0,
                head_base + run0 + 1,
                head_base + run1,
                head_base + run1 + 1,
                head_base + k_blocks_per_head - 1,
            };
            const size_t row = size_t(h * q_blocks_per_head + qb) * max_kept;
            for (int i = 0; i < max_kept; ++i) {
                index[row + size_t(i)] = blocks[i];
            }
        }
    }
    return index;
}

static void test_cache_full_k_4250_d128(AttentionRunner & runner) {
    constexpr int qo_len = 96;
    constexpr int kv_len = 4250;
    constexpr int n_heads = 2;
    constexpr int n_kv_heads = 2;
    constexpr int head_dim = 128;

    const std::vector<float> q = gen_q(qo_len, n_heads, head_dim, 201);
    const std::vector<ggml_fp16_t> k = gen_kv_f16(kv_len, n_kv_heads, head_dim, 202);
    std::vector<ggml_fp16_t> v = gen_kv_f16(kv_len, n_kv_heads, head_dim, 203);
    for (ggml_fp16_t & value : v) {
        value = ggml_fp32_to_fp16(0.5f * ggml_fp16_to_fp32(value));
    }
    const std::vector<int32_t> block_index =
        fragmented_per_head_block_index(qo_len, kv_len, n_heads, n_kv_heads);
    const std::vector<float> dense_mask =
        dense_mask_from_block_index(block_index, qo_len, kv_len, n_heads, n_kv_heads);

    const std::vector<float> cpu =
        runner.run_cpu_dense_masked(4, qo_len, kv_len, n_heads, n_kv_heads, head_dim, q, k, v, dense_mask);
    std::fprintf(stderr,
                 "[EXPECT] cache-mode expected=FULL_K qo=%d kv=%d heads=%d kv_heads=%d d=%d\n",
                 qo_len,
                 kv_len,
                 n_heads,
                 n_kv_heads,
                 head_dim);
    const std::vector<float> sparse =
        runner.run_htp_sparse_index(4, qo_len, kv_len, n_heads, n_kv_heads, head_dim, q, k, v, block_index);
    const DiffStats out_stats = diff_stats(cpu, sparse);
    std::printf("[CACHE_MODE_DIAG] expected=FULL_K out_max_abs=%.9g mean_abs=%.9g rms_abs=%.9g "
                "worst=%zu cpu=%.9g sparse=%.9g\n",
                out_stats.max_abs,
                out_stats.mean_abs,
                out_stats.rms_abs,
                out_stats.worst,
                double(cpu[out_stats.worst]),
                double(sparse[out_stats.worst]));
    std::fflush(stdout);
    GGML_ASSERT(out_stats.max_abs <= 3e-3);

    const RefAttentionWithLse ref =
        reference_attention_with_lse(qo_len, kv_len, n_heads, n_kv_heads, head_dim, q, k, v, &dense_mask);
    const std::vector<float> packed =
        runner.run_htp_flash_with_lse_packed(4, qo_len, kv_len, n_heads, n_kv_heads, head_dim, q, k, v, &block_index);
    const PackedWithLseDiffStats packed_stats =
        packed_with_lse_diff_stats(ref.out, ref.lse, packed, qo_len, n_heads, head_dim);
    std::printf("[CACHE_MODE] expected=FULL_K out_max_abs=%.9g packed_out_max_abs=%.9g "
                "lse_max_abs=%.9g out_worst=%zu packed_out_worst=%zu lse_worst=%zu\n",
                out_stats.max_abs,
                packed_stats.out_max_abs,
                packed_stats.lse_max_abs,
                out_stats.worst,
                packed_stats.out_worst,
                packed_stats.lse_worst);
    GGML_ASSERT(packed_stats.out_max_abs <= 3e-3);
    GGML_ASSERT(packed_stats.lse_max_abs <= 5e-3);
}

static void test_cache_mode_output(AttentionRunner & runner,
                                   const char * expected_mode,
                                   int kv_len,
                                   int head_dim,
                                   int seed) {
    constexpr int qo_len = 96;
    constexpr int n_heads = 2;
    constexpr int n_kv_heads = 2;

    const std::vector<float> q = gen_q(qo_len, n_heads, head_dim, seed);
    const std::vector<ggml_fp16_t> k = gen_kv_f16(kv_len, n_kv_heads, head_dim, seed + 1);
    std::vector<ggml_fp16_t> v = gen_kv_f16(kv_len, n_kv_heads, head_dim, seed + 2);
    for (ggml_fp16_t & value : v) {
        value = ggml_fp32_to_fp16(0.5f * ggml_fp16_to_fp32(value));
    }
    const std::vector<int32_t> block_index =
        fragmented_per_head_block_index(qo_len, kv_len, n_heads, n_kv_heads);
    const std::vector<float> dense_mask =
        dense_mask_from_block_index(block_index, qo_len, kv_len, n_heads, n_kv_heads);
    const std::vector<float> cpu =
        runner.run_cpu_dense_masked(4, qo_len, kv_len, n_heads, n_kv_heads, head_dim, q, k, v, dense_mask);

    std::fprintf(stderr,
                 "[EXPECT] cache-mode expected=%s qo=%d kv=%d heads=%d kv_heads=%d d=%d\n",
                 expected_mode,
                 qo_len,
                 kv_len,
                 n_heads,
                 n_kv_heads,
                 head_dim);
    const std::vector<float> sparse =
        runner.run_htp_sparse_index(4, qo_len, kv_len, n_heads, n_kv_heads, head_dim, q, k, v, block_index);
    const DiffStats stats = diff_stats(cpu, sparse);
    std::printf("[CACHE_MODE] expected=%s out_max_abs=%.9g mean_abs=%.9g rms_abs=%.9g "
                "worst=%zu cpu=%.9g sparse=%.9g\n",
                expected_mode,
                stats.max_abs,
                stats.mean_abs,
                stats.rms_abs,
                stats.worst,
                double(cpu[stats.worst]),
                double(sparse[stats.worst]));
    std::fflush(stdout);
    GGML_ASSERT(stats.max_abs <= 3e-3);
}

static void test_cache_full_kv_4250(AttentionRunner & runner) {
    test_cache_mode_output(runner, "FULL_KV", 4250, 64, 211);
}

static void test_cache_chunked_5281_d128(AttentionRunner & runner) {
    test_cache_mode_output(runner, "CHUNKED", 5281, 128, 221);
}

static void test_cache_d128_multi_kv(AttentionRunner & runner) {
    constexpr int qo_len = 96;
    constexpr int kv_len = 4250;
    constexpr int n_heads = 2;
    constexpr int n_kv_heads = 2;
    constexpr int head_dim = 128;
    constexpr int max_kept = 5;
    constexpr int q_blocks_per_head = 3;
    const int k_blocks_per_head = (kv_len + SD_HTP_BLOCK_SPARSE_MASK_BLOCK - 1) /
                                  SD_HTP_BLOCK_SPARSE_MASK_BLOCK;

    const std::vector<float> q = gen_q(qo_len, n_heads, head_dim, 231);
    const std::vector<ggml_fp16_t> k = gen_kv_f16(kv_len, n_kv_heads, head_dim, 232);
    std::vector<ggml_fp16_t> v = gen_kv_f16(kv_len, n_kv_heads, head_dim, 233);
    for (ggml_fp16_t & value : v) {
        value = ggml_fp32_to_fp16(0.5f * ggml_fp16_to_fp32(value));
    }

    std::vector<int32_t> block_index =
        fragmented_per_head_block_index(qo_len, kv_len, n_heads, n_kv_heads);
    const size_t tail_only_row = size_t(1 * q_blocks_per_head + 2) * max_kept;
    block_index[tail_only_row] = 1 * k_blocks_per_head + k_blocks_per_head - 1;
    for (int i = 1; i < max_kept; ++i) {
        block_index[tail_only_row + size_t(i)] = -1;
    }

    const std::vector<float> dense_mask =
        dense_mask_from_block_index(block_index, qo_len, kv_len, n_heads, n_kv_heads);
    const RefAttentionWithLse ref =
        reference_attention_with_lse(qo_len, kv_len, n_heads, n_kv_heads, head_dim, q, k, v, &dense_mask);
    const std::vector<float> sparse =
        runner.run_htp_sparse_index(4, qo_len, kv_len, n_heads, n_kv_heads, head_dim, q, k, v, block_index);
    const DiffStats out_stats = diff_stats(ref.out, sparse);
    const std::vector<float> packed =
        runner.run_htp_flash_with_lse_packed(4, qo_len, kv_len, n_heads, n_kv_heads, head_dim, q, k, v, &block_index);
    const PackedWithLseDiffStats packed_stats =
        packed_with_lse_diff_stats(ref.out, ref.lse, packed, qo_len, n_heads, head_dim);
    std::printf("[CACHE_MULTI_KV] expected=FULL_K out_max_abs=%.9g packed_out_max_abs=%.9g "
                "lse_max_abs=%.9g out_worst=%zu packed_out_worst=%zu lse_worst=%zu\n",
                out_stats.max_abs,
                packed_stats.out_max_abs,
                packed_stats.lse_max_abs,
                out_stats.worst,
                packed_stats.out_worst,
                packed_stats.lse_worst);
    GGML_ASSERT(out_stats.max_abs <= 3e-3);
    GGML_ASSERT(packed_stats.out_max_abs <= 3e-3);
    GGML_ASSERT(packed_stats.lse_max_abs <= 5e-3);
}

struct EquivalentBenchmarkMasks {
    std::vector<int32_t> sparse_index;
    std::vector<ggml_fp16_t> dense_mask;
    int max_kept = 0;
    int min_kept = 0;
    double mean_kept = 0.0;
};

static EquivalentBenchmarkMasks make_equivalent_benchmark_masks(int qo_len,
                                                                 int kv_len,
                                                                 int n_heads,
                                                                 int n_kv_heads) {
    GGML_ASSERT(n_heads == n_kv_heads);
    const int block_q = SD_HTP_BLOCK_SPARSE_MASK_BLOCK;
    const int block_k = SD_HTP_BLOCK_SPARSE_MASK_BLOCK;
    const int q_blocks_per_head = (qo_len + block_q - 1) / block_q;
    const int k_blocks_per_head = (kv_len + block_k - 1) / block_k;
    const int n_q_blocks = n_heads * q_blocks_per_head;
    GGML_ASSERT(q_blocks_per_head == 13 && k_blocks_per_head == 13);
    const int max_kept = k_blocks_per_head;

    // Captured from the first 410-token joint-attention r99 mask in the production trace.
    const std::vector<std::vector<int32_t>> local_rows = {
        {0, 1, 2, 8, 9, 10, 11, 12},
        {1, 8, 9, 10, 11, 12},
        {0, 1, 2, 8, 9, 10, 11, 12},
        {1, 8, 9, 10, 11, 12},
        {0, 1, 2, 8, 9, 10, 11, 12},
        {1, 8, 9, 10, 11, 12},
        {0, 1, 2, 8, 9, 10, 11, 12},
        {1, 8, 9, 10, 11, 12},
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
    };
    int min_kept = max_kept;
    int total_kept = 0;
    for (int qb = 0; qb < q_blocks_per_head; ++qb) {
        const auto & row = local_rows[size_t(qb)];
        min_kept = std::min(min_kept, int(row.size()));
        total_kept += int(row.size());
    }

    EquivalentBenchmarkMasks masks;
    masks.max_kept = max_kept;
    masks.min_kept = min_kept;
    masks.mean_kept = double(total_kept) / double(q_blocks_per_head);
    masks.sparse_index.assign(size_t(n_q_blocks) * size_t(max_kept), -1);
    for (int h = 0; h < n_heads; ++h) {
        const int k_block_base = h * k_blocks_per_head;
        for (int qb = 0; qb < q_blocks_per_head; ++qb) {
            const int dst_qb = h * q_blocks_per_head + qb;
            const auto & row = local_rows[size_t(qb)];
            for (size_t slot = 0; slot < row.size(); ++slot) {
                masks.sparse_index[size_t(dst_qb) * size_t(max_kept) + slot] = k_block_base + row[slot];
            }
        }
    }

    const int kv_pad_len = ((kv_len + 63) / 64) * 64;
    const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f);
    const ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity());
    masks.dense_mask.assign(size_t(kv_pad_len) * size_t(qo_len), neg_inf);
    for (int qt = 0; qt < qo_len; ++qt) {
        const auto & row = local_rows[size_t(qt / block_q)];
        for (int kt = 0; kt < kv_len; ++kt) {
            const int kb = kt / block_k;
            if (std::find(row.begin(), row.end(), kb) != row.end()) {
                masks.dense_mask[size_t(qt) * size_t(kv_pad_len) + size_t(kt)] = zero;
            }
        }
    }
    return masks;
}

static void test_exact_residual_rho_reference_formula(AttentionRunner & runner) {
    constexpr int qo_len = 96;
    constexpr int kv_len = 128;
    constexpr int n_heads = 4;
    constexpr int n_kv_heads = 2;
    constexpr int head_dim = 64;
    constexpr float keep_ratio = 0.50f;

    std::vector<float> q = gen_q(qo_len, n_heads, head_dim, 11);
    std::vector<ggml_fp16_t> k = gen_kv_f16(kv_len, n_kv_heads, head_dim, 12);
    std::vector<ggml_fp16_t> v = gen_kv_f16(kv_len, n_kv_heads, head_dim, 13);

    std::vector<int32_t> block_index =
        runner.make_block_index(4, qo_len, kv_len, n_heads, n_kv_heads, head_dim, q, k, keep_ratio);
    std::vector<float> sparse_dense_mask =
        dense_mask_from_block_index(block_index, qo_len, kv_len, n_heads, n_kv_heads);

    const RefAttentionWithLse dense_ref =
        reference_attention_with_lse(qo_len, kv_len, n_heads, n_kv_heads, head_dim, q, k, v, nullptr);
    const RefAttentionWithLse sparse_ref =
        reference_attention_with_lse(qo_len, kv_len, n_heads, n_kv_heads, head_dim, q, k, v, &sparse_dense_mask);

    std::vector<float> recovered(dense_ref.out.size());
    double max_diff = 0.0;
    size_t worst = 0;
    for (int qt = 0; qt < qo_len; ++qt) {
        for (int h = 0; h < n_heads; ++h) {
            const float rho = std::exp(sparse_ref.lse[size_t(qt) * size_t(n_heads) + size_t(h)] -
                                       dense_ref.lse[size_t(qt) * size_t(n_heads) + size_t(h)]);
            GGML_ASSERT(rho >= 0.0f && rho <= 1.0001f);
            for (int d = 0; d < head_dim; ++d) {
                const size_t out_i = (size_t(qt) * size_t(n_heads) + size_t(h)) * size_t(head_dim) + size_t(d);
                const float residual = dense_ref.out[out_i] - rho * sparse_ref.out[out_i];
                recovered[out_i] = rho * sparse_ref.out[out_i] + residual;
                const double diff = std::fabs(double(recovered[out_i]) - double(dense_ref.out[out_i]));
                if (diff > max_diff) {
                    max_diff = diff;
                    worst = out_i;
                }
            }
        }
    }

    std::printf("[PASS] exact residual rho reference formula max_abs=%.9g worst=%zu\n",
                max_diff,
                worst);
    GGML_ASSERT(max_diff <= 1e-5);
}

static void test_cache_outputs_survive_until_persistence(AttentionRunner & runner) {
    constexpr int n_tensors = 8;
    const std::vector<float> values = runner.run_cache_output_lifetime_test(n_tensors, 4096);
    GGML_ASSERT(values.size() == n_tensors);
    for (int i = 0; i < n_tensors; ++i) {
        std::fprintf(stderr,
                     "[CACHE-LIFETIME] index=%d expected=%.1f got=%.9g\n",
                     i,
                     double(i + 1),
                     double(values[size_t(i)]));
        GGML_ASSERT(std::fabs(values[size_t(i)] - float(i + 1)) <= 1e-6f);
    }
}

static void test_attention_lse_node_matches_reference(AttentionRunner & runner) {
    constexpr int qo_len = 64;
    constexpr int kv_len = 96;
    constexpr int n_heads = 4;
    constexpr int n_kv_heads = 2;
    constexpr int head_dim = 64;
    constexpr float keep_ratio = 0.50f;

    std::vector<float> q = gen_q(qo_len, n_heads, head_dim, 21);
    std::vector<ggml_fp16_t> k = gen_kv_f16(kv_len, n_kv_heads, head_dim, 22);
    std::vector<ggml_fp16_t> v = gen_kv_f16(kv_len, n_kv_heads, head_dim, 23);

    std::vector<int32_t> block_index =
        runner.make_block_index(4, qo_len, kv_len, n_heads, n_kv_heads, head_dim, q, k, keep_ratio);
    std::vector<float> sparse_dense_mask =
        dense_mask_from_block_index(block_index, qo_len, kv_len, n_heads, n_kv_heads);

    const RefAttentionWithLse dense_ref =
        reference_attention_with_lse(qo_len, kv_len, n_heads, n_kv_heads, head_dim, q, k, v, nullptr);
    const RefAttentionWithLse sparse_ref =
        reference_attention_with_lse(qo_len, kv_len, n_heads, n_kv_heads, head_dim, q, k, v, &sparse_dense_mask);

    const std::vector<float> dense_lse =
        runner.run_cpu_lse(4, qo_len, kv_len, n_heads, n_kv_heads, head_dim, q, k, nullptr);
    const std::vector<float> sparse_lse =
        runner.run_cpu_lse(4, qo_len, kv_len, n_heads, n_kv_heads, head_dim, q, k, &sparse_dense_mask);
    const std::vector<float> sparse_index_lse =
        runner.run_cpu_lse_index(4, qo_len, kv_len, n_heads, n_kv_heads, head_dim, q, k, block_index);

    GGML_ASSERT(dense_lse.size() == dense_ref.lse.size());
    GGML_ASSERT(sparse_lse.size() == sparse_ref.lse.size());
    GGML_ASSERT(sparse_index_lse.size() == sparse_ref.lse.size());
    size_t worst_dense = 0;
    size_t worst_sparse = 0;
    size_t worst_sparse_index = 0;
    const double dense_diff = max_abs_diff(dense_ref.lse, dense_lse, &worst_dense);
    const double sparse_diff = max_abs_diff(sparse_ref.lse, sparse_lse, &worst_sparse);
    const double sparse_index_diff = max_abs_diff(sparse_ref.lse, sparse_index_lse, &worst_sparse_index);
    std::printf("[PASS] attention lse node dense_max_abs=%.9g sparse_max_abs=%.9g sparse_index_max_abs=%.9g worst_dense=%zu worst_sparse=%zu worst_sparse_index=%zu\n",
                dense_diff,
                sparse_diff,
                sparse_index_diff,
                worst_dense,
                worst_sparse,
                worst_sparse_index);
    GGML_ASSERT(dense_diff <= 1e-5);
    GGML_ASSERT(sparse_diff <= 1e-5);
    GGML_ASSERT(sparse_index_diff <= 1e-5);
}

static void test_htp_flash_attn_with_lse_matches_reference(AttentionRunner & runner) {
    {
        constexpr int qo_len = 64;
        constexpr int kv_len = 96;
        constexpr int n_heads = 1;
        constexpr int n_kv_heads = 1;
        constexpr int head_dim = 64;

        std::vector<float> q = gen_q(qo_len, n_heads, head_dim, 61);
        std::vector<ggml_fp16_t> k = gen_kv_f16(kv_len, n_kv_heads, head_dim, 62);
        std::vector<ggml_fp16_t> v = gen_kv_f16(kv_len, n_kv_heads, head_dim, 63);
        const RefAttentionWithLse dense_ref =
            reference_attention_with_lse(qo_len, kv_len, n_heads, n_kv_heads, head_dim, q, k, v, nullptr);
        const std::vector<float> dense_no_lse =
            runner.run(4, qo_len, kv_len, n_heads, n_kv_heads, head_dim, q, k, v, false, false);
        const std::vector<int32_t> full_index =
            full_keep_block_index(qo_len, kv_len, n_heads, n_kv_heads);
        const std::vector<float> sparse_packed =
            runner.run_htp_flash_with_lse_packed(4, qo_len, kv_len, n_heads, n_kv_heads, head_dim, q, k, v, &full_index);
        const PackedWithLseDiffStats stats =
            packed_with_lse_diff_stats(dense_no_lse, dense_ref.lse, sparse_packed, qo_len, n_heads, head_dim);
        std::printf("[DIAG] single-head full-keep sparse with-lse out_max_abs=%.9g lse_max_abs=%.9g out_worst=%zu lse_worst=%zu\n",
                    stats.out_max_abs,
                    stats.lse_max_abs,
                    stats.out_worst,
                    stats.lse_worst);
    }
    {
        constexpr int qo_len = 64;
        constexpr int kv_len = 96;
        constexpr int n_heads = 4;
        constexpr int n_kv_heads = 1;
        constexpr int head_dim = 64;

        std::vector<float> q = gen_q(qo_len, n_heads, head_dim, 71);
        std::vector<ggml_fp16_t> k = gen_kv_f16(kv_len, n_kv_heads, head_dim, 72);
        std::vector<ggml_fp16_t> v = gen_kv_f16(kv_len, n_kv_heads, head_dim, 73);
        const RefAttentionWithLse dense_ref =
            reference_attention_with_lse(qo_len, kv_len, n_heads, n_kv_heads, head_dim, q, k, v, nullptr);
        const std::vector<float> dense_no_lse =
            runner.run(4, qo_len, kv_len, n_heads, n_kv_heads, head_dim, q, k, v, false, false);
        const std::vector<int32_t> full_index =
            full_keep_block_index(qo_len, kv_len, n_heads, n_kv_heads);
        const std::vector<float> sparse_packed =
            runner.run_htp_flash_with_lse_packed(4, qo_len, kv_len, n_heads, n_kv_heads, head_dim, q, k, v, &full_index);
        const PackedWithLseDiffStats stats =
            packed_with_lse_diff_stats(dense_no_lse, dense_ref.lse, sparse_packed, qo_len, n_heads, head_dim);
        std::printf("[DIAG] multi-q-head single-kv full-keep sparse with-lse out_max_abs=%.9g lse_max_abs=%.9g out_worst=%zu lse_worst=%zu\n",
                    stats.out_max_abs,
                    stats.lse_max_abs,
                    stats.out_worst,
                    stats.lse_worst);
    }

    constexpr int qo_len = 64;
    constexpr int kv_len = 96;
    constexpr int n_heads = 4;
    constexpr int n_kv_heads = 4;
    constexpr int head_dim = 64;
    constexpr float keep_ratio = 0.50f;

    std::vector<float> q = gen_q(qo_len, n_heads, head_dim, 41);
    std::vector<ggml_fp16_t> k = gen_kv_f16(kv_len, n_kv_heads, head_dim, 42);
    std::vector<ggml_fp16_t> v = gen_kv_f16(kv_len, n_kv_heads, head_dim, 43);

    std::vector<int32_t> block_index =
        runner.make_block_index(4, qo_len, kv_len, n_heads, n_kv_heads, head_dim, q, k, keep_ratio);
    std::vector<float> sparse_dense_mask =
        dense_mask_from_block_index(block_index, qo_len, kv_len, n_heads, n_kv_heads);

    const RefAttentionWithLse dense_ref =
        reference_attention_with_lse(qo_len, kv_len, n_heads, n_kv_heads, head_dim, q, k, v, nullptr);
    const RefAttentionWithLse sparse_ref =
        reference_attention_with_lse(qo_len, kv_len, n_heads, n_kv_heads, head_dim, q, k, v, &sparse_dense_mask);

    std::fprintf(stderr, "[EXPECT] dense_htp_with_lse op_index=5 qo=%d kv=%d heads=%d kv_heads=%d d=%d\n",
                 qo_len, kv_len, n_heads, n_kv_heads, head_dim);
    const std::vector<float> dense_no_lse =
        runner.run(4, qo_len, kv_len, n_heads, n_kv_heads, head_dim, q, k, v, false, false);
    const std::vector<float> dense_packed =
        runner.run_htp_flash_with_lse_packed(4, qo_len, kv_len, n_heads, n_kv_heads, head_dim, q, k, v, nullptr);
    std::fprintf(stderr, "[EXPECT] sparse_htp_with_lse op_index=6 qo=%d kv=%d heads=%d kv_heads=%d d=%d\n",
                 qo_len, kv_len, n_heads, n_kv_heads, head_dim);
    const std::vector<float> sparse_no_lse =
        runner.run_htp_sparse_index(4, qo_len, kv_len, n_heads, n_kv_heads, head_dim, q, k, v, block_index);
    const std::vector<float> sparse_packed =
        runner.run_htp_flash_with_lse_packed(4, qo_len, kv_len, n_heads, n_kv_heads, head_dim, q, k, v, &block_index);
    const std::vector<int32_t> full_index =
        full_keep_block_index(qo_len, kv_len, n_heads, n_kv_heads);
    const std::vector<float> full_sparse_packed =
        runner.run_htp_flash_with_lse_packed(4, qo_len, kv_len, n_heads, n_kv_heads, head_dim, q, k, v, &full_index);

    GGML_ASSERT(dense_packed.size() == size_t(qo_len) * size_t(n_heads) * size_t(head_dim + 1));
    GGML_ASSERT(sparse_packed.size() == size_t(qo_len) * size_t(n_heads) * size_t(head_dim + 1));
    GGML_ASSERT(full_sparse_packed.size() == size_t(qo_len) * size_t(n_heads) * size_t(head_dim + 1));

    const PackedWithLseDiffStats dense_stats =
        packed_with_lse_diff_stats(dense_no_lse, dense_ref.lse, dense_packed, qo_len, n_heads, head_dim);
    const PackedWithLseDiffStats sparse_stats =
        packed_with_lse_diff_stats(sparse_no_lse, sparse_ref.lse, sparse_packed, qo_len, n_heads, head_dim);
    size_t sparse_linear_worst = 0;
    const double sparse_linear_prefix_diff =
        packed_linear_prefix_diff(sparse_no_lse, sparse_packed, &sparse_linear_worst);
    const size_t expected_out_size = size_t(qo_len) * size_t(n_heads) * size_t(head_dim);
    GGML_ASSERT(expected_out_size > 0);
    GGML_ASSERT(!sparse_ref.out.empty() && sparse_ref.out.size() == expected_out_size);
    GGML_ASSERT(!sparse_no_lse.empty() && sparse_no_lse.size() == expected_out_size);
    size_t sparse_no_lse_ref_worst = 0;
    const double sparse_no_lse_ref_diff =
        max_abs_diff(sparse_ref.out, sparse_no_lse, &sparse_no_lse_ref_worst);
    const PackedWithLseDiffStats full_sparse_stats =
        packed_with_lse_diff_stats(dense_no_lse, dense_ref.lse, full_sparse_packed, qo_len, n_heads, head_dim);
    std::printf("[PASS] htp flash-attn with-lse packed "
                "dense_out_max_abs=%.9g dense_lse_max_abs=%.9g dense_out_worst=%zu dense_lse_worst=%zu "
                "sparse_out_max_abs=%.9g sparse_lse_max_abs=%.9g sparse_out_worst=%zu sparse_lse_worst=%zu "
                "sparse_no_lse_linear_prefix_max_abs=%.9g sparse_no_lse_linear_worst=%zu "
                "sparse_no_lse_ref_max_abs=%.9g sparse_no_lse_ref_worst=%zu "
                "full_sparse_out_max_abs=%.9g full_sparse_lse_max_abs=%.9g "
                "full_sparse_out_worst=%zu full_sparse_lse_worst=%zu\n",
                dense_stats.out_max_abs,
                dense_stats.lse_max_abs,
                dense_stats.out_worst,
                dense_stats.lse_worst,
                sparse_stats.out_max_abs,
                sparse_stats.lse_max_abs,
                sparse_stats.out_worst,
                sparse_stats.lse_worst,
                sparse_linear_prefix_diff,
                sparse_linear_worst,
                sparse_no_lse_ref_diff,
                sparse_no_lse_ref_worst,
                full_sparse_stats.out_max_abs,
                full_sparse_stats.lse_max_abs,
                full_sparse_stats.out_worst,
                full_sparse_stats.lse_worst);
    GGML_ASSERT(dense_stats.out_max_abs <= 1e-5);
    GGML_ASSERT(dense_stats.lse_max_abs <= 5e-3);
    GGML_ASSERT(sparse_stats.out_max_abs <= 1e-5);
    GGML_ASSERT(sparse_stats.lse_max_abs <= 5e-3);
    GGML_ASSERT(sparse_no_lse_ref_diff <= 3e-3);
    GGML_ASSERT(full_sparse_stats.out_max_abs <= 1e-5);
    GGML_ASSERT(full_sparse_stats.lse_max_abs <= 5e-3);
}

static void test_htp_flash_attn_with_lse_production_layout(AttentionRunner & runner) {
    constexpr int qo_len = 410;
    constexpr int kv_len = 410;
    constexpr int n_heads = 24;
    constexpr int n_kv_heads = 24;
    constexpr int head_dim = 64;

    std::vector<float> q = gen_q(qo_len, n_heads, head_dim, 81);
    std::vector<ggml_fp16_t> k = gen_kv_f16(kv_len, n_kv_heads, head_dim, 82);
    std::vector<ggml_fp16_t> v = gen_kv_f16(kv_len, n_kv_heads, head_dim, 83);
    const std::vector<int32_t> block_index =
        runner.make_block_index(4, qo_len, kv_len, n_heads, n_kv_heads, head_dim, q, k, 0.99f);

    const std::vector<float> dense_no_lse =
        runner.run(4, qo_len, kv_len, n_heads, n_kv_heads, head_dim, q, k, v, false, false);
    const std::vector<float> dense_lse =
        runner.run_cpu_lse(4, qo_len, kv_len, n_heads, n_kv_heads, head_dim, q, k, nullptr);
    const std::vector<float> dense_packed =
        runner.run_htp_flash_with_lse_packed(4, qo_len, kv_len, n_heads, n_kv_heads, head_dim, q, k, v, nullptr);
    const PackedWithLseDiffStats dense_stats =
        packed_with_lse_diff_stats(dense_no_lse, dense_lse, dense_packed, qo_len, n_heads, head_dim);

    const std::vector<float> sparse_no_lse =
        runner.run_htp_sparse_index(4, qo_len, kv_len, n_heads, n_kv_heads, head_dim, q, k, v, block_index);
    const std::vector<float> sparse_lse =
        runner.run_cpu_lse_index(4, qo_len, kv_len, n_heads, n_kv_heads, head_dim, q, k, block_index);
    const std::vector<float> sparse_packed =
        runner.run_htp_flash_with_lse_packed(4, qo_len, kv_len, n_heads, n_kv_heads, head_dim, q, k, v, &block_index);
    const PackedWithLseDiffStats sparse_stats =
        packed_with_lse_diff_stats(sparse_no_lse, sparse_lse, sparse_packed, qo_len, n_heads, head_dim);

    std::printf("[PRODUCTION] htp flash-attn with-lse packed "
                "dense_out_max_abs=%.9g dense_lse_max_abs=%.9g dense_out_worst=%zu dense_lse_worst=%zu "
                "sparse_out_max_abs=%.9g sparse_lse_max_abs=%.9g sparse_out_worst=%zu sparse_lse_worst=%zu\n",
                dense_stats.out_max_abs,
                dense_stats.lse_max_abs,
                dense_stats.out_worst,
                dense_stats.lse_worst,
                sparse_stats.out_max_abs,
                sparse_stats.lse_max_abs,
                sparse_stats.out_worst,
                sparse_stats.lse_worst);

    GGML_ASSERT(dense_stats.out_max_abs <= 1e-5);
    GGML_ASSERT(dense_stats.lse_max_abs <= 5e-3);
    GGML_ASSERT(sparse_stats.out_max_abs <= 1e-5);
    GGML_ASSERT(sparse_stats.lse_max_abs <= 5e-3);
}

static void test_exact_rho_anchor_graph_sets_sparse_flash_block_params(AttentionRunner & runner) {
    constexpr int qo_len = 64;
    constexpr int kv_len = 96;
    constexpr int n_heads = 4;
    constexpr int n_kv_heads = 2;
    constexpr int head_dim = 64;

    std::vector<float> q = gen_q(qo_len, n_heads, head_dim, 51);
    std::vector<ggml_fp16_t> k = gen_kv_f16(kv_len, n_kv_heads, head_dim, 52);
    std::vector<ggml_fp16_t> v = gen_kv_f16(kv_len, n_kv_heads, head_dim, 53);

    const int missing =
        runner.count_exact_rho_anchor_flash_nodes_missing_block_params(qo_len, kv_len, n_heads, n_kv_heads, head_dim, q, k, v);
    GGML_ASSERT(missing == 0);
}

static void test_exact_rho_recovered_graph_matches_dense(AttentionRunner & runner) {
    constexpr int qo_len = 64;
    constexpr int kv_len = 96;
    constexpr int n_heads = 4;
    constexpr int n_kv_heads = 2;
    constexpr int head_dim = 64;
    constexpr float keep_ratio = 0.50f;

    std::vector<float> q = gen_q(qo_len, n_heads, head_dim, 31);
    std::vector<ggml_fp16_t> k = gen_kv_f16(kv_len, n_kv_heads, head_dim, 32);
    std::vector<ggml_fp16_t> v = gen_kv_f16(kv_len, n_kv_heads, head_dim, 33);

    std::vector<int32_t> block_index =
        runner.make_block_index(4, qo_len, kv_len, n_heads, n_kv_heads, head_dim, q, k, keep_ratio);
    const std::vector<float> dense =
        runner.run(4, qo_len, kv_len, n_heads, n_kv_heads, head_dim, q, k, v, false, false);
    const std::vector<float> recovered =
        runner.run_exact_rho_recovered_graph(4, qo_len, kv_len, n_heads, n_kv_heads, head_dim, q, k, v, block_index);

    size_t worst = 0;
    const double diff = max_abs_diff(dense, recovered, &worst);
    GGML_ASSERT(diff <= 3e-3);
    std::printf("[PASS] exact rho recovered graph max_abs=%.9g worst=%zu dense=%.9g recovered=%.9g\n",
                diff,
                worst,
                dense.empty() ? 0.0 : double(dense[worst]),
                recovered.empty() ? 0.0 : double(recovered[worst]));
}

static void test_dense_exp2_paths_match_cpu(AttentionRunner & runner) {
    test_diff_helpers_handle_non_finite_values();
    test_packed_diff_helpers_handle_non_finite_values();

    struct DenseExp2Case {
        int qo_len;
        int kv_len;
        int n_heads;
        int n_kv_heads;
        int head_dim;
        const char * name;
    };

    const DenseExp2Case cases[] = {
        {65, 95, 4, 4, 64, "dense-d64-tail"},
        {65, 95, 4, 4, 80, "sp-hdim-d80-tail"},
        {33, 97, 4, 2, 96, "sp-hdim-d96-gqa-tail"},
    };

    bool all_passed = true;
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        const DenseExp2Case & test_case = cases[i];
        std::vector<float> q =
            gen_q(test_case.qo_len, test_case.n_heads, test_case.head_dim, 101 + int(i) * 3);
        std::vector<ggml_fp16_t> k =
            gen_kv_f16(test_case.kv_len, test_case.n_kv_heads, test_case.head_dim, 102 + int(i) * 3);
        std::vector<ggml_fp16_t> v =
            gen_kv_f16(test_case.kv_len, test_case.n_kv_heads, test_case.head_dim, 103 + int(i) * 3);
        std::vector<float> dense_mask(size_t(test_case.kv_len) * size_t(test_case.qo_len) *
                                         size_t(test_case.n_heads),
                                     0.0f);

        const std::vector<float> cpu =
            runner.run_cpu_dense_masked(4,
                                        test_case.qo_len,
                                        test_case.kv_len,
                                        test_case.n_heads,
                                        test_case.n_kv_heads,
                                        test_case.head_dim,
                                        q,
                                        k,
                                        v,
                                        dense_mask);
        std::fprintf(stderr,
                     "[EXPECT] phase1-exp2 case=%s expected_htp_core=%s qo=%d kv=%d heads=%d kv_heads=%d d=%d\n",
                     test_case.name,
                     test_case.head_dim % 64 == 0 ? "simple_flash_attn" : "simple_flash_attn_sp_hdim",
                     test_case.qo_len,
                     test_case.kv_len,
                     test_case.n_heads,
                     test_case.n_kv_heads,
                     test_case.head_dim);
        const std::vector<float> htp =
            runner.run(4,
                       test_case.qo_len,
                       test_case.kv_len,
                       test_case.n_heads,
                       test_case.n_kv_heads,
                       test_case.head_dim,
                       q,
                       k,
                       v,
                       false,
                       false);
        const size_t expected_output_size =
            size_t(test_case.qo_len) * size_t(test_case.n_heads) * size_t(test_case.head_dim);
        GGML_ASSERT(expected_output_size > 0);
        GGML_ASSERT(!cpu.empty() && cpu.size() == expected_output_size);
        GGML_ASSERT(!htp.empty() && htp.size() == expected_output_size);
        const DiffStats stats = diff_stats(cpu, htp);
        const bool passed = stats.max_abs <= 1e-2 && stats.rms_abs <= 1e-2;
        std::printf("[%s] phase1-exp2 case=%s max_abs=%.9g mean_abs=%.9g rms_abs=%.9g "
                    "ref_rms=%.9g rel_rms=%.9g worst=%zu cpu=%.9g htp=%.9g\n",
                    passed ? "PASS" : "FAIL",
                    test_case.name,
                    stats.max_abs,
                    stats.mean_abs,
                    stats.rms_abs,
                    stats.ref_rms,
                    stats.rel_rms,
                    stats.worst,
                    cpu.empty() ? 0.0 : double(cpu[stats.worst]),
                    htp.empty() ? 0.0 : double(htp[stats.worst]));
        std::fflush(stdout);
        all_passed = all_passed && passed;
    }
    GGML_ASSERT(all_passed);
}

static void run_case(AttentionRunner & runner, int qo_len, int kv_len, int n_heads, int n_kv_heads, int head_dim) {
    std::vector<float> q = gen_q(qo_len, n_heads, head_dim, 1);
    std::vector<ggml_fp16_t> k = gen_kv_f16(kv_len, n_kv_heads, head_dim, 2);
    std::vector<ggml_fp16_t> v = gen_kv_f16(kv_len, n_kv_heads, head_dim, 3);

    std::fprintf(stderr, "[EXPECT] dense_htp op_index=5 qo=%d kv=%d heads=%d kv_heads=%d d=%d\n",
                 qo_len, kv_len, n_heads, n_kv_heads, head_dim);
    std::vector<float> dense = runner.run(4, qo_len, kv_len, n_heads, n_kv_heads, head_dim, q, k, v, false, false);
    std::fprintf(stderr, "[EXPECT] sparse_htp_ratio1 op_index=6 qo=%d kv=%d heads=%d kv_heads=%d d=%d\n",
                 qo_len, kv_len, n_heads, n_kv_heads, head_dim);
    std::vector<float> sparse = runner.run(4, qo_len, kv_len, n_heads, n_kv_heads, head_dim, q, k, v, true, false, 1.0f);

    size_t worst = 0;
    const double diff = max_abs_diff(dense, sparse, &worst);
    std::printf("[DIFF] qo=%d kv=%d heads=%d kv_heads=%d d=%d max_abs=%.9g worst=%zu dense=%.9g sparse=%.9g\n",
                qo_len,
                kv_len,
                n_heads,
                n_kv_heads,
                head_dim,
                diff,
                worst,
                dense.empty() ? 0.0 : double(dense[worst]),
                sparse.empty() ? 0.0 : double(sparse[worst]));
    GGML_ASSERT(diff <= 1e-3);

    for (const float keep_ratio : {0.99f, 0.95f}) {
        std::fprintf(stderr,
                     "[EXPECT] sparse_htp_ratio%.2f op_index=6 qo=%d kv=%d heads=%d kv_heads=%d d=%d\n",
                     double(keep_ratio),
                     qo_len,
                     kv_len,
                     n_heads,
                     n_kv_heads,
                     head_dim);
        std::vector<float> partial_sparse =
            runner.run(4, qo_len, kv_len, n_heads, n_kv_heads, head_dim, q, k, v, true, false, keep_ratio);
        const DiffStats stats = diff_stats(dense, partial_sparse);
        std::printf("[APPROX_DIFF] ratio=%.2f qo=%d kv=%d heads=%d kv_heads=%d d=%d max_abs=%.9g mean_abs=%.9g rms_abs=%.9g ref_rms=%.9g rel_rms=%.9g worst=%zu dense=%.9g sparse=%.9g\n",
                    double(keep_ratio),
                    qo_len,
                    kv_len,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    stats.max_abs,
                    stats.mean_abs,
                    stats.rms_abs,
                    stats.ref_rms,
                    stats.rel_rms,
                    stats.worst,
                    dense.empty() ? 0.0 : double(dense[stats.worst]),
                    partial_sparse.empty() ? 0.0 : double(partial_sparse[stats.worst]));

        std::fprintf(stderr,
                     "[EXPECT] same_mask_ratio%.2f make_index qo=%d kv=%d heads=%d kv_heads=%d d=%d\n",
                     double(keep_ratio),
                     qo_len,
                     kv_len,
                     n_heads,
                     n_kv_heads,
                     head_dim);
        std::vector<int32_t> block_index =
            runner.make_block_index(4, qo_len, kv_len, n_heads, n_kv_heads, head_dim, q, k, keep_ratio);
        std::vector<float> dense_mask =
            dense_mask_from_block_index(block_index, qo_len, kv_len, n_heads, n_kv_heads);
        std::vector<float> cpu_masked =
            runner.run_cpu_dense_masked(4, qo_len, kv_len, n_heads, n_kv_heads, head_dim, q, k, v, dense_mask);
        std::vector<float> htp_sparse_same_mask =
            runner.run_htp_sparse_index(4, qo_len, kv_len, n_heads, n_kv_heads, head_dim, q, k, v, block_index);
        const size_t expected_output_size = size_t(qo_len) * size_t(n_heads) * size_t(head_dim);
        GGML_ASSERT(expected_output_size > 0);
        GGML_ASSERT(!cpu_masked.empty() && cpu_masked.size() == expected_output_size);
        GGML_ASSERT(!htp_sparse_same_mask.empty() && htp_sparse_same_mask.size() == expected_output_size);
        const DiffStats same_mask_stats = diff_stats(cpu_masked, htp_sparse_same_mask);
        std::printf("[SAME_MASK_DIFF] ratio=%.2f qo=%d kv=%d heads=%d kv_heads=%d d=%d max_abs=%.9g mean_abs=%.9g rms_abs=%.9g ref_rms=%.9g rel_rms=%.9g worst=%zu cpu=%.9g sparse=%.9g\n",
                    double(keep_ratio),
                    qo_len,
                    kv_len,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    same_mask_stats.max_abs,
                    same_mask_stats.mean_abs,
                    same_mask_stats.rms_abs,
                    same_mask_stats.ref_rms,
                    same_mask_stats.rel_rms,
                    same_mask_stats.worst,
                    cpu_masked.empty() ? 0.0 : double(cpu_masked[same_mask_stats.worst]),
                    htp_sparse_same_mask.empty() ? 0.0 : double(htp_sparse_same_mask[same_mask_stats.worst]));
        GGML_ASSERT(same_mask_stats.max_abs <= 3e-3);
    }
}

static void run_equivalent_mask_benchmark(AttentionRunner & runner) {
    constexpr int qo_len = 410;
    constexpr int kv_len = 410;
    constexpr int n_heads = 24;
    constexpr int n_kv_heads = 24;
    constexpr int head_dim = 64;
    constexpr int warmup_pairs = 5;
    constexpr int measured_pairs = 30;

    std::vector<float> q = gen_q(qo_len, n_heads, head_dim, 91);
    std::vector<ggml_fp16_t> k = gen_kv_f16(kv_len, n_kv_heads, head_dim, 92);
    std::vector<ggml_fp16_t> v = gen_kv_f16(kv_len, n_kv_heads, head_dim, 93);
    const EquivalentBenchmarkMasks masks =
        make_equivalent_benchmark_masks(qo_len, kv_len, n_heads, n_kv_heads);
    GGML_ASSERT(masks.min_kept < masks.max_kept);

    std::fprintf(stderr,
                 "[HTP_ATTN_BENCH] begin qo=%d kv=%d heads=%d kv_heads=%d d=%d "
                 "warmup_pairs=%d measured_pairs=%d min_kept=%d max_kept=%d mean_kept=%.3f\n",
                 qo_len,
                 kv_len,
                 n_heads,
                 n_kv_heads,
                 head_dim,
                 warmup_pairs,
                 measured_pairs,
                 masks.min_kept,
                 masks.max_kept,
                 masks.mean_kept);

    for (int pair = 0; pair < warmup_pairs + measured_pairs; ++pair) {
        std::vector<float> simple;
        std::vector<float> sparse;
        const bool simple_first = (pair % 2) == 0;
        if (simple_first) {
            simple = runner.run_htp_dense_f16_mask(4, qo_len, kv_len, n_heads, n_kv_heads, head_dim, q, k, v, masks.dense_mask);
            sparse = runner.run_htp_sparse_index(4, qo_len, kv_len, n_heads, n_kv_heads, head_dim, q, k, v, masks.sparse_index);
        } else {
            sparse = runner.run_htp_sparse_index(4, qo_len, kv_len, n_heads, n_kv_heads, head_dim, q, k, v, masks.sparse_index);
            simple = runner.run_htp_dense_f16_mask(4, qo_len, kv_len, n_heads, n_kv_heads, head_dim, q, k, v, masks.dense_mask);
        }

        if (pair == 0) {
            size_t worst = 0;
            const double diff = max_abs_diff(simple, sparse, &worst);
            std::fprintf(stderr,
                         "[HTP_ATTN_BENCH] equivalent_mask_max_abs=%.9g worst=%zu simple=%.9g sparse=%.9g\n",
                         diff,
                         worst,
                         double(simple[worst]),
                         double(sparse[worst]));
            GGML_ASSERT(diff <= 3e-3);
        }
        if (pair + 1 == warmup_pairs) {
            std::fprintf(stderr, "[HTP_ATTN_BENCH] warmup_complete\n");
        }
    }
    std::fprintf(stderr, "[HTP_ATTN_BENCH] complete\n");
}

static void run_image_size_r90_benchmark(AttentionRunner & runner) {
    struct BenchmarkCase {
        const char * attention;
        int image_size;
        int seq_len;
    };

    constexpr int n_heads = 24;
    constexpr int n_kv_heads = 24;
    constexpr int head_dim = 64;
    constexpr float keep_ratio = 0.90f;
    constexpr int warmup_pairs = 3;
    constexpr int measured_pairs = 15;
    const BenchmarkCase cases[] = {
        {"self", 256, 256},
        {"joint", 256, 410},
        {"self", 512, 1024},
        {"joint", 512, 1178},
        {"self", 1024, 4096},
        {"joint", 1024, 4250},
    };

    GGML_ASSERT(sd_htp_block_sparse_score_enabled(keep_ratio));
    for (size_t case_idx = 0; case_idx < std::size(cases); ++case_idx) {
        const BenchmarkCase & test_case = cases[case_idx];
        const int qo_len = test_case.seq_len;
        const int kv_len = test_case.seq_len;
        std::vector<float> q = gen_q(qo_len, n_heads, head_dim, 101 + int(case_idx) * 3);
        std::vector<ggml_fp16_t> k =
            gen_kv_f16(kv_len, n_kv_heads, head_dim, 102 + int(case_idx) * 3);
        std::vector<ggml_fp16_t> v =
            gen_kv_f16(kv_len, n_kv_heads, head_dim, 103 + int(case_idx) * 3);

        const bool is_joint = std::strcmp(test_case.attention, "joint") == 0;
        const int protected_start = is_joint ? test_case.image_size * test_case.image_size / 256 : 0;
        const int protected_len = is_joint ? 154 : 0;
        std::vector<int32_t> sparse_index = runner.make_block_index(4,
                                                                    qo_len,
                                                                    kv_len,
                                                                    n_heads,
                                                                    n_kv_heads,
                                                                    head_dim,
                                                                    q,
                                                                    k,
                                                                    keep_ratio,
                                                                    protected_start,
                                                                    protected_len,
                                                                    true);
        const int q_blocks_per_head = (qo_len + SD_HTP_BLOCK_SPARSE_MASK_BLOCK - 1) /
                                      SD_HTP_BLOCK_SPARSE_MASK_BLOCK;
        const int k_blocks_per_head = (kv_len + SD_HTP_BLOCK_SPARSE_MASK_BLOCK - 1) /
                                      SD_HTP_BLOCK_SPARSE_MASK_BLOCK;
        GGML_ASSERT(sparse_index.size() ==
                    size_t(n_heads) * size_t(q_blocks_per_head) * size_t(k_blocks_per_head));
        size_t kept_blocks = 0;
        for (const int32_t block : sparse_index) {
            kept_blocks += block >= 0 ? 1 : 0;
        }
        const size_t total_blocks = size_t(n_heads) * size_t(q_blocks_per_head) * size_t(k_blocks_per_head);
        const double actual_keep = total_blocks == 0 ? 0.0 : double(kept_blocks) / double(total_blocks);

        const int kv_pad_len = ((kv_len + 63) / 64) * 64;
        const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f);
        std::vector<ggml_fp16_t> dense_mask(size_t(qo_len) * size_t(kv_pad_len), zero);

        std::fprintf(stderr,
                     "[HTP_IMAGE_ATTN_BENCH] begin image=%d attention=%s qo=%d kv=%d heads=%d "
                     "kv_heads=%d d=%d requested_keep=%.3f actual_block_keep=%.6f kept=%zu total=%zu "
                     "protected_start=%d protected_len=%d "
                     "warmup_pairs=%d measured_pairs=%d\n",
                     test_case.image_size,
                     test_case.attention,
                     qo_len,
                     kv_len,
                     n_heads,
                     n_kv_heads,
                     head_dim,
                     double(keep_ratio),
                     actual_keep,
                     kept_blocks,
                     total_blocks,
                     protected_start,
                     protected_len,
                     warmup_pairs,
                     measured_pairs);
        std::fflush(stderr);

        for (int pair = 0; pair < warmup_pairs + measured_pairs; ++pair) {
            const bool dense_first = (pair % 2) == 0;
            if (dense_first) {
                (void) runner.run_htp_dense_f16_mask(
                    4, qo_len, kv_len, n_heads, n_kv_heads, head_dim, q, k, v, dense_mask);
                (void) runner.run_htp_sparse_index(
                    4, qo_len, kv_len, n_heads, n_kv_heads, head_dim, q, k, v, sparse_index);
            } else {
                (void) runner.run_htp_sparse_index(
                    4, qo_len, kv_len, n_heads, n_kv_heads, head_dim, q, k, v, sparse_index);
                (void) runner.run_htp_dense_f16_mask(
                    4, qo_len, kv_len, n_heads, n_kv_heads, head_dim, q, k, v, dense_mask);
            }
            if (pair + 1 == warmup_pairs) {
                std::fprintf(stderr,
                             "[HTP_IMAGE_ATTN_BENCH] warmup_complete image=%d attention=%s\n",
                             test_case.image_size,
                             test_case.attention);
                std::fflush(stderr);
            }
        }
        std::fprintf(stderr,
                     "[HTP_IMAGE_ATTN_BENCH] case_complete image=%d attention=%s\n",
                     test_case.image_size,
                     test_case.attention);
        std::fflush(stderr);
    }
    std::fprintf(stderr, "[HTP_IMAGE_ATTN_BENCH] complete\n");
}

struct NamedTestCase {
    const char * name;
    bool implemented;
};

static const NamedTestCase named_test_cases[] = {
    {"phase1-exp2", true},
    {"phase1-lse-rho", true},
    {"production-410", true},
    {"cache-full-kv-4250", true},
    {"cache-full-k-4250-d128", true},
    {"cache-chunked-5281-d128", true},
    {"cache-d128-multi-kv", true},
    {"static-halo-baseline", true},
    {"static-halo-baseline-d128", true},
    {"static-halo-l2", true},
    {"static-halo-l2-d128", true},
    {"static-halo-l2-2h", true},
    {"static-halo-l2-mixed-fallback", true},
    {"static-halo-l2-6x4", true},
    {"static-halo-l2-5x3", true},
    {"static-halo-userdma", true},
    {"static-halo-userdma-2h", true},
    {"static-halo-userdma-mixed-fallback", true},
    {"static-halo-userdma-d128", true},
    {"static-halo-userdma-6x4", true},
    {"static-halo-userdma-5x3", true},
    {"static-halo-userdma-fallback", true},
    {"static-halo-stable-normal-high-logit", true},
    {"benchmark-410", true},
    {"benchmark-image-sizes-r90", true},
};

static void print_named_test_cases(FILE * stream) {
    std::fprintf(stream, "Available test cases:\n");
    for (const NamedTestCase & test_case : named_test_cases) {
        std::fprintf(stream, "  %s%s\n", test_case.name, test_case.implemented ? "" : " (not implemented)");
    }
}

static const NamedTestCase * find_named_test_case(const std::string & name) {
    for (const NamedTestCase & test_case : named_test_cases) {
        if (name == test_case.name) {
            return &test_case;
        }
    }
    return nullptr;
}

static void run_phase1_lse_rho_tests(AttentionRunner & runner) {
    test_exact_residual_rho_reference_formula(runner);
    test_attention_lse_node_matches_reference(runner);
    test_exact_rho_anchor_graph_sets_sparse_flash_block_params(runner);
    test_htp_flash_attn_with_lse_matches_reference(runner);
    test_htp_flash_attn_with_lse_production_layout(runner);
    test_exact_rho_recovered_graph_matches_dense(runner);
}

int main(int argc, char ** argv) {
    if (argc == 2 && std::string(argv[1]) == "--list") {
        print_named_test_cases(stdout);
        return 0;
    }

    std::string selected_case;
    if (argc == 1) {
        if (sd_env_flag_enabled("SD_HTP_ATTENTION_BENCHMARK")) {
            selected_case = "benchmark-410";
        }
    } else if (argc == 3 && std::string(argv[1]) == "--case" && argv[2][0] != '\0') {
        selected_case = argv[2];
    } else {
        std::fprintf(stderr, "Usage: %s [--list | --case NAME]\n", argv[0]);
        print_named_test_cases(stderr);
        return 2;
    }

    if (!selected_case.empty()) {
        const NamedTestCase * test_case = find_named_test_case(selected_case);
        if (test_case == nullptr) {
            std::fprintf(stderr, "Unknown test case: %s\n", selected_case.c_str());
            print_named_test_cases(stderr);
            return 2;
        }
        if (!test_case->implemented) {
            std::fprintf(stderr,
                         "[UNIMPLEMENTED] test case %s is reserved for phase 2\n",
                         selected_case.c_str());
            return 3;
        }
    }

    test_block_sparse_keep_ratio_policy();
    test_block_sparse_mask_format_is_explicit_for_ambiguous_shapes();
    test_flash_attn_static_halo_metadata();
    test_halo_kv_load_mode_selector();

    if (!ggml_backend_htp_runtime_available()) {
        std::printf("[SKIP] HTP runtime unavailable (%s)\n", ggml_backend_htp_runtime_error());
        return 0;
    }

    ggml_backend_t cpu = ggml_backend_init_by_name("CPU", nullptr);
    ggml_backend_htp_set_hybrid_enabled(true);
    ggml_backend_t htp = ggml_backend_init_by_name("HTP", nullptr);
    GGML_ASSERT(cpu != nullptr && htp != nullptr);

    AttentionRunner runner(cpu, htp);

    if (selected_case == "benchmark-410") {
        run_equivalent_mask_benchmark(runner);
    } else if (selected_case == "benchmark-image-sizes-r90") {
        run_image_size_r90_benchmark(runner);
    } else if (selected_case == "phase1-exp2") {
        test_dense_exp2_paths_match_cpu(runner);
    } else if (selected_case == "phase1-lse-rho") {
        run_phase1_lse_rho_tests(runner);
    } else if (selected_case == "production-410") {
        test_htp_flash_attn_with_lse_production_layout(runner);
        run_case(runner, 410, 410, 24, 24, 64);
    } else if (selected_case == "cache-full-k-4250-d128") {
        test_cache_full_k_4250_d128(runner);
    } else if (selected_case == "cache-full-kv-4250") {
        test_cache_full_kv_4250(runner);
    } else if (selected_case == "cache-chunked-5281-d128") {
        test_cache_chunked_5281_d128(runner);
    } else if (selected_case == "cache-d128-multi-kv") {
        test_cache_d128_multi_kv(runner);
    } else if (selected_case == "static-halo-baseline") {
        test_static_halo_characterization(runner, "baseline");
    } else if (selected_case == "static-halo-baseline-d128") {
        test_static_halo_characterization(runner, "baseline", 1, 128);
    } else if (selected_case == "static-halo-l2") {
        test_static_halo_characterization(runner, "l2fetch");
    } else if (selected_case == "static-halo-l2-d128") {
        test_static_halo_characterization(runner, "l2fetch", 1, 128);
    } else if (selected_case == "static-halo-l2-2h") {
        test_static_halo_characterization(runner, "l2fetch", 2);
    } else if (selected_case == "static-halo-l2-mixed-fallback") {
        test_static_halo_characterization(runner, "l2fetch", 2, 64, 16, 32, true, true);
    } else if (selected_case == "static-halo-l2-6x4") {
        test_static_halo_characterization(runner, "l2fetch", 1, 64, 24, 32);
    } else if (selected_case == "static-halo-l2-5x3") {
        test_static_halo_characterization(runner, "l2fetch", 1, 64, 20, 24);
    } else if (selected_case == "static-halo-userdma") {
        test_static_halo_characterization(runner, "userdma");
    } else if (selected_case == "static-halo-userdma-2h") {
        test_static_halo_characterization(runner, "userdma", 2);
    } else if (selected_case == "static-halo-userdma-mixed-fallback") {
        test_static_halo_characterization(runner, "userdma", 2, 64, 16, 32, true, true);
    } else if (selected_case == "static-halo-userdma-d128") {
        test_static_halo_characterization(runner, "userdma", 1, 128);
    } else if (selected_case == "static-halo-userdma-6x4") {
        test_static_halo_characterization(runner, "userdma", 1, 64, 24, 32);
    } else if (selected_case == "static-halo-userdma-5x3") {
        test_static_halo_characterization(runner, "userdma", 1, 64, 20, 24);
    } else if (selected_case == "static-halo-userdma-fallback") {
        test_static_halo_characterization(runner, "userdma", 1, 64, 16, 32, false);
    } else if (selected_case == "static-halo-stable-normal-high-logit") {
        test_static_halo_stable_normal_high_logit(runner);
    } else {
        test_cache_outputs_survive_until_persistence(runner);
        test_dense_exp2_paths_match_cpu(runner);
        run_phase1_lse_rho_tests(runner);

        run_case(runner, 256, 256, 24, 24, 64);
        run_case(runner, 410, 410, 24, 24, 64);
        run_case(runner, 512, 512, 24, 24, 64);
        test_static_halo_characterization(runner, "baseline");
        test_static_halo_characterization(runner, "l2fetch");
        test_static_halo_characterization(runner, "userdma");
    }

    ggml_backend_free(htp);
    ggml_backend_free(cpu);

    if (selected_case.empty()) {
        std::printf("\n=== HTP dense vs sparse full-keep test PASSED ===\n");
    } else {
        std::printf("\n[PASS] test case %s\n", selected_case.c_str());
    }
    return 0;
}
