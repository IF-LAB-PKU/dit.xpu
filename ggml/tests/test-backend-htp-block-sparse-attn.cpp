// Runner-style HTP dense vs block-sparse full-keep attention regression.
// Dense uses the same HTP kernel contract as inference (Q/O F32, K/V F16, F16 zero mask).
// Sparse uses the real inference block-sparse path with keep_ratio=1.0 generated from score/mask.

#include "../../src/ggml_extend.hpp"

#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-htp.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

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
        setenv(name, value, 1);
    }

    ~EnvGuard() {
        if (had_old) {
            setenv(name, old_value.c_str(), 1);
        } else {
            unsetenv(name);
        }
    }
};

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

static double max_abs_diff(const std::vector<float> & a, const std::vector<float> & b, size_t * worst = nullptr) {
    GGML_ASSERT(a.size() == b.size());
    double max_diff = 0.0;
    size_t worst_idx = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double diff = std::fabs(double(a[i]) - double(b[i]));
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
        const double diff = double(got[i]) - double(ref[i]);
        const double abs_diff = std::fabs(diff);
        if (abs_diff > stats.max_abs) {
            stats.max_abs = abs_diff;
            stats.worst = i;
        }
        sum_abs += abs_diff;
        sum_sq += diff * diff;
        ref_sq += double(ref[i]) * double(ref[i]);
    }
    if (!ref.empty()) {
        stats.mean_abs = sum_abs / double(ref.size());
        stats.rms_abs = std::sqrt(sum_sq / double(ref.size()));
        stats.ref_rms = std::sqrt(ref_sq / double(ref.size()));
        stats.rel_rms = stats.ref_rms > 0.0 ? stats.rms_abs / stats.ref_rms : 0.0;
    }
    return stats;
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
        EnvGuard keep_ratio("SD_HTP_BLOCK_SPARSE_KEEP_RATIO", block_sparse_enabled ? keep_ratio_buf : "0.8");

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
                                          float keep_ratio_value) {
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

            const int block_q = SD_HTP_BLOCK_SPARSE_MASK_BLOCK;
            const int block_k = SD_HTP_BLOCK_SPARSE_MASK_BLOCK;
            const int n_k_blocks = n_kv_heads * ((kv_len + block_k - 1) / block_k);
            const int top_k = std::max(1, std::min(sd_env_int_value("SD_HTP_BLOCK_SPARSE_TOPK", 2), n_k_blocks));
            ggml_tensor * q_mean = ggml_ext_block_sparse_q_mean_node(runner_ctx.ggml_ctx, q, block_q);
            ggml_tensor * k_mean = ggml_ext_block_sparse_k_mean_node(runner_ctx.ggml_ctx, k, block_k);
            ggml_tensor * score  = ggml_ext_block_sparse_score_topk_node(runner_ctx.ggml_ctx, q_mean, k_mean, block_q, block_k, top_k, false);
            ggml_tensor * index  = ggml_ext_block_sparse_index_from_score_node(runner_ctx.ggml_ctx, score, block_q, block_k, top_k, false, 0, 0, n_heads, n_kv_heads);

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

    std::vector<float> run_htp_sparse_index(int n_threads,
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
            out->op_params[4] = block_q;
            out->op_params[5] = block_k;
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
            sparse_out->op_params[4] = block_q;
            sparse_out->op_params[5] = block_k;
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
                                                                   1.0f / std::sqrt(float(head_dim)));
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
    }
}

int main() {
    if (!ggml_backend_htp_runtime_available()) {
        std::printf("[SKIP] HTP runtime unavailable (%s)\n", ggml_backend_htp_runtime_error());
        return 0;
    }

    ggml_backend_t cpu = ggml_backend_init_by_name("CPU", nullptr);
    ggml_backend_htp_set_hybrid_enabled(true);
    ggml_backend_t htp = ggml_backend_init_by_name("HTP", nullptr);
    GGML_ASSERT(cpu != nullptr && htp != nullptr);

    AttentionRunner runner(cpu, htp);

    test_exact_residual_rho_reference_formula(runner);
    test_attention_lse_node_matches_reference(runner);
    test_exact_rho_recovered_graph_matches_dense(runner);

    run_case(runner, 256, 256, 24, 24, 64);
    run_case(runner, 410, 410, 24, 24, 64);
    run_case(runner, 512, 512, 24, 24, 64);

    ggml_backend_free(htp);
    ggml_backend_free(cpu);

    std::printf("\n=== HTP dense vs sparse full-keep test PASSED ===\n");
    return 0;
}
