// Runner-style HTP dense vs block-sparse full-keep attention regression.
// Dense uses the same HTP kernel contract as inference (Q/O F32, K/V F16, F16 zero mask).
// Sparse uses the real inference block-sparse path with keep_ratio=1.0 generated from score/mask.

#include "../../src/ggml_extend.hpp"

#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-htp.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
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
                           bool force_full_keep) {
        set_flash_attention_enabled(true);
        set_htp_attention_enabled(true);
        set_block_sparse_attention_enabled(block_sparse_enabled);

        EnvGuard htp_ops("SD_HTP_OPS", "1");
        EnvGuard htp_compute("SD_HTP_COMPUTE", "1");
        EnvGuard htp_attention("SD_HTP_ATTENTION", "1");
        EnvGuard block_sparse_attn("SD_HTP_BLOCK_SPARSE_ATTN", block_sparse_enabled ? "1" : "0");
        EnvGuard full_keep_flag("SD_HTP_BLOCK_SPARSE_FORCE_FULL_KEEP", force_full_keep ? "1" : "0");
        EnvGuard keep_ratio("SD_HTP_BLOCK_SPARSE_KEEP_RATIO", block_sparse_enabled ? "1.0" : "0.8");

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
};

static void run_case(AttentionRunner & runner, int qo_len, int kv_len, int n_heads, int n_kv_heads, int head_dim) {
    std::vector<float> q = gen_q(qo_len, n_heads, head_dim, 1);
    std::vector<ggml_fp16_t> k = gen_kv_f16(kv_len, n_kv_heads, head_dim, 2);
    std::vector<ggml_fp16_t> v = gen_kv_f16(kv_len, n_kv_heads, head_dim, 3);

    std::fprintf(stderr, "[EXPECT] dense_htp op_index=5 qo=%d kv=%d heads=%d kv_heads=%d d=%d\n",
                 qo_len, kv_len, n_heads, n_kv_heads, head_dim);
    std::vector<float> dense = runner.run(4, qo_len, kv_len, n_heads, n_kv_heads, head_dim, q, k, v, false, false);
    std::fprintf(stderr, "[EXPECT] sparse_htp_ratio1 op_index=6 qo=%d kv=%d heads=%d kv_heads=%d d=%d\n",
                 qo_len, kv_len, n_heads, n_kv_heads, head_dim);
    std::vector<float> sparse = runner.run(4, qo_len, kv_len, n_heads, n_kv_heads, head_dim, q, k, v, true, false);

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

    run_case(runner, 256, 256, 24, 24, 64);
    run_case(runner, 512, 512, 24, 24, 64);

    ggml_backend_free(htp);
    ggml_backend_free(cpu);

    std::printf("\n=== HTP dense vs sparse full-keep test PASSED ===\n");
    return 0;
}
