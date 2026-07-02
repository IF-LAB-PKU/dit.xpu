// CPU-vs-HTP hybrid attention precision comparison test.
//
// Both paths run through the same CPU-runtime + HTP-hybrid execution model:
//   - CPU reference: SD_HTP_OPS=1, SD_HTP_COMPUTE=1, SD_HTP_ATTENTION=0
//     Builds decomposed attention, so CPU keeps the attention flow while
//     eligible MUL_MAT nodes may still route to HTP.
//   - HTP dense path: SD_HTP_OPS=1, SD_HTP_COMPUTE=1, SD_HTP_ATTENTION=1
//     Builds FLASH_ATTN_EXT and routes the view-backed dense attention node to
//     HTP dispatch (op_index=5).

#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-htp.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "../../src/ggml_extend.hpp"

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

static int pad64(int n) {
    return (n + 63) & ~63;
}

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

// ---------------------------------------------------------------------------
// deterministic test data  (standard ggml contiguous layout)
// ---------------------------------------------------------------------------

static std::vector<float> gen_q(int qo_len, int n_heads, int head_dim, int seed) {
    std::vector<float> data(size_t(qo_len) * n_heads * head_dim);
    // ggml layout: ne = [n_heads * head_dim, qo_len, 1]
    // Memory order (inner to outer): d, h, q
    for (int q = 0; q < qo_len; ++q) {
        for (int h = 0; h < n_heads; ++h) {
            for (int d = 0; d < head_dim; ++d) {
                float v = 0.01f * float(((h * 37 + q * 17 + d * 11 + seed * 13) % 257) - 128);
                data[(size_t(q) * n_heads * head_dim) + (h * head_dim) + d] = v;
            }
        }
    }
    return data;
}

static std::vector<ggml_fp16_t> gen_kv_f16(int seq_len, int n_heads, int head_dim, int seed) {
    std::vector<ggml_fp16_t> data(size_t(seq_len) * n_heads * head_dim);
    // ggml layout: ne = [n_heads * head_dim, seq_len, 1]
    // Memory order (inner to outer): d, h, s
    for (int s = 0; s < seq_len; ++s) {
        for (int h = 0; h < n_heads; ++h) {
            for (int d = 0; d < head_dim; ++d) {
                float v = 0.015f * float(((h * 41 + s * 19 + d * 7 + seed * 23) % 193) - 96);
                data[(size_t(s) * n_heads * head_dim) + (h * head_dim) + d] = ggml_fp32_to_fp16(v);
            }
        }
    }
    return data;
}

static std::vector<float> gen_mask_f32(int qo_len, int kv_len) {
    std::vector<float> data(size_t(qo_len) * kv_len, 0.0f);
    return data;
}

// ---------------------------------------------------------------------------
// inference-style hybrid attention runner
// ---------------------------------------------------------------------------

struct AttnResult {
    std::vector<float> output;
    int64_t            elapsed_us;
};

struct AttentionRunner : GGMLRunner {
    AttentionRunner(ggml_backend_t runtime_backend, ggml_backend_t htp_backend)
        : GGMLRunner(runtime_backend, runtime_backend) {
        set_htp_backend(htp_backend);
    }

    std::string get_desc() override {
        return "attention-precision";
    }

    AttnResult run(int n_threads,
                   int qo_len, int kv_len,
                   int n_heads, int n_kv_heads, int head_dim,
                   const std::vector<float>      & q_data,
                   const std::vector<ggml_fp16_t> & k_data,
                   const std::vector<ggml_fp16_t> & v_data,
                   const std::vector<float>      & mask_data,
                   bool flash_attn_enabled,
                   bool htp_attention_enabled) {
        set_flash_attention_enabled(flash_attn_enabled);
        set_htp_attention_enabled(htp_attention_enabled);

        sd::Tensor<float>       q_sd({n_heads * head_dim, qo_len, 1}, q_data);
        sd::Tensor<ggml_fp16_t> k_sd({n_kv_heads * head_dim, kv_len, 1}, k_data);
        sd::Tensor<ggml_fp16_t> v_sd({n_kv_heads * head_dim, kv_len, 1}, v_data);
        sd::Tensor<float>       m_sd({kv_len, qo_len, 1, 1}, mask_data);

        auto get_graph = [&]() -> ggml_cgraph * {
            GGMLRunnerContext runner_ctx = get_context();

            ggml_tensor * q = make_input(q_sd);
            ggml_tensor * k = make_input(k_sd);
            ggml_tensor * v = make_input(v_sd);
            ggml_tensor * m = make_input(m_sd);

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
                                                       false);

            ggml_cgraph * graph = ggml_new_graph(runner_ctx.ggml_ctx);
            ggml_build_forward_expand(graph, out);
            return graph;
        };

        int64_t t0 = ggml_time_us();
        auto result = GGMLRunner::compute<float>(get_graph, n_threads, true);
        int64_t t1 = ggml_time_us();

        if (!result.has_value()) {
            return {{}, -1};
        }

        return {std::move(result->values()), t1 - t0};
    }
};

// ---------------------------------------------------------------------------
// comparison & reporting
// ---------------------------------------------------------------------------

struct CmpStats {
    double max_abs_err;
    double mean_abs_err;
    double max_rel_err;
    double mean_rel_err;
    size_t worst_idx;
    double cpu_val_at_worst;
    double htp_val_at_worst;
    double rmse;
    std::vector<int> hist_buckets;
};

static CmpStats compare(const std::vector<float> & cpu, const std::vector<float> & htp) {
    CmpStats s{};
    if (cpu.size() != htp.size()) {
        fprintf(stderr, "size mismatch: cpu=%zu htp=%zu\n", cpu.size(), htp.size());
        return s;
    }

    s.max_abs_err  = 0.0;
    s.mean_abs_err = 0.0;
    s.max_rel_err  = 0.0;
    s.mean_rel_err = 0.0;
    s.rmse         = 0.0;
    s.worst_idx    = 0;

    const double thresholds[] = {1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1,
                                 std::numeric_limits<double>::infinity()};
    s.hist_buckets.assign(sizeof(thresholds) / sizeof(thresholds[0]), 0);

    double sum_abs = 0.0;
    double sum_sq  = 0.0;
    double sum_rel = 0.0;

    for (size_t i = 0; i < cpu.size(); ++i) {
        double a = cpu[i];
        double b = htp[i];
        double abs_err = std::fabs(a - b);
        double rel_err = abs_err / (std::fabs(a) + 1e-8);

        sum_abs += abs_err;
        sum_sq  += abs_err * abs_err;
        sum_rel += rel_err;

        if (abs_err > s.max_abs_err) {
            s.max_abs_err      = abs_err;
            s.worst_idx        = i;
            s.cpu_val_at_worst = a;
            s.htp_val_at_worst = b;
        }
        if (rel_err > s.max_rel_err) {
            s.max_rel_err = rel_err;
        }

        for (size_t bkt = 0; bkt < s.hist_buckets.size(); ++bkt) {
            if (abs_err <= thresholds[bkt]) {
                s.hist_buckets[bkt]++;
                break;
            }
        }
    }

    s.mean_abs_err = sum_abs / (double) cpu.size();
    s.rmse         = std::sqrt(sum_sq / (double) cpu.size());
    s.mean_rel_err = sum_rel / (double) cpu.size();

    return s;
}

static void print_stats(const char * label, const CmpStats & s, size_t total_elems) {
    printf("  %s:\n", label);
    printf("    max_abs_err  = %.6e  (cpu=%.6f  htp=%.6f)\n",
           s.max_abs_err, s.cpu_val_at_worst, s.htp_val_at_worst);
    printf("    mean_abs_err = %.6e\n", s.mean_abs_err);
    printf("    rmse         = %.6e\n", s.rmse);
    printf("    max_rel_err  = %.6e\n", s.max_rel_err);
    printf("    mean_rel_err = %.6e\n", s.mean_rel_err);
    printf("    abs_err histogram (of %zu elements):\n", total_elems);
    const char * bucket_labels[] = {"<1e-7", "<1e-6", "<1e-5", "<1e-4", "<1e-3", "<1e-2", "<1e-1", ">=1e-1"};
    int cumulative = 0;
    for (size_t i = 0; i < s.hist_buckets.size(); ++i) {
        cumulative += s.hist_buckets[i];
        printf("      %-8s  %8d  (%5.1f%% cumulative)\n",
               bucket_labels[i], s.hist_buckets[i],
               100.0 * (double) cumulative / (double) total_elems);
    }
}

// ---------------------------------------------------------------------------
// test cases
// ---------------------------------------------------------------------------

struct CaseParams {
    int qo_len;
    int kv_len;
    int n_heads;
    int n_kv_heads;
    int head_dim;
    const char * desc;
};

static bool run_case(ggml_backend_t cpu_backend, ggml_backend_t htp_backend, const CaseParams & p) {
    printf("\n--- %s (qo=%d kv=%d heads=%d kv_heads=%d d=%d) ---\n",
           p.desc, p.qo_len, p.kv_len, p.n_heads, p.n_kv_heads, p.head_dim);

    // Generate data once in standard ggml layout
    std::vector<float>      q    = gen_q(p.qo_len, p.n_heads, p.head_dim, 1);
    std::vector<ggml_fp16_t> k   = gen_kv_f16(p.kv_len, p.n_kv_heads, p.head_dim, 2);
    std::vector<ggml_fp16_t> v   = gen_kv_f16(p.kv_len, p.n_kv_heads, p.head_dim, 3);
    std::vector<float>       mask = gen_mask_f32(p.qo_len, p.kv_len);

    AttentionRunner runner(cpu_backend, htp_backend);

    printf("  Running CPU reference (SD_HTP_ATTENTION=0)...\n");
    EnvGuard env_ops("SD_HTP_OPS", "1");
    EnvGuard env_compute("SD_HTP_COMPUTE", "1");
    EnvGuard env_attn0("SD_HTP_ATTENTION", "0");
    AttnResult cpu_result = runner.run(1,
                                       p.qo_len,
                                       p.kv_len,
                                       p.n_heads,
                                       p.n_kv_heads,
                                       p.head_dim,
                                       q,
                                       k,
                                       v,
                                       mask,
                                       false,
                                       false);
    if (cpu_result.elapsed_us < 0) {
        fprintf(stderr, "FAIL: CPU reference run failed\n");
        return false;
    }
    printf("  CPU reference elapsed: %.3f ms, output elements: %zu\n",
           cpu_result.elapsed_us / 1000.0, cpu_result.output.size());

    printf("  Running HTP dense (SD_HTP_ATTENTION=1)...\n");
    EnvGuard env_attn1("SD_HTP_ATTENTION", "1");
    AttnResult htp_result = runner.run(1,
                                       p.qo_len,
                                       p.kv_len,
                                       p.n_heads,
                                       p.n_kv_heads,
                                       p.head_dim,
                                       q,
                                       k,
                                       v,
                                       mask,
                                       true,
                                       true);
    if (htp_result.elapsed_us < 0) {
        fprintf(stderr, "FAIL: HTP run failed\n");
        return false;
    }
    printf("  HTP dense elapsed: %.3f ms, output elements: %zu\n",
           htp_result.elapsed_us / 1000.0, htp_result.output.size());

    // Compare
    CmpStats stats = compare(cpu_result.output, htp_result.output);
    print_stats("CPU reference vs HTP dense attention", stats, cpu_result.output.size());

    if (stats.max_abs_err > 1.0) {
        printf("  WARNING: max_abs_err > 1.0 — significant deviation\n");
    }
    if (stats.rmse > 0.1) {
        printf("  WARNING: rmse > 0.1 — widespread error\n");
    }

    return stats.max_abs_err <= 1e-2 && stats.rmse <= 1e-2;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
    printf("=== HTP Attention Precision Test (inference-style attention routing) ===\n\n");
    ggml_backend_t cpu_backend = ggml_backend_init_by_name("CPU", nullptr);
    if (!cpu_backend) {
        fprintf(stderr, "FAIL: CPU backend init failed\n");
        return 1;
    }

    ggml_backend_t htp_backend = nullptr;
    if (ggml_backend_htp_runtime_available()) {
        ggml_backend_htp_set_hybrid_enabled(true);
        htp_backend = ggml_backend_init_by_name("HTP", nullptr);
        if (!htp_backend) {
            fprintf(stderr, "FAIL: HTP backend init failed\n");
            ggml_backend_free(cpu_backend);
            return 1;
        }
    } else {
        const char * err = ggml_backend_htp_runtime_error();
        printf("SKIP: HTP runtime unavailable (%s)\n", err ? err : "unknown");
        ggml_backend_free(cpu_backend);
        return 0;
    }

    CaseParams cases[] = {
        { 256,  256,  4,  2,  64, "tiny GQA"},
        { 256,  256,  8,  4,  64, "small GQA"},
        { 256,  256, 24, 24,  64, "SD3.5 self-attn (qo=kv=256)"},
        { 410,  512, 24, 24,  64, "SD3.5 cross-attn (qo=410 kv=512)"},
        { 256,  512,  8,  2,  64, "GQA (n_heads=8 kv_heads=2)"},
    };

    int passed  = 0;
    int total   = 0;

    for (const auto & c : cases) {
        total++;
        bool ok = run_case(cpu_backend, htp_backend, c);
        if (ok) {
            passed++;
            printf("  [PASS] %s\n", c.desc);
        } else {
            printf("  [FAIL] %s\n", c.desc);
        }
    }

    printf("\n=== Results: %d/%d passed ===\n", passed, total);

    ggml_backend_free(htp_backend);
    ggml_backend_free(cpu_backend);

    return (passed == total) ? 0 : 1;
}
