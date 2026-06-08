// Test a simple permuted F16 weight x F32 activation -> F32 output matmul on the HTP backend.

#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-htp.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static ggml_context * make_ctx(size_t tensor_count, size_t extra = 16 * 1024) {
    ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead() * tensor_count + ggml_graph_overhead() + extra,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    GGML_ASSERT(ctx != nullptr);
    return ctx;
}

static std::vector<float> gen_matrix(int64_t rows, int64_t cols, int seed) {
    std::vector<float> data(size_t(rows * cols));
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t c = 0; c < cols; ++c) {
            data[r * cols + c] = 0.01f * float(((r * (11 + seed) + c * (7 + seed)) % 401) - 200);
        }
    }
    return data;
}

static std::vector<ggml_fp16_t> fp32_to_fp16(const std::vector<float> & src) {
    std::vector<ggml_fp16_t> dst(src.size());
    ggml_fp32_to_fp16_row(src.data(), dst.data(), src.size());
    return dst;
}

static std::vector<ggml_fp16_t> unpack_permuted_f16_weight(const std::vector<ggml_fp16_t> & packed, int64_t k, int64_t n) {
    GGML_ASSERT(k % 32 == 0);
    GGML_ASSERT(n % 32 == 0);
    GGML_ASSERT(packed.size() == size_t(k * n));

    std::vector<ggml_fp16_t> unpacked(packed.size());
    for (int64_t i = 0; i < k; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            const int64_t i0 = i / 32;
            const int64_t i1 = i % 32;
            const int64_t j0 = j / 32;
            const int64_t j1 = j % 32;

            const int64_t tile_idx = j0 * (k / 32) + i0;
            const ggml_fp16_t * tile = packed.data() + tile_idx * 32 * 32;
            unpacked[j * k + i] = tile[(i1 & ~int64_t{1}) * 32 + j1 * 2 + (i1 & int64_t{1})];
        }
    }
    return unpacked;
}

static void test_pack_f16_weight_roundtrip(int64_t k, int64_t n) {
    std::vector<float> weight_ref      = gen_matrix(n, k, 7);
    std::vector<ggml_fp16_t> weight_nk = fp32_to_fp16(weight_ref);
    std::vector<ggml_fp16_t> packed(weight_nk.size());

    ggml_backend_htp_pack_f16_weight(packed.data(), weight_nk.data(), k, n);
    std::vector<ggml_fp16_t> unpacked = unpack_permuted_f16_weight(packed, k, n);

    if (std::memcmp(weight_nk.data(), unpacked.data(), weight_nk.size() * sizeof(ggml_fp16_t)) != 0) {
        for (size_t i = 0; i < weight_nk.size(); ++i) {
            if (weight_nk[i] != unpacked[i]) {
                std::fprintf(stderr, "pack/unpack mismatch at %zu\n", i);
                break;
            }
        }
        GGML_ASSERT(false);
    }
}

static std::vector<float> run_matmul_backend(ggml_backend_t backend, const std::vector<ggml_fp16_t> & weight_data,
                                             const std::vector<float> & act_mk, int64_t k, int64_t n, int64_t m) {
    ggml_context * ctx = make_ctx(8);
    ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, k, n);
    ggml_tensor * act    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, m);
    ggml_tensor * dst    = ggml_mul_mat(ctx, weight, act);
    GGML_ASSERT(weight != nullptr && act != nullptr && dst != nullptr);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    GGML_ASSERT(graph != nullptr);
    ggml_build_forward_expand(graph, dst);

    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
    GGML_ASSERT(buft != nullptr);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    GGML_ASSERT(buffer != nullptr);

    ggml_backend_tensor_set(weight, weight_data.data(), 0, weight_data.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(act, act_mk.data(), 0, act_mk.size() * sizeof(float));

    const bool htp_backend = ggml_backend_is_htp(backend);
    if (htp_backend) {
        ggml_backend_htp_register_packed_f16_weight(weight, weight);
    }

    const ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (htp_backend) {
        ggml_backend_htp_unregister_packed_f16_weight(weight);
    }
    if (status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "graph compute failed: status=%d\n", int(status));
    }
    GGML_ASSERT(status == GGML_STATUS_SUCCESS);

    std::vector<float> out(size_t(m * n));
    ggml_backend_tensor_get(dst, out.data(), 0, out.size() * sizeof(float));

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return out;
}

static void test_shape(int64_t k, int64_t n, int64_t m) {
    ggml_backend_t htp = ggml_backend_init_by_name("HTP", nullptr);
    GGML_ASSERT(htp != nullptr);

    if (!ggml_backend_htp_runtime_available()) {
        const char * err = ggml_backend_htp_runtime_error();
        std::printf("[SKIP] MATMUL_F16 (%lld,%lld,%lld): runtime unavailable (%s)\n",
                    (long long) k, (long long) n, (long long) m, err ? err : "unknown");
        ggml_backend_free(htp);
        return;
    }

    ggml_backend_dev_t dev = ggml_backend_get_device(htp);
    GGML_ASSERT(dev != nullptr);

    ggml_context * support_ctx = make_ctx(4);
    ggml_tensor * w = ggml_new_tensor_2d(support_ctx, GGML_TYPE_F16, k, n);
    ggml_tensor * a = ggml_new_tensor_2d(support_ctx, GGML_TYPE_F32, k, m);
    ggml_tensor * d = ggml_mul_mat(support_ctx, w, a);
    GGML_ASSERT(d != nullptr);

    // Permuted F16 matmul must be reported as supported for direct HTP dispatch.
    GGML_ASSERT(ggml_backend_dev_supports_op(dev, d));
    ggml_free(support_ctx);

    std::vector<float> weight_ref     = gen_matrix(n, k, 0);
    std::vector<ggml_fp16_t> weight_nk = fp32_to_fp16(weight_ref);
    std::vector<ggml_fp16_t> weight_packed(weight_nk.size());
    ggml_backend_htp_pack_f16_weight(weight_packed.data(), weight_nk.data(), k, n);
    std::vector<float> act_mk          = gen_matrix(m, k, 1);

    ggml_backend_t cpu = ggml_backend_init_by_name("CPU", nullptr);
    GGML_ASSERT(cpu != nullptr);
    auto cpu_out = run_matmul_backend(cpu, weight_nk, act_mk, k, n, m);
    auto htp_out = run_matmul_backend(htp, weight_packed, act_mk, k, n, m);

    double max_diff = 0.0;
    size_t worst_i  = 0;
    for (size_t i = 0; i < cpu_out.size(); ++i) {
        const double diff = std::fabs(double(cpu_out[i]) - double(htp_out[i]));
        if (diff > max_diff) {
            max_diff = diff;
            worst_i  = i;
        }
    }

    if (max_diff > 5e-1) {
        std::fprintf(stderr, "HTP mismatch at %zu: cpu=%.8f htp=%.8f diff=%.2e\n",
                     worst_i, cpu_out[worst_i], htp_out[worst_i], max_diff);
        const size_t n_dump = std::min<size_t>(16, cpu_out.size());
        for (size_t i = 0; i < n_dump; ++i) {
            std::fprintf(stderr, "  out[%zu]: cpu=%.8f htp=%.8f\n", i, cpu_out[i], htp_out[i]);
        }
        GGML_ASSERT(false);
    }

    std::printf("[PASS] MATMUL_F16 (%lld,%lld,%lld) max_abs_diff=%.2e\n",
                (long long) k, (long long) n, (long long) m, max_diff);

    ggml_backend_free(cpu);
    ggml_backend_free(htp);
}

int main() {
    test_pack_f16_weight_roundtrip(64, 96);
    test_pack_f16_weight_roundtrip(1024, 1024);

    if (const char * single = std::getenv("GGML_HTP_MATMUL_F16_CASE")) {
        long long k = 0;
        long long n = 0;
        long long m = 0;
        if (std::sscanf(single, "%llux%llux%llu", &k, &n, &m) != 3 || k <= 0 || n <= 0 || m <= 0) {
            std::fprintf(stderr, "invalid GGML_HTP_MATMUL_F16_CASE=%s, expected <k>x<n>x<m>\n", single);
            return 2;
        }
        test_shape(k, n, m);
        std::printf("\n=== HTP MATMUL_F16 test PASSED ===\n");
        return 0;
    }

    test_shape(64, 96, 5);
    test_shape(1024, 1024, 1);
    test_shape(1536, 2432, 32);

    std::printf("\n=== HTP MATMUL_F16 test PASSED ===\n");
    return 0;
}
