// 测试 HTP 后端 RMS_NORM 算子的计算正确性
// ===========================================================================
//
// 验证流程：
//   1. 生成伪随机 F32 输入数据 (src_host)
//   2. CPU 参考: rms_norm_ref() 按行计算 y = x / sqrt(mean(x^2) + eps)
//   3. HTP 执行: 构建单节点 ggml_rms_norm 图 → RPCMEM 缓冲区分配 →
//               ggml_backend_tensor_set 写入 → ggml_backend_graph_compute 计算 →
//               ggml_backend_tensor_get 读回
//   4. 逐元素比较: |ref[i] - htp[i]| <= 1e-3
//

#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-htp.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// ── CPU reference ──────────────────────────────────────────────────────────

static void rms_norm_ref(float * dst, const float * src, int ne0, int ne1, float eps) {
    for (int row = 0; row < ne1; ++row) {
        const float * x = src + row * ne0;
        float * y       = dst + row * ne0;

        float sum = 0.0f;
        for (int col = 0; col < ne0; ++col) {
            sum += x[col] * x[col];
        }
        const float mean  = sum / float(ne0);
        const float scale = 1.0f / std::sqrt(mean + eps);
        for (int col = 0; col < ne0; ++col) {
            y[col] = x[col] * scale;
        }
    }
}

// ── helper: create ggml context ────────────────────────────────────────────

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

// ── run RMS_NORM graph and compare with reference ──────────────────────────

static void test_rms_norm_shape(int ne0, int ne1, float eps) {
    ggml_backend_t backend = ggml_backend_init_by_name("HTP", nullptr);
    GGML_ASSERT(backend != nullptr);
    GGML_ASSERT(ggml_backend_is_htp(backend));

    if (!ggml_backend_htp_runtime_available()) {
        const char * err = ggml_backend_htp_runtime_error();
        std::printf("[SKIP] RMS_NORM (%d,%d): runtime unavailable (%s)\n",
                    ne0, ne1, err ? err : "unknown");
        ggml_backend_free(backend);
        return;
    }

    const size_t elem_count = size_t(ne0) * ne1;
    const size_t byte_count = elem_count * sizeof(float);

    ggml_context * ctx = make_ctx(8);
    ggml_tensor * src = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, ne1);
    GGML_ASSERT(src != nullptr);

    ggml_tensor * dst = ggml_rms_norm(ctx, src, eps);
    GGML_ASSERT(dst != nullptr);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    GGML_ASSERT(graph != nullptr);
    ggml_build_forward_expand(graph, dst);

    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
    GGML_ASSERT(buft != nullptr);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    GGML_ASSERT(buffer != nullptr);

    // init input data
    std::vector<float> src_host(elem_count);
    std::vector<float> ref_host(elem_count);
    std::vector<float> dst_host(elem_count, 0.0f);

    for (size_t i = 0; i < elem_count; ++i) {
        src_host[i] = ((int(i) * 37) % 2000) * 0.02f - 20.0f;
    }
    ggml_backend_tensor_set(src, src_host.data(), 0, byte_count);

    // compute on HTP
    ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "RMS_NORM(%d,%d) HTP graph compute failed: status=%d\n",
                     ne0, ne1, int(status));
    }
    GGML_ASSERT(status == GGML_STATUS_SUCCESS);

    // readback & compare
    ggml_backend_tensor_get(dst, dst_host.data(), 0, byte_count);
    rms_norm_ref(ref_host.data(), src_host.data(), ne0, ne1, eps);

    double max_diff = 0.0;
    for (size_t i = 0; i < elem_count; ++i) {
        double diff = std::fabs(double(ref_host[i]) - double(dst_host[i]));
        if (diff > max_diff) max_diff = diff;
        if (diff > 1e-3) {
            std::fprintf(stderr, "RMS_NORM(%d,%d) mismatch at %zu: ref=%.6f htp=%.6f diff=%.6e\n",
                         ne0, ne1, i, ref_host[i], dst_host[i], diff);
            GGML_ASSERT(false);
        }
    }

    std::printf("[PASS] RMS_NORM (%d,%d) max_diff=%.2e\n", ne0, ne1, max_diff);

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
}

// ───────────────────────────────────────────────────────────────────────────

int main() {
    const float eps = 1e-5f;

    if (const char * single = std::getenv("GGML_HTP_RMS_CASE")) {
        int ne0 = 0;
        int ne1 = 0;
        if (std::sscanf(single, "%dx%d", &ne0, &ne1) != 2 || ne0 <= 0 || ne1 <= 0) {
            std::fprintf(stderr, "invalid GGML_HTP_RMS_CASE=%s, expected <ne0>x<ne1>\n", single);
            return 2;
        }

        test_rms_norm_shape(ne0, ne1, eps);
        std::printf("\n=== HTP RMS_NORM test PASSED ===\n");
        return 0;
    }

    test_rms_norm_shape(64,  1,  eps);
    test_rms_norm_shape(256, 1,  eps);
    test_rms_norm_shape(128, 4,  eps);
    test_rms_norm_shape(256, 3,  eps);
    test_rms_norm_shape(512, 2,  eps);
    test_rms_norm_shape(1024, 1, eps);
    test_rms_norm_shape(768, 8,  eps);

    std::printf("\n=== HTP RMS_NORM test PASSED ===\n");
    return 0;
}
