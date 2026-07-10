#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

static size_t idx4(int64_t i0, int64_t i1, int64_t i2, int64_t i3,
                   int64_t ne0, int64_t ne1, int64_t ne2) {
    return size_t(i0 + ne0*(i1 + ne1*(i2 + ne2*i3)));
}

static void require_close(float actual, float expected, float tol, const char * label) {
    const float diff = std::fabs(actual - expected);
    if (!(diff <= tol)) {
        std::fprintf(stderr, "%s mismatch: actual=%g expected=%g diff=%g tol=%g\n",
                     label, actual, expected, diff, tol);
        std::abort();
    }
}

static float reference_lse(const std::vector<float> & q,
                           const std::vector<float> & k,
                           const std::vector<ggml_fp16_t> & mask,
                           int64_t D,
                           int64_t Lq,
                           int64_t Lk,
                           int64_t H,
                           int64_t B,
                           float scale,
                           int64_t qt,
                           int64_t h,
                           int64_t b) {
    float max_score = -std::numeric_limits<float>::infinity();

    for (int64_t kt = 0; kt < Lk; ++kt) {
        float score = 0.0f;
        for (int64_t d = 0; d < D; ++d) {
            score += q[idx4(d, qt, h, b, D, Lq, H)] * k[idx4(d, kt, h, b, D, Lk, H)];
        }
        score *= scale;
        score += ggml_fp16_to_fp32(mask[idx4(kt, qt, 0, b, Lk, Lq, 1)]);
        max_score = std::fmax(max_score, score);
    }

    float exp_sum = 0.0f;
    for (int64_t kt = 0; kt < Lk; ++kt) {
        float score = 0.0f;
        for (int64_t d = 0; d < D; ++d) {
            score += q[idx4(d, qt, h, b, D, Lq, H)] * k[idx4(d, kt, h, b, D, Lk, H)];
        }
        score *= scale;
        score += ggml_fp16_to_fp32(mask[idx4(kt, qt, 0, b, Lk, Lq, 1)]);
        exp_sum += std::exp(score - max_score);
    }

    return max_score + std::log(exp_sum);
}

static void run_case(int64_t Lq, int64_t Lk, int n_threads) {
    const int64_t D  = 5;
    const int64_t DV = 3;
    const int64_t H  = 2;
    const int64_t B  = 1;
    const float scale = 0.37f;

    std::vector<float> q(size_t(D*Lq*H*B));
    std::vector<float> k(size_t(D*Lk*H*B));
    std::vector<float> v(size_t(DV*Lk*H*B));
    std::vector<ggml_fp16_t> mask(size_t(Lk*Lq*B));

    for (size_t i = 0; i < q.size(); ++i) {
        q[i] = 0.07f * float(int(i % 17) - 8);
    }
    for (size_t i = 0; i < k.size(); ++i) {
        k[i] = 0.05f * float(int((i*3) % 19) - 9);
    }
    for (size_t i = 0; i < v.size(); ++i) {
        v[i] = 0.03f * float(int((i*5) % 23) - 11);
    }
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t qt = 0; qt < Lq; ++qt) {
            for (int64_t kt = 0; kt < Lk; ++kt) {
                const bool masked = (kt + 2*qt) % 7 == 3;
                const float value = masked ? -std::numeric_limits<float>::infinity()
                                           : 0.02f * float((kt % 5) - 2);
                mask[idx4(kt, qt, 0, b, Lk, Lq, 1)] = ggml_fp32_to_fp16(value);
            }
        }
    }

    ggml_backend_t backend = ggml_backend_cpu_init();
    GGML_ASSERT(backend != nullptr);
    ggml_backend_cpu_set_n_threads(backend, n_threads);

    struct ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead()*16 + ggml_graph_overhead() + 1024*1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);

    ggml_tensor * tq = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D,  Lq, H, B);
    ggml_tensor * tk = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D,  Lk, H, B);
    ggml_tensor * tv = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, DV, Lk, H, B);
    ggml_tensor * tm = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, Lk, Lq, 1, B);

    ggml_tensor * out = ggml_flash_attn_ext_with_lse(ctx, tq, tk, tv, tm, scale, 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);

    GGML_ASSERT(out->ne[0] == DV + 1);
    GGML_ASSERT(out->ne[1] == H);
    GGML_ASSERT(out->ne[2] == Lq);
    GGML_ASSERT(out->ne[3] == B);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    GGML_ASSERT(ggml_gallocr_alloc_graph(galloc, graph));

    ggml_backend_tensor_set(tq, q.data(), 0, q.size()*sizeof(float));
    ggml_backend_tensor_set(tk, k.data(), 0, k.size()*sizeof(float));
    ggml_backend_tensor_set(tv, v.data(), 0, v.size()*sizeof(float));
    ggml_backend_tensor_set(tm, mask.data(), 0, mask.size()*sizeof(ggml_fp16_t));

    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> got(size_t(ggml_nelements(out)));
    ggml_backend_tensor_get(out, got.data(), 0, got.size()*sizeof(float));

    for (int64_t b = 0; b < B; ++b) {
        for (int64_t qt = 0; qt < Lq; ++qt) {
            for (int64_t h = 0; h < H; ++h) {
                const float expected = reference_lse(q, k, mask, D, Lq, Lk, H, B, scale, qt, h, b);
                const float actual = got[idx4(DV, h, qt, b, DV + 1, H, Lq)];
                require_close(actual, expected, 2e-5f, "flash-attn LSE");
            }
        }
    }

    ggml_gallocr_free(galloc);
    ggml_free(ctx);
    ggml_backend_free(backend);
}

int main() {
    run_case(/*Lq=*/3, /*Lk=*/11,  /*n_threads=*/1);
    run_case(/*Lq=*/1, /*Lk=*/640, /*n_threads=*/4);
    return 0;
}
