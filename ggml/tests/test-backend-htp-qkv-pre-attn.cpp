// Benchmark the real SD3.5 MMDiT pre-attention graph on the HTP backend.

#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-htp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <vector>

namespace {

constexpr int64_t kHiddenSize = 1536;
constexpr int64_t kHeads = 24;
constexpr int64_t kHeadDim = 64;
constexpr int64_t kQkvSize = 3 * kHiddenSize;

int env_positive_int(const char * name, int fallback) {
    const char * value = std::getenv(name);
    if (value == nullptr || *value == '\0') {
        return fallback;
    }
    char * end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    return end != value && *end == '\0' && parsed > 0 && parsed <= INT32_MAX ? int(parsed) : fallback;
}

int64_t now_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

float activation_value(int64_t row, int64_t col) {
    return float(((row * 11 + col * 5) % 37) - 18) * 0.01f;
}

float weight_value(int64_t row, int64_t col) {
    return float(((row * 13 + col * 7) % 101) - 50) * 0.001f;
}

struct PreAttentionGraph {
    ggml_context * ctx = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * weight = nullptr;
    ggml_tensor * activation = nullptr;
    ggml_tensor * bias = nullptr;
    ggml_tensor * q_gamma = nullptr;
    ggml_tensor * k_gamma = nullptr;
    ggml_tensor * q = nullptr;
    ggml_tensor * k = nullptr;
    ggml_tensor * v = nullptr;
    ggml_tensor * mask = nullptr;
    ggml_tensor * output = nullptr;
    ggml_tensor * packed_storage = nullptr;
};

static void packed_qkv_noop(struct ggml_tensor *, int, int, void *) {}

PreAttentionGraph build_pre_attention_graph(int tokens, int image_tokens,
                                             int image_block_rows, int image_block_cols,
                                             bool packed_graph) {
    const ggml_init_params params = {
        /* .mem_size   = */ 4 * 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    PreAttentionGraph result;
    result.ctx = ggml_init(params);
    GGML_ASSERT(result.ctx != nullptr);

    result.weight = ggml_new_tensor_2d(result.ctx, GGML_TYPE_F16, kHiddenSize, kQkvSize);
    result.activation = ggml_new_tensor_2d(result.ctx, GGML_TYPE_F32, kHiddenSize, tokens);
    result.bias = ggml_new_tensor_1d(result.ctx, GGML_TYPE_F32, kQkvSize);
    result.q_gamma = ggml_new_tensor_1d(result.ctx, GGML_TYPE_F32, kHeadDim);
    result.k_gamma = ggml_new_tensor_1d(result.ctx, GGML_TYPE_F32, kHeadDim);

    if (packed_graph) {
        const int token_tiles = (tokens + 31) / 32;
        const int64_t qkv_elems = 3 * kHeads * token_tiles * 2 * 1024;
        ggml_tensor * args[] = { result.activation, result.weight, result.bias,
                                 result.q_gamma, result.k_gamma };
        ggml_tensor * qkv = ggml_custom_4d(
            result.ctx, GGML_TYPE_F16, qkv_elems, 1, 1, 1,
            args, 5, packed_qkv_noop, 1, nullptr);
        ggml_set_name(qkv, "qkv_mul_rms_packed");
        result.packed_storage = qkv;
        const size_t qkv_bytes = (size_t) kHeads * token_tiles * 2 * 1024 * sizeof(ggml_fp16_t);
        result.q = ggml_view_4d(result.ctx, qkv, kHeadDim, tokens, kHeads, 1,
                                sizeof(ggml_fp16_t), kHeadDim * sizeof(ggml_fp16_t),
                                (size_t) token_tiles * 2 * 1024 * sizeof(ggml_fp16_t), 0);
        result.k = ggml_view_4d(result.ctx, qkv, kHeadDim, tokens, kHeads, 1,
                                sizeof(ggml_fp16_t), kHeadDim * sizeof(ggml_fp16_t),
                                (size_t) token_tiles * 2 * 1024 * sizeof(ggml_fp16_t), qkv_bytes);
        result.v = ggml_view_4d(result.ctx, qkv, kHeadDim, tokens, kHeads, 1,
                                sizeof(ggml_fp16_t), kHeadDim * sizeof(ggml_fp16_t),
                                (size_t) token_tiles * 2 * 1024 * sizeof(ggml_fp16_t), 2 * qkv_bytes);
        result.q->op = GGML_OP_NONE;
        result.k->op = GGML_OP_NONE;
        result.v->op = GGML_OP_NONE;
        result.q->type = GGML_TYPE_F16;
        result.k->type = GGML_TYPE_F16;
        result.v->type = GGML_TYPE_F16;
        result.mask = ggml_new_tensor_4d(result.ctx, GGML_TYPE_I32,
                                         (tokens + 31) / 32, kHeads * ((tokens + 31) / 32), 1, 1);
        ggml_set_name(result.mask, "block_sparse_index");
        result.output = ggml_flash_attn_ext_with_lse(
            result.ctx, result.q, result.k, result.v, result.mask,
            1.0f / std::sqrt(float(kHeadDim)), 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(result.output, GGML_PREC_F32);
        ggml_flash_attn_ext_set_block_sparse(
            result.output, 32, 32, GGML_FLASH_ATTN_EXT_MASK_BLOCK_INDEX);
        ggml_flash_attn_ext_set_static_halo_2d(
            result.output, image_tokens, tokens - image_tokens,
            image_block_rows, image_block_cols);
        ggml_set_name(result.output, "packed_sparse_attention");
        result.graph = ggml_new_graph_custom(result.ctx, 64, false);
        GGML_ASSERT(result.graph != nullptr);
        ggml_build_forward_expand(result.graph, result.output);
        return result;
    }

    // Linear(dim, 3 * dim, bias=true).
    ggml_tensor * qkv = ggml_mul_mat(result.ctx, result.weight, result.activation);
    qkv = ggml_add_inplace(result.ctx, qkv, result.bias);

    // split_qkv(): reshape + permute + cont, followed by three views.
    qkv = ggml_reshape_4d(result.ctx, qkv, kHiddenSize, 3, tokens, 1);
    qkv = ggml_cont(result.ctx, ggml_permute(result.ctx, qkv, 0, 3, 1, 2));
    const int64_t branch_offset = qkv->nb[2] * qkv->ne[2];
    ggml_tensor * q = ggml_view_3d(result.ctx, qkv, qkv->ne[0], qkv->ne[1], qkv->ne[2],
                                   qkv->nb[1], qkv->nb[2], branch_offset * 0);
    ggml_tensor * k = ggml_view_3d(result.ctx, qkv, qkv->ne[0], qkv->ne[1], qkv->ne[2],
                                   qkv->nb[1], qkv->nb[2], branch_offset * 1);
    result.v = ggml_view_3d(result.ctx, qkv, qkv->ne[0], qkv->ne[1], qkv->ne[2],
                            qkv->nb[1], qkv->nb[2], branch_offset * 2);

    q = ggml_reshape_4d(result.ctx, q, kHeadDim, kHeads, tokens, 1);
    k = ggml_reshape_4d(result.ctx, k, kHeadDim, kHeads, tokens, 1);
    q = ggml_mul_inplace(result.ctx, ggml_rms_norm(result.ctx, q, 1.0e-6f), result.q_gamma);
    k = ggml_mul_inplace(result.ctx, ggml_rms_norm(result.ctx, k, 1.0e-6f), result.k_gamma);
    result.q = ggml_reshape_3d(result.ctx, q, kHiddenSize, tokens, 1);
    result.k = ggml_reshape_3d(result.ctx, k, kHiddenSize, tokens, 1);

    const int blocks = (tokens + 31) / 32;
    result.mask = ggml_new_tensor_4d(result.ctx, GGML_TYPE_I32,
                                     blocks, kHeads * blocks, 1, 1);
    ggml_set_name(result.mask, "block_sparse_index");
    ggml_tensor * q_attn = ggml_permute(
        result.ctx, ggml_reshape_4d(result.ctx, result.q, kHeadDim, kHeads, tokens, 1),
        0, 2, 1, 3);
    ggml_tensor * k_f16 = ggml_cast(result.ctx, result.k, GGML_TYPE_F16);
    ggml_tensor * v_f16 = ggml_cast(result.ctx, result.v, GGML_TYPE_F16);
    ggml_tensor * k_attn = ggml_permute(
        result.ctx, ggml_reshape_4d(result.ctx, k_f16, kHeadDim, kHeads, tokens, 1),
        0, 2, 1, 3);
    ggml_tensor * v_attn = ggml_permute(
        result.ctx, ggml_reshape_4d(result.ctx, v_f16, kHeadDim, kHeads, tokens, 1),
        0, 2, 1, 3);
    result.output = ggml_flash_attn_ext_with_lse(
        result.ctx, q_attn, k_attn, v_attn, result.mask,
        1.0f / std::sqrt(float(kHeadDim)), 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(result.output, GGML_PREC_F32);
    ggml_flash_attn_ext_set_block_sparse(
        result.output, 32, 32, GGML_FLASH_ATTN_EXT_MASK_BLOCK_INDEX);
    ggml_flash_attn_ext_set_static_halo_2d(
        result.output, image_tokens, tokens - image_tokens,
        image_block_rows, image_block_cols);

    result.graph = ggml_new_graph_custom(result.ctx, 64, false);
    GGML_ASSERT(result.graph != nullptr);
    ggml_build_forward_expand(result.graph, result.output);
    return result;
}

std::vector<int32_t> make_static_halo_index(int tokens, int image_block_rows,
                                            int image_block_cols) {
    const int blocks = (tokens + 31) / 32;
    const int image_blocks = image_block_rows * image_block_cols;
    std::vector<int32_t> index(size_t(kHeads) * blocks * blocks, -1);
    for (int h = 0; h < kHeads; ++h) {
        for (int qb = 0; qb < blocks; ++qb) {
            int32_t * row = index.data() + (size_t(h) * blocks + qb) * blocks;
            int count = 0;
            if (qb >= image_blocks) {
                for (int kb = 0; kb < blocks; ++kb) {
                    row[count++] = h * blocks + kb;
                }
                continue;
            }
            const int qy = qb / image_block_cols;
            const int qx = qb % image_block_cols;
            for (int ky = std::max(0, qy - 1);
                 ky <= std::min(image_block_rows - 1, qy + 1); ++ky) {
                for (int kx = std::max(0, qx - 1);
                     kx <= std::min(image_block_cols - 1, qx + 1); ++kx) {
                    row[count++] = h * blocks + ky * image_block_cols + kx;
                }
            }
            for (int kb = image_blocks; kb < blocks; ++kb) {
                row[count++] = h * blocks + kb;
            }
        }
    }
    return index;
}

void run_case(int resolution, int warmups, int runs) {
    GGML_ASSERT(resolution >= 256 && resolution % 16 == 0);
    const int token_h = resolution / 16;
    const int token_w = resolution / 16;
    const int image_tokens = token_h * token_w;
    const bool self_graph = std::getenv("GGML_HTP_QKV_SELF_GRAPH") != nullptr &&
                            std::strcmp(std::getenv("GGML_HTP_QKV_SELF_GRAPH"), "0") != 0;
    const int tokens = self_graph ? image_tokens : image_tokens + 154;
    const int image_block_rows = token_h / 4;
    const int image_block_cols = token_w / 8;
    ggml_backend_htp_set_hybrid_enabled(true);
    ggml_backend_t htp = ggml_backend_init_by_name("HTP", nullptr);
    GGML_ASSERT(htp != nullptr);
    if (!ggml_backend_htp_runtime_available()) {
        const char * error = ggml_backend_htp_runtime_error();
        std::printf("[SKIP] SD3.5 QKV pre-attention: runtime unavailable (%s)\n",
                    error != nullptr ? error : "unknown");
        ggml_backend_free(htp);
        ggml_backend_htp_set_hybrid_enabled(false);
        return;
    }
    ggml_backend_htp_set_n_threads(htp, env_positive_int("GGML_HTP_QKV_PRE_ATTN_THREADS", 8));

    const bool packed_graph = std::getenv("GGML_HTP_QKV_PACKED_GRAPH") != nullptr &&
                              std::strcmp(std::getenv("GGML_HTP_QKV_PACKED_GRAPH"), "0") != 0;
    PreAttentionGraph f = build_pre_attention_graph(
        tokens, image_tokens, image_block_rows, image_block_cols, packed_graph);
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(htp);
    GGML_ASSERT(buft != nullptr);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors_from_buft(f.ctx, buft);
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> activation(size_t(tokens) * kHiddenSize);
    for (int64_t token = 0; token < tokens; ++token) {
        for (int64_t col = 0; col < kHiddenSize; ++col) {
            activation[size_t(token) * kHiddenSize + col] = activation_value(token, col);
        }
    }
    std::vector<ggml_fp16_t> weight(size_t(kHiddenSize) * kQkvSize);
    std::vector<ggml_fp16_t> packed_weight(weight.size());
    for (int64_t out = 0; out < kQkvSize; ++out) {
        for (int64_t in = 0; in < kHiddenSize; ++in) {
            weight[size_t(out) * kHiddenSize + in] = ggml_fp32_to_fp16(weight_value(in, out));
        }
    }
    ggml_backend_htp_pack_f16_weight(packed_weight.data(), weight.data(), kHiddenSize, kQkvSize);

    std::vector<float> bias(kQkvSize);
    for (int64_t i = 0; i < kQkvSize; ++i) {
        bias[i] = float((i % 17) - 8) * 0.01f;
    }
    std::vector<float> q_gamma(kHeadDim);
    std::vector<float> k_gamma(kHeadDim);
    for (int64_t i = 0; i < kHeadDim; ++i) {
        q_gamma[i] = 1.0f + float(i % 7) * 0.01f;
        k_gamma[i] = 1.0f - float(i % 5) * 0.01f;
    }

    ggml_backend_tensor_set(f.activation, activation.data(), 0, activation.size() * sizeof(float));
    ggml_backend_tensor_set(f.weight, packed_weight.data(), 0, packed_weight.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(f.bias, bias.data(), 0, bias.size() * sizeof(float));
    ggml_backend_tensor_set(f.q_gamma, q_gamma.data(), 0, q_gamma.size() * sizeof(float));
    ggml_backend_tensor_set(f.k_gamma, k_gamma.data(), 0, k_gamma.size() * sizeof(float));
    const std::vector<int32_t> mask = make_static_halo_index(
        tokens, image_block_rows, image_block_cols);
    ggml_backend_tensor_set(f.mask, mask.data(), 0, mask.size() * sizeof(int32_t));
    ggml_backend_htp_register_packed_f16_weight(f.weight, f.weight);

    for (int i = 0; i < warmups; ++i) {
        GGML_ASSERT(ggml_backend_graph_compute(htp, f.graph) == GGML_STATUS_SUCCESS);
    }
    std::vector<int64_t> elapsed;
    elapsed.reserve(runs);
    for (int i = 0; i < runs; ++i) {
        const int64_t start = now_us();
        const ggml_status status = ggml_backend_graph_compute(htp, f.graph);
        const int64_t end = now_us();
        GGML_ASSERT(status == GGML_STATUS_SUCCESS);
        elapsed.push_back(end - start);
    }

    float o0 = 0.0f;
    float lse0 = 0.0f;
    float q_samples[3] = {};
    float k_samples[3] = {};
    float v_samples[3] = {};
    ggml_backend_tensor_get(f.output, &o0, 0, sizeof(o0));
    ggml_backend_tensor_get(f.output, &lse0, kHeadDim * sizeof(float), sizeof(lse0));
    if (!packed_graph) {
        const size_t sample_offsets[] = { 0, size_t(kHeadDim), size_t(kHiddenSize) };
        for (int i = 0; i < 3; ++i) {
            ggml_backend_tensor_get(f.q, &q_samples[i], sample_offsets[i] * sizeof(float), sizeof(float));
            ggml_backend_tensor_get(f.k, &k_samples[i], sample_offsets[i] * sizeof(float), sizeof(float));
            ggml_backend_tensor_get(f.v, &v_samples[i], sample_offsets[i] * sizeof(float), sizeof(float));
        }
    }
    double q_sum = 0.0;
    double k_sum = 0.0;
    double v_sum = 0.0;
    double q_sumsq = 0.0;
    double k_sumsq = 0.0;
    double v_sumsq = 0.0;
    float ref_o0 = 0.0f;
    float ref_lse0 = 0.0f;
    if (!packed_graph && resolution == 256) {
        std::vector<float> q_all(size_t(tokens) * kHiddenSize);
        std::vector<float> k_all(size_t(tokens) * kHiddenSize);
        std::vector<float> v_all(size_t(tokens) * kHiddenSize);
        ggml_backend_tensor_get(f.q, q_all.data(), 0, q_all.size() * sizeof(float));
        ggml_backend_tensor_get(f.k, k_all.data(), 0, k_all.size() * sizeof(float));
        ggml_backend_tensor_get(f.v, v_all.data(), 0, v_all.size() * sizeof(float));
        for (size_t i = 0; i < q_all.size(); ++i) {
            q_sum += q_all[i]; k_sum += k_all[i]; v_sum += v_all[i];
            q_sumsq += q_all[i] * q_all[i];
            k_sumsq += k_all[i] * k_all[i];
            v_sumsq += v_all[i] * v_all[i];
        }
        const int blocks = (tokens + 31) / 32;
        const int32_t * row = mask.data();
        float max_score = -std::numeric_limits<float>::infinity();
        for (int pos = 0; pos < blocks && row[pos] >= 0; ++pos) {
            const int kb = row[pos] % blocks;
            for (int kt = kb * 32; kt < std::min(tokens, kb * 32 + 32); ++kt) {
                float score = 0.0f;
                for (int d = 0; d < kHeadDim; ++d) {
                    const float kf16 = ggml_fp16_to_fp32(ggml_fp32_to_fp16(
                        k_all[size_t(kt) * kHiddenSize + d]));
                    score += q_all[d] * kf16;
                }
                score /= std::sqrt(float(kHeadDim));
                max_score = std::max(max_score, score);
            }
        }
        float denom = 0.0f;
        float numerator = 0.0f;
        for (int pos = 0; pos < blocks && row[pos] >= 0; ++pos) {
            const int kb = row[pos] % blocks;
            for (int kt = kb * 32; kt < std::min(tokens, kb * 32 + 32); ++kt) {
                float score = 0.0f;
                for (int d = 0; d < kHeadDim; ++d) {
                    const float kf16 = ggml_fp16_to_fp32(ggml_fp32_to_fp16(
                        k_all[size_t(kt) * kHiddenSize + d]));
                    score += q_all[d] * kf16;
                }
                const float p = std::exp(score / std::sqrt(float(kHeadDim)) - max_score);
                denom += p;
                numerator += p * ggml_fp16_to_fp32(ggml_fp32_to_fp16(
                    v_all[size_t(kt) * kHiddenSize]));
            }
        }
        ref_o0 = numerator / denom;
        ref_lse0 = max_score + std::log(denom);
    }
    GGML_ASSERT(std::isfinite(o0) && std::isfinite(lse0));

    std::vector<int64_t> sorted = elapsed;
    std::sort(sorted.begin(), sorted.end());
    const double median = sorted.size() % 2 != 0
        ? double(sorted[sorted.size() / 2])
        : 0.5 * double(sorted[sorted.size() / 2 - 1] + sorted[sorted.size() / 2]);
    const double average = double(std::accumulate(elapsed.begin(), elapsed.end(), int64_t{0})) / elapsed.size();
    std::printf("sd35_qkv_sparse_e2e: graph=%s path=%s resolution=%d tokens=%d warmups=%d runs=%d "
                "avg_us=%.1f median_us=%.1f min_us=%lld samples_us=",
                self_graph ? "self" : "joint", packed_graph ? "packed" : "legacy",
                resolution, tokens, warmups, runs,
                average, median, (long long) sorted.front());
    for (size_t i = 0; i < elapsed.size(); ++i) {
        std::printf("%s%lld", i == 0 ? "" : ",", (long long) elapsed[i]);
    }
    std::printf(" o0=%g lse0=%g q0=%g k0=%g v0=%g q_h1=%g k_h1=%g v_h1=%g "
                "q_t1=%g k_t1=%g v_t1=%g q_sum=%.9g k_sum=%.9g v_sum=%.9g "
                "q_sumsq=%.9g k_sumsq=%.9g v_sumsq=%.9g ref_o0=%g ref_lse0=%g nodes=%d\n",
                o0, lse0, q_samples[0], k_samples[0], v_samples[0],
                q_samples[1], k_samples[1], v_samples[1],
                q_samples[2], k_samples[2], v_samples[2],
                q_sum, k_sum, v_sum, q_sumsq, k_sumsq, v_sumsq, ref_o0, ref_lse0,
                ggml_graph_n_nodes(f.graph));

    ggml_backend_htp_unregister_packed_f16_weight(f.weight);
    ggml_backend_buffer_free(buffer);
    ggml_free(f.ctx);
    ggml_backend_free(htp);
    ggml_backend_htp_set_hybrid_enabled(false);
}

} // namespace

int main() {
    const int resolution = env_positive_int("GGML_HTP_QKV_E2E_RESOLUTION", 256);
    const int warmups = env_positive_int("GGML_HTP_QKV_PRE_ATTN_WARMUPS", 5);
    const int runs = env_positive_int("GGML_HTP_QKV_PRE_ATTN_RUNS", 3);
    run_case(resolution, warmups, runs);
    return 0;
}
