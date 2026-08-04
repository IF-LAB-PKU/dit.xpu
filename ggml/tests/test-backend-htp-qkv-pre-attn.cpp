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

int env_positive_int(const char * name, int fallback) {
    const char * value = std::getenv(name);
    if (value == nullptr || *value == '\0') {
        return fallback;
    }
    char * end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    return end != value && *end == '\0' && parsed > 0 && parsed <= INT32_MAX ? int(parsed) : fallback;
}

int head_dim_from_env() {
    const int value = env_positive_int("GGML_HTP_QKV_HEAD_DIM", 64);
    GGML_ASSERT(value == 64 || value == 128);
    return value;
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
                                             bool packed_graph, int head_dim) {
    const int64_t qkv_hidden_size = kHeads * head_dim;
    const int64_t qkv_size = 3 * qkv_hidden_size;
    const ggml_init_params params = {
        /* .mem_size   = */ 4 * 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    PreAttentionGraph result;
    result.ctx = ggml_init(params);
    GGML_ASSERT(result.ctx != nullptr);

    result.weight = ggml_new_tensor_2d(result.ctx, GGML_TYPE_F16, kHiddenSize, qkv_size);
    result.activation = ggml_new_tensor_2d(result.ctx, GGML_TYPE_F32, kHiddenSize, tokens);
    result.bias = ggml_new_tensor_1d(result.ctx, GGML_TYPE_F32, qkv_size);
    result.q_gamma = ggml_new_tensor_1d(result.ctx, GGML_TYPE_F32, head_dim);
    result.k_gamma = ggml_new_tensor_1d(result.ctx, GGML_TYPE_F32, head_dim);

    if (packed_graph) {
        const int token_tiles = (tokens + 31) / 32;
        const int64_t qkv_elems = 3 * kHeads * token_tiles * (head_dim / 32) * 1024;
        ggml_tensor * args[] = { result.activation, result.weight, result.bias,
                                 result.q_gamma, result.k_gamma };
        ggml_tensor * qkv = ggml_custom_4d(
            result.ctx, GGML_TYPE_F16, qkv_elems, 1, 1, 1,
            args, 5, packed_qkv_noop, 1, nullptr);
        ggml_set_name(qkv, "qkv_mul_rms_packed");
        result.packed_storage = qkv;
        const size_t qkv_bytes = (size_t) kHeads * token_tiles * (size_t) (head_dim / 32) * 1024 * sizeof(ggml_fp16_t);
        const size_t head_stride = (size_t) token_tiles * (size_t) (head_dim / 32) * 1024 * sizeof(ggml_fp16_t);
        result.q = ggml_view_4d(result.ctx, qkv, head_dim, tokens, kHeads, 1,
                                sizeof(ggml_fp16_t), (size_t) head_dim * sizeof(ggml_fp16_t), head_stride, 0);
        result.k = ggml_view_4d(result.ctx, qkv, head_dim, tokens, kHeads, 1,
                                sizeof(ggml_fp16_t), (size_t) head_dim * sizeof(ggml_fp16_t), head_stride, qkv_bytes);
        result.v = ggml_view_4d(result.ctx, qkv, head_dim, tokens, kHeads, 1,
                                sizeof(ggml_fp16_t), (size_t) head_dim * sizeof(ggml_fp16_t), head_stride, 2 * qkv_bytes);
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
            1.0f / std::sqrt(float(head_dim)), 0.0f, 0.0f);
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
    qkv = ggml_reshape_4d(result.ctx, qkv, qkv_hidden_size, 3, tokens, 1);
    qkv = ggml_cont(result.ctx, ggml_permute(result.ctx, qkv, 0, 3, 1, 2));
    const int64_t branch_offset = qkv->nb[2] * qkv->ne[2];
    ggml_tensor * q = ggml_view_3d(result.ctx, qkv, qkv->ne[0], qkv->ne[1], qkv->ne[2],
                                   qkv->nb[1], qkv->nb[2], branch_offset * 0);
    ggml_tensor * k = ggml_view_3d(result.ctx, qkv, qkv->ne[0], qkv->ne[1], qkv->ne[2],
                                   qkv->nb[1], qkv->nb[2], branch_offset * 1);
    result.v = ggml_view_3d(result.ctx, qkv, qkv->ne[0], qkv->ne[1], qkv->ne[2],
                            qkv->nb[1], qkv->nb[2], branch_offset * 2);

    q = ggml_reshape_4d(result.ctx, q, head_dim, kHeads, tokens, 1);
    k = ggml_reshape_4d(result.ctx, k, head_dim, kHeads, tokens, 1);
    q = ggml_mul_inplace(result.ctx, ggml_rms_norm(result.ctx, q, 1.0e-6f), result.q_gamma);
    k = ggml_mul_inplace(result.ctx, ggml_rms_norm(result.ctx, k, 1.0e-6f), result.k_gamma);
    result.q = ggml_reshape_3d(result.ctx, q, qkv_hidden_size, tokens, 1);
    result.k = ggml_reshape_3d(result.ctx, k, qkv_hidden_size, tokens, 1);

    const int blocks = (tokens + 31) / 32;
    result.mask = ggml_new_tensor_4d(result.ctx, GGML_TYPE_I32,
                                     blocks, kHeads * blocks, 1, 1);
    ggml_set_name(result.mask, "block_sparse_index");
    ggml_tensor * q_layout = ggml_cont(result.ctx, ggml_permute(
        result.ctx, ggml_reshape_4d(result.ctx, result.q, head_dim, tokens, kHeads, 1),
        0, 2, 1, 3));
    ggml_tensor * q_attn = ggml_permute(result.ctx, q_layout, 0, 2, 1, 3);
    ggml_tensor * k_f16 = ggml_cast(result.ctx, result.k, GGML_TYPE_F16);
    ggml_tensor * v_f16 = ggml_cast(result.ctx, result.v, GGML_TYPE_F16);
    ggml_tensor * k_layout = ggml_cont(result.ctx, ggml_permute(
        result.ctx, ggml_reshape_4d(result.ctx, k_f16, head_dim, tokens, kHeads, 1),
        0, 2, 1, 3));
    ggml_tensor * v_layout = ggml_cont(result.ctx, ggml_permute(
        result.ctx, ggml_reshape_4d(result.ctx, v_f16, head_dim, tokens, kHeads, 1),
        0, 2, 1, 3));
    ggml_tensor * k_attn = ggml_permute(result.ctx, k_layout, 0, 2, 1, 3);
    ggml_tensor * v_attn = ggml_permute(result.ctx, v_layout, 0, 2, 1, 3);
    result.output = ggml_flash_attn_ext_with_lse(
        result.ctx, q_attn, k_attn, v_attn, result.mask,
        1.0f / std::sqrt(float(head_dim)), 0.0f, 0.0f);
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

std::vector<int32_t> make_flux2_direct_halo_index(int image_tokens, int text_tokens,
                                                  int image_block_rows, int image_block_cols) {
    const int tokens = image_tokens + text_tokens;
    const int blocks = (tokens + 31) / 32;
    std::vector<int32_t> index(size_t(kHeads) * blocks * blocks, -1);
    GGML_ASSERT(image_tokens == image_block_rows * image_block_cols * 32);
    const int image_blocks = image_block_rows * image_block_cols;
    for (int h = 0; h < kHeads; ++h) {
        for (int qb = 0; qb < blocks; ++qb) {
            int32_t * row = index.data() + (size_t(h) * blocks + qb) * blocks;
            int kept = 0;
            if (qb >= image_blocks) {
                for (int kb = 0; kb < blocks; ++kb) {
                    row[kept++] = h * blocks + kb;
                }
                continue;
            }
            const int qy = qb / image_block_cols;
            const int qx = qb % image_block_cols;
            for (int ky = std::max(0, qy - 1);
                 ky <= std::min(image_block_rows - 1, qy + 1); ++ky) {
                for (int kx = std::max(0, qx - 1);
                     kx <= std::min(image_block_cols - 1, qx + 1); ++kx) {
                    row[kept++] = h * blocks + ky * image_block_cols + kx;
                }
            }
            for (int kb = image_blocks; kb < blocks; ++kb) {
                row[kept++] = h * blocks + kb;
            }
        }
    }
    return index;
}

std::vector<int32_t> make_static_halo_index(int tokens, int image_block_rows,
                                            int image_block_cols) {
    const int blocks = (tokens + 31) / 32;
    const int image_blocks = image_block_rows * image_block_cols;
    std::vector<int32_t> index(size_t(kHeads) * blocks * blocks, -1);
    for (int h = 0; h < kHeads; ++h) {
        for (int qb = 0; qb < blocks; ++qb) {
            int32_t * row = index.data() + (size_t(h) * blocks + qb) * blocks;
            int kept = 0;
            if (qb >= image_blocks) {
                for (int kb = 0; kb < blocks; ++kb) {
                    row[kept++] = h * blocks + kb;
                }
                continue;
            }
            const int qy = qb / image_block_cols;
            const int qx = qb % image_block_cols;
            for (int ky = std::max(0, qy - 1);
                 ky <= std::min(image_block_rows - 1, qy + 1); ++ky) {
                for (int kx = std::max(0, qx - 1);
                     kx <= std::min(image_block_cols - 1, qx + 1); ++kx) {
                    row[kept++] = h * blocks + ky * image_block_cols + kx;
                }
            }
            for (int kb = image_blocks; kb < blocks; ++kb) {
                row[kept++] = h * blocks + kb;
            }
        }
    }
    return index;
}

void run_case(int resolution, int warmups, int runs) {
    GGML_ASSERT(resolution >= 256 && resolution % 16 == 0);
    const int head_dim = head_dim_from_env();
    const int64_t qkv_hidden_size = kHeads * head_dim;
    const int64_t qkv_size = 3 * qkv_hidden_size;
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
        tokens, image_tokens, image_block_rows, image_block_cols, packed_graph, head_dim);
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
    std::vector<ggml_fp16_t> weight(size_t(kHiddenSize) * qkv_size);
    std::vector<ggml_fp16_t> packed_weight(weight.size());
    for (int64_t out = 0; out < qkv_size; ++out) {
        for (int64_t in = 0; in < kHiddenSize; ++in) {
            weight[size_t(out) * kHiddenSize + in] = ggml_fp32_to_fp16(weight_value(in, out));
        }
    }
    ggml_backend_htp_pack_f16_weight(packed_weight.data(), weight.data(), kHiddenSize, qkv_size);

    std::vector<float> bias(qkv_size);
    for (int64_t i = 0; i < qkv_size; ++i) {
        bias[i] = float((i % 17) - 8) * 0.01f;
    }
    std::vector<float> q_gamma(head_dim);
    std::vector<float> k_gamma(head_dim);
    for (int64_t i = 0; i < head_dim; ++i) {
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
    ggml_backend_tensor_get(f.output, &lse0, size_t(head_dim) * sizeof(float), sizeof(lse0));
    if (!packed_graph) {
        const size_t sample_offsets[] = { 0, size_t(head_dim), size_t(qkv_hidden_size) };
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
        std::vector<float> q_all(size_t(tokens) * qkv_hidden_size);
        std::vector<float> k_all(size_t(tokens) * qkv_hidden_size);
        std::vector<float> v_all(size_t(tokens) * qkv_hidden_size);
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
                for (int d = 0; d < head_dim; ++d) {
                    const float kf16 = ggml_fp16_to_fp32(ggml_fp32_to_fp16(
                        k_all[size_t(kt) * qkv_hidden_size + d]));
                    score += q_all[d] * kf16;
                }
                score /= std::sqrt(float(head_dim));
                max_score = std::max(max_score, score);
            }
        }
        float denom = 0.0f;
        float numerator = 0.0f;
        for (int pos = 0; pos < blocks && row[pos] >= 0; ++pos) {
            const int kb = row[pos] % blocks;
            for (int kt = kb * 32; kt < std::min(tokens, kb * 32 + 32); ++kt) {
                float score = 0.0f;
                for (int d = 0; d < head_dim; ++d) {
                    const float kf16 = ggml_fp16_to_fp32(ggml_fp32_to_fp16(
                        k_all[size_t(kt) * qkv_hidden_size + d]));
                    score += q_all[d] * kf16;
                }
                const float p = std::exp(score / std::sqrt(float(head_dim)) - max_score);
                denom += p;
                numerator += p * ggml_fp16_to_fp32(ggml_fp32_to_fp16(
                    v_all[size_t(kt) * qkv_hidden_size]));
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

constexpr int64_t kFlux2HiddenSize = 3072;
constexpr int64_t kFlux2HeadDim = 128;
constexpr int kFlux2ImageTokens = 256;
constexpr int kFlux2TextTokens = 512;

struct Flux2Qkv {
    ggml_tensor * q = nullptr;
    ggml_tensor * k = nullptr;
    ggml_tensor * v = nullptr;
};

struct Flux2Graph {
    ggml_context * ctx = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * image_activation = nullptr;
    ggml_tensor * text_activation = nullptr;
    ggml_tensor * image_weight = nullptr;
    ggml_tensor * text_weight = nullptr;
    ggml_tensor * image_bias = nullptr;
    ggml_tensor * text_bias = nullptr;
    ggml_tensor * image_q_gamma = nullptr;
    ggml_tensor * image_k_gamma = nullptr;
    ggml_tensor * text_q_gamma = nullptr;
    ggml_tensor * text_k_gamma = nullptr;
    ggml_tensor * rope = nullptr;
    ggml_tensor * mask = nullptr;
    ggml_tensor * output = nullptr;
    ggml_tensor * debug_q = nullptr;
    ggml_tensor * debug_k = nullptr;
    ggml_tensor * debug_v = nullptr;
    ggml_tensor * packed_storage = nullptr;
};

Flux2Qkv build_flux2_legacy_branch(ggml_context * ctx,
                                   ggml_tensor * activation,
                                   ggml_tensor * weight,
                                   ggml_tensor * bias,
                                   ggml_tensor * q_gamma,
                                   ggml_tensor * k_gamma) {
    constexpr int64_t qkv_hidden = kHeads * kFlux2HeadDim;
    ggml_tensor * qkv = ggml_add_inplace(ctx, ggml_mul_mat(ctx, weight, activation), bias);
    const size_t head_stride = size_t(kFlux2HeadDim) * qkv->nb[0];
    Flux2Qkv result;
    result.q = ggml_view_4d(ctx, qkv, kFlux2HeadDim, kHeads, activation->ne[1], 1,
                            head_stride, qkv->nb[1], qkv->nb[2], 0);
    result.k = ggml_view_4d(ctx, qkv, kFlux2HeadDim, kHeads, activation->ne[1], 1,
                            head_stride, qkv->nb[1], qkv->nb[2], qkv_hidden * qkv->nb[0]);
    result.v = ggml_view_4d(ctx, qkv, kFlux2HeadDim, kHeads, activation->ne[1], 1,
                            head_stride, qkv->nb[1], qkv->nb[2], 2 * qkv_hidden * qkv->nb[0]);
    result.q = ggml_mul_inplace(ctx, ggml_rms_norm(ctx, result.q, 1.0e-6f), q_gamma);
    result.k = ggml_mul_inplace(ctx, ggml_rms_norm(ctx, result.k, 1.0e-6f), k_gamma);
    return result;
}

ggml_tensor * apply_flux2_legacy_rope(ggml_context * ctx,
                                      ggml_tensor * x,
                                      ggml_tensor * rope) {
    const int64_t head_dim = x->ne[0];
    const int64_t heads = x->ne[1];
    const int64_t tokens = x->ne[2];
    const int64_t batch = x->ne[3];
    x = ggml_cont(ctx, ggml_permute(ctx, x, 0, 2, 1, 3));
    x = ggml_reshape_4d(ctx, x, 2, head_dim / 2, tokens, heads * batch);
    x = ggml_cont(ctx, ggml_permute(ctx, x, 3, 0, 1, 2));

    const size_t x_offset = x->nb[2] * x->ne[2];
    ggml_tensor * x0 = ggml_view_3d(ctx, x, x->ne[0], x->ne[1], x->ne[2],
                                    x->nb[1], x->nb[2], 0);
    ggml_tensor * x1 = ggml_view_3d(ctx, x, x->ne[0], x->ne[1], x->ne[2],
                                    x->nb[1], x->nb[2], x_offset);
    x0 = ggml_reshape_4d(ctx, x0, 1, x0->ne[0], x0->ne[1], x0->ne[2]);
    x1 = ggml_reshape_4d(ctx, x1, 1, x1->ne[0], x1->ne[1], x1->ne[2]);
    ggml_tensor * repeated_shape = ggml_new_tensor_4d(ctx, x0->type, 2, x0->ne[1], x0->ne[2], x0->ne[3]);
    x0 = ggml_repeat(ctx, x0, repeated_shape);
    x1 = ggml_repeat(ctx, x1, repeated_shape);

    rope = ggml_cont(ctx, ggml_permute(ctx, rope, 3, 0, 1, 2));
    const size_t rope_offset = rope->nb[2] * rope->ne[2];
    ggml_tensor * rope0 = ggml_view_3d(ctx, rope, rope->ne[0], rope->ne[1], rope->ne[2],
                                       rope->nb[1], rope->nb[2], 0);
    ggml_tensor * rope1 = ggml_view_3d(ctx, rope, rope->ne[0], rope->ne[1], rope->ne[2],
                                       rope->nb[1], rope->nb[2], rope_offset);
    ggml_tensor * output = ggml_add_inplace(ctx, ggml_mul(ctx, x0, rope0), ggml_mul(ctx, x1, rope1));
    return ggml_reshape_3d(ctx, output, head_dim, tokens, heads * batch);
}

Flux2Qkv build_flux2_packed_views(ggml_context * ctx, ggml_tensor * storage, int tokens) {
    const int token_tiles = (tokens + 31) / 32;
    const size_t head_stride = size_t(token_tiles) * size_t(kFlux2HeadDim / 32) * 1024 * sizeof(ggml_fp16_t);
    const size_t qkv_bytes = size_t(kHeads) * head_stride;
    Flux2Qkv result;
    result.q = ggml_view_4d(ctx, storage, kFlux2HeadDim, tokens, kHeads, 1,
                            sizeof(ggml_fp16_t), kFlux2HeadDim * sizeof(ggml_fp16_t), head_stride, 0);
    result.k = ggml_view_4d(ctx, storage, kFlux2HeadDim, tokens, kHeads, 1,
                            sizeof(ggml_fp16_t), kFlux2HeadDim * sizeof(ggml_fp16_t), head_stride, qkv_bytes);
    result.v = ggml_view_4d(ctx, storage, kFlux2HeadDim, tokens, kHeads, 1,
                            sizeof(ggml_fp16_t), kFlux2HeadDim * sizeof(ggml_fp16_t), head_stride, 2 * qkv_bytes);
    for (ggml_tensor * tensor : {result.q, result.k, result.v}) {
        tensor->op = GGML_OP_NONE;
        tensor->type = GGML_TYPE_F16;
    }
    return result;
}

Flux2Graph build_flux2_graph(bool dual, bool packed) {
    constexpr int tokens = kFlux2ImageTokens + kFlux2TextTokens;
    constexpr int64_t qkv_size = 3 * kHeads * kFlux2HeadDim;
    const ggml_init_params params = {
        /* .mem_size   = */ 8 * 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    Flux2Graph result;
    result.ctx = ggml_init(params);
    GGML_ASSERT(result.ctx != nullptr);

    result.image_activation = ggml_new_tensor_2d(result.ctx, GGML_TYPE_F32, kFlux2HiddenSize,
                                                  dual ? kFlux2ImageTokens : tokens);
    result.image_weight = ggml_new_tensor_2d(result.ctx, GGML_TYPE_F16, kFlux2HiddenSize, qkv_size);
    result.image_bias = ggml_new_tensor_1d(result.ctx, GGML_TYPE_F32, qkv_size);
    result.image_q_gamma = ggml_new_tensor_1d(result.ctx, GGML_TYPE_F32, kFlux2HeadDim);
    result.image_k_gamma = ggml_new_tensor_1d(result.ctx, GGML_TYPE_F32, kFlux2HeadDim);
    if (dual) {
        result.text_activation = ggml_new_tensor_2d(result.ctx, GGML_TYPE_F32, kFlux2HiddenSize, kFlux2TextTokens);
        result.text_weight = ggml_new_tensor_2d(result.ctx, GGML_TYPE_F16, kFlux2HiddenSize, qkv_size);
        result.text_bias = ggml_new_tensor_1d(result.ctx, GGML_TYPE_F32, qkv_size);
        result.text_q_gamma = ggml_new_tensor_1d(result.ctx, GGML_TYPE_F32, kFlux2HeadDim);
        result.text_k_gamma = ggml_new_tensor_1d(result.ctx, GGML_TYPE_F32, kFlux2HeadDim);
    }
    result.rope = ggml_new_tensor_4d(result.ctx, GGML_TYPE_F32, 2, 2, kFlux2HeadDim / 2, tokens);

    Flux2Qkv qkv;
    if (packed) {
        const int token_tiles = (tokens + 31) / 32;
        const int64_t packed_elems = 3 * kHeads * token_tiles * (kFlux2HeadDim / 32) * 1024;
        ggml_tensor * storage = nullptr;
        if (dual) {
            ggml_tensor * args[] = {
                result.image_activation, result.image_weight, result.image_bias,
                result.image_q_gamma, result.image_k_gamma,
                result.text_activation, result.text_weight, result.text_bias,
                result.text_q_gamma, result.text_k_gamma, result.rope,
            };
            storage = ggml_custom_4d(result.ctx, GGML_TYPE_F16, packed_elems, 1, 1, 1,
                                     args, 11, packed_qkv_noop, 1, nullptr);
            ggml_set_name(storage, "qkv_mul_rms_packed_dual_rope");
        } else {
            ggml_tensor * args[] = {
                result.image_activation, result.image_weight, result.image_bias,
                result.image_q_gamma, result.image_k_gamma, result.rope,
            };
            storage = ggml_custom_4d(result.ctx, GGML_TYPE_F16, packed_elems, 1, 1, 1,
                                     args, 6, packed_qkv_noop, 1, nullptr);
            ggml_set_name(storage, "qkv_mul_rms_packed_rope");
        }
        result.packed_storage = storage;
        qkv = build_flux2_packed_views(result.ctx, storage, tokens);
    } else {
        Flux2Qkv image = build_flux2_legacy_branch(
            result.ctx, result.image_activation, result.image_weight, result.image_bias,
            result.image_q_gamma, result.image_k_gamma);
        if (dual) {
            Flux2Qkv text = build_flux2_legacy_branch(
                result.ctx, result.text_activation, result.text_weight, result.text_bias,
                result.text_q_gamma, result.text_k_gamma);
            qkv.q = ggml_concat(result.ctx, image.q, text.q, 2);
            qkv.k = ggml_concat(result.ctx, image.k, text.k, 2);
            qkv.v = ggml_concat(result.ctx, image.v, text.v, 2);
        } else {
            qkv = image;
        }
        qkv.q = apply_flux2_legacy_rope(result.ctx, qkv.q, result.rope);
        qkv.k = apply_flux2_legacy_rope(result.ctx, qkv.k, result.rope);
        qkv.k = ggml_cast(result.ctx, qkv.k, GGML_TYPE_F16);
        qkv.v = ggml_cast(result.ctx, ggml_permute(result.ctx, qkv.v, 0, 2, 1, 3), GGML_TYPE_F16);
    }
    result.debug_q = qkv.q;
    result.debug_k = qkv.k;
    result.debug_v = qkv.v;

    constexpr int blocks = (tokens + 31) / 32;
    result.mask = ggml_new_tensor_4d(result.ctx, GGML_TYPE_I32, blocks, kHeads * blocks, 1, 1);
    ggml_set_name(result.mask, "block_sparse_index");
    result.output = ggml_flash_attn_ext_with_lse(
        result.ctx, qkv.q, qkv.k, qkv.v, result.mask,
        1.0f / std::sqrt(float(kFlux2HeadDim)), 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(result.output, GGML_PREC_F32);
    ggml_flash_attn_ext_set_block_sparse(
        result.output, 32, 32, GGML_FLASH_ATTN_EXT_MASK_BLOCK_INDEX);
    ggml_flash_attn_ext_set_static_halo_2d(
        result.output, kFlux2ImageTokens, kFlux2TextTokens, 4, 2);
    ggml_set_name(result.output, packed ? "packed_sparse_attention" : "reordered_sparse_attention");
    result.graph = ggml_new_graph_custom(result.ctx, 128, false);
    GGML_ASSERT(result.graph != nullptr);
    ggml_build_forward_expand(result.graph, result.output);
    return result;
}

void set_flux2_branch_data(ggml_tensor * activation,
                           ggml_tensor * weight,
                           ggml_tensor * bias,
                           ggml_tensor * q_gamma,
                           ggml_tensor * k_gamma,
                           int branch_offset) {
    constexpr int64_t qkv_size = 3 * kHeads * kFlux2HeadDim;
    std::vector<float> activation_data(size_t(activation->ne[1]) * kFlux2HiddenSize);
    for (int64_t token = 0; token < activation->ne[1]; ++token) {
        for (int64_t col = 0; col < kFlux2HiddenSize; ++col) {
            activation_data[size_t(token) * kFlux2HiddenSize + col] =
                activation_value(token + branch_offset, col + branch_offset);
        }
    }
    std::vector<ggml_fp16_t> weight_data(size_t(kFlux2HiddenSize) * qkv_size);
    std::vector<ggml_fp16_t> packed_weight(weight_data.size());
    for (int64_t out = 0; out < qkv_size; ++out) {
        for (int64_t in = 0; in < kFlux2HiddenSize; ++in) {
            weight_data[size_t(out) * kFlux2HiddenSize + in] =
                ggml_fp32_to_fp16(weight_value(in + branch_offset, out + branch_offset));
        }
    }
    ggml_backend_htp_pack_f16_weight(
        packed_weight.data(), weight_data.data(), kFlux2HiddenSize, qkv_size);
    std::vector<float> bias_data(qkv_size);
    for (int64_t i = 0; i < qkv_size; ++i) {
        bias_data[i] = float(((i + branch_offset) % 19) - 9) * 0.007f;
    }
    std::vector<float> q_gamma_data(kFlux2HeadDim);
    std::vector<float> k_gamma_data(kFlux2HeadDim);
    for (int64_t i = 0; i < kFlux2HeadDim; ++i) {
        q_gamma_data[i] = 0.9f + float((i + branch_offset) % 11) * 0.013f;
        k_gamma_data[i] = 1.1f - float((i + branch_offset) % 7) * 0.009f;
    }
    ggml_backend_tensor_set(activation, activation_data.data(), 0, activation_data.size() * sizeof(float));
    ggml_backend_tensor_set(weight, packed_weight.data(), 0, packed_weight.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(bias, bias_data.data(), 0, bias_data.size() * sizeof(float));
    ggml_backend_tensor_set(q_gamma, q_gamma_data.data(), 0, q_gamma_data.size() * sizeof(float));
    ggml_backend_tensor_set(k_gamma, k_gamma_data.data(), 0, k_gamma_data.size() * sizeof(float));
    ggml_backend_htp_register_packed_f16_weight(weight, weight);
}

uint64_t fnv1a_hash(const void * data, size_t size) {
    const auto * bytes = static_cast<const uint8_t *>(data);
    uint64_t hash = UINT64_C(1469598103934665603);
    for (size_t i = 0; i < size; ++i) {
        hash ^= bytes[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

float flux2_packed_value(const std::vector<ggml_fp16_t> & packed,
                         int tokens, int token, int head, int dim, int region, int kind) {
    const int token_tiles = (tokens + 31) / 32;
    const size_t head_elems = size_t(token_tiles) * (kFlux2HeadDim / 32) * 1024;
    const size_t qkv_bytes = size_t(kHeads) * head_elems;
    size_t tile = 0;
    size_t in_tile = 0;
    if (kind == 0) {
        tile = size_t(head) * head_elems + ((size_t) (token / 32) * (kFlux2HeadDim / 32) + dim / 32) * 1024;
        in_tile = size_t(((token % 32) & ~1) * 32 + (dim % 32) * 2 + (token & 1));
    } else if (kind == 1) {
        tile = size_t(head) * head_elems + ((size_t) (token / 32) * (kFlux2HeadDim / 32) + dim / 32) * 1024;
        in_tile = size_t(((dim % 32) & ~1) * 32 + (token % 32) * 2 + (dim & 1));
    } else {
        tile = size_t(head) * head_elems + ((size_t) (dim / 32) * token_tiles + token / 32) * 1024;
        in_tile = size_t(((token % 32) & ~1) * 32 + (dim % 32) * 2 + (token & 1));
    }
    return ggml_fp16_to_fp32(packed[size_t(region) * qkv_bytes + tile + in_tile]);
}

void run_flux2_case(int warmups, int runs) {
    const bool dual = std::getenv("GGML_HTP_FLUX2_DUAL") != nullptr &&
                      std::strcmp(std::getenv("GGML_HTP_FLUX2_DUAL"), "0") != 0;
    const bool packed = std::getenv("GGML_HTP_QKV_PACKED_GRAPH") != nullptr &&
                        std::strcmp(std::getenv("GGML_HTP_QKV_PACKED_GRAPH"), "0") != 0;
    ggml_backend_htp_set_hybrid_enabled(true);
    ggml_backend_t htp = ggml_backend_init_by_name("HTP", nullptr);
    GGML_ASSERT(htp != nullptr && ggml_backend_htp_runtime_available());
    ggml_backend_htp_set_n_threads(htp, env_positive_int("GGML_HTP_QKV_PRE_ATTN_THREADS", 8));

    Flux2Graph f = build_flux2_graph(dual, packed);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors_from_buft(
        f.ctx, ggml_backend_get_default_buffer_type(htp));
    GGML_ASSERT(buffer != nullptr);
    set_flux2_branch_data(f.image_activation, f.image_weight, f.image_bias,
                          f.image_q_gamma, f.image_k_gamma, 0);
    if (dual) {
        set_flux2_branch_data(f.text_activation, f.text_weight, f.text_bias,
                              f.text_q_gamma, f.text_k_gamma, 17);
    }
    std::vector<float> rope_data(size_t(kFlux2ImageTokens + kFlux2TextTokens) *
                                 size_t(kFlux2HeadDim / 2) * 4);
    for (int token = 0; token < kFlux2ImageTokens + kFlux2TextTokens; ++token) {
        for (int d = 0; d < kFlux2HeadDim / 2; ++d) {
            const float angle = 0.013f * float(token + 1) * float(d + 1);
            float * m = rope_data.data() + (size_t(token) * (kFlux2HeadDim / 2) + d) * 4;
            m[0] = std::cos(angle);
            m[1] = -std::sin(angle);
            m[2] = std::sin(angle);
            m[3] = std::cos(angle);
        }
    }
    ggml_backend_tensor_set(f.rope, rope_data.data(), 0, rope_data.size() * sizeof(float));
    const std::vector<int32_t> mask = make_flux2_direct_halo_index(
        kFlux2ImageTokens, kFlux2TextTokens, 4, 2);
    ggml_backend_tensor_set(f.mask, mask.data(), 0, mask.size() * sizeof(int32_t));

    for (int i = 0; i < warmups; ++i) {
        GGML_ASSERT(ggml_backend_graph_compute(htp, f.graph) == GGML_STATUS_SUCCESS);
    }
    std::vector<int64_t> elapsed;
    elapsed.reserve(runs);
    for (int i = 0; i < runs; ++i) {
        const int64_t start = now_us();
        GGML_ASSERT(ggml_backend_graph_compute(htp, f.graph) == GGML_STATUS_SUCCESS);
        elapsed.push_back(now_us() - start);
    }

    std::vector<float> output(ggml_nelements(f.output));
    ggml_backend_tensor_get(f.output, output.data(), 0, output.size() * sizeof(float));
    bool finite = true;
    for (float value : output) finite = finite && std::isfinite(value);
    GGML_ASSERT(finite);

    std::vector<ggml_fp16_t> packed_values;
    if (packed) {
        packed_values.resize(ggml_nelements(f.packed_storage));
        ggml_backend_tensor_get(f.packed_storage, packed_values.data(), 0,
                                packed_values.size() * sizeof(ggml_fp16_t));
    }
    const int tokens = kFlux2ImageTokens + kFlux2TextTokens;
    const int blocks = tokens / 32;
    std::vector<float> q_cpu(size_t(kHeads) * tokens * kFlux2HeadDim);
    std::vector<float> k_cpu(q_cpu.size());
    std::vector<float> v_cpu(q_cpu.size());
    auto raw_value = [](ggml_tensor * tensor, int token, int head, int dim) {
        ggml_fp16_t value16 = 0;
        float value32 = 0.0f;
        const size_t offset = size_t(head) * tensor->nb[2] + size_t(token) * tensor->nb[1] + size_t(dim) * tensor->nb[0];
        if (tensor->type == GGML_TYPE_F32) {
            ggml_backend_tensor_get(tensor, &value32, offset, sizeof(value32));
            return value32;
        }
        ggml_backend_tensor_get(tensor, &value16, offset, sizeof(value16));
        return ggml_fp16_to_fp32(value16);
    };
    for (int head = 0; head < kHeads; ++head) {
        for (int token = 0; token < tokens; ++token) {
            for (int dim = 0; dim < kFlux2HeadDim; ++dim) {
                const size_t index = (size_t(head) * tokens + token) * kFlux2HeadDim + dim;
                if (packed) {
                    q_cpu[index] = flux2_packed_value(packed_values, tokens, token, head, dim, 0, 0);
                    k_cpu[index] = flux2_packed_value(packed_values, tokens, token, head, dim, 1, 1);
                    v_cpu[index] = flux2_packed_value(packed_values, tokens, token, head, dim, 2, 2);
                } else {
                    q_cpu[index] = raw_value(f.debug_q, token, head, dim);
                    k_cpu[index] = raw_value(f.debug_k, token, head, dim);
                    v_cpu[index] = raw_value(f.debug_v, token, head, dim);
                }
            }
        }
    }
    const size_t q0 = 0;
    float max_score = -std::numeric_limits<float>::infinity();
    for (int block = 0; block < blocks; ++block) {
        const bool image_key = block < 8 && (block == 0 || block == 1 || block == 2 || block == 3);
        const bool text_key = block >= 8;
        if (!image_key && !text_key) continue;
        for (int token = block * 32; token < block * 32 + 32; ++token) {
            float score = 0.0f;
            for (int dim = 0; dim < kFlux2HeadDim; ++dim) {
                score += q_cpu[q0 + dim] * k_cpu[(size_t(token) * kFlux2HeadDim) + dim];
            }
            max_score = std::max(max_score, score / std::sqrt(float(kFlux2HeadDim)));
        }
    }
    float denom = 0.0f;
    float numerator = 0.0f;
    for (int block = 0; block < blocks; ++block) {
        const bool image_key = block < 4;
        const bool text_key = block >= 8;
        if (!image_key && !text_key) continue;
        for (int token = block * 32; token < block * 32 + 32; ++token) {
            float score = 0.0f;
            for (int dim = 0; dim < kFlux2HeadDim; ++dim) {
                score += q_cpu[q0 + dim] * k_cpu[(size_t(token) * kFlux2HeadDim) + dim];
            }
            const float probability = std::exp(score / std::sqrt(float(kFlux2HeadDim)) - max_score);
            denom += probability;
            numerator += probability * v_cpu[(size_t(token) * kFlux2HeadDim)];
        }
    }
    const float oracle_o0 = numerator / denom;
    const float oracle_lse = max_score + std::log(denom);
    float actual_o0 = 0.0f;
    float actual_lse = 0.0f;
    ggml_backend_tensor_get(f.output, &actual_o0, 0, sizeof(actual_o0));
    ggml_backend_tensor_get(f.output, &actual_lse, size_t(kFlux2HeadDim) * sizeof(float), sizeof(actual_lse));
    const float oracle_err_o0 = std::fabs(oracle_o0 - actual_o0);
    const float oracle_err_lse = std::fabs(oracle_lse - actual_lse);
    const char * output_path = std::getenv("GGML_HTP_FLUX2_OUTPUT");
    if (output_path != nullptr && *output_path != '\0') {
        FILE * file = std::fopen(output_path, "wb");
        GGML_ASSERT(file != nullptr);
        const size_t written = std::fwrite(output.data(), sizeof(float), output.size(), file);
        GGML_ASSERT(std::fclose(file) == 0 && written == output.size());
    }

    std::vector<int64_t> sorted = elapsed;
    std::sort(sorted.begin(), sorted.end());
    const double average = double(std::accumulate(elapsed.begin(), elapsed.end(), int64_t{0})) / elapsed.size();
    const double median = sorted.size() % 2 != 0 ? double(sorted[sorted.size() / 2]) :
        0.5 * double(sorted[sorted.size() / 2 - 1] + sorted[sorted.size() / 2]);
    const size_t p95_index = (95 * sorted.size() + 99) / 100 - 1;
    const char * loader = std::getenv("SD_HTP_HALO_KV_LOAD");
    std::printf("flux2_qkv_sparse_e2e: device=8gen3 soc=SM8650 stream=%s layout=%s loader=%s "
                "image_tokens=%d text_tokens=%d heads=%lld head_dim=%lld warmups=%d runs=%d "
                "avg_us=%.1f median_us=%.1f p95_us=%lld min_us=%lld nodes=%d "
                "output_hash=%016llx oracle_err_o0=%g oracle_err_lse=%g\n",
                dual ? "double" : "single", packed ? "packed_no_reorder" : "reordered",
                loader != nullptr ? loader : "default", kFlux2ImageTokens, kFlux2TextTokens,
                (long long) kHeads, (long long) kFlux2HeadDim, warmups, runs, average, median,
                (long long) sorted[p95_index], (long long) sorted.front(), ggml_graph_n_nodes(f.graph),
                (unsigned long long) fnv1a_hash(output.data(), output.size() * sizeof(float)),
                oracle_err_o0, oracle_err_lse);

    ggml_backend_htp_unregister_packed_f16_weight(f.image_weight);
    if (dual) ggml_backend_htp_unregister_packed_f16_weight(f.text_weight);
    ggml_backend_buffer_free(buffer);
    ggml_free(f.ctx);
    ggml_backend_free(htp);
    ggml_backend_htp_set_hybrid_enabled(false);
}

} // namespace

int main() {
    const int warmups = env_positive_int("GGML_HTP_QKV_PRE_ATTN_WARMUPS", 5);
    const int runs = env_positive_int("GGML_HTP_QKV_PRE_ATTN_RUNS", 3);
    if (std::getenv("GGML_HTP_FLUX2_PRE_ATTN") != nullptr &&
        std::strcmp(std::getenv("GGML_HTP_FLUX2_PRE_ATTN"), "0") != 0) {
        run_flux2_case(warmups, runs);
        return 0;
    }
    const int resolution = env_positive_int("GGML_HTP_QKV_E2E_RESOLUTION", 256);
    run_case(resolution, warmups, runs);
    return 0;
}
