#include "test-backend-htp-attention-common.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>

#include "../../src/ggml_extend.hpp"

namespace htp_attention_common {
namespace {

constexpr ImageSize kImageSizes[] = {
    {256, 256, 4, 2},
    {512, 1024, 8, 4},
    {1024, 4096, 16, 8},
};

constexpr CaseSpec kCases[] = {
    {"sd3.5", "joint", 24, 64, kImageTokens, kTextTokens, true, true, false, false, true},
    {"sd3.5", "self2", 24, 64, kImageTokens, 0, false, true, false, false, true},
    {"sd3.5-large", "joint", 38, 64, kImageTokens, kTextTokens, true, true, false, false, true},
    {"sd3.5-large", "self2", 38, 64, kImageTokens, 0, false, true, false, false, true},
    {"flux2", "double-stream", 24, 128, kImageTokens, kTextTokens, true, true, true, false, true},
    {"flux2", "single-stream", 24, 128, kImageTokens, kTextTokens, false, true, true, false, true},
    {"z-image-turbo", "joint", 30, 128, kImageTokens, kTextTokens, true, false, true, false, true},
    {"z-image-turbo", "context-refiner", 30, 128, kImageTokens, 0, false, false, true, true, false},
    {"z-image-turbo", "noise-refiner", 30, 128, kImageTokens, 0, false, false, true, false, true},
    {"z-image-base", "joint", 30, 128, kImageTokens, kTextTokens, true, false, true, false, true},
    {"z-image-base", "context-refiner", 30, 128, kImageTokens, 0, false, false, true, true, false},
    {"z-image-base", "noise-refiner", 30, 128, kImageTokens, 0, false, false, true, false, true},
};

static float activation_value(int token, int col, int seed) {
    return float(((token * 11 + col * 5 + seed * 17) % 37) - 18) * 0.01f;
}

static float weight_value(int input, int output, int seed) {
    uint32_t value = uint32_t(input) * 0x9e3779b1u ^ uint32_t(output) * 0x85ebca6bu ^
                     uint32_t(seed) * 0xc2b2ae35u;
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    return float(int(value % 101u) - 50) * 0.001f;
}

static std::vector<float> identity_rope(int tokens, int head_dim) {
    std::vector<float> rope(size_t(tokens) * (head_dim / 2) * 4, 0.0f);
    for (int t = 0; t < tokens; ++t) {
        for (int d = 0; d < head_dim / 2; ++d) {
            const size_t base = (size_t(t) * (head_dim / 2) + d) * 4;
            rope[base + 0] = 1.0f;
            rope[base + 3] = 1.0f;
        }
    }
    return rope;
}

static void packed_qkv_noop(ggml_tensor *, int, int, void *) {}

static BranchData make_branch(const CaseSpec & spec, int tokens, int seed) {
    const int qkv = 3 * spec.heads * spec.head_dim;
    BranchData data;
    data.tokens = tokens;
    data.activation.resize(size_t(tokens) * kInputDim);
    data.weight.resize(size_t(kInputDim) * qkv);
    data.packed_weight.resize(data.weight.size());
    if (spec.bias) data.bias.resize(qkv);
    data.q_gamma.resize(spec.head_dim);
    data.k_gamma.resize(spec.head_dim);
    for (int t = 0; t < tokens; ++t) {
        for (int i = 0; i < kInputDim; ++i) {
            data.activation[size_t(t) * kInputDim + i] = activation_value(t, i, seed);
        }
    }
    for (int o = 0; o < qkv; ++o) {
        for (int i = 0; i < kInputDim; ++i) {
            data.weight[size_t(o) * kInputDim + i] = ggml_fp32_to_fp16(weight_value(i, o, seed));
        }
        if (spec.bias) data.bias[o] = float(((o + seed * 3) % 19) - 9) * 0.007f;
    }
    for (int d = 0; d < spec.head_dim; ++d) {
        data.q_gamma[d] = 0.9f + float((d + seed) % 11) * 0.013f;
        data.k_gamma[d] = 1.1f - float((d + seed) % 7) * 0.009f;
    }
    ggml_backend_htp_pack_f16_weight(data.packed_weight.data(), data.weight.data(), kInputDim, qkv);
    return data;
}

static void apply_identity_rope(ggml_context * ctx, ggml_tensor ** q, ggml_tensor ** k,
                                ggml_tensor * rope, int head_dim) {
    auto apply = [ctx, rope, head_dim](ggml_tensor * x) {
        const int64_t tokens = x->ne[2];
        const int64_t heads = x->ne[1];
        x = ggml_cont(ctx, ggml_permute(ctx, x, 0, 2, 1, 3));
        x = ggml_reshape_4d(ctx, x, 2, head_dim / 2, tokens, heads);
        x = ggml_cont(ctx, ggml_permute(ctx, x, 3, 0, 1, 2));

        const size_t x_offset = x->nb[2] * x->ne[2];
        ggml_tensor * x0 = ggml_view_3d(ctx, x, x->ne[0], x->ne[1], x->ne[2], x->nb[1], x->nb[2], 0);
        ggml_tensor * x1 = ggml_view_3d(ctx, x, x->ne[0], x->ne[1], x->ne[2], x->nb[1], x->nb[2], x_offset);
        x0 = ggml_reshape_4d(ctx, x0, 1, x0->ne[0], x0->ne[1], x0->ne[2]);
        x1 = ggml_reshape_4d(ctx, x1, 1, x1->ne[0], x1->ne[1], x1->ne[2]);
        ggml_tensor * repeated_shape = ggml_new_tensor_4d(ctx, x0->type, 2, x0->ne[1], x0->ne[2], x0->ne[3]);
        x0 = ggml_repeat(ctx, x0, repeated_shape);
        x1 = ggml_repeat(ctx, x1, repeated_shape);

        ggml_tensor * rope_tensor = ggml_cont(ctx, ggml_permute(ctx, rope, 3, 0, 1, 2));
        const size_t rope_offset = rope_tensor->nb[2] * rope_tensor->ne[2];
        ggml_tensor * rope0 = ggml_view_3d(ctx, rope_tensor, rope_tensor->ne[0], rope_tensor->ne[1], rope_tensor->ne[2], rope_tensor->nb[1], rope_tensor->nb[2], 0);
        ggml_tensor * rope1 = ggml_view_3d(ctx, rope_tensor, rope_tensor->ne[0], rope_tensor->ne[1], rope_tensor->ne[2], rope_tensor->nb[1], rope_tensor->nb[2], rope_offset);
        ggml_tensor * output = ggml_add_inplace(ctx, ggml_mul(ctx, x0, rope0), ggml_mul(ctx, x1, rope1));
        return ggml_reshape_3d(ctx, output, head_dim, tokens, heads);
    };
    *q = apply(*q);
    *k = apply(*k);
}

struct BranchQkv {
    ggml_tensor * q = nullptr;
    ggml_tensor * k = nullptr;
    ggml_tensor * v = nullptr;
};

static BranchQkv build_standard_branch(ggml_context * ctx, const CaseSpec & spec,
                                        ggml_tensor * activation, ggml_tensor * weight,
                                        ggml_tensor * bias, ggml_tensor * q_gamma,
                                        ggml_tensor * k_gamma) {
    const int hidden = spec.heads * spec.head_dim;
    ggml_tensor * qkv = ggml_mul_mat(ctx, weight, activation);
    if (bias != nullptr) qkv = ggml_add_inplace(ctx, qkv, bias);
    if (spec.rope) {
        const size_t head_stride = size_t(spec.head_dim) * qkv->nb[0];
        BranchQkv result;
        result.q = ggml_view_4d(ctx, qkv, spec.head_dim, spec.heads, activation->ne[1], 1,
                                head_stride, qkv->nb[1], qkv->nb[2], 0);
        result.k = ggml_view_4d(ctx, qkv, spec.head_dim, spec.heads, activation->ne[1], 1,
                                head_stride, qkv->nb[1], qkv->nb[2], size_t(hidden) * qkv->nb[0]);
        result.v = ggml_view_4d(ctx, qkv, spec.head_dim, spec.heads, activation->ne[1], 1,
                                head_stride, qkv->nb[1], qkv->nb[2], size_t(2 * hidden) * qkv->nb[0]);
        result.q = ggml_mul_inplace(ctx, ggml_rms_norm(ctx, result.q, 1.0e-6f), q_gamma);
        result.k = ggml_mul_inplace(ctx, ggml_rms_norm(ctx, result.k, 1.0e-6f), k_gamma);
        return result;
    }
    qkv = ggml_reshape_4d(ctx, qkv, hidden, 3, activation->ne[1], 1);
    qkv = ggml_cont(ctx, ggml_permute(ctx, qkv, 0, 3, 1, 2));
    const size_t branch_stride = qkv->nb[2] * qkv->ne[2];
    BranchQkv result;
    result.q = ggml_view_3d(ctx, qkv, qkv->ne[0], qkv->ne[1], qkv->ne[2],
                            qkv->nb[1], qkv->nb[2], 0);
    result.k = ggml_view_3d(ctx, qkv, qkv->ne[0], qkv->ne[1], qkv->ne[2],
                            qkv->nb[1], qkv->nb[2], branch_stride);
    result.v = ggml_view_3d(ctx, qkv, qkv->ne[0], qkv->ne[1], qkv->ne[2],
                            qkv->nb[1], qkv->nb[2], 2 * branch_stride);
    result.q = ggml_reshape_4d(ctx, result.q, spec.head_dim, spec.heads,
                                activation->ne[1], 1);
    result.k = ggml_reshape_4d(ctx, result.k, spec.head_dim, spec.heads,
                                activation->ne[1], 1);
    result.q = ggml_mul_inplace(ctx, ggml_rms_norm(ctx, result.q, 1.0e-6f), q_gamma);
    result.k = ggml_mul_inplace(ctx, ggml_rms_norm(ctx, result.k, 1.0e-6f), k_gamma);
    result.q = ggml_reshape_3d(ctx, result.q, hidden, activation->ne[1], 1);
    result.k = ggml_reshape_3d(ctx, result.k, hidden, activation->ne[1], 1);
    result.v = ggml_reshape_3d(ctx, result.v, hidden, activation->ne[1], 1);
    return result;
}

static void standard_attention_views(ggml_context * ctx, const CaseSpec & spec,
                                     ggml_tensor ** q, ggml_tensor ** k, ggml_tensor ** v,
                                     ggml_tensor * rope) {
    if (spec.rope) {
        ggml_tensor * q4 = ggml_reshape_4d(ctx, *q, spec.head_dim, spec.heads, (*q)->ne[2], 1);
        ggml_tensor * k4 = ggml_reshape_4d(ctx, *k, spec.head_dim, spec.heads, (*k)->ne[2], 1);
        apply_identity_rope(ctx, &q4, &k4, rope, spec.head_dim);

        ggml_tensor * q_base = ggml_cont(ctx, ggml_permute(ctx, q4, 0, 2, 1, 3));
        ggml_tensor * k_base = ggml_cont(ctx, ggml_permute(ctx, k4, 0, 2, 1, 3));
        ggml_tensor * v_base = ggml_cont(ctx, *v);
        *q = ggml_permute(ctx, q_base, 0, 2, 1, 3);
        *k = ggml_permute(ctx, ggml_cast(ctx, k_base, GGML_TYPE_F16), 0, 2, 1, 3);
        *v = ggml_permute(ctx, ggml_cast(ctx, v_base, GGML_TYPE_F16), 0, 2, 1, 3);
    } else {
        const int tokens = (*q)->ne[1];
        ggml_tensor * q_base = ggml_reshape_3d(ctx, *q, spec.head_dim, spec.heads, tokens);
        ggml_tensor * k_base = ggml_reshape_3d(ctx, *k, spec.head_dim, spec.heads, tokens);
        ggml_tensor * v_base = ggml_reshape_3d(ctx, *v, spec.head_dim, spec.heads, tokens);
        *q = ggml_permute(ctx, q_base, 0, 2, 1, 3);
        *k = ggml_permute(ctx, ggml_cast(ctx, k_base, GGML_TYPE_F16), 0, 2, 1, 3);
        *v = ggml_permute(ctx, ggml_cast(ctx, v_base, GGML_TYPE_F16), 0, 2, 1, 3);
    }
}

static void add_standard_projection_inputs(Graph & graph, const CaseSpec & spec) {
    const int branches = spec.dual ? 2 : 1;
    const int branch_tokens[] = {spec.dual ? spec.image_tokens : total_tokens(spec), spec.text_tokens};
    const int qkv = 3 * spec.heads * spec.head_dim;
    for (int b = 0; b < branches; ++b) {
        graph.activations.push_back(ggml_new_tensor_2d(graph.ctx, GGML_TYPE_F32, kInputDim, branch_tokens[b]));
        graph.weights.push_back(ggml_new_tensor_2d(graph.ctx, GGML_TYPE_F16, kInputDim, qkv));
        graph.biases.push_back(spec.bias ? ggml_new_tensor_1d(graph.ctx, GGML_TYPE_F32, qkv) : nullptr);
        graph.q_gammas.push_back(ggml_new_tensor_1d(graph.ctx, GGML_TYPE_F32, spec.head_dim));
        graph.k_gammas.push_back(ggml_new_tensor_1d(graph.ctx, GGML_TYPE_F32, spec.head_dim));
        ggml_set_name(graph.activations.back(), b == 0 ? "attention_input_x_image" : "attention_input_x_text");
        ggml_set_name(graph.weights.back(), b == 0 ? "attention_qkv_weight_image" : "attention_qkv_weight_text");
    }
    if (spec.rope) {
        graph.rope = ggml_new_tensor_4d(graph.ctx, GGML_TYPE_F32, 2, 2, spec.head_dim / 2, total_tokens(spec));
        ggml_set_name(graph.rope, "attention_rope");
    }
}

static void build_standard_attention(Graph & graph, const CaseSpec & spec, Route route) {
    const int branches = spec.dual ? 2 : 1;
    BranchQkv qkv[2];
    for (int b = 0; b < branches; ++b) {
        qkv[b] = build_standard_branch(graph.ctx, spec, graph.activations[b], graph.weights[b],
                                       graph.biases[b], graph.q_gammas[b], graph.k_gammas[b]);
    }
    ggml_tensor * q = qkv[0].q;
    ggml_tensor * k = qkv[0].k;
    ggml_tensor * v = qkv[0].v;
    if (spec.dual) {
        const int concat_axis = spec.rope ? 2 : 1;
        q = ggml_concat(graph.ctx, q, qkv[1].q, concat_axis);
        k = ggml_concat(graph.ctx, k, qkv[1].k, concat_axis);
        v = ggml_concat(graph.ctx, v, qkv[1].v, concat_axis);
    }
    standard_attention_views(graph.ctx, spec, &q, &k, &v, graph.rope);
    graph.debug_q = q;
    graph.debug_k = k;
    graph.debug_v = v;

    const int tokens = total_tokens(spec);
    const int blocks = (tokens + kBlock - 1) / kBlock;
    ggml_tensor * attention_output = nullptr;
    if (route == Route::Dense) {
        const int padded_tokens = ((tokens + 63) / 64) * 64;
        graph.mask = ggml_new_tensor_2d(graph.ctx, GGML_TYPE_F16, padded_tokens, tokens);
        ggml_set_name(graph.mask, "dense_attention_mask");
        attention_output = ggml_flash_attn_ext(
            graph.ctx, q, k, v, graph.mask, 1.0f / std::sqrt(float(spec.head_dim)), 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(attention_output, GGML_PREC_F32);
        ggml_set_name(attention_output, "dense_attention");
    } else {
        graph.mask = ggml_new_tensor_4d(graph.ctx, GGML_TYPE_I32, blocks, spec.heads * blocks, 1, 1);
        ggml_set_name(graph.mask, "block_sparse_index");
        attention_output = ggml_flash_attn_ext(
            graph.ctx, q, k, v, graph.mask, 1.0f / std::sqrt(float(spec.head_dim)), 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(attention_output, GGML_PREC_F32);
        ggml_flash_attn_ext_set_block_sparse(attention_output, kBlock, kBlock,
                                              GGML_FLASH_ATTN_EXT_MASK_BLOCK_INDEX);
        ggml_flash_attn_ext_set_static_halo_2d(attention_output, spec.image_tokens, spec.text_tokens,
                                                spec.image_block_rows, spec.image_block_cols);
        ggml_set_name(attention_output, "external_sparse_attention");
    }
    graph.output = ggml_reshape_3d(graph.ctx, ggml_cont(graph.ctx, attention_output),
                                   spec.heads * spec.head_dim, tokens, 1);
}

static void build_packed_attention(Graph & graph, const CaseSpec & spec) {
    add_standard_projection_inputs(graph, spec);
    const int tokens = total_tokens(spec);
    const int qkv = 3 * spec.heads * spec.head_dim;
    const int tiles = (tokens + 31) / 32;
    const int64_t elems = int64_t(3) * spec.heads * tiles * (spec.head_dim / 32) * 1024;
    if (spec.dual) {
        ggml_tensor * args[] = {
            graph.activations[0], graph.weights[0], graph.biases[0], graph.q_gammas[0], graph.k_gammas[0],
            graph.activations[1], graph.weights[1], graph.biases[1], graph.q_gammas[1], graph.k_gammas[1], graph.rope,
        };
        graph.packed_storage = ggml_custom_4d(graph.ctx, GGML_TYPE_F16, elems, 1, 1, 1,
                                               args, spec.rope ? 11 : 10, packed_qkv_noop, 1, nullptr);
        ggml_set_name(graph.packed_storage, spec.rope ? "qkv_mul_rms_packed_dual_rope" : "qkv_mul_rms_packed_dual");
    } else {
        ggml_tensor * args[] = {
            graph.activations[0], graph.weights[0], graph.biases[0], graph.q_gammas[0], graph.k_gammas[0], graph.rope,
        };
        graph.packed_storage = ggml_custom_4d(graph.ctx, GGML_TYPE_F16, elems, 1, 1, 1,
                                               args, spec.rope ? 6 : 5, packed_qkv_noop, 1, nullptr);
        ggml_set_name(graph.packed_storage, spec.rope ? "qkv_mul_rms_packed_rope" : "qkv_mul_rms_packed");
    }
    const size_t head_stride = size_t(tiles) * (spec.head_dim / 32) * 1024 * sizeof(ggml_fp16_t);
    const size_t region = size_t(spec.heads) * head_stride;
    ggml_tensor * q = ggml_view_4d(graph.ctx, graph.packed_storage, spec.head_dim, tokens, spec.heads, 1,
                                    sizeof(ggml_fp16_t), size_t(spec.head_dim) * sizeof(ggml_fp16_t), head_stride, 0);
    ggml_tensor * k = ggml_view_4d(graph.ctx, graph.packed_storage, spec.head_dim, tokens, spec.heads, 1,
                                    sizeof(ggml_fp16_t), size_t(spec.head_dim) * sizeof(ggml_fp16_t), head_stride, region);
    ggml_tensor * v = ggml_view_4d(graph.ctx, graph.packed_storage, spec.head_dim, tokens, spec.heads, 1,
                                    sizeof(ggml_fp16_t), size_t(spec.head_dim) * sizeof(ggml_fp16_t), head_stride, 2 * region);
    for (ggml_tensor * tensor : {q, k, v}) {
        tensor->op = GGML_OP_NONE;
        tensor->type = GGML_TYPE_F16;
    }
    graph.mask = ggml_new_tensor_4d(graph.ctx, GGML_TYPE_I32, (tokens + 31) / 32,
                                    spec.heads * ((tokens + 31) / 32), 1, 1);
    ggml_set_name(graph.mask, "block_sparse_index");
    ggml_tensor * packed_output = ggml_flash_attn_ext_with_lse(
        graph.ctx, q, k, v, graph.mask, 1.0f / std::sqrt(float(spec.head_dim)), 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(packed_output, GGML_PREC_F32);
    ggml_flash_attn_ext_set_block_sparse(packed_output, kBlock, kBlock,
                                          GGML_FLASH_ATTN_EXT_MASK_BLOCK_INDEX);
    ggml_flash_attn_ext_set_static_halo_2d(packed_output, spec.image_tokens, spec.text_tokens,
                                            spec.image_block_rows, spec.image_block_cols);
    ggml_set_name(packed_output, "packed_sparse_attention");
    ggml_tensor * output_view = ggml_view_3d(graph.ctx, packed_output, spec.head_dim, spec.heads, tokens,
                                             packed_output->nb[1], packed_output->nb[2], 0);
    graph.output = ggml_reshape_3d(graph.ctx, ggml_cont(graph.ctx, output_view),
                                   spec.heads * spec.head_dim, tokens, 1);
    (void) qkv;
}

} // namespace

const ImageSize * image_sizes() { return kImageSizes; }
int image_size_count() { return int(sizeof(kImageSizes) / sizeof(kImageSizes[0])); }
const CaseSpec * cases() { return kCases; }
int case_count() { return int(sizeof(kCases) / sizeof(kCases[0])); }
int total_tokens(const CaseSpec & spec) { return spec.image_tokens + spec.text_tokens; }

bool selected(const ImageSize & size, const CaseSpec & spec) {
    const char * resolution = std::getenv("HTP_ATTN_TEST_RESOLUTION");
    if (resolution == nullptr) resolution = std::getenv("HTP_ATTN_PERF_TEST_RESOLUTION");
    if (resolution != nullptr && resolution[0] != '\0' && std::atoi(resolution) != size.resolution) return false;
    const char * model = std::getenv("HTP_ATTN_TEST_MODEL");
    if (model == nullptr) model = std::getenv("HTP_ATTN_PERF_TEST_MODEL");
    if (model != nullptr && model[0] != '\0' && std::strcmp(model, spec.model) != 0) return false;
    const char * family = std::getenv("HTP_ATTN_TEST_FAMILY");
    if (family == nullptr) family = std::getenv("HTP_ATTN_PERF_TEST_FAMILY");
    if (family != nullptr && family[0] != '\0' && std::strcmp(family, spec.family) != 0) return false;
    return true;
}

InputData make_input(const CaseSpec & spec) {
    InputData data;
    data.branches.push_back(make_branch(spec, spec.dual ? spec.image_tokens : total_tokens(spec), 1));
    if (spec.dual) data.branches.push_back(make_branch(spec, spec.text_tokens, 7));
    data.block_index = make_block_index(spec);
    if (spec.rope) data.rope = identity_rope(total_tokens(spec), spec.head_dim);
    return data;
}

void append_reference_branch(const CaseSpec & spec, const BranchData & branch,
                             bool sparse, Reference & ref) {
    const int hidden = spec.heads * spec.head_dim;
    const int qkv = 3 * hidden;
    const size_t old_tokens = ref.q.size() / hidden;
    ref.q.resize((old_tokens + branch.tokens) * hidden);
    ref.k.resize((old_tokens + branch.tokens) * hidden);
    ref.v.resize((old_tokens + branch.tokens) * hidden);
    std::vector<float> projected(qkv);
    for (int t = 0; t < branch.tokens; ++t) {
        for (int o = 0; o < qkv; ++o) {
            float sum = spec.bias ? branch.bias[o] : 0.0f;
            for (int i = 0; i < kInputDim; ++i) {
                sum += ggml_fp16_to_fp32(branch.weight[size_t(o) * kInputDim + i]) *
                       ggml_fp16_to_fp32(ggml_fp32_to_fp16(branch.activation[size_t(t) * kInputDim + i]));
            }
            projected[o] = sum;
        }
        for (int h = 0; h < spec.heads; ++h) {
            float q_ss = 0.0f;
            float k_ss = 0.0f;
            for (int d = 0; d < spec.head_dim; ++d) {
                q_ss += projected[h * spec.head_dim + d] * projected[h * spec.head_dim + d];
                k_ss += projected[hidden + h * spec.head_dim + d] * projected[hidden + h * spec.head_dim + d];
            }
            const float q_scale = 1.0f / std::sqrt(q_ss / spec.head_dim + 1.0e-6f);
            const float k_scale = 1.0f / std::sqrt(k_ss / spec.head_dim + 1.0e-6f);
            for (int d = 0; d < spec.head_dim; ++d) {
                const size_t dst = (old_tokens + t) * hidden + h * spec.head_dim + d;
                float q = projected[h * spec.head_dim + d] * q_scale * branch.q_gamma[d];
                const float k = projected[hidden + h * spec.head_dim + d] * k_scale * branch.k_gamma[d];
                const float v = projected[2 * hidden + h * spec.head_dim + d];
                if (sparse) q = ggml_fp16_to_fp32(ggml_fp32_to_fp16(q));
                ref.q[dst] = q;
                ref.k[dst] = ggml_fp32_to_fp16(k);
                ref.v[dst] = ggml_fp32_to_fp16(v);
            }
        }
    }
}

std::vector<int32_t> make_block_index(const CaseSpec & spec) {
    const int tokens = total_tokens(spec);
    const int blocks = (tokens + kBlock - 1) / kBlock;
    const int image_blocks = spec.image_tokens / kBlock;
    std::vector<int32_t> index(size_t(spec.heads) * blocks * blocks, -1);
    for (int h = 0; h < spec.heads; ++h) {
        for (int qb = 0; qb < blocks; ++qb) {
            int32_t * row = index.data() + (size_t(h) * blocks + qb) * blocks;
            int kept = 0;
            if (spec.full_keep || qb >= image_blocks) {
                for (int kb = 0; kb < blocks; ++kb) row[kept++] = h * blocks + kb;
                continue;
            }
            const int qy = qb / spec.image_block_cols;
            const int qx = qb % spec.image_block_cols;
            for (int ky = std::max(0, qy - 1); ky <= std::min(spec.image_block_rows - 1, qy + 1); ++ky) {
                for (int kx = std::max(0, qx - 1); kx <= std::min(spec.image_block_cols - 1, qx + 1); ++kx) {
                    row[kept++] = h * blocks + ky * spec.image_block_cols + kx;
                }
            }
            for (int kb = image_blocks; kb < blocks; ++kb) row[kept++] = h * blocks + kb;
        }
    }
    return index;
}

std::vector<float> reference_attention(const CaseSpec & spec, const Reference & ref,
                                       const std::vector<int32_t> * index) {
    const int tokens = total_tokens(spec);
    const int hidden = spec.heads * spec.head_dim;
    const int blocks = (tokens + kBlock - 1) / kBlock;
    std::vector<float> result(size_t(tokens) * spec.heads * spec.head_dim);
    std::vector<int> keys;
    keys.reserve(tokens);
    for (int qt = 0; qt < tokens; ++qt) {
        for (int h = 0; h < spec.heads; ++h) {
            keys.clear();
            if (index == nullptr) {
                for (int kt = 0; kt < tokens; ++kt) keys.push_back(kt);
            } else {
                const int qb = qt / kBlock;
                const int32_t * row = index->data() + (size_t(h) * blocks + qb) * blocks;
                for (int p = 0; p < blocks && row[p] >= 0; ++p) {
                    const int kb = row[p] % blocks;
                    for (int kt = kb * kBlock; kt < std::min(tokens, (kb + 1) * kBlock); ++kt) keys.push_back(kt);
                }
            }
            float max_score = -std::numeric_limits<float>::infinity();
            std::vector<float> scores(keys.size());
            for (size_t i = 0; i < keys.size(); ++i) {
                float score = 0.0f;
                const int kt = keys[i];
                for (int d = 0; d < spec.head_dim; ++d) {
                    score += ref.q[size_t(qt) * hidden + h * spec.head_dim + d] *
                             ggml_fp16_to_fp32(ref.k[size_t(kt) * hidden + h * spec.head_dim + d]);
                }
                scores[i] = score / std::sqrt(float(spec.head_dim));
                max_score = std::max(max_score, scores[i]);
            }
            float denom = 0.0f;
            for (float score : scores) denom += std::exp(score - max_score);
            const size_t base = (size_t(qt) * spec.heads + h) * spec.head_dim;
            for (int d = 0; d < spec.head_dim; ++d) {
                float value = 0.0f;
                for (size_t i = 0; i < keys.size(); ++i) {
                    value += std::exp(scores[i] - max_score) *
                             ggml_fp16_to_fp32(ref.v[size_t(keys[i]) * hidden + h * spec.head_dim + d]);
                }
                result[base + d] = value / denom;
            }
        }
    }
    return result;
}

Graph build_graph(const CaseSpec & spec, Route route) {
    Graph graph;
    graph.route = route;
    const ggml_init_params params = {128 * 1024 * 1024, nullptr, true};
    graph.ctx = ggml_init(params);
    GGML_ASSERT(graph.ctx != nullptr);
    if (route == Route::PackedSparse) {
        build_packed_attention(graph, spec);
    } else {
        add_standard_projection_inputs(graph, spec);
        build_standard_attention(graph, spec, route);
    }
    graph.graph = ggml_new_graph_custom(graph.ctx, 512, false);
    GGML_ASSERT(graph.graph != nullptr);
    ggml_build_forward_expand(graph.graph, graph.output);
    return graph;
}

bool upload_graph(Graph & graph, const InputData & data, ggml_backend_t backend) {
    graph.buffer = ggml_backend_alloc_ctx_tensors_from_buft(
        graph.ctx, ggml_backend_get_default_buffer_type(backend));
    if (graph.buffer == nullptr || data.branches.size() != graph.activations.size()) return false;
    for (size_t i = 0; i < data.branches.size(); ++i) {
        const BranchData & branch = data.branches[i];
        ggml_backend_tensor_set(graph.activations[i], branch.activation.data(), 0,
                                branch.activation.size() * sizeof(float));
        ggml_backend_tensor_set(graph.weights[i], branch.packed_weight.data(), 0,
                                branch.packed_weight.size() * sizeof(ggml_fp16_t));
        if (graph.biases[i] != nullptr) {
            ggml_backend_tensor_set(graph.biases[i], branch.bias.data(), 0,
                                    branch.bias.size() * sizeof(float));
        }
        ggml_backend_tensor_set(graph.q_gammas[i], branch.q_gamma.data(), 0,
                                branch.q_gamma.size() * sizeof(float));
        ggml_backend_tensor_set(graph.k_gammas[i], branch.k_gamma.data(), 0,
                                branch.k_gamma.size() * sizeof(float));
        ggml_backend_htp_register_packed_f16_weight(graph.weights[i], graph.weights[i]);
    }
    if (graph.rope != nullptr) {
        ggml_backend_tensor_set(graph.rope, data.rope.data(), 0, data.rope.size() * sizeof(float));
    }
    if (graph.mask->type == GGML_TYPE_I32) {
        ggml_backend_tensor_set(graph.mask, data.block_index.data(), 0,
                                data.block_index.size() * sizeof(int32_t));
    } else {
        const int tokens = graph.mask->ne[1];
        const int padded_tokens = graph.mask->ne[0];
        std::vector<ggml_fp16_t> mask(ggml_nelements(graph.mask), ggml_fp32_to_fp16(-INFINITY));
        for (int query = 0; query < tokens; ++query) {
            std::fill_n(mask.data() + size_t(query) * padded_tokens, tokens,
                        ggml_fp32_to_fp16(0.0f));
        }
        ggml_backend_tensor_set(graph.mask, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
    }
    return true;
}

bool graph_contract(const Graph & graph, const CaseSpec & spec) {
    if (graph.graph == nullptr || graph.output == nullptr || graph.activations.empty()) {
        return false;
    }
    int attention_nodes = 0;
    int projection_nodes = 0;
    int packed_nodes = 0;
    bool has_activation = false;
    for (int i = 0; i < ggml_graph_n_nodes(graph.graph); ++i) {
        const ggml_tensor * node = ggml_graph_node(graph.graph, i);
        if (node == nullptr) continue;
        if (node->op == GGML_OP_FLASH_ATTN_EXT) {
            ++attention_nodes;
        }
        if (node->op == GGML_OP_MUL_MAT) {
            ++projection_nodes;
        }
        if (node->op == GGML_OP_CUSTOM &&
            std::strncmp(node->name, "qkv_mul_rms_packed", std::strlen("qkv_mul_rms_packed")) == 0) {
            ++packed_nodes;
        }
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            const ggml_tensor * src = node->src[j];
            if (src != nullptr && std::strncmp(src->name, "attention_input_x_", 18) == 0) {
                has_activation = true;
            }
        }
    }
    if (attention_nodes != 1 || !has_activation) return false;
    if (graph.route == Route::PackedSparse) {
        return packed_nodes == 1 && projection_nodes == 0;
    }
    return packed_nodes == 0 && projection_nodes >= (spec.dual ? 2 : 1);
}

void free_graph(Graph & graph) {
    for (ggml_tensor * weight : graph.weights) ggml_backend_htp_unregister_packed_f16_weight(weight);
    if (graph.buffer != nullptr) ggml_backend_buffer_free(graph.buffer);
    if (graph.ctx != nullptr) ggml_free(graph.ctx);
    graph = {};
}

int run_fixture_contract_test() {
    const CaseSpec & spec = kCases[4];
    Graph dense = build_graph(spec, Route::Dense);
    Graph external = build_graph(spec, Route::ExternalSparseL2);
    Graph chunk = build_graph(spec, Route::ExternalSparseChunk);
    Graph packed = build_graph(spec, Route::PackedSparse);
    const bool ok = graph_contract(dense, spec) && graph_contract(external, spec) &&
                    graph_contract(chunk, spec) && graph_contract(packed, spec);
    free_graph(dense);
    free_graph(external);
    free_graph(chunk);
    free_graph(packed);
    return ok ? 0 : 1;
}

} // namespace htp_attention_common
