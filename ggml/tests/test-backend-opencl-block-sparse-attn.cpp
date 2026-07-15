#include "ggml-htp.h"
#include "ggml-opencl-htp.h"
#include "ggml-opencl.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <fcntl.h>

namespace {

constexpr int64_t k_head_dim = 64;
constexpr int64_t k_block_size = 32;
constexpr uint8_t k_guard = 0xa5;

enum class mask_pattern {
    full,
    sparse_terminated,
    invalid_ids,
    empty,
};

struct attention_case {
    const char * name;
    int64_t lq;
    int64_t lk;
    int64_t heads;
    int64_t kv_heads;
    int64_t batch;
    ggml_flash_attn_ext_mask_format format;
    mask_pattern pattern;
    bool nonzero_offsets;
    bool permuted_kv;
    bool compare_htp_sparse;
    bool compare_htp_dense;
};

struct mask_data {
    int64_t ne0 = 0;
    int64_t ne1 = 0;
    std::vector<int32_t> values;
};

struct generated_inputs {
    std::vector<float> q;
    std::vector<ggml_fp16_t> k;
    std::vector<ggml_fp16_t> v;
    std::vector<float> k_f32;
    std::vector<float> v_f32;
};

struct owned_rpcmem_buffer {
    ggml_backend_buffer_t buffer = nullptr;
    ggml_backend_htp_rpcmem_info info = {};
    size_t tensor_offset = 0;
};

void test_ordinary_opencl_support_gating(ggml_backend_t opencl) {
    const ggml_init_params params = {
        /* .mem_size   = */ 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    GGML_ASSERT(ctx != nullptr);

    auto make_attention = [&](ggml_type q_type, ggml_type kv_type, ggml_type mask_type, bool with_lse) {
        ggml_tensor * q = ggml_new_tensor_4d(ctx, q_type, 64, 32, 4, 1);
        ggml_tensor * k = ggml_new_tensor_4d(ctx, kv_type, 64, 32, 4, 1);
        ggml_tensor * v = ggml_new_tensor_4d(ctx, kv_type, 64, 32, 4, 1);
        ggml_tensor * mask = mask_type == GGML_TYPE_COUNT ? nullptr :
            ggml_new_tensor_4d(ctx, mask_type, 32, 32, 1, 1);
        return with_lse ?
            ggml_flash_attn_ext_with_lse(ctx, q, k, v, mask, 0.125f, 0.0f, 0.0f) :
            ggml_flash_attn_ext(ctx, q, k, v, mask, 0.125f, 0.0f, 0.0f);
    };

    ggml_tensor * f32_dense = make_attention(GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F16, false);
    ggml_tensor * mixed_dense = make_attention(GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_F16, false);
    ggml_tensor * maskless = make_attention(GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_COUNT, false);
    GGML_ASSERT(ggml_backend_supports_op(opencl, f32_dense));
    GGML_ASSERT(ggml_backend_supports_op(opencl, mixed_dense));
    GGML_ASSERT(ggml_backend_supports_op(opencl, maskless));

    ggml_tensor * i32_matrix = make_attention(GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_I32, true);
    ggml_flash_attn_ext_set_block_sparse(
        i32_matrix, k_block_size, k_block_size, GGML_FLASH_ATTN_EXT_MASK_BLOCK_MATRIX);
    ggml_tensor * i32_index = make_attention(GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_I32, true);
    ggml_flash_attn_ext_set_block_sparse(
        i32_index, k_block_size, k_block_size, GGML_FLASH_ATTN_EXT_MASK_BLOCK_INDEX);
    GGML_ASSERT(!ggml_backend_supports_op(opencl, i32_matrix));
    GGML_ASSERT(!ggml_backend_supports_op(opencl, i32_index));

    GGML_ASSERT(!ggml_backend_supports_op(
        opencl, make_attention(GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F16, true)));
    GGML_ASSERT(!ggml_backend_supports_op(
        opencl, make_attention(GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_COUNT, true)));
    ggml_tensor f32_mask = *f32_dense->src[3];
    f32_mask.type = GGML_TYPE_F32;
    ggml_tensor wrong_mask_type = *f32_dense;
    wrong_mask_type.src[3] = &f32_mask;
    GGML_ASSERT(!ggml_backend_supports_op(opencl, &wrong_mask_type));

    ggml_tensor tagged_block_sparse = *mixed_dense;
    tagged_block_sparse.op_params[4] = k_block_size;
    tagged_block_sparse.op_params[5] = k_block_size;
    tagged_block_sparse.op_params[7] = GGML_FLASH_ATTN_EXT_MASK_BLOCK_MATRIX;
    GGML_ASSERT(!ggml_backend_supports_op(opencl, &tagged_block_sparse));

    ggml_tensor mixed_k = *mixed_dense->src[1];
    mixed_k.ne[0] = 80;
    ggml_tensor mixed_dimensions = *mixed_dense;
    mixed_dimensions.src[1] = &mixed_k;
    GGML_ASSERT(!ggml_backend_supports_op(opencl, &mixed_dimensions));

    ggml_free(ctx);
}

size_t q_index(const attention_case & c, int64_t d, int64_t qt, int64_t h, int64_t b) {
    return static_cast<size_t>(d + k_head_dim * (qt + c.lq * (h + c.heads * b)));
}

size_t kv_index(const attention_case & c, int64_t d, int64_t kt, int64_t h, int64_t b) {
    return static_cast<size_t>(d + k_head_dim * (kt + c.lk * (h + c.kv_heads * b)));
}

size_t packed_index(const attention_case & c, int64_t d, int64_t h, int64_t qt, int64_t b) {
    return static_cast<size_t>(d + (k_head_dim + 1) * (h + c.heads * (qt + c.lq * b)));
}

int64_t q_blocks_per_head(const attention_case & c) {
    return (c.lq + k_block_size - 1) / k_block_size;
}

int64_t k_blocks_per_head(const attention_case & c) {
    return (c.lk + k_block_size - 1) / k_block_size;
}

mask_data make_mask(const attention_case & c) {
    const int64_t q_blocks = q_blocks_per_head(c);
    const int64_t k_blocks = k_blocks_per_head(c);
    const int64_t total_q_blocks = c.heads * q_blocks;
    const int64_t total_k_blocks = c.kv_heads * k_blocks;
    const int64_t heads_per_kv = c.heads / c.kv_heads;
    mask_data result;
    result.ne1 = total_q_blocks;

    if (c.format == GGML_FLASH_ATTN_EXT_MASK_BLOCK_MATRIX) {
        result.ne0 = total_k_blocks;
        result.values.assign(
            static_cast<size_t>(result.ne0 * result.ne1), c.pattern == mask_pattern::full ? 0 : 1);
        if (c.pattern == mask_pattern::sparse_terminated) {
            for (int64_t h = 0; h < c.heads; ++h) {
                const int64_t kvh = h / heads_per_kv;
                for (int64_t qb = 0; qb < q_blocks; ++qb) {
                    const int64_t qb_abs = h * q_blocks + qb;
                    const int64_t first = kvh * k_blocks + qb % k_blocks;
                    const int64_t last = (kvh + 1) * k_blocks - 1;
                    result.values[static_cast<size_t>(first + result.ne0 * qb_abs)] = 0;
                    result.values[static_cast<size_t>(last + result.ne0 * qb_abs)] = 0;
                }
            }
        }
        return result;
    }

    if (c.pattern == mask_pattern::full) {
        result.ne0 = k_blocks;
    } else if (c.pattern == mask_pattern::invalid_ids) {
        result.ne0 = 7;
    } else {
        result.ne0 = 4;
    }
    result.values.assign(static_cast<size_t>(result.ne0 * result.ne1), -1);
    for (int64_t h = 0; h < c.heads; ++h) {
        const int64_t kvh = h / heads_per_kv;
        const int64_t local_base = kvh * k_blocks;
        for (int64_t qb = 0; qb < q_blocks; ++qb) {
            int32_t * row = result.values.data() + (h * q_blocks + qb) * result.ne0;
            if (c.pattern == mask_pattern::full) {
                for (int64_t kb = 0; kb < k_blocks; ++kb) {
                    row[kb] = static_cast<int32_t>(local_base + kb);
                }
            } else if (c.pattern == mask_pattern::sparse_terminated) {
                row[0] = static_cast<int32_t>(local_base + qb % k_blocks);
                row[1] = static_cast<int32_t>(local_base + (qb + 1) % k_blocks);
                row[2] = -1;
                row[3] = -1;
            } else if (c.pattern == mask_pattern::invalid_ids) {
                const int64_t foreign_kvh = c.kv_heads > 1 ? (kvh + 1) % c.kv_heads : kvh;
                row[0] = static_cast<int32_t>(foreign_kvh * k_blocks);
                row[1] = static_cast<int32_t>(total_k_blocks + 9);
                row[2] = static_cast<int32_t>(local_base + qb % k_blocks);
                row[3] = -7;
                row[4] = static_cast<int32_t>(local_base + k_blocks - 1);
                row[5] = -1;
                row[6] = -1;
            }
        }
    }
    return result;
}

bool block_is_kept(
    const attention_case & c,
    const mask_data & mask,
    int64_t query_head,
    int64_t query_block,
    int64_t kv_head,
    int64_t key_block) {
    const int64_t q_blocks = q_blocks_per_head(c);
    const int64_t k_blocks = k_blocks_per_head(c);
    const int64_t qb_abs = query_head * q_blocks + query_block;
    const int64_t kb_abs = kv_head * k_blocks + key_block;
    if (c.format == GGML_FLASH_ATTN_EXT_MASK_BLOCK_MATRIX) {
        return mask.values[static_cast<size_t>(kb_abs + mask.ne0 * qb_abs)] == 0;
    }

    const int32_t * row = mask.values.data() + qb_abs * mask.ne0;
    for (int64_t i = 0; i < mask.ne0; ++i) {
        const int32_t candidate = row[i];
        if (candidate == -1) {
            break;
        }
        if (candidate < 0 || candidate >= c.kv_heads * k_blocks ||
            candidate / k_blocks != kv_head) {
            continue;
        }
        if (candidate == kb_abs) {
            return true;
        }
    }
    return false;
}

generated_inputs make_inputs(const attention_case & c) {
    generated_inputs inputs;
    inputs.q.resize(static_cast<size_t>(k_head_dim * c.lq * c.heads * c.batch));
    inputs.k.resize(static_cast<size_t>(k_head_dim * c.lk * c.kv_heads * c.batch));
    inputs.v.resize(inputs.k.size());
    inputs.k_f32.resize(inputs.k.size());
    inputs.v_f32.resize(inputs.v.size());

    for (int64_t b = 0; b < c.batch; ++b) {
        for (int64_t h = 0; h < c.heads; ++h) {
            for (int64_t qt = 0; qt < c.lq; ++qt) {
                for (int64_t d = 0; d < k_head_dim; ++d) {
                    const float phase =
                        0.017f * float((d + 1) * (qt + 1)) + 0.61f * float(h) + 0.29f * float(b);
                    inputs.q[q_index(c, d, qt, h, b)] =
                        0.45f * std::sin(phase) + 0.05f * std::cos(0.13f * float(d + 3 * h));
                }
            }
        }
        for (int64_t h = 0; h < c.kv_heads; ++h) {
            for (int64_t kt = 0; kt < c.lk; ++kt) {
                for (int64_t d = 0; d < k_head_dim; ++d) {
                    const size_t index = kv_index(c, d, kt, h, b);
                    const float k_value =
                        0.42f * std::cos(0.011f * float((d + 2) * (kt + 1)) +
                                         0.83f * float(h) + 0.23f * float(b)) +
                        0.025f * float(h);
                    const float v_value = 0.75f * (
                        0.36f * std::sin(0.019f * float((d + 1) * (kt + 3)) +
                                         0.97f * float(h) + 0.31f * float(b)) +
                        0.4f * float(h) + 0.07f * float(b));
                    inputs.k[index] = ggml_fp32_to_fp16(k_value);
                    inputs.v[index] = ggml_fp32_to_fp16(v_value);
                    inputs.k_f32[index] = ggml_fp16_to_fp32(inputs.k[index]);
                    inputs.v_f32[index] = ggml_fp16_to_fp32(inputs.v[index]);
                }
            }
        }
    }
    return inputs;
}

std::vector<float> reference_attention(
    const attention_case & c,
    const generated_inputs & inputs,
    const mask_data & mask,
    float scale) {
    const int64_t heads_per_kv = c.heads / c.kv_heads;
    std::vector<float> packed(
        static_cast<size_t>((k_head_dim + 1) * c.heads * c.lq * c.batch), 0.0f);
    std::vector<float> scores(static_cast<size_t>(c.lk));

    for (int64_t b = 0; b < c.batch; ++b) {
        for (int64_t qt = 0; qt < c.lq; ++qt) {
            for (int64_t h = 0; h < c.heads; ++h) {
                const int64_t kvh = h / heads_per_kv;
                float maximum = -std::numeric_limits<float>::infinity();
                std::fill(scores.begin(), scores.end(), -std::numeric_limits<float>::infinity());
                for (int64_t kt = 0; kt < c.lk; ++kt) {
                    if (!block_is_kept(c, mask, h, qt / k_block_size, kvh, kt / k_block_size)) {
                        continue;
                    }
                    float dot = 0.0f;
                    for (int64_t d = 0; d < k_head_dim; ++d) {
                        dot += inputs.q[q_index(c, d, qt, h, b)] *
                               inputs.k_f32[kv_index(c, d, kt, kvh, b)];
                    }
                    scores[static_cast<size_t>(kt)] = dot * scale;
                    maximum = std::max(maximum, scores[static_cast<size_t>(kt)]);
                }

                const size_t row = packed_index(c, 0, h, qt, b);
                if (!std::isfinite(maximum)) {
                    packed[row + k_head_dim] = -std::numeric_limits<float>::infinity();
                    continue;
                }
                float sum = 0.0f;
                for (int64_t kt = 0; kt < c.lk; ++kt) {
                    const float score = scores[static_cast<size_t>(kt)];
                    if (!std::isfinite(score)) {
                        continue;
                    }
                    const float weight = std::exp(score - maximum);
                    sum += weight;
                    for (int64_t d = 0; d < k_head_dim; ++d) {
                        packed[row + static_cast<size_t>(d)] +=
                            weight * inputs.v_f32[kv_index(c, d, kt, kvh, b)];
                    }
                }
                GGML_ASSERT(sum > 0.0f);
                for (int64_t d = 0; d < k_head_dim; ++d) {
                    packed[row + static_cast<size_t>(d)] /= sum;
                }
                packed[row + k_head_dim] = maximum + std::log(sum);
            }
        }
    }
    return packed;
}

owned_rpcmem_buffer allocate_tensor_buffer(
    ggml_backend_t htp,
    ggml_tensor * tensor,
    size_t offset) {
    owned_rpcmem_buffer owned;
    const size_t tensor_size = ggml_backend_buft_get_alloc_size(
        ggml_backend_get_default_buffer_type(htp), tensor);
    const size_t total_size = offset + tensor_size + 4096;
    owned.buffer = ggml_backend_alloc_buffer(htp, total_size);
    GGML_ASSERT(owned.buffer != nullptr);
    GGML_ASSERT(ggml_backend_htp_get_rpcmem_info(owned.buffer, &owned.info));
    GGML_ASSERT(owned.info.logical_size == total_size);
    owned.tensor_offset = offset;
    GGML_ASSERT(ggml_backend_tensor_alloc(
        owned.buffer, tensor, static_cast<uint8_t *>(owned.info.base) + offset) == GGML_STATUS_SUCCESS);
    std::memset(owned.info.base, k_guard, owned.info.logical_size);
    return owned;
}

ggml_opencl_htp_import_descriptor make_import_descriptor(const owned_rpcmem_buffer & owned) {
    return {
        /* .backing         = */ owned.buffer,
        /* .base            = */ owned.info.base,
        /* .fd              = */ owned.info.fd,
        /* .logical_size    = */ owned.info.logical_size,
        /* .allocation_size = */ owned.info.allocation_size,
    };
}

template<typename T>
void write_scalar(ggml_tensor * tensor, int64_t i0, int64_t i1, int64_t i2, int64_t i3, T value) {
    uint8_t * address = static_cast<uint8_t *>(tensor->data) +
        i0 * tensor->nb[0] + i1 * tensor->nb[1] + i2 * tensor->nb[2] + i3 * tensor->nb[3];
    std::memcpy(address, &value, sizeof(value));
}

template<typename T>
T read_scalar(const ggml_tensor * tensor, int64_t i0, int64_t i1, int64_t i2, int64_t i3) {
    const uint8_t * address = static_cast<const uint8_t *>(tensor->data) +
        i0 * tensor->nb[0] + i1 * tensor->nb[1] + i2 * tensor->nb[2] + i3 * tensor->nb[3];
    T value;
    std::memcpy(&value, address, sizeof(value));
    return value;
}

void fill_tensors(
    const attention_case & c,
    const generated_inputs & inputs,
    const mask_data & mask_data,
    ggml_tensor * q,
    ggml_tensor * k,
    ggml_tensor * v,
    ggml_tensor * mask) {
    for (int64_t b = 0; b < c.batch; ++b) {
        for (int64_t h = 0; h < c.heads; ++h) {
            for (int64_t qt = 0; qt < c.lq; ++qt) {
                for (int64_t d = 0; d < k_head_dim; ++d) {
                    write_scalar(q, d, qt, h, b, inputs.q[q_index(c, d, qt, h, b)]);
                }
            }
        }
        for (int64_t h = 0; h < c.kv_heads; ++h) {
            for (int64_t kt = 0; kt < c.lk; ++kt) {
                for (int64_t d = 0; d < k_head_dim; ++d) {
                    const size_t index = kv_index(c, d, kt, h, b);
                    write_scalar(k, d, kt, h, b, inputs.k[index]);
                    write_scalar(v, d, kt, h, b, inputs.v[index]);
                }
            }
        }
    }
    for (int64_t i1 = 0; i1 < mask_data.ne1; ++i1) {
        for (int64_t i0 = 0; i0 < mask_data.ne0; ++i0) {
            write_scalar(mask, i0, i1, 0, 0, mask_data.values[static_cast<size_t>(i0 + mask_data.ne0 * i1)]);
        }
    }
}

std::vector<uint8_t> snapshot(const owned_rpcmem_buffer & owned) {
    const uint8_t * data = static_cast<const uint8_t *>(owned.info.base);
    return std::vector<uint8_t>(data, data + owned.info.logical_size);
}

void assert_output_guards(const attention_case & c, const ggml_tensor * packed, const owned_rpcmem_buffer & owned) {
    std::vector<uint8_t> touched(owned.info.logical_size, 0);
    const size_t base_offset = static_cast<const uint8_t *>(packed->data) -
                               static_cast<const uint8_t *>(owned.info.base);
    for (int64_t b = 0; b < c.batch; ++b) {
        for (int64_t qt = 0; qt < c.lq; ++qt) {
            for (int64_t h = 0; h < c.heads; ++h) {
                for (int64_t d = 0; d <= k_head_dim; ++d) {
                    const size_t offset = base_offset + d * packed->nb[0] + h * packed->nb[1] +
                                          qt * packed->nb[2] + b * packed->nb[3];
                    GGML_ASSERT(offset + sizeof(float) <= touched.size());
                    std::fill(touched.begin() + offset, touched.begin() + offset + sizeof(float), 1);
                }
            }
        }
    }
    const uint8_t * bytes = static_cast<const uint8_t *>(owned.info.base);
    for (size_t i = 0; i < touched.size(); ++i) {
        if (!touched[i]) {
            GGML_ASSERT(bytes[i] == k_guard);
        }
    }
}

struct diff_stats {
    float out_max_abs = 0.0f;
    float lse_max_abs = 0.0f;
    bool valid = true;
};

diff_stats compare_packed(
    const attention_case & c,
    const ggml_tensor * actual_tensor,
    const std::vector<float> & expected) {
    diff_stats stats;
    for (int64_t b = 0; b < c.batch; ++b) {
        for (int64_t qt = 0; qt < c.lq; ++qt) {
            for (int64_t h = 0; h < c.heads; ++h) {
                const size_t row = packed_index(c, 0, h, qt, b);
                for (int64_t d = 0; d < k_head_dim; ++d) {
                    const float actual = read_scalar<float>(actual_tensor, d, h, qt, b);
                    const float reference = expected[row + static_cast<size_t>(d)];
                    stats.valid = stats.valid && std::isfinite(actual) && std::isfinite(reference);
                    if (std::isfinite(actual) && std::isfinite(reference)) {
                        stats.out_max_abs = std::max(stats.out_max_abs, std::fabs(actual - reference));
                    }
                }
                const float actual_lse = read_scalar<float>(actual_tensor, k_head_dim, h, qt, b);
                const float reference_lse = expected[row + k_head_dim];
                if (std::isfinite(reference_lse)) {
                    stats.valid = stats.valid && std::isfinite(actual_lse);
                    if (std::isfinite(actual_lse)) {
                        stats.lse_max_abs = std::max(stats.lse_max_abs, std::fabs(actual_lse - reference_lse));
                    }
                } else {
                    stats.valid = stats.valid && std::isinf(actual_lse) && std::signbit(actual_lse);
                }
            }
        }
    }
    return stats;
}

diff_stats compare_vectors(
    const attention_case & c,
    const std::vector<float> & actual,
    const std::vector<float> & expected) {
    GGML_ASSERT(actual.size() == expected.size());
    ggml_tensor tensor = {};
    tensor.type = GGML_TYPE_F32;
    tensor.ne[0] = k_head_dim + 1;
    tensor.ne[1] = c.heads;
    tensor.ne[2] = c.lq;
    tensor.ne[3] = c.batch;
    tensor.nb[0] = sizeof(float);
    tensor.nb[1] = tensor.nb[0] * tensor.ne[0];
    tensor.nb[2] = tensor.nb[1] * tensor.ne[1];
    tensor.nb[3] = tensor.nb[2] * tensor.ne[2];
    tensor.data = const_cast<float *>(actual.data());
    return compare_packed(c, &tensor, expected);
}

generated_inputs batch_slice_inputs(
    const attention_case & c,
    const generated_inputs & inputs,
    int64_t batch) {
    generated_inputs result;
    const size_t q_count = static_cast<size_t>(k_head_dim * c.lq * c.heads);
    const size_t kv_count = static_cast<size_t>(k_head_dim * c.lk * c.kv_heads);
    result.q.assign(inputs.q.begin() + batch * q_count, inputs.q.begin() + (batch + 1) * q_count);
    result.k.assign(inputs.k.begin() + batch * kv_count, inputs.k.begin() + (batch + 1) * kv_count);
    result.v.assign(inputs.v.begin() + batch * kv_count, inputs.v.begin() + (batch + 1) * kv_count);
    return result;
}

std::vector<float> run_htp_packed(
    ggml_backend_t htp,
    const attention_case & source_case,
    const generated_inputs & source_inputs,
    const mask_data & sparse_mask,
    bool dense) {
    attention_case c = source_case;
    c.batch = 1;
    const ggml_init_params params = {
        /* .mem_size   = */ 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    GGML_ASSERT(ctx != nullptr);
    ggml_tensor * q_storage = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, k_head_dim, c.heads, c.lq, 1);
    ggml_tensor * k_storage = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, k_head_dim, c.kv_heads, c.lk, 1);
    ggml_tensor * v_storage = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, k_head_dim, c.kv_heads, c.lk, 1);
    ggml_tensor * q = ggml_view_4d(
        ctx, q_storage, k_head_dim, c.lq, c.heads, 1,
        q_storage->nb[2], q_storage->nb[1], q_storage->nb[3], 0);
    ggml_tensor * k = ggml_view_4d(
        ctx, k_storage, k_head_dim, c.lk, c.kv_heads, 1,
        k_storage->nb[2], k_storage->nb[1], k_storage->nb[3], 0);
    ggml_tensor * v = ggml_view_4d(
        ctx, v_storage, k_head_dim, c.lk, c.kv_heads, 1,
        v_storage->nb[2], v_storage->nb[1], v_storage->nb[3], 0);
    q->op = GGML_OP_NONE;
    k->op = GGML_OP_NONE;
    v->op = GGML_OP_NONE;

    std::vector<float> q_physical(source_inputs.q.size());
    std::vector<ggml_fp16_t> k_physical(source_inputs.k.size());
    std::vector<ggml_fp16_t> v_physical(source_inputs.v.size());
    for (int64_t h = 0; h < c.heads; ++h) {
        for (int64_t qt = 0; qt < c.lq; ++qt) {
            for (int64_t d = 0; d < k_head_dim; ++d) {
                q_physical[static_cast<size_t>(d + k_head_dim * (h + c.heads * qt))] =
                    source_inputs.q[q_index(c, d, qt, h, 0)];
            }
        }
    }
    for (int64_t h = 0; h < c.kv_heads; ++h) {
        for (int64_t kt = 0; kt < c.lk; ++kt) {
            for (int64_t d = 0; d < k_head_dim; ++d) {
                const size_t physical = static_cast<size_t>(d + k_head_dim * (h + c.kv_heads * kt));
                const size_t logical = kv_index(c, d, kt, h, 0);
                k_physical[physical] = source_inputs.k[logical];
                v_physical[physical] = source_inputs.v[logical];
            }
        }
    }

    std::vector<ggml_fp16_t> dense_mask;
    ggml_tensor * mask = nullptr;
    if (dense) {
        const int64_t padded_lk = ((c.lk + 63) / 64) * 64;
        dense_mask.assign(
            static_cast<size_t>(padded_lk * c.lq),
            ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity()));
        for (int64_t qt = 0; qt < c.lq; ++qt) {
            std::fill_n(dense_mask.data() + qt * padded_lk, c.lk, ggml_fp32_to_fp16(0.0f));
        }
        mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, padded_lk, c.lq, 1, 1);
    } else {
        mask = ggml_new_tensor_4d(ctx, GGML_TYPE_I32, sparse_mask.ne0, sparse_mask.ne1, 1, 1);
    }
    ggml_tensor * packed = ggml_flash_attn_ext_with_lse(
        ctx, q, k, v, mask, 1.0f / std::sqrt(float(k_head_dim)), 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(packed, GGML_PREC_F32);
    if (!dense) {
        ggml_flash_attn_ext_set_block_sparse(packed, k_block_size, k_block_size, c.format);
    }
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, packed);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors_from_buft(
        ctx, ggml_backend_get_default_buffer_type(htp));
    GGML_ASSERT(buffer != nullptr);
    ggml_backend_tensor_set(q_storage, q_physical.data(), 0, q_physical.size() * sizeof(float));
    ggml_backend_tensor_set(k_storage, k_physical.data(), 0, k_physical.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(v_storage, v_physical.data(), 0, v_physical.size() * sizeof(ggml_fp16_t));
    if (dense) {
        ggml_backend_tensor_set(mask, dense_mask.data(), 0, dense_mask.size() * sizeof(ggml_fp16_t));
    } else {
        ggml_backend_tensor_set(mask, sparse_mask.values.data(), 0, sparse_mask.values.size() * sizeof(int32_t));
    }
    GGML_ASSERT(packed->buffer != nullptr && ggml_is_contiguous(packed));
    GGML_ASSERT(q->view_src == q_storage && q_storage->buffer != nullptr && ggml_is_contiguous(q_storage));
    GGML_ASSERT(k->view_src == k_storage && k_storage->buffer != nullptr && ggml_is_contiguous(k_storage));
    GGML_ASSERT(v->view_src == v_storage && v_storage->buffer != nullptr && ggml_is_contiguous(v_storage));
    GGML_ASSERT(mask->buffer != nullptr && ggml_is_contiguous(mask));
    GGML_ASSERT(ggml_backend_buft_is_rpcmem(ggml_backend_buffer_get_type(packed->buffer)));
    GGML_ASSERT(ggml_backend_buft_is_rpcmem(ggml_backend_buffer_get_type(q_storage->buffer)));
    GGML_ASSERT(ggml_backend_buft_is_rpcmem(ggml_backend_buffer_get_type(mask->buffer)));
    GGML_ASSERT(ggml_backend_supports_op(htp, packed));
    const bool hybrid_was_enabled = ggml_backend_htp_get_hybrid_enabled();
    ggml_backend_htp_set_hybrid_enabled(true);
    const ggml_status status = ggml_backend_graph_compute(htp, graph);
    ggml_backend_htp_set_hybrid_enabled(hybrid_was_enabled);
    if (status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "%s HTP %s packed compute failed: %d\n",
                     c.name, dense ? "dense" : "sparse", int(status));
    }
    GGML_ASSERT(status == GGML_STATUS_SUCCESS);
    std::vector<float> result(static_cast<size_t>(ggml_nelements(packed)));
    ggml_backend_tensor_get(packed, result.data(), 0, result.size() * sizeof(float));
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return result;
}

void run_case(
    ggml_opencl_htp_sidecar * sidecar,
    ggml_backend_t htp,
    const attention_case & c,
    bool check_preconditions) {
    GGML_ASSERT(c.heads % c.kv_heads == 0);
    const float scale = 1.0f / std::sqrt(float(k_head_dim));
    const mask_data sparse_mask = make_mask(c);
    const generated_inputs inputs = make_inputs(c);
    const std::vector<float> reference = reference_attention(c, inputs, sparse_mask, scale);
    const ggml_init_params params = {
        /* .mem_size   = */ 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    GGML_ASSERT(ctx != nullptr);
    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, k_head_dim, c.lq, c.heads, c.batch);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, k_head_dim, c.lk, c.kv_heads, c.batch);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, k_head_dim, c.lk, c.kv_heads, c.batch);
    if (c.permuted_kv) {
        k->nb[0] = sizeof(ggml_fp16_t);
        k->nb[2] = k_head_dim * k->nb[0];
        k->nb[1] = c.kv_heads * k->nb[2];
        k->nb[3] = c.lk * k->nb[1];
        std::copy(std::begin(k->nb), std::end(k->nb), std::begin(v->nb));
    }
    ggml_tensor * mask = ggml_new_tensor_4d(
        ctx, GGML_TYPE_I32, sparse_mask.ne0, sparse_mask.ne1, 1, 1);
    ggml_tensor * packed = ggml_flash_attn_ext_with_lse(ctx, q, k, v, mask, scale, 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(packed, GGML_PREC_F32);
    ggml_flash_attn_ext_set_block_sparse(packed, k_block_size, k_block_size, c.format);

    const size_t q_offset = c.nonzero_offsets ? 64 : 0;
    const size_t k_offset = c.nonzero_offsets ? 96 : 0;
    const size_t v_offset = c.nonzero_offsets ? 128 : 0;
    const size_t mask_offset = c.nonzero_offsets ? 160 : 0;
    const size_t output_offset = c.nonzero_offsets ? 192 : 0;
    std::vector<owned_rpcmem_buffer> buffers;
    buffers.push_back(allocate_tensor_buffer(htp, q, q_offset));
    buffers.push_back(allocate_tensor_buffer(htp, k, k_offset));
    buffers.push_back(allocate_tensor_buffer(htp, v, v_offset));
    buffers.push_back(allocate_tensor_buffer(htp, mask, mask_offset));
    buffers.push_back(allocate_tensor_buffer(htp, packed, output_offset));
    fill_tensors(c, inputs, sparse_mask, q, k, v, mask);
    const std::vector<uint8_t> q_snapshot = snapshot(buffers[0]);
    const std::vector<uint8_t> k_snapshot = snapshot(buffers[1]);
    const std::vector<uint8_t> v_snapshot = snapshot(buffers[2]);
    const std::vector<uint8_t> mask_snapshot = snapshot(buffers[3]);
    const std::vector<uint8_t> output_snapshot = snapshot(buffers[4]);

    char reason[1024] = {};
    ggml_cgraph * cgraph = ggml_new_graph(ctx);
    ggml_graph_add_node(cgraph, packed);
    std::vector<ggml_opencl_htp_import_descriptor> descriptors;
    for (const owned_rpcmem_buffer & buffer : buffers) {
        descriptors.push_back(make_import_descriptor(buffer));
    }
    descriptors.push_back(descriptors.front());

    auto expect_preflight_unsupported = [&] (
        const char * name,
        ggml_cgraph * rejected_cgraph,
        const std::vector<ggml_opencl_htp_import_descriptor> & rejected_descriptors,
        const char * expected_reason = nullptr) {
        ggml_opencl_htp_graph * rejected_graph = nullptr;
        reason[0] = '\0';
        const ggml_opencl_htp_result result = ggml_backend_opencl_htp_graph_preflight(
            sidecar, rejected_cgraph,
            rejected_descriptors.empty() ? nullptr : rejected_descriptors.data(),
            rejected_descriptors.size(), &rejected_graph, reason, sizeof(reason));
        if (result != GGML_OPENCL_HTP_UNSUPPORTED || rejected_graph != nullptr || reason[0] == '\0') {
            std::fprintf(
                stderr, "%s preflight rejection failed: result=%d graph=%p reason=%s\n",
                name, result, static_cast<void *>(rejected_graph), reason);
        }
        GGML_ASSERT(result == GGML_OPENCL_HTP_UNSUPPORTED);
        GGML_ASSERT(rejected_graph == nullptr);
        GGML_ASSERT(reason[0] != '\0');
        GGML_ASSERT(expected_reason == nullptr || std::strstr(reason, expected_reason) != nullptr);
        GGML_ASSERT(snapshot(buffers[0]) == q_snapshot);
        GGML_ASSERT(snapshot(buffers[1]) == k_snapshot);
        GGML_ASSERT(snapshot(buffers[2]) == v_snapshot);
        GGML_ASSERT(snapshot(buffers[3]) == mask_snapshot);
        GGML_ASSERT(snapshot(buffers[4]) == output_snapshot);
        for (const owned_rpcmem_buffer & buffer : buffers) {
            GGML_ASSERT(fcntl(buffer.info.fd, F_GETFD) != -1);
        }
    };

    if (check_preconditions) {
        GGML_ASSERT(setenv("SD_HTP_SPARSE_OPENCL", "0", 1) == 0);
        expect_preflight_unsupported("feature disabled", cgraph, descriptors, "SD_HTP_SPARSE_OPENCL=1");
        GGML_ASSERT(setenv("SD_HTP_SPARSE_OPENCL", "1", 1) == 0);

        ggml_cgraph * no_candidate_graph = ggml_new_graph(ctx);
        ggml_graph_add_node(no_candidate_graph, q);
        expect_preflight_unsupported("no candidate", no_candidate_graph, descriptors, "no eligible");

        ggml_tensor i32_no_lse = *packed;
        i32_no_lse.op_params[6] = 0;
        ggml_cgraph * no_lse_graph = ggml_new_graph(ctx);
        ggml_graph_add_node(no_lse_graph, &i32_no_lse);
        expect_preflight_unsupported("I32 without LSE", no_lse_graph, descriptors, "no eligible");

        std::vector<ggml_opencl_htp_import_descriptor> missing_descriptor = descriptors;
        missing_descriptor.erase(missing_descriptor.begin() + 4);
        expect_preflight_unsupported("missing descriptor", cgraph, missing_descriptor, "missing RPCMEM descriptor");

        std::vector<ggml_opencl_htp_import_descriptor> invalid_descriptor = descriptors;
        invalid_descriptor[4].fd = -1;
        expect_preflight_unsupported("invalid descriptor", cgraph, invalid_descriptor, "invalid");

        std::vector<ggml_opencl_htp_import_descriptor> inconsistent_duplicate = descriptors;
        --inconsistent_duplicate.back().logical_size;
        expect_preflight_unsupported(
            "inconsistent duplicate", cgraph, inconsistent_duplicate, "inconsistent");

        std::vector<ggml_opencl_htp_import_descriptor> misaligned_base = descriptors;
        misaligned_base[4].base = static_cast<uint8_t *>(misaligned_base[4].base) + 1;
        expect_preflight_unsupported("misaligned base", cgraph, misaligned_base, "not aligned");

        std::vector<ggml_opencl_htp_import_descriptor> zero_logical_size = descriptors;
        zero_logical_size[4].logical_size = 0;
        expect_preflight_unsupported("zero logical size", cgraph, zero_logical_size, "nonzero");

        std::vector<ggml_opencl_htp_import_descriptor> short_allocation = descriptors;
        short_allocation[4].allocation_size = short_allocation[4].logical_size - 1;
        expect_preflight_unsupported("short allocation", cgraph, short_allocation, "smaller than logical");

        std::vector<ggml_opencl_htp_import_descriptor> short_fd = descriptors;
        short_fd[4].allocation_size += 4096;
        expect_preflight_unsupported("short fd", cgraph, short_fd, "fd");

        std::vector<ggml_opencl_htp_import_descriptor> oversized_descriptor = descriptors;
        oversized_descriptor[4].allocation_size = SIZE_MAX;
        expect_preflight_unsupported("oversized allocation", cgraph, oversized_descriptor, "OpenCL maximum");

        ggml_tensor invalid_mask = *mask;
        ++invalid_mask.ne[0];
        ggml_tensor invalid_shape = *packed;
        invalid_shape.src[3] = &invalid_mask;
        ggml_tensor predicate_miss = *packed;
        predicate_miss.op_params[7] = GGML_FLASH_ATTN_EXT_MASK_DEFAULT;
        ggml_cgraph * predicate_miss_graph = ggml_new_graph(ctx);
        ggml_graph_add_node(predicate_miss_graph, packed);
        ggml_graph_add_node(predicate_miss_graph, &predicate_miss);
        ggml_opencl_htp_graph * exact_predicate_session = nullptr;
        GGML_ASSERT(ggml_backend_opencl_htp_graph_preflight(
            sidecar, predicate_miss_graph, descriptors.data(), descriptors.size(),
            &exact_predicate_session, reason, sizeof(reason)) == GGML_OPENCL_HTP_OK);
        GGML_ASSERT(exact_predicate_session != nullptr);
        GGML_ASSERT(ggml_backend_opencl_htp_graph_routed_node_count(exact_predicate_session) == 1);
        GGML_ASSERT(ggml_backend_opencl_htp_graph_routes_node(exact_predicate_session, packed));
        GGML_ASSERT(!ggml_backend_opencl_htp_graph_routes_node(exact_predicate_session, &predicate_miss));
        GGML_ASSERT(ggml_backend_opencl_htp_graph_live_alias_count(exact_predicate_session) == 5);
        GGML_ASSERT(ggml_backend_opencl_htp_graph_live_event_count(exact_predicate_session) == 0);
        ggml_backend_opencl_htp_graph_free(&exact_predicate_session);
        GGML_ASSERT(snapshot(buffers[4]) == output_snapshot);

        ggml_cgraph * invalid_shape_graph = ggml_new_graph(ctx);
        ggml_graph_add_node(invalid_shape_graph, packed);
        ggml_graph_add_node(invalid_shape_graph, &invalid_shape);
        expect_preflight_unsupported(
            "eligible node with invalid mask shape", invalid_shape_graph, descriptors, "unsupported");

        ggml_tensor invalid_layout_k = *k;
        invalid_layout_k.nb[1] += sizeof(ggml_fp16_t);
        invalid_layout_k.nb[2] = invalid_layout_k.nb[1] * static_cast<size_t>(invalid_layout_k.ne[1]);
        invalid_layout_k.nb[3] = invalid_layout_k.nb[2] * static_cast<size_t>(invalid_layout_k.ne[2]);
        ggml_tensor invalid_layout = *packed;
        invalid_layout.src[1] = &invalid_layout_k;
        ggml_cgraph * invalid_layout_graph = ggml_new_graph(ctx);
        ggml_graph_add_node(invalid_layout_graph, &invalid_layout);
        expect_preflight_unsupported("invalid K layout", invalid_layout_graph, descriptors, "unsupported");

        ggml_tensor invalid_span = *packed;
        invalid_span.data = static_cast<uint8_t *>(buffers[4].info.base) +
                            buffers[4].info.logical_size - sizeof(float);
        ggml_cgraph * invalid_span_graph = ggml_new_graph(ctx);
        ggml_graph_add_node(invalid_span_graph, &invalid_span);
        expect_preflight_unsupported("invalid output span", invalid_span_graph, descriptors, "binding preflight");

        ggml_tensor overlapping_q = *q;
        overlapping_q.buffer = packed->buffer;
        overlapping_q.data = packed->data;
        ggml_tensor input_overlap = *packed;
        input_overlap.src[0] = &overlapping_q;
        ggml_cgraph * input_overlap_graph = ggml_new_graph(ctx);
        ggml_graph_add_node(input_overlap_graph, &input_overlap);
        expect_preflight_unsupported(
            "output/input overlap", input_overlap_graph, descriptors, "overlaps an input span");

        ggml_tensor overlapping_output = *packed;
        ggml_cgraph * overlapping_outputs_graph = ggml_new_graph(ctx);
        ggml_graph_add_node(overlapping_outputs_graph, packed);
        ggml_graph_add_node(overlapping_outputs_graph, &overlapping_output);
        expect_preflight_unsupported(
            "cross-output overlap", overlapping_outputs_graph, descriptors, "output spans overlap");

        ggml_tensor * second_output = ggml_flash_attn_ext_with_lse(
            ctx, q, k, v, mask, scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(second_output, GGML_PREC_F32);
        ggml_flash_attn_ext_set_block_sparse(
            second_output, k_block_size, k_block_size, c.format);
        owned_rpcmem_buffer second_buffer = allocate_tensor_buffer(htp, second_output, 256);
        const std::vector<uint8_t> second_snapshot = snapshot(second_buffer);
        std::vector<ggml_opencl_htp_import_descriptor> two_node_descriptors = descriptors;
        two_node_descriptors.push_back(make_import_descriptor(second_buffer));
        ggml_cgraph * two_node_graph = ggml_new_graph(ctx);
        ggml_graph_add_node(two_node_graph, packed);
        ggml_graph_add_node(two_node_graph, second_output);
        ggml_opencl_htp_graph * two_node_session = nullptr;
        GGML_ASSERT(ggml_backend_opencl_htp_graph_preflight(
            sidecar, two_node_graph, two_node_descriptors.data(), two_node_descriptors.size(),
            &two_node_session, reason, sizeof(reason)) == GGML_OPENCL_HTP_OK);
        GGML_ASSERT(two_node_session != nullptr);
        GGML_ASSERT(ggml_backend_opencl_htp_graph_routed_node_count(two_node_session) == 2);
        GGML_ASSERT(ggml_backend_opencl_htp_graph_routes_node(two_node_session, packed));
        GGML_ASSERT(ggml_backend_opencl_htp_graph_routes_node(two_node_session, second_output));
        GGML_ASSERT(ggml_backend_opencl_htp_graph_live_alias_count(two_node_session) == 6);
        GGML_ASSERT(ggml_backend_opencl_htp_graph_live_event_count(two_node_session) == 0);
        GGML_ASSERT(snapshot(buffers[4]) == output_snapshot);
        GGML_ASSERT(snapshot(second_buffer) == second_snapshot);
        ggml_backend_opencl_htp_graph_free(&two_node_session);
        GGML_ASSERT(fcntl(second_buffer.info.fd, F_GETFD) != -1);
        ggml_backend_buffer_free(second_buffer.buffer);
    }

    ggml_opencl_htp_graph * graph = nullptr;
    GGML_ASSERT(ggml_backend_opencl_htp_graph_preflight(
        sidecar, cgraph, descriptors.data(), descriptors.size(),
        &graph, reason, sizeof(reason)) == GGML_OPENCL_HTP_OK);
    GGML_ASSERT(graph != nullptr);
    GGML_ASSERT(ggml_backend_opencl_htp_graph_routed_node_count(graph) == 1);
    GGML_ASSERT(ggml_backend_opencl_htp_graph_routes_node(graph, packed));
    GGML_ASSERT(!ggml_backend_opencl_htp_graph_routes_node(graph, q));
    ggml_tensor equivalent_but_unfrozen = *packed;
    GGML_ASSERT(!ggml_backend_opencl_htp_graph_routes_node(graph, &equivalent_but_unfrozen));
    GGML_ASSERT(ggml_backend_opencl_htp_graph_live_alias_count(graph) == 5);
    GGML_ASSERT(ggml_backend_opencl_htp_graph_live_event_count(graph) == 0);
    GGML_ASSERT(snapshot(buffers[4]) == output_snapshot);
    auto expect_pre_enqueue_unsupported = [&](
        const char * name,
        const ggml_tensor * node,
        size_t expected_alias_count) {
        reason[0] = '\0';
        const ggml_opencl_htp_result rejected =
            ggml_backend_opencl_htp_graph_compute_block_sparse_attn_sync(
                graph, node, reason, sizeof(reason));
        const bool output_unchanged = snapshot(buffers[4]) == output_snapshot;
        if (rejected != GGML_OPENCL_HTP_UNSUPPORTED || reason[0] == '\0' ||
            ggml_backend_opencl_htp_graph_live_alias_count(graph) != expected_alias_count ||
            ggml_backend_opencl_htp_graph_live_event_count(graph) != 0 || !output_unchanged) {
            std::fprintf(
                stderr,
                "%s pre-enqueue rejection failed: result=%d aliases=%zu events=%zu output_unchanged=%s reason=%s\n",
                name,
                rejected,
                ggml_backend_opencl_htp_graph_live_alias_count(graph),
                ggml_backend_opencl_htp_graph_live_event_count(graph),
                output_unchanged ? "yes" : "no",
                reason);
        }
        GGML_ASSERT(rejected == GGML_OPENCL_HTP_UNSUPPORTED);
        GGML_ASSERT(reason[0] != '\0');
        GGML_ASSERT(ggml_backend_opencl_htp_graph_live_alias_count(graph) == expected_alias_count);
        GGML_ASSERT(ggml_backend_opencl_htp_graph_live_event_count(graph) == 0);
        GGML_ASSERT(output_unchanged);
    };
    if (check_preconditions) {
        GGML_ASSERT(ggml_backend_opencl_htp_graph_live_alias_count(graph) == 5);

        ggml_tensor unsupported_op = *packed;
        unsupported_op.op = GGML_OP_ADD;
        expect_pre_enqueue_unsupported("unsupported op", &unsupported_op, 5);

        ggml_tensor overlapping_q = *q;
        overlapping_q.buffer = packed->buffer;
        overlapping_q.data = packed->data;
        ggml_tensor overlapping_input = *packed;
        overlapping_input.src[0] = &overlapping_q;
        expect_pre_enqueue_unsupported("output/input overlap", &overlapping_input, 5);

        ggml_tensor overlapping_rows = *packed;
        overlapping_rows.nb[2] = overlapping_rows.nb[0];
        expect_pre_enqueue_unsupported("overlapping output rows", &overlapping_rows, 5);

        auto set_row_stride = [](ggml_tensor * tensor, size_t row_stride) {
            tensor->nb[1] = row_stride;
            tensor->nb[2] = row_stride * static_cast<size_t>(tensor->ne[1]);
            tensor->nb[3] = tensor->nb[2] * static_cast<size_t>(tensor->ne[2]);
        };
        const size_t kv_row_bytes = static_cast<size_t>(k_head_dim) * sizeof(ggml_fp16_t);

        ggml_tensor gapped_k = *k;
        set_row_stride(&gapped_k, kv_row_bytes + sizeof(ggml_fp16_t));
        ggml_tensor gapped_k_output = *packed;
        gapped_k_output.src[1] = &gapped_k;
        expect_pre_enqueue_unsupported("gapped K strides", &gapped_k_output, 5);

        ggml_tensor overlapping_k = *k;
        set_row_stride(&overlapping_k, kv_row_bytes - sizeof(ggml_fp16_t));
        ggml_tensor overlapping_k_output = *packed;
        overlapping_k_output.src[1] = &overlapping_k;
        expect_pre_enqueue_unsupported("overlapping K strides", &overlapping_k_output, 5);

        ggml_tensor gapped_v = *v;
        set_row_stride(&gapped_v, kv_row_bytes + sizeof(ggml_fp16_t));
        ggml_tensor gapped_v_output = *packed;
        gapped_v_output.src[2] = &gapped_v;
        expect_pre_enqueue_unsupported("gapped V strides", &gapped_v_output, 5);

        ggml_tensor overlapping_v = *v;
        set_row_stride(&overlapping_v, kv_row_bytes - sizeof(ggml_fp16_t));
        ggml_tensor overlapping_v_output = *packed;
        overlapping_v_output.src[2] = &overlapping_v;
        expect_pre_enqueue_unsupported("overlapping V strides", &overlapping_v_output, 5);
    }

    const ggml_opencl_htp_result result =
        ggml_backend_opencl_htp_graph_compute_block_sparse_attn_sync(
            graph, packed, reason, sizeof(reason));
    if (result != GGML_OPENCL_HTP_OK) {
        std::fprintf(stderr, "%s compute failed: %s\n", c.name, reason);
    }
    GGML_ASSERT(result == GGML_OPENCL_HTP_OK);
    GGML_ASSERT(snapshot(buffers[0]) == q_snapshot);
    GGML_ASSERT(snapshot(buffers[1]) == k_snapshot);
    GGML_ASSERT(snapshot(buffers[2]) == v_snapshot);
    GGML_ASSERT(snapshot(buffers[3]) == mask_snapshot);
    assert_output_guards(c, packed, buffers[4]);

    const diff_stats cpu_stats = compare_packed(c, packed, reference);
    std::printf(
        "[OPENCL_BLOCK_SPARSE] case=%s opencl_cpu_out=%.9g opencl_cpu_lse=%.9g valid=%s guards=yes",
        c.name, cpu_stats.out_max_abs, cpu_stats.lse_max_abs, cpu_stats.valid ? "yes" : "no");
    GGML_ASSERT(cpu_stats.valid);
    GGML_ASSERT(cpu_stats.out_max_abs <= 3e-3f);
    GGML_ASSERT(cpu_stats.lse_max_abs <= 5e-3f);
    if (c.pattern == mask_pattern::empty) {
        for (int64_t b = 0; b < c.batch; ++b) {
            for (int64_t qt = 0; qt < c.lq; ++qt) {
                for (int64_t h = 0; h < c.heads; ++h) {
                    for (int64_t d = 0; d < k_head_dim; ++d) {
                        GGML_ASSERT(read_scalar<float>(packed, d, h, qt, b) == 0.0f);
                    }
                    const float lse = read_scalar<float>(packed, k_head_dim, h, qt, b);
                    GGML_ASSERT(std::isinf(lse) && std::signbit(lse));
                }
            }
        }
    }

    if (c.compare_htp_sparse || c.compare_htp_dense) {
        std::vector<float> htp_sparse_all;
        std::vector<float> htp_dense_all;
        for (int64_t b = 0; b < c.batch; ++b) {
            const generated_inputs one_batch_inputs = batch_slice_inputs(c, inputs, b);
            if (c.compare_htp_sparse) {
                const std::vector<float> one = run_htp_packed(htp, c, one_batch_inputs, sparse_mask, false);
                htp_sparse_all.insert(htp_sparse_all.end(), one.begin(), one.end());
            }
            if (c.compare_htp_dense) {
                const std::vector<float> one = run_htp_packed(htp, c, one_batch_inputs, sparse_mask, true);
                htp_dense_all.insert(htp_dense_all.end(), one.begin(), one.end());
            }
        }
        if (c.compare_htp_sparse) {
            const diff_stats htp_cpu = compare_vectors(c, htp_sparse_all, reference);
            const diff_stats opencl_htp = compare_packed(c, packed, htp_sparse_all);
            std::printf(
                " htp_sparse_cpu_out=%.9g htp_sparse_cpu_lse=%.9g"
                " opencl_htp_sparse_out=%.9g opencl_htp_sparse_lse=%.9g",
                htp_cpu.out_max_abs, htp_cpu.lse_max_abs,
                opencl_htp.out_max_abs, opencl_htp.lse_max_abs);
            GGML_ASSERT(htp_cpu.valid && htp_cpu.out_max_abs <= 3e-3f && htp_cpu.lse_max_abs <= 5e-3f);
            GGML_ASSERT(opencl_htp.valid && opencl_htp.out_max_abs <= 3e-3f && opencl_htp.lse_max_abs <= 5e-3f);
        }
        if (c.compare_htp_dense) {
            const diff_stats htp_cpu = compare_vectors(c, htp_dense_all, reference);
            const diff_stats opencl_htp = compare_packed(c, packed, htp_dense_all);
            float rho_max_abs = 0.0f;
            for (int64_t b = 0; b < c.batch; ++b) {
                for (int64_t qt = 0; qt < c.lq; ++qt) {
                    for (int64_t h = 0; h < c.heads; ++h) {
                        const size_t lse = packed_index(c, k_head_dim, h, qt, b);
                        rho_max_abs = std::max(
                            rho_max_abs,
                            std::fabs(std::exp(read_scalar<float>(packed, k_head_dim, h, qt, b) -
                                                    htp_dense_all[lse]) - 1.0f));
                    }
                }
            }
            std::printf(
                " htp_dense_cpu_out=%.9g htp_dense_cpu_lse=%.9g"
                " opencl_htp_dense_out=%.9g opencl_htp_dense_lse=%.9g rho=%.9g",
                htp_cpu.out_max_abs, htp_cpu.lse_max_abs,
                opencl_htp.out_max_abs, opencl_htp.lse_max_abs, rho_max_abs);
            GGML_ASSERT(htp_cpu.valid && htp_cpu.out_max_abs <= 3e-3f && htp_cpu.lse_max_abs <= 5e-3f);
            GGML_ASSERT(opencl_htp.valid && opencl_htp.out_max_abs <= 3e-3f && opencl_htp.lse_max_abs <= 5e-3f);
            GGML_ASSERT(rho_max_abs <= 5e-3f);
        }
    }
    if (c.heads != c.kv_heads) {
        // Existing HTP sparse-index GQA coverage is diagnostic-only, so CPU is the correctness oracle here.
        std::printf(" htp_gqa_oracle=skipped_existing_diagnostic_only");
    }
    std::printf("\n");

    GGML_ASSERT(ggml_backend_opencl_htp_graph_finish(
        graph, reason, sizeof(reason)) == GGML_OPENCL_HTP_OK);
    GGML_ASSERT(ggml_backend_opencl_htp_graph_routed_node_count(graph) == 0);
    GGML_ASSERT(!ggml_backend_opencl_htp_graph_routes_node(graph, packed));
    ggml_backend_opencl_htp_graph_free(&graph);
    ggml_free(ctx);
    for (auto it = buffers.rbegin(); it != buffers.rend(); ++it) {
        GGML_ASSERT(fcntl(it->info.fd, F_GETFD) != -1);
        ggml_backend_buffer_free(it->buffer);
    }
}

} // namespace

int main() {
    GGML_ASSERT(setenv("SD_HTP_SPARSE_OPENCL", "1", 1) == 0);
    GGML_ASSERT(setenv("SD_HTP_OPS", "1", 1) == 0);
    GGML_ASSERT(setenv("SD_HTP_COMPUTE", "1", 1) == 0);
    GGML_ASSERT(setenv("SD_HTP_ATTENTION", "1", 1) == 0);

    ggml_opencl_qcom_requirements requirements = {};
    char reason[1024] = {};
    ggml_opencl_htp_sidecar * sidecar =
        ggml_backend_opencl_htp_sidecar_create(&requirements, reason, sizeof(reason));
    if (sidecar == nullptr) {
        std::fprintf(stderr, "OpenCL HTP sidecar creation failed: %s\n", reason);
        return 1;
    }
    ggml_backend_t htp = ggml_backend_htp_init();
    GGML_ASSERT(htp != nullptr && ggml_backend_htp_runtime_available());
    ggml_backend_t opencl = ggml_backend_opencl_init();
    GGML_ASSERT(opencl != nullptr);
    test_ordinary_opencl_support_gating(opencl);

    const attention_case cases[] = {
        { "matrix_full_keep", 64, 96, 4, 4, 1,
          GGML_FLASH_ATTN_EXT_MASK_BLOCK_MATRIX, mask_pattern::full,
          false, false, true, true },
        { "matrix_sparse", 64, 96, 4, 4, 1,
          GGML_FLASH_ATTN_EXT_MASK_BLOCK_MATRIX, mask_pattern::sparse_terminated,
          false, false, false, false },
        { "matrix_empty", 32, 32, 1, 1, 1,
          GGML_FLASH_ATTN_EXT_MASK_BLOCK_MATRIX, mask_pattern::empty,
          false, false, false, false },
        { "index_empty", 33, 65, 2, 2, 1,
          GGML_FLASH_ATTN_EXT_MASK_BLOCK_INDEX, mask_pattern::empty,
          false, false, false, false },
        { "index_full_keep_tail", 65, 97, 4, 4, 1,
          GGML_FLASH_ATTN_EXT_MASK_BLOCK_INDEX, mask_pattern::full,
          false, false, true, true },
        { "index_sparse_terminated", 65, 97, 4, 4, 1,
          GGML_FLASH_ATTN_EXT_MASK_BLOCK_INDEX, mask_pattern::sparse_terminated,
          false, false, true, false },
        { "index_invalid_ids_gqa", 65, 97, 4, 2, 1,
          GGML_FLASH_ATTN_EXT_MASK_BLOCK_INDEX, mask_pattern::invalid_ids,
          false, false, false, false },
        { "gqa_4_to_1", 65, 97, 4, 1, 1,
          GGML_FLASH_ATTN_EXT_MASK_BLOCK_INDEX, mask_pattern::sparse_terminated,
          false, false, false, false },
        { "batch2_offsets_permuted", 65, 97, 4, 4, 2,
          GGML_FLASH_ATTN_EXT_MASK_BLOCK_INDEX, mask_pattern::sparse_terminated,
          true, true, true, false },
        { "production_410x24_tail", 410, 410, 24, 24, 1,
          GGML_FLASH_ATTN_EXT_MASK_BLOCK_INDEX, mask_pattern::sparse_terminated,
          false, false, false, false },
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        run_case(sidecar, htp, cases[i], i == 0 || i == 3);
    }

    ggml_backend_free(opencl);
    ggml_backend_free(htp);
    ggml_backend_opencl_htp_sidecar_free(sidecar);
    return 0;
}
