#include "ggml-htp.h"
#include "ggml-opencl-htp.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace {

constexpr int64_t k_head_dim = 64;
constexpr int64_t k_block_size = 32;

struct owned_rpcmem_buffer {
    ggml_backend_buffer_t buffer = nullptr;
    ggml_backend_htp_rpcmem_info info = {};
};

owned_rpcmem_buffer allocate_tensor_buffer(ggml_backend_t htp, ggml_tensor * tensor) {
    owned_rpcmem_buffer owned;
    const size_t size = ggml_backend_buft_get_alloc_size(
        ggml_backend_get_default_buffer_type(htp), tensor);
    GGML_ASSERT(size >= ggml_nbytes(tensor));
    owned.buffer = ggml_backend_alloc_buffer(htp, size);
    GGML_ASSERT(owned.buffer != nullptr);
    GGML_ASSERT(ggml_backend_buffer_get_size(owned.buffer) == size);
    GGML_ASSERT(ggml_backend_htp_get_rpcmem_info(owned.buffer, &owned.info));
    GGML_ASSERT(owned.info.logical_size == size);
    GGML_ASSERT(ggml_backend_tensor_alloc(
        owned.buffer, tensor, owned.info.base) == GGML_STATUS_SUCCESS);
    return owned;
}

ggml_opencl_htp_import_descriptor make_import_descriptor(
    const owned_rpcmem_buffer & owned) {
    return {
        /* .backing         = */ owned.buffer,
        /* .base            = */ owned.info.base,
        /* .fd              = */ owned.info.fd,
        /* .logical_size    = */ owned.info.logical_size,
        /* .allocation_size = */ owned.info.allocation_size,
    };
}

void import_buffer(
    ggml_opencl_htp_graph * graph,
    const owned_rpcmem_buffer & owned,
    char * reason,
    size_t reason_capacity) {
    const ggml_opencl_htp_import_descriptor descriptor = make_import_descriptor(owned);
    const ggml_opencl_htp_result result = ggml_backend_opencl_htp_graph_import(
        graph, &descriptor, reason, reason_capacity);
    if (result != GGML_OPENCL_HTP_OK) {
        std::fprintf(stderr, "RPCMEM import failed: %s\n", reason);
    }
    GGML_ASSERT(result == GGML_OPENCL_HTP_OK);
}

void free_buffers(std::vector<owned_rpcmem_buffer> & buffers) {
    for (auto it = buffers.rbegin(); it != buffers.rend(); ++it) {
        ggml_backend_buffer_free(it->buffer);
    }
    buffers.clear();
}

size_t q_index(int64_t d, int64_t qt, int64_t h, int64_t b, int64_t lq, int64_t heads) {
    return static_cast<size_t>(d + k_head_dim * (qt + lq * (h + heads * b)));
}

size_t kv_index(int64_t d, int64_t kt, int64_t h, int64_t b, int64_t lk, int64_t heads) {
    return static_cast<size_t>(d + k_head_dim * (kt + lk * (h + heads * b)));
}

size_t packed_index(int64_t d, int64_t h, int64_t qt, int64_t b, int64_t lq, int64_t heads) {
    return static_cast<size_t>(d + (k_head_dim + 1) * (h + heads * (qt + lq * b)));
}

size_t mask_index(int64_t kb_abs, int64_t qb_abs, int64_t total_k_blocks) {
    return static_cast<size_t>(kb_abs + total_k_blocks * qb_abs);
}

struct attention_case {
    const char * name;
    int64_t lq;
    int64_t lk;
    int64_t heads;
    int64_t kv_heads;
    bool sparse;
    bool empty;
};

std::vector<int32_t> make_mask(const attention_case & test_case) {
    const int64_t q_blocks_per_head = (test_case.lq + k_block_size - 1) / k_block_size;
    const int64_t k_blocks_per_head = (test_case.lk + k_block_size - 1) / k_block_size;
    const int64_t total_q_blocks = test_case.heads * q_blocks_per_head;
    const int64_t total_k_blocks = test_case.kv_heads * k_blocks_per_head;
    std::vector<int32_t> mask(
        static_cast<size_t>(total_q_blocks * total_k_blocks),
        (test_case.sparse || test_case.empty) ? 1 : 0);

    if (test_case.sparse) {
        GGML_ASSERT(q_blocks_per_head == 2 && k_blocks_per_head == 3);
        GGML_ASSERT(test_case.heads == test_case.kv_heads);
        for (int64_t h = 0; h < test_case.heads; ++h) {
            const int64_t qb0 = h * q_blocks_per_head;
            const int64_t qb1 = qb0 + 1;
            mask[mask_index(h * k_blocks_per_head + 0, qb0, total_k_blocks)] = 0;
            mask[mask_index(h * k_blocks_per_head + 2, qb0, total_k_blocks)] = 0;
            mask[mask_index(h * k_blocks_per_head + 1, qb1, total_k_blocks)] = 0;
        }
    }
    return mask;
}

struct generated_inputs {
    std::vector<float> q;
    std::vector<ggml_fp16_t> k;
    std::vector<ggml_fp16_t> v;
    std::vector<float> k_f32;
    std::vector<float> v_f32;
};

generated_inputs make_inputs(const attention_case & test_case) {
    constexpr int64_t batch = 1;
    generated_inputs inputs;
    inputs.q.resize(static_cast<size_t>(k_head_dim * test_case.lq * test_case.heads * batch));
    inputs.k.resize(static_cast<size_t>(k_head_dim * test_case.lk * test_case.kv_heads * batch));
    inputs.v.resize(inputs.k.size());
    inputs.k_f32.resize(inputs.k.size());
    inputs.v_f32.resize(inputs.v.size());

    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t h = 0; h < test_case.heads; ++h) {
            for (int64_t qt = 0; qt < test_case.lq; ++qt) {
                for (int64_t d = 0; d < k_head_dim; ++d) {
                    const float phase = 0.017f * float((d + 1) * (qt + 1)) + 0.61f * float(h);
                    inputs.q[q_index(d, qt, h, b, test_case.lq, test_case.heads)] =
                        0.45f * std::sin(phase) + 0.05f * std::cos(0.13f * float(d + 3 * h));
                }
            }
        }
        for (int64_t h = 0; h < test_case.kv_heads; ++h) {
            for (int64_t kt = 0; kt < test_case.lk; ++kt) {
                for (int64_t d = 0; d < k_head_dim; ++d) {
                    const size_t index = kv_index(d, kt, h, b, test_case.lk, test_case.kv_heads);
                    const float k_value =
                        0.42f * std::cos(0.011f * float((d + 2) * (kt + 1)) + 0.83f * float(h)) +
                        0.025f * float(h);
                    const float v_value =
                        0.36f * std::sin(0.019f * float((d + 1) * (kt + 3)) + 0.97f * float(h)) +
                        0.4f * float(h);
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
    const attention_case & test_case,
    const generated_inputs & inputs,
    const std::vector<int32_t> & mask,
    float scale) {
    constexpr int64_t batch = 1;
    const int64_t q_blocks_per_head = (test_case.lq + k_block_size - 1) / k_block_size;
    const int64_t k_blocks_per_head = (test_case.lk + k_block_size - 1) / k_block_size;
    const int64_t total_k_blocks = test_case.kv_heads * k_blocks_per_head;
    const int64_t heads_per_kv = test_case.heads / test_case.kv_heads;
    std::vector<float> packed(
        static_cast<size_t>((k_head_dim + 1) * test_case.heads * test_case.lq * batch), 0.0f);
    std::vector<float> scores(static_cast<size_t>(test_case.lk));

    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t qt = 0; qt < test_case.lq; ++qt) {
            for (int64_t h = 0; h < test_case.heads; ++h) {
                const int64_t kv_head = h / heads_per_kv;
                const int64_t qb_abs = h * q_blocks_per_head + qt / k_block_size;
                float maximum = -std::numeric_limits<float>::infinity();
                std::fill(scores.begin(), scores.end(), -std::numeric_limits<float>::infinity());

                for (int64_t kt = 0; kt < test_case.lk; ++kt) {
                    const int64_t kb_abs = kv_head * k_blocks_per_head + kt / k_block_size;
                    if (mask[mask_index(kb_abs, qb_abs, total_k_blocks)] != 0) {
                        continue;
                    }
                    float dot = 0.0f;
                    for (int64_t d = 0; d < k_head_dim; ++d) {
                        dot += inputs.q[q_index(d, qt, h, b, test_case.lq, test_case.heads)] *
                               inputs.k_f32[kv_index(
                                   d, kt, kv_head, b, test_case.lk, test_case.kv_heads)];
                    }
                    const float score = dot * scale;
                    scores[static_cast<size_t>(kt)] = score;
                    maximum = std::max(maximum, score);
                }

                const size_t row = packed_index(0, h, qt, b, test_case.lq, test_case.heads);
                if (!std::isfinite(maximum)) {
                    packed[row + k_head_dim] = -std::numeric_limits<float>::infinity();
                    continue;
                }

                float sum = 0.0f;
                for (int64_t kt = 0; kt < test_case.lk; ++kt) {
                    const float score = scores[static_cast<size_t>(kt)];
                    if (!std::isfinite(score)) {
                        continue;
                    }
                    const float weight = std::exp(score - maximum);
                    sum += weight;
                    for (int64_t d = 0; d < k_head_dim; ++d) {
                        packed[row + static_cast<size_t>(d)] += weight * inputs.v_f32[kv_index(
                            d, kt, kv_head, b, test_case.lk, test_case.kv_heads)];
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

void run_case(
    ggml_opencl_htp_sidecar * sidecar,
    ggml_backend_t htp,
    const attention_case & test_case) {
    constexpr int64_t batch = 1;
    const float scale = 1.0f / std::sqrt(float(k_head_dim));
    const int64_t q_blocks_per_head = (test_case.lq + k_block_size - 1) / k_block_size;
    const int64_t k_blocks_per_head = (test_case.lk + k_block_size - 1) / k_block_size;
    const int64_t total_q_blocks = test_case.heads * q_blocks_per_head;
    const int64_t total_k_blocks = test_case.kv_heads * k_blocks_per_head;
    const std::vector<int32_t> mask_data = make_mask(test_case);
    const generated_inputs inputs = make_inputs(test_case);
    const std::vector<float> reference = reference_attention(test_case, inputs, mask_data, scale);

    const ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead() * 12 + 4096,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    GGML_ASSERT(ctx != nullptr);

    ggml_tensor * q = ggml_new_tensor_4d(
        ctx, GGML_TYPE_F32, k_head_dim, test_case.lq, test_case.heads, batch);
    ggml_tensor * k = ggml_new_tensor_4d(
        ctx, GGML_TYPE_F16, k_head_dim, test_case.lk, test_case.kv_heads, batch);
    ggml_tensor * v = ggml_new_tensor_4d(
        ctx, GGML_TYPE_F16, k_head_dim, test_case.lk, test_case.kv_heads, batch);
    ggml_tensor * mask = ggml_new_tensor_4d(
        ctx, GGML_TYPE_I32, total_k_blocks, total_q_blocks, 1, 1);
    ggml_tensor * packed = ggml_flash_attn_ext_with_lse(
        ctx, q, k, v, mask, scale, 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(packed, GGML_PREC_F32);
    ggml_flash_attn_ext_set_block_sparse(
        packed, k_block_size, k_block_size, GGML_FLASH_ATTN_EXT_MASK_BLOCK_MATRIX);

    GGML_ASSERT(packed->ne[0] == k_head_dim + 1);
    GGML_ASSERT(packed->ne[1] == test_case.heads);
    GGML_ASSERT(packed->ne[2] == test_case.lq);
    GGML_ASSERT(packed->ne[3] == batch);

    std::vector<owned_rpcmem_buffer> buffers;
    buffers.reserve(5);
    buffers.push_back(allocate_tensor_buffer(htp, q));
    buffers.push_back(allocate_tensor_buffer(htp, k));
    buffers.push_back(allocate_tensor_buffer(htp, v));
    buffers.push_back(allocate_tensor_buffer(htp, mask));
    buffers.push_back(allocate_tensor_buffer(htp, packed));

    std::memcpy(q->data, inputs.q.data(), inputs.q.size() * sizeof(inputs.q[0]));
    std::memcpy(k->data, inputs.k.data(), inputs.k.size() * sizeof(inputs.k[0]));
    std::memcpy(v->data, inputs.v.data(), inputs.v.size() * sizeof(inputs.v[0]));
    std::memcpy(mask->data, mask_data.data(), mask_data.size() * sizeof(mask_data[0]));
    std::fill_n(
        static_cast<float *>(packed->data), ggml_nelements(packed),
        std::numeric_limits<float>::quiet_NaN());
    auto output_is_sentinel = [&]() {
        const float * values = static_cast<const float *>(packed->data);
        return std::all_of(
            values, values + ggml_nelements(packed),
            [](float value) { return std::isnan(value); });
    };

    char reason[1024] = {};
    ggml_opencl_htp_graph * graph = nullptr;
    GGML_ASSERT(ggml_backend_opencl_htp_graph_create(
        sidecar, &graph, reason, sizeof(reason)) == GGML_OPENCL_HTP_OK);

    GGML_ASSERT(ggml_backend_opencl_htp_graph_compute_block_sparse_attn_sync(
        graph, packed, reason, sizeof(reason)) == GGML_OPENCL_HTP_UNSUPPORTED);
    GGML_ASSERT(reason[0] != '\0');
    GGML_ASSERT(ggml_backend_opencl_htp_graph_live_alias_count(graph) == 0);
    GGML_ASSERT(ggml_backend_opencl_htp_graph_live_event_count(graph) == 0);
    GGML_ASSERT(output_is_sentinel());

    for (const owned_rpcmem_buffer & buffer : buffers) {
        import_buffer(graph, buffer, reason, sizeof(reason));
    }
    GGML_ASSERT(ggml_backend_opencl_htp_graph_live_alias_count(graph) == 5);

    ggml_tensor unsupported = *packed;
    unsupported.op = GGML_OP_ADD;
    GGML_ASSERT(ggml_backend_opencl_htp_graph_compute_block_sparse_attn_sync(
        graph, &unsupported, reason, sizeof(reason)) == GGML_OPENCL_HTP_UNSUPPORTED);
    GGML_ASSERT(reason[0] != '\0');
    GGML_ASSERT(ggml_backend_opencl_htp_graph_live_alias_count(graph) == 5);
    GGML_ASSERT(ggml_backend_opencl_htp_graph_live_event_count(graph) == 0);
    GGML_ASSERT(output_is_sentinel());

    ggml_tensor overlapping_q = *q;
    overlapping_q.buffer = packed->buffer;
    overlapping_q.data = packed->data;
    ggml_tensor overlapping_output = *packed;
    overlapping_output.src[0] = &overlapping_q;
    GGML_ASSERT(ggml_backend_opencl_htp_graph_compute_block_sparse_attn_sync(
        graph, &overlapping_output, reason, sizeof(reason)) == GGML_OPENCL_HTP_UNSUPPORTED);
    GGML_ASSERT(reason[0] != '\0');
    GGML_ASSERT(ggml_backend_opencl_htp_graph_live_alias_count(graph) == 5);
    GGML_ASSERT(ggml_backend_opencl_htp_graph_live_event_count(graph) == 0);
    GGML_ASSERT(output_is_sentinel());

    ggml_tensor overlapping_rows = *packed;
    overlapping_rows.nb[2] = overlapping_rows.nb[0];
    const ggml_opencl_htp_result overlapping_rows_result =
        ggml_backend_opencl_htp_graph_compute_block_sparse_attn_sync(
            graph, &overlapping_rows, reason, sizeof(reason));
    const bool overlapping_rows_preserved_sentinel = output_is_sentinel();
    if (overlapping_rows_result != GGML_OPENCL_HTP_UNSUPPORTED ||
        !overlapping_rows_preserved_sentinel) {
        std::fprintf(
            stderr,
            "[EXPECTED_RED] overlapping output rows result=%d sentinel_preserved=%s reason=%s\n",
            overlapping_rows_result,
            overlapping_rows_preserved_sentinel ? "yes" : "no",
            reason);
    }
    GGML_ASSERT(overlapping_rows_result == GGML_OPENCL_HTP_UNSUPPORTED);
    GGML_ASSERT(overlapping_rows_preserved_sentinel);
    GGML_ASSERT(ggml_backend_opencl_htp_graph_live_alias_count(graph) == 5);
    GGML_ASSERT(ggml_backend_opencl_htp_graph_live_event_count(graph) == 0);

    const ggml_opencl_htp_result compute_result =
        ggml_backend_opencl_htp_graph_compute_block_sparse_attn_sync(
            graph, packed, reason, sizeof(reason));
    if (compute_result != GGML_OPENCL_HTP_OK) {
        std::fprintf(stderr, "%s compute failed: %s\n", test_case.name, reason);
    }
    GGML_ASSERT(compute_result == GGML_OPENCL_HTP_OK);
    GGML_ASSERT(ggml_backend_opencl_htp_graph_live_event_count(graph) == 0);

    const float * actual = static_cast<const float *>(packed->data);
    float out_max_abs = 0.0f;
    float lse_max_abs = 0.0f;
    bool values_valid = true;
    for (int64_t qt = 0; qt < test_case.lq; ++qt) {
        for (int64_t h = 0; h < test_case.heads; ++h) {
            const size_t row = packed_index(0, h, qt, 0, test_case.lq, test_case.heads);
            for (int64_t d = 0; d < k_head_dim; ++d) {
                const float expected_value = reference[row + static_cast<size_t>(d)];
                const float actual_value = actual[row + static_cast<size_t>(d)];
                values_valid = values_valid && std::isfinite(expected_value) && std::isfinite(actual_value);
                if (std::isfinite(expected_value) && std::isfinite(actual_value)) {
                    out_max_abs = std::max(out_max_abs, std::fabs(expected_value - actual_value));
                }
                if (test_case.empty) {
                    values_valid = values_valid && actual_value == 0.0f;
                }
            }

            const float expected_lse = reference[row + k_head_dim];
            const float actual_lse = actual[row + k_head_dim];
            if (std::isfinite(expected_lse)) {
                values_valid = values_valid && std::isfinite(actual_lse);
                if (std::isfinite(actual_lse)) {
                    lse_max_abs = std::max(lse_max_abs, std::fabs(expected_lse - actual_lse));
                }
            } else {
                values_valid = values_valid && std::isinf(actual_lse) && std::signbit(actual_lse) &&
                               std::isinf(expected_lse) && std::signbit(expected_lse);
            }
        }
    }

    std::printf(
        "[OPENCL_BLOCK_SPARSE] case=%s out_max_abs=%.9g lse_max_abs=%.9g valid=%s\n",
        test_case.name, out_max_abs, lse_max_abs, values_valid ? "yes" : "no");
    GGML_ASSERT(values_valid);
    GGML_ASSERT(out_max_abs <= 3e-3f);
    GGML_ASSERT(lse_max_abs <= 5e-3f);

    GGML_ASSERT(ggml_backend_opencl_htp_graph_finish(
        graph, reason, sizeof(reason)) == GGML_OPENCL_HTP_OK);
    GGML_ASSERT(ggml_backend_opencl_htp_graph_live_alias_count(graph) == 0);
    GGML_ASSERT(ggml_backend_opencl_htp_graph_live_event_count(graph) == 0);
    ggml_backend_opencl_htp_graph_free(&graph);
    ggml_free(ctx);
    free_buffers(buffers);
}

} // namespace

int main() {
    GGML_ASSERT(setenv("SD_HTP_SPARSE_OPENCL", "1", 1) == 0);

    ggml_opencl_qcom_requirements requirements = {};
    char reason[1024] = {};
    ggml_opencl_htp_sidecar * sidecar =
        ggml_backend_opencl_htp_sidecar_create(&requirements, reason, sizeof(reason));
    if (sidecar == nullptr) {
        std::fprintf(stderr, "OpenCL HTP sidecar creation failed: %s\n", reason);
        return 1;
    }

    ggml_backend_t htp = ggml_backend_htp_init();
    GGML_ASSERT(htp != nullptr);
    GGML_ASSERT(ggml_backend_htp_runtime_available());

    run_case(sidecar, htp, {"full_keep", 64, 96, 4, 4, false, false});
    run_case(sidecar, htp, {"sparse_heads", 64, 96, 4, 4, true, false});
    run_case(sidecar, htp, {"empty", 32, 32, 1, 1, false, true});

    ggml_backend_free(htp);
    ggml_backend_opencl_htp_sidecar_free(sidecar);
    return 0;
}
