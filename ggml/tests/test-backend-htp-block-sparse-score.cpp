#include "../../src/ggml_extend.hpp"

#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>
#include <ggml-htp.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>

static ggml_context * make_ctx(size_t tensor_count, size_t extra = 256 * 1024) {
    ggml_init_params params = {
        .mem_size   = ggml_tensor_overhead() * tensor_count + ggml_graph_overhead() + extra,
        .mem_buffer = nullptr,
        .no_alloc   = true,
    };
    ggml_context * ctx = ggml_init(params);
    GGML_ASSERT(ctx != nullptr);
    return ctx;
}

static int ceil_div_int(int a, int b) {
    return (a + b - 1) / b;
}

static std::vector<ggml_fp16_t> make_mean_data(int head_dim, int n_blocks, int seed) {
    std::vector<ggml_fp16_t> data(size_t(head_dim) * size_t(n_blocks));
    for (int b = 0; b < n_blocks; ++b) {
        for (int d = 0; d < head_dim; ++d) {
            const float v = 0.01f * float(((b * 29 + d * 7 + seed * 13) % 251) - 125);
            data[size_t(b) * size_t(head_dim) + size_t(d)] = ggml_fp32_to_fp16(v);
        }
    }
    return data;
}

static std::vector<ggml_fp16_t> make_score_data(int n_q_blocks, int n_k_blocks) {
    std::vector<ggml_fp16_t> data(size_t(n_q_blocks) * size_t(n_k_blocks));
    for (int qb = 0; qb < n_q_blocks; ++qb) {
        for (int kb = 0; kb < n_k_blocks; ++kb) {
            const int v0 = (qb * 37 + kb * 17 + (qb ^ kb) * 3) % 257;
            const int v1 = (qb * 11 - kb * 19 + 1024) % 113;
            const float v = 0.015f * float(v0 - 128) + 0.001f * float(v1 - 56);
            data[size_t(qb) * size_t(n_k_blocks) + size_t(kb)] = ggml_fp32_to_fp16(v);
        }
    }
    return data;
}

static std::vector<int32_t> reference_score_to_mask(const std::vector<ggml_fp16_t> & score,
                                                    int n_q_blocks,
                                                    int n_k_blocks,
                                                    int block_q,
                                                    int block_k,
                                                    float keep_ratio,
                                                    bool causal,
                                                    int protected_prefix_len,
                                                    int n_heads,
                                                    int n_kv_heads) {
    std::vector<int32_t> block_mask(size_t(n_q_blocks) * size_t(n_k_blocks), 1);
    std::vector<std::pair<float, int>> candidates;
    candidates.reserve(size_t(n_k_blocks));

    n_heads = std::max(1, n_heads);
    n_kv_heads = std::max(1, n_kv_heads);
    protected_prefix_len = std::max(0, protected_prefix_len);
    keep_ratio = std::clamp(keep_ratio, 0.0f, 1.0f);
    GGML_ASSERT(n_heads % n_kv_heads == 0);
    GGML_ASSERT(n_q_blocks % n_heads == 0);
    GGML_ASSERT(n_k_blocks % n_kv_heads == 0);
    const int gqa = std::max(1, n_heads / n_kv_heads);
    const int q_blocks_per_head = n_q_blocks / n_heads;
    const int k_blocks_per_head = n_k_blocks / n_kv_heads;
    const int protected_q_blocks_per_head = std::min(q_blocks_per_head, (protected_prefix_len + block_q - 1) / block_q);
    const int protected_k_blocks_per_head = std::min(k_blocks_per_head, (protected_prefix_len + block_k - 1) / block_k);

    for (int qb = 0; qb < n_q_blocks; ++qb) {
        const int q_head = qb / q_blocks_per_head;
        const int q_block_in_head = qb % q_blocks_per_head;
        const int kv_head = std::min(n_kv_heads - 1, q_head / gqa);
        const bool protect_q = q_block_in_head < protected_q_blocks_per_head;
        for (int kb = 0; kb < n_k_blocks; ++kb) {
            const int k_head = kb / k_blocks_per_head;
            const int k_block_in_head = kb % k_blocks_per_head;
            if (k_head == kv_head && (protect_q || k_block_in_head < protected_k_blocks_per_head)) {
                block_mask[size_t(qb) * size_t(n_k_blocks) + size_t(kb)] = 0;
            }
        }
    }

    for (int qb = 0; qb < n_q_blocks; ++qb) {
        const int q_head = qb / q_blocks_per_head;
        const int q_block_in_head = qb % q_blocks_per_head;
        const int kv_head = std::min(n_kv_heads - 1, q_head / gqa);
        if (q_block_in_head < protected_q_blocks_per_head) {
            continue;
        }
        if (keep_ratio >= 1.0f) {
            for (int kb = 0; kb < n_k_blocks; ++kb) {
                const int k_head = kb / k_blocks_per_head;
                if (k_head != kv_head) {
                    continue;
                }
                const int k_block_in_head = kb % k_blocks_per_head;
                if (k_block_in_head < protected_k_blocks_per_head) {
                    continue;
                }
                if (causal && k_block_in_head * block_k > q_block_in_head * block_q + block_q - 1) {
                    continue;
                }
                block_mask[size_t(qb) * size_t(n_k_blocks) + size_t(kb)] = 0;
            }
            continue;
        }

        candidates.clear();
        for (int kb = 0; kb < n_k_blocks; ++kb) {
            const int k_head = kb / k_blocks_per_head;
            if (k_head != kv_head) {
                continue;
            }
            const int k_block_in_head = kb % k_blocks_per_head;
            if (k_block_in_head < protected_k_blocks_per_head) {
                continue;
            }
            if (causal && k_block_in_head * block_k > q_block_in_head * block_q + block_q - 1) {
                continue;
            }
            const float s = ggml_fp16_to_fp32(score[size_t(qb) * size_t(n_k_blocks) + size_t(kb)]);
            candidates.emplace_back(s, kb);
        }
        if (candidates.empty()) {
            continue;
        }

        std::sort(candidates.begin(), candidates.end(), [](const auto & a, const auto & b) {
            if (a.first != b.first) {
                return a.first > b.first;
            }
            return a.second < b.second;
        });

        const float max_score = candidates.front().first;
        double total_weight = 0.0;
        for (const auto & cand : candidates) {
            total_weight += std::exp(double(cand.first - max_score));
        }
        if (total_weight <= 0.0) {
            continue;
        }

        double kept_weight = 0.0;
        for (const auto & cand : candidates) {
            kept_weight += std::exp(double(cand.first - max_score));
            block_mask[size_t(qb) * size_t(n_k_blocks) + size_t(cand.second)] = 0;
            if (kept_weight / total_weight >= keep_ratio) {
                break;
            }
        }
    }

    return block_mask;
}

static std::vector<int32_t> read_i32_tensor(const ggml_tensor * t) {
    std::vector<int32_t> data(size_t(ggml_nelements(t)));
    ggml_backend_tensor_get(t, data.data(), 0, data.size() * sizeof(int32_t));
    return data;
}

static std::vector<float> read_f32_tensor(const ggml_tensor * t) {
    std::vector<float> data(size_t(ggml_nelements(t)));
    ggml_backend_tensor_get(t, data.data(), 0, data.size() * sizeof(float));
    return data;
}

static void test_patch_major_mapping_roundtrip() {
    const std::vector<std::pair<int, int>> shapes = {
        {16, 16},
        {20, 20},
        {16, 32},
        {13, 17},
    };

    for (const auto & shape : shapes) {
        const int h = shape.first;
        const int w = shape.second;
        GGML_ASSERT(sd_patch_major_token_count(h, w) == h * w);
        std::vector<int> seen(size_t(h) * size_t(w), 0);
        for (int row = 0; row < h * w; ++row) {
            const int pm = sd_patch_major_index_from_row_major(row, h, w, 4);
            GGML_ASSERT(pm >= 0 && pm < h * w);
            GGML_ASSERT(sd_row_major_index_from_patch_major(pm, h, w, 4) == row);
            seen[size_t(pm)]++;
        }
        for (int v : seen) {
            GGML_ASSERT(v == 1);
        }
    }

    std::printf("[PASS] patch-major mapping roundtrip\n");
}

static void test_patch_major_shift_mapping_roundtrip() {
    const std::vector<std::pair<int, int>> shapes = {
        {16, 16},
        {20, 20},
        {16, 32},
        {13, 17},
        {3, 5},
    };
    const std::vector<std::pair<int, int>> shifts = {
        {0, 0},
        {-2, -2},
        {2, 2},
        {-7, 5},
    };

    for (const auto & shape : shapes) {
        const int h = shape.first;
        const int w = shape.second;
        for (const auto & shift : shifts) {
            std::vector<int> seen(size_t(h) * size_t(w), 0);
            for (int row = 0; row < h * w; ++row) {
                const int pm = sd_patch_major_shifted_index_from_row_major(row, h, w, 4, shift.first, shift.second);
                GGML_ASSERT(pm >= 0 && pm < h * w);
                GGML_ASSERT(sd_row_major_index_from_patch_major_shifted(pm, h, w, 4, shift.first, shift.second) == row);
                seen[size_t(pm)]++;
            }
            for (int v : seen) {
                GGML_ASSERT(v == 1);
            }
        }
    }

    std::printf("[PASS] patch-major shifted mapping roundtrip\n");
}

static void test_patch_major_reorder_tensor_roundtrip() {
    constexpr int c = 3;
    constexpr int h = 20;
    constexpr int w = 20;
    constexpr int n = 1;
    const int l = h * w;
    std::vector<float> input(size_t(c) * size_t(l) * size_t(n));
    for (int token = 0; token < l; ++token) {
        for (int ch = 0; ch < c; ++ch) {
            input[size_t(token) * size_t(c) + size_t(ch)] = float(token * 10 + ch);
        }
    }

    ggml_context * ctx = make_ctx(16, input.size() * sizeof(float) * 3 + 256 * 1024);
    ggml_tensor * src = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, c, l, n);
    ggml_tensor * reordered = sd_patch_major_reorder_node(ctx, src, h, w, 4, false);
    ggml_tensor * restored = sd_patch_major_reorder_node(ctx, reordered, h, w, 4, true);
    GGML_ASSERT(src != nullptr && reordered != nullptr && restored != nullptr);

    ggml_backend_t cpu = ggml_backend_init_by_name("CPU", nullptr);
    GGML_ASSERT(cpu != nullptr);
    ggml_backend_cpu_set_n_threads(cpu, 4);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_get_default_buffer_type(cpu));
    GGML_ASSERT(buffer != nullptr);

    ggml_backend_tensor_set(src, input.data(), 0, input.size() * sizeof(float));
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, restored);
    GGML_ASSERT(ggml_backend_graph_compute(cpu, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(cpu);

    const std::vector<float> actual = read_f32_tensor(restored);
    GGML_ASSERT(actual == input);

    ggml_backend_buffer_free(buffer);
    ggml_backend_free(cpu);
    ggml_free(ctx);

    std::printf("[PASS] patch-major tensor reorder roundtrip\n");
}

static void test_patch_major_shifted_reorder_tensor_roundtrip() {
    constexpr int c = 3;
    constexpr int h = 13;
    constexpr int w = 17;
    constexpr int n = 1;
    const int l = h * w;
    std::vector<float> input(size_t(c) * size_t(l) * size_t(n));
    for (int token = 0; token < l; ++token) {
        for (int ch = 0; ch < c; ++ch) {
            input[size_t(token) * size_t(c) + size_t(ch)] = float(token * 10 + ch);
        }
    }

    ggml_context * ctx = make_ctx(16, input.size() * sizeof(float) * 3 + 256 * 1024);
    ggml_tensor * src = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, c, l, n);
    ggml_tensor * reordered = sd_patch_major_reorder_node(ctx, src, h, w, 4, false, -2, -2);
    ggml_tensor * restored = sd_patch_major_reorder_node(ctx, reordered, h, w, 4, true, -2, -2);
    GGML_ASSERT(src != nullptr && reordered != nullptr && restored != nullptr);

    ggml_backend_t cpu = ggml_backend_init_by_name("CPU", nullptr);
    GGML_ASSERT(cpu != nullptr);
    ggml_backend_cpu_set_n_threads(cpu, 4);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_get_default_buffer_type(cpu));
    GGML_ASSERT(buffer != nullptr);

    ggml_backend_tensor_set(src, input.data(), 0, input.size() * sizeof(float));
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, restored);
    GGML_ASSERT(ggml_backend_graph_compute(cpu, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(cpu);

    const std::vector<float> actual = read_f32_tensor(restored);
    GGML_ASSERT(actual == input);

    ggml_backend_buffer_free(buffer);
    ggml_backend_free(cpu);
    ggml_free(ctx);

    std::printf("[PASS] patch-major shifted tensor reorder roundtrip\n");
}

static void test_patch_major_shifted_to_shifted_reorder_tensor_roundtrip() {
    constexpr int c = 3;
    constexpr int h = 13;
    constexpr int w = 17;
    constexpr int n = 1;
    const int l = h * w;
    std::vector<float> input(size_t(c) * size_t(l) * size_t(n));
    for (int token = 0; token < l; ++token) {
        for (int ch = 0; ch < c; ++ch) {
            input[size_t(token) * size_t(c) + size_t(ch)] = float(token * 10 + ch);
        }
    }

    ggml_context * ctx = make_ctx(20, input.size() * sizeof(float) * 4 + 256 * 1024);
    ggml_tensor * src = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, c, l, n);
    ggml_tensor * unshifted = sd_patch_major_reorder_node(ctx, src, h, w, 4, false, 0, 0);
    ggml_tensor * shifted = sd_patch_major_reorder_node(ctx, unshifted, h, w, 4, false, -2, -2, 0, 0, true);
    ggml_tensor * restored = sd_patch_major_reorder_node(ctx, shifted, h, w, 4, true, -2, -2);
    GGML_ASSERT(src != nullptr && unshifted != nullptr && shifted != nullptr && restored != nullptr);

    ggml_backend_t cpu = ggml_backend_init_by_name("CPU", nullptr);
    GGML_ASSERT(cpu != nullptr);
    ggml_backend_cpu_set_n_threads(cpu, 4);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_get_default_buffer_type(cpu));
    GGML_ASSERT(buffer != nullptr);

    ggml_backend_tensor_set(src, input.data(), 0, input.size() * sizeof(float));
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, restored);
    GGML_ASSERT(ggml_backend_graph_compute(cpu, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(cpu);

    const std::vector<float> actual = read_f32_tensor(restored);
    GGML_ASSERT(actual == input);

    ggml_backend_buffer_free(buffer);
    ggml_backend_free(cpu);
    ggml_free(ctx);

    std::printf("[PASS] patch-major shifted-to-shifted tensor reorder roundtrip\n");
}

static void test_block_sparse_step_shift_policy() {
    GGML_ASSERT(sd_htp_block_sparse_step_shift(false, 0) == 0);
    GGML_ASSERT(sd_htp_block_sparse_step_shift(false, 1) == 0);
    GGML_ASSERT(sd_htp_block_sparse_step_shift(false, 3) == 0);
    GGML_ASSERT(sd_htp_block_sparse_step_shift(false, 6) == 0);
    GGML_ASSERT(sd_htp_block_sparse_step_shift(true, 0) == 0);
    GGML_ASSERT(sd_htp_block_sparse_step_shift(true, 1) == 0);
    GGML_ASSERT(sd_htp_block_sparse_step_shift(true, 2) == -2);
    GGML_ASSERT(sd_htp_block_sparse_step_shift(true, 3) == 0);
    GGML_ASSERT(sd_htp_block_sparse_step_shift(true, 4) == -2);
    GGML_ASSERT(sd_htp_block_sparse_step_shift(true, 5) == 0);
    GGML_ASSERT(sd_htp_block_sparse_step_shift(true, 6) == -2);
    GGML_ASSERT(sd_htp_block_sparse_step_shift(true, -1) == 0);

    std::printf("[PASS] block sparse step shift policy\n");
}

static void test_block_sparse_dense_interval_policy() {
    GGML_ASSERT(sd_htp_block_sparse_scheduled_step_index(0, 0) == 0);
    GGML_ASSERT(sd_htp_block_sparse_scheduled_step_index(1, 0) == 1);
    GGML_ASSERT(sd_htp_block_sparse_scheduled_step_index(4, 0) == 4);
    GGML_ASSERT(sd_htp_block_sparse_scheduled_step_index(1, -3) == 1);

    GGML_ASSERT(sd_htp_block_sparse_scheduled_step_index(1, 2) == 1);
    GGML_ASSERT(sd_htp_block_sparse_scheduled_step_index(2, 2) == 2);
    GGML_ASSERT(sd_htp_block_sparse_scheduled_step_index(3, 2) == 0);
    GGML_ASSERT(sd_htp_block_sparse_scheduled_step_index(4, 2) == 3);
    GGML_ASSERT(sd_htp_block_sparse_scheduled_step_index(5, 2) == 4);
    GGML_ASSERT(sd_htp_block_sparse_scheduled_step_index(6, 2) == 0);

    GGML_ASSERT(sd_htp_block_sparse_scheduled_step_index(1, 1) == 1);
    GGML_ASSERT(sd_htp_block_sparse_scheduled_step_index(2, 1) == 0);
    GGML_ASSERT(sd_htp_block_sparse_scheduled_step_index(3, 1) == 2);
    GGML_ASSERT(sd_htp_block_sparse_scheduled_step_index(4, 1) == 0);

    std::printf("[PASS] block sparse dense interval policy\n");
}

static void test_block_sparse_residual_mode_policy() {
    GGML_ASSERT(sd_htp_block_sparse_residual_mode(false, false, 0) == SD_HTP_BLOCK_SPARSE_RESIDUAL_OFF);
    GGML_ASSERT(sd_htp_block_sparse_residual_mode(false, true, 0) == SD_HTP_BLOCK_SPARSE_RESIDUAL_OFF);
    GGML_ASSERT(sd_htp_block_sparse_residual_mode(true, false, 0) == SD_HTP_BLOCK_SPARSE_RESIDUAL_OFF);
    GGML_ASSERT(sd_htp_block_sparse_residual_mode(true, false, 1) == SD_HTP_BLOCK_SPARSE_RESIDUAL_APPLY);
    GGML_ASSERT(sd_htp_block_sparse_residual_mode(true, false, 3) == SD_HTP_BLOCK_SPARSE_RESIDUAL_APPLY);
    GGML_ASSERT(sd_htp_block_sparse_residual_mode(true, true, 0) == SD_HTP_BLOCK_SPARSE_RESIDUAL_ANCHOR);
    GGML_ASSERT(sd_htp_block_sparse_residual_mode(true, true, 3) == SD_HTP_BLOCK_SPARSE_RESIDUAL_ANCHOR);

    std::printf("[PASS] block sparse residual mode policy\n");
}

static void test_block_sparse_residual_warmup_anchor_policy() {
    GGML_ASSERT(!sd_htp_block_sparse_warmup_residual_anchor_step(0, 4, 20, 4));
    GGML_ASSERT(!sd_htp_block_sparse_warmup_residual_anchor_step(3, 4, 20, 4));
    GGML_ASSERT(sd_htp_block_sparse_warmup_residual_anchor_step(4, 4, 20, 4));
    GGML_ASSERT(!sd_htp_block_sparse_warmup_residual_anchor_step(5, 4, 20, 4));
    GGML_ASSERT(!sd_htp_block_sparse_warmup_residual_anchor_step(4, 0, 20, 4));
    GGML_ASSERT(!sd_htp_block_sparse_warmup_residual_anchor_step(20, 20, 20, 0));
    GGML_ASSERT(!sd_htp_block_sparse_warmup_residual_anchor_step(16, 16, 20, 4));

    std::printf("[PASS] block sparse residual warmup anchor policy\n");
}

static void test_block_sparse_residual_anchor_has_sparse_consumer_policy() {
    GGML_ASSERT(sd_htp_block_sparse_residual_anchor_has_sparse_consumer(4, 20, 4));
    GGML_ASSERT(sd_htp_block_sparse_residual_anchor_has_sparse_consumer(8, 20, 4));
    GGML_ASSERT(sd_htp_block_sparse_residual_anchor_has_sparse_consumer(12, 20, 4));
    GGML_ASSERT(!sd_htp_block_sparse_residual_anchor_has_sparse_consumer(16, 20, 4));
    GGML_ASSERT(!sd_htp_block_sparse_residual_anchor_has_sparse_consumer(20, 20, 0));

    std::printf("[PASS] block sparse residual anchor consumer policy\n");
}

static void test_block_sparse_residual_cache_key_policy() {
    GGML_ASSERT(sd_htp_block_sparse_residual_cache_key(0, "joint", 3) ==
                "htp_block_sparse_residual_b0_joint_s3");
    GGML_ASSERT(sd_htp_block_sparse_residual_cache_key(1, "self", 4) ==
                "htp_block_sparse_residual_b1_self_s4");
    GGML_ASSERT(sd_htp_block_sparse_residual_cache_key(2, "self2", 5) ==
                "htp_block_sparse_residual_b2_self2_s5");

    std::printf("[PASS] block sparse residual cache key policy\n");
}

static void test_patch_score_mask_promotes_to_dsp_index() {
    constexpr int n_q16 = 4;
    constexpr int n_k16 = 5;
    constexpr int n_heads = 1;
    constexpr int n_kv_heads = 1;
    constexpr int max_kept32 = 3;

    std::vector<int32_t> mask16(size_t(n_q16) * size_t(n_k16), 1);
    mask16[0 * n_k16 + 0] = 0;
    mask16[1 * n_k16 + 1] = 0;
    mask16[2 * n_k16 + 2] = 0;
    mask16[3 * n_k16 + 4] = 0;

    std::vector<int32_t> index32(size_t(2) * size_t(max_kept32), -1);
    ggml_ext_block_sparse_promote_16_mask_to_32_index(mask16.data(),
                                                      index32.data(),
                                                      n_q16,
                                                      n_k16,
                                                      max_kept32,
                                                      n_heads,
                                                      n_kv_heads);

    const std::vector<int32_t> expected = {
        0, -1, -1,
        1, 2, -1,
    };
    GGML_ASSERT(index32 == expected);

    std::printf("[PASS] patch score mask promotes to dsp index\n");
}

static void test_patch_score_mask_2d_halo_keeps_surrounding_blocks() {
    constexpr int n_q16 = 16;
    constexpr int n_k16 = 16;
    constexpr int n_heads = 1;
    constexpr int n_kv_heads = 1;

    std::vector<int32_t> mask16(size_t(n_q16) * size_t(n_k16), 1);
    mask16[5 * n_k16 + 15] = 0;

    ggml_ext_block_sparse_force_patch_halo_mask(mask16.data(),
                                                n_q16,
                                                n_k16,
                                                16,
                                                16,
                                                n_heads,
                                                n_kv_heads);

    const std::vector<int32_t> expected_kept = {
        0, 1, 2,
        4, 5, 6,
        8, 9, 10,
        15,
    };
    std::vector<int32_t> actual_kept;
    for (int kb = 0; kb < n_k16; ++kb) {
        if (mask16[5 * n_k16 + kb] == 0) {
            actual_kept.push_back(kb);
        }
    }
    GGML_ASSERT(actual_kept == expected_kept);

    std::printf("[PASS] patch score mask 2d halo keeps surrounding blocks\n");
}

static void test_patch_score_mask_center_sink_keeps_center_blocks() {
    constexpr int n_q16 = 16;
    constexpr int n_k16 = 16;
    constexpr int n_heads = 1;
    constexpr int n_kv_heads = 1;

    std::vector<int32_t> mask16(size_t(n_q16) * size_t(n_k16), 1);
    ggml_ext_block_sparse_force_center_sink_mask(mask16.data(),
                                                 n_q16,
                                                 n_k16,
                                                 16,
                                                 16,
                                                 n_heads,
                                                 n_kv_heads);

    const std::vector<int32_t> expected_kept = {5, 6, 9, 10};
    for (int qb = 0; qb < n_q16; ++qb) {
        std::vector<int32_t> actual_kept;
        for (int kb = 0; kb < n_k16; ++kb) {
            if (mask16[size_t(qb) * size_t(n_k16) + size_t(kb)] == 0) {
                actual_kept.push_back(kb);
            }
        }
        GGML_ASSERT(actual_kept == expected_kept);
    }

    std::printf("[PASS] patch score mask center sink keeps center blocks\n");
}

static void test_patch_score_node_promotes_to_dsp_index() {
    constexpr int n_q16 = 4;
    constexpr int n_k16 = 5;
    constexpr int n_heads = 1;
    constexpr int n_kv_heads = 1;
    constexpr int max_kept32 = 3;

    setenv("SD_HTP_BLOCK_SPARSE_KEEP_RATIO", "0.01", 1);

    std::vector<ggml_fp16_t> score(size_t(n_q16) * size_t(n_k16), ggml_fp32_to_fp16(-10.0f));
    score[0 * n_k16 + 0] = ggml_fp32_to_fp16(10.0f);
    score[1 * n_k16 + 1] = ggml_fp32_to_fp16(10.0f);
    score[2 * n_k16 + 2] = ggml_fp32_to_fp16(10.0f);
    score[3 * n_k16 + 4] = ggml_fp32_to_fp16(10.0f);

    ggml_context * ctx = make_ctx(8, score.size() * sizeof(ggml_fp16_t) + 256 * 1024);
    ggml_tensor * score_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_k16, n_q16);
    ggml_tensor * index_t = ggml_ext_block_sparse_index_32_from_16_score_node(ctx,
                                                                              score_t,
                                                                              2,
                                                                              false,
                                                                              0,
                                                                              0,
                                                                              n_heads,
                                                                              n_kv_heads);
    GGML_ASSERT(score_t != nullptr && index_t != nullptr);
    GGML_ASSERT(index_t->ne[0] == max_kept32);
    GGML_ASSERT(index_t->ne[1] == 2);

    ggml_backend_t cpu = ggml_backend_init_by_name("CPU", nullptr);
    GGML_ASSERT(cpu != nullptr);
    ggml_backend_cpu_set_n_threads(cpu, 4);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_get_default_buffer_type(cpu));
    GGML_ASSERT(buffer != nullptr);

    ggml_backend_tensor_set(score_t, score.data(), 0, score.size() * sizeof(ggml_fp16_t));
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, index_t);
    GGML_ASSERT(ggml_backend_graph_compute(cpu, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(cpu);

    const std::vector<int32_t> index32 = read_i32_tensor(index_t);
    const std::vector<int32_t> expected = {
        0, -1, -1,
        1, 2, -1,
    };
    GGML_ASSERT(index32 == expected);

    ggml_backend_buffer_free(buffer);
    ggml_backend_free(cpu);
    ggml_free(ctx);

    std::printf("[PASS] patch score node promotes to dsp index\n");
}

static std::vector<int32_t> run_mask_node_cpu(int n_q_blocks,
                                              int n_k_blocks,
                                              int block_q,
                                              int block_k,
                                              const std::vector<ggml_fp16_t> & score,
                                              bool causal,
                                              int protected_prefix_len,
                                              int n_heads,
                                              int n_kv_heads) {
    ggml_context * ctx = make_ctx(8, score.size() * sizeof(ggml_fp16_t) + 256 * 1024);
    ggml_tensor * score_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_k_blocks, n_q_blocks);
    ggml_tensor * mask_t = ggml_ext_block_sparse_mask_from_score_node(ctx,
                                                                      score_t,
                                                                      block_q,
                                                                      block_k,
                                                                      2,
                                                                      causal,
                                                                      0,
                                                                      protected_prefix_len,
                                                                      n_heads,
                                                                      n_kv_heads);
    GGML_ASSERT(score_t != nullptr && mask_t != nullptr);

    ggml_backend_t cpu = ggml_backend_init_by_name("CPU", nullptr);
    GGML_ASSERT(cpu != nullptr);
    ggml_backend_cpu_set_n_threads(cpu, 4);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_get_default_buffer_type(cpu));
    GGML_ASSERT(buffer != nullptr);

    ggml_backend_tensor_set(score_t, score.data(), 0, score.size() * sizeof(ggml_fp16_t));
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, mask_t);
    GGML_ASSERT(ggml_backend_graph_compute(cpu, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(cpu);

    std::vector<int32_t> mask = read_i32_tensor(mask_t);
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(cpu);
    ggml_free(ctx);
    return mask;
}

static std::vector<int32_t> run_index_node_cpu(int n_q_blocks,
                                               int n_k_blocks,
                                               int block_q,
                                               int block_k,
                                               const std::vector<ggml_fp16_t> & score,
                                               bool causal,
                                               int protected_prefix_len,
                                               int n_heads,
                                               int n_kv_heads,
                                               int * max_kept_out) {
    ggml_context * ctx = make_ctx(8, score.size() * sizeof(ggml_fp16_t) + 256 * 1024);
    ggml_tensor * score_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_k_blocks, n_q_blocks);
    ggml_tensor * index_t = ggml_ext_block_sparse_index_from_score_node(ctx,
                                                                        score_t,
                                                                        block_q,
                                                                        block_k,
                                                                        2,
                                                                        causal,
                                                                        0,
                                                                        protected_prefix_len,
                                                                        n_heads,
                                                                        n_kv_heads);
    GGML_ASSERT(score_t != nullptr && index_t != nullptr);

    ggml_backend_t cpu = ggml_backend_init_by_name("CPU", nullptr);
    GGML_ASSERT(cpu != nullptr);
    ggml_backend_cpu_set_n_threads(cpu, 4);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_get_default_buffer_type(cpu));
    GGML_ASSERT(buffer != nullptr);

    ggml_backend_tensor_set(score_t, score.data(), 0, score.size() * sizeof(ggml_fp16_t));
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, index_t);
    GGML_ASSERT(ggml_backend_graph_compute(cpu, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(cpu);

    if (max_kept_out != nullptr) {
        *max_kept_out = static_cast<int>(index_t->ne[0]);
    }
    std::vector<int32_t> index = read_i32_tensor(index_t);
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(cpu);
    ggml_free(ctx);
    return index;
}

static std::vector<int32_t> run_index_node_cpu_range(int n_q_blocks,
                                                     int n_k_blocks,
                                                     int block_q,
                                                     int block_k,
                                                     const std::vector<ggml_fp16_t> & score,
                                                     bool causal,
                                                     int protected_start,
                                                     int protected_len,
                                                     int n_heads,
                                                     int n_kv_heads,
                                                     int * max_kept_out) {
    ggml_context * ctx = make_ctx(8, score.size() * sizeof(ggml_fp16_t) + 256 * 1024);
    ggml_tensor * score_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_k_blocks, n_q_blocks);
    ggml_tensor * index_t = ggml_ext_block_sparse_index_from_score_node(ctx,
                                                                        score_t,
                                                                        block_q,
                                                                        block_k,
                                                                        2,
                                                                        causal,
                                                                        protected_start,
                                                                        protected_len,
                                                                        n_heads,
                                                                        n_kv_heads);
    GGML_ASSERT(score_t != nullptr && index_t != nullptr);

    ggml_backend_t cpu = ggml_backend_init_by_name("CPU", nullptr);
    GGML_ASSERT(cpu != nullptr);
    ggml_backend_cpu_set_n_threads(cpu, 4);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_get_default_buffer_type(cpu));
    GGML_ASSERT(buffer != nullptr);

    ggml_backend_tensor_set(score_t, score.data(), 0, score.size() * sizeof(ggml_fp16_t));
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, index_t);
    GGML_ASSERT(ggml_backend_graph_compute(cpu, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(cpu);

    if (max_kept_out != nullptr) {
        *max_kept_out = static_cast<int>(index_t->ne[0]);
    }
    std::vector<int32_t> index = read_i32_tensor(index_t);
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(cpu);
    ggml_free(ctx);
    return index;
}

static void test_score_to_index_suffix_protection() {
    constexpr int block_q = 32;
    constexpr int block_k = 32;
    constexpr int n_heads = 24;
    constexpr int n_kv_heads = 24;
    constexpr int qo_len = 410;
    constexpr int kv_len = 410;
    constexpr int image_tokens = 256;
    constexpr int text_tokens = 154;
    const int n_q_blocks = ceil_div_int(qo_len, block_q) * n_heads;
    const int n_k_blocks = ceil_div_int(kv_len, block_k) * n_kv_heads;
    const std::vector<ggml_fp16_t> score = make_score_data(n_q_blocks, n_k_blocks);

    setenv("SD_HTP_BLOCK_SPARSE_KEEP_RATIO", "0.5", 1);

    int max_kept = 0;
    const std::vector<int32_t> index = run_index_node_cpu_range(n_q_blocks,
                                                                n_k_blocks,
                                                                block_q,
                                                                block_k,
                                                                score,
                                                                false,
                                                                image_tokens,
                                                                text_tokens,
                                                                n_heads,
                                                                n_kv_heads,
                                                                &max_kept);

    const int blocks_per_head = ceil_div_int(qo_len, block_q);
    const int protected_first_block = image_tokens / block_q;
    const int protected_last_block = ceil_div_int(image_tokens + text_tokens, block_q) - 1;
    GGML_ASSERT(max_kept == blocks_per_head);

    for (int h = 0; h < n_heads; ++h) {
        const int q_text_block = h * blocks_per_head + protected_first_block;
        std::vector<int32_t> kept;
        for (int i = 0; i < max_kept; ++i) {
            const int32_t kb = index[size_t(q_text_block) * size_t(max_kept) + size_t(i)];
            if (kb >= 0) {
                kept.push_back(kb);
            }
        }
        GGML_ASSERT(kept.size() == size_t(blocks_per_head));
        GGML_ASSERT(kept.front() == h * blocks_per_head);
        GGML_ASSERT(kept.back() == h * blocks_per_head + blocks_per_head - 1);

        const int q_image_block = h * blocks_per_head;
        bool saw_text_k = false;
        for (int i = 0; i < max_kept; ++i) {
            const int32_t kb = index[size_t(q_image_block) * size_t(max_kept) + size_t(i)];
            if (kb < 0) {
                continue;
            }
            const int k_block_in_head = kb - h * blocks_per_head;
            if (k_block_in_head >= protected_first_block && k_block_in_head <= protected_last_block) {
                saw_text_k = true;
            }
        }
        GGML_ASSERT(saw_text_k);
    }

    std::printf("[PASS] score-to-index suffix protection image=%d text=%d max_kept=%d\n",
                image_tokens,
                text_tokens,
                max_kept);
}

static void test_score_to_index_equivalence(int qo_len,
                                            int kv_len,
                                            float keep_ratio,
                                            int protected_prefix_len) {
    constexpr int block_q = 32;
    constexpr int block_k = 32;
    constexpr int n_heads = 24;
    constexpr int n_kv_heads = 24;
    const bool causal = false;
    const int n_q_blocks = ceil_div_int(qo_len, block_q) * n_heads;
    const int n_k_blocks = ceil_div_int(kv_len, block_k) * n_kv_heads;
    const std::vector<ggml_fp16_t> score = make_score_data(n_q_blocks, n_k_blocks);

    char keep_ratio_buf[32];
    std::snprintf(keep_ratio_buf, sizeof(keep_ratio_buf), "%.8g", double(keep_ratio));
    setenv("SD_HTP_BLOCK_SPARSE_KEEP_RATIO", keep_ratio_buf, 1);

    const std::vector<int32_t> mask = run_mask_node_cpu(n_q_blocks,
                                                        n_k_blocks,
                                                        block_q,
                                                        block_k,
                                                        score,
                                                        causal,
                                                        protected_prefix_len,
                                                        n_heads,
                                                        n_kv_heads);
    int max_kept = 0;
    const std::vector<int32_t> index = run_index_node_cpu(n_q_blocks,
                                                          n_k_blocks,
                                                          block_q,
                                                          block_k,
                                                          score,
                                                          causal,
                                                          protected_prefix_len,
                                                          n_heads,
                                                          n_kv_heads,
                                                          &max_kept);

    GGML_ASSERT(max_kept == ggml_ext_block_sparse_max_kept_blocks(n_q_blocks, n_k_blocks, n_heads, n_kv_heads));
    GGML_ASSERT(index.size() == size_t(max_kept) * size_t(n_q_blocks));

    for (int qb = 0; qb < n_q_blocks; ++qb) {
        std::vector<int32_t> kept_from_mask;
        for (int kb = 0; kb < n_k_blocks; ++kb) {
            if (mask[size_t(qb) * size_t(n_k_blocks) + size_t(kb)] == 0) {
                kept_from_mask.push_back(kb);
            }
        }

        std::vector<int32_t> kept_from_index;
        bool saw_pad = false;
        for (int i = 0; i < max_kept; ++i) {
            const int32_t kb = index[size_t(qb) * size_t(max_kept) + size_t(i)];
            if (kb < 0) {
                saw_pad = true;
                continue;
            }
            if (saw_pad || kb >= n_k_blocks) {
                std::fprintf(stderr,
                             "index row format mismatch qb=%d i=%d kb=%d max_kept=%d n_k_blocks=%d\n",
                             qb,
                             i,
                             kb,
                             max_kept,
                             n_k_blocks);
                GGML_ASSERT(false);
            }
            kept_from_index.push_back(kb);
        }

        if (kept_from_mask != kept_from_index) {
            std::fprintf(stderr,
                         "mask/index mismatch qo=%d kv=%d keep_ratio=%.3f protected=%d qb=%d mask_kept=%zu index_kept=%zu\n",
                         qo_len,
                         kv_len,
                         keep_ratio,
                         protected_prefix_len,
                         qb,
                         kept_from_mask.size(),
                         kept_from_index.size());
            GGML_ASSERT(false);
        }
    }

    std::printf("[PASS] score-to-index equivalence qo=%d kv=%d keep_ratio=%.3f protected=%d max_kept=%d\n",
                qo_len,
                kv_len,
                keep_ratio,
                protected_prefix_len,
                max_kept);
}

static void test_score_to_mask_reference(int n_q_blocks, int n_k_blocks, float keep_ratio, int protected_prefix_len) {
    constexpr int block_q = 32;
    constexpr int block_k = 32;
    constexpr int n_heads = 24;
    constexpr int n_kv_heads = 24;
    const bool causal = false;
    const std::vector<ggml_fp16_t> score = make_score_data(n_q_blocks, n_k_blocks);

    char keep_ratio_buf[32];
    std::snprintf(keep_ratio_buf, sizeof(keep_ratio_buf), "%.8g", double(keep_ratio));
    setenv("SD_HTP_BLOCK_SPARSE_KEEP_RATIO", keep_ratio_buf, 1);

    ggml_init_params params = {
        .mem_size   = ggml_tensor_overhead() * 2 + ggml_graph_overhead() + score.size() * sizeof(ggml_fp16_t) + 4096,
        .mem_buffer = nullptr,
        .no_alloc   = false,
    };
    ggml_context * ctx = ggml_init(params);
    GGML_ASSERT(ctx != nullptr);
    ggml_tensor * score_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_k_blocks, n_q_blocks);
    GGML_ASSERT(score_t != nullptr);
    std::memcpy(score_t->data, score.data(), score.size() * sizeof(ggml_fp16_t));

    const std::vector<int32_t> expected = reference_score_to_mask(score,
                                                                  n_q_blocks,
                                                                  n_k_blocks,
                                                                  block_q,
                                                                  block_k,
                                                                  keep_ratio,
                                                                  causal,
                                                                  protected_prefix_len,
                                                                  n_heads,
                                                                  n_kv_heads);
    const std::vector<int32_t> actual = ggml_ext_block_sparse_score_to_mask(score_t,
                                                                            block_q,
                                                                            block_k,
                                                                            2,
                                                                            causal,
                                                                            0,
                                                                            protected_prefix_len,
                                                                            n_heads,
                                                                            n_kv_heads);
    const std::vector<int32_t> actual_node = run_mask_node_cpu(n_q_blocks,
                                                              n_k_blocks,
                                                              block_q,
                                                              block_k,
                                                              score,
                                                              causal,
                                                              protected_prefix_len,
                                                              n_heads,
                                                              n_kv_heads);
    ggml_free(ctx);

    GGML_ASSERT(expected.size() == actual.size());
    GGML_ASSERT(expected.size() == actual_node.size());
    int diff_count = 0;
    size_t first_diff = 0;
    for (size_t i = 0; i < expected.size(); ++i) {
        if (expected[i] != actual[i]) {
            if (diff_count == 0) {
                first_diff = i;
            }
            ++diff_count;
        }
    }
    for (size_t i = 0; i < expected.size(); ++i) {
        if (expected[i] != actual_node[i]) {
            std::fprintf(stderr,
                         "score-to-mask node mismatch shape=[%d,%d] keep_ratio=%.3f protected=%d first_diff=%zu expected=%d actual=%d\n",
                         n_q_blocks,
                         n_k_blocks,
                         keep_ratio,
                         protected_prefix_len,
                         i,
                         expected[i],
                         actual_node[i]);
            GGML_ASSERT(false);
        }
    }
    if (diff_count != 0) {
        std::fprintf(stderr,
                     "score-to-mask mismatch shape=[%d,%d] keep_ratio=%.3f protected=%d first_diff=%zu expected=%d actual=%d diff_count=%d\n",
                     n_q_blocks,
                     n_k_blocks,
                     keep_ratio,
                     protected_prefix_len,
                     first_diff,
                     expected[first_diff],
                     actual[first_diff],
                     diff_count);
        GGML_ASSERT(false);
    }
    std::printf("[PASS] score-to-mask reference shape=[%d,%d] keep_ratio=%.3f protected=%d\n",
                n_q_blocks,
                n_k_blocks,
                keep_ratio,
                protected_prefix_len);
}

static std::vector<ggml_fp16_t> read_f16_tensor(const ggml_tensor * t) {
    std::vector<ggml_fp16_t> data(size_t(ggml_nelements(t)));
    ggml_backend_tensor_get(t, data.data(), 0, data.size() * sizeof(ggml_fp16_t));
    return data;
}

static std::vector<ggml_fp16_t> run_score_node(ggml_backend_t backend,
                                               int n_q_blocks,
                                               int n_k_blocks,
                                               int head_dim,
                                               int block_q,
                                               int block_k,
                                               const std::vector<ggml_fp16_t> & q_mean,
                                               const std::vector<ggml_fp16_t> & k_mean) {
    ggml_context * ctx = make_ctx(8);

    ggml_tensor * q_mean_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, head_dim, n_q_blocks);
    ggml_tensor * k_mean_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, head_dim, n_k_blocks);
    ggml_tensor * score_t = ggml_ext_block_sparse_score_topk_node(ctx, q_mean_t, k_mean_t, block_q, block_k, 2, false);
    GGML_ASSERT(q_mean_t && k_mean_t && score_t);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_get_default_buffer_type(backend));
    GGML_ASSERT(buffer != nullptr);

    ggml_backend_tensor_set(q_mean_t, q_mean.data(), 0, q_mean.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(k_mean_t, k_mean.data(), 0, k_mean.size() * sizeof(ggml_fp16_t));

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, score_t);
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend);

    std::vector<ggml_fp16_t> score = read_f16_tensor(score_t);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return score;
}

static void compare_scores(const std::vector<ggml_fp16_t> & expected,
                           const std::vector<ggml_fp16_t> & actual,
                           int n_q_blocks,
                           int n_k_blocks) {
    GGML_ASSERT(expected.size() == actual.size());
    double max_diff = 0.0;
    size_t worst = 0;
    for (size_t i = 0; i < expected.size(); ++i) {
        const double diff = std::fabs(double(ggml_fp16_to_fp32(expected[i])) -
                                      double(ggml_fp16_to_fp32(actual[i])));
        if (diff > max_diff) {
            max_diff = diff;
            worst = i;
        }
    }
    if (max_diff > 1e-2) {
        std::fprintf(stderr,
                     "block sparse score mismatch at %zu: cpu=%.8f htp=%.8f diff=%.3e shape=[%d,%d]\n",
                     worst,
                     ggml_fp16_to_fp32(expected[worst]),
                     ggml_fp16_to_fp32(actual[worst]),
                     max_diff,
                     n_q_blocks,
                     n_k_blocks);
        GGML_ASSERT(false);
    }
    std::printf("[PASS] block sparse score shape=[%d,%d] max_diff=%.3e\n", n_q_blocks, n_k_blocks, max_diff);
}

static void test_flat_mean_score_shape(int qo_len, int kv_len, int n_heads, int n_kv_heads, int head_dim) {
    constexpr int block_q = 32;
    constexpr int block_k = 32;
    const int n_q_blocks = ceil_div_int(qo_len, block_q) * n_heads;
    const int n_k_blocks = ceil_div_int(kv_len, block_k) * n_kv_heads;

    ggml_context * ctx = make_ctx(16);
    ggml_tensor * q_t = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, qo_len, n_heads);
    ggml_tensor * k_t = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, head_dim, kv_len, n_kv_heads);
    ggml_tensor * q_mean_t = ggml_ext_block_sparse_q_mean_node(ctx, q_t, block_q);
    ggml_tensor * k_mean_t = ggml_ext_block_sparse_k_mean_node(ctx, k_t, block_k);
    ggml_tensor * score_t = ggml_ext_block_sparse_score_topk_node(ctx, q_mean_t, k_mean_t, block_q, block_k, 2, false);
    GGML_ASSERT(q_mean_t != nullptr && k_mean_t != nullptr && score_t != nullptr);
    if (q_mean_t->ne[0] != head_dim || q_mean_t->ne[1] != n_q_blocks || q_mean_t->ne[2] != 1 ||
        k_mean_t->ne[0] != head_dim || k_mean_t->ne[1] != n_k_blocks || k_mean_t->ne[2] != 1 ||
        score_t->ne[0] != n_k_blocks || score_t->ne[1] != n_q_blocks) {
        std::fprintf(stderr,
                     "flat block shape mismatch: qo=%d kv=%d heads=%d kv_heads=%d d=%d "
                     "q_mean=[%lld,%lld,%lld] expected=[%d,%d,1] "
                     "k_mean=[%lld,%lld,%lld] expected=[%d,%d,1] "
                     "score=[%lld,%lld] expected=[%d,%d]\n",
                     qo_len,
                     kv_len,
                     n_heads,
                     n_kv_heads,
                     head_dim,
                     (long long) q_mean_t->ne[0],
                     (long long) q_mean_t->ne[1],
                     (long long) q_mean_t->ne[2],
                     head_dim,
                     n_q_blocks,
                     (long long) k_mean_t->ne[0],
                     (long long) k_mean_t->ne[1],
                     (long long) k_mean_t->ne[2],
                     head_dim,
                     n_k_blocks,
                     (long long) score_t->ne[0],
                     (long long) score_t->ne[1],
                     n_k_blocks,
                     n_q_blocks);
        GGML_ASSERT(false);
    }
    ggml_free(ctx);

    std::printf("[PASS] flat mean/score shape qo=%d kv=%d heads=%d kv_heads=%d score=[%d,%d]\n",
                qo_len,
                kv_len,
                n_heads,
                n_kv_heads,
                n_k_blocks,
                n_q_blocks);
}

static void test_score_shape(int qo_len, int kv_len, int n_heads, int n_kv_heads, int head_dim) {
    constexpr int block_q = 32;
    constexpr int block_k = 32;
    const int n_q_blocks = ceil_div_int(qo_len, block_q) * n_heads;
    const int n_k_blocks = ceil_div_int(kv_len, block_k) * n_kv_heads;

    ggml_backend_t htp = ggml_backend_init_by_name("HTP", nullptr);
    GGML_ASSERT(htp != nullptr);

    if (!ggml_backend_htp_runtime_available()) {
        const char * err = ggml_backend_htp_runtime_error();
        std::printf("[SKIP] HTP runtime unavailable (%s)\n", err ? err : "unknown");
        ggml_backend_free(htp);
        return;
    }

    {
        ggml_context * probe_ctx = make_ctx(8);
        ggml_tensor * q_mean_t = ggml_new_tensor_2d(probe_ctx, GGML_TYPE_F16, head_dim, n_q_blocks);
        ggml_tensor * k_mean_t = ggml_new_tensor_2d(probe_ctx, GGML_TYPE_F16, head_dim, n_k_blocks);
        ggml_tensor * score_t = ggml_ext_block_sparse_score_topk_node(probe_ctx, q_mean_t, k_mean_t, block_q, block_k, 2, false);
        GGML_ASSERT(score_t != nullptr);
        GGML_ASSERT(ggml_backend_supports_op(htp, score_t));
        ggml_free(probe_ctx);
    }

    ggml_backend_t cpu = ggml_backend_init_by_name("CPU", nullptr);
    GGML_ASSERT(cpu != nullptr);

    const std::vector<ggml_fp16_t> q_mean = make_mean_data(head_dim, n_q_blocks, 1);
    const std::vector<ggml_fp16_t> k_mean = make_mean_data(head_dim, n_k_blocks, 2);
    std::vector<ggml_fp16_t> expected = run_score_node(cpu, n_q_blocks, n_k_blocks, head_dim, block_q, block_k, q_mean, k_mean);
    std::vector<ggml_fp16_t> actual = run_score_node(htp, n_q_blocks, n_k_blocks, head_dim, block_q, block_k, q_mean, k_mean);

    compare_scores(expected, actual, n_q_blocks, n_k_blocks);

    ggml_backend_free(cpu);
    ggml_backend_free(htp);
}

int main() {
    test_patch_major_mapping_roundtrip();
    test_patch_major_shift_mapping_roundtrip();
    test_patch_major_reorder_tensor_roundtrip();
    test_patch_major_shifted_reorder_tensor_roundtrip();
    test_patch_major_shifted_to_shifted_reorder_tensor_roundtrip();
    test_block_sparse_step_shift_policy();
    test_block_sparse_dense_interval_policy();
    test_block_sparse_residual_mode_policy();
    test_block_sparse_residual_warmup_anchor_policy();
    test_block_sparse_residual_anchor_has_sparse_consumer_policy();
    test_block_sparse_residual_cache_key_policy();
    test_patch_score_mask_promotes_to_dsp_index();
    test_patch_score_mask_2d_halo_keeps_surrounding_blocks();
    test_patch_score_mask_center_sink_keeps_center_blocks();
    test_patch_score_node_promotes_to_dsp_index();
    test_score_to_mask_reference(24, 24, 0.95f, 0);
    test_score_to_mask_reference(192, 192, 0.95f, 0);
    test_score_to_mask_reference(312, 312, 0.95f, 154);
    test_score_to_index_equivalence(256, 256, 1.0f, 0);
    test_score_to_index_equivalence(256, 256, 0.98f, 0);
    test_score_to_index_equivalence(256, 256, 0.95f, 0);
    test_score_to_index_equivalence(256, 256, 0.90f, 0);
    test_score_to_index_equivalence(410, 410, 1.0f, 154);
    test_score_to_index_equivalence(410, 410, 0.98f, 154);
    test_score_to_index_equivalence(410, 410, 0.95f, 154);
    test_score_to_index_equivalence(410, 410, 0.90f, 154);
    test_score_to_index_suffix_protection();
    test_flat_mean_score_shape(256, 256, 24, 24, 64);
    test_flat_mean_score_shape(410, 410, 24, 24, 64);
    test_score_shape(96, 128, 24, 24, 64);
    test_score_shape(410, 410, 24, 24, 64);
    std::printf("\n=== HTP BLOCK_SPARSE_SCORE test PASSED ===\n");
    return 0;
}
