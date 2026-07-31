#include "ggml-htp.h"
#include "ggml-opencl-htp.h"
#include "ggml-opencl.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>

namespace {

constexpr int64_t k_head_dim = 64;
constexpr int64_t k_default_heads = 1;
constexpr int64_t k_block_size = 32;
constexpr int k_warmup_runs = 4;
constexpr int k_measured_runs = 20;

int64_t benchmark_heads() {
    static const int64_t heads = [] {
        const char * value = std::getenv("ATTN_PERF_HEADS");
        if (value == nullptr || value[0] == '\0') {
            return k_default_heads;
        }
        char * end = nullptr;
        const long long parsed = std::strtoll(value, &end, 10);
        GGML_ASSERT(end != value && *end == '\0' && parsed > 0);
        return int64_t(parsed);
    }();
    return heads;
}

struct benchmark_case {
    const char * name;
    int64_t sequence_length;
    int image_height;
    int image_width;
    int text_tokens;
};

struct htp_tensor {
    ggml_backend_buffer_t buffer = nullptr;
    ggml_backend_htp_rpcmem_info info = {};
};

struct timing_stats {
    double mean_ms = 0.0;
    double p50_ms = 0.0;
    double p95_ms = 0.0;
};

htp_tensor allocate_htp_tensor(ggml_backend_t htp, ggml_tensor * tensor) {
    htp_tensor result;
    const size_t size = ggml_backend_buft_get_alloc_size(
        ggml_backend_get_default_buffer_type(htp), tensor);
    result.buffer = ggml_backend_alloc_buffer(htp, size);
    GGML_ASSERT(result.buffer != nullptr);
    GGML_ASSERT(ggml_backend_htp_get_rpcmem_info(result.buffer, &result.info));
    GGML_ASSERT(ggml_backend_tensor_alloc(result.buffer, tensor, result.info.base) == GGML_STATUS_SUCCESS);
    return result;
}

ggml_opencl_htp_import_descriptor import_descriptor(const htp_tensor & tensor) {
    return {
        /* .backing         = */ tensor.buffer,
        /* .base            = */ tensor.info.base,
        /* .fd              = */ tensor.info.fd,
        /* .logical_size    = */ tensor.info.logical_size,
        /* .allocation_size = */ tensor.info.allocation_size,
    };
}

void fill_qkv(
    ggml_tensor * q,
    ggml_tensor * k,
    ggml_tensor * v,
    int64_t sequence_length) {
    const int64_t heads = benchmark_heads();
    std::vector<float> q_values(static_cast<size_t>(k_head_dim * sequence_length * heads));
    std::vector<ggml_fp16_t> k_values(static_cast<size_t>(k_head_dim * sequence_length * heads));
    std::vector<ggml_fp16_t> v_values(k_values.size());
    for (int64_t head = 0; head < heads; ++head) {
        for (int64_t token = 0; token < sequence_length; ++token) {
            for (int64_t dim = 0; dim < k_head_dim; ++dim) {
                const size_t index = static_cast<size_t>(
                    dim + k_head_dim * (token + sequence_length * head));
                q_values[index] = 0.35f * std::sin(0.013f * float((dim + 1) * (token + 1)) +
                                                    0.19f * float(head));
                k_values[index] = ggml_fp32_to_fp16(
                    0.42f * std::cos(0.017f * float((dim + 3) * (token + 1)) +
                                      0.11f * float(head)));
                v_values[index] = ggml_fp32_to_fp16(
                    0.37f * std::sin(0.023f * float((dim + 5) * (token + 1)) +
                                      0.07f * float(head)));
            }
        }
    }
    ggml_backend_tensor_set(q, q_values.data(), 0, q_values.size() * sizeof(float));
    ggml_backend_tensor_set(k, k_values.data(), 0, k_values.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(v, v_values.data(), 0, v_values.size() * sizeof(ggml_fp16_t));
}

std::vector<int32_t> make_direct_halo_mask(const benchmark_case & c, double * kept_ratio) {
    const int64_t heads = benchmark_heads();
    GGML_ASSERT(c.sequence_length == int64_t(c.image_height) * c.image_width + c.text_tokens);
    GGML_ASSERT(c.image_height % 4 == 0 && c.image_width % 8 == 0);
    const int64_t blocks = (c.sequence_length + k_block_size - 1) / k_block_size;
    const int image_block_rows = c.image_height / 4;
    const int image_block_columns = c.image_width / 8;
    const int image_blocks = image_block_rows * image_block_columns;
    GGML_ASSERT(image_blocks + (c.text_tokens + k_block_size - 1) / k_block_size == blocks);

    std::vector<int32_t> index(static_cast<size_t>(heads * blocks * blocks), -1);
    int64_t kept = 0;
    for (int64_t head = 0; head < heads; ++head) {
        for (int64_t query_block = 0; query_block < blocks; ++query_block) {
            int32_t * row = index.data() + (head * blocks + query_block) * blocks;
            int count = 0;
            if (query_block >= image_blocks) {
                for (int64_t key_block = 0; key_block < blocks; ++key_block) {
                    row[count++] = static_cast<int32_t>(head * blocks + key_block);
                }
            } else {
                const int query_y = int(query_block) / image_block_columns;
                const int query_x = int(query_block) % image_block_columns;
                for (int key_y = std::max(0, query_y - 1);
                     key_y <= std::min(image_block_rows - 1, query_y + 1);
                     ++key_y) {
                    for (int key_x = std::max(0, query_x - 1);
                         key_x <= std::min(image_block_columns - 1, query_x + 1);
                         ++key_x) {
                        row[count++] = static_cast<int32_t>(
                            head * blocks + key_y * image_block_columns + key_x);
                    }
                }
                for (int64_t text_block = image_blocks; text_block < blocks; ++text_block) {
                    row[count++] = static_cast<int32_t>(head * blocks + text_block);
                }
            }
            kept += count;
        }
    }
    *kept_ratio = double(kept) / double(heads * blocks * blocks);
    return index;
}

timing_stats summarize(std::vector<int64_t> samples_us) {
    GGML_ASSERT(!samples_us.empty());
    std::sort(samples_us.begin(), samples_us.end());
    const double sum_us = std::accumulate(samples_us.begin(), samples_us.end(), 0.0);
    const size_t p95_index = (samples_us.size() * 95 + 99) / 100 - 1;
    return {
        /* .mean_ms = */ sum_us / double(samples_us.size()) / 1000.0,
        /* .p50_ms  = */ double(samples_us[samples_us.size() / 2]) / 1000.0,
        /* .p95_ms  = */ double(samples_us[p95_index]) / 1000.0,
    };
}

template<typename Run>
timing_stats time_synchronous(Run && run) {
    for (int i = 0; i < k_warmup_runs; ++i) {
        run();
    }
    std::vector<int64_t> samples_us;
    samples_us.reserve(k_measured_runs);
    for (int i = 0; i < k_measured_runs; ++i) {
        const int64_t begin_us = ggml_time_us();
        run();
        samples_us.push_back(ggml_time_us() - begin_us);
    }
    return summarize(std::move(samples_us));
}

timing_stats benchmark_opencl_dense(ggml_backend_t opencl, const benchmark_case & c) {
    const int64_t heads = benchmark_heads();
    const ggml_init_params params = {
        /* .mem_size   = */ 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    GGML_ASSERT(ctx != nullptr);
    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, k_head_dim, c.sequence_length, heads, 1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, k_head_dim, c.sequence_length, heads, 1);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, k_head_dim, c.sequence_length, heads, 1);
    const int64_t padded_k = ((c.sequence_length + 63) / 64) * 64;
    ggml_tensor * mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, padded_k, c.sequence_length, 1, 1);
    ggml_tensor * output = ggml_flash_attn_ext(
        ctx, q, k, v, mask, 1.0f / std::sqrt(float(k_head_dim)), 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(output, GGML_PREC_F32);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors_from_buft(
        ctx, ggml_backend_get_default_buffer_type(opencl));
    GGML_ASSERT(buffer != nullptr);

    fill_qkv(q, k, v, c.sequence_length);
    std::vector<ggml_fp16_t> dense_mask(static_cast<size_t>(padded_k * c.sequence_length),
                                         ggml_fp32_to_fp16(-INFINITY));
    for (int64_t token = 0; token < c.sequence_length; ++token) {
        std::fill_n(dense_mask.data() + token * padded_k, c.sequence_length, ggml_fp32_to_fp16(0.0f));
    }
    ggml_backend_tensor_set(mask, dense_mask.data(), 0, dense_mask.size() * sizeof(ggml_fp16_t));
    GGML_ASSERT(ggml_backend_supports_op(opencl, output));
    const timing_stats result = time_synchronous([&] {
        GGML_ASSERT(ggml_backend_graph_compute(opencl, graph) == GGML_STATUS_SUCCESS);
        ggml_backend_synchronize(opencl);
    });
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return result;
}

timing_stats benchmark_htp(
    ggml_backend_t htp,
    const benchmark_case & c,
    bool sparse,
    double * kept_ratio) {
    const int64_t heads = benchmark_heads();
    const ggml_init_params params = {
        /* .mem_size   = */ 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    GGML_ASSERT(ctx != nullptr);
    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, k_head_dim, c.sequence_length, heads, 1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, k_head_dim, c.sequence_length, heads, 1);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, k_head_dim, c.sequence_length, heads, 1);
    const int64_t blocks = (c.sequence_length + k_block_size - 1) / k_block_size;
    std::vector<int32_t> sparse_mask;
    std::vector<ggml_fp16_t> dense_mask;
    ggml_tensor * mask = nullptr;
    if (sparse) {
        sparse_mask = make_direct_halo_mask(c, kept_ratio);
        mask = ggml_new_tensor_4d(ctx, GGML_TYPE_I32, blocks, heads * blocks, 1, 1);
    } else {
        const int64_t padded_k = ((c.sequence_length + 63) / 64) * 64;
        dense_mask.assign(static_cast<size_t>(padded_k * c.sequence_length), ggml_fp32_to_fp16(0.0f));
        mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, padded_k, c.sequence_length, 1, 1);
    }
    ggml_tensor * output = ggml_flash_attn_ext_with_lse(
        ctx, q, k, v, mask, 1.0f / std::sqrt(float(k_head_dim)), 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(output, GGML_PREC_F32);
    if (sparse) {
        ggml_flash_attn_ext_set_block_sparse(
            output, k_block_size, k_block_size, GGML_FLASH_ATTN_EXT_MASK_BLOCK_INDEX);
        if (c.text_tokens > 0) {
            ggml_flash_attn_ext_set_static_halo_2d(
                output,
                c.image_height * c.image_width,
                c.text_tokens,
                c.image_height / 4,
                c.image_width / 8);
        }
    }
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors_from_buft(
        ctx, ggml_backend_get_default_buffer_type(htp));
    GGML_ASSERT(buffer != nullptr);
    fill_qkv(q, k, v, c.sequence_length);
    if (sparse) {
        std::memcpy(mask->data, sparse_mask.data(), sparse_mask.size() * sizeof(int32_t));
    } else {
        ggml_backend_tensor_set(mask, dense_mask.data(), 0, dense_mask.size() * sizeof(ggml_fp16_t));
    }
    GGML_ASSERT(ggml_backend_supports_op(htp, output));
    const bool hybrid_was_enabled = ggml_backend_htp_get_hybrid_enabled();
    GGML_ASSERT(setenv("SD_HTP_SPARSE_OPENCL", "0", 1) == 0);
    ggml_backend_htp_set_hybrid_enabled(true);
    const timing_stats result = time_synchronous([&] {
        GGML_ASSERT(ggml_backend_graph_compute(htp, graph) == GGML_STATUS_SUCCESS);
        ggml_backend_synchronize(htp);
    });
    ggml_backend_htp_set_hybrid_enabled(hybrid_was_enabled);
    GGML_ASSERT(setenv("SD_HTP_SPARSE_OPENCL", "1", 1) == 0);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return result;
}

timing_stats benchmark_sparse(
    ggml_opencl_htp_sidecar * sidecar,
    ggml_backend_t htp,
    const benchmark_case & c,
    double * kept_ratio) {
    const int64_t heads = benchmark_heads();
    const ggml_init_params params = {
        /* .mem_size   = */ 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    GGML_ASSERT(ctx != nullptr);
    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, k_head_dim, c.sequence_length, heads, 1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, k_head_dim, c.sequence_length, heads, 1);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, k_head_dim, c.sequence_length, heads, 1);
    const int64_t blocks = (c.sequence_length + k_block_size - 1) / k_block_size;
    const std::vector<int32_t> sparse_mask = make_direct_halo_mask(c, kept_ratio);
    ggml_tensor * mask = ggml_new_tensor_4d(ctx, GGML_TYPE_I32, blocks, heads * blocks, 1, 1);
    ggml_tensor * output = ggml_flash_attn_ext_with_lse(
        ctx, q, k, v, mask, 1.0f / std::sqrt(float(k_head_dim)), 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(output, GGML_PREC_F32);
    ggml_flash_attn_ext_set_block_sparse(
        output, k_block_size, k_block_size, GGML_FLASH_ATTN_EXT_MASK_BLOCK_INDEX);

    htp_tensor q_buffer = allocate_htp_tensor(htp, q);
    htp_tensor k_buffer = allocate_htp_tensor(htp, k);
    htp_tensor v_buffer = allocate_htp_tensor(htp, v);
    htp_tensor mask_buffer = allocate_htp_tensor(htp, mask);
    htp_tensor output_buffer = allocate_htp_tensor(htp, output);
    fill_qkv(q, k, v, c.sequence_length);
    std::memcpy(mask->data, sparse_mask.data(), sparse_mask.size() * sizeof(int32_t));

    ggml_cgraph * cgraph = ggml_new_graph(ctx);
    ggml_graph_add_node(cgraph, output);
    const std::vector<ggml_opencl_htp_import_descriptor> descriptors = {
        import_descriptor(q_buffer), import_descriptor(k_buffer), import_descriptor(v_buffer),
        import_descriptor(mask_buffer), import_descriptor(output_buffer),
    };
    char reason[1024] = {};
    ggml_opencl_htp_graph * graph = nullptr;
    GGML_ASSERT(ggml_backend_opencl_htp_graph_preflight(
        sidecar, cgraph, descriptors.data(), descriptors.size(), &graph, reason, sizeof(reason)) ==
                GGML_OPENCL_HTP_OK);
    GGML_ASSERT(graph != nullptr);
    const timing_stats result = time_synchronous([&] {
        reason[0] = '\0';
        GGML_ASSERT(ggml_backend_opencl_htp_graph_compute_block_sparse_attn_sync(
            graph, output, reason, sizeof(reason)) == GGML_OPENCL_HTP_OK);
    });
    GGML_ASSERT(ggml_backend_opencl_htp_graph_finish(graph, reason, sizeof(reason)) == GGML_OPENCL_HTP_OK);
    ggml_backend_opencl_htp_graph_free(&graph);
    ggml_backend_buffer_free(output_buffer.buffer);
    ggml_backend_buffer_free(mask_buffer.buffer);
    ggml_backend_buffer_free(v_buffer.buffer);
    ggml_backend_buffer_free(k_buffer.buffer);
    ggml_backend_buffer_free(q_buffer.buffer);
    ggml_free(ctx);
    return result;
}

void run_case(
    ggml_opencl_htp_sidecar * sidecar,
    ggml_backend_t htp,
    ggml_backend_t opencl,
    const char * attention_type,
    const benchmark_case & c) {
    const char * only_backend = std::getenv("ATTN_PERF_BACKEND");
    auto backend_selected = [only_backend](const char * name) {
        return only_backend == nullptr || only_backend[0] == '\0' || std::strcmp(only_backend, name) == 0;
    };
    double kept_ratio = 0.0;
    auto run_backend = [&](const char * name, auto && run) {
        timing_stats result;
        if (!backend_selected(name)) {
            return result;
        }
        std::fprintf(stderr, "[ATTN_PERF_STAGE] resolution=%s type=%s backend=%s begin\n",
                     c.name, attention_type, name);
        std::fflush(stderr);
        result = run();
        std::printf(
            "[ATTN_PERF_RESULT] resolution=%s type=%s qo=%lld heads=%lld backend=%s "
            "mean=%.3fms p50=%.3fms p95=%.3fms\n",
            c.name, attention_type, (long long) c.sequence_length, (long long) benchmark_heads(), name,
            result.mean_ms, result.p50_ms, result.p95_ms);
        std::fflush(stdout);
        return result;
    };
    run_backend("htp_dense_lse", [&] { return benchmark_htp(htp, c, false, nullptr); });
    run_backend("htp_sparse_lse", [&] { return benchmark_htp(htp, c, true, &kept_ratio); });
    run_backend("opencl_dense", [&] { return benchmark_opencl_dense(opencl, c); });
    run_backend("opencl_sparse_lse", [&] { return benchmark_sparse(sidecar, htp, c, &kept_ratio); });
    if (backend_selected("htp_sparse_lse") || backend_selected("opencl_sparse_lse")) {
        std::printf(
            "[ATTN_PERF_MASK] resolution=%s type=%s direct_halo_kept=%.2f%%\n",
            c.name, attention_type, kept_ratio * 100.0);
        std::fflush(stdout);
    }
}

} // namespace

int main() {
    GGML_ASSERT(setenv("SD_HTP_SPARSE_OPENCL", "1", 1) == 0);
    GGML_ASSERT(setenv("SD_HTP_OPS", "1", 1) == 0);
    GGML_ASSERT(setenv("SD_HTP_COMPUTE", "1", 1) == 0);
    GGML_ASSERT(setenv("SD_HTP_ATTENTION", "1", 1) == 0);
    GGML_ASSERT(setenv("SD_HTP_BLOCK_SPARSE_ATTN", "1", 1) == 0);

    char reason[1024] = {};
    ggml_opencl_qcom_requirements requirements = {};
    ggml_opencl_htp_sidecar * sidecar =
        ggml_backend_opencl_htp_sidecar_create(&requirements, reason, sizeof(reason));
    if (sidecar == nullptr) {
        std::fprintf(stderr, "OpenCL HTP sidecar creation failed: %s\n", reason);
        return 1;
    }
    ggml_backend_t htp = ggml_backend_htp_init();
    ggml_backend_t opencl = ggml_backend_opencl_init();
    GGML_ASSERT(htp != nullptr && ggml_backend_htp_runtime_available());
    GGML_ASSERT(opencl != nullptr);

    const char * only = std::getenv("ATTN_PERF_ONLY");
    auto selected = [only](const char * name) {
        return only == nullptr || only[0] == '\0' || std::strcmp(only, name) == 0;
    };
    // These are the actual joint and self-attention sequence shapes for each resolution.
    if (selected("256-joint")) {
        run_case(sidecar, htp, opencl, "joint", { "256", 410, 16, 16, 154 });
    }
    if (selected("256-self")) {
        run_case(sidecar, htp, opencl, "self", { "256", 256, 16, 16, 0 });
    }
    if (selected("512-joint")) {
        run_case(sidecar, htp, opencl, "joint", { "512", 1178, 32, 32, 154 });
    }
    if (selected("512-self")) {
        run_case(sidecar, htp, opencl, "self", { "512", 1024, 32, 32, 0 });
    }
    if (selected("1024-joint")) {
        run_case(sidecar, htp, opencl, "joint", { "1024", 4250, 64, 64, 154 });
    }
    if (selected("1024-self")) {
        run_case(sidecar, htp, opencl, "self", { "1024", 4096, 64, 64, 0 });
    }
    if (selected("2048-joint")) {
        run_case(sidecar, htp, opencl, "joint", { "2048", 16538, 128, 128, 154 });
    }
    if (selected("4096-joint")) {
        run_case(sidecar, htp, opencl, "joint", { "4096", 65690, 256, 256, 154 });
    }

    ggml_backend_free(opencl);
    ggml_backend_free(htp);
    ggml_backend_opencl_htp_sidecar_free(sidecar);
    return 0;
}
