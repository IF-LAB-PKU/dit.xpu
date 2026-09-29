// CPU attention kernel benchmark: dense vs block-sparse ("block halo").
//
// Both routes build the same q/k/v inputs from the shared attention fixture.
// The sparse route swaps the f16 mask for the I32 block index that the block
// halo policy produces and selects ggml_flash_attn_ext_with_lse, which sends
// the CPU backend to the block-index kernel. Timing covers only
// ggml_backend_graph_compute for the single attention node.

#include <ggml-backend.h>
#include <ggml-cpu.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "test-backend-htp-attention-common.hpp"

namespace {

enum class Route { Dense, Sparse };

struct TimingStats {
    double min_us = 0.0;
    double median_us = 0.0;
    double mean_us = 0.0;
    double p95_us = 0.0;
    double max_us = 0.0;
};

struct AttentionGraph {
    ggml_context * ctx = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_tensor * q_storage = nullptr;
    ggml_tensor * k_storage = nullptr;
    ggml_tensor * v_storage = nullptr;
    ggml_tensor * mask = nullptr;
    ggml_tensor * output = nullptr;
};

static const char * route_name(Route route) {
    return route == Route::Dense ? "cpu-dense" : "cpu-sparse-block-halo";
}

static int env_int(const char * name, int fallback, int minimum) {
    const char * value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') return fallback;
    char * end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    return end != value && end != nullptr && *end == '\0' && parsed >= minimum && parsed <= INT32_MAX
        ? static_cast<int>(parsed) : fallback;
}

static const char * env_filter(const char * cpu_name, const char * common_name) {
    const char * value = std::getenv(cpu_name);
    if (value != nullptr && value[0] != '\0') return value;
    return std::getenv(common_name);
}

static bool selected_cpu_case(const htp_attention_common::ImageSize & size,
                              const htp_attention_common::CaseSpec & spec) {
    const char * resolution = env_filter("CPU_ATTN_TEST_RESOLUTION", "HTP_ATTN_TEST_RESOLUTION");
    if (resolution != nullptr && std::atoi(resolution) != size.resolution) return false;
    const char * model = env_filter("CPU_ATTN_TEST_MODEL", "HTP_ATTN_TEST_MODEL");
    if (model != nullptr && std::strcmp(model, spec.model) != 0) return false;
    const char * family = env_filter("CPU_ATTN_TEST_FAMILY", "HTP_ATTN_TEST_FAMILY");
    if (family != nullptr && std::strcmp(family, spec.family) != 0) return false;
    return true;
}

static bool selected_route(Route route) {
    const char * selected = std::getenv("CPU_ATTN_SPARSE_ROUTE");
    if (selected == nullptr || selected[0] == '\0') return true;
    return std::strcmp(selected, route == Route::Dense ? "dense" : "sparse") == 0;
}

static TimingStats summarize(std::vector<int64_t> samples) {
    TimingStats result;
    std::sort(samples.begin(), samples.end());
    result.min_us = static_cast<double>(samples.front());
    result.median_us = static_cast<double>(samples[samples.size() / 2]);
    result.max_us = static_cast<double>(samples.back());
    double total = 0.0;
    for (const int64_t sample : samples) total += static_cast<double>(sample);
    result.mean_us = total / static_cast<double>(samples.size());
    const size_t p95_index = (samples.size() * 95 + 99) / 100 - 1;
    result.p95_us = static_cast<double>(samples[p95_index]);
    return result;
}

static bool parse_temperature(const std::string & dump, const char * key, double * value) {
    const std::string needle = std::string("mName=") + (key != nullptr ? key : "");
    for (size_t line_begin = 0; line_begin < dump.size();) {
        const size_t found_line_end = dump.find('\n', line_begin);
        const size_t line_end = found_line_end == std::string::npos ? dump.size() : found_line_end;
        const size_t pos = dump.find(needle, line_begin);
        if (pos < line_end) {
            const size_t name_end = pos + needle.size();
            const bool exact_name = name_end == line_end || dump[name_end] == ',' || dump[name_end] == '}';
            const size_t value_pos = dump.find("mValue=", line_begin);
            if (!exact_name || value_pos >= line_end) {
                if (line_end == dump.size()) break;
                line_begin = line_end + 1;
                continue;
            }
            char * parsed_end = nullptr;
            const char * value_begin = dump.c_str() + value_pos + 7;
            const double parsed = std::strtod(value_begin, &parsed_end);
            if (parsed_end != value_begin && parsed_end <= dump.c_str() + line_end && std::isfinite(parsed)) {
                *value = parsed;
                return true;
            }
        }
        if (line_end == dump.size()) break;
        line_begin = line_end + 1;
    }
    return false;
}

static bool read_device_temperature(double * value) {
#if defined(__ANDROID__)
    FILE * pipe = popen("/system/bin/dumpsys thermalservice 2>/dev/null", "r");
    if (pipe == nullptr) return false;
    std::string dump;
    char buffer[512];
    while (std::fgets(buffer, sizeof(buffer), pipe) != nullptr) dump += buffer;
    pclose(pipe);
    return parse_temperature(dump, "shell_skin", value) || parse_temperature(dump, "skin", value);
#else
    (void) value;
    return false;
#endif
}

static bool device_has_sd_cli() {
#if defined(__ANDROID__)
    FILE * pipe = popen("/system/bin/pidof sd-cli 2>/dev/null", "r");
    if (pipe == nullptr) return false;
    char buffer[128] = {};
    const bool running = std::fgets(buffer, sizeof(buffer), pipe) != nullptr && buffer[0] != '\0';
    pclose(pipe);
    return running;
#else
    return false;
#endif
}

static bool wait_for_cooldown(double * initial_temperature) {
    if (std::getenv("CPU_ATTN_PERF_SKIP_COOLDOWN") != nullptr) {
        *initial_temperature = -1.0;
        return true;
    }
#if !defined(__ANDROID__)
    std::fprintf(stderr, "cooldown requires Android; set CPU_ATTN_PERF_SKIP_COOLDOWN=1 for host-only runs\n");
    return false;
#else
    const double limit_c = env_int("CPU_ATTN_PERF_COOLDOWN_C", 40, 0);
    const int poll_ms = env_int("CPU_ATTN_PERF_COOLDOWN_POLL_MS", 5000, 100);
    const int timeout_s = env_int("CPU_ATTN_PERF_COOLDOWN_TIMEOUT_S", 1800, 1);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeout_s);
    while (std::chrono::steady_clock::now() < deadline) {
        double temperature = 0.0;
        if (!device_has_sd_cli() && read_device_temperature(&temperature) && temperature < limit_c) {
            double confirmed = 0.0;
            if (!device_has_sd_cli() && read_device_temperature(&confirmed) && confirmed < limit_c) {
                *initial_temperature = confirmed;
                return true;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(poll_ms));
    }
    std::fprintf(stderr, "cooldown timeout: device did not become idle below %.2f C\n", limit_c);
    return false;
#endif
}

static AttentionGraph build_graph(const htp_attention_common::CaseSpec & spec, Route route) {
    AttentionGraph result;
    const ggml_init_params params = {64 * 1024 * 1024, nullptr, true};
    result.ctx = ggml_init(params);
    if (result.ctx == nullptr) return result;

    const int tokens = htp_attention_common::total_tokens(spec);
    const int blocks = (tokens + htp_attention_common::kBlock - 1) / htp_attention_common::kBlock;

    result.q_storage = ggml_new_tensor_4d(
        result.ctx, GGML_TYPE_F32, spec.head_dim, spec.heads, tokens, 1);
    result.k_storage = ggml_new_tensor_4d(
        result.ctx, GGML_TYPE_F16, spec.head_dim, spec.heads, tokens, 1);
    result.v_storage = ggml_new_tensor_4d(
        result.ctx, GGML_TYPE_F16, spec.head_dim, spec.heads, tokens, 1);
    ggml_tensor * q = ggml_permute(result.ctx, result.q_storage, 0, 2, 1, 3);
    ggml_tensor * k = ggml_permute(result.ctx, result.k_storage, 0, 2, 1, 3);
    ggml_tensor * v = ggml_permute(result.ctx, result.v_storage, 0, 2, 1, 3);

    const float scale = 1.0f / std::sqrt(static_cast<float>(spec.head_dim));
    if (route == Route::Dense) {
        const int padded_tokens = ((tokens + 63) / 64) * 64;
        result.mask = ggml_new_tensor_2d(result.ctx, GGML_TYPE_F16, padded_tokens, tokens);
        result.output = ggml_flash_attn_ext(result.ctx, q, k, v, result.mask, scale, 0.0f, 0.0f);
        ggml_set_name(result.output, "cpu_dense_attention");
    } else {
        result.mask = ggml_new_tensor_4d(
            result.ctx, GGML_TYPE_I32, blocks, spec.heads * blocks, 1, 1);
        result.output = ggml_flash_attn_ext_with_lse(
            result.ctx, q, k, v, result.mask, scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_block_sparse(
            result.output,
            htp_attention_common::kBlock,
            htp_attention_common::kBlock,
            GGML_FLASH_ATTN_EXT_MASK_BLOCK_INDEX);
        ggml_flash_attn_ext_set_static_halo_2d(
            result.output,
            spec.image_tokens,
            spec.text_tokens,
            spec.image_block_rows,
            spec.image_block_cols);
        ggml_set_name(result.output, "cpu_sparse_block_halo_attention");
    }
    ggml_flash_attn_ext_set_prec(result.output, GGML_PREC_F32);
    result.graph = ggml_new_graph_custom(result.ctx, 16, false);
    ggml_build_forward_expand(result.graph, result.output);
    return result;
}

static bool upload_graph(AttentionGraph & graph,
                         const htp_attention_common::InputData & data,
                         const htp_attention_common::Reference & reference,
                         Route route,
                         ggml_backend_t backend) {
    graph.buffer = ggml_backend_alloc_ctx_tensors_from_buft(
        graph.ctx, ggml_backend_get_default_buffer_type(backend));
    if (graph.buffer == nullptr) return false;

    ggml_backend_tensor_set(graph.q_storage, reference.q.data(), 0,
                            reference.q.size() * sizeof(float));
    ggml_backend_tensor_set(graph.k_storage, reference.k.data(), 0,
                            reference.k.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(graph.v_storage, reference.v.data(), 0,
                            reference.v.size() * sizeof(ggml_fp16_t));

    if (route == Route::Sparse) {
        if (data.block_index.size() != size_t(ggml_nelements(graph.mask))) return false;
        ggml_backend_tensor_set(graph.mask, data.block_index.data(), 0,
                                data.block_index.size() * sizeof(int32_t));
    } else {
        const int tokens = graph.mask->ne[1];
        const int padded_tokens = graph.mask->ne[0];
        std::vector<ggml_fp16_t> mask(ggml_nelements(graph.mask), ggml_fp32_to_fp16(-INFINITY));
        for (int query = 0; query < tokens; ++query) {
            std::fill_n(mask.data() + static_cast<size_t>(query) * padded_tokens,
                        tokens, ggml_fp32_to_fp16(0.0f));
        }
        ggml_backend_tensor_set(graph.mask, mask.data(), 0,
                                mask.size() * sizeof(ggml_fp16_t));
    }
    return true;
}

static void free_graph(AttentionGraph & graph) {
    if (graph.buffer != nullptr) ggml_backend_buffer_free(graph.buffer);
    if (graph.ctx != nullptr) ggml_free(graph.ctx);
    graph = {};
}

// Verifies the kernel output against the fixture reference. The sparse route is
// checked against the same block index the kernel consumed, so a kernel that
// silently ran dense (or misread the index) cannot pass.
static bool validate_graph(const AttentionGraph & graph,
                           const htp_attention_common::CaseSpec & spec,
                           const htp_attention_common::InputData & data,
                           const htp_attention_common::Reference & reference,
                           Route route,
                           const char * model, const char * family, int resolution) {
    const int tokens = htp_attention_common::total_tokens(spec);
    const int head_dim = spec.head_dim;
    const int heads = spec.heads;

    std::vector<float> actual(ggml_nelements(graph.output));
    ggml_backend_tensor_get(graph.output, actual.data(), 0, actual.size() * sizeof(float));

    std::vector<float> expected_output;
    std::vector<float> expected_lse;
    if (route == Route::Sparse) {
        const htp_attention_common::AttentionReference expected =
            htp_attention_common::reference_attention_with_lse(spec, reference, &data.block_index);
        expected_output = expected.output;
        expected_lse = expected.lse;
    } else {
        expected_output = htp_attention_common::reference_attention(spec, reference, nullptr);
    }

    // ggml_flash_attn_ext() builds the result as { head_dim (+1 for LSE), heads, tokens }.
    const size_t token_bytes = graph.output->nb[2];
    const size_t head_bytes = graph.output->nb[1];
    double output_sq = 0.0;
    double lse_sq = 0.0;
    double output_max = 0.0;
    double lse_max = 0.0;
    for (int token = 0; token < tokens; ++token) {
        for (int head = 0; head < heads; ++head) {
            const float * row = reinterpret_cast<const float *>(
                reinterpret_cast<const char *>(actual.data()) + token * token_bytes + head * head_bytes);
            const size_t base = (size_t(token) * heads + head) * head_dim;
            for (int d = 0; d < head_dim; ++d) {
                const double error = double(row[d]) - double(expected_output[base + d]);
                output_max = std::max(output_max, std::fabs(error));
                output_sq += error * error;
            }
            if (route == Route::Sparse) {
                const double error = double(row[head_dim]) - double(expected_lse[size_t(token) * heads + head]);
                lse_max = std::max(lse_max, std::fabs(error));
                lse_sq += error * error;
            }
        }
    }
    const double output_rmse = std::sqrt(output_sq / double(expected_output.size()));
    const double lse_rmse = expected_lse.empty()
        ? 0.0 : std::sqrt(lse_sq / double(expected_lse.size()));

    const bool pass = output_max <= 5.0e-3 && output_rmse <= 5.0e-3 &&
                      lse_max <= 5.0e-3 && lse_rmse <= 5.0e-3;
    std::printf("[%s] model=%s family=%s resolution=%d route=%s output=(%.9g,%.9g) lse=(%.9g,%.9g)\n",
                pass ? "PASS" : "FAIL", model, family, resolution, route_name(route),
                output_max, output_rmse, lse_max, lse_rmse);
    std::fflush(stdout);
    return pass;
}

// Fraction of the (head, query block) rows of the index that keep a KV block.
static double kept_block_mean(const htp_attention_common::CaseSpec & spec,
                              const htp_attention_common::InputData & data) {
    const int tokens = htp_attention_common::total_tokens(spec);
    const int blocks = (tokens + htp_attention_common::kBlock - 1) / htp_attention_common::kBlock;
    const size_t rows = size_t(spec.heads) * blocks;
    size_t kept = 0;
    for (size_t row = 0; row < rows; ++row) {
        for (int p = 0; p < blocks; ++p) {
            if (data.block_index[row * blocks + p] < 0) break;
            ++kept;
        }
    }
    return rows > 0 ? double(kept) / double(rows) : 0.0;
}

static bool run_route(ggml_backend_t backend,
                      const htp_attention_common::ImageSize & size,
                      const htp_attention_common::CaseSpec & base,
                      Route route,
                      int warmups, int samples) {
    double initial_temperature = 0.0;
    if (!wait_for_cooldown(&initial_temperature)) return false;

    htp_attention_common::CaseSpec spec = base;
    spec.image_tokens = size.image_tokens;
    spec.image_block_rows = size.block_rows;
    spec.image_block_cols = size.block_cols;
    spec.resolution = size.resolution;

    AttentionGraph graph = build_graph(spec, route);
    if (graph.ctx == nullptr || graph.graph == nullptr || graph.output == nullptr) {
        std::fprintf(stderr, "[FAIL] graph setup model=%s family=%s resolution=%d route=%s\n",
                     spec.model, spec.family, spec.resolution, route_name(route));
        free_graph(graph);
        return false;
    }

    const htp_attention_common::InputData data = htp_attention_common::make_input(spec);
    htp_attention_common::Reference reference;
    for (const htp_attention_common::BranchData & branch : data.branches) {
        htp_attention_common::append_reference_branch(spec, branch, route == Route::Sparse, reference);
    }
    if (!upload_graph(graph, data, reference, route, backend)) {
        std::fprintf(stderr, "[FAIL] upload model=%s family=%s resolution=%d route=%s\n",
                     spec.model, spec.family, spec.resolution, route_name(route));
        free_graph(graph);
        return false;
    }

    for (int i = 0; i < warmups; ++i) {
        if (ggml_backend_graph_compute(backend, graph.graph) != GGML_STATUS_SUCCESS) {
            free_graph(graph);
            return false;
        }
    }

    const bool correct = validate_graph(graph, spec, data, reference, route,
                                        spec.model, spec.family, spec.resolution);

    std::vector<int64_t> elapsed;
    elapsed.reserve(static_cast<size_t>(samples));
    for (int i = 0; i < samples; ++i) {
        const auto begin = std::chrono::steady_clock::now();
        if (ggml_backend_graph_compute(backend, graph.graph) != GGML_STATUS_SUCCESS) {
            free_graph(graph);
            return false;
        }
        const auto end = std::chrono::steady_clock::now();
        elapsed.push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count());
    }

    const TimingStats stats = summarize(std::move(elapsed));
    std::printf("[PERF] model=%s family=%s resolution=%d image_tokens=%d route=%s "
                "keep_blocks_per_query=%.3f temp_c=%.2f samples=%d cpu_attention_min_us=%.1f "
                "cpu_attention_median_us=%.1f cpu_attention_mean_us=%.1f "
                "cpu_attention_p95_us=%.1f cpu_attention_max_us=%.1f\n",
                spec.model, spec.family, spec.resolution, spec.image_tokens, route_name(route),
                kept_block_mean(spec, data), initial_temperature, samples,
                stats.min_us, stats.median_us, stats.mean_us, stats.p95_us, stats.max_us);
    std::fflush(stdout);
    free_graph(graph);
    return correct;
}

} // namespace

int main() {
    ggml_backend_t backend = ggml_backend_cpu_init();
    if (backend == nullptr) {
        std::fprintf(stderr, "[FAIL] CPU backend unavailable\n");
        return 2;
    }
    ggml_backend_cpu_set_n_threads(
        backend, env_int("CPU_ATTN_KERNEL_THREADS", 4, 1));
    const int warmups = env_int("CPU_ATTN_KERNEL_WARMUPS", 3, 0);
    const int samples = env_int("CPU_ATTN_KERNEL_SAMPLES", 10, 1);

    int measured = 0;
    int expected = 0;
    for (int si = 0; si < htp_attention_common::image_size_count(); ++si) {
        const auto & size = htp_attention_common::image_sizes()[si];
        for (int ci = 0; ci < htp_attention_common::case_count(); ++ci) {
            const auto & spec = htp_attention_common::cases()[ci];
            if (!spec.sparse || !selected_cpu_case(size, spec)) continue;
            for (const Route route : {Route::Dense, Route::Sparse}) {
                if (!selected_route(route)) continue;
                ++expected;
                measured += run_route(backend, size, spec, route, warmups, samples) ? 1 : 0;
            }
        }
    }
    ggml_backend_free(backend);
    std::printf("TOTAL: %d/%d CPU dense/block-halo attention cases measured\n", measured, expected);
    return measured == expected ? 0 : 1;
}
