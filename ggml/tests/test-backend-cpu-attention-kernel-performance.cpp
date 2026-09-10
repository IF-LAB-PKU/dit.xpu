// CPU dense attention operator benchmark.
// Timing covers only ggml_backend_graph_compute for one FLASH_ATTN_EXT node.

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

static AttentionGraph build_graph(const htp_attention_common::CaseSpec & spec,
                                  const htp_attention_common::ImageSize & size) {
    AttentionGraph result;
    const ggml_init_params params = {64 * 1024 * 1024, nullptr, true};
    result.ctx = ggml_init(params);
    if (result.ctx == nullptr) return result;

    const int tokens = size.image_tokens + spec.text_tokens;
    result.q_storage = ggml_new_tensor_4d(
        result.ctx, GGML_TYPE_F32, spec.head_dim, spec.heads, tokens, 1);
    result.k_storage = ggml_new_tensor_4d(
        result.ctx, GGML_TYPE_F16, spec.head_dim, spec.heads, tokens, 1);
    result.v_storage = ggml_new_tensor_4d(
        result.ctx, GGML_TYPE_F16, spec.head_dim, spec.heads, tokens, 1);
    ggml_tensor * q = ggml_permute(result.ctx, result.q_storage, 0, 2, 1, 3);
    ggml_tensor * k = ggml_permute(result.ctx, result.k_storage, 0, 2, 1, 3);
    ggml_tensor * v = ggml_permute(result.ctx, result.v_storage, 0, 2, 1, 3);

    const int padded_tokens = ((tokens + 63) / 64) * 64;
    result.mask = ggml_new_tensor_2d(result.ctx, GGML_TYPE_F16, padded_tokens, tokens);
    result.output = ggml_flash_attn_ext(
        result.ctx, q, k, v, result.mask,
        1.0f / std::sqrt(static_cast<float>(spec.head_dim)), 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(result.output, GGML_PREC_F32);
    ggml_set_name(result.output, "cpu_dense_attention");
    result.graph = ggml_new_graph_custom(result.ctx, 16, false);
    ggml_build_forward_expand(result.graph, result.output);
    return result;
}

static bool upload_graph(AttentionGraph & graph,
                         const htp_attention_common::CaseSpec & spec,
                         const htp_attention_common::ImageSize & size,
                         ggml_backend_t backend) {
    graph.buffer = ggml_backend_alloc_ctx_tensors_from_buft(
        graph.ctx, ggml_backend_get_default_buffer_type(backend));
    if (graph.buffer == nullptr) return false;

    htp_attention_common::CaseSpec sized = spec;
    sized.image_tokens = size.image_tokens;
    sized.image_block_rows = size.block_rows;
    sized.image_block_cols = size.block_cols;
    sized.resolution = size.resolution;
    const htp_attention_common::InputData data = htp_attention_common::make_input(sized);
    htp_attention_common::Reference reference;
    for (const htp_attention_common::BranchData & branch : data.branches) {
        htp_attention_common::append_reference_branch(sized, branch, false, reference);
    }

    ggml_backend_tensor_set(graph.q_storage, reference.q.data(), 0,
                            reference.q.size() * sizeof(float));
    ggml_backend_tensor_set(graph.k_storage, reference.k.data(), 0,
                            reference.k.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(graph.v_storage, reference.v.data(), 0,
                            reference.v.size() * sizeof(ggml_fp16_t));

    const int tokens = graph.mask->ne[1];
    const int padded_tokens = graph.mask->ne[0];
    std::vector<ggml_fp16_t> mask(ggml_nelements(graph.mask), ggml_fp32_to_fp16(-INFINITY));
    for (int query = 0; query < tokens; ++query) {
        std::fill_n(mask.data() + static_cast<size_t>(query) * padded_tokens,
                    tokens, ggml_fp32_to_fp16(0.0f));
    }
    ggml_backend_tensor_set(graph.mask, mask.data(), 0,
                            mask.size() * sizeof(ggml_fp16_t));

    int attention_nodes = 0;
    for (int i = 0; i < ggml_graph_n_nodes(graph.graph); ++i) {
        const ggml_tensor * node = ggml_graph_node(graph.graph, i);
        if (node != nullptr && node->op == GGML_OP_FLASH_ATTN_EXT) ++attention_nodes;
    }
    return attention_nodes == 1 && graph.output->op == GGML_OP_FLASH_ATTN_EXT;
}

static void free_graph(AttentionGraph & graph) {
    if (graph.buffer != nullptr) ggml_backend_buffer_free(graph.buffer);
    if (graph.ctx != nullptr) ggml_free(graph.ctx);
    graph = {};
}

static bool run_case(ggml_backend_t backend,
                     const htp_attention_common::ImageSize & size,
                     const htp_attention_common::CaseSpec & base,
                     int warmups, int samples) {
    double initial_temperature = 0.0;
    if (!wait_for_cooldown(&initial_temperature)) return false;
    htp_attention_common::CaseSpec spec = base;
    spec.image_tokens = size.image_tokens;
    spec.image_block_rows = size.block_rows;
    spec.image_block_cols = size.block_cols;
    spec.resolution = size.resolution;
    AttentionGraph graph = build_graph(spec, size);
    if (graph.ctx == nullptr || graph.graph == nullptr || graph.output == nullptr ||
        !upload_graph(graph, spec, size, backend)) {
        std::fprintf(stderr, "[FAIL] graph setup model=%s family=%s resolution=%d\n",
                     spec.model, spec.family, spec.resolution);
        free_graph(graph);
        return false;
    }

    for (int i = 0; i < warmups; ++i) {
        if (ggml_backend_graph_compute(backend, graph.graph) != GGML_STATUS_SUCCESS) {
            free_graph(graph);
            return false;
        }
    }

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
    std::printf("[PERF] model=%s family=%s resolution=%d image_tokens=%d route=cpu-dense "
                "temp_c=%.2f samples=%d cpu_attention_min_us=%.1f cpu_attention_median_us=%.1f "
                "cpu_attention_mean_us=%.1f cpu_attention_p95_us=%.1f "
                "cpu_attention_max_us=%.1f\n",
                spec.model, spec.family, spec.resolution, spec.image_tokens, initial_temperature, samples,
                stats.min_us, stats.median_us, stats.mean_us, stats.p95_us, stats.max_us);
    std::fflush(stdout);
    free_graph(graph);
    return true;
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
            ++expected;
            measured += run_case(backend, size, spec, warmups, samples) ? 1 : 0;
        }
    }
    ggml_backend_free(backend);
    std::printf("TOTAL: %d/%d CPU dense attention cases measured\n", measured, expected);
    return measured == expected ? 0 : 1;
}
