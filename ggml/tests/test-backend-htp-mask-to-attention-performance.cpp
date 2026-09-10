// Measure route-specific mask/index preparation through HTP attention output.
// The graph contains only Q/K/V inputs, route preparation, and attention.

#include <ggml-backend.h>
#include <ggml-htp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "../../src/ggml_extend.hpp"

namespace {

struct CaseSpec {
    const char * model;
    const char * family;
    int heads;
    int head_dim;
    int text_tokens;
};

struct ImageSize {
    int resolution;
    int image_tokens;
    int block_rows;
    int block_cols;
};

enum class Route {
    Dense,
    Halo,
    TopP,
};

struct TimingStats {
    double min_us = 0.0;
    double median_us = 0.0;
    double mean_us = 0.0;
    double p95_us = 0.0;
    double max_us = 0.0;
};

struct Graph {
    ggml_context * ctx = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_tensor * q = nullptr;
    ggml_tensor * k = nullptr;
    ggml_tensor * v = nullptr;
    ggml_tensor * mask = nullptr;
    ggml_tensor * output = nullptr;
};

static const ImageSize kSizes[] = {
    {256, 256, 4, 2},
    {512, 1024, 8, 4},
    {1024, 4096, 16, 8},
};

static const CaseSpec kCases[] = {
    {"sd3.5", "joint", 24, 64, 32},
    {"sd3.5", "self2", 24, 64, 0},
    {"sd3.5-large", "joint", 38, 64, 32},
    {"sd3.5-large", "self2", 38, 64, 0},
    {"flux2", "double-stream", 24, 128, 32},
    {"flux2", "single-stream", 24, 128, 32},
    {"z-image-turbo", "joint", 30, 128, 32},
    {"z-image-turbo", "noise-refiner", 30, 128, 0},
    {"z-image-base", "joint", 30, 128, 32},
    {"z-image-base", "noise-refiner", 30, 128, 0},
};

static int env_int(const char * name, int fallback, int minimum) {
    const char * value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') return fallback;
    char * end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    return end != value && end != nullptr && *end == '\0' && parsed >= minimum
        ? static_cast<int>(parsed) : fallback;
}

static double env_double(const char * name, double fallback) {
    const char * value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') return fallback;
    char * end = nullptr;
    const double parsed = std::strtod(value, &end);
    return end != value && end != nullptr && *end == '\0' && std::isfinite(parsed)
        ? parsed : fallback;
}

static bool parse_temperature(const std::string & dump, const char * sensor, double * value) {
    const std::string name = std::string("mName=") + sensor;
    for (size_t begin = 0; begin < dump.size();) {
        const size_t found_end = dump.find('\n', begin);
        const size_t end = found_end == std::string::npos ? dump.size() : found_end;
        const size_t sensor_pos = dump.find(name, begin);
        if (sensor_pos < end) {
            // dumpsys thermalservice currently prints mValue before mName.
            const size_t value_pos = dump.find("mValue=", begin);
            if (value_pos >= begin && value_pos < sensor_pos) {
                char * parsed_end = nullptr;
                const char * value_begin = dump.c_str() + value_pos + 7;
                const double parsed = std::strtod(value_begin, &parsed_end);
                if (parsed_end != value_begin && parsed_end <= dump.c_str() + end && std::isfinite(parsed)) {
                    *value = parsed;
                    return true;
                }
            }
        }
        if (end == dump.size()) break;
        begin = end + 1;
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
    // Some Android builds make pidof match the invoking shell/command line;
    // inspect process comm names so the benchmark does not block on itself.
    FILE * pipe = popen("for p in /proc/[0-9]*; do c=$(cat \"$p/comm\" 2>/dev/null); [ \"$c\" = sd-cli ] && echo 1 && break; done", "r");
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
    if (std::getenv("HTP_MASK_ATTN_SKIP_COOLDOWN") != nullptr) {
        *initial_temperature = -1.0;
        return true;
    }
#if !defined(__ANDROID__)
    std::fprintf(stderr, "cooldown requires Android; set HTP_MASK_ATTN_SKIP_COOLDOWN=1 for host-only runs\n");
    return false;
#else
    const double limit_c = env_double("HTP_MASK_ATTN_COOLDOWN_C", 40.0);
    const int poll_ms = env_int("HTP_MASK_ATTN_COOLDOWN_POLL_MS", 5000, 100);
    const int timeout_s = env_int("HTP_MASK_ATTN_COOLDOWN_TIMEOUT_S", 1800, 1);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeout_s);
    while (std::chrono::steady_clock::now() < deadline) {
        double temperature = 0.0;
        if (read_device_temperature(&temperature) && temperature < limit_c) {
            double confirmed = 0.0;
            if (read_device_temperature(&confirmed) && confirmed < limit_c) {
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

static bool selected(const ImageSize & size, const CaseSpec & spec) {
    const char * value = std::getenv("HTP_MASK_ATTN_RESOLUTION");
    if (value != nullptr && value[0] != '\0' && std::atoi(value) != size.resolution) return false;
    value = std::getenv("HTP_MASK_ATTN_MODEL");
    if (value != nullptr && value[0] != '\0' && std::strcmp(value, spec.model) != 0) return false;
    value = std::getenv("HTP_MASK_ATTN_FAMILY");
    if (value != nullptr && value[0] != '\0' && std::strcmp(value, spec.family) != 0) return false;
    return true;
}

static const char * route_name(Route route) {
    switch (route) {
        case Route::Dense: return "htp-dense";
        case Route::Halo: return "halo-sparse";
        case Route::TopP: return "top-p-sparse";
    }
    return "unknown";
}

static bool selected_route(Route route) {
    const char * value = std::getenv("HTP_MASK_ATTN_ROUTE");
    return value == nullptr || value[0] == '\0' || std::strcmp(value, route_name(route)) == 0;
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

static void set_route_environment(Route route, double top_p) {
    setenv("SD_HTP_OPS", "1", 1);
    setenv("SD_HTP_COMPUTE", "1", 1);
    setenv("SD_HTP_ATTENTION", "1", 1);
    setenv("SD_HTP_BLOCK_SPARSE_RESIDUAL_CACHE", "0", 1);
    setenv("SD_HTP_BLOCK_SPARSE_RESIDUAL_EXACT_RHO", "0", 1);
    setenv("SD_HTP_SPARSE_OPENCL", "0", 1);
    setenv("SD_HTP_SPARSE_OPENCL_RESIDUAL", "0", 1);
    setenv("SD_HTP_HALO_KV_LOAD", "l2fetch", 1);
    setenv("SD_HTP_BLOCK_SPARSE_CENTER_SINK", "0", 1);
    if (route == Route::Dense) {
        setenv("SD_HTP_BLOCK_SPARSE_ATTN", "0", 1);
        setenv("SD_HTP_BLOCK_SPARSE_PATCH_HALO", "0", 1);
        setenv("SD_HTP_BLOCK_SPARSE_KEEP_RATIO", "1", 1);
    } else if (route == Route::Halo) {
        setenv("SD_HTP_BLOCK_SPARSE_ATTN", "1", 1);
        setenv("SD_HTP_BLOCK_SPARSE_PATCH_HALO", "1", 1);
        setenv("SD_HTP_BLOCK_SPARSE_KEEP_RATIO", "0", 1);
        setenv("SD_HTP_BLOCK_SPARSE_HOST_INDEX", "0", 1);
    } else {
        setenv("SD_HTP_BLOCK_SPARSE_ATTN", "1", 1);
        setenv("SD_HTP_BLOCK_SPARSE_PATCH_HALO", "0", 1);
        setenv("SD_HTP_BLOCK_SPARSE_KEEP_RATIO", std::to_string(top_p).c_str(), 1);
        setenv("SD_HTP_BLOCK_SPARSE_HOST_INDEX", "0", 1);
    }
}

static Graph build_graph(ggml_backend_t backend, const ImageSize & size,
                         const CaseSpec & spec, Route route, double top_p) {
    set_route_environment(route, top_p);
    Graph graph;
    const ggml_init_params params = {256 * 1024 * 1024, nullptr, true};
    graph.ctx = ggml_init(params);
    GGML_ASSERT(graph.ctx != nullptr);
    ggml_tensor * one = ggml_new_tensor_1d(graph.ctx, GGML_TYPE_F32, 1);
    ggml_set_name(one, "ggml_runner_build_in_tensor:one");

    const int tokens = size.image_tokens + spec.text_tokens;
    graph.q = ggml_new_tensor_3d(graph.ctx, GGML_TYPE_F32, spec.head_dim, tokens, spec.heads);
    graph.k = ggml_new_tensor_3d(graph.ctx, GGML_TYPE_F16, spec.head_dim, tokens, spec.heads);
    graph.v = ggml_new_tensor_3d(graph.ctx, GGML_TYPE_F16, spec.head_dim, tokens, spec.heads);
    ggml_set_name(graph.q, "mask_to_attention_q");
    ggml_set_name(graph.k, "mask_to_attention_k");
    ggml_set_name(graph.v, "mask_to_attention_v");

    if (route == Route::Dense) {
        // Keep the preparation node in F32. The shared attention builder adds
        // any HTP alignment padding in F32 and performs the final F16 cast.
        graph.mask = ggml_ext_zeros(graph.ctx, tokens, tokens, 1, 1);
        ggml_set_name(graph.mask, "dense_mask_prepare");
        graph.output = ggml_ext_attention_ext(
            graph.ctx, backend, graph.q, graph.k, graph.v, spec.heads, graph.mask,
            true, true, 1.0f, false);
    } else {
        const int protected_start = spec.text_tokens > 0 ? size.image_tokens : 0;
        graph.output = ggml_ext_attention_ext(
            graph.ctx, backend, graph.q, graph.k, graph.v, spec.heads, nullptr,
            true, true, 1.0f, true, protected_start, spec.text_tokens,
            SD_HTP_BLOCK_SPARSE_RESIDUAL_OFF, 1.0f, 1.0f, false, nullptr, nullptr,
            nullptr, nullptr, nullptr, size.block_rows * 4, size.block_cols * 8);
    }
    ggml_set_name(graph.output, "mask_to_attention_output");
    graph.graph = ggml_new_graph_custom(graph.ctx, 512, false);
    ggml_build_forward_expand(graph.graph, graph.output);
    return graph;
}

static bool upload_graph(Graph & graph, ggml_backend_t backend) {
    graph.buffer = ggml_backend_alloc_ctx_tensors_from_buft(
        graph.ctx, ggml_backend_get_default_buffer_type(backend));
    if (graph.buffer == nullptr) return false;
    const int64_t elements = ggml_nelements(graph.q);
    std::vector<float> q(static_cast<size_t>(elements));
    std::vector<ggml_fp16_t> k(static_cast<size_t>(elements));
    std::vector<ggml_fp16_t> v(static_cast<size_t>(elements));
    for (int64_t i = 0; i < elements; ++i) {
        q[static_cast<size_t>(i)] = float((i * 17) % 101 - 50) * 0.001f;
        k[static_cast<size_t>(i)] = ggml_fp32_to_fp16(float((i * 13) % 97 - 48) * 0.001f);
        v[static_cast<size_t>(i)] = ggml_fp32_to_fp16(float((i * 7) % 89 - 44) * 0.001f);
    }
    ggml_backend_tensor_set(graph.q, q.data(), 0, q.size() * sizeof(float));
    ggml_backend_tensor_set(graph.k, k.data(), 0, k.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(graph.v, v.data(), 0, v.size() * sizeof(ggml_fp16_t));
    const float one = 1.0f;
    ggml_tensor * built_in_one = ggml_get_tensor(graph.ctx, "ggml_runner_build_in_tensor:one");
    if (built_in_one != nullptr) ggml_backend_tensor_set(built_in_one, &one, 0, sizeof(one));
    return true;
}

static void free_graph(Graph & graph) {
    if (graph.buffer != nullptr) ggml_backend_buffer_free(graph.buffer);
    if (graph.ctx != nullptr) ggml_free(graph.ctx);
    graph = {};
}

static bool run_case(ggml_backend_t backend, const ImageSize & size,
                     const CaseSpec & spec, Route route, int warmups, int samples,
                     double top_p) {
    double initial_temperature = 0.0;
    if (!wait_for_cooldown(&initial_temperature)) return false;
    Graph graph = build_graph(backend, size, spec, route, top_p);
    if (!upload_graph(graph, backend)) {
        free_graph(graph);
        return false;
    }
    for (int i = 0; i < warmups; ++i) {
        ggml_backend_htp_reset_attention_profile(backend);
        if (ggml_backend_graph_compute(backend, graph.graph) != GGML_STATUS_SUCCESS) {
            free_graph(graph);
            return false;
        }
    }
    std::vector<int64_t> total_samples;
    std::vector<int64_t> attention_samples;
    total_samples.reserve(static_cast<size_t>(samples));
    attention_samples.reserve(static_cast<size_t>(samples));
    for (int i = 0; i < samples; ++i) {
        ggml_backend_htp_reset_attention_profile(backend);
        const auto begin = std::chrono::steady_clock::now();
        if (ggml_backend_graph_compute(backend, graph.graph) != GGML_STATUS_SUCCESS) {
            free_graph(graph);
            return false;
        }
        const auto end = std::chrono::steady_clock::now();
        const int64_t total_us = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
        const int64_t attention_us = ggml_backend_htp_get_last_attention_elapsed_us(backend);
        if (attention_us < 0 || total_us < 0) {
            free_graph(graph);
            return false;
        }
        total_samples.push_back(total_us);
        attention_samples.push_back(attention_us);
    }
    const TimingStats total = summarize(std::move(total_samples));
    const TimingStats attention = summarize(std::move(attention_samples));
    std::printf("[PERF] model=%s family=%s resolution=%d route=%s top_p=%.3f temp_c=%.2f samples=%d "
                "mask_to_attention_total_mean_us=%.1f mask_to_attention_total_p95_us=%.1f "
                "htp_attention_mean_us=%.1f pre_attention_overhead_mean_us=%.1f\n",
                spec.model, spec.family, size.resolution, route_name(route), top_p, initial_temperature, samples,
                total.mean_us, total.p95_us, attention.mean_us,
                total.mean_us - attention.mean_us);
    std::fflush(stdout);
    free_graph(graph);
    return true;
}

} // namespace

int main() {
    setenv("SD_HTP_OPS", "1", 1);
    setenv("SD_HTP_COMPUTE", "1", 1);
    setenv("SD_HTP_ATTENTION", "1", 1);
    setenv("SD_HTP_HALO_KV_TRACE", "0", 1);
    setenv("SD_HTP_HALO_REQUIRE_USERDMA_TRACE", "0", 1);
    setenv("SD_HTP_HALO_REQUIRE_L2_TRACE", "0", 1);
    ggml_backend_htp_set_hybrid_enabled(true);
    ggml_backend_t backend = ggml_backend_init_by_name("HTP", nullptr);
    if (backend == nullptr || !ggml_backend_htp_runtime_available()) {
        std::fprintf(stderr, "[FAIL] HTP runtime unavailable: %s\n",
                     ggml_backend_htp_runtime_error() != nullptr ? ggml_backend_htp_runtime_error() : "unknown");
        if (backend != nullptr) ggml_backend_free(backend);
        return 2;
    }
    ggml_backend_htp_set_n_threads(backend, 8);
    const int warmups = env_int("HTP_MASK_ATTN_WARMUPS", 3, 0);
    const int samples = env_int("HTP_MASK_ATTN_SAMPLES", 10, 1);
    const double top_p = env_double("HTP_MASK_ATTN_TOP_P", 0.90);
    const Route routes[] = {Route::Dense, Route::Halo, Route::TopP};
    int expected = 0;
    int measured = 0;
    for (const ImageSize & size : kSizes) {
        for (const CaseSpec & spec : kCases) {
            if (!selected(size, spec)) continue;
            for (const Route route : routes) {
                if (!selected_route(route)) continue;
                ++expected;
                measured += run_case(backend, size, spec, route, warmups, samples, top_p) ? 1 : 0;
            }
        }
    }
    std::printf("TOTAL: %d/%d mask-to-attention routes measured\n", measured, expected);
    ggml_backend_free(backend);
    ggml_backend_htp_set_hybrid_enabled(false);
    return measured == expected ? 0 : 1;
}
