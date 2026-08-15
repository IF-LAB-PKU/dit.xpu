// HTP attention-kernel-only benchmark.
// The reported samples come from the DSP attention op elapsed_us field and do
// not include QKV projection, graph scheduling, or host-side graph time.

#include <ggml-backend.h>
#include <ggml-htp.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if defined(__ANDROID__)
#include <unistd.h>
#endif

#include "test-backend-htp-attention-common.hpp"

namespace {

struct TimingStats {
    double min_us = 0.0;
    double median_us = 0.0;
    double mean_us = 0.0;
    double p95_us = 0.0;
    double max_us = 0.0;
};

static int env_int(const char * name, int fallback, int minimum) {
    const char * value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') return fallback;
    char * end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    return end != value && end != nullptr && *end == '\0' && parsed >= minimum && parsed <= INT32_MAX
        ? static_cast<int>(parsed) : fallback;
}

static double env_double(const char * name, double fallback) {
    const char * value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') return fallback;
    char * end = nullptr;
    const double parsed = std::strtod(value, &end);
    return end != value && end != nullptr && *end == '\0' && std::isfinite(parsed) ? parsed : fallback;
}

static bool parse_temperature(const std::string & dump, const char * sensor, double * value) {
    const std::string name = std::string("mName=") + sensor;
    for (size_t line_begin = 0; line_begin < dump.size();) {
        const size_t found_line_end = dump.find('\n', line_begin);
        const size_t line_end = found_line_end == std::string::npos ? dump.size() : found_line_end;
        const size_t sensor_pos = dump.find(name, line_begin);
        if (sensor_pos < line_end) {
            const size_t name_end = sensor_pos + name.size();
            const bool exact_name = name_end == line_end || dump[name_end] == ',' || dump[name_end] == '}';
            const size_t value_pos = dump.find("mValue=", line_begin);
            if (exact_name && value_pos < line_end) {
                char * end = nullptr;
                const char * value_begin = dump.c_str() + value_pos + 7;
                const double parsed = std::strtod(value_begin, &end);
                if (end != value_begin && end <= dump.c_str() + line_end && std::isfinite(parsed)) {
                    *value = parsed;
                    return true;
                }
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
    if (std::getenv("HTP_ATTN_PERF_SKIP_COOLDOWN") != nullptr) {
        *initial_temperature = -1.0;
        return true;
    }
#if !defined(__ANDROID__)
    std::fprintf(stderr, "cooldown requires Android; set HTP_ATTN_PERF_SKIP_COOLDOWN=1 for host-only runs\n");
    return false;
#else
    const double limit_c = env_double("HTP_ATTN_PERF_COOLDOWN_C", 40.0);
    const int poll_ms = env_int("HTP_ATTN_PERF_COOLDOWN_POLL_MS", 5000, 100);
    const int timeout_s = env_int("HTP_ATTN_PERF_COOLDOWN_TIMEOUT_S", 900, 1);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeout_s);
    while (std::chrono::steady_clock::now() < deadline) {
        double temperature = 0.0;
        if (!device_has_sd_cli() && read_device_temperature(&temperature) && temperature < limit_c) {
            double confirmed_temperature = 0.0;
            if (!device_has_sd_cli() && read_device_temperature(&confirmed_temperature) &&
                confirmed_temperature < limit_c) {
                *initial_temperature = confirmed_temperature;
                return true;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(poll_ms));
    }
    std::fprintf(stderr, "cooldown timeout: device did not become idle below %.2f C\n", limit_c);
    return false;
#endif
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

static const char * route_name(htp_attention_common::Route route, const char * loader) {
    if (route == htp_attention_common::Route::Dense) return "dense";
    if (route == htp_attention_common::Route::ExternalSparseBaseline) {
        return "sparse-baseline-unpacked";
    }
    if (route == htp_attention_common::Route::ExternalSparseChunk) {
        return "sparse-chunk-unpacked";
    }
    if (route == htp_attention_common::Route::PackedSparse && std::strcmp(loader, "chunked") == 0) {
        return "sparse-chunk-packed";
    }
    if (route == htp_attention_common::Route::PackedSparse) return "sparse-l2-packed";
    return "sparse-l2-unpacked";
}

static bool selected_route(const char * route) {
    const char * selected = std::getenv("HTP_ATTN_KERNEL_ROUTE");
    return selected == nullptr || selected[0] == '\0' ||
           std::strcmp(selected, route) == 0;
}

static bool run_route(ggml_backend_t backend,
                      const htp_attention_common::ImageSize & size,
                      const htp_attention_common::CaseSpec & base,
                      htp_attention_common::Route route,
                      const char * loader,
                      int warmups,
                      int samples) {
    double initial_temperature = 0.0;
    if (!wait_for_cooldown(&initial_temperature)) return false;

    htp_attention_common::CaseSpec spec = base;
    spec.image_tokens = size.image_tokens;
    spec.image_block_rows = size.block_rows;
    spec.image_block_cols = size.block_cols;
    spec.resolution = size.resolution;

    setenv("SD_HTP_BLOCK_SPARSE_ATTN", route == htp_attention_common::Route::Dense ? "0" : "1", 1);
    setenv("SD_HTP_PACKED_ATTENTION", route == htp_attention_common::Route::PackedSparse ? "1" : "0", 1);
    setenv("SD_HTP_HALO_KV_LOAD", loader, 1);
    setenv("SD_HTP_HALO_KV_TRACE", "0", 1);
    setenv("SD_HTP_HALO_REQUIRE_USERDMA_TRACE", "0", 1);
    setenv("SD_HTP_HALO_REQUIRE_L2_TRACE", "0", 1);

    htp_attention_common::Graph graph = htp_attention_common::build_graph(spec, route);
    const htp_attention_common::InputData data = htp_attention_common::make_input(spec);
    if (!htp_attention_common::graph_contract(graph, spec) ||
        !htp_attention_common::upload_graph(graph, data, backend)) {
        htp_attention_common::free_graph(graph);
        return false;
    }

    for (int i = 0; i < warmups; ++i) {
        ggml_backend_htp_reset_attention_profile(backend);
        if (ggml_backend_graph_compute(backend, graph.graph) != GGML_STATUS_SUCCESS ||
            ggml_backend_htp_get_last_attention_elapsed_us(backend) < 0) {
            htp_attention_common::free_graph(graph);
            return false;
        }
    }

    std::vector<int64_t> attention_elapsed;
    attention_elapsed.reserve(static_cast<size_t>(samples));
    for (int i = 0; i < samples; ++i) {
        ggml_backend_htp_reset_attention_profile(backend);
        if (ggml_backend_graph_compute(backend, graph.graph) != GGML_STATUS_SUCCESS) {
            htp_attention_common::free_graph(graph);
            return false;
        }
        const int64_t elapsed = ggml_backend_htp_get_last_attention_elapsed_us(backend);
        if (elapsed < 0) {
            htp_attention_common::free_graph(graph);
            return false;
        }
        attention_elapsed.push_back(elapsed);
    }

    const TimingStats stats = summarize(std::move(attention_elapsed));
    std::printf("[PERF] model=%s family=%s resolution=%d image_tokens=%d route=%s loader=%s temp_c=%.2f "
                "samples=%d htp_attention_min_us=%.1f htp_attention_median_us=%.1f "
                "htp_attention_mean_us=%.1f htp_attention_p95_us=%.1f htp_attention_max_us=%.1f\n",
                spec.model, spec.family, spec.resolution, spec.image_tokens,
                route_name(route, loader), loader, initial_temperature, samples,
                stats.min_us, stats.median_us, stats.mean_us, stats.p95_us, stats.max_us);
    std::fflush(stdout);
    htp_attention_common::free_graph(graph);
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

    const int warmups = env_int("HTP_ATTN_KERNEL_WARMUPS", 3, 0);
    const int samples = env_int("HTP_ATTN_KERNEL_SAMPLES", 10, 1);
    int measured = 0;
    int expected = 0;
    for (int si = 0; si < htp_attention_common::image_size_count(); ++si) {
        const htp_attention_common::ImageSize & size = htp_attention_common::image_sizes()[si];
        for (int ci = 0; ci < htp_attention_common::case_count(); ++ci) {
            const htp_attention_common::CaseSpec & spec = htp_attention_common::cases()[ci];
            if (!htp_attention_common::selected(size, spec)) continue;
            if (!spec.sparse) continue;

            const auto measure = [&](const char * name,
                                     htp_attention_common::Route route,
                                     const char * loader) {
                if (!selected_route(name)) return;
                ++expected;
                measured += run_route(
                    backend, size, spec, route, loader, warmups, samples) ? 1 : 0;
            };
            measure("dense", htp_attention_common::Route::Dense, "baseline");
            measure("sparse-baseline-unpacked",
                    htp_attention_common::Route::ExternalSparseBaseline, "baseline");
            measure("sparse-chunk-unpacked",
                    htp_attention_common::Route::ExternalSparseChunk, "chunked");
            measure("sparse-chunk-packed",
                    htp_attention_common::Route::PackedSparse, "chunked");
            measure("sparse-l2-packed",
                    htp_attention_common::Route::PackedSparse, "l2fetch");
            measure("sparse-l2-unpacked",
                    htp_attention_common::Route::ExternalSparseL2, "l2fetch");
        }
    }
    ggml_backend_free(backend);
    ggml_backend_htp_set_hybrid_enabled(false);
    std::printf("TOTAL: %d/%d attention-kernel routes measured\n", measured, expected);
    return measured == expected ? 0 : 1;
}
