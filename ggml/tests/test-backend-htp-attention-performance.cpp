// Full-graph HTP attention benchmark. Correctness and performance share the
// same X -> QKV projection/preprocess -> attention fixture.

#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-htp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "test-backend-htp-attention-common.hpp"

extern "C" int htp_ops_resolve_halo_kv_load_mode(void);
extern "C" int htp_ops_resolve_halo_kv_load_mode_for_node(
    const struct ggml_tensor * node, bool * used_node_override);

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

static bool test_temperature_parser() {
    const std::string dump =
        "Cached temperatures:\n"
        "\tTemperature{mValue=36.0, mType=3, mName=shell_skin, mStatus=0}\n"
        "Current temperatures from HAL:\n"
        "\tTemperature{mValue=15.0, mType=8, mName=socd, mStatus=0}\n"
        "\tTemperature{mValue=34.847, mType=3, mName=skin, mStatus=0}\n";
    double value = -1.0;
    const bool shell_ok = parse_temperature(dump, "shell_skin", &value) && std::fabs(value - 36.0) <= 1.0e-6;
    const bool skin_ok = parse_temperature(dump, "skin", &value) && std::fabs(value - 34.847) <= 1.0e-6;
    if (!shell_ok || !skin_ok) {
        std::fprintf(stderr, "[FAIL] thermalservice same-record temperature parser\n");
        return false;
    }
    std::printf("[PASS] thermalservice same-record temperature parser\n");
    return true;
}

static bool test_chunked_loader_contract() {
    const char * old_value = std::getenv("SD_HTP_HALO_KV_LOAD");
    const std::string old_value_copy = old_value != nullptr ? old_value : "";
    setenv("SD_HTP_HALO_KV_LOAD", "chunked", 1);
    const int resolved = htp_ops_resolve_halo_kv_load_mode();
    if (old_value != nullptr) {
        setenv("SD_HTP_HALO_KV_LOAD", old_value_copy.c_str(), 1);
    } else {
        unsetenv("SD_HTP_HALO_KV_LOAD");
    }
    if (resolved != 3) {
        std::fprintf(stderr, "[FAIL] chunked loader resolved to %d, expected 3\n", resolved);
        return false;
    }
    std::printf("[PASS] chunked loader contract\n");
    return true;
}

static bool test_node_loader_override_contract() {
    const ggml_init_params params = { 1024 * 1024, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        std::fprintf(stderr, "[FAIL] node loader resolver context init\n");
        return false;
    }
    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 16, 4, 2, 1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 16, 4, 2, 1);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 16, 2, 4, 1);
    ggml_tensor * node = ggml_flash_attn_ext_with_lse(ctx, q, k, v, nullptr, 1.0f, 0.0f, 0.0f);
    if (node == nullptr) {
        ggml_free(ctx);
        std::fprintf(stderr, "[FAIL] node loader resolver node init\n");
        return false;
    }

    const char * old_value = std::getenv("SD_HTP_HALO_KV_LOAD");
    const std::string old_value_copy = old_value != nullptr ? old_value : "";
    const struct Case {
        const char * global_value;
        ggml_flash_attn_ext_halo_kv_load_mode node_value;
        int expected;
        bool expected_override;
    } cases[] = {
        {"l2fetch", GGML_FLASH_ATTN_EXT_HALO_KV_LOAD_CHUNKED, 3, true},
        {"userdma", GGML_FLASH_ATTN_EXT_HALO_KV_LOAD_CHUNKED, 3, true},
        {"baseline", GGML_FLASH_ATTN_EXT_HALO_KV_LOAD_CHUNKED, 3, true},
        {"l2fetch", GGML_FLASH_ATTN_EXT_HALO_KV_LOAD_INHERIT, 1, false},
        {"userdma", GGML_FLASH_ATTN_EXT_HALO_KV_LOAD_INHERIT, 2, false},
        {"baseline", GGML_FLASH_ATTN_EXT_HALO_KV_LOAD_INHERIT, 0, false},
    };
    bool ok = true;
    for (const auto & test_case : cases) {
        setenv("SD_HTP_HALO_KV_LOAD", test_case.global_value, 1);
        ggml_flash_attn_ext_set_halo_kv_load_mode(node, test_case.node_value);
        bool used_override = false;
        const int resolved = htp_ops_resolve_halo_kv_load_mode_for_node(node, &used_override);
        if (resolved != test_case.expected || used_override != test_case.expected_override) {
            std::fprintf(stderr, "[FAIL] node loader global=%s node=%d resolved=%d override=%d\n",
                         test_case.global_value, static_cast<int>(test_case.node_value),
                         resolved, used_override ? 1 : 0);
            ok = false;
        }
    }
    if (old_value != nullptr) {
        setenv("SD_HTP_HALO_KV_LOAD", old_value_copy.c_str(), 1);
    } else {
        unsetenv("SD_HTP_HALO_KV_LOAD");
    }
    ggml_free(ctx);
    if (ok) std::printf("[PASS] node loader override contract\n");
    return ok;
}

static TimingStats summarize_timings(std::vector<int64_t> samples) {
    GGML_ASSERT(!samples.empty());
    std::sort(samples.begin(), samples.end());
    TimingStats stats;
    stats.min_us = static_cast<double>(samples.front());
    stats.median_us = static_cast<double>(samples[samples.size() / 2]);
    stats.max_us = static_cast<double>(samples.back());
    double total = 0.0;
    for (const int64_t sample : samples) total += static_cast<double>(sample);
    stats.mean_us = total / static_cast<double>(samples.size());
    stats.p95_us = static_cast<double>(samples[(samples.size() * 95 + 99) / 100 - 1]);
    return stats;
}

static bool benchmark_case(ggml_backend_t htp,
                           const htp_attention_common::ImageSize & size,
                           const htp_attention_common::CaseSpec & base,
                           htp_attention_common::Route route,
                           const char * loader,
                           int warmups,
                           int samples) {
    double initial_temperature = -1.0;
    if (!wait_for_cooldown(&initial_temperature)) return false;

    const htp_attention_common::CaseSpec spec =
        htp_attention_common::make_inference_case(size, base, route);
    setenv("SD_HTP_BLOCK_SPARSE_ATTN", route == htp_attention_common::Route::Dense ? "0" : "1", 1);
    setenv("SD_HTP_PACKED_ATTENTION", route == htp_attention_common::Route::PackedSparse ? "1" : "0", 1);
    setenv("SD_HTP_HALO_KV_LOAD", loader, 1);
    setenv("SD_HTP_HALO_REQUIRE_USERDMA_TRACE", "0", 1);
    setenv("SD_HTP_HALO_REQUIRE_L2_TRACE", "0", 1);
    if (spec.external_inference || spec.packed_inference) {
        setenv("SD_HTP_BLOCK_SPARSE_KEEP_RATIO", "0", 1);
        setenv("SD_HTP_BLOCK_SPARSE_PATCH_HALO", "1", 1);
        setenv("SD_HTP_PACKED_Z_IMAGE", spec.packed_inference ? "1" : "0", 1);
        setenv("SD_HTP_SPARSE_OPENCL", "0", 1);
    }

    htp_attention_common::Graph graph = htp_attention_common::build_graph(spec, route, htp);
    const htp_attention_common::InputData data = htp_attention_common::make_input(spec);
    const bool ready = htp_attention_common::graph_contract(graph, spec) &&
                       htp_attention_common::upload_graph(graph, data, htp);
    std::vector<int64_t> elapsed;
    if (ready) {
        for (int i = 0; i < warmups; ++i) {
            if (ggml_backend_graph_compute(htp, graph.graph) != GGML_STATUS_SUCCESS) {
                htp_attention_common::free_graph(graph);
                return false;
            }
        }
        elapsed.reserve(static_cast<size_t>(samples));
        for (int i = 0; i < samples; ++i) {
            const int64_t begin = ggml_time_us();
            if (ggml_backend_graph_compute(htp, graph.graph) != GGML_STATUS_SUCCESS) {
                htp_attention_common::free_graph(graph);
                return false;
            }
            elapsed.push_back(ggml_time_us() - begin);
        }
    }
    htp_attention_common::free_graph(graph);
    if (!ready || elapsed.empty()) return false;

    const TimingStats stats = summarize_timings(std::move(elapsed));
    const char * route_name = route == htp_attention_common::Route::Dense ? "dense" :
                              route == htp_attention_common::Route::PackedSparse ? "packed-sparse" :
                              "external-sparse";
    std::printf("[PERF] model=%s family=%s resolution=%d image_tokens=%d route=%s loader=%s "
                "temp_c=%.2f samples=%d full_graph_min_us=%.1f full_graph_median_us=%.1f "
                "full_graph_mean_us=%.1f full_graph_p95_us=%.1f full_graph_max_us=%.1f\n",
                spec.model, spec.family, spec.resolution, spec.image_tokens, route_name, loader,
                initial_temperature, samples, stats.min_us, stats.median_us, stats.mean_us,
                stats.p95_us, stats.max_us);
    std::fflush(stdout);
    return true;
}

static int benchmark_matrix(ggml_backend_t htp, int warmups, int samples) {
    int measured = 0;
    int expected = 0;
    const bool dense_only = std::getenv("HTP_ATTN_PERF_DENSE_ONLY") != nullptr;
    for (int si = 0; si < htp_attention_common::image_size_count(); ++si) {
        const htp_attention_common::ImageSize & size = htp_attention_common::image_sizes()[si];
        for (int ci = 0; ci < htp_attention_common::case_count(); ++ci) {
            const htp_attention_common::CaseSpec & spec = htp_attention_common::cases()[ci];
            if (!htp_attention_common::selected(size, spec)) continue;
            ++expected;
            measured += benchmark_case(htp, size, spec, htp_attention_common::Route::Dense,
                                       "none", warmups, samples) ? 1 : 0;
            if (spec.sparse && !dense_only) {
                ++expected;
                measured += benchmark_case(htp, size, spec, htp_attention_common::Route::ExternalSparseL2,
                                           "l2fetch", warmups, samples) ? 1 : 0;
                ++expected;
                measured += benchmark_case(htp, size, spec, htp_attention_common::Route::PackedSparse,
                                           "l2fetch", warmups, samples) ? 1 : 0;
            }
        }
    }
    std::printf("TOTAL: %d/%d shared full_graph routes measured\n", measured, expected);
    return measured == expected ? 0 : 1;
}

} // namespace

int main() {
    if (std::getenv("HTP_ATTN_SHARED_FIXTURE_CONTRACT") != nullptr) {
        return htp_attention_common::run_fixture_contract_test();
    }
    if (std::getenv("HTP_ATTN_CHUNKED_ROUTE_CONTRACT") != nullptr) {
        return test_chunked_loader_contract() ? 0 : 1;
    }
    if (std::getenv("HTP_ATTN_NODE_LOADER_CONTRACT") != nullptr) {
        return test_node_loader_override_contract() ? 0 : 1;
    }
    if (std::getenv("HTP_ATTN_PERF_TEST_TEMPERATURE_PARSER") != nullptr) {
        return test_temperature_parser() ? 0 : 1;
    }

    setenv("SD_HTP_OPS", "1", 1);
    setenv("SD_HTP_COMPUTE", "1", 1);
    setenv("SD_HTP_ATTENTION", "1", 1);
    ggml_backend_htp_set_hybrid_enabled(true);
    ggml_backend_t htp = ggml_backend_init_by_name("HTP", nullptr);
    if (htp == nullptr || !ggml_backend_htp_runtime_available()) {
        std::fprintf(stderr, "[FAIL] HTP runtime unavailable: %s\n",
                     ggml_backend_htp_runtime_error() != nullptr ? ggml_backend_htp_runtime_error() : "unknown");
        if (htp != nullptr) ggml_backend_free(htp);
        return 2;
    }
    ggml_backend_htp_set_n_threads(htp, 8);

    const int warmups = env_int("HTP_ATTN_PERF_WARMUPS", 3, 0);
    const int samples = env_int("HTP_ATTN_PERF_SAMPLES", 10, 1);
    const int status = benchmark_matrix(htp, warmups, samples);
    ggml_backend_free(htp);
    ggml_backend_htp_set_hybrid_enabled(false);
    return status;
}
