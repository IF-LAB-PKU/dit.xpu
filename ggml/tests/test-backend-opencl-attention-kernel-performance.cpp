// OpenCL attention-kernel-only benchmark.
// The reported samples cover attention submission through completion, matching
// the HTP attention worker-submit-to-worker-completion timing window.

#include <ggml-backend.h>
#include <ggml-htp.h>
#include <ggml-opencl.h>

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

enum class Route {
    Dense,
    Sparse,
};

enum class VLayout {
    TokenHead,
    HeadToken,
};

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
    return end != value && end != nullptr && *end == '\0' &&
                   parsed >= minimum && parsed <= INT_MAX
        ? static_cast<int>(parsed)
        : fallback;
}

static double env_double(const char * name, double fallback) {
    const char * value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') return fallback;
    char * end = nullptr;
    const double parsed = std::strtod(value, &end);
    return end != value && end != nullptr && *end == '\0' && std::isfinite(parsed)
        ? parsed
        : fallback;
}

static bool env_enabled(const char * name) {
    const char * value = std::getenv(name);
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

static bool parse_temperature(const std::string & dump, const char * sensor, double * value) {
    const std::string name = std::string("mName=") + sensor;
    for (size_t line_begin = 0; line_begin < dump.size();) {
        const size_t found_line_end = dump.find('\n', line_begin);
        const size_t line_end =
            found_line_end == std::string::npos ? dump.size() : found_line_end;
        const size_t sensor_pos = dump.find(name, line_begin);
        if (sensor_pos < line_end) {
            const size_t name_end = sensor_pos + name.size();
            const bool exact_name =
                name_end == line_end || dump[name_end] == ',' || dump[name_end] == '}';
            const size_t value_pos = dump.find("mValue=", line_begin);
            if (exact_name && value_pos < line_end) {
                char * end = nullptr;
                const char * value_begin = dump.c_str() + value_pos + 7;
                const double parsed = std::strtod(value_begin, &end);
                if (end != value_begin && end <= dump.c_str() + line_end &&
                    std::isfinite(parsed)) {
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
    return parse_temperature(dump, "shell_skin", value) ||
           parse_temperature(dump, "skin", value);
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
    const bool running =
        std::fgets(buffer, sizeof(buffer), pipe) != nullptr && buffer[0] != '\0';
    pclose(pipe);
    return running;
#else
    return false;
#endif
}

static bool wait_for_cooldown(double * initial_temperature) {
    if (std::getenv("OPENCL_ATTN_PERF_SKIP_COOLDOWN") != nullptr) {
        *initial_temperature = -1.0;
        return true;
    }
#if !defined(__ANDROID__)
    std::fprintf(
        stderr,
        "cooldown requires Android; set OPENCL_ATTN_PERF_SKIP_COOLDOWN=1 for host runs\n");
    return false;
#else
    const double limit_c = env_double("OPENCL_ATTN_PERF_COOLDOWN_C", 40.0);
    const int poll_ms = env_int("OPENCL_ATTN_PERF_COOLDOWN_POLL_MS", 5000, 100);
    const int timeout_s = env_int("OPENCL_ATTN_PERF_COOLDOWN_TIMEOUT_S", 900, 1);
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(timeout_s);
    while (std::chrono::steady_clock::now() < deadline) {
        double temperature = 0.0;
        if (!device_has_sd_cli() && read_device_temperature(&temperature) &&
            temperature < limit_c) {
            double confirmed_temperature = 0.0;
            if (!device_has_sd_cli() &&
                read_device_temperature(&confirmed_temperature) &&
                confirmed_temperature < limit_c) {
                *initial_temperature = confirmed_temperature;
                return true;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(poll_ms));
    }
    std::fprintf(
        stderr, "cooldown timeout: device did not become idle below %.2f C\n", limit_c);
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

static const char * route_name(Route route) {
    return route == Route::Dense ? "opencl-dense" : "opencl-sparse";
}

static const char * v_layout_name(VLayout layout) {
    return layout == VLayout::TokenHead ? "D,L,H,B" : "D,H,L,B";
}

static bool selected_v_layout(VLayout layout) {
    const char * selected = std::getenv("OPENCL_ATTN_TEST_V_LAYOUT");
    if (selected == nullptr || selected[0] == '\0') return true;
    return std::strcmp(
        selected, layout == VLayout::TokenHead ? "token-head" : "head-token") == 0;
}

static bool selected_route(Route route) {
    const char * selected = std::getenv("OPENCL_ATTN_TEST_ROUTE");
    return selected == nullptr || selected[0] == '\0' ||
           std::strcmp(selected, route == Route::Dense ? "dense" : "sparse") == 0;
}

static void configure_sparse_route() {
    setenv("SD_HTP_OPS", "1", 1);
    setenv("SD_HTP_COMPUTE", "1", 1);
    setenv("SD_HTP_ATTENTION", "1", 1);
    setenv("SD_HTP_BLOCK_SPARSE_ATTN", "1", 1);
    setenv("SD_HTP_PACKED_ATTENTION", "0", 1);
    // The HTP route state machine requires the residual flag to select the
    // OpenCL anchor sidecar route, even when this benchmark has no residual
    // tensor in its attention graph.
    setenv("SD_HTP_BLOCK_SPARSE_RESIDUAL_CACHE", "1", 1);
    setenv("SD_HTP_SPARSE_OPENCL", "1", 1);
    setenv("SD_HTP_SPARSE_OPENCL_STRICT", "1", 1);
    setenv("SD_HTP_SPARSE_OPENCL_ASYNC", "0", 1);
    setenv("SD_HTP_SPARSE_OPENCL_EARLY_SUBMIT", "0", 1);
    setenv("SD_HTP_SPARSE_OPENCL_SUBMIT_THREAD", "0", 1);
    setenv("SD_HTP_SPARSE_OPENCL_RESIDUAL", "1", 1);
    setenv("SD_HTP_HALO_KV_LOAD", "l2fetch", 1);
}

static AttentionGraph build_attention_graph(
    const htp_attention_common::CaseSpec & spec,
    Route route,
    VLayout v_layout) {
    AttentionGraph result;
    const ggml_init_params params = {64 * 1024 * 1024, nullptr, true};
    result.ctx = ggml_init(params);
    if (result.ctx == nullptr) return result;

    const int tokens = htp_attention_common::total_tokens(spec);
    const int blocks = (tokens + htp_attention_common::kBlock - 1) /
                       htp_attention_common::kBlock;
    result.q_storage = ggml_new_tensor_4d(
        result.ctx, GGML_TYPE_F32, spec.head_dim, spec.heads, tokens, 1);
    result.k_storage = ggml_new_tensor_4d(
        result.ctx, GGML_TYPE_F16, spec.head_dim, spec.heads, tokens, 1);
    result.v_storage = ggml_new_tensor_4d(
        result.ctx, GGML_TYPE_F16, spec.head_dim, spec.heads, tokens, 1);
    ggml_tensor * q = ggml_permute(result.ctx, result.q_storage, 0, 2, 1, 3);
    ggml_tensor * k = ggml_permute(result.ctx, result.k_storage, 0, 2, 1, 3);
    ggml_tensor * v = v_layout == VLayout::TokenHead
                          ? ggml_permute(result.ctx, result.v_storage, 0, 2, 1, 3)
                          : result.v_storage;

    if (route == Route::Dense) {
        const int padded_tokens = ((tokens + 63) / 64) * 64;
        result.mask =
            ggml_new_tensor_2d(result.ctx, GGML_TYPE_F16, padded_tokens, tokens);
        result.output = ggml_flash_attn_ext(
            result.ctx,
            q,
            k,
            v,
            result.mask,
            1.0f / std::sqrt(static_cast<float>(spec.head_dim)),
            0.0f,
            0.0f);
        ggml_set_name(result.output, "opencl_dense_attention");
    } else {
        result.mask = ggml_new_tensor_4d(
            result.ctx, GGML_TYPE_I32, blocks, spec.heads * blocks, 1, 1);
        result.output = ggml_flash_attn_ext_with_lse(
            result.ctx,
            q,
            k,
            v,
            result.mask,
            1.0f / std::sqrt(static_cast<float>(spec.head_dim)),
            0.0f,
            0.0f);
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
        ggml_set_name(result.output, "opencl_sparse_attention");
    }
    ggml_flash_attn_ext_set_prec(result.output, GGML_PREC_F32);
    result.graph = ggml_new_graph_custom(result.ctx, 16, false);
    ggml_build_forward_expand(result.graph, result.output);
    return result;
}

static bool upload_attention_graph(
    AttentionGraph & graph,
    const htp_attention_common::CaseSpec & spec,
    Route route,
    ggml_backend_t backend) {
    const ggml_backend_buffer_type_t buft =
        route == Route::Dense
            ? ggml_backend_get_default_buffer_type(backend)
            : ggml_backend_htp_qkv_buffer_type();
    graph.buffer = ggml_backend_alloc_ctx_tensors_from_buft(graph.ctx, buft);
    if (graph.buffer == nullptr) {
        std::fprintf(stderr, "[FAIL] attention graph buffer allocation failed route=%s\n",
                     route_name(route));
        return false;
    }

    const htp_attention_common::InputData data =
        htp_attention_common::make_input(spec);
    htp_attention_common::Reference reference;
    for (const htp_attention_common::BranchData & branch : data.branches) {
        htp_attention_common::append_reference_branch(
            spec, branch, route == Route::Sparse, reference);
    }
    ggml_backend_tensor_set(
        graph.q_storage,
        reference.q.data(),
        0,
        reference.q.size() * sizeof(float));
    ggml_backend_tensor_set(
        graph.k_storage,
        reference.k.data(),
        0,
        reference.k.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(
        graph.v_storage,
        reference.v.data(),
        0,
        reference.v.size() * sizeof(ggml_fp16_t));

    if (route == Route::Sparse) {
        ggml_backend_tensor_set(
            graph.mask,
            data.block_index.data(),
            0,
            data.block_index.size() * sizeof(int32_t));
    } else {
        const int tokens = graph.mask->ne[1];
        const int padded_tokens = graph.mask->ne[0];
        std::vector<ggml_fp16_t> mask(
            ggml_nelements(graph.mask), ggml_fp32_to_fp16(-INFINITY));
        for (int query = 0; query < tokens; ++query) {
            std::fill_n(
                mask.data() + static_cast<size_t>(query) * padded_tokens,
                tokens,
                ggml_fp32_to_fp16(0.0f));
        }
        ggml_backend_tensor_set(
            graph.mask, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
    }
    const int node_count = ggml_graph_n_nodes(graph.graph);
    int attention_nodes = 0;
    for (int i = 0; i < node_count; ++i) {
        const ggml_tensor * node = ggml_graph_node(graph.graph, i);
        if (node != nullptr && node->op == GGML_OP_FLASH_ATTN_EXT) {
            ++attention_nodes;
        }
    }
    if (attention_nodes != 1 || graph.output->op != GGML_OP_FLASH_ATTN_EXT) {
        std::fprintf(
            stderr,
            "[FAIL] attention graph contract route=%s nodes=%d attention_nodes=%d output_op=%s\n",
            route_name(route),
            node_count,
            attention_nodes,
            ggml_op_name(graph.output->op));
        return false;
    }
    return true;
}

static void free_attention_graph(AttentionGraph & graph) {
    if (graph.buffer != nullptr) ggml_backend_buffer_free(graph.buffer);
    if (graph.ctx != nullptr) ggml_free(graph.ctx);
    graph = {};
}

static bool run_route(
    ggml_backend_t dense_backend,
    ggml_backend_t sparse_backend,
    const htp_attention_common::ImageSize & size,
    const htp_attention_common::CaseSpec & base,
    Route route,
    VLayout v_layout,
    int warmups,
    int samples) {
    double initial_temperature = 0.0;
    if (!wait_for_cooldown(&initial_temperature)) return false;

    htp_attention_common::CaseSpec spec = base;
    spec.image_tokens = size.image_tokens;
    spec.image_block_rows = size.block_rows;
    spec.image_block_cols = size.block_cols;
    spec.resolution = size.resolution;

    ggml_backend_t backend = dense_backend;
    if (route == Route::Sparse) {
        configure_sparse_route();
        backend = sparse_backend;
    } else {
        setenv("SD_HTP_BLOCK_SPARSE_ATTN", "0", 1);
        setenv("SD_HTP_SPARSE_OPENCL", "0", 1);
    }

    AttentionGraph graph = build_attention_graph(spec, route, v_layout);
    if (graph.ctx == nullptr || graph.graph == nullptr || graph.output == nullptr ||
        !upload_attention_graph(graph, spec, route, backend)) {
        std::fprintf(
            stderr,
            "[FAIL] graph setup model=%s family=%s resolution=%d route=%s v_layout=%s\n",
            spec.model,
            spec.family,
            spec.resolution,
            route_name(route),
            v_layout_name(v_layout));
        free_attention_graph(graph);
        return false;
    }

    ggml_backend_opencl_set_attention_profiling_enabled(true);
    for (int i = 0; i < warmups; ++i) {
        ggml_backend_opencl_reset_attention_profile();
        if (ggml_backend_graph_compute(backend, graph.graph) != GGML_STATUS_SUCCESS ||
            ggml_backend_opencl_get_last_attention_elapsed_us() < 0) {
            std::fprintf(
                stderr,
                "[FAIL] warmup model=%s family=%s resolution=%d route=%s\n",
                spec.model,
                spec.family,
                spec.resolution,
                route_name(route));
            ggml_backend_opencl_set_attention_profiling_enabled(false);
            free_attention_graph(graph);
            return false;
        }
    }

    std::vector<int64_t> attention_elapsed;
    attention_elapsed.reserve(static_cast<size_t>(samples));
    const bool diagnostic_profile =
        route == Route::Sparse && env_enabled("SD_HTP_SPARSE_OPENCL_PROFILE");
    for (int i = 0; i < samples; ++i) {
        ggml_backend_opencl_reset_attention_profile();
        if (ggml_backend_graph_compute(backend, graph.graph) != GGML_STATUS_SUCCESS) {
            std::fprintf(
                stderr,
                "[FAIL] compute model=%s family=%s resolution=%d route=%s sample=%d\n",
                spec.model,
                spec.family,
                spec.resolution,
                route_name(route),
                i);
            ggml_backend_opencl_set_attention_profiling_enabled(false);
            free_attention_graph(graph);
            return false;
        }
        const int64_t elapsed = ggml_backend_opencl_get_last_attention_elapsed_us();
        if (elapsed < 0) {
            std::fprintf(
                stderr,
                "[FAIL] missing timing model=%s family=%s resolution=%d route=%s sample=%d\n",
                spec.model,
                spec.family,
                spec.resolution,
                route_name(route),
                i);
            ggml_backend_opencl_set_attention_profiling_enabled(false);
            free_attention_graph(graph);
            return false;
        }
        attention_elapsed.push_back(elapsed);
        if (diagnostic_profile) {
            ggml_opencl_attention_profile profile = {};
            if (!ggml_backend_opencl_get_last_attention_profile(&profile) ||
                profile.event_queued_ns > profile.event_submit_ns ||
                profile.event_submit_ns > profile.event_start_ns ||
                profile.event_start_ns > profile.event_end_ns) {
                std::fprintf(
                    stderr,
                    "[FAIL] invalid OpenCL event profile model=%s family=%s "
                    "resolution=%d sample=%d\n",
                    spec.model, spec.family, spec.resolution, i);
                ggml_backend_opencl_set_attention_profiling_enabled(false);
                free_attention_graph(graph);
                return false;
            }
            const double queued_submit_us =
                static_cast<double>(profile.event_submit_ns - profile.event_queued_ns) / 1000.0;
            const double submit_start_us =
                static_cast<double>(profile.event_start_ns - profile.event_submit_ns) / 1000.0;
            const double kernel_us =
                static_cast<double>(profile.event_end_ns - profile.event_start_ns) / 1000.0;
            const double queued_end_us =
                static_cast<double>(profile.event_end_ns - profile.event_queued_ns) / 1000.0;
            if (kernel_us > queued_end_us) {
                std::fprintf(
                    stderr,
                    "[FAIL] inconsistent OpenCL event duration model=%s family=%s "
                    "resolution=%d sample=%d\n",
                    spec.model, spec.family, spec.resolution, i);
                ggml_backend_opencl_set_attention_profiling_enabled(false);
                free_attention_graph(graph);
                return false;
            }
            std::printf(
                "[OCL_ATTN_SAMPLE] model=%s family=%s resolution=%d sample=%d "
                "host_total_us=%lld validation_us=%lld binding_us=%lld args_us=%lld "
                "enqueue_us=%lld flush_us=%lld wait_us=%lld "
                "event_queued_submit_us=%.3f event_submit_start_us=%.3f "
                "event_kernel_us=%.3f event_queued_end_us=%.3f\n",
                spec.model, spec.family, spec.resolution, i,
                static_cast<long long>(profile.host_total_us),
                static_cast<long long>(profile.validation_us),
                static_cast<long long>(profile.binding_us),
                static_cast<long long>(profile.args_us),
                static_cast<long long>(profile.enqueue_us),
                static_cast<long long>(profile.flush_us),
                static_cast<long long>(profile.wait_us),
                queued_submit_us, submit_start_us, kernel_us, queued_end_us);
            std::fflush(stdout);
        }
    }
    ggml_backend_opencl_set_attention_profiling_enabled(false);

    if (route == Route::Sparse) {
        const htp_attention_common::InputData data =
            htp_attention_common::make_input(spec);
        htp_attention_common::Reference reference;
        for (const htp_attention_common::BranchData & branch : data.branches) {
            htp_attention_common::append_reference_branch(spec, branch, true, reference);
        }
        const htp_attention_common::AttentionReference expected =
            htp_attention_common::reference_attention_with_lse(
                spec, reference, &data.block_index);
        std::vector<float> packed(ggml_nelements(graph.output));
        ggml_backend_tensor_get(
            graph.output, packed.data(), 0, packed.size() * sizeof(float));
        double output_mse = 0.0;
        double lse_mse = 0.0;
        float output_max_abs = 0.0f;
        float lse_max_abs = 0.0f;
        const int rows = htp_attention_common::total_tokens(spec) * spec.heads;
        for (int row = 0; row < rows; ++row) {
            for (int d = 0; d < spec.head_dim; ++d) {
                const float error =
                    packed[size_t(row) * (spec.head_dim + 1) + d] -
                    expected.output[size_t(row) * spec.head_dim + d];
                output_max_abs = std::max(output_max_abs, std::fabs(error));
                output_mse += double(error) * error;
            }
            const float lse_error =
                packed[size_t(row) * (spec.head_dim + 1) + spec.head_dim] -
                expected.lse[row];
            lse_max_abs = std::max(lse_max_abs, std::fabs(lse_error));
            lse_mse += double(lse_error) * lse_error;
        }
        const float output_rmse = std::sqrt(
            float(output_mse / expected.output.size()));
        const float lse_rmse = std::sqrt(float(lse_mse / expected.lse.size()));
        if (output_max_abs > 5e-3f || output_rmse > 5e-3f ||
            lse_max_abs > 5e-3f || lse_rmse > 5e-3f) {
            std::fprintf(
                stderr,
                "[FAIL] sparse correctness model=%s resolution=%d v_layout=%s "
                "output=(%.9g,%.9g) lse=(%.9g,%.9g)\n",
                spec.model, spec.resolution, v_layout_name(v_layout),
                output_max_abs, output_rmse, lse_max_abs, lse_rmse);
            free_attention_graph(graph);
            return false;
        }
    }

    const TimingStats stats = summarize(std::move(attention_elapsed));
    std::printf(
        "[PERF] model=%s family=%s resolution=%d image_tokens=%d route=%s v_layout=%s "
        "temp_c=%.2f samples=%d opencl_attention_min_us=%.1f "
        "opencl_attention_median_us=%.1f opencl_attention_mean_us=%.1f "
        "opencl_attention_p95_us=%.1f opencl_attention_max_us=%.1f\n",
        spec.model,
        spec.family,
        spec.resolution,
        spec.image_tokens,
        route_name(route),
        v_layout_name(v_layout),
        initial_temperature,
        samples,
        stats.min_us,
        stats.median_us,
        stats.mean_us,
        stats.p95_us,
        stats.max_us);
    std::fflush(stdout);
    free_attention_graph(graph);
    return true;
}

}  // namespace

int main() {
    const int warmups = env_int("OPENCL_ATTN_KERNEL_WARMUPS", 3, 0);
    const int samples = env_int("OPENCL_ATTN_KERNEL_SAMPLES", 10, 1);

    ggml_backend_t dense_backend = ggml_backend_opencl_init();
    if (dense_backend == nullptr) {
        std::fprintf(stderr, "[FAIL] OpenCL backend unavailable\n");
        return 2;
    }

    configure_sparse_route();
    ggml_backend_htp_set_hybrid_enabled(true);
    ggml_backend_t sparse_backend = ggml_backend_init_by_name("HTP", nullptr);
    if (sparse_backend == nullptr || !ggml_backend_htp_runtime_available()) {
        std::fprintf(
            stderr,
            "[FAIL] HTP runtime unavailable for OpenCL sparse sidecar: %s\n",
            ggml_backend_htp_runtime_error() != nullptr
                ? ggml_backend_htp_runtime_error()
                : "unknown");
        if (sparse_backend != nullptr) ggml_backend_free(sparse_backend);
        ggml_backend_free(dense_backend);
        ggml_backend_htp_set_hybrid_enabled(false);
        return 2;
    }
    ggml_backend_htp_set_n_threads(sparse_backend, 8);

    int measured = 0;
    int expected = 0;
    for (int si = 0; si < htp_attention_common::image_size_count(); ++si) {
        const htp_attention_common::ImageSize & size =
            htp_attention_common::image_sizes()[si];
        for (int ci = 0; ci < htp_attention_common::case_count(); ++ci) {
            const htp_attention_common::CaseSpec & spec =
                htp_attention_common::cases()[ci];
            if (!htp_attention_common::selected(size, spec) || !spec.sparse) continue;

            for (const Route route : {Route::Dense, Route::Sparse}) {
                if (!selected_route(route)) continue;
                const VLayout layouts[] = {
                    VLayout::TokenHead,
                    VLayout::HeadToken,
                };
                const int layout_count = route == Route::Sparse ? 2 : 1;
                for (int layout_index = 0; layout_index < layout_count; ++layout_index) {
                    if (!selected_v_layout(layouts[layout_index])) continue;
                    ++expected;
                    measured += run_route(
                        dense_backend,
                        sparse_backend,
                        size,
                        spec,
                        route,
                        layouts[layout_index],
                        warmups,
                        samples)
                        ? 1
                        : 0;
                }
            }
        }
    }

    ggml_backend_opencl_set_attention_profiling_enabled(false);
    ggml_backend_free(sparse_backend);
    ggml_backend_free(dense_backend);
    ggml_backend_htp_set_hybrid_enabled(false);
    std::printf("TOTAL: %d/%d OpenCL attention-kernel routes measured\n", measured, expected);
    return measured == expected ? 0 : 1;
}
