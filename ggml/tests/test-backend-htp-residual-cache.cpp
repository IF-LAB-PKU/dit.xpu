#include "ggml-backend.h"
#include "ggml-htp.h"
#include "ggml.h"
#include "ggml_extend.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

namespace {

constexpr int k_skip = 77;

struct graph_order_result {
    size_t buffer_size = 0;
    int cache_count = 0;
};

bool graph_contains(ggml_cgraph * graph, const ggml_tensor * tensor) {
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        if (ggml_graph_node(graph, i) == tensor) {
            return true;
        }
    }
    return false;
}

bool cache_expands_persisted_tensor() {
    ggml_init_params params = {
        /*.mem_size   =*/ 4 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        return false;
    }

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 32, false);
    ggml_tensor * source = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 16, 16);
    ggml_tensor * view = ggml_transpose(ctx, source);
    ggml_tensor * persisted = nullptr;
    const std::function<ggml_tensor *(const std::string &, ggml_tensor *)>
        cache_tensor = [&](const std::string &, ggml_tensor * tensor) {
            persisted = ggml_cont(ctx, tensor);
            ggml_set_name(persisted, "graph_order_persisted_cache");
            return persisted;
        };

    sd_cache_tensor_and_expand_graph(graph, cache_tensor, "cache", view);
    const bool expanded = persisted != nullptr && graph_contains(graph, persisted);
    ggml_free(ctx);
    return expanded;
}

bool opencl_residual_route_requires_all_prerequisites() {
    unsetenv("SD_HTP_SPARSE_OPENCL");
    unsetenv("SD_HTP_SPARSE_OPENCL_ASYNC");
    unsetenv("SD_HTP_SPARSE_OPENCL_EARLY_SUBMIT");
    unsetenv("SD_HTP_BLOCK_SPARSE_RESIDUAL_CACHE");
    unsetenv("SD_HTP_BLOCK_SPARSE_RESIDUAL_EXACT_RHO");
    if (sd_htp_opencl_residual_route_prerequisites()) {
        return false;
    }

    setenv("SD_HTP_SPARSE_OPENCL", "1", 1);
    setenv("SD_HTP_SPARSE_OPENCL_ASYNC", "1", 1);
    setenv("SD_HTP_SPARSE_OPENCL_EARLY_SUBMIT", "1", 1);
    setenv("SD_HTP_BLOCK_SPARSE_RESIDUAL_CACHE", "1", 1);
    if (sd_htp_opencl_residual_route_prerequisites()) {
        return false;
    }

    setenv("SD_HTP_BLOCK_SPARSE_RESIDUAL_EXACT_RHO", "1", 1);
    return sd_htp_opencl_residual_route_prerequisites();
}

bool cache_storage_updates_avoid_rebuilding_disjoint_branches() {
    const std::vector<std::unordered_set<std::string>> existing = {
        {"residual_branch_0", "rho_branch_0"},
    };

    const auto append = sd_plan_cache_storage_update(
        existing,
        {"residual_branch_1", "rho_branch_1"},
        false);
    if (append.mode != SD_CACHE_STORAGE_APPEND ||
        !append.replace_segment_indices.empty()) {
        return false;
    }

    const auto replace = sd_plan_cache_storage_update(
        existing,
        {"residual_branch_0", "rho_branch_0"},
        false);
    if (replace.mode != SD_CACHE_STORAGE_REPLACE_SEGMENTS ||
        replace.replace_segment_indices != std::vector<size_t>{0}) {
        return false;
    }

    const auto partial = sd_plan_cache_storage_update(
        existing,
        {"residual_branch_0"},
        false);
    if (partial.mode != SD_CACHE_STORAGE_MERGE) {
        return false;
    }

    const auto graph_cut = sd_plan_cache_storage_update(
        existing,
        {"residual_branch_1", "rho_branch_1"},
        true);
    return graph_cut.mode == SD_CACHE_STORAGE_MERGE;
}

graph_order_result build_graph_order_fixture(bool expand_cache_immediately) {
    constexpr int k_layers = 8;
    constexpr int64_t k_elements = 256 * 1024;

    ggml_init_params params = {
        /*.mem_size   =*/ 16 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        return {};
    }

    ggml_backend_t backend = ggml_backend_cpu_init();
    if (backend == nullptr) {
        ggml_free(ctx);
        return {};
    }
    ggml_gallocr_t allocator =
        ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 4 * k_layers + 16, false);

    graph_order_result result;
    std::vector<ggml_tensor *> cache_outputs;
    ggml_tensor * state = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, k_elements);
    ggml_set_name(state, "graph_order_input");

    const auto cache_tensor = [&](const std::string &, ggml_tensor * tensor) {
        ++result.cache_count;
        cache_outputs.push_back(tensor);
        return tensor;
    };

    for (int layer = 0; layer < k_layers; ++layer) {
        ggml_tensor * branch = ggml_sqr(ctx, state);
        ggml_tensor * cache = ggml_sum(ctx, branch);
        ggml_format_name(cache, "graph_order_cache_%d", layer);
        if (expand_cache_immediately) {
            sd_cache_tensor_and_expand_graph(graph, cache_tensor, cache->name, cache);
        } else {
            cache_tensor(cache->name, cache);
        }
        state = ggml_neg(ctx, state);
        ggml_format_name(state, "graph_order_main_%d", layer);
    }

    ggml_build_forward_expand(graph, state);
    if (!expand_cache_immediately) {
        for (ggml_tensor * cache : cache_outputs) {
            ggml_set_output(cache);
            ggml_build_forward_expand(graph, cache);
        }
    }

    if (ggml_gallocr_reserve(allocator, graph)) {
        result.buffer_size = ggml_gallocr_get_buffer_size(allocator, 0);
    }

    ggml_gallocr_free(allocator);
    ggml_backend_free(backend);
    ggml_free(ctx);
    return result;
}

bool check(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
    }
    return condition;
}

int run_fp16_test() {
    const std::vector<float> dense = {
        0.0f, 1.0f, -1.0f, 0.0001220703125f,
        3.1415927f, -2.7182817f, 12.5f, -15.75f,
        0.33333334f, -0.66666669f, 102.0f, -96.0f,
    };
    const std::vector<float> sparse = {
        0.0f, 0.5f, -0.5f, 0.00006103515625f,
        2.75f, -3.0f, 10.0f, -12.0f,
        0.25f, -0.75f, 100.0f, -100.0f,
    };
    const std::vector<float> rho = {
        1.0f, 0.9999f, 1.0001f, 0.5f,
        0.125f, 0.875f, 1.25f, 0.75f,
        1.3333333f, 0.6666667f, 1.019f, 0.959f,
    };

    std::vector<float> reference(dense.size());
    std::vector<ggml_fp16_t> cached(dense.size());
    std::vector<float> recovered(dense.size());
    double squared_error = 0.0;
    float max_abs = 0.0f;

    for (size_t i = 0; i < dense.size(); ++i) {
        reference[i] = dense[i] - sparse[i] * rho[i];
        cached[i] = ggml_fp32_to_fp16(reference[i]);
        const float residual_f32 = ggml_fp16_to_fp32(cached[i]);
        recovered[i] = sparse[i] * rho[i] + residual_f32;
        if (!check(std::isfinite(residual_f32), "FP16 residual must be finite") ||
            !check(std::isfinite(recovered[i]), "recombined output must be finite")) {
            return 1;
        }
        const float error = std::fabs(residual_f32 - reference[i]);
        max_abs = std::max(max_abs, error);
        squared_error += static_cast<double>(error) * error;
    }

    const float rmse = static_cast<float>(
        std::sqrt(squared_error / static_cast<double>(reference.size())));
    std::printf(
        "fp16 residual_type=%s rho_type=%s max_abs=%.9g rmse=%.9g nan=0 inf=0\n",
        ggml_type_name(GGML_TYPE_F16), ggml_type_name(GGML_TYPE_F32), max_abs, rmse);
    if (!check(max_abs <= 5e-3f, "FP16 residual max_abs exceeds 5e-3") ||
        !check(rmse <= 1e-3f, "FP16 residual RMSE exceeds 1e-3")) {
        return 1;
    }
    std::puts("PASS fp16");
    return 0;
}

int run_graph_order_test() {
    const graph_order_result tail = build_graph_order_fixture(false);
    const graph_order_result immediate = build_graph_order_fixture(true);
    std::printf("graph_order tail_bytes=%zu immediate_bytes=%zu caches=%d\n",
                tail.buffer_size, immediate.buffer_size, immediate.cache_count);
    if (!check(tail.buffer_size > 0 && immediate.buffer_size > 0,
               "graph-order allocator fixture must reserve both graphs") ||
        !check(tail.cache_count == 8 && immediate.cache_count == 8,
               "graph-order fixture must register every cache output") ||
        !check(cache_expands_persisted_tensor(),
               "cache expansion must use the tensor returned by persistence") ||
        !check(opencl_residual_route_requires_all_prerequisites(),
               "OpenCL residual route must require every prerequisite") ||
        !check(cache_storage_updates_avoid_rebuilding_disjoint_branches(),
               "disjoint branch caches must append without rebuilding old cache storage") ||
        !check(sd_htp_release_compute_buffer_after_residual_anchor(
                   SD_HTP_BLOCK_SPARSE_RESIDUAL_ANCHOR, false),
               "CPU residual anchor must release its compute arena") ||
        !check(!sd_htp_release_compute_buffer_after_residual_anchor(
                    SD_HTP_BLOCK_SPARSE_RESIDUAL_ANCHOR, true),
               "OpenCL residual anchor must preserve deferred arena ownership") ||
        !check(!sd_htp_release_compute_buffer_after_residual_anchor(
                    SD_HTP_BLOCK_SPARSE_RESIDUAL_APPLY, false),
               "sparse apply must keep the existing compute-buffer policy") ||
        !check(immediate.buffer_size * 2 < tail.buffer_size,
               "per-layer cache expansion must cut peak allocation by more than half")) {
        return 1;
    }
    std::puts("PASS graph-order");
    return 0;
}

#if defined(GGML_USE_HTP)

struct cache_pair {
    ggml_backend_htp_async_cache_slot * residual = nullptr;
    ggml_backend_htp_async_cache_slot * rho = nullptr;
};

void free_cache_pair(cache_pair & pair) {
    ggml_backend_htp_async_cache_slot_free(pair.rho);
    ggml_backend_htp_async_cache_slot_free(pair.residual);
    pair = {};
}

cache_pair create_cache_pair(ggml_backend_t backend, uint64_t generation) {
    constexpr int64_t residual_ne[GGML_MAX_DIMS] = { 16, 2, 4, 1 };
    constexpr int64_t rho_ne[GGML_MAX_DIMS] = { 1, 2, 4, 1 };
    cache_pair pair;
    char reason[512] = {};
    pair.residual = ggml_backend_htp_async_cache_slot_create(
        backend, "htp_block_sparse_residual_b0_self_s0", generation,
        GGML_TYPE_F16, residual_ne, reason, sizeof(reason));
    if (pair.residual == nullptr) {
        std::fprintf(stderr, "residual slot creation failed: %s\n", reason);
        return pair;
    }
    pair.rho = ggml_backend_htp_async_cache_slot_create(
        backend, "htp_block_sparse_residual_rho_b0_self_s0", generation,
        GGML_TYPE_F32, rho_ne, reason, sizeof(reason));
    if (pair.rho == nullptr) {
        std::fprintf(stderr, "rho slot creation failed: %s\n", reason);
        free_cache_pair(pair);
    }
    return pair;
}

int run_lifecycle_test() {
    setenv("SD_HTP_SPARSE_OPENCL", "1", 1);
    setenv("SD_HTP_SPARSE_OPENCL_STRICT", "1", 1);
    if (!ggml_backend_htp_runtime_available()) {
        std::fprintf(stderr, "SKIP: HTP runtime unavailable: %s\n",
                     ggml_backend_htp_runtime_error());
        return k_skip;
    }

    ggml_backend_t backend = ggml_backend_htp_init();
    if (backend == nullptr) {
        std::fprintf(stderr, "SKIP: HTP backend initialization failed: %s\n",
                     ggml_backend_htp_runtime_error());
        return k_skip;
    }

    cache_pair generation_1 = create_cache_pair(backend, 1);
    if (generation_1.residual == nullptr || generation_1.rho == nullptr) {
        ggml_backend_free(backend);
        return 1;
    }
    const size_t generation_1_imports =
        ggml_backend_htp_opencl_live_cache_import_count(backend);
    std::printf("generation 1 imports=%zu\n", generation_1_imports);
    if (!check(generation_1_imports == 2, "generation 1 must own two imports")) {
        free_cache_pair(generation_1);
        ggml_backend_free(backend);
        return 1;
    }

    char reason[512] = {};
    constexpr int64_t duplicate_ne[GGML_MAX_DIMS] = { 16, 2, 4, 1 };
    ggml_backend_htp_async_cache_slot * duplicate =
        ggml_backend_htp_async_cache_slot_create(
            backend, "htp_block_sparse_residual_b0_self_s0", 2,
            GGML_TYPE_F16, duplicate_ne, reason, sizeof(reason));
    if (!check(duplicate == nullptr,
               "a cache key must reject a second live generation")) {
        ggml_backend_htp_async_cache_slot_free(duplicate);
        free_cache_pair(generation_1);
        ggml_backend_free(backend);
        return 1;
    }

    free_cache_pair(generation_1);
    if (!check(ggml_backend_htp_opencl_live_cache_import_count(backend) == 0,
               "old generation imports must be released before replacement")) {
        ggml_backend_free(backend);
        return 1;
    }

    cache_pair generation_2 = create_cache_pair(backend, 2);
    const size_t generation_2_imports =
        ggml_backend_htp_opencl_live_cache_import_count(backend);
    std::printf("generation 2 imports=%zu\n", generation_2_imports);
    if (generation_2.residual == nullptr || generation_2.rho == nullptr ||
        !check(generation_2_imports == 2,
               "generation 2 must replace generation 1 without import growth")) {
        free_cache_pair(generation_2);
        ggml_backend_free(backend);
        return 1;
    }

    free_cache_pair(generation_2);
    const size_t final_imports =
        ggml_backend_htp_opencl_live_cache_import_count(backend);
    const size_t final_events =
        ggml_backend_htp_opencl_live_cache_event_count(backend);
    std::printf("final imports=%zu events=%zu\n", final_imports, final_events);
    const bool clean =
        check(final_imports == 0, "all persistent imports must be released") &&
        check(final_events == 0, "all cache events must be released");
    ggml_backend_free(backend);
    if (!clean) {
        return 1;
    }
    std::puts("PASS lifecycle");
    return 0;
}

#else

int run_lifecycle_test() {
    std::puts("SKIP: test was built without HTP support");
    return k_skip;
}

#endif

}  // namespace

int main() {
    const char * mode = std::getenv("HTP_RESIDUAL_CACHE_TEST");
    if (mode == nullptr || mode[0] == '\0' || std::strcmp(mode, "fp16") == 0) {
        return run_fp16_test();
    }
    if (std::strcmp(mode, "lifecycle") == 0) {
        return run_lifecycle_test();
    }
    if (std::strcmp(mode, "graph-order") == 0) {
        return run_graph_order_test();
    }
    std::fprintf(stderr, "unknown HTP_RESIDUAL_CACHE_TEST mode: %s\n", mode);
    return 2;
}
