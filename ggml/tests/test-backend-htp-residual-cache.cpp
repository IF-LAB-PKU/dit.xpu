#include "ggml-backend.h"
#include "ggml-htp.h"
#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

namespace {

constexpr int k_skip = 77;

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
    std::fprintf(stderr, "unknown HTP_RESIDUAL_CACHE_TEST mode: %s\n", mode);
    return 2;
}
