// CPU-reference vs HTP correctness for the shared full attention fixture.

#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-htp.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "test-backend-htp-attention-common.hpp"

namespace {

struct Metrics {
    double max_abs = 0.0;
    double rmse = 0.0;
    size_t worst = 0;
    bool finite = true;
};

static Metrics compare(const std::vector<float> & expected,
                       const std::vector<float> & actual) {
    Metrics metrics;
    if (expected.size() != actual.size() || expected.empty()) {
        metrics.finite = false;
        return metrics;
    }
    double sum_sq = 0.0;
    for (size_t i = 0; i < expected.size(); ++i) {
        if (!std::isfinite(expected[i]) || !std::isfinite(actual[i])) {
            metrics.finite = false;
            continue;
        }
        const double diff = std::fabs(double(expected[i]) - double(actual[i]));
        if (diff > metrics.max_abs) {
            metrics.max_abs = diff;
            metrics.worst = i;
        }
        sum_sq += diff * diff;
    }
    metrics.rmse = std::sqrt(sum_sq / expected.size());
    return metrics;
}

static const char * route_name(htp_attention_common::Route route) {
    switch (route) {
        case htp_attention_common::Route::Dense: return "dense";
        case htp_attention_common::Route::PackedSparse: return "packed-sparse";
        default: return "external-sparse";
    }
}

static bool run_case(ggml_backend_t htp,
                     const htp_attention_common::ImageSize & size,
                     const htp_attention_common::CaseSpec & base,
                     htp_attention_common::Route route) {
    const htp_attention_common::CaseSpec spec =
        htp_attention_common::make_inference_case(size, base, route);

    const bool block_sparse = route != htp_attention_common::Route::Dense;
    const bool q_is_f16 = route == htp_attention_common::Route::PackedSparse;
    const htp_attention_common::InputData data = htp_attention_common::make_input(spec);
    htp_attention_common::Reference reference;
    for (const htp_attention_common::BranchData & branch : data.branches) {
        htp_attention_common::append_reference_branch(spec, branch, q_is_f16, reference);
    }
    const std::vector<float> expected = htp_attention_common::reference_attention(
        spec, reference, block_sparse ? &data.block_index : nullptr);

    setenv("SD_HTP_BLOCK_SPARSE_ATTN", block_sparse ? "1" : "0", 1);
    setenv("SD_HTP_PACKED_ATTENTION", q_is_f16 ? "1" : "0", 1);
    setenv("SD_HTP_HALO_KV_LOAD", block_sparse ? "l2fetch" : "none", 1);
    if (spec.external_inference || spec.packed_inference) {
        setenv("SD_HTP_BLOCK_SPARSE_KEEP_RATIO", "0", 1);
        setenv("SD_HTP_BLOCK_SPARSE_PATCH_HALO", "1", 1);
        setenv("SD_HTP_PACKED_Z_IMAGE", spec.packed_inference ? "1" : "0", 1);
        setenv("SD_HTP_SPARSE_OPENCL", "0", 1);
    }
    htp_attention_common::Graph graph = htp_attention_common::build_graph(spec, route, htp);
    const bool ready = htp_attention_common::graph_contract(graph, spec) &&
                       htp_attention_common::upload_graph(graph, data, htp) &&
                       size_t(ggml_nelements(graph.output)) == expected.size();

    ggml_status status = GGML_STATUS_FAILED;
    std::vector<float> actual(expected.size(), 0.0f);
    if (ready) {
        status = ggml_backend_graph_compute(htp, graph.graph);
        if (status == GGML_STATUS_SUCCESS) {
            ggml_backend_tensor_get(graph.output, actual.data(), 0,
                                    actual.size() * sizeof(float));
        }
    }
    const Metrics metrics = compare(expected, actual);
    const bool pass = ready && status == GGML_STATUS_SUCCESS && metrics.finite &&
                      metrics.max_abs <= 5.0e-3 && metrics.rmse <= 5.0e-3;
    std::printf("[%s] %s/%s/resolution=%d route=%s full_graph_correctness=1 "
                "max_abs=%.9g rmse=%.9g",
                pass ? "PASS" : "FAIL", spec.model, spec.family, spec.resolution,
                route_name(route), metrics.max_abs, metrics.rmse);
    if (!pass && metrics.worst < expected.size() && metrics.worst < actual.size()) {
        std::printf(" worst=%zu ref=%g got=%g status=%d",
                    metrics.worst, expected[metrics.worst], actual[metrics.worst], int(status));
    }
    std::printf("\n");
    htp_attention_common::free_graph(graph);
    return pass;
}

static int run_matrix(ggml_backend_t htp) {
    int passed = 0;
    int expected = 0;
    const bool dense_only = std::getenv("HTP_ATTN_TEST_DENSE_ONLY") != nullptr;
    const char * route_filter = std::getenv("HTP_ATTN_TEST_ROUTE");
    const auto route_enabled = [route_filter](const char * name) {
        return route_filter == nullptr || route_filter[0] == '\0' ||
               std::strcmp(route_filter, name) == 0;
    };
    for (int si = 0; si < htp_attention_common::image_size_count(); ++si) {
        const htp_attention_common::ImageSize & size = htp_attention_common::image_sizes()[si];
        for (int ci = 0; ci < htp_attention_common::case_count(); ++ci) {
            const htp_attention_common::CaseSpec & spec = htp_attention_common::cases()[ci];
            if (!htp_attention_common::selected(size, spec)) continue;
            if (route_enabled("dense")) {
                ++expected;
                passed += run_case(htp, size, spec, htp_attention_common::Route::Dense) ? 1 : 0;
            }
            if (spec.sparse && !dense_only) {
                if (route_enabled("external-sparse")) {
                    ++expected;
                    passed += run_case(htp, size, spec,
                                       htp_attention_common::Route::ExternalSparseL2) ? 1 : 0;
                }
                if (route_enabled("packed-sparse")) {
                    ++expected;
                    passed += run_case(htp, size, spec,
                                       htp_attention_common::Route::PackedSparse) ? 1 : 0;
                }
            }
        }
    }
    std::printf("TOTAL: %d/%d shared fixture routes passed\n", passed, expected);
    return passed == expected ? 0 : 1;
}

} // namespace

int main() {
    if (std::getenv("HTP_ATTN_SHARED_FIXTURE_CONTRACT") != nullptr) {
        return htp_attention_common::run_fixture_contract_test();
    }
    setenv("SD_HTP_OPS", "1", 1);
    setenv("SD_HTP_COMPUTE", "1", 1);
    setenv("SD_HTP_ATTENTION", "1", 1);
    setenv("SD_HTP_HALO_REQUIRE_USERDMA_TRACE", "0", 1);
    setenv("SD_HTP_HALO_REQUIRE_L2_TRACE", "0", 1);
    ggml_backend_htp_set_hybrid_enabled(true);
    ggml_backend_t htp = ggml_backend_init_by_name("HTP", nullptr);
    if (htp == nullptr || !ggml_backend_htp_runtime_available()) {
        std::fprintf(stderr, "[FAIL] HTP runtime unavailable: %s\n",
                     ggml_backend_htp_runtime_error() != nullptr ? ggml_backend_htp_runtime_error() : "unknown");
        if (htp != nullptr) ggml_backend_free(htp);
        return 2;
    }
    ggml_backend_htp_set_n_threads(htp, 8);
    const int status = run_matrix(htp);
    ggml_backend_free(htp);
    ggml_backend_htp_set_hybrid_enabled(false);
    return status;
}
