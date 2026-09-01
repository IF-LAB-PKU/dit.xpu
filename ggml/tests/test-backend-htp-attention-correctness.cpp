// CPU-reference vs HTP correctness for the shared full attention fixture.

#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-htp.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

static bool read_f32_rows(const ggml_tensor * tensor, std::vector<float> & output) {
    if (tensor == nullptr || tensor->type != GGML_TYPE_F32 ||
        size_t(ggml_nelements(tensor)) != output.size()) {
        return false;
    }
    const size_t row_bytes = size_t(tensor->ne[0]) * sizeof(float);
    for (int64_t z = 0; z < tensor->ne[2]; ++z) {
        for (int64_t y = 0; y < tensor->ne[1]; ++y) {
            const size_t src_offset = size_t(z) * tensor->nb[2] + size_t(y) * tensor->nb[1];
            const size_t dst_offset = size_t(z * tensor->ne[1] + y) * size_t(tensor->ne[0]);
            ggml_backend_tensor_get(tensor, output.data() + dst_offset, src_offset, row_bytes);
        }
    }
    return true;
}

static const char * route_name(htp_attention_common::Route route) {
    switch (route) {
        case htp_attention_common::Route::Dense: return "dense";
        case htp_attention_common::Route::PackedSparse: return "packed-sparse";
        case htp_attention_common::Route::FusedAnchor: return "fused-anchor";
        default: return "external-sparse";
    }
}

static bool run_fused_case(
    ggml_backend_t htp,
    const htp_attention_common::ImageSize & size,
    const htp_attention_common::CaseSpec & base) {
    using namespace htp_attention_common;
    const CaseSpec spec = make_inference_case(size, base, Route::FusedAnchor);
    const InputData data = make_input(spec);
    Reference reference;
    for (const BranchData & branch : data.branches) {
        append_reference_branch(spec, branch, false, reference);
    }
    const AttentionReference dense_reference =
        reference_attention_with_lse(spec, reference, nullptr);
    const AttentionReference sparse_reference =
        reference_attention_with_lse(spec, reference, &data.block_index);
    const int rows = total_tokens(spec) * spec.heads;
    std::vector<float> rho_reference(rows);
    std::vector<float> residual_reference(size_t(rows) * spec.head_dim);
    for (int row = 0; row < rows; ++row) {
        const float rho = std::exp(
            sparse_reference.lse[row] - dense_reference.lse[row]);
        rho_reference[row] = rho;
        for (int d = 0; d < spec.head_dim; ++d) {
            const size_t index = size_t(row) * spec.head_dim + d;
            residual_reference[index] =
                dense_reference.output[index] - rho * sparse_reference.output[index];
        }
    }

    Graph graph = build_graph(spec, Route::FusedAnchor, htp);
    if (std::getenv("HTP_ATTN_DIRECT_OPERATOR") != nullptr) {
        setenv("SD_HTP_SPARSE_OPENCL_RESIDUAL", "0", 1);
        setenv("SD_HTP_SPARSE_OPENCL_STRICT", "0", 1);
    }
    const bool contract_ready = graph_contract(graph, spec);
    const bool upload_ready = contract_ready && upload_graph(graph, data, htp);
    const bool ready = upload_ready &&
                       graph.dense_output != nullptr && graph.dense_lse != nullptr &&
                       graph.sparse_output != nullptr && graph.sparse_lse != nullptr &&
                       graph.rho != nullptr && graph.residual != nullptr;
    ggml_status status = GGML_STATUS_FAILED;
    std::vector<float> dense_output(dense_reference.output.size());
    std::vector<float> dense_lse(dense_reference.lse.size());
    std::vector<float> sparse_output(sparse_reference.output.size());
    std::vector<float> sparse_lse(sparse_reference.lse.size());
    std::vector<float> rho(rho_reference.size());
    std::vector<ggml_fp16_t> residual_f16(residual_reference.size());
    std::vector<float> residual(residual_reference.size());
    if (ready) {
        status = ggml_backend_graph_compute(htp, graph.graph);
        if (status == GGML_STATUS_SUCCESS) {
            read_f32_rows(graph.dense_output, dense_output);
            read_f32_rows(graph.dense_lse, dense_lse);
            read_f32_rows(graph.sparse_output, sparse_output);
            read_f32_rows(graph.sparse_lse, sparse_lse);
            ggml_backend_tensor_get(graph.rho, rho.data(), 0,
                                    rho.size() * sizeof(float));
            ggml_backend_tensor_get(graph.residual, residual_f16.data(), 0,
                                    residual_f16.size() * sizeof(ggml_fp16_t));
            for (size_t i = 0; i < residual.size(); ++i) {
                residual[i] = ggml_fp16_to_fp32(residual_f16[i]);
            }
        }
    }
    const Metrics dense_o = compare(dense_reference.output, dense_output);
    const Metrics sparse_o = compare(sparse_reference.output, sparse_output);
    const Metrics dense_lse_metrics = compare(dense_reference.lse, dense_lse);
    const Metrics sparse_lse_metrics = compare(sparse_reference.lse, sparse_lse);
    const Metrics rho_metrics = compare(rho_reference, rho);
    const Metrics residual_metrics = compare(residual_reference, residual);
    double text_dense_sparse_max_abs = 0.0;
    bool text_dense_sparse_finite = true;
    if (spec.text_tokens > 0) {
        for (int token = spec.image_tokens; token < total_tokens(spec); ++token) {
            for (int head = 0; head < spec.heads; ++head) {
                const size_t row = size_t(token * spec.heads + head);
                const size_t base = row * size_t(spec.head_dim);
                for (int d = 0; d < spec.head_dim; ++d) {
                    const float dense = dense_output[base + d];
                    const float sparse = sparse_output[base + d];
                    if (!std::isfinite(dense) || !std::isfinite(sparse)) {
                        text_dense_sparse_finite = false;
                    } else {
                        text_dense_sparse_max_abs = std::max(
                            text_dense_sparse_max_abs, std::fabs(double(dense) - double(sparse)));
                    }
                }
                const float dense_lse_value = dense_lse[row];
                const float sparse_lse_value = sparse_lse[row];
                if (!std::isfinite(dense_lse_value) || !std::isfinite(sparse_lse_value)) {
                    text_dense_sparse_finite = false;
                } else {
                    text_dense_sparse_max_abs = std::max(
                        text_dense_sparse_max_abs,
                        std::fabs(double(dense_lse_value) - double(sparse_lse_value)));
                }
            }
        }
    }
    const bool pass = ready && status == GGML_STATUS_SUCCESS &&
                      dense_o.finite && sparse_o.finite &&
                      dense_lse_metrics.finite && sparse_lse_metrics.finite &&
                      rho_metrics.finite && residual_metrics.finite &&
                      dense_o.max_abs <= 5e-3 && dense_o.rmse <= 5e-3 &&
                      sparse_o.max_abs <= 5e-3 && sparse_o.rmse <= 5e-3 &&
                      dense_lse_metrics.max_abs <= 5e-3 &&
                      sparse_lse_metrics.max_abs <= 5e-3 &&
                      rho_metrics.max_abs <= 2e-3 &&
                      residual_metrics.max_abs <= 5e-3 &&
                      residual_metrics.rmse <= 1e-3 &&
                      text_dense_sparse_finite && text_dense_sparse_max_abs <= 1e-6;
    std::printf(
        "[%s] %s/%s/resolution=%d route=fused-anchor d=%d "
        "dense_o=(%.9g,%.9g) sparse_o=(%.9g,%.9g) "
        "dense_lse=%.9g sparse_lse=%.9g rho=%.9g residual=(%.9g,%.9g) "
        "text_dense_sparse_max_abs=%.9g status=%d ready_contract=%d ready_upload=%d\n",
        pass ? "PASS" : "FAIL", spec.model, spec.family, spec.resolution,
        spec.head_dim, dense_o.max_abs, dense_o.rmse,
        sparse_o.max_abs, sparse_o.rmse, dense_lse_metrics.max_abs,
        sparse_lse_metrics.max_abs, rho_metrics.max_abs,
        residual_metrics.max_abs, residual_metrics.rmse, text_dense_sparse_max_abs, int(status),
        contract_ready ? 1 : 0, upload_ready ? 1 : 0);
    if (!pass) {
        std::printf(
            "  dense_o_worst=%zu ref=%.9g got=%.9g "
            "sparse_o_worst=%zu ref=%.9g got=%.9g\n",
            dense_o.worst,
            dense_o.worst < dense_reference.output.size() ? dense_reference.output[dense_o.worst] : 0.0f,
            dense_o.worst < dense_output.size() ? dense_output[dense_o.worst] : 0.0f,
            sparse_o.worst,
            sparse_o.worst < sparse_reference.output.size() ? sparse_reference.output[sparse_o.worst] : 0.0f,
            sparse_o.worst < sparse_output.size() ? sparse_output[sparse_o.worst] : 0.0f);
        std::printf(
            "  residual_worst=%zu ref=%.9g got=%.9g\n",
            residual_metrics.worst,
            residual_metrics.worst < residual_reference.size() ? residual_reference[residual_metrics.worst] : 0.0f,
            residual_metrics.worst < residual.size() ? residual[residual_metrics.worst] : 0.0f);
    }
    free_graph(graph);
    return pass;
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
            if (std::strncmp(spec.model, "z-image", 7) == 0) {
                if (std::strcmp(spec.family, "context-refiner") == 0) {
                    if (route_enabled("dense")) {
                        ++expected;
                        passed += run_case(htp, size, spec, htp_attention_common::Route::Dense) ? 1 : 0;
                    }
                } else if (route_enabled("z-image-pair") || route_filter == nullptr) {
                    ++expected;
                    passed += htp_attention_common::run_z_image_route_pair(htp, size, spec) ? 1 : 0;
                }
                continue;
            }
            if (route_enabled("dense")) {
                ++expected;
                passed += run_case(htp, size, spec, htp_attention_common::Route::Dense) ? 1 : 0;
            }
            if (spec.sparse && !dense_only) {
                if (route_enabled("fused-anchor")) {
                    ++expected;
                    passed += run_fused_case(htp, size, spec) ? 1 : 0;
                }
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
    const int status = std::getenv("HTP_ATTN_TEST_PLACEMENT_ONLY") != nullptr
        ? htp_attention_common::run_hybrid_placement_contract_test(htp)
        : run_matrix(htp);
    ggml_backend_free(htp);
    ggml_backend_htp_set_hybrid_enabled(false);
    return status;
}
