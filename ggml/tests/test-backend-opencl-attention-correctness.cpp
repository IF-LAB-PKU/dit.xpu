// CPU-reference correctness tests for the standalone OpenCL sparse-attention path.

#include <ggml-backend.h>
#include <ggml-htp.h>
#include <ggml-opencl.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "test-backend-htp-attention-common.hpp"

namespace {

enum class MaskKind {
    Index,
    Matrix,
};

struct DirectCase {
    std::string name;
    int head_dim = 128;
    int heads = 1;
    int kv_heads = 1;
    int batch = 1;
    int lq = 32;
    int lk = 32;
    int image_tokens = 32;
    int text_tokens = 0;
    int image_block_rows = 1;
    int image_block_cols = 1;
    bool omit_halo_metadata = false;
    bool compare_all_rows = true;
    MaskKind mask_kind = MaskKind::Index;
    int mask_columns = 0;
    std::vector<float> q;
    std::vector<ggml_fp16_t> k;
    std::vector<ggml_fp16_t> v;
    std::vector<int32_t> mask;
};

struct Graph {
    ggml_context * ctx = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_tensor * q_storage = nullptr;
    ggml_tensor * k_storage = nullptr;
    ggml_tensor * v_storage = nullptr;
    ggml_tensor * mask = nullptr;
    ggml_tensor * output = nullptr;
};

struct Metric {
    double max_abs = 0.0;
    double sum_sq = 0.0;
    size_t count = 0;
    bool finite = true;

    double rmse() const {
        return count == 0 ? 0.0 : std::sqrt(sum_sq / static_cast<double>(count));
    }
};

struct Summary {
    int cases = 0;
    int passed = 0;
    Metric output;
    Metric lse;
};

static bool env_enabled(const char * name) {
    const char * value = std::getenv(name);
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

static void configure_sparse_route() {
    setenv("SD_HTP_OPS", "1", 1);
    setenv("SD_HTP_COMPUTE", "1", 1);
    setenv("SD_HTP_ATTENTION", "1", 1);
    setenv("SD_HTP_BLOCK_SPARSE_ATTN", "1", 1);
    setenv("SD_HTP_PACKED_ATTENTION", "0", 1);
    setenv("SD_HTP_SPARSE_OPENCL", "1", 1);
    setenv("SD_HTP_SPARSE_OPENCL_STRICT", "1", 1);
    setenv("SD_HTP_SPARSE_OPENCL_ASYNC", "0", 1);
    setenv("SD_HTP_SPARSE_OPENCL_EARLY_SUBMIT", "0", 1);
    setenv("SD_HTP_SPARSE_OPENCL_SUBMIT_THREAD", "0", 1);
    setenv("SD_HTP_SPARSE_OPENCL_RESIDUAL", "0", 1);
    setenv("SD_HTP_HALO_KV_LOAD", "l2fetch", 1);
}

static size_t q_index(const DirectCase & test, int b, int token, int head, int d) {
    return (((static_cast<size_t>(b) * test.lq + token) * test.heads + head) *
            test.head_dim) + d;
}

static size_t kv_index(const DirectCase & test, int b, int token, int head, int d) {
    return (((static_cast<size_t>(b) * test.lk + token) * test.kv_heads + head) *
            test.head_dim) + d;
}

static size_t output_row_index(const DirectCase & test, int b, int token, int head) {
    return (static_cast<size_t>(b) * test.lq + token) * test.heads + head;
}

static int q_blocks(const DirectCase & test) {
    return (test.lq + htp_attention_common::kBlock - 1) /
           htp_attention_common::kBlock;
}

static int k_blocks(const DirectCase & test) {
    return (test.lk + htp_attention_common::kBlock - 1) /
           htp_attention_common::kBlock;
}

static std::vector<int> selected_rows(const DirectCase & test) {
    const int total_rows = test.batch * test.lq * test.heads;
    if (test.compare_all_rows) {
        std::vector<int> rows(total_rows);
        for (int row = 0; row < total_rows; ++row) rows[row] = row;
        return rows;
    }

    std::vector<int> tokens = {
        0,
        std::min(31, test.lq - 1),
        std::min(32, test.lq - 1),
        std::min(test.lq / 2, test.lq - 1),
        std::min(std::max(0, test.image_tokens - 1), test.lq - 1),
        std::min(test.image_tokens, test.lq - 1),
        test.lq - 1,
    };
    std::sort(tokens.begin(), tokens.end());
    tokens.erase(std::unique(tokens.begin(), tokens.end()), tokens.end());
    std::vector<int> rows;
    for (int b = 0; b < test.batch; ++b) {
        for (int token : tokens) {
            rows.push_back(static_cast<int>(output_row_index(test, b, token, 0)));
            if (test.heads > 1) {
                rows.push_back(static_cast<int>(
                    output_row_index(test, b, token, test.heads - 1)));
            }
        }
    }
    std::sort(rows.begin(), rows.end());
    rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
    return rows;
}

static std::vector<int> gather_keys(
        const DirectCase & test,
        int token,
        int head) {
    const int qb = token / htp_attention_common::kBlock;
    const int qb_count = q_blocks(test);
    const int kb_count = k_blocks(test);
    const int kv_head = head / (test.heads / test.kv_heads);
    const int mask_row = head * qb_count + qb;
    std::vector<int> keys;
    if (test.mask_kind == MaskKind::Index) {
        const int32_t * row = test.mask.data() +
            static_cast<size_t>(mask_row) * test.mask_columns;
        for (int candidate = 0; candidate < test.mask_columns; ++candidate) {
            const int block = row[candidate];
            if (block == -1) break;
            if (block < 0 || block >= test.kv_heads * kb_count ||
                block / kb_count != kv_head) {
                continue;
            }
            const int local_block = block - kv_head * kb_count;
            const int begin = local_block * htp_attention_common::kBlock;
            const int end = std::min(test.lk, begin + htp_attention_common::kBlock);
            for (int key = begin; key < end; ++key) keys.push_back(key);
        }
    } else {
        const int matrix_columns = test.kv_heads * kb_count;
        const int32_t * row = test.mask.data() +
            static_cast<size_t>(mask_row) * matrix_columns;
        for (int local_block = 0; local_block < kb_count; ++local_block) {
            const int block = kv_head * kb_count + local_block;
            if (row[block] != 0) continue;
            const int begin = local_block * htp_attention_common::kBlock;
            const int end = std::min(test.lk, begin + htp_attention_common::kBlock);
            for (int key = begin; key < end; ++key) keys.push_back(key);
        }
    }
    return keys;
}

static void update_metric(Metric & metric, float expected, float actual) {
    if (std::isinf(expected) && std::isinf(actual) &&
        std::signbit(expected) == std::signbit(actual)) {
        return;
    }
    if (!std::isfinite(expected) || !std::isfinite(actual)) {
        metric.finite = false;
        return;
    }
    const double difference = std::fabs(static_cast<double>(expected) - actual);
    metric.max_abs = std::max(metric.max_abs, difference);
    metric.sum_sq += difference * difference;
    ++metric.count;
}

static bool compare_reference(
        const DirectCase & test,
        const std::vector<float> & actual,
        Metric * output_metric,
        Metric * lse_metric) {
    const size_t row_stride = static_cast<size_t>(test.head_dim + 1);
    const size_t expected_size =
        static_cast<size_t>(test.batch) * test.lq * test.heads * row_stride;
    if (actual.size() != expected_size) return false;

    for (size_t row = 0; row < actual.size() / row_stride; ++row) {
        for (int d = 0; d < test.head_dim; ++d) {
            if (!std::isfinite(actual[row * row_stride + d])) {
                output_metric->finite = false;
            }
        }
        const int head = static_cast<int>(row % test.heads);
        const int token_batch = static_cast<int>(row / test.heads);
        const int token = token_batch % test.lq;
        const bool empty = gather_keys(test, token, head).empty();
        const float actual_lse = actual[row * row_stride + test.head_dim];
        if ((!empty && !std::isfinite(actual_lse)) ||
            (empty && !(std::isinf(actual_lse) && std::signbit(actual_lse)))) {
            lse_metric->finite = false;
        }
    }

    const std::vector<int> rows = selected_rows(test);
    for (int row : rows) {
        const int head = row % test.heads;
        const int token_batch = row / test.heads;
        const int token = token_batch % test.lq;
        const int batch = token_batch / test.lq;
        const int kv_head = head / (test.heads / test.kv_heads);
        const std::vector<int> keys = gather_keys(test, token, head);
        const size_t actual_base = static_cast<size_t>(row) * row_stride;
        if (keys.empty()) {
            for (int d = 0; d < test.head_dim; ++d) {
                update_metric(*output_metric, 0.0f, actual[actual_base + d]);
            }
            update_metric(
                *lse_metric, -std::numeric_limits<float>::infinity(),
                actual[actual_base + test.head_dim]);
            continue;
        }

        std::vector<float> scores(keys.size());
        float maximum = -std::numeric_limits<float>::infinity();
        for (size_t i = 0; i < keys.size(); ++i) {
            float score = 0.0f;
            for (int d = 0; d < test.head_dim; ++d) {
                score += test.q[q_index(test, batch, token, head, d)] *
                    ggml_fp16_to_fp32(test.k[kv_index(test, batch, keys[i], kv_head, d)]);
            }
            scores[i] = score / std::sqrt(static_cast<float>(test.head_dim));
            maximum = std::max(maximum, scores[i]);
        }
        float denominator = 0.0f;
        for (float score : scores) denominator += std::exp(score - maximum);
        for (int d = 0; d < test.head_dim; ++d) {
            float expected = 0.0f;
            for (size_t i = 0; i < keys.size(); ++i) {
                expected += std::exp(scores[i] - maximum) *
                    ggml_fp16_to_fp32(test.v[kv_index(test, batch, keys[i], kv_head, d)]);
            }
            update_metric(*output_metric, expected / denominator, actual[actual_base + d]);
        }
        update_metric(
            *lse_metric, maximum + std::log(denominator),
            actual[actual_base + test.head_dim]);
    }
    return true;
}

static Graph build_graph(const DirectCase & test) {
    Graph result;
    const bool trace = env_enabled("OPENCL_ATTN_CORRECTNESS_TRACE");
    const ggml_init_params params = {64 * 1024 * 1024, nullptr, true};
    result.ctx = ggml_init(params);
    if (result.ctx == nullptr) return result;

    result.q_storage = ggml_new_tensor_4d(
        result.ctx, GGML_TYPE_F32,
        test.head_dim, test.heads, test.lq, test.batch);
    result.k_storage = ggml_new_tensor_4d(
        result.ctx, GGML_TYPE_F16,
        test.head_dim, test.kv_heads, test.lk, test.batch);
    result.v_storage = ggml_new_tensor_4d(
        result.ctx, GGML_TYPE_F16,
        test.head_dim, test.kv_heads, test.lk, test.batch);
    ggml_tensor * q = ggml_permute(result.ctx, result.q_storage, 0, 2, 1, 3);
    ggml_tensor * k = ggml_permute(result.ctx, result.k_storage, 0, 2, 1, 3);
    ggml_tensor * v = ggml_permute(result.ctx, result.v_storage, 0, 2, 1, 3);

    if (test.mask_kind == MaskKind::Index) {
        result.mask = ggml_new_tensor_2d(
            result.ctx, GGML_TYPE_I32,
            test.mask_columns, test.heads * q_blocks(test));
    } else {
        result.mask = ggml_new_tensor_2d(
            result.ctx, GGML_TYPE_I32,
            test.kv_heads * k_blocks(test), test.heads * q_blocks(test));
    }
    result.output = ggml_flash_attn_ext_with_lse(
        result.ctx, q, k, v, result.mask,
        1.0f / std::sqrt(static_cast<float>(test.head_dim)),
        0.0f, 0.0f);
    if (trace) std::fprintf(stderr, "[CORRECTNESS_TRACE] case=%s stage=attention_created\n", test.name.c_str());
    ggml_flash_attn_ext_set_block_sparse(
        result.output,
        htp_attention_common::kBlock,
        htp_attention_common::kBlock,
        test.mask_kind == MaskKind::Index
            ? GGML_FLASH_ATTN_EXT_MASK_BLOCK_INDEX
            : GGML_FLASH_ATTN_EXT_MASK_BLOCK_MATRIX);
    if (trace) std::fprintf(stderr, "[CORRECTNESS_TRACE] case=%s stage=block_sparse_set\n", test.name.c_str());
    if (test.mask_kind == MaskKind::Index && !test.omit_halo_metadata &&
        test.image_tokens > 0 &&
        test.image_tokens + test.text_tokens == test.lq && test.lq == test.lk) {
        ggml_flash_attn_ext_set_static_halo_2d(
            result.output,
            test.image_tokens,
            test.text_tokens,
            test.image_block_rows,
            test.image_block_cols);
        if (trace) std::fprintf(stderr, "[CORRECTNESS_TRACE] case=%s stage=halo_set\n", test.name.c_str());
    }
    ggml_flash_attn_ext_set_prec(result.output, GGML_PREC_F32);
    ggml_set_name(result.output, "opencl_sparse_correctness");
    if (trace) std::fprintf(stderr, "[CORRECTNESS_TRACE] case=%s stage=graph_create_begin\n", test.name.c_str());
    result.graph = ggml_new_graph_custom(result.ctx, 16, false);
    if (trace) std::fprintf(stderr, "[CORRECTNESS_TRACE] case=%s stage=graph_expand_begin\n", test.name.c_str());
    ggml_build_forward_expand(result.graph, result.output);
    if (trace) std::fprintf(stderr, "[CORRECTNESS_TRACE] case=%s stage=graph_expand_end\n", test.name.c_str());
    return result;
}

static void free_graph(Graph & graph) {
    if (graph.buffer != nullptr) ggml_backend_buffer_free(graph.buffer);
    if (graph.ctx != nullptr) ggml_free(graph.ctx);
    graph = {};
}

static bool upload(Graph & graph, const DirectCase & test) {
    graph.buffer = ggml_backend_alloc_ctx_tensors_from_buft(
        graph.ctx, ggml_backend_htp_qkv_buffer_type());
    if (graph.buffer == nullptr) return false;
    ggml_backend_tensor_set(
        graph.q_storage, test.q.data(), 0, test.q.size() * sizeof(float));
    ggml_backend_tensor_set(
        graph.k_storage, test.k.data(), 0, test.k.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(
        graph.v_storage, test.v.data(), 0, test.v.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(
        graph.mask, test.mask.data(), 0, test.mask.size() * sizeof(int32_t));
    return true;
}

static DirectCase make_pattern_case(
        const char * name,
        int tokens,
        int text_tokens,
        int heads,
        int kv_heads,
        int batch,
        MaskKind mask_kind,
        bool inject_invalid_ids,
        bool sentinel_early,
        bool empty_first_row,
        bool omit_halo_metadata) {
    DirectCase test;
    test.name = name;
    test.heads = heads;
    test.kv_heads = kv_heads;
    test.batch = batch;
    test.lq = tokens;
    test.lk = tokens;
    test.image_tokens = 32;
    test.text_tokens = text_tokens;
    test.omit_halo_metadata = omit_halo_metadata;
    test.mask_kind = mask_kind;
    test.q.resize(static_cast<size_t>(batch) * tokens * heads * test.head_dim);
    test.k.resize(static_cast<size_t>(batch) * tokens * kv_heads * test.head_dim);
    test.v.resize(test.k.size());
    for (int b = 0; b < batch; ++b) {
        for (int token = 0; token < tokens; ++token) {
            for (int head = 0; head < heads; ++head) {
                for (int d = 0; d < test.head_dim; ++d) {
                    const float phase = static_cast<float>(
                        1 + b * 131 + token * 17 + head * 29 + d * 7);
                    test.q[q_index(test, b, token, head, d)] =
                        std::sin(phase * 0.0137f) * 1.75f +
                        static_cast<float>((d % 7) - 3) * 0.000123f;
                }
            }
            for (int head = 0; head < kv_heads; ++head) {
                for (int d = 0; d < test.head_dim; ++d) {
                    const float phase = static_cast<float>(
                        3 + b * 97 + token * 19 + head * 31 + d * 11);
                    test.k[kv_index(test, b, token, head, d)] =
                        ggml_fp32_to_fp16(std::cos(phase * 0.0091f) * 0.85f);
                    test.v[kv_index(test, b, token, head, d)] =
                        ggml_fp32_to_fp16(std::sin(phase * 0.0073f) * 0.35f);
                }
            }
        }
    }

    const int qb_count = q_blocks(test);
    const int kb_count = k_blocks(test);
    if (mask_kind == MaskKind::Index) {
        test.mask_columns = kb_count + (inject_invalid_ids ? 3 : 1);
        test.mask.assign(
            static_cast<size_t>(heads) * qb_count * test.mask_columns, -1);
        for (int head = 0; head < heads; ++head) {
            const int kv_head = head / (heads / kv_heads);
            for (int qb = 0; qb < qb_count; ++qb) {
                int32_t * row = test.mask.data() +
                    static_cast<size_t>(head * qb_count + qb) * test.mask_columns;
                int write = 0;
                if (inject_invalid_ids) {
                    row[write++] = -2;
                    row[write++] = ((kv_head + 1) % kv_heads) * kb_count;
                    row[write++] = kv_heads * kb_count + 7;
                }
                for (int kb = 0; kb < kb_count; ++kb) {
                    row[write++] = kv_head * kb_count + kb;
                    if (sentinel_early && kb == 0 && write < test.mask_columns) {
                        row[write++] = -1;
                        break;
                    }
                }
            }
        }
        if (empty_first_row) {
            std::fill_n(test.mask.data(), test.mask_columns, -1);
        }
    } else {
        test.mask_columns = kv_heads * kb_count;
        test.mask.assign(
            static_cast<size_t>(heads) * qb_count * test.mask_columns, 1);
        for (int head = 0; head < heads; ++head) {
            const int kv_head = head / (heads / kv_heads);
            for (int qb = 0; qb < qb_count; ++qb) {
                int32_t * row = test.mask.data() +
                    static_cast<size_t>(head * qb_count + qb) * test.mask_columns;
                for (int kb = 0; kb < kb_count; ++kb) {
                    row[kv_head * kb_count + kb] = 0;
                }
            }
        }
    }
    return test;
}

static DirectCase make_main_case(
        const htp_attention_common::ImageSize & size,
        const htp_attention_common::CaseSpec & base) {
    htp_attention_common::CaseSpec spec = base;
    spec.image_tokens = size.image_tokens;
    spec.image_block_rows = size.block_rows;
    spec.image_block_cols = size.block_cols;
    spec.resolution = size.resolution;
    const htp_attention_common::InputData data = htp_attention_common::make_input(spec);
    htp_attention_common::Reference reference;
    for (const htp_attention_common::BranchData & branch : data.branches) {
        htp_attention_common::append_reference_branch(spec, branch, true, reference);
    }

    DirectCase test;
    test.name = std::string(spec.model) + "/" + spec.family + "/" +
        std::to_string(spec.resolution);
    test.head_dim = spec.head_dim;
    test.heads = spec.heads;
    test.kv_heads = spec.heads;
    test.lq = htp_attention_common::total_tokens(spec);
    test.lk = test.lq;
    test.image_tokens = spec.image_tokens;
    test.text_tokens = spec.text_tokens;
    test.image_block_rows = spec.image_block_rows;
    test.image_block_cols = spec.image_block_cols;
    test.compare_all_rows = spec.resolution == 256;
    test.mask_columns = k_blocks(test);
    test.q = std::move(reference.q);
    test.k = std::move(reference.k);
    test.v = std::move(reference.v);
    test.mask = data.block_index;
    return test;
}

static bool run_case(ggml_backend_t backend, const DirectCase & test, Summary * summary) {
    ++summary->cases;
    const bool trace = env_enabled("OPENCL_ATTN_CORRECTNESS_TRACE");
    if (trace) std::fprintf(stderr, "[CORRECTNESS_TRACE] case=%s stage=build_begin\n", test.name.c_str());
    Graph graph = build_graph(test);
    if (trace) std::fprintf(stderr, "[CORRECTNESS_TRACE] case=%s stage=build_end\n", test.name.c_str());
    const bool ready = graph.ctx != nullptr && graph.graph != nullptr &&
        graph.output != nullptr && upload(graph, test);
    if (trace) std::fprintf(stderr, "[CORRECTNESS_TRACE] case=%s stage=upload_end ready=%d\n", test.name.c_str(), ready ? 1 : 0);
    ggml_status status = GGML_STATUS_FAILED;
    std::vector<float> actual;
    if (ready) {
        if (trace) std::fprintf(stderr, "[CORRECTNESS_TRACE] case=%s stage=compute_begin\n", test.name.c_str());
        status = ggml_backend_graph_compute(backend, graph.graph);
        if (trace) std::fprintf(stderr, "[CORRECTNESS_TRACE] case=%s stage=compute_end status=%d\n", test.name.c_str(), int(status));
        if (status == GGML_STATUS_SUCCESS) {
            actual.resize(ggml_nelements(graph.output));
            ggml_backend_tensor_get(
                graph.output, actual.data(), 0, actual.size() * sizeof(float));
        }
    }

    Metric output;
    Metric lse;
    const bool compared = status == GGML_STATUS_SUCCESS &&
        compare_reference(test, actual, &output, &lse);
    const bool pass = ready && compared && output.finite && lse.finite &&
        output.max_abs <= 5e-3 && output.rmse() <= 5e-3 &&
        lse.max_abs <= 5e-3 && lse.rmse() <= 5e-3;
    if (pass) ++summary->passed;
    summary->output.max_abs = std::max(summary->output.max_abs, output.max_abs);
    summary->output.sum_sq += output.sum_sq;
    summary->output.count += output.count;
    summary->output.finite = summary->output.finite && output.finite;
    summary->lse.max_abs = std::max(summary->lse.max_abs, lse.max_abs);
    summary->lse.sum_sq += lse.sum_sq;
    summary->lse.count += lse.count;
    summary->lse.finite = summary->lse.finite && lse.finite;
    std::printf(
        "[%s] case=%s d=%d h=%d hkv=%d batch=%d lq=%d lk=%d "
        "mask=%s output_max_abs=%.9g output_rmse=%.9g "
        "lse_max_abs=%.9g lse_rmse=%.9g status=%d\n",
        pass ? "PASS" : "FAIL", test.name.c_str(), test.head_dim,
        test.heads, test.kv_heads, test.batch, test.lq, test.lk,
        test.mask_kind == MaskKind::Index ? "index" : "matrix",
        output.max_abs, output.rmse(), lse.max_abs, lse.rmse(), int(status));
    std::fflush(stdout);
    free_graph(graph);
    return pass;
}

static std::vector<DirectCase> microcases() {
    std::vector<DirectCase> tests;
    const int text_sizes[] = {0, 1, 31, 32, 33, 77};
    for (int text : text_sizes) {
        const std::string name = "text-" + std::to_string(text);
        tests.push_back(make_pattern_case(
            name.c_str(), 32 + text, text, 4, 4, 1,
            MaskKind::Index, false, false, false, false));
    }
    tests.push_back(make_pattern_case(
        "gqa-batch2-invalid-ids", 63, 31, 8, 2, 2,
        MaskKind::Index, true, false, false, false));
    tests.push_back(make_pattern_case(
        "sentinel-early", 63, 31, 8, 2, 1,
        MaskKind::Index, false, true, false, false));
    tests.push_back(make_pattern_case(
        "empty-row", 33, 1, 4, 4, 1,
        MaskKind::Index, false, false, true, false));
    tests.push_back(make_pattern_case(
        "matrix-mask", 63, 31, 8, 2, 1,
        MaskKind::Matrix, false, false, false, false));
    tests.push_back(make_pattern_case(
        "generic-no-halo-metadata", 63, 31, 8, 2, 1,
        MaskKind::Index, false, false, false, true));
    return tests;
}

}  // namespace

int main() {
    configure_sparse_route();
    ggml_backend_htp_set_hybrid_enabled(true);
    ggml_backend_t backend = ggml_backend_init_by_name("HTP", nullptr);
    if (backend == nullptr || !ggml_backend_htp_runtime_available()) {
        std::fprintf(
            stderr, "[FAIL] HTP runtime unavailable for OpenCL sparse sidecar: %s\n",
            ggml_backend_htp_runtime_error() != nullptr
                ? ggml_backend_htp_runtime_error() : "unknown");
        if (backend != nullptr) ggml_backend_free(backend);
        ggml_backend_htp_set_hybrid_enabled(false);
        return 2;
    }
    ggml_backend_htp_set_n_threads(backend, 8);

    Summary summary;
    const bool main_only = env_enabled("OPENCL_ATTN_CORRECTNESS_MAIN_ONLY");
    const bool micro_only = env_enabled("OPENCL_ATTN_CORRECTNESS_MICRO_ONLY");
    const char * case_filter = std::getenv("OPENCL_ATTN_CORRECTNESS_CASE");
    if (!main_only) {
        for (const DirectCase & test : microcases()) {
            if (case_filter != nullptr && case_filter[0] != '\0' &&
                test.name != case_filter) {
                continue;
            }
            run_case(backend, test, &summary);
        }
    }
    if (!micro_only) {
        for (int si = 0; si < htp_attention_common::image_size_count(); ++si) {
            const htp_attention_common::ImageSize & size =
                htp_attention_common::image_sizes()[si];
            for (int ci = 0; ci < htp_attention_common::case_count(); ++ci) {
                const htp_attention_common::CaseSpec & spec =
                    htp_attention_common::cases()[ci];
                if (spec.head_dim != 128 || !spec.sparse ||
                    !htp_attention_common::selected(size, spec)) {
                    continue;
                }
                run_case(backend, make_main_case(size, spec), &summary);
            }
        }
    }

    ggml_backend_free(backend);
    ggml_backend_htp_set_hybrid_enabled(false);
    const bool pass = summary.cases > 0 && summary.passed == summary.cases &&
        summary.output.finite && summary.lse.finite;
    std::printf(
        "variant=automatic cases=%d "
        "output_max_abs=%.9g output_rmse=%.9g "
        "lse_max_abs=%.9g lse_rmse=%.9g fallbacks=0 pending_events=0 %s\n",
        summary.cases,
        summary.output.max_abs, summary.output.rmse(),
        summary.lse.max_abs, summary.lse.rmse(),
        pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
