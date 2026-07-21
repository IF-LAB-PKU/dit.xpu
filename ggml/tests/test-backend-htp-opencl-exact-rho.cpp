#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-htp.h"
#include "ggml.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include <unistd.h>

namespace {

constexpr int64_t k_head_dim = 64;
constexpr int64_t k_q_tokens = 32;
constexpr int64_t k_kv_tokens = 64;
constexpr int64_t k_heads = 2;
constexpr int64_t k_kv_heads = 2;
constexpr int64_t k_batch = 1;
constexpr int32_t k_block_size = 32;
constexpr int64_t k_packed_dim = k_head_dim + 1;
constexpr float k_scale = 0.125f;

class env_guard {
public:
    env_guard(const char * name, const char * value) : name_(name) {
        const char * previous = std::getenv(name);
        if (previous != nullptr) {
            had_previous_ = true;
            previous_ = previous;
        }
        GGML_ASSERT(setenv(name, value, 1) == 0);
    }

    ~env_guard() {
        if (had_previous_) {
            setenv(name_.c_str(), previous_.c_str(), 1);
        } else {
            unsetenv(name_.c_str());
        }
    }

private:
    std::string name_;
    std::string previous_;
    bool had_previous_ = false;
};

struct graph_fixture {
    ggml_context * ctx = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * q_storage = nullptr;
    ggml_tensor * q = nullptr;
    ggml_tensor * k_storage = nullptr;
    ggml_tensor * v_storage = nullptr;
    ggml_tensor * k = nullptr;
    ggml_tensor * v = nullptr;
    ggml_tensor * dense_mask = nullptr;
    ggml_tensor * sparse_mask = nullptr;
    ggml_tensor * dense_packed = nullptr;
    ggml_tensor * sparse_packed = nullptr;
    ggml_tensor * rho = nullptr;
    ggml_tensor * residual = nullptr;
    ggml_tensor * recovered = nullptr;
    ggml_gallocr_t allocator = nullptr;
};

struct run_result {
    ggml_status status = GGML_STATUS_FAILED;
    std::string trace;
    std::vector<float> dense_packed;
    std::vector<float> sparse_packed;
    std::vector<float> rho;
    std::vector<float> residual;
    std::vector<float> recovered;
};

size_t q_index(int64_t d, int64_t qt, int64_t h, int64_t b = 0) {
    return static_cast<size_t>(d + k_head_dim * (qt + k_q_tokens * (h + k_heads * b)));
}

size_t kv_index(int64_t d, int64_t kt, int64_t h, int64_t b = 0) {
    return static_cast<size_t>(d + k_head_dim * (kt + k_kv_tokens * (h + k_kv_heads * b)));
}

size_t packed_index(int64_t d, int64_t h, int64_t qt, int64_t b = 0) {
    return static_cast<size_t>(d + k_packed_dim * (h + k_heads * (qt + k_q_tokens * b)));
}

std::vector<float> make_q() {
    std::vector<float> result(static_cast<size_t>(k_head_dim * k_q_tokens * k_heads * k_batch));
    for (int64_t h = 0; h < k_heads; ++h) {
        for (int64_t qt = 0; qt < k_q_tokens; ++qt) {
            for (int64_t d = 0; d < k_head_dim; ++d) {
                const float phase = 0.017f * float((d + 1) * (qt + 1)) + 0.61f * float(h);
                result[q_index(d, qt, h)] =
                    0.45f * std::sin(phase) + 0.05f * std::cos(0.13f * float(d + 3 * h));
            }
        }
    }
    return result;
}

void make_kv(std::vector<ggml_fp16_t> & k, std::vector<ggml_fp16_t> & v) {
    k.resize(static_cast<size_t>(k_head_dim * k_kv_tokens * k_kv_heads * k_batch));
    v.resize(k.size());
    for (int64_t h = 0; h < k_kv_heads; ++h) {
        for (int64_t kt = 0; kt < k_kv_tokens; ++kt) {
            for (int64_t d = 0; d < k_head_dim; ++d) {
                const size_t index = kv_index(d, kt, h);
                const float k_value =
                    0.42f * std::cos(0.011f * float((d + 2) * (kt + 1)) + 0.83f * float(h));
                const float v_value = 0.75f * (
                    0.36f * std::sin(0.019f * float((d + 1) * (kt + 3)) + 0.97f * float(h)) +
                    0.4f * float(h));
                k[index] = ggml_fp32_to_fp16(k_value);
                v[index] = ggml_fp32_to_fp16(v_value);
            }
        }
    }
}

std::vector<float> reference_attention(
        const std::vector<float> & q,
        const std::vector<ggml_fp16_t> & k,
        const std::vector<ggml_fp16_t> & v,
        bool sparse) {
    std::vector<float> packed(
        static_cast<size_t>(k_packed_dim * k_heads * k_q_tokens * k_batch), 0.0f);
    std::vector<float> scores(static_cast<size_t>(k_kv_tokens));

    for (int64_t qt = 0; qt < k_q_tokens; ++qt) {
        for (int64_t h = 0; h < k_heads; ++h) {
            const int64_t kv_head = h / (k_heads / k_kv_heads);
            float maximum = -std::numeric_limits<float>::infinity();
            for (int64_t kt = 0; kt < k_kv_tokens; ++kt) {
                if (sparse && kt >= k_block_size) {
                    scores[static_cast<size_t>(kt)] = -std::numeric_limits<float>::infinity();
                    continue;
                }
                float dot = 0.0f;
                for (int64_t d = 0; d < k_head_dim; ++d) {
                    dot += q[q_index(d, qt, h)] *
                        ggml_fp16_to_fp32(k[kv_index(d, kt, kv_head)]);
                }
                const float score = dot * k_scale;
                scores[static_cast<size_t>(kt)] = score;
                maximum = std::max(maximum, score);
            }

            float sum = 0.0f;
            const size_t row = packed_index(0, h, qt);
            for (int64_t kt = 0; kt < k_kv_tokens; ++kt) {
                if (!std::isfinite(scores[static_cast<size_t>(kt)])) {
                    continue;
                }
                const float weight = std::exp(scores[static_cast<size_t>(kt)] - maximum);
                sum += weight;
                for (int64_t d = 0; d < k_head_dim; ++d) {
                    packed[row + static_cast<size_t>(d)] +=
                        weight * ggml_fp16_to_fp32(v[kv_index(d, kt, kv_head)]);
                }
            }
            for (int64_t d = 0; d < k_head_dim; ++d) {
                packed[row + static_cast<size_t>(d)] /= sum;
            }
            packed[row + k_head_dim] = maximum + std::log(sum);
        }
    }
    return packed;
}

void fill_sparse_index(ggml_tensor * dst, int ith, int nth, void * userdata) {
    GGML_UNUSED(nth);
    GGML_UNUSED(userdata);
    if (ith != 0) {
        return;
    }
    GGML_ASSERT(dst != nullptr && dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_nelements(dst) == k_heads);
    auto * values = static_cast<int32_t *>(dst->data);
    values[0] = 0;
    values[1] = 2;
}

graph_fixture make_graph(ggml_backend_t backend) {
    const ggml_init_params params = {
        /* .mem_size   = */ 4 * 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    graph_fixture f;
    f.ctx = ggml_init(params);
    GGML_ASSERT(f.ctx != nullptr);

    f.q_storage = ggml_new_tensor_4d(
        f.ctx, GGML_TYPE_F32, k_head_dim, k_heads, k_q_tokens, k_batch);
    f.q = ggml_view_4d(
        f.ctx, f.q_storage, k_head_dim, k_q_tokens, k_heads, k_batch,
        f.q_storage->nb[2], f.q_storage->nb[1], f.q_storage->nb[3], 0);
    f.k_storage = ggml_new_tensor_4d(
        f.ctx, GGML_TYPE_F16, k_head_dim, k_kv_heads, k_kv_tokens, k_batch);
    f.v_storage = ggml_new_tensor_4d(
        f.ctx, GGML_TYPE_F16, k_head_dim, k_kv_heads, k_kv_tokens, k_batch);
    f.k = ggml_view_4d(
        f.ctx, f.k_storage, k_head_dim, k_kv_tokens, k_kv_heads, k_batch,
        f.k_storage->nb[2], f.k_storage->nb[1], f.k_storage->nb[3], 0);
    f.v = ggml_view_4d(
        f.ctx, f.v_storage, k_head_dim, k_kv_tokens, k_kv_heads, k_batch,
        f.v_storage->nb[2], f.v_storage->nb[1], f.v_storage->nb[3], 0);
    f.k->op = GGML_OP_NONE;
    f.v->op = GGML_OP_NONE;
    f.q->op = GGML_OP_NONE;
    f.dense_mask = ggml_new_tensor_4d(f.ctx, GGML_TYPE_F16, k_kv_tokens, k_q_tokens, 1, 1);
    // Production halo masks are input-less GGML_OP_CUSTOM side branches. They
    // must be prepared before their sparse attention can be submitted early.
    f.sparse_mask = ggml_custom_4d(
        f.ctx, GGML_TYPE_I32, 1, k_heads, 1, 1,
        nullptr, 0, fill_sparse_index, 1, nullptr);
    ggml_set_name(f.q_storage, "exact_rho_q_storage");
    ggml_set_name(f.q, "exact_rho_q");
    ggml_set_name(f.k_storage, "exact_rho_k_storage");
    ggml_set_name(f.v_storage, "exact_rho_v_storage");
    ggml_set_name(f.k, "exact_rho_k");
    ggml_set_name(f.v, "exact_rho_v");
    ggml_set_name(f.dense_mask, "exact_rho_dense_mask");
    ggml_set_name(f.sparse_mask, "exact_rho_sparse_index");
    ggml_set_input(f.q_storage);
    ggml_set_input(f.k_storage);
    ggml_set_input(f.v_storage);
    ggml_set_input(f.dense_mask);
    ggml_set_input(f.sparse_mask);

    f.dense_packed = ggml_flash_attn_ext_with_lse(
        f.ctx, f.q, f.k, f.v, f.dense_mask, k_scale, 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(f.dense_packed, GGML_PREC_F32);
    ggml_set_name(f.dense_packed, "exact_rho_dense_packed");

    f.sparse_packed = ggml_flash_attn_ext_with_lse(
        f.ctx, f.q, f.k, f.v, f.sparse_mask, k_scale, 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(f.sparse_packed, GGML_PREC_F32);
    ggml_flash_attn_ext_set_block_sparse(
        f.sparse_packed, k_block_size, k_block_size, GGML_FLASH_ATTN_EXT_MASK_BLOCK_INDEX);
    ggml_set_name(f.sparse_packed, "exact_rho_sparse_packed");

    ggml_tensor * dense_o = ggml_view_4d(
        f.ctx, f.dense_packed, k_head_dim, k_heads, k_q_tokens, k_batch,
        f.dense_packed->nb[1], f.dense_packed->nb[2], f.dense_packed->nb[3], 0);
    ggml_tensor * sparse_o = ggml_view_4d(
        f.ctx, f.sparse_packed, k_head_dim, k_heads, k_q_tokens, k_batch,
        f.sparse_packed->nb[1], f.sparse_packed->nb[2], f.sparse_packed->nb[3], 0);
    ggml_tensor * dense_lse = ggml_view_4d(
        f.ctx, f.dense_packed, 1, k_heads, k_q_tokens, k_batch,
        f.dense_packed->nb[1], f.dense_packed->nb[2], f.dense_packed->nb[3],
        k_head_dim * sizeof(float));
    ggml_tensor * sparse_lse = ggml_view_4d(
        f.ctx, f.sparse_packed, 1, k_heads, k_q_tokens, k_batch,
        f.sparse_packed->nb[1], f.sparse_packed->nb[2], f.sparse_packed->nb[3],
        k_head_dim * sizeof(float));
    ggml_set_name(dense_o, "exact_rho_dense_o");
    ggml_set_name(sparse_o, "exact_rho_sparse_o");
    ggml_set_name(dense_lse, "exact_rho_dense_lse");
    ggml_set_name(sparse_lse, "exact_rho_sparse_lse");

    f.rho = ggml_exp(f.ctx, ggml_sub(f.ctx, sparse_lse, dense_lse));
    ggml_set_name(f.rho, "exact_rho_rho");
    ggml_tensor * scaled_sparse = ggml_mul(f.ctx, sparse_o, f.rho);
    ggml_set_name(scaled_sparse, "exact_rho_scaled_sparse");
    f.residual = ggml_sub(f.ctx, dense_o, scaled_sparse);
    ggml_set_name(f.residual, "exact_rho_residual");
    f.recovered = ggml_add(f.ctx, ggml_mul(f.ctx, sparse_o, f.rho), f.residual);
    ggml_set_name(f.recovered, "exact_rho_recovered");
    ggml_set_output(f.dense_packed);
    ggml_set_output(f.sparse_packed);
    ggml_set_output(f.rho);
    ggml_set_output(f.residual);
    ggml_set_output(f.recovered);

    f.graph = ggml_new_graph(f.ctx);
    GGML_ASSERT(f.graph != nullptr);
    // Match the production anchor graph: the dense model result is expanded
    // first, while the exact-rho residual is a cache-only side branch added
    // afterward. Early OpenCL dispatch must not depend on cache-root order.
    ggml_build_forward_expand(f.graph, dense_o);
    ggml_build_forward_expand(f.graph, f.recovered);
    f.allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    GGML_ASSERT(f.allocator != nullptr);
    GGML_ASSERT(ggml_gallocr_alloc_graph(f.allocator, f.graph));
    return f;
}

void initialize_graph_inputs(graph_fixture & fixture) {
    const std::vector<float> q = make_q();
    std::vector<float> q_physical(q.size());
    for (int64_t h = 0; h < k_heads; ++h) {
        for (int64_t qt = 0; qt < k_q_tokens; ++qt) {
            for (int64_t d = 0; d < k_head_dim; ++d) {
                const size_t physical = static_cast<size_t>(
                    d + k_head_dim * (h + k_heads * qt));
                q_physical[physical] = q[q_index(d, qt, h)];
            }
        }
    }

    std::vector<ggml_fp16_t> k;
    std::vector<ggml_fp16_t> v;
    make_kv(k, v);
    std::vector<ggml_fp16_t> k_physical(k.size());
    std::vector<ggml_fp16_t> v_physical(v.size());
    for (int64_t h = 0; h < k_kv_heads; ++h) {
        for (int64_t kt = 0; kt < k_kv_tokens; ++kt) {
            for (int64_t d = 0; d < k_head_dim; ++d) {
                const size_t logical = kv_index(d, kt, h);
                const size_t physical = static_cast<size_t>(
                    d + k_head_dim * (h + k_kv_heads * kt));
                k_physical[physical] = k[logical];
                v_physical[physical] = v[logical];
            }
        }
    }

    const std::vector<ggml_fp16_t> dense_mask(
        static_cast<size_t>(k_kv_tokens * k_q_tokens), ggml_fp32_to_fp16(0.0f));
    ggml_backend_tensor_set(
        fixture.q_storage, q_physical.data(), 0, q_physical.size() * sizeof(float));
    ggml_backend_tensor_set(
        fixture.k_storage, k_physical.data(), 0, k_physical.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(
        fixture.v_storage, v_physical.data(), 0, v_physical.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(
        fixture.dense_mask, dense_mask.data(), 0, dense_mask.size() * sizeof(ggml_fp16_t));
}

std::string capture_graph_compute(
        ggml_backend_t backend,
        ggml_cgraph * graph,
        ggml_status * status) {
    FILE * capture = std::tmpfile();
    GGML_ASSERT(capture != nullptr);
    const int saved_stderr = dup(STDERR_FILENO);
    GGML_ASSERT(saved_stderr >= 0);
    GGML_ASSERT(std::fflush(stderr) == 0);
    GGML_ASSERT(dup2(fileno(capture), STDERR_FILENO) >= 0);

    *status = ggml_backend_graph_compute(backend, graph);

    GGML_ASSERT(std::fflush(stderr) == 0);
    GGML_ASSERT(dup2(saved_stderr, STDERR_FILENO) >= 0);
    close(saved_stderr);
    GGML_ASSERT(std::fseek(capture, 0, SEEK_END) == 0);
    const long length = std::ftell(capture);
    GGML_ASSERT(length >= 0);
    GGML_ASSERT(std::fseek(capture, 0, SEEK_SET) == 0);
    std::string result(static_cast<size_t>(length), '\0');
    if (length > 0) {
        GGML_ASSERT(std::fread(result.data(), 1, result.size(), capture) == result.size());
    }
    std::fclose(capture);
    return result;
}

std::vector<float> read_f32(const ggml_tensor * tensor) {
    std::vector<float> result(static_cast<size_t>(ggml_nelements(tensor)));
    ggml_backend_tensor_get(tensor, result.data(), 0, result.size() * sizeof(float));
    return result;
}

run_result run_graph(
        ggml_backend_t backend,
        graph_fixture & f,
        bool sidecar_enabled,
        bool async_enabled) {
    env_guard sidecar("SD_HTP_SPARSE_OPENCL", sidecar_enabled ? "1" : "0");
    env_guard async_sidecar("SD_HTP_SPARSE_OPENCL_ASYNC", async_enabled ? "1" : "0");
    initialize_graph_inputs(f);
    run_result result;
    result.trace = capture_graph_compute(backend, f.graph, &result.status);
    if (result.status == GGML_STATUS_SUCCESS) {
        result.dense_packed = read_f32(f.dense_packed);
        result.sparse_packed = read_f32(f.sparse_packed);
        result.rho = read_f32(f.rho);
        result.residual = read_f32(f.residual);
        result.recovered = read_f32(f.recovered);
    }
    return result;
}

int count_attention_trace(const std::string & trace, const char * route, int block_sparse);

void require_async_event_order(const std::string & trace) {
    const char * expected[] = {
        "gpu_sparse_early_enqueue",
        "gpu_sparse_enqueue",
        "gpu_sparse_flush",
        "htp_dense_begin",
        "htp_dense_end",
        "gpu_sparse_wait_begin",
        "gpu_sparse_wait_end",
        "rho_consumer_begin",
    };
    struct event_record {
        const char * name;
        int64_t ts_us;
    } records[sizeof(expected) / sizeof(expected[0])] = {};
    size_t position = 0;
    size_t record_count = 0;
    for (const char * event : expected) {
        const size_t found = trace.find(event, position);
        if (found == std::string::npos) {
            std::fprintf(
                stderr,
                "[FAIL] missing or out-of-order async event '%s' after offset %zu\n--- trace ---\n%s",
                event, position, trace.c_str());
            GGML_ASSERT(false && "OpenCL sparse/HTP dense overlap event order mismatch");
        }
        const size_t line_begin = trace.rfind('\n', found);
        const size_t line_end = trace.find('\n', found);
        const size_t line_start = line_begin == std::string::npos ? 0 : line_begin + 1;
        const size_t line_length = line_end == std::string::npos ?
            std::string::npos : line_end - line_start;
        const std::string line = trace.substr(line_start, line_length);
        const std::string timestamp_prefix = "ts_us=";
        const size_t timestamp_begin = line.find(timestamp_prefix);
        GGML_ASSERT(timestamp_begin != std::string::npos);
        const size_t timestamp_value_begin = timestamp_begin + timestamp_prefix.size();
        const size_t timestamp_end = line.find_first_not_of("0123456789", timestamp_value_begin);
        const std::string timestamp = line.substr(
            timestamp_value_begin,
            timestamp_end == std::string::npos ? std::string::npos : timestamp_end - timestamp_value_begin);
        GGML_ASSERT(!timestamp.empty());
        records[record_count++] = { event, std::strtoll(timestamp.c_str(), nullptr, 10) };
        std::printf(
            "[ASYNC_ORDER] %s\n",
            line.c_str());
        position = found + std::strlen(event);
    }
    for (size_t i = 1; i < record_count; ++i) {
        GGML_ASSERT(records[i - 1].ts_us <= records[i].ts_us);
    }
    GGML_ASSERT(records[0].ts_us <= records[1].ts_us);
    GGML_ASSERT(records[1].ts_us < records[3].ts_us);
    GGML_ASSERT(records[3].ts_us < records[5].ts_us);
    GGML_ASSERT(records[4].ts_us <= records[5].ts_us);
    GGML_ASSERT(records[5].ts_us <= records[6].ts_us);
    GGML_ASSERT(records[6].ts_us <= records[7].ts_us);
    GGML_ASSERT(count_attention_trace(trace, "opencl", 1) == 1);
}

int count_attention_trace(const std::string & trace, const char * route, int block_sparse) {
    int count = 0;
    size_t begin = 0;
    const std::string route_text = std::string("attention_timing route=") + route;
    const std::string sparse_text = std::string("block_sparse=") + (block_sparse ? "1" : "0");
    while (begin < trace.size()) {
        const size_t end = trace.find('\n', begin);
        const std::string line = trace.substr(begin, end == std::string::npos ? end : end - begin);
        if (line.find(route_text) != std::string::npos && line.find(sparse_text) != std::string::npos) {
            ++count;
        }
        if (end == std::string::npos) {
            break;
        }
        begin = end + 1;
    }
    return count;
}

float max_abs_diff(const std::vector<float> & lhs, const std::vector<float> & rhs) {
    GGML_ASSERT(lhs.size() == rhs.size());
    float worst = 0.0f;
    for (size_t i = 0; i < lhs.size(); ++i) {
        GGML_ASSERT(std::isfinite(lhs[i]) && std::isfinite(rhs[i]));
        worst = std::max(worst, std::fabs(lhs[i] - rhs[i]));
    }
    return worst;
}

struct packed_error {
    float o = 0.0f;
    float lse = 0.0f;
};

packed_error packed_max_abs_diff(
        const std::vector<float> & actual,
        const std::vector<float> & reference,
        const char * label) {
    GGML_ASSERT(actual.size() == reference.size());
    packed_error result;
    for (int64_t qt = 0; qt < k_q_tokens; ++qt) {
        for (int64_t h = 0; h < k_heads; ++h) {
            const size_t row = packed_index(0, h, qt);
            for (int64_t d = 0; d < k_head_dim; ++d) {
                GGML_ASSERT(std::isfinite(actual[row + static_cast<size_t>(d)]));
                GGML_ASSERT(std::isfinite(reference[row + static_cast<size_t>(d)]));
                result.o = std::max(
                    result.o,
                    std::fabs(actual[row + static_cast<size_t>(d)] -
                              reference[row + static_cast<size_t>(d)]));
            }
            if (!std::isfinite(actual[row + k_head_dim])) {
                std::fprintf(
                    stderr,
                    "%s has non-finite LSE at qt=%lld head=%lld index=%zu value=%g\n",
                    label,
                    (long long) qt,
                    (long long) h,
                    row + k_head_dim,
                    actual[row + k_head_dim]);
            }
            GGML_ASSERT(std::isfinite(actual[row + k_head_dim]));
            GGML_ASSERT(std::isfinite(reference[row + k_head_dim]));
            result.lse = std::max(
                result.lse,
                std::fabs(actual[row + k_head_dim] - reference[row + k_head_dim]));
        }
    }
    return result;
}

void require_routes(const run_result & off, const run_result & on) {
    const int off_sparse_htp = count_attention_trace(off.trace, "htp", 1);
    const int off_dense_htp = count_attention_trace(off.trace, "htp", 0);
    const int on_sparse_opencl = count_attention_trace(on.trace, "opencl", 1);
    const int on_sparse_htp = count_attention_trace(on.trace, "htp", 1);
    const int on_dense_htp = count_attention_trace(on.trace, "htp", 0);
    std::printf(
        "exact-rho routes off[sparse_htp=%d dense_htp=%d] "
        "on[sparse_opencl=%d sparse_htp=%d dense_htp=%d]\n",
        off_sparse_htp, off_dense_htp, on_sparse_opencl, on_sparse_htp, on_dense_htp);
    if (off_sparse_htp != 1 || off_dense_htp != 1 ||
        on_sparse_opencl != 1 || on_sparse_htp != 0 || on_dense_htp != 1) {
        std::fprintf(stderr,
            "[FAIL] hybrid route mismatch off[sparse_htp=%d dense_htp=%d] "
            "on[sparse_opencl=%d sparse_htp=%d dense_htp=%d]\n"
            "--- sidecar off trace ---\n%s--- sidecar on trace ---\n%s",
            off_sparse_htp, off_dense_htp, on_sparse_opencl, on_sparse_htp, on_dense_htp,
            off.trace.c_str(), on.trace.c_str());
        GGML_ASSERT(false && "sidecar-on sparse attention was not routed to OpenCL");
    }
}

void require_numerics(
        const run_result & off,
        const run_result & on,
        const std::vector<float> & dense_ref,
        const std::vector<float> & sparse_ref) {
    constexpr float opencl_tolerance = 2.0e-4f;
    const packed_error off_dense_error = packed_max_abs_diff(off.dense_packed, dense_ref, "off dense");
    const packed_error off_sparse_error = packed_max_abs_diff(off.sparse_packed, sparse_ref, "off sparse");
    const packed_error on_dense_error = packed_max_abs_diff(on.dense_packed, dense_ref, "on dense");
    const packed_error on_sparse_error = packed_max_abs_diff(on.sparse_packed, sparse_ref, "on sparse");
    std::fprintf(
        stderr,
        "exact-rho packed errors off[dense_o=%.9g dense_lse=%.9g sparse_o=%.9g sparse_lse=%.9g] "
        "on[dense_o=%.9g dense_lse=%.9g sparse_o=%.9g sparse_lse=%.9g]\n",
        off_dense_error.o, off_dense_error.lse, off_sparse_error.o, off_sparse_error.lse,
        on_dense_error.o, on_dense_error.lse, on_sparse_error.o, on_sparse_error.lse);
    constexpr float htp_o_tolerance = 3.0e-3f;
    constexpr float htp_lse_tolerance = 5.0e-3f;
    GGML_ASSERT(off_dense_error.o <= htp_o_tolerance);
    GGML_ASSERT(off_dense_error.lse <= htp_lse_tolerance);
    GGML_ASSERT(off_sparse_error.o <= htp_o_tolerance);
    GGML_ASSERT(off_sparse_error.lse <= htp_lse_tolerance);
    GGML_ASSERT(on_dense_error.o <= htp_o_tolerance);
    GGML_ASSERT(on_dense_error.lse <= htp_lse_tolerance);
    GGML_ASSERT(on_sparse_error.o <= opencl_tolerance);
    GGML_ASSERT(on_sparse_error.lse <= opencl_tolerance);
    GGML_ASSERT(max_abs_diff(off.dense_packed, on.dense_packed) <= 1.0e-6f);

    std::vector<float> rho_ref(on.rho.size());
    std::vector<float> rho_cpu_ref(on.rho.size());
    std::vector<float> recovered_ref(on.recovered.size());
    std::vector<float> recovered_cpu_ref(on.recovered.size());
    for (int64_t qt = 0; qt < k_q_tokens; ++qt) {
        for (int64_t h = 0; h < k_heads; ++h) {
            const size_t packed_row = packed_index(0, h, qt);
            const size_t rho_row = static_cast<size_t>(h + k_heads * qt);
            rho_ref[rho_row] = std::exp(
                on.sparse_packed[packed_row + k_head_dim] -
                on.dense_packed[packed_row + k_head_dim]);
            rho_cpu_ref[rho_row] = std::exp(
                sparse_ref[packed_row + k_head_dim] - dense_ref[packed_row + k_head_dim]);
            for (int64_t d = 0; d < k_head_dim; ++d) {
                const size_t output_index = static_cast<size_t>(d + k_head_dim * (h + k_heads * qt));
                recovered_ref[output_index] = on.dense_packed[packed_row + static_cast<size_t>(d)];
                recovered_cpu_ref[output_index] = dense_ref[packed_row + static_cast<size_t>(d)];
                const float recovered_from_residual =
                    on.residual[output_index] + rho_ref[rho_row] *
                    on.sparse_packed[packed_row + static_cast<size_t>(d)];
                GGML_ASSERT(std::fabs(recovered_from_residual - recovered_ref[output_index]) <= 2.0e-5f);
            }
        }
    }
    GGML_ASSERT(max_abs_diff(on.rho, rho_ref) <= 2.0e-5f);
    GGML_ASSERT(max_abs_diff(on.rho, rho_cpu_ref) <= 3.0e-3f);
    GGML_ASSERT(max_abs_diff(on.recovered, recovered_ref) <= 2.0e-5f);
    GGML_ASSERT(max_abs_diff(on.recovered, recovered_cpu_ref) <= htp_o_tolerance);
    GGML_ASSERT(max_abs_diff(off.recovered, on.recovered) <= 2.0e-5f);

    std::printf(
        "exact-rho errors off[dense=%.9g sparse=%.9g] on[dense=%.9g sparse=%.9g] "
        "rho=%.9g rho_cpu=%.9g recovered=%.9g recovered_cpu=%.9g parity=%.9g\n",
        off_dense_error.o, off_sparse_error.o, on_dense_error.o, on_sparse_error.o,
        max_abs_diff(on.rho, rho_ref), max_abs_diff(on.rho, rho_cpu_ref),
        max_abs_diff(on.recovered, recovered_ref), max_abs_diff(on.recovered, recovered_cpu_ref),
        max_abs_diff(off.recovered, on.recovered));
}

void require_async_parity(const run_result & sync, const run_result & async) {
    GGML_ASSERT(max_abs_diff(sync.dense_packed, async.dense_packed) <= 1.0e-6f);
    GGML_ASSERT(max_abs_diff(sync.sparse_packed, async.sparse_packed) <= 2.0e-5f);
    GGML_ASSERT(max_abs_diff(sync.rho, async.rho) <= 2.0e-5f);
    GGML_ASSERT(max_abs_diff(sync.residual, async.residual) <= 2.0e-5f);
    GGML_ASSERT(max_abs_diff(sync.recovered, async.recovered) <= 2.0e-5f);
}

void require_task11_failure_modes(
        ggml_backend_t backend,
        graph_fixture & fixture,
        const run_result & sync_reference) {
    auto checkpoint = [](const char * name) {
        std::printf("Task 11 failure mode: %s ts_us=%lld\n", name, (long long) ggml_time_us());
        std::fflush(stdout);
    };
    checkpoint("preflight fallback begin");
    {
        env_guard reject_preflight("SD_HTP_SPARSE_OPENCL_TEST_REJECT_PREFLIGHT", "1");
        env_guard strict("SD_HTP_SPARSE_OPENCL_STRICT", "0");
        const run_result fallback = run_graph(backend, fixture, true, true);
        GGML_ASSERT(fallback.status == GGML_STATUS_SUCCESS);
        GGML_ASSERT(count_attention_trace(fallback.trace, "opencl", 1) == 0);
        GGML_ASSERT(count_attention_trace(fallback.trace, "htp", 1) == 1);
        GGML_ASSERT(count_attention_trace(fallback.trace, "htp", 0) == 1);
    }
    checkpoint("preflight fallback end; strict begin");
    {
        env_guard reject_preflight("SD_HTP_SPARSE_OPENCL_TEST_REJECT_PREFLIGHT", "1");
        env_guard strict("SD_HTP_SPARSE_OPENCL_STRICT", "1");
        const run_result strict_failure = run_graph(backend, fixture, true, true);
        GGML_ASSERT(strict_failure.status == GGML_STATUS_FAILED);
        GGML_ASSERT(count_attention_trace(strict_failure.trace, "opencl", 1) == 0);
        GGML_ASSERT(count_attention_trace(strict_failure.trace, "htp", 1) == 0);
    }
    checkpoint("strict end; enqueue failure begin");
    {
        env_guard fail_enqueue("SD_HTP_SPARSE_OPENCL_TEST_FAIL_ENQUEUE", "1");
        const run_result enqueue_failure = run_graph(backend, fixture, true, true);
        GGML_ASSERT(enqueue_failure.status == GGML_STATUS_FAILED);
        GGML_ASSERT(count_attention_trace(enqueue_failure.trace, "opencl", 1) == 1);
        GGML_ASSERT(count_attention_trace(enqueue_failure.trace, "htp", 1) == 0);
    }
    checkpoint("enqueue failure end; wait failure begin");
    {
        env_guard fail_wait("SD_HTP_SPARSE_OPENCL_TEST_FAIL_WAIT", "1");
        const run_result wait_failure = run_graph(backend, fixture, true, true);
        GGML_ASSERT(wait_failure.status == GGML_STATUS_FAILED);
        GGML_ASSERT(count_attention_trace(wait_failure.trace, "opencl", 1) == 1);
        GGML_ASSERT(count_attention_trace(wait_failure.trace, "htp", 1) == 0);
        GGML_ASSERT(count_attention_trace(wait_failure.trace, "htp", 0) == 1);
    }
    checkpoint("wait failure end; abort after enqueue begin");
    {
        env_guard abort_after_enqueue("SD_HTP_SPARSE_OPENCL_TEST_ABORT_AFTER_ENQUEUE", "1");
        const run_result aborted = run_graph(backend, fixture, true, true);
        GGML_ASSERT(aborted.status == GGML_STATUS_ABORTED);
        GGML_ASSERT(count_attention_trace(aborted.trace, "opencl", 1) == 1);
        GGML_ASSERT(count_attention_trace(aborted.trace, "htp", 1) == 0);
    }
    checkpoint("abort after enqueue end; stress begin");

    {
        env_guard stress_trace("SD_HTP_SPARSE_OPENCL_TRACE", "0");
        constexpr int stress_iterations = 100;
        std::printf("Task 11 abort/rebuild recovery stress: begin %d iterations\n", stress_iterations);
        std::fflush(stdout);
        int64_t checkpoint_begin_us = ggml_time_us();
        double checkpoint_abort_ms = 0.0;
        double checkpoint_recovery_ms = 0.0;
        for (int iteration = 0; iteration < stress_iterations; ++iteration) {
            const int64_t iteration_begin_us = ggml_time_us();
            graph_fixture rebuilt = make_graph(backend);
            {
                env_guard abort_after_enqueue(
                    "SD_HTP_SPARSE_OPENCL_TEST_ABORT_AFTER_ENQUEUE", "1");
                const run_result aborted = run_graph(backend, rebuilt, true, true);
                GGML_ASSERT(aborted.status == GGML_STATUS_ABORTED);
            }
            const int64_t abort_end_us = ggml_time_us();
            const bool verify_recovery =
                (iteration + 1) % 10 == 0 || iteration + 1 == stress_iterations;
            if (verify_recovery) {
                const run_result recovered = run_graph(backend, rebuilt, true, true);
                GGML_ASSERT(recovered.status == GGML_STATUS_SUCCESS);
                GGML_ASSERT(max_abs_diff(
                    recovered.recovered, sync_reference.recovered) <= 2.0e-5f);
            }
            ggml_gallocr_free(rebuilt.allocator);
            ggml_free(rebuilt.ctx);
            const int64_t iteration_end_us = ggml_time_us();
            checkpoint_abort_ms += (abort_end_us - iteration_begin_us) / 1000.0;
            checkpoint_recovery_ms +=
                verify_recovery ? (iteration_end_us - abort_end_us) / 1000.0 : 0.0;
            if ((iteration + 1) % 10 == 0 || iteration + 1 == stress_iterations) {
                std::printf(
                    "Task 11 abort/rebuild recovery stress: %d/%d "
                    "wall_ms=%.3f abort_ms=%.3f recovery_ms=%.3f\n",
                    iteration + 1,
                    stress_iterations,
                    (iteration_end_us - checkpoint_begin_us) / 1000.0,
                    checkpoint_abort_ms,
                    checkpoint_recovery_ms);
                std::fflush(stdout);
                checkpoint_begin_us = iteration_end_us;
                checkpoint_abort_ms = 0.0;
                checkpoint_recovery_ms = 0.0;
            }
        }
    }
}

} // namespace

int main() {
    env_guard initial_sidecar("SD_HTP_SPARSE_OPENCL", "1");
    env_guard sidecar_trace("SD_HTP_SPARSE_OPENCL_TRACE", "1");
    env_guard early_submit("SD_HTP_SPARSE_OPENCL_EARLY_SUBMIT", "1");
    env_guard htp_ops("SD_HTP_OPS", "1");
    env_guard htp_compute("SD_HTP_COMPUTE", "1");
    env_guard htp_attention("SD_HTP_ATTENTION", "1");

    ggml_backend_t backend = ggml_backend_htp_init();
    GGML_ASSERT(backend != nullptr);
    if (!ggml_backend_htp_runtime_available()) {
        std::fprintf(
            stderr, "HTP runtime unavailable: %s\n",
            ggml_backend_htp_runtime_error() != nullptr ?
                ggml_backend_htp_runtime_error() : "unknown error");
    }
    GGML_ASSERT(ggml_backend_htp_runtime_available());
    ggml_backend_htp_set_hybrid_enabled(true);
    ggml_backend_htp_set_n_threads(backend, 2);

    graph_fixture fixture = make_graph(backend);
    const std::vector<float> q = make_q();
    std::vector<ggml_fp16_t> k;
    std::vector<ggml_fp16_t> v;
    make_kv(k, v);

    const std::vector<float> dense_ref = reference_attention(q, k, v, false);
    const std::vector<float> sparse_ref = reference_attention(q, k, v, true);
    const run_result off = run_graph(backend, fixture, false, false);
    const run_result sync = run_graph(backend, fixture, true, false);
    const run_result async = run_graph(backend, fixture, true, true);
    GGML_ASSERT(off.status == GGML_STATUS_SUCCESS);
    GGML_ASSERT(sync.status == GGML_STATUS_SUCCESS);
    GGML_ASSERT(async.status == GGML_STATUS_SUCCESS);
    require_routes(off, sync);
    require_routes(off, async);
    require_numerics(off, sync, dense_ref, sparse_ref);
    require_numerics(off, async, dense_ref, sparse_ref);
    require_async_parity(sync, async);
    require_async_event_order(async.trace);
    require_task11_failure_modes(backend, fixture, sync);

    ggml_gallocr_free(fixture.allocator);
    ggml_free(fixture.ctx);
    ggml_backend_free(backend);
    ggml_backend_htp_set_hybrid_enabled(false);
    std::printf("test-backend-htp-opencl-exact-rho passed\n");
    return 0;
}
