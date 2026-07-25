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

extern "C" {
bool htp_ops_opencl_node_needs_wait(const ggml_tensor * dst);
int htp_ops_opencl_wait_node(const ggml_tensor * dst);
}

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
constexpr float k_live_alias_sentinel = 17.25f;
constexpr uint64_t k_persistent_cache_generation = 41;
constexpr uint64_t k_routed_residual_generation = 101;
constexpr uint64_t k_cross_graph_residual_generation_0 = 201;
constexpr uint64_t k_cross_graph_residual_generation_1 = 202;
constexpr uint64_t k_cross_graph_arena_generation_0 = 301;
constexpr uint64_t k_cross_graph_arena_generation_1 = 302;
constexpr uint64_t k_delayed_consumer_cache_generation = 203;
constexpr uint64_t k_delayed_consumer_arena_generation = 303;
constexpr uint64_t k_fault_rho_generation = 401;
constexpr uint64_t k_fault_residual_generation = 402;
constexpr uint64_t k_fault_wait_generation = 403;
constexpr uint64_t k_fault_recovery_generation = 404;
constexpr uint64_t k_abort_residual_queue_generation = 405;
constexpr uint64_t k_abort_residual_enqueue_generation = 406;
constexpr uint64_t k_consumer_failure_generation = 407;
constexpr uint64_t k_queue_fallback_generation = 408;
constexpr uint64_t k_queue_strict_generation = 409;
constexpr uint64_t k_residual_flush_generation = 410;
constexpr uint64_t k_residual_profile_generation = 411;
constexpr uint64_t k_queue_disabled_generation = 412;
constexpr uint64_t k_consumer_failure_arena_generation = 307;

class env_guard {
public:
    env_guard(const char * name, const char * value) : name_(name) {
        const char * previous = std::getenv(name);
        if (previous != nullptr) {
            had_previous_ = true;
            previous_ = previous;
        }
        if (value != nullptr) {
            GGML_ASSERT(setenv(name, value, 1) == 0);
        } else {
            GGML_ASSERT(unsetenv(name) == 0);
        }
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
    ggml_tensor * sparse_mask_clobber = nullptr;
    ggml_tensor * dense_packed = nullptr;
    ggml_tensor * sparse_packed = nullptr;
    ggml_tensor * rho = nullptr;
    ggml_tensor * residual = nullptr;
    ggml_tensor * recovered = nullptr;
    ggml_tensor * live_alias = nullptr;
    ggml_tensor * live_probe = nullptr;
    ggml_tensor * dense_clobber = nullptr;
    ggml_tensor * priority_sparse_mask = nullptr;
    ggml_tensor * priority_sparse_packed = nullptr;
    ggml_gallocr_t allocator = nullptr;
    ggml_backend_htp_async_cache_slot * rho_slot = nullptr;
    ggml_backend_htp_async_cache_slot * residual_slot = nullptr;
    ggml_backend_htp_async_cache_slot * sparse_slot = nullptr;
    bool residual_only_route = false;
};

struct run_result {
    ggml_status status = GGML_STATUS_FAILED;
    std::string trace;
    std::vector<float> dense_packed;
    std::vector<float> sparse_packed;
    std::vector<float> rho;
    std::vector<float> residual;
    std::vector<uint8_t> rho_cache_bytes;
    std::vector<uint8_t> residual_cache_bytes;
    std::vector<float> recovered;
    std::vector<float> live_probe;
};

struct persistent_cache_probe {
    ggml_backend_htp_async_cache_slot * slot = nullptr;
    ggml_backend_buffer_t backing = nullptr;
    uint64_t import_identity = 0;
    std::vector<float> expected;
};

persistent_cache_probe make_persistent_cache_probe(ggml_backend_t backend) {
    const int64_t ne[GGML_MAX_DIMS] = {
        k_head_dim, k_heads, k_q_tokens, k_batch,
    };
    char reason[512] = {};
    persistent_cache_probe probe;
    probe.slot = ggml_backend_htp_async_cache_slot_create(
        backend,
        "htp_block_sparse_residual_b0_self_s0",
        k_persistent_cache_generation,
        GGML_TYPE_F32,
        ne,
        reason,
        sizeof(reason));
    if (probe.slot == nullptr) {
        std::fprintf(stderr, "persistent cache slot creation failed: %s\n", reason);
    }
    GGML_ASSERT(probe.slot != nullptr);

    ggml_tensor * tensor = ggml_backend_htp_async_cache_slot_tensor(probe.slot);
    GGML_ASSERT(tensor != nullptr);
    GGML_ASSERT(tensor->buffer != nullptr);
    probe.backing = tensor->buffer;

    ggml_backend_htp_rpcmem_info info = {};
    GGML_ASSERT(ggml_backend_htp_async_cache_slot_rpcmem_info(probe.slot, &info));
    GGML_ASSERT(info.base != nullptr);
    GGML_ASSERT(info.fd >= 0);
    GGML_ASSERT(info.logical_size >= ggml_nbytes(tensor));
    GGML_ASSERT(info.allocation_size >= info.logical_size);

    probe.import_identity =
        ggml_backend_htp_async_cache_slot_opencl_import_identity(probe.slot);
    GGML_ASSERT(probe.import_identity != 0);
    probe.expected.resize(static_cast<size_t>(ggml_nelements(tensor)));
    for (size_t i = 0; i < probe.expected.size(); ++i) {
        probe.expected[i] = 0.25f + static_cast<float>(i % 97) * 0.03125f;
    }
    ggml_backend_tensor_set(
        tensor, probe.expected.data(), 0, probe.expected.size() * sizeof(float));
    return probe;
}

void require_persistent_cache_probe_unchanged(const persistent_cache_probe & probe) {
    GGML_ASSERT(probe.slot != nullptr);
    ggml_tensor * tensor = ggml_backend_htp_async_cache_slot_tensor(probe.slot);
    GGML_ASSERT(tensor != nullptr);
    GGML_ASSERT(tensor->buffer == probe.backing);
    GGML_ASSERT(
        ggml_backend_htp_async_cache_slot_opencl_import_identity(probe.slot) ==
        probe.import_identity);
    std::vector<float> actual(probe.expected.size());
    ggml_backend_tensor_get(tensor, actual.data(), 0, actual.size() * sizeof(float));
    GGML_ASSERT(actual == probe.expected);
}

std::vector<uint8_t> byte_snapshot(const std::vector<float> & values) {
    std::vector<uint8_t> bytes(values.size() * sizeof(float));
    if (!bytes.empty()) {
        std::memcpy(bytes.data(), values.data(), bytes.size());
    }
    return bytes;
}

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

void clobber_sparse_index(ggml_tensor * dst, int ith, int nth, void * userdata) {
    GGML_UNUSED(nth);
    GGML_UNUSED(userdata);
    if (ith != 0) {
        return;
    }
    const char * enabled = std::getenv("SD_HTP_SPARSE_OPENCL_TEST_CLOBBER_MASK");
    if (enabled == nullptr || enabled[0] == '\0' || std::strcmp(enabled, "0") == 0) {
        return;
    }
    std::fill_n(
        static_cast<int32_t *>(dst->data),
        static_cast<size_t>(ggml_nelements(dst)),
        -1);
}

void capture_live_alias(
        ggml_tensor * dst,
        const ggml_tensor * src,
        int ith,
        int nth,
        void * userdata) {
    GGML_UNUSED(nth);
    GGML_UNUSED(userdata);
    if (ith != 0) {
        return;
    }
    const char * delay_value = std::getenv("SD_HTP_SPARSE_OPENCL_TEST_LIVE_ALIAS_DELAY_US");
    if (delay_value != nullptr && delay_value[0] != '\0') {
        char * end = nullptr;
        const long delay_us = std::strtol(delay_value, &end, 10);
        if (end != delay_value && *end == '\0' && delay_us > 0 && delay_us <= 1000000) {
            usleep(static_cast<useconds_t>(delay_us));
        }
    }
    GGML_ASSERT(ggml_nbytes(dst) == ggml_nbytes(src));
    std::memcpy(dst->data, src->data, ggml_nbytes(dst));
}

void capture_deferred_cache_input(
        ggml_tensor * dst,
        const ggml_tensor * src,
        int ith,
        int nth,
        void * userdata) {
    GGML_UNUSED(nth);
    GGML_UNUSED(userdata);
    if (ith != 0) {
        return;
    }
    std::fprintf(
        stderr,
        "htp_opencl_event name=deferred_cache_consumer_begin ts_us=%lld "
        "consumer=%s\n",
        (long long) ggml_time_us(),
        dst->name[0] != '\0' ? dst->name : "<unnamed>");
    GGML_ASSERT(ggml_nbytes(dst) == ggml_nbytes(src));
    std::memcpy(dst->data, src->data, ggml_nbytes(dst));
}

void compute_exact_rho_tagged(
        ggml_tensor * dst, int ith, int nth, void * userdata) {
    GGML_UNUSED(userdata);
    const ggml_tensor * dense = dst->src[0];
    const ggml_tensor * sparse = dst->src[1];
    GGML_ASSERT(dense != nullptr && sparse != nullptr);
    const int64_t rows = dense->ne[1] * dense->ne[2] * dense->ne[3];
    const int64_t head_dim = dense->ne[0] - 1;
    for (int64_t row = ith; row < rows; row += nth) {
        static_cast<float *>(dst->data)[row] = std::exp(
            static_cast<const float *>(sparse->data)[row * dense->ne[0] + head_dim] -
            static_cast<const float *>(dense->data)[row * dense->ne[0] + head_dim]);
    }
}

void compute_exact_residual_tagged(
        ggml_tensor * dst, int ith, int nth, void * userdata) {
    GGML_UNUSED(userdata);
    const ggml_tensor * dense = dst->src[0];
    const ggml_tensor * sparse = dst->src[1];
    const ggml_tensor * rho = dst->src[2];
    GGML_ASSERT(dense != nullptr && sparse != nullptr && rho != nullptr);
    const int64_t head_dim = dense->ne[0] - 1;
    const int64_t rows = dense->ne[1] * dense->ne[2] * dense->ne[3];
    for (int64_t row = ith; row < rows; row += nth) {
        const float scale = static_cast<const float *>(rho->data)[row];
        for (int64_t d = 0; d < head_dim; ++d) {
            static_cast<float *>(dst->data)[row * head_dim + d] =
                static_cast<const float *>(dense->data)[row * dense->ne[0] + d] -
                static_cast<const float *>(sparse->data)[row * sparse->ne[0] + d] * scale;
        }
    }
}

graph_fixture make_graph(
        ggml_backend_t backend,
        uint64_t residual_generation = k_routed_residual_generation,
        bool add_priority_sparse_probe = false) {
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
    f.sparse_mask_clobber = ggml_custom_4d(
        f.ctx, GGML_TYPE_I32, 1, k_heads, 1, 1,
        nullptr, 0, clobber_sparse_index, 1, nullptr);
    ggml_set_name(f.q_storage, "exact_rho_q_storage");
    ggml_set_name(f.q, "exact_rho_q");
    ggml_set_name(f.k_storage, "exact_rho_k_storage");
    ggml_set_name(f.v_storage, "exact_rho_v_storage");
    ggml_set_name(f.k, "exact_rho_k");
    ggml_set_name(f.v, "exact_rho_v");
    ggml_set_name(f.dense_mask, "exact_rho_dense_mask");
    ggml_set_name(f.sparse_mask, "exact_rho_sparse_index");
    ggml_set_name(f.sparse_mask_clobber, "exact_rho_sparse_index_clobber");
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

    const bool routed_residual =
        std::getenv("SD_HTP_SPARSE_OPENCL_RESIDUAL") != nullptr;
    f.residual_only_route = routed_residual;
    if (routed_residual) {
        ggml_tensor * rho_args[] = { f.dense_packed, f.sparse_packed };
        f.rho = ggml_custom_4d(
            f.ctx, GGML_TYPE_F32, 1, k_heads, k_q_tokens, k_batch,
            rho_args, 2, compute_exact_rho_tagged, GGML_N_TASKS_MAX, nullptr);
        ggml_set_name(f.rho, "htp_block_sparse_residual_rho_b1_self_s0");
        ggml_tensor * residual_args[] = { f.dense_packed, f.sparse_packed, f.rho };
        f.residual = ggml_custom_4d(
            f.ctx, GGML_TYPE_F32, k_head_dim, k_heads, k_q_tokens, k_batch,
            residual_args, 3, compute_exact_residual_tagged, GGML_N_TASKS_MAX, nullptr);
        ggml_set_name(f.residual, "htp_block_sparse_residual_b1_self_s0");
    } else {
        ggml_tensor * rho_sub = ggml_sub(f.ctx, sparse_lse, dense_lse);
        ggml_set_name(rho_sub, "htp_cpu_residual_rho_sub");
        f.rho = ggml_exp(f.ctx, rho_sub);
        ggml_set_name(f.rho, "htp_block_sparse_residual_rho_test");
        ggml_tensor * scaled_sparse = ggml_mul(f.ctx, sparse_o, f.rho);
        ggml_set_name(scaled_sparse, "htp_cpu_residual_scale_mul");
        f.residual = ggml_sub(f.ctx, dense_o, scaled_sparse);
        ggml_set_name(f.residual, "htp_block_sparse_residual_test");
    }
    f.recovered = routed_residual
        ? ggml_cont_4d(f.ctx, dense_o, k_head_dim, k_heads, k_q_tokens, k_batch)
        : ggml_add(f.ctx, ggml_mul(f.ctx, sparse_o, f.rho), f.residual);
    ggml_set_name(f.recovered, "exact_rho_recovered");
    f.live_alias = ggml_new_tensor_4d(
        f.ctx, GGML_TYPE_F32, k_packed_dim, k_heads, k_q_tokens, k_batch);
    ggml_set_name(f.live_alias, "exact_rho_live_alias");
    ggml_set_input(f.live_alias);
    f.live_probe = ggml_map_custom1(f.ctx, f.live_alias, capture_live_alias, 1, nullptr);
    ggml_set_name(f.live_probe, "exact_rho_live_probe");
    ggml_set_output(f.live_probe);
    if (!routed_residual) {
        ggml_set_output(f.dense_packed);
    }
    ggml_set_output(f.sparse_packed);
    ggml_set_output(f.rho);
    ggml_set_output(f.residual);
    ggml_set_output(f.recovered);

    f.graph = ggml_new_graph(f.ctx);
    GGML_ASSERT(f.graph != nullptr);
    // Match the production anchor graph: the dense model result is expanded
    // first, while the exact-rho residual is a cache-only side branch added
    // afterward. Early OpenCL dispatch must not depend on cache-root order.
    ggml_build_forward_expand(f.graph, f.sparse_mask_clobber);
    ggml_build_forward_expand(f.graph, dense_o);
    ggml_build_forward_expand(f.graph, f.live_probe);
    ggml_build_forward_expand(f.graph, f.recovered);
    if (routed_residual) {
        ggml_build_forward_expand(f.graph, f.residual);
        if (add_priority_sparse_probe) {
            ggml_tensor * priority_mask_args[] = { f.dense_packed };
            f.priority_sparse_mask = ggml_custom_4d(
                f.ctx, GGML_TYPE_I32, 1, k_heads, 1, 1,
                priority_mask_args, 1, fill_sparse_index, 1, nullptr);
            ggml_set_name(
                f.priority_sparse_mask, "exact_rho_priority_sparse_index");
            f.priority_sparse_packed = ggml_flash_attn_ext_with_lse(
                f.ctx, f.q, f.k, f.v, f.priority_sparse_mask,
                k_scale, 0.0f, 0.0f);
            ggml_flash_attn_ext_set_prec(
                f.priority_sparse_packed, GGML_PREC_F32);
            ggml_flash_attn_ext_set_block_sparse(
                f.priority_sparse_packed, k_block_size, k_block_size,
                GGML_FLASH_ATTN_EXT_MASK_BLOCK_INDEX);
            ggml_set_name(
                f.priority_sparse_packed, "exact_rho_priority_sparse_packed");
            ggml_set_output(f.priority_sparse_packed);
            ggml_build_forward_expand(f.graph, f.priority_sparse_packed);
        }
        // This real HTP attention is deliberately placed after the residual
        // markers. Without the dense lifetime root, gallocr can assign it the
        // first dense output's released arena span while the GPU still reads it.
        f.dense_clobber = ggml_flash_attn_ext_with_lse(
            f.ctx, f.q, f.k, f.v, f.dense_mask, k_scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(f.dense_clobber, GGML_PREC_F32);
        ggml_set_name(f.dense_clobber, "exact_rho_post_residual_htp_dense");
        ggml_set_output(f.dense_clobber);
        ggml_build_forward_expand(f.graph, f.dense_clobber);
        GGML_ASSERT(
            ggml_backend_htp_opencl_retain_deferred_residual_inputs(f.graph) == 1);
        GGML_ASSERT((f.dense_packed->flags & GGML_TENSOR_FLAG_OUTPUT) != 0);
    }
    f.allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    GGML_ASSERT(f.allocator != nullptr);
    GGML_ASSERT(ggml_gallocr_alloc_graph(f.allocator, f.graph));
    if (routed_residual) {
        GGML_ASSERT(f.dense_clobber->data != f.dense_packed->data);
    }
    if (routed_residual) {
        const int64_t rho_ne[GGML_MAX_DIMS] = { 1, k_heads, k_q_tokens, k_batch };
        const int64_t residual_ne[GGML_MAX_DIMS] = {
            k_head_dim, k_heads, k_q_tokens, k_batch,
        };
        const int64_t sparse_ne[GGML_MAX_DIMS] = {
            k_packed_dim, k_heads, k_q_tokens, k_batch,
        };
        char reason[512] = {};
        f.rho_slot = ggml_backend_htp_async_cache_slot_create(
            backend, f.rho->name, residual_generation,
            GGML_TYPE_F32, rho_ne, reason, sizeof(reason));
        GGML_ASSERT(f.rho_slot != nullptr);
        f.residual_slot = ggml_backend_htp_async_cache_slot_create(
            backend, f.residual->name, residual_generation,
            GGML_TYPE_F32, residual_ne, reason, sizeof(reason));
        GGML_ASSERT(f.residual_slot != nullptr);
        f.sparse_slot = ggml_backend_htp_async_cache_slot_create(
            backend, "test_residual_only_sparse_destination",
            residual_generation,
            GGML_TYPE_F32, sparse_ne, reason, sizeof(reason));
        GGML_ASSERT(f.sparse_slot != nullptr);
        ggml_tensor * rho_slot_tensor =
            ggml_backend_htp_async_cache_slot_tensor(f.rho_slot);
        ggml_tensor * residual_slot_tensor =
            ggml_backend_htp_async_cache_slot_tensor(f.residual_slot);
        ggml_tensor * sparse_slot_tensor =
            ggml_backend_htp_async_cache_slot_tensor(f.sparse_slot);
        f.rho->buffer = rho_slot_tensor->buffer;
        f.rho->data = rho_slot_tensor->data;
        f.residual->buffer = residual_slot_tensor->buffer;
        f.residual->data = residual_slot_tensor->data;
        f.sparse_packed->buffer = sparse_slot_tensor->buffer;
        f.sparse_packed->data = sparse_slot_tensor->data;
    }
    // Model the production allocator reuse that makes the future sparse output
    // share storage with a still-live main-graph intermediate.
    if (!routed_residual) {
        f.live_alias->buffer = f.sparse_packed->buffer;
        f.live_alias->data = f.sparse_packed->data;
    }
    f.sparse_mask_clobber->buffer = f.sparse_mask->buffer;
    f.sparse_mask_clobber->data = f.sparse_mask->data;
    return f;
}

void free_graph_fixture(graph_fixture & fixture) {
    ggml_backend_htp_async_cache_slot_free(fixture.sparse_slot);
    ggml_backend_htp_async_cache_slot_free(fixture.residual_slot);
    ggml_backend_htp_async_cache_slot_free(fixture.rho_slot);
    fixture.residual_slot = nullptr;
    fixture.sparse_slot = nullptr;
    fixture.rho_slot = nullptr;
    ggml_gallocr_free(fixture.allocator);
    fixture.allocator = nullptr;
    ggml_free(fixture.ctx);
    fixture.ctx = nullptr;
}

void initialize_graph_inputs(graph_fixture & fixture) {
    auto require_buffer = [](const ggml_tensor * tensor) {
        if (tensor == nullptr || tensor->buffer == nullptr) {
            std::fprintf(stderr, "fixture tensor buffer missing: %s\n",
                         tensor != nullptr && tensor->name[0] != '\0' ?
                             tensor->name : "<unnamed>");
        }
        GGML_ASSERT(tensor != nullptr && tensor->buffer != nullptr);
    };
    require_buffer(fixture.q_storage);
    require_buffer(fixture.k_storage);
    require_buffer(fixture.v_storage);
    require_buffer(fixture.dense_mask);
    require_buffer(fixture.live_alias);
    if (fixture.residual_only_route) {
        require_buffer(fixture.sparse_packed);
    }
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
    const std::vector<float> live_alias(
        static_cast<size_t>(ggml_nelements(fixture.live_alias)), k_live_alias_sentinel);
    ggml_backend_tensor_set(
        fixture.live_alias, live_alias.data(), 0, live_alias.size() * sizeof(float));
    if (fixture.residual_only_route) {
        const std::vector<float> sparse_sentinel(
            static_cast<size_t>(ggml_nelements(fixture.sparse_packed)),
            k_live_alias_sentinel);
        ggml_backend_tensor_set(
            fixture.sparse_packed,
            sparse_sentinel.data(),
            0,
            sparse_sentinel.size() * sizeof(float));
    }
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
        result.rho_cache_bytes = byte_snapshot(result.rho);
        result.residual_cache_bytes = byte_snapshot(result.residual);
        result.recovered = read_f32(f.recovered);
        result.live_probe = read_f32(f.live_probe);
    }
    return result;
}

int count_attention_trace(const std::string & trace, const char * route, int block_sparse);

int64_t require_event_timestamp(const std::string & trace, const char * event) {
    const size_t found = trace.find(event);
    if (found == std::string::npos) {
        std::fprintf(stderr, "[FAIL] missing async event '%s'\n--- trace ---\n%s",
                     event, trace.c_str());
        GGML_ASSERT(false && "OpenCL sparse/HTP dense event missing");
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
    std::printf("[ASYNC_ORDER] %s\n", line.c_str());
    return std::strtoll(timestamp.c_str(), nullptr, 10);
}

int64_t require_event_timestamp_for_producer(
        const std::string & trace,
        const char * event,
        const char * producer) {
    const std::string event_text = std::string("name=") + event;
    const std::string producer_text = std::string("producer=") + producer;
    size_t line_begin = 0;
    while (line_begin < trace.size()) {
        const size_t line_end = trace.find('\n', line_begin);
        const std::string line = trace.substr(
            line_begin,
            line_end == std::string::npos ? std::string::npos : line_end - line_begin);
        if (line.find(event_text) != std::string::npos &&
            line.find(producer_text) != std::string::npos) {
            const std::string timestamp_prefix = "ts_us=";
            const size_t timestamp_begin = line.find(timestamp_prefix);
            GGML_ASSERT(timestamp_begin != std::string::npos);
            const size_t value_begin = timestamp_begin + timestamp_prefix.size();
            const size_t value_end = line.find_first_not_of("0123456789", value_begin);
            std::printf("[ASYNC_ORDER] %s\n", line.c_str());
            return std::strtoll(
                line.substr(value_begin, value_end - value_begin).c_str(), nullptr, 10);
        }
        if (line_end == std::string::npos) {
            break;
        }
        line_begin = line_end + 1;
    }
    std::fprintf(
        stderr,
        "[FAIL] missing async event '%s' for producer '%s'\n--- trace ---\n%s",
        event, producer, trace.c_str());
    GGML_ASSERT(false && "OpenCL/HTP producer event missing");
    return 0;
}

uint64_t require_trace_u64_field(
        const std::string & trace,
        const char * line_marker,
        const char * field) {
    const size_t marker = trace.find(line_marker);
    GGML_ASSERT(marker != std::string::npos);
    const std::string prefix = std::string(field) + "=";
    const size_t begin = trace.find(prefix, marker);
    GGML_ASSERT(begin != std::string::npos);
    const size_t value_begin = begin + prefix.size();
    const size_t value_end = trace.find_first_not_of("0123456789", value_begin);
    GGML_ASSERT(value_end != value_begin);
    return std::strtoull(
        trace.substr(value_begin, value_end - value_begin).c_str(), nullptr, 10);
}

void require_async_event_order(const std::string & trace) {
    GGML_ASSERT(trace.find("htp_opencl_submit_timing") != std::string::npos);
    GGML_ASSERT(trace.find("cached=1") != std::string::npos);
    const int64_t early = require_event_timestamp(trace, "gpu_sparse_early_enqueue");
    const int64_t job = require_event_timestamp(trace, "gpu_sparse_submit_job");
    const int64_t dense_begin = require_event_timestamp(trace, "htp_dense_begin");
    const int64_t dense_end = require_event_timestamp(trace, "htp_dense_end");
    const int64_t enqueue = require_event_timestamp(trace, "gpu_sparse_enqueue");
    const int64_t flush = require_event_timestamp(trace, "gpu_sparse_flush");
    const int64_t wait_begin = require_event_timestamp(trace, "gpu_sparse_wait_begin");
    const int64_t wait_end = require_event_timestamp(trace, "gpu_sparse_wait_end");
    const int64_t static_wait_begin =
        require_event_timestamp(trace, "gpu_sparse_static_wait_begin");
    const int64_t static_wait_end =
        require_event_timestamp(trace, "gpu_sparse_static_wait_end");
    const int64_t consumer = require_event_timestamp(trace, "rho_consumer_begin");
    GGML_ASSERT(early <= job);
    GGML_ASSERT(job < dense_begin);
    GGML_ASSERT(dense_begin < dense_end);
    GGML_ASSERT(enqueue <= flush);
    const bool waited_for_publication =
        trace.find("gpu_sparse_publication_wait_begin") != std::string::npos;
    if (waited_for_publication) {
        const int64_t publication_wait_begin =
            require_event_timestamp(trace, "gpu_sparse_publication_wait_begin");
        const int64_t publication_wait_end =
            require_event_timestamp(trace, "gpu_sparse_publication_wait_end");
        GGML_ASSERT(publication_wait_begin <= enqueue);
        GGML_ASSERT(flush <= publication_wait_end);
        GGML_ASSERT(publication_wait_end <= wait_begin);
    }
    GGML_ASSERT(flush <= wait_begin);
    GGML_ASSERT(static_wait_begin <= wait_begin);
    GGML_ASSERT(wait_begin <= wait_end);
    GGML_ASSERT(wait_end <= static_wait_end);
    GGML_ASSERT(wait_end <= consumer);
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
    GGML_ASSERT(sync.rho_cache_bytes == byte_snapshot(sync.rho));
    GGML_ASSERT(sync.residual_cache_bytes == byte_snapshot(sync.residual));
    GGML_ASSERT(async.rho_cache_bytes == byte_snapshot(async.rho));
    GGML_ASSERT(async.residual_cache_bytes == byte_snapshot(async.residual));
}

void require_early_output_lifetime(ggml_backend_t backend, graph_fixture & fixture) {
    auto require_sentinel = [](const run_result & result) {
        GGML_ASSERT(result.status == GGML_STATUS_SUCCESS);
        for (float value : result.live_probe) {
            GGML_ASSERT(value == k_live_alias_sentinel);
        }
    };
    {
        env_guard delayed_probe("SD_HTP_SPARSE_OPENCL_TEST_LIVE_ALIAS_DELAY_US", "50000");
        env_guard inline_submission("SD_HTP_SPARSE_OPENCL_SUBMIT_THREAD", "0");
        require_sentinel(run_graph(backend, fixture, true, true));
    }
    {
        env_guard delayed_probe("SD_HTP_SPARSE_OPENCL_TEST_LIVE_ALIAS_DELAY_US", "100000");
        env_guard delayed_submission("SD_HTP_SPARSE_OPENCL_TEST_SUBMIT_DELAY_US", "20000");
        require_sentinel(run_graph(backend, fixture, true, true));
    }
}

void require_early_mask_lifetime(
        ggml_backend_t backend,
        graph_fixture & fixture,
        const run_result & sync_reference) {
    env_guard clobber_mask("SD_HTP_SPARSE_OPENCL_TEST_CLOBBER_MASK", "1");
    env_guard delayed_submission("SD_HTP_SPARSE_OPENCL_TEST_SUBMIT_DELAY_US", "50000");
    const run_result result = run_graph(backend, fixture, true, true);
    GGML_ASSERT(result.status == GGML_STATUS_SUCCESS);
    require_async_parity(sync_reference, result);
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
    checkpoint("strict end; submit thread fallback begin");
    {
        env_guard fail_start("SD_HTP_SPARSE_OPENCL_TEST_FAIL_SUBMIT_THREAD_START", "1");
        const run_result fallback = run_graph(backend, fixture, true, true);
        GGML_ASSERT(fallback.status == GGML_STATUS_SUCCESS);
        GGML_ASSERT(fallback.trace.find("submission thread start failed") != std::string::npos);
        GGML_ASSERT(fallback.trace.find("falling back to inline submission") != std::string::npos);
        GGML_ASSERT(fallback.trace.find("gpu_sparse_submit_job") == std::string::npos);
        GGML_ASSERT(count_attention_trace(fallback.trace, "opencl", 1) == 1);
        GGML_ASSERT(count_attention_trace(fallback.trace, "htp", 1) == 0);
        require_async_parity(sync_reference, fallback);
    }
    checkpoint("submit thread fallback end; enqueue failure begin");
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
        env_guard abort_after_queue("SD_HTP_SPARSE_OPENCL_TEST_ABORT_AFTER_QUEUE", "1");
        env_guard delayed_submission("SD_HTP_SPARSE_OPENCL_TEST_SUBMIT_DELAY_US", "50000");
        const run_result aborted = run_graph(backend, fixture, true, true);
        GGML_ASSERT(aborted.status == GGML_STATUS_ABORTED);
        GGML_ASSERT(aborted.trace.find("gpu_sparse_submit_job") != std::string::npos);
        GGML_ASSERT(aborted.trace.find("gpu_sparse_enqueue") == std::string::npos);
    }
    checkpoint("abort after queue end; abort after enqueue begin");
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
            free_graph_fixture(rebuilt);
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

void require_task12_residual_failure_modes(
        ggml_backend_t backend,
        const run_result & sync_reference) {
    const size_t baseline_imports =
        ggml_backend_htp_opencl_live_cache_import_count(backend);
    const size_t baseline_events =
        ggml_backend_htp_opencl_live_cache_event_count(backend);
    const int64_t ne[GGML_MAX_DIMS] = { 4, 2, 1, 1 };
    char reason[512] = {};
    {
        env_guard fail_cache_alloc(
            "SD_HTP_SPARSE_OPENCL_TEST_FAIL_CACHE_ALLOC", "1");
        ggml_backend_htp_async_cache_slot * slot =
            ggml_backend_htp_async_cache_slot_create(
                backend, "task12_cache_alloc", 1, GGML_TYPE_F32,
                ne, reason, sizeof(reason));
        GGML_ASSERT(slot == nullptr);
        GGML_ASSERT(std::strstr(reason, "injected cache allocation failure") != nullptr);
    }
    {
        env_guard fail_cache_import(
            "SD_HTP_SPARSE_OPENCL_TEST_FAIL_CACHE_IMPORT", "1");
        ggml_backend_htp_async_cache_slot * slot =
            ggml_backend_htp_async_cache_slot_create(
                backend, "task12_cache_import", 1, GGML_TYPE_F32,
                ne, reason, sizeof(reason));
        GGML_ASSERT(slot == nullptr);
        GGML_ASSERT(std::strstr(reason, "injected cache import failure") != nullptr);
    }

    env_guard residual_route("SD_HTP_SPARSE_OPENCL_RESIDUAL", "1");
    env_guard residual_queue("SD_HTP_SPARSE_OPENCL_RESIDUAL_QUEUE", "1");
    auto require_failure = [&](const char * env_name,
                               uint64_t generation,
                               const char * failure_text) {
        env_guard injected(env_name, "1");
        graph_fixture fixture = make_graph(backend, generation);
        const run_result result = run_graph(backend, fixture, true, true);
        if (result.status != GGML_STATUS_FAILED) {
            std::fprintf(stderr, "expected %s failure\n--- trace ---\n%s",
                         env_name, result.trace.c_str());
        }
        GGML_ASSERT(result.status == GGML_STATUS_FAILED);
        GGML_ASSERT(result.trace.find(failure_text) != std::string::npos);
        GGML_ASSERT(ggml_backend_htp_opencl_deferred_arena_count(backend) == 0);
        free_graph_fixture(fixture);
    };
    require_failure(
        "SD_HTP_SPARSE_OPENCL_TEST_FAIL_RHO_ENQUEUE",
        k_fault_rho_generation,
        "injected rho enqueue failure");
    require_failure(
        "SD_HTP_SPARSE_OPENCL_TEST_FAIL_RESIDUAL_ENQUEUE",
        k_fault_residual_generation,
        "injected residual enqueue failure");
    require_failure(
        "SD_HTP_SPARSE_OPENCL_TEST_FAIL_RESIDUAL_WAIT",
        k_fault_wait_generation,
        "injected residual event wait failure");
    require_failure(
        "SD_HTP_SPARSE_OPENCL_TEST_FAIL_RESIDUAL_QUEUE_FLUSH",
        k_residual_flush_generation,
        "injected residual queue flush failure");

    graph_fixture recovery = make_graph(backend, k_fault_recovery_generation);
    const run_result recovered = run_graph(backend, recovery, true, true);
    GGML_ASSERT(recovered.status == GGML_STATUS_SUCCESS);
    GGML_ASSERT(max_abs_diff(sync_reference.dense_packed,
                             recovered.dense_packed) <= 1.0e-6f);
    GGML_ASSERT(max_abs_diff(sync_reference.rho, recovered.rho) <= 2.0e-5f);
    GGML_ASSERT(max_abs_diff(sync_reference.residual,
                             recovered.residual) <= 2.0e-5f);
    GGML_ASSERT(ggml_backend_htp_opencl_deferred_arena_count(backend) == 0);
    free_graph_fixture(recovery);
    GGML_ASSERT(
        ggml_backend_htp_opencl_live_cache_import_count(backend) ==
        baseline_imports);
    GGML_ASSERT(
        ggml_backend_htp_opencl_live_cache_event_count(backend) ==
        baseline_events);
    GGML_ASSERT(
        ggml_backend_htp_opencl_cache_input_binding_count(backend) == 0);
}

void require_residual_queue_creation_modes(
        ggml_backend_t backend) {
    env_guard residual_route("SD_HTP_SPARSE_OPENCL_RESIDUAL", "1");
    env_guard residual_queue("SD_HTP_SPARSE_OPENCL_RESIDUAL_QUEUE", nullptr);
    env_guard fail_queue_create(
        "SD_HTP_SPARSE_OPENCL_TEST_FAIL_RESIDUAL_QUEUE_CREATE", "1");
    {
        env_guard non_strict("SD_HTP_SPARSE_OPENCL_STRICT", "0");
        graph_fixture fixture = make_graph(
            backend, k_queue_fallback_generation);
        const run_result fallback = run_graph(backend, fixture, true, true);
        GGML_ASSERT(fallback.status == GGML_STATUS_SUCCESS);
        GGML_ASSERT(fallback.trace.find(
            "residual queue unavailable; using sparse queue") !=
            std::string::npos);
        GGML_ASSERT(fallback.trace.find(
            "gpu_rho_residual_enqueue queue=sparse") != std::string::npos);
        free_graph_fixture(fixture);
    }
    {
        env_guard strict("SD_HTP_SPARSE_OPENCL_STRICT", "1");
        graph_fixture fixture = make_graph(
            backend, k_queue_strict_generation);
        const run_result rejected = run_graph(backend, fixture, true, true);
        GGML_ASSERT(rejected.status == GGML_STATUS_FAILED);
        GGML_ASSERT(rejected.trace.find(
            "injected residual queue creation failure") != std::string::npos);
        free_graph_fixture(fixture);
    }
    {
        env_guard residual_queue_disabled(
            "SD_HTP_SPARSE_OPENCL_RESIDUAL_QUEUE", "0");
        env_guard strict("SD_HTP_SPARSE_OPENCL_STRICT", "1");
        graph_fixture fixture = make_graph(
            backend, k_queue_disabled_generation);
        const run_result disabled = run_graph(backend, fixture, true, true);
        GGML_ASSERT(disabled.status == GGML_STATUS_SUCCESS);
        GGML_ASSERT(disabled.trace.find(
            "gpu_rho_residual_enqueue queue=sparse") != std::string::npos);
        GGML_ASSERT(disabled.trace.find(
            "injected residual queue creation failure") == std::string::npos);
        free_graph_fixture(fixture);
    }
}

void require_task12_abort_and_consumer_failure(
        ggml_backend_t backend) {
    env_guard residual_route("SD_HTP_SPARSE_OPENCL_RESIDUAL", "1");
    env_guard residual_queue("SD_HTP_SPARSE_OPENCL_RESIDUAL_QUEUE", "1");
    {
        env_guard abort_after_queue(
            "SD_HTP_SPARSE_OPENCL_TEST_ABORT_AFTER_RESIDUAL_QUEUE", "1");
        env_guard delayed_submission(
            "SD_HTP_SPARSE_OPENCL_TEST_SUBMIT_DELAY_US", "100000");
        graph_fixture fixture = make_graph(
            backend, k_abort_residual_queue_generation);
        const run_result aborted = run_graph(backend, fixture, true, true);
        GGML_ASSERT(aborted.status == GGML_STATUS_ABORTED);
        GGML_ASSERT(aborted.trace.find("gpu_residual_submit_job") != std::string::npos);
        GGML_ASSERT(aborted.trace.find("gpu_rho_residual_enqueue") == std::string::npos);
        GGML_ASSERT(ggml_backend_htp_opencl_deferred_arena_count(backend) == 0);
        free_graph_fixture(fixture);
    }
    {
        env_guard abort_after_enqueue(
            "SD_HTP_SPARSE_OPENCL_TEST_ABORT_AFTER_RESIDUAL_ENQUEUE", "1");
        env_guard delayed_probe(
            "SD_HTP_SPARSE_OPENCL_TEST_LIVE_ALIAS_DELAY_US", "100000");
        graph_fixture fixture = make_graph(
            backend, k_abort_residual_enqueue_generation);
        const run_result aborted = run_graph(backend, fixture, true, true);
        GGML_ASSERT(aborted.status == GGML_STATUS_ABORTED);
        GGML_ASSERT(aborted.trace.find("gpu_rho_residual_enqueue") != std::string::npos);
        GGML_ASSERT(ggml_backend_htp_opencl_deferred_arena_count(backend) == 0);
        free_graph_fixture(fixture);
    }
    {
        env_guard delayed_submission(
            "SD_HTP_SPARSE_OPENCL_TEST_SUBMIT_DELAY_US", "500000");
        graph_fixture producer = make_graph(
            backend, k_consumer_failure_generation);
        initialize_graph_inputs(producer);
        ggml_backend_htp_opencl_set_compute_arena_generation(
            backend, k_consumer_failure_arena_generation);
        ggml_status producer_status = GGML_STATUS_FAILED;
        capture_graph_compute(backend, producer.graph, &producer_status);
        GGML_ASSERT(producer_status == GGML_STATUS_SUCCESS);

        ggml_tensor * cache_input = ggml_dup_tensor(producer.ctx, producer.rho);
        ggml_set_name(cache_input, "failed_deferred_rho_cache_input");
        ggml_set_input(cache_input);
        cache_input->buffer = producer.rho->buffer;
        cache_input->data = producer.rho->data;
        cache_input->extra = producer.rho->extra;
        ggml_tensor * cache_copy = ggml_map_custom1(
            producer.ctx, cache_input, capture_deferred_cache_input, 1, nullptr);
        ggml_set_name(cache_copy, "failed_deferred_rho_cache_copy");
        ggml_set_output(cache_copy);
        ggml_cgraph * consumer_graph = ggml_new_graph(producer.ctx);
        ggml_build_forward_expand(consumer_graph, cache_copy);
        ggml_gallocr_t consumer_allocator =
            ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        GGML_ASSERT(consumer_allocator != nullptr);
        GGML_ASSERT(ggml_gallocr_alloc_graph(consumer_allocator, consumer_graph));

        char reason[512] = {};
        GGML_ASSERT(ggml_backend_htp_opencl_bind_cache_input(
            backend, cache_input, producer.rho->name,
            k_consumer_failure_generation, reason, sizeof(reason)));
        ggml_backend_htp_opencl_test_fail_next_cache_wait(backend);
        ggml_status consumer_status = GGML_STATUS_SUCCESS;
        const std::string consumer_trace = capture_graph_compute(
            backend, consumer_graph, &consumer_status);
        GGML_ASSERT(consumer_status == GGML_STATUS_FAILED);
        GGML_ASSERT(consumer_trace.find(
            "injected residual event wait failure") != std::string::npos);
        GGML_ASSERT(
            ggml_backend_htp_opencl_cache_input_binding_count(backend) == 0);

        bool released = false;
        GGML_ASSERT(ggml_backend_htp_opencl_release_arena_generation(
            backend, k_consumer_failure_arena_generation, true, &released,
            reason, sizeof(reason)));
        GGML_ASSERT(released);
        ggml_gallocr_free(consumer_allocator);
        free_graph_fixture(producer);
    }
}

void require_strict_dense_only_graph_allowed(ggml_backend_t backend) {
    const ggml_init_params params = {
        /* .mem_size   = */ 256 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    GGML_ASSERT(ctx != nullptr);
    ggml_tensor * lhs = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 16);
    ggml_tensor * rhs = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 16);
    ggml_set_input(lhs);
    ggml_set_input(rhs);
    ggml_set_name(lhs, "strict_dense_only_lhs");
    ggml_set_name(rhs, "strict_dense_only_rhs");
    ggml_tensor * sum = ggml_add(ctx, lhs, rhs);
    ggml_set_name(sum, "strict_dense_only_sum");
    ggml_set_output(sum);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    GGML_ASSERT(graph != nullptr);
    ggml_build_forward_expand(graph, sum);
    ggml_gallocr_t allocator = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(backend));
    GGML_ASSERT(allocator != nullptr);
    GGML_ASSERT(ggml_gallocr_alloc_graph(allocator, graph));
    const float lhs_data[16] = {};
    const float rhs_data[16] = {};
    ggml_backend_tensor_set(lhs, lhs_data, 0, sizeof(lhs_data));
    ggml_backend_tensor_set(rhs, rhs_data, 0, sizeof(rhs_data));

    env_guard sidecar("SD_HTP_SPARSE_OPENCL", "1");
    env_guard async_sidecar("SD_HTP_SPARSE_OPENCL_ASYNC", "1");
    env_guard strict("SD_HTP_SPARSE_OPENCL_STRICT", "1");
    ggml_status status = GGML_STATUS_FAILED;
    const std::string trace = capture_graph_compute(backend, graph, &status);
    GGML_ASSERT(status == GGML_STATUS_SUCCESS);
    GGML_ASSERT(trace.find(
        "graph has no eligible sparse with-LSE attention nodes") !=
        std::string::npos);

    ggml_gallocr_free(allocator);
    ggml_free(ctx);
}

} // namespace

int main() {
    env_guard initial_sidecar("SD_HTP_SPARSE_OPENCL", "1");
    env_guard sidecar_trace("SD_HTP_SPARSE_OPENCL_TRACE", "1");
    env_guard sidecar_profile("SD_HTP_SPARSE_OPENCL_PROFILE", "1");
    env_guard early_submit("SD_HTP_SPARSE_OPENCL_EARLY_SUBMIT", "1");
    env_guard submit_thread("SD_HTP_SPARSE_OPENCL_SUBMIT_THREAD", "1");
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
    ggml_backend_htp_set_n_threads(backend, 4);
    require_residual_queue_creation_modes(backend);
    require_strict_dense_only_graph_allowed(backend);

    persistent_cache_probe persistent_probe = make_persistent_cache_probe(backend);

    graph_fixture fixture = make_graph(backend);
    const std::vector<float> q = make_q();
    std::vector<ggml_fp16_t> k;
    std::vector<ggml_fp16_t> v;
    make_kv(k, v);

    const std::vector<float> dense_ref = reference_attention(q, k, v, false);
    const std::vector<float> sparse_ref = reference_attention(q, k, v, true);
    const run_result off = run_graph(backend, fixture, false, false);
    require_persistent_cache_probe_unchanged(persistent_probe);
    const run_result sync = run_graph(backend, fixture, true, false);
    require_persistent_cache_probe_unchanged(persistent_probe);
    run_result inline_async;
    {
        env_guard inline_submission("SD_HTP_SPARSE_OPENCL_SUBMIT_THREAD", "0");
        inline_async = run_graph(backend, fixture, true, true);
    }
    run_result async;
    {
        env_guard delayed_submission("SD_HTP_SPARSE_OPENCL_TEST_SUBMIT_DELAY_US", "50000");
        async = run_graph(backend, fixture, true, true);
    }
    require_persistent_cache_probe_unchanged(persistent_probe);
    GGML_ASSERT(off.trace.find("htp_cpu_residual_timing") == std::string::npos);
    {
        env_guard residual_profile("SD_HTP_RESIDUAL_PROFILE", "1");
        const run_result profiled = run_graph(backend, fixture, false, false);
        GGML_ASSERT(profiled.status == GGML_STATUS_SUCCESS);
        GGML_ASSERT(profiled.trace.find(
            "htp_cpu_residual_timing stage=rho_sub") != std::string::npos);
        GGML_ASSERT(profiled.trace.find(
            "htp_cpu_residual_timing stage=rho_exp") != std::string::npos);
        GGML_ASSERT(profiled.trace.find(
            "htp_cpu_residual_timing stage=scale_mul") != std::string::npos);
        GGML_ASSERT(profiled.trace.find(
            "htp_cpu_residual_timing stage=residual_sub") != std::string::npos);
    }
    GGML_ASSERT(off.status == GGML_STATUS_SUCCESS);
    GGML_ASSERT(sync.status == GGML_STATUS_SUCCESS);
    GGML_ASSERT(inline_async.status == GGML_STATUS_SUCCESS);
    GGML_ASSERT(async.status == GGML_STATUS_SUCCESS);
    require_routes(off, sync);
    require_routes(off, inline_async);
    require_routes(off, async);
    require_numerics(off, sync, dense_ref, sparse_ref);
    require_numerics(off, inline_async, dense_ref, sparse_ref);
    require_numerics(off, async, dense_ref, sparse_ref);
    require_async_parity(sync, inline_async);
    require_async_parity(sync, async);
    require_async_event_order(async.trace);
    require_early_output_lifetime(backend, fixture);
    require_early_mask_lifetime(backend, fixture, sync);
    require_task11_failure_modes(backend, fixture, sync);
    require_task12_residual_failure_modes(backend, sync);
    require_task12_abort_and_consumer_failure(backend);

    {
        env_guard residual_route("SD_HTP_SPARSE_OPENCL_RESIDUAL", "1");
        env_guard residual_queue("SD_HTP_SPARSE_OPENCL_RESIDUAL_QUEUE", nullptr);
        graph_fixture routed_fixture = make_graph(backend);
        env_guard delayed_residual_submission(
            "SD_HTP_SPARSE_OPENCL_TEST_SUBMIT_DELAY_US", "250000");
        const run_result routed = run_graph(backend, routed_fixture, true, true);
        if (routed.status != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "--- routed residual failure trace ---\n%s",
                         routed.trace.c_str());
        }
        GGML_ASSERT(routed.status == GGML_STATUS_SUCCESS);
        std::fprintf(
            stderr,
            "routed residual errors dense=%.9g sparse=%.9g rho=%.9g residual=%.9g\n",
            max_abs_diff(sync.dense_packed, routed.dense_packed),
            max_abs_diff(sync.sparse_packed, routed.sparse_packed),
            max_abs_diff(sync.rho, routed.rho),
            max_abs_diff(sync.residual, routed.residual));
        GGML_ASSERT(max_abs_diff(sync.dense_packed, routed.dense_packed) <= 1.0e-6f);
        GGML_ASSERT(max_abs_diff(sync.rho, routed.rho) <= 2.0e-5f);
        GGML_ASSERT(max_abs_diff(sync.residual, routed.residual) <= 2.0e-5f);
        GGML_ASSERT(routed.trace.find("gpu_rho_residual_enqueue") != std::string::npos);
        GGML_ASSERT(routed.trace.find(
            "gpu_sparse_enqueue queue=sparse") != std::string::npos);
        GGML_ASSERT(routed.trace.find(
            "gpu_rho_residual_enqueue queue=residual") != std::string::npos);
        GGML_ASSERT(routed.trace.find("gpu_rho_enqueue") == std::string::npos);
        GGML_ASSERT(routed.trace.find("gpu_residual_enqueue") == std::string::npos);
        GGML_ASSERT(routed.trace.find("gpu_residual_route") != std::string::npos);
        GGML_ASSERT(routed.trace.find(
            "htp_opencl_residual_host_timing") == std::string::npos);
        GGML_ASSERT(routed.trace.find("rho_consumer_begin") == std::string::npos);
        GGML_ASSERT(routed.trace.find("op=SUB") == std::string::npos);
        const uint64_t sparse_device_end = require_trace_u64_field(
            routed.trace, "htp_opencl_residual_timing", "sparse_end_ns");
        const uint64_t fused_device_start = require_trace_u64_field(
            routed.trace, "htp_opencl_residual_timing", "fused_start_ns");
        GGML_ASSERT(sparse_device_end <= fused_device_start);
        const int64_t routed_dense_end =
            require_event_timestamp(routed.trace, "htp_dense_end");
        const int64_t residual_submit =
            require_event_timestamp(routed.trace, "gpu_residual_submit_job");
        const int64_t next_htp_node =
            require_event_timestamp(routed.trace, "next_htp_node_begin");
        const int64_t residual_enqueue =
            require_event_timestamp(routed.trace, "gpu_rho_residual_enqueue");
        const int64_t post_residual_htp_node =
            require_event_timestamp_for_producer(
                routed.trace,
                "htp_dense_begin",
                "exact_rho_post_residual_htp_dense");
        const int64_t residual_complete =
            require_event_timestamp(routed.trace, "gpu_residual_complete");
        const int64_t graph_finish_wait_begin =
            require_event_timestamp(routed.trace, "gpu_residual_graph_finish_wait_begin");
        const int64_t graph_finish_wait_end =
            require_event_timestamp(routed.trace, "gpu_residual_graph_finish_wait_end");
        GGML_ASSERT(routed_dense_end <= residual_submit);
        GGML_ASSERT(residual_submit < next_htp_node);
        GGML_ASSERT(residual_submit <= residual_enqueue);
        GGML_ASSERT(residual_submit < post_residual_htp_node);
        GGML_ASSERT(post_residual_htp_node < residual_complete);
        GGML_ASSERT(graph_finish_wait_begin <= residual_complete);
        GGML_ASSERT(residual_complete <= graph_finish_wait_end);
        GGML_ASSERT(routed.trace.find("gpu_sparse_publish_begin") == std::string::npos);
        GGML_ASSERT(routed.trace.find("gpu_sparse_publish_end") == std::string::npos);
        for (float value : routed.sparse_packed) {
            GGML_ASSERT(value == k_live_alias_sentinel);
        }
        free_graph_fixture(routed_fixture);

        env_guard residual_profile("SD_HTP_RESIDUAL_PROFILE", "1");
        graph_fixture profiled_fixture = make_graph(
            backend, k_residual_profile_generation);
        const run_result profiled = run_graph(
            backend, profiled_fixture, true, true);
        GGML_ASSERT(profiled.status == GGML_STATUS_SUCCESS);
        GGML_ASSERT(profiled.trace.find(
            "htp_opencl_residual_host_timing") != std::string::npos);
        free_graph_fixture(profiled_fixture);
    }

    {
        env_guard residual_route("SD_HTP_SPARSE_OPENCL_RESIDUAL", "1");
        env_guard residual_queue("SD_HTP_SPARSE_OPENCL_RESIDUAL_QUEUE", "1");
        env_guard hold_residual(
            "SD_HTP_SPARSE_OPENCL_TEST_HOLD_RESIDUAL_FOR_SPARSE", "1");
        graph_fixture priority_fixture = make_graph(
            backend, k_routed_residual_generation, true);
        const run_result priority = run_graph(
            backend, priority_fixture, true, true);
        if (priority.status != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "--- priority failure trace ---\n%s",
                         priority.trace.c_str());
        }
        GGML_ASSERT(priority.status == GGML_STATUS_SUCCESS);
        const int64_t residual_publish = require_event_timestamp(
            priority.trace, "gpu_residual_submit_job");
        const int64_t sparse_publish = require_event_timestamp_for_producer(
            priority.trace, "gpu_sparse_submit_job",
            "exact_rho_priority_sparse_packed");
        const int64_t sparse_enqueue = require_event_timestamp_for_producer(
            priority.trace, "gpu_sparse_enqueue",
            "exact_rho_priority_sparse_packed");
        const int64_t residual_enqueue = require_event_timestamp(
            priority.trace, "gpu_rho_residual_enqueue");
        GGML_ASSERT(residual_publish < sparse_publish);
        GGML_ASSERT(sparse_publish <= sparse_enqueue);
        GGML_ASSERT(sparse_enqueue < residual_enqueue);
        free_graph_fixture(priority_fixture);
    }

    {
        env_guard residual_route("SD_HTP_SPARSE_OPENCL_RESIDUAL", "1");
        env_guard delayed_residual_submission(
            "SD_HTP_SPARSE_OPENCL_TEST_SUBMIT_DELAY_US", "250000");
        env_guard sidecar("SD_HTP_SPARSE_OPENCL", "1");
        env_guard async_sidecar("SD_HTP_SPARSE_OPENCL_ASYNC", "1");

        graph_fixture branch0 = make_graph(
            backend, k_cross_graph_residual_generation_0);
        initialize_graph_inputs(branch0);
        ggml_backend_htp_opencl_set_compute_arena_generation(
            backend, k_cross_graph_arena_generation_0);
        ggml_status branch0_status = GGML_STATUS_FAILED;
        const std::string branch0_trace = capture_graph_compute(
            backend, branch0.graph, &branch0_status);
        GGML_ASSERT(branch0_status == GGML_STATUS_SUCCESS);
        GGML_ASSERT(ggml_backend_htp_opencl_arena_generation_pending(
            backend, k_cross_graph_arena_generation_0));
        GGML_ASSERT(branch0_trace.find("gpu_residual_submit_job") != std::string::npos);
        GGML_ASSERT(
            branch0_trace.find("gpu_residual_graph_finish_wait_begin") ==
            std::string::npos);

        bool released = true;
        char reason[512] = {};
        GGML_ASSERT(ggml_backend_htp_opencl_release_arena_generation(
            backend, k_cross_graph_arena_generation_0, false, &released,
            reason, sizeof(reason)));
        GGML_ASSERT(!released);

        graph_fixture branch1 = make_graph(
            backend, k_cross_graph_residual_generation_1);
        initialize_graph_inputs(branch1);
        ggml_backend_htp_opencl_set_compute_arena_generation(
            backend, k_cross_graph_arena_generation_1);
        ggml_status branch1_status = GGML_STATUS_FAILED;
        const std::string branch1_trace = capture_graph_compute(
            backend, branch1.graph, &branch1_status);
        GGML_ASSERT(branch1_status == GGML_STATUS_SUCCESS);
        GGML_ASSERT(branch1_trace.find("htp_dense_begin") != std::string::npos);
        GGML_ASSERT(ggml_backend_htp_opencl_deferred_arena_count(backend) >= 1);

        GGML_ASSERT(ggml_backend_htp_opencl_release_arena_generation(
            backend, k_cross_graph_arena_generation_0, true, &released,
            reason, sizeof(reason)));
        GGML_ASSERT(released);
        GGML_ASSERT(ggml_backend_htp_opencl_release_arena_generation(
            backend, k_cross_graph_arena_generation_1, true, &released,
            reason, sizeof(reason)));
        GGML_ASSERT(released);
        GGML_ASSERT(ggml_backend_htp_opencl_deferred_arena_count(backend) == 0);

        GGML_ASSERT(max_abs_diff(sync.rho, read_f32(branch0.rho)) <= 2.0e-5f);
        GGML_ASSERT(max_abs_diff(sync.residual, read_f32(branch0.residual)) <= 2.0e-5f);
        GGML_ASSERT(max_abs_diff(sync.rho, read_f32(branch1.rho)) <= 2.0e-5f);
        GGML_ASSERT(max_abs_diff(sync.residual, read_f32(branch1.residual)) <= 2.0e-5f);
        free_graph_fixture(branch1);
        free_graph_fixture(branch0);
    }

    {
        env_guard residual_route("SD_HTP_SPARSE_OPENCL_RESIDUAL", "1");
        env_guard delayed_residual_submission(
            "SD_HTP_SPARSE_OPENCL_TEST_SUBMIT_DELAY_US", "500000");
        env_guard sidecar("SD_HTP_SPARSE_OPENCL", "1");
        env_guard async_sidecar("SD_HTP_SPARSE_OPENCL_ASYNC", "1");

        graph_fixture producer = make_graph(
            backend, k_delayed_consumer_cache_generation);
        initialize_graph_inputs(producer);
        ggml_backend_htp_opencl_set_compute_arena_generation(
            backend, k_delayed_consumer_arena_generation);
        ggml_status producer_status = GGML_STATUS_FAILED;
        const std::string producer_trace = capture_graph_compute(
            backend, producer.graph, &producer_status);
        GGML_ASSERT(producer_status == GGML_STATUS_SUCCESS);
        GGML_ASSERT(ggml_backend_htp_opencl_arena_generation_pending(
            backend, k_delayed_consumer_arena_generation));
        GGML_ASSERT(producer_trace.find("gpu_residual_submit_job") != std::string::npos);

        ggml_tensor * cache_input = ggml_dup_tensor(producer.ctx, producer.rho);
        ggml_set_name(cache_input, "deferred_rho_cache_input");
        ggml_set_input(cache_input);
        cache_input->buffer = producer.rho->buffer;
        cache_input->data = producer.rho->data;
        cache_input->extra = producer.rho->extra;
        ggml_tensor * cache_copy = ggml_map_custom1(
            producer.ctx, cache_input, capture_deferred_cache_input, 1, nullptr);
        ggml_set_name(cache_copy, "deferred_rho_cache_copy");
        ggml_set_output(cache_copy);
        ggml_cgraph * consumer_graph = ggml_new_graph(producer.ctx);
        GGML_ASSERT(consumer_graph != nullptr);
        ggml_build_forward_expand(consumer_graph, cache_copy);
        ggml_gallocr_t consumer_allocator =
            ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        GGML_ASSERT(consumer_allocator != nullptr);
        GGML_ASSERT(ggml_gallocr_alloc_graph(consumer_allocator, consumer_graph));

        char reason[512] = {};
        GGML_ASSERT(ggml_backend_htp_opencl_bind_cache_input(
            backend,
            cache_input,
            producer.rho->name,
            k_delayed_consumer_cache_generation,
            reason,
            sizeof(reason)));
        ggml_status consumer_status = GGML_STATUS_FAILED;
        const std::string consumer_trace = capture_graph_compute(
            backend, consumer_graph, &consumer_status);
        GGML_ASSERT(consumer_status == GGML_STATUS_SUCCESS);
        const int64_t wait_begin = require_event_timestamp(
            consumer_trace, "gpu_residual_cache_wait_begin");
        const int64_t wait_end = require_event_timestamp(
            consumer_trace, "gpu_residual_cache_wait_end");
        const int64_t consumer_begin = require_event_timestamp(
            consumer_trace, "deferred_cache_consumer_begin");
        GGML_ASSERT(wait_begin <= wait_end);
        GGML_ASSERT(wait_end <= consumer_begin);
        GGML_ASSERT(max_abs_diff(sync.rho, read_f32(cache_copy)) <= 2.0e-5f);

        ggml_tensor * ready_cache_input = ggml_dup_tensor(producer.ctx, producer.rho);
        ggml_set_name(ready_cache_input, "ready_deferred_rho_cache_input");
        ggml_set_input(ready_cache_input);
        ready_cache_input->buffer = producer.rho->buffer;
        ready_cache_input->data = producer.rho->data;
        ready_cache_input->extra = producer.rho->extra;
        ggml_tensor * ready_cache_copy = ggml_map_custom1(
            producer.ctx, ready_cache_input, capture_deferred_cache_input, 1, nullptr);
        ggml_set_name(ready_cache_copy, "ready_deferred_rho_cache_copy");
        ggml_set_output(ready_cache_copy);
        ggml_cgraph * ready_consumer_graph = ggml_new_graph(producer.ctx);
        GGML_ASSERT(ready_consumer_graph != nullptr);
        ggml_build_forward_expand(ready_consumer_graph, ready_cache_copy);
        ggml_gallocr_t ready_consumer_allocator =
            ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        GGML_ASSERT(ready_consumer_allocator != nullptr);
        GGML_ASSERT(ggml_gallocr_alloc_graph(
            ready_consumer_allocator, ready_consumer_graph));
        GGML_ASSERT(ggml_backend_htp_opencl_bind_cache_input(
            backend, ready_cache_input, producer.rho->name,
            k_delayed_consumer_cache_generation, reason, sizeof(reason)));
        GGML_ASSERT(htp_ops_opencl_node_needs_wait(ready_cache_copy));
        GGML_ASSERT(htp_ops_opencl_wait_node(ready_cache_copy) == 0);
        GGML_ASSERT(htp_ops_opencl_node_needs_wait(ready_cache_copy));
        ggml_status ready_consumer_status = GGML_STATUS_FAILED;
        capture_graph_compute(
            backend, ready_consumer_graph, &ready_consumer_status);
        GGML_ASSERT(ready_consumer_status == GGML_STATUS_SUCCESS);
        GGML_ASSERT(
            ggml_backend_htp_opencl_cache_input_binding_count(backend) == 0);
        ggml_gallocr_free(ready_consumer_allocator);

        bool released = false;
        GGML_ASSERT(ggml_backend_htp_opencl_release_arena_generation(
            backend, k_delayed_consumer_arena_generation, false, &released,
            reason, sizeof(reason)));
        GGML_ASSERT(released);
        ggml_gallocr_free(consumer_allocator);
        free_graph_fixture(producer);
    }

    free_graph_fixture(fixture);
    require_persistent_cache_probe_unchanged(persistent_probe);
    ggml_backend_htp_async_cache_slot_free(persistent_probe.slot);
    ggml_backend_free(backend);
    ggml_backend_htp_set_hybrid_enabled(false);
    std::printf("test-backend-htp-opencl-exact-rho passed\n");
    return 0;
}
