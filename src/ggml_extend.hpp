#ifndef __GGML_EXTEND_HPP__
#define __GGML_EXTEND_HPP__

#include <assert.h>
#include <inttypes.h>
#include <stdarg.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-htp.h"
#include "ggml.h"
#include "ggml_extend_backend.h"
#include "ggml_graph_cut.h"
#include "ggml_qkv_buffer_plan.hpp"

#include "htp_tensor_flags.h"
#include "htp_packed_staging.h"
#include "ggml/src/ggml-htp/htp_param_placement.h"
#include "ggml/src/ggml-htp/session_placement.h"
#include "ggml/src/ggml-htp/session_ownership.h"
#include "model.h"
#include "tensor.hpp"

#include "rng.hpp"
#include "tensor_ggml.hpp"
#include "util.h"

#define EPS 1e-05f

extern "C" bool htp_ops_support_op_runtime(const struct ggml_tensor * dst);

#ifndef __STATIC_INLINE__
#define __STATIC_INLINE__ static inline
#endif

#ifndef SD_UNUSED
#define SD_UNUSED(x) (void)(x)
#endif

// 统一解析布尔环境变量，避免把 "0" 这种显式关闭值误判成开启。
__STATIC_INLINE__ bool sd_env_flag_enabled(const char* name) {
    const char* value = name != nullptr ? std::getenv(name) : nullptr;
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

__STATIC_INLINE__ bool sd_env_flag_enabled_default(const char* name, bool default_value) {
    const char* value = name != nullptr ? std::getenv(name) : nullptr;
    if (value == nullptr || value[0] == '\0') {
        return default_value;
    }
    return std::strcmp(value, "0") != 0;
}

__STATIC_INLINE__ const char* sd_env_value(const char* name) {
    const char* value = name != nullptr ? std::getenv(name) : nullptr;
    return value != nullptr && value[0] != '\0' ? value : nullptr;
}

__STATIC_INLINE__ int sd_env_int_value(const char* name, int default_value) {
    const char* value = sd_env_value(name);
    return value != nullptr ? std::atoi(value) : default_value;
}

__STATIC_INLINE__ float sd_env_float_value(const char* name, float default_value) {
    const char* value = sd_env_value(name);
    return value != nullptr ? std::strtof(value, nullptr) : default_value;
}

__STATIC_INLINE__ int sd_env_ratio_to_keep_count(const char* name, int total, float default_ratio) {
    if (total <= 0) {
        return 0;
    }
    const float ratio = std::clamp(sd_env_float_value(name, default_ratio), 0.0f, 1.0f);
    return std::max(1, std::min(total, static_cast<int>(std::ceil(ratio * float(total)))));
}

// pack_for_htp and unpack_from_htp removed - not used in actual inference with diffusion_flash_attn=false

constexpr int SD_HTP_BLOCK_SPARSE_MASK_BLOCK = 32;
constexpr int SD_HTP_BLOCK_SPARSE_PATCH_BLOCK = 16;
constexpr int SD_HTP_BLOCK_SPARSE_PATCH_SIZE = 4;
constexpr int SD_HTP_BLOCK_SPARSE_STEP_SHIFT_PERIOD = 2;

__STATIC_INLINE__ float sd_htp_block_sparse_keep_ratio() {
    return std::clamp(sd_env_float_value("SD_HTP_BLOCK_SPARSE_KEEP_RATIO", 0.0f), 0.0f, 1.0f);
}

__STATIC_INLINE__ bool sd_htp_block_sparse_score_enabled(float keep_ratio) {
    return keep_ratio != 0.0f && keep_ratio != 1.0f;
}

__STATIC_INLINE__ bool sd_htp_block_sparse_score_enabled() {
    return sd_htp_block_sparse_score_enabled(sd_htp_block_sparse_keep_ratio());
}

__STATIC_INLINE__ bool sd_htp_block_sparse_direct_halo_enabled(float keep_ratio, bool patch_halo) {
    return keep_ratio == 0.0f && patch_halo;
}

__STATIC_INLINE__ int sd_htp_block_sparse_step_shift(bool enabled, int sparse_step_index) {
    if (!enabled || sparse_step_index <= 0) {
        return 0;
    }
    return sparse_step_index % SD_HTP_BLOCK_SPARSE_STEP_SHIFT_PERIOD == 0 ? -2 : 0;
}

__STATIC_INLINE__ int sd_htp_block_sparse_scheduled_step_index(int candidate_sparse_step_index,
                                                              int dense_after_sparse_steps) {
    if (candidate_sparse_step_index <= 0) {
        return 0;
    }
    if (dense_after_sparse_steps <= 0) {
        return candidate_sparse_step_index;
    }
    const int cycle = dense_after_sparse_steps + 1;
    if (candidate_sparse_step_index % cycle == 0) {
        return 0;
    }
    return candidate_sparse_step_index - candidate_sparse_step_index / cycle;
}

enum sd_htp_block_sparse_residual_mode {
    SD_HTP_BLOCK_SPARSE_RESIDUAL_OFF    = 0,
    SD_HTP_BLOCK_SPARSE_RESIDUAL_APPLY  = 1,
    SD_HTP_BLOCK_SPARSE_RESIDUAL_ANCHOR = 2,
};

enum sd_htp_route_mode {
    SD_HTP_ROUTE_DISABLED = 0,
    SD_HTP_ROUTE_FUSED_OPENCL_RESIDUAL,
    SD_HTP_ROUTE_OPENCL_ANCHOR,
    SD_HTP_ROUTE_HTP_CPU,
    SD_HTP_ROUTE_INVALID,
};

struct sd_htp_attention_policy {
    bool sparse_policy_active = false;
    bool packed_qkv_allowed   = false;
    bool residual_eligible    = false;
};

struct sd_htp_residual_binding {
    std::string residual_key;
    std::string rho_key;
    ggml_tensor* cached_residual = nullptr;
    ggml_tensor* cached_rho      = nullptr;
    int slot                     = -1;
};

__STATIC_INLINE__ sd_htp_route_mode sd_htp_route_mode_from_flags(bool fused,
                                                                  bool opencl_sparse,
                                                                  bool opencl_residual) {
    if (fused && !opencl_sparse && opencl_residual) {
        return SD_HTP_ROUTE_FUSED_OPENCL_RESIDUAL;
    }
    if (!fused && opencl_sparse && opencl_residual) {
        return SD_HTP_ROUTE_OPENCL_ANCHOR;
    }
    if (!fused && !opencl_sparse && !opencl_residual) {
        return SD_HTP_ROUTE_HTP_CPU;
    }
    return SD_HTP_ROUTE_INVALID;
}

__STATIC_INLINE__ ggml_flash_attn_ext_halo_kv_load_mode
sd_htp_sparse_halo_kv_load_override(sd_htp_route_mode route, int residual_mode) {
    return route == SD_HTP_ROUTE_HTP_CPU &&
                   residual_mode == SD_HTP_BLOCK_SPARSE_RESIDUAL_ANCHOR
        ? GGML_FLASH_ATTN_EXT_HALO_KV_LOAD_CHUNKED
        : GGML_FLASH_ATTN_EXT_HALO_KV_LOAD_INHERIT;
}

__STATIC_INLINE__ void sd_htp_apply_sparse_halo_kv_load_override(
    ggml_tensor* candidate, sd_htp_route_mode route, int residual_mode) {
    if (candidate == nullptr || candidate->op != GGML_OP_FLASH_ATTN_EXT ||
        ggml_flash_attn_ext_get_sparse_policy(candidate) !=
            GGML_FLASH_ATTN_EXT_SPARSE_POLICY_STATIC_HALO ||
        ggml_flash_attn_ext_get_mask_format(candidate) !=
            GGML_FLASH_ATTN_EXT_MASK_BLOCK_INDEX) {
        return;
    }
    const auto override_mode = sd_htp_sparse_halo_kv_load_override(route, residual_mode);
    if (override_mode != GGML_FLASH_ATTN_EXT_HALO_KV_LOAD_INHERIT) {
        ggml_flash_attn_ext_set_halo_kv_load_mode(candidate, override_mode);
    }
}

__STATIC_INLINE__ sd_htp_attention_policy sd_htp_attention_policy_for(
    bool block_sparse_step,
    int residual_mode,
    bool has_image_region) {
    const bool anchor = residual_mode == SD_HTP_BLOCK_SPARSE_RESIDUAL_ANCHOR;
    const bool apply  = residual_mode == SD_HTP_BLOCK_SPARSE_RESIDUAL_APPLY;
    return {
        has_image_region && (block_sparse_step || anchor),
        has_image_region && block_sparse_step && !anchor,
        has_image_region && (anchor || apply),
    };
}

__STATIC_INLINE__ bool sd_htp_patch_major_layout_enabled(bool sparse_attn_enabled,
                                                          int residual_mode) {
    return sparse_attn_enabled || residual_mode == SD_HTP_BLOCK_SPARSE_RESIDUAL_ANCHOR;
}

// Residual caches use the canonical dense O stride. Every packed apply must
// materialize its logical O view before adding the cache, regardless of which
// rho path produced that cache.
__STATIC_INLINE__ bool sd_htp_packed_residual_apply_needs_contiguous(
    bool packed_layout, int residual_mode) {
    return packed_layout &&
           residual_mode == SD_HTP_BLOCK_SPARSE_RESIDUAL_APPLY;
}

__STATIC_INLINE__ bool sd_htp_packed_residual_apply_uses_lse(
    bool packed_layout, int residual_mode) {
    SD_UNUSED(residual_mode);
    return packed_layout;
}

__STATIC_INLINE__ bool sd_htp_sparse_attention_needs_lse(
    bool packed_layout, int residual_mode, float kv_scale,
    bool anchor_can_cache_exact_rho) {
    return sd_htp_packed_residual_apply_uses_lse(packed_layout, residual_mode) ||
           (!packed_layout && kv_scale != 1.0f) ||
           anchor_can_cache_exact_rho;
}

__STATIC_INLINE__ uint64_t sd_htp_residual_trace_hash(const ggml_tensor* tensor) {
    if (tensor == nullptr || tensor->data == nullptr) {
        return 0;
    }
    const uint8_t* bytes = static_cast<const uint8_t*>(tensor->data);
    uint64_t hash = UINT64_C(14695981039346656037);
    const size_t size = ggml_nbytes(tensor);
    for (size_t i = 0; i < size; ++i) {
        hash ^= bytes[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

__STATIC_INLINE__ sd_htp_block_sparse_residual_mode sd_htp_block_sparse_residual_mode(bool enabled,
                                                                                      bool periodic_dense,
                                                                                      int sparse_step_index) {
    if (!enabled) {
        return SD_HTP_BLOCK_SPARSE_RESIDUAL_OFF;
    }
    if (periodic_dense) {
        return SD_HTP_BLOCK_SPARSE_RESIDUAL_ANCHOR;
    }
    if (sparse_step_index > 0) {
        return SD_HTP_BLOCK_SPARSE_RESIDUAL_APPLY;
    }
    return SD_HTP_BLOCK_SPARSE_RESIDUAL_OFF;
}

__STATIC_INLINE__ bool sd_htp_block_sparse_residual_anchor_has_sparse_consumer(int step_index,
                                                                               int total_steps,
                                                                               int dense_tail_steps) {
    if (step_index <= 0 || total_steps <= 0) {
        return false;
    }
    const int next_step = step_index + 1;
    if (next_step > total_steps) {
        return false;
    }
    const int first_tail_step = dense_tail_steps > 0 ? total_steps - dense_tail_steps + 1 : total_steps + 1;
    return next_step < first_tail_step;
}

__STATIC_INLINE__ bool sd_htp_block_sparse_warmup_residual_anchor_step(int step_index,
                                                                       int warmup_steps,
                                                                       int total_steps,
                                                                       int dense_tail_steps) {
    if (step_index <= 0 || warmup_steps <= 0 || total_steps <= 0) {
        return false;
    }
    if (step_index != warmup_steps) {
        return false;
    }
    return sd_htp_block_sparse_residual_anchor_has_sparse_consumer(step_index, total_steps, dense_tail_steps);
}

__STATIC_INLINE__ float sd_htp_block_sparse_residual_rho_value(float rho) {
    return std::clamp(rho, 0.0f, 1.0f);
}

__STATIC_INLINE__ std::string sd_htp_block_sparse_residual_cache_key(int branch_index,
                                                                     const char* attention_kind,
                                                                     int attention_slot) {
    std::string key = "htp_block_sparse_residual_b" + std::to_string(branch_index);
    if (attention_kind != nullptr && attention_kind[0] != '\0') {
        key += "_";
        key += attention_kind;
    }
    key += "_s";
    key += std::to_string(attention_slot);
    return key;
}

__STATIC_INLINE__ std::string sd_htp_block_sparse_residual_rho_cache_key(int branch_index,
                                                                         const char* attention_kind,
                                                                         int attention_slot) {
    std::string key = "htp_block_sparse_residual_rho_b" + std::to_string(branch_index);
    if (attention_kind != nullptr && attention_kind[0] != '\0') {
        key += "_";
        key += attention_kind;
    }
    key += "_s";
    key += std::to_string(attention_slot);
    return key;
}

__STATIC_INLINE__ int sd_patch_major_token_count(int token_h, int token_w) {
    return std::max(0, token_h) * std::max(0, token_w);
}

__STATIC_INLINE__ int sd_patch_major_index_from_row_major(int row_major,
                                                         int token_h,
                                                         int token_w,
                                                         int patch) {
    GGML_ASSERT(row_major >= 0);
    GGML_ASSERT(token_h > 0 && token_w > 0 && patch > 0);
    GGML_ASSERT(row_major < token_h * token_w);
    const int y = row_major / token_w;
    const int x = row_major % token_w;
    const int patch_cols = (token_w + patch - 1) / patch;

    int out = 0;
    for (int by = 0; by < y / patch; ++by) {
        const int bh = std::min(patch, token_h - by * patch);
        for (int bx = 0; bx < patch_cols; ++bx) {
            const int bw = std::min(patch, token_w - bx * patch);
            out += bh * bw;
        }
    }
    for (int bx = 0; bx < x / patch; ++bx) {
        const int bw = std::min(patch, token_w - bx * patch);
        const int bh = std::min(patch, token_h - (y / patch) * patch);
        out += bh * bw;
    }

    const int local_y = y % patch;
    const int local_x = x % patch;
    const int bw = std::min(patch, token_w - (x / patch) * patch);
    return out + local_y * bw + local_x;
}

__STATIC_INLINE__ int sd_row_major_index_from_patch_major(int patch_major,
                                                         int token_h,
                                                         int token_w,
                                                         int patch) {
    GGML_ASSERT(patch_major >= 0);
    GGML_ASSERT(token_h > 0 && token_w > 0 && patch > 0);
    GGML_ASSERT(patch_major < token_h * token_w);
    const int patch_rows = (token_h + patch - 1) / patch;
    const int patch_cols = (token_w + patch - 1) / patch;
    int remaining = patch_major;
    for (int by = 0; by < patch_rows; ++by) {
        const int bh = std::min(patch, token_h - by * patch);
        for (int bx = 0; bx < patch_cols; ++bx) {
            const int bw = std::min(patch, token_w - bx * patch);
            const int block_tokens = bh * bw;
            if (remaining < block_tokens) {
                const int local_y = remaining / bw;
                const int local_x = remaining % bw;
                return (by * patch + local_y) * token_w + (bx * patch + local_x);
            }
            remaining -= block_tokens;
        }
    }
    return -1;
}

__STATIC_INLINE__ int sd_positive_mod(int v, int m) {
    GGML_ASSERT(m > 0);
    const int r = v % m;
    return r < 0 ? r + m : r;
}

__STATIC_INLINE__ int sd_patch_major_shifted_index_from_row_major(int row_major,
                                                                  int token_h,
                                                                  int token_w,
                                                                  int patch,
                                                                  int shift_y,
                                                                  int shift_x) {
    GGML_ASSERT(row_major >= 0);
    GGML_ASSERT(token_h > 0 && token_w > 0 && patch > 0);
    GGML_ASSERT(row_major < token_h * token_w);
    const int y = row_major / token_w;
    const int x = row_major % token_w;
    const int shifted_y = sd_positive_mod(y + shift_y, token_h);
    const int shifted_x = sd_positive_mod(x + shift_x, token_w);
    const int shifted_row = shifted_y * token_w + shifted_x;
    return sd_patch_major_index_from_row_major(shifted_row, token_h, token_w, patch);
}

__STATIC_INLINE__ int sd_row_major_index_from_patch_major_shifted(int patch_major,
                                                                  int token_h,
                                                                  int token_w,
                                                                  int patch,
                                                                  int shift_y,
                                                                  int shift_x) {
    GGML_ASSERT(patch_major >= 0);
    GGML_ASSERT(token_h > 0 && token_w > 0 && patch > 0);
    GGML_ASSERT(patch_major < token_h * token_w);
    const int shifted_row = sd_row_major_index_from_patch_major(patch_major, token_h, token_w, patch);
    GGML_ASSERT(shifted_row >= 0);
    const int shifted_y = shifted_row / token_w;
    const int shifted_x = shifted_row % token_w;
    const int y = sd_positive_mod(shifted_y - shift_y, token_h);
    const int x = sd_positive_mod(shifted_x - shift_x, token_w);
    return y * token_w + x;
}

struct sd_patch_major_reorder_params {
    int token_h;
    int token_w;
    int patch;
    int inverse;
    int shift_y;
    int shift_x;
    int src_shift_y;
    int src_shift_x;
    int src_patch_major;
};

__STATIC_INLINE__ void sd_patch_major_reorder_custom(ggml_tensor* dst,
                                                     int ith,
                                                     int nth,
                                                     void* userdata) {
    const auto* params = static_cast<const sd_patch_major_reorder_params*>(userdata);
    const ggml_tensor* src = dst->src[0];
    GGML_ASSERT(src != nullptr && params != nullptr);
    GGML_ASSERT(src->type == dst->type);
    GGML_ASSERT(src->ne[0] == dst->ne[0] && src->ne[1] == dst->ne[1] && src->ne[2] == dst->ne[2]);
    GGML_ASSERT(src->ne[1] == int64_t(params->token_h) * int64_t(params->token_w));
    GGML_ASSERT(ggml_blck_size(src->type) == 1);

    const size_t type_size = ggml_type_size(src->type);
    GGML_ASSERT(src->nb[0] == static_cast<int64_t>(type_size));
    GGML_ASSERT(dst->nb[0] == static_cast<int64_t>(type_size));
    const size_t row_bytes = static_cast<size_t>(src->ne[0]) * type_size;

    for (int64_t n = ith; n < src->ne[2]; n += nth) {
        for (int64_t dst_l = 0; dst_l < src->ne[1]; ++dst_l) {
            int src_l = -1;
            if (params->inverse != 0) {
                src_l = sd_patch_major_shifted_index_from_row_major(static_cast<int>(dst_l),
                                                                    params->token_h,
                                                                    params->token_w,
                                                                    params->patch,
                                                                    params->shift_y,
                                                                    params->shift_x);
            } else if (params->src_patch_major != 0) {
                const int row = sd_row_major_index_from_patch_major_shifted(static_cast<int>(dst_l),
                                                                            params->token_h,
                                                                            params->token_w,
                                                                            params->patch,
                                                                            params->shift_y,
                                                                            params->shift_x);
                src_l = sd_patch_major_shifted_index_from_row_major(row,
                                                                    params->token_h,
                                                                    params->token_w,
                                                                    params->patch,
                                                                    params->src_shift_y,
                                                                    params->src_shift_x);
            } else {
                src_l = sd_row_major_index_from_patch_major_shifted(static_cast<int>(dst_l),
                                                                    params->token_h,
                                                                    params->token_w,
                                                                    params->patch,
                                                                    params->shift_y,
                                                                    params->shift_x);
            }
            GGML_ASSERT(src_l >= 0);
            const char* src_row = static_cast<const char*>(src->data) + n * src->nb[2] + int64_t(src_l) * src->nb[1];
            char* dst_row = static_cast<char*>(dst->data) + n * dst->nb[2] + dst_l * dst->nb[1];
            std::memcpy(dst_row, src_row, row_bytes);
        }
    }
}

__STATIC_INLINE__ ggml_tensor* sd_patch_major_reorder_node(ggml_context* ctx,
                                                           ggml_tensor* x,
                                                           int token_h,
                                                           int token_w,
                                                           int patch,
                                                           bool inverse,
                                                           int shift_y = 0,
                                                           int shift_x = 0,
                                                           int src_shift_y = 0,
                                                           int src_shift_x = 0,
                                                           bool src_patch_major = false) {
    GGML_ASSERT(ctx != nullptr && x != nullptr);
    GGML_ASSERT(token_h > 0 && token_w > 0 && patch > 0);
    GGML_ASSERT(x->ne[1] == int64_t(token_h) * int64_t(token_w));
    ggml_tensor* args[] = {x};
    auto* params = static_cast<sd_patch_major_reorder_params*>(
        ggml_new_buffer(ctx, sizeof(sd_patch_major_reorder_params)));
    params->token_h = token_h;
    params->token_w = token_w;
    params->patch = patch;
    params->inverse = inverse ? 1 : 0;
    params->shift_y = shift_y;
    params->shift_x = shift_x;
    params->src_shift_y = src_shift_y;
    params->src_shift_x = src_shift_x;
    params->src_patch_major = src_patch_major ? 1 : 0;
    ggml_tensor* result = ggml_custom_4d(ctx,
                                         x->type,
                                         x->ne[0],
                                         x->ne[1],
                                         x->ne[2],
                                         x->ne[3],
                                         args,
                                         1,
                                         sd_patch_major_reorder_custom,
                                         GGML_N_TASKS_MAX,
                                         params);
    ggml_set_name(result, inverse ? "patch_major_inverse_reorder" : "patch_major_reorder");
    return result;
}

__STATIC_INLINE__ ggml_tensor* sd_optional_patch_major_reorder_node(
        ggml_context* ctx,
        ggml_tensor* x,
        int token_h,
        int token_w,
        int patch,
        bool enabled,
        bool inverse,
        int shift_y = 0,
        int shift_x = 0,
        int src_shift_y = 0,
        int src_shift_x = 0,
        bool src_patch_major = false) {
    if (!enabled) {
        return x;
    }
    return sd_patch_major_reorder_node(ctx,
                                       x,
                                       token_h,
                                       token_w,
                                       patch,
                                       inverse,
                                       shift_y,
                                       shift_x,
                                       src_shift_y,
                                       src_shift_x,
                                       src_patch_major);
}

__STATIC_INLINE__ void ggml_ext_block_sparse_promote_16_mask_to_32_index(const int32_t* mask16,
                                                                         int32_t* index32,
                                                                         int n_q16_blocks,
                                                                         int n_k16_blocks,
                                                                         int max_kept32,
                                                                         int n_heads,
                                                                         int n_kv_heads) {
    GGML_ASSERT(mask16 != nullptr && index32 != nullptr);
    GGML_ASSERT(n_q16_blocks > 0 && n_k16_blocks > 0 && max_kept32 > 0);
    n_heads = std::max(1, n_heads);
    n_kv_heads = std::max(1, n_kv_heads);
    GGML_ASSERT(n_q16_blocks % n_heads == 0);
    GGML_ASSERT(n_k16_blocks % n_kv_heads == 0);
    const int q16_per_head = n_q16_blocks / n_heads;
    const int k16_per_head = n_k16_blocks / n_kv_heads;
    const int q32_per_head = (q16_per_head + 1) / 2;
    const int k32_per_head = (k16_per_head + 1) / 2;
    const int n_q32_blocks = n_heads * q32_per_head;
    const int n_k32_blocks = n_kv_heads * k32_per_head;

    std::vector<int32_t> mask32(size_t(n_q32_blocks) * size_t(n_k32_blocks), 1);
    for (int qb16 = 0; qb16 < n_q16_blocks; ++qb16) {
        const int q_head = qb16 / q16_per_head;
        const int q32 = q_head * q32_per_head + (qb16 % q16_per_head) / 2;
        for (int kb16 = 0; kb16 < n_k16_blocks; ++kb16) {
            if (mask16[size_t(qb16) * size_t(n_k16_blocks) + size_t(kb16)] != 0) {
                continue;
            }
            const int k_head = kb16 / k16_per_head;
            const int k32 = k_head * k32_per_head + (kb16 % k16_per_head) / 2;
            mask32[size_t(q32) * size_t(n_k32_blocks) + size_t(k32)] = 0;
        }
    }

    for (int qb32 = 0; qb32 < n_q32_blocks; ++qb32) {
        int32_t* row = index32 + size_t(qb32) * size_t(max_kept32);
        std::fill(row, row + max_kept32, -1);
        int kept = 0;
        for (int kb32 = 0; kb32 < n_k32_blocks; ++kb32) {
            if (mask32[size_t(qb32) * size_t(n_k32_blocks) + size_t(kb32)] != 0) {
                continue;
            }
            GGML_ASSERT(kept < max_kept32);
            row[kept++] = kb32;
        }
    }
}

__STATIC_INLINE__ bool ggml_ext_block_sparse_fill_direct_halo_index_32(int32_t* index32,
                                                                        int max_kept32,
                                                                        int qo_len,
                                                                        int kv_len,
                                                                        int protected_start,
                                                                        int protected_len,
                                                                        int token_h,
                                                                        int token_w,
                                                                        int n_heads,
                                                                        int n_kv_heads,
                                                                        bool center_sink) {
    if (index32 == nullptr || max_kept32 <= 0 || qo_len <= 0 || kv_len <= 0 ||
        token_h <= 0 || token_w <= 0 || token_h % 4 != 0 || token_w % 8 != 0 ||
        n_heads <= 0 || n_kv_heads <= 0 || n_heads % n_kv_heads != 0) {
        return false;
    }

    protected_start = std::max(0, protected_start);
    protected_len = std::max(0, protected_len);
    const int image_tokens = token_h * token_w;
    const int protected_end = protected_start + protected_len;
    const int covered_tokens = protected_len > 0 ? protected_end : image_tokens;
    if ((protected_len > 0 && protected_start != image_tokens) ||
        (protected_len == 0 && protected_start != 0) ||
        qo_len != covered_tokens || kv_len != covered_tokens ||
        image_tokens % SD_HTP_BLOCK_SPARSE_MASK_BLOCK != 0) {
        return false;
    }

    const int grid_h = token_h / 4;
    const int grid_w = token_w / 8;
    const int image_blocks = grid_h * grid_w;
    const int q_blocks_per_head = (qo_len + SD_HTP_BLOCK_SPARSE_MASK_BLOCK - 1) /
                                  SD_HTP_BLOCK_SPARSE_MASK_BLOCK;
    const int k_blocks_per_head = (kv_len + SD_HTP_BLOCK_SPARSE_MASK_BLOCK - 1) /
                                  SD_HTP_BLOCK_SPARSE_MASK_BLOCK;
    if (image_blocks > q_blocks_per_head || image_blocks > k_blocks_per_head ||
        max_kept32 < k_blocks_per_head) {
        return false;
    }

    const int n_q_blocks = n_heads * q_blocks_per_head;
    std::fill(index32, index32 + size_t(n_q_blocks) * size_t(max_kept32), -1);
    const int gqa = n_heads / n_kv_heads;
    const int protected_first_block = protected_start / SD_HTP_BLOCK_SPARSE_MASK_BLOCK;
    const int protected_end_block = (protected_end + SD_HTP_BLOCK_SPARSE_MASK_BLOCK - 1) /
                                    SD_HTP_BLOCK_SPARSE_MASK_BLOCK;
    const int center_y0 = (grid_h - 1) / 2;
    const int center_y1 = grid_h / 2;
    const int center_x0 = (grid_w - 1) / 2;
    const int center_x1 = grid_w / 2;
    std::vector<uint8_t> keep(size_t(k_blocks_per_head), 0);
    for (int qb = 0; qb < n_q_blocks; ++qb) {
        const int q_head = qb / q_blocks_per_head;
        const int q_block_in_head = qb % q_blocks_per_head;
        const int kv_head = q_head / gqa;
        int32_t* row = index32 + size_t(qb) * size_t(max_kept32);
        int kept = 0;
        if (protected_len > 0 &&
            q_block_in_head >= protected_first_block && q_block_in_head < protected_end_block) {
            for (int kb = 0; kb < k_blocks_per_head; ++kb) {
                row[kept++] = kv_head * k_blocks_per_head + kb;
            }
            continue;
        }
        if (q_block_in_head >= image_blocks) {
            continue;
        }

        const int qy = q_block_in_head / grid_w;
        const int qx = q_block_in_head % grid_w;
        std::fill(keep.begin(), keep.end(), 0);
        for (int ky = std::max(0, qy - 1); ky <= std::min(grid_h - 1, qy + 1); ++ky) {
            for (int kx = std::max(0, qx - 1); kx <= std::min(grid_w - 1, qx + 1); ++kx) {
                keep[size_t(ky * grid_w + kx)] = 1;
            }
        }
        if (center_sink) {
            for (int ky = center_y0; ky <= center_y1; ++ky) {
                for (int kx = center_x0; kx <= center_x1; ++kx) {
                    keep[size_t(ky * grid_w + kx)] = 1;
                }
            }
        }
        for (int kb = protected_first_block; protected_len > 0 && kb < protected_end_block; ++kb) {
            keep[size_t(kb)] = 1;
        }
        for (int kb = 0; kb < k_blocks_per_head; ++kb) {
            if (keep[size_t(kb)] != 0) {
                row[kept++] = kv_head * k_blocks_per_head + kb;
            }
        }
    }
    return true;
}

__STATIC_INLINE__ void ggml_ext_block_sparse_fill_full_index_32(int32_t* index32,
                                                                 int max_kept32,
                                                                 int qo_len,
                                                                 int kv_len,
                                                                 int n_heads,
                                                                 int n_kv_heads) {
    GGML_ASSERT(index32 != nullptr && max_kept32 > 0);
    GGML_ASSERT(qo_len > 0 && kv_len > 0);
    GGML_ASSERT(n_heads > 0 && n_kv_heads > 0 && n_heads % n_kv_heads == 0);
    const int q_blocks_per_head = (qo_len + SD_HTP_BLOCK_SPARSE_MASK_BLOCK - 1) /
                                  SD_HTP_BLOCK_SPARSE_MASK_BLOCK;
    const int k_blocks_per_head = (kv_len + SD_HTP_BLOCK_SPARSE_MASK_BLOCK - 1) /
                                  SD_HTP_BLOCK_SPARSE_MASK_BLOCK;
    GGML_ASSERT(max_kept32 >= k_blocks_per_head);
    const int n_q_blocks = n_heads * q_blocks_per_head;
    const int gqa = n_heads / n_kv_heads;
    std::fill(index32, index32 + size_t(n_q_blocks) * size_t(max_kept32), -1);
    for (int qb = 0; qb < n_q_blocks; ++qb) {
        const int q_head = qb / q_blocks_per_head;
        const int kv_head = q_head / gqa;
        int32_t* row = index32 + size_t(qb) * size_t(max_kept32);
        for (int kb = 0; kb < k_blocks_per_head; ++kb) {
            row[kb] = kv_head * k_blocks_per_head + kb;
        }
    }
}

__STATIC_INLINE__ void ggml_ext_block_sparse_force_patch_halo_mask(int32_t* mask,
                                                                   int n_q_blocks,
                                                                   int n_k_blocks,
                                                                   int image_q_blocks,
                                                                   int image_k_blocks,
                                                                   int n_heads,
                                                                   int n_kv_heads) {
    GGML_ASSERT(mask != nullptr);
    GGML_ASSERT(n_q_blocks > 0 && n_k_blocks > 0);
    n_heads = std::max(1, n_heads);
    n_kv_heads = std::max(1, n_kv_heads);
    GGML_ASSERT(n_heads % n_kv_heads == 0);
    GGML_ASSERT(n_q_blocks % n_heads == 0);
    GGML_ASSERT(n_k_blocks % n_kv_heads == 0);
    const int gqa = std::max(1, n_heads / n_kv_heads);
    const int q_blocks_per_head = n_q_blocks / n_heads;
    const int k_blocks_per_head = n_k_blocks / n_kv_heads;
    image_q_blocks = std::max(0, std::min(image_q_blocks, q_blocks_per_head));
    image_k_blocks = std::max(0, std::min(image_k_blocks, k_blocks_per_head));
    const int q_grid_w = static_cast<int>(std::sqrt(static_cast<double>(image_q_blocks)));
    const int k_grid_w = static_cast<int>(std::sqrt(static_cast<double>(image_k_blocks)));
    if (q_grid_w <= 0 || k_grid_w <= 0 ||
        q_grid_w * q_grid_w != image_q_blocks ||
        k_grid_w * k_grid_w != image_k_blocks ||
        q_grid_w != k_grid_w) {
        return;
    }

    const int grid_w = q_grid_w;
    const int grid_h = image_q_blocks / grid_w;
    for (int qb = 0; qb < n_q_blocks; ++qb) {
        const int q_head = qb / q_blocks_per_head;
        const int q_block_in_head = qb % q_blocks_per_head;
        if (q_block_in_head >= image_q_blocks) {
            continue;
        }
        const int kv_head = std::min(n_kv_heads - 1, q_head / gqa);
        const int qy = q_block_in_head / grid_w;
        const int qx = q_block_in_head % grid_w;
        for (int dy = -1; dy <= 1; ++dy) {
            const int ky = qy + dy;
            if (ky < 0 || ky >= grid_h) {
                continue;
            }
            for (int dx = -1; dx <= 1; ++dx) {
                const int kx = qx + dx;
                if (kx < 0 || kx >= grid_w) {
                    continue;
                }
                const int k_block_in_head = ky * grid_w + kx;
                if (k_block_in_head >= image_k_blocks) {
                    continue;
                }
                const int kb = kv_head * k_blocks_per_head + k_block_in_head;
                mask[size_t(qb) * size_t(n_k_blocks) + size_t(kb)] = 0;
            }
        }
    }
}

__STATIC_INLINE__ void ggml_ext_block_sparse_force_center_sink_mask(int32_t* mask,
                                                                    int n_q_blocks,
                                                                    int n_k_blocks,
                                                                    int image_q_blocks,
                                                                    int image_k_blocks,
                                                                    int n_heads,
                                                                    int n_kv_heads) {
    GGML_ASSERT(mask != nullptr);
    GGML_ASSERT(n_q_blocks > 0 && n_k_blocks > 0);
    n_heads = std::max(1, n_heads);
    n_kv_heads = std::max(1, n_kv_heads);
    GGML_ASSERT(n_heads % n_kv_heads == 0);
    GGML_ASSERT(n_q_blocks % n_heads == 0);
    GGML_ASSERT(n_k_blocks % n_kv_heads == 0);
    const int gqa = std::max(1, n_heads / n_kv_heads);
    const int q_blocks_per_head = n_q_blocks / n_heads;
    const int k_blocks_per_head = n_k_blocks / n_kv_heads;
    image_q_blocks = std::max(0, std::min(image_q_blocks, q_blocks_per_head));
    image_k_blocks = std::max(0, std::min(image_k_blocks, k_blocks_per_head));
    const int q_grid_w = static_cast<int>(std::sqrt(static_cast<double>(image_q_blocks)));
    const int k_grid_w = static_cast<int>(std::sqrt(static_cast<double>(image_k_blocks)));
    if (q_grid_w <= 0 || k_grid_w <= 0 ||
        q_grid_w * q_grid_w != image_q_blocks ||
        k_grid_w * k_grid_w != image_k_blocks ||
        q_grid_w != k_grid_w) {
        return;
    }

    const int grid_w = q_grid_w;
    const int grid_h = image_q_blocks / grid_w;
    const int center_y0 = (grid_h - 1) / 2;
    const int center_y1 = grid_h / 2;
    const int center_x0 = (grid_w - 1) / 2;
    const int center_x1 = grid_w / 2;
    for (int qb = 0; qb < n_q_blocks; ++qb) {
        const int q_head = qb / q_blocks_per_head;
        const int q_block_in_head = qb % q_blocks_per_head;
        if (q_block_in_head >= image_q_blocks) {
            continue;
        }
        const int kv_head = std::min(n_kv_heads - 1, q_head / gqa);
        for (int ky = center_y0; ky <= center_y1; ++ky) {
            for (int kx = center_x0; kx <= center_x1; ++kx) {
                const int k_block_in_head = ky * grid_w + kx;
                if (k_block_in_head >= image_k_blocks) {
                    continue;
                }
                const int kb = kv_head * k_blocks_per_head + k_block_in_head;
                mask[size_t(qb) * size_t(n_k_blocks) + size_t(kb)] = 0;
            }
        }
    }
}

__STATIC_INLINE__ uint64_t sd_fnv1a64_update(uint64_t hash, const void* data, size_t size) {
    const uint8_t* bytes = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < size; ++i) {
        hash ^= bytes[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

__STATIC_INLINE__ ggml_tensor* sd_backend_htp_get_packed_f16_weight(const ggml_tensor* original) {
#ifdef GGML_USE_HTP
    return ggml_backend_htp_get_packed_f16_weight(original);
#else
    SD_UNUSED(original);
    return nullptr;
#endif
}

__STATIC_INLINE__ void sd_backend_htp_register_packed_f16_weight(const ggml_tensor* original, ggml_tensor* packed) {
#ifdef GGML_USE_HTP
    ggml_backend_htp_register_packed_f16_weight(original, packed);
#else
    SD_UNUSED(original);
    SD_UNUSED(packed);
#endif
}

__STATIC_INLINE__ void sd_backend_htp_unregister_packed_f16_weight(const ggml_tensor* original) {
#ifdef GGML_USE_HTP
    ggml_backend_htp_unregister_packed_f16_weight(original);
#else
    SD_UNUSED(original);
#endif
}

__STATIC_INLINE__ void sd_backend_htp_set_n_threads(ggml_backend_t backend, int n_threads) {
#ifdef GGML_USE_HTP
    ggml_backend_htp_set_n_threads(backend, n_threads);
#else
    SD_UNUSED(backend);
    SD_UNUSED(n_threads);
#endif
}

__STATIC_INLINE__ void sd_backend_htp_set_hybrid_enabled(bool enabled) {
#ifdef GGML_USE_HTP
    ggml_backend_htp_set_hybrid_enabled(enabled);
#else
    SD_UNUSED(enabled);
#endif
}

// HTP residual-cache helpers are optional at link time. Keep the runner
// buildable for CPU/other backends while preserving the real calls when the
// HTP backend is present.
__STATIC_INLINE__ ggml_backend_htp_async_cache_slot* sd_backend_htp_async_cache_slot_create(
        ggml_backend_t backend,
        const char* cache_key,
        uint64_t generation,
        enum ggml_type type,
        const int64_t ne[GGML_MAX_DIMS],
        char* reason,
        size_t reason_capacity) {
#ifdef GGML_USE_HTP
    return ggml_backend_htp_async_cache_slot_create(
        backend, cache_key, generation, type, ne, reason, reason_capacity);
#else
    SD_UNUSED(backend); SD_UNUSED(cache_key); SD_UNUSED(generation); SD_UNUSED(type);
    SD_UNUSED(ne); SD_UNUSED(reason); SD_UNUSED(reason_capacity);
    return nullptr;
#endif
}

__STATIC_INLINE__ void sd_backend_htp_async_cache_slot_free(
        ggml_backend_htp_async_cache_slot* slot) {
#ifdef GGML_USE_HTP
    ggml_backend_htp_async_cache_slot_free(slot);
#else
    SD_UNUSED(slot);
#endif
}

__STATIC_INLINE__ ggml_tensor* sd_backend_htp_async_cache_slot_tensor(
        ggml_backend_htp_async_cache_slot* slot) {
#ifdef GGML_USE_HTP
    return ggml_backend_htp_async_cache_slot_tensor(slot);
#else
    SD_UNUSED(slot);
    return nullptr;
#endif
}

__STATIC_INLINE__ uint64_t sd_backend_htp_async_cache_slot_opencl_import_identity(
        const ggml_backend_htp_async_cache_slot* slot) {
#ifdef GGML_USE_HTP
    return ggml_backend_htp_async_cache_slot_opencl_import_identity(slot);
#else
    SD_UNUSED(slot);
    return 0;
#endif
}

__STATIC_INLINE__ bool sd_backend_htp_async_cache_slot_snapshot(
        const ggml_backend_htp_async_cache_slot* slot,
        ggml_htp_cache_slot_snapshot* snapshot) {
#ifdef GGML_USE_HTP
    return ggml_backend_htp_async_cache_slot_snapshot(slot, snapshot);
#else
    SD_UNUSED(slot);
    if (snapshot != nullptr) {
        *snapshot = {};
    }
    return false;
#endif
}

__STATIC_INLINE__ bool sd_backend_htp_set_graph_memory_ledger(
        ggml_backend_t backend,
        const ggml_htp_memory_ledger_snapshot* snapshot) {
#ifdef GGML_USE_HTP
    return ggml_backend_htp_set_graph_memory_ledger(backend, snapshot);
#else
    SD_UNUSED(backend); SD_UNUSED(snapshot);
    return false;
#endif
}

__STATIC_INLINE__ bool sd_backend_htp_get_lifetime_snapshot(
        ggml_backend_t backend,
        ggml_htp_lifetime_snapshot* snapshot) {
#ifdef GGML_USE_HTP
    return ggml_backend_htp_get_lifetime_snapshot(backend, snapshot);
#else
    SD_UNUSED(backend);
    if (snapshot != nullptr) {
        *snapshot = {};
    }
    return false;
#endif
}

__STATIC_INLINE__ size_t sd_backend_htp_opencl_retain_deferred_residual_inputs(
        ggml_cgraph* graph) {
#ifdef GGML_USE_HTP
    return ggml_backend_htp_opencl_retain_deferred_residual_inputs(graph);
#else
    SD_UNUSED(graph);
    return 0;
#endif
}

__STATIC_INLINE__ void sd_backend_htp_opencl_set_compute_arena_generation(
        ggml_backend_t backend, uint64_t generation) {
#ifdef GGML_USE_HTP
    ggml_backend_htp_opencl_set_compute_arena_generation(backend, generation);
#else
    SD_UNUSED(backend); SD_UNUSED(generation);
#endif
}

__STATIC_INLINE__ bool sd_backend_htp_opencl_arena_generation_pending(
        ggml_backend_t backend, uint64_t generation) {
#ifdef GGML_USE_HTP
    return ggml_backend_htp_opencl_arena_generation_pending(backend, generation);
#else
    SD_UNUSED(backend); SD_UNUSED(generation);
    return false;
#endif
}

__STATIC_INLINE__ bool sd_backend_htp_opencl_release_arena_generation(
        ggml_backend_t backend,
        uint64_t generation,
        bool wait,
        bool* released,
        char* reason,
        size_t reason_capacity) {
#ifdef GGML_USE_HTP
    return ggml_backend_htp_opencl_release_arena_generation(
        backend, generation, wait, released, reason, reason_capacity);
#else
    SD_UNUSED(backend); SD_UNUSED(generation); SD_UNUSED(wait);
    SD_UNUSED(reason); SD_UNUSED(reason_capacity);
    if (released != nullptr) {
        *released = true;
    }
    return true;
#endif
}

__STATIC_INLINE__ bool sd_backend_htp_opencl_bind_cache_input(
        ggml_backend_t backend,
        const ggml_tensor* input,
        const char* cache_key,
        uint64_t generation,
        char* reason,
        size_t reason_capacity) {
#ifdef GGML_USE_HTP
    return ggml_backend_htp_opencl_bind_cache_input(
        backend, input, cache_key, generation, reason, reason_capacity);
#else
    SD_UNUSED(backend); SD_UNUSED(input); SD_UNUSED(cache_key); SD_UNUSED(generation);
    SD_UNUSED(reason); SD_UNUSED(reason_capacity);
    return false;
#endif
}

__STATIC_INLINE__ bool sd_backend_htp_opencl_cache_query(
        ggml_backend_t backend,
        const char* cache_key,
        uint64_t generation,
        bool* ready,
        char* reason,
        size_t reason_capacity) {
#ifdef GGML_USE_HTP
    return ggml_backend_htp_opencl_cache_query(
        backend, cache_key, generation, ready, reason, reason_capacity);
#else
    SD_UNUSED(backend); SD_UNUSED(cache_key); SD_UNUSED(generation);
    SD_UNUSED(reason); SD_UNUSED(reason_capacity);
    if (ready != nullptr) {
        *ready = false;
    }
    return false;
#endif
}

__STATIC_INLINE__ bool sd_backend_htp_opencl_cache_wait(
        ggml_backend_t backend,
        const char* cache_key,
        uint64_t generation,
        bool* waited,
        char* reason,
        size_t reason_capacity) {
#ifdef GGML_USE_HTP
    return ggml_backend_htp_opencl_cache_wait(
        backend, cache_key, generation, waited, reason, reason_capacity);
#else
    SD_UNUSED(backend); SD_UNUSED(cache_key); SD_UNUSED(generation);
    SD_UNUSED(reason); SD_UNUSED(reason_capacity);
    if (waited != nullptr) {
        *waited = false;
    }
    return false;
#endif
}

__STATIC_INLINE__ int align_up_offset(int n, int multiple) {
    return (multiple - n % multiple) % multiple;
}

__STATIC_INLINE__ int align_up(int n, int multiple) {
    return n + align_up_offset(n, multiple);
}

__STATIC_INLINE__ void ggml_log_callback_default(ggml_log_level level, const char* text, void*) {
    switch (level) {
        case GGML_LOG_LEVEL_DEBUG:
            LOG_DEBUG(text);
            break;
        case GGML_LOG_LEVEL_INFO:
            LOG_INFO(text);
            break;
        case GGML_LOG_LEVEL_WARN:
            LOG_WARN(text);
            break;
        case GGML_LOG_LEVEL_ERROR:
            LOG_ERROR(text);
            break;
        default:
            LOG_DEBUG(text);
    }
}

#if GGML_MAX_NAME < 128
#warning "GGML_MAX_NAME < 128: name buffer is smaller than the sd.cpp default, but routing helpers remain usable"
#endif

// n-mode tensor-matrix product
// example: 2-mode product
// A: [ne03, k, ne01, ne00]
// B: k rows, m columns => [k, m]
// result is [ne03, m, ne01, ne00]
__STATIC_INLINE__ ggml_tensor* ggml_ext_mul_n_mode(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b, int mode = 0) {
    // reshape A
    // swap 0th and nth axis
    a           = ggml_cont(ctx, ggml_permute(ctx, a, mode, mode != 1 ? 1 : 0, mode != 2 ? 2 : 0, mode != 3 ? 3 : 0));
    int64_t ne1 = a->ne[1];
    int64_t ne2 = a->ne[2];
    int64_t ne3 = a->ne[3];
    // make 2D
    a = ggml_cont(ctx, ggml_reshape_2d(ctx, a, a->ne[0], (ne3 * ne2 * ne1)));

    ggml_tensor* result = ggml_cont(ctx, ggml_transpose(ctx, ggml_mul_mat(ctx, a, b)));

    // reshape output (same shape as a after permutation except first dim)
    result = ggml_reshape_4d(ctx, result, result->ne[0], ne1, ne2, ne3);
    // swap back 0th and nth axis
    result = ggml_permute(ctx, result, mode, mode != 1 ? 1 : 0, mode != 2 ? 2 : 0, mode != 3 ? 3 : 0);
    return result;
}

// Kronecker product
// [ne03,ne02,ne01,ne00] x [ne13,ne12,ne11,ne10] => [ne03*ne13,ne02*ne12,ne01*ne11,ne00*ne10]
__STATIC_INLINE__ ggml_tensor* ggml_ext_kronecker(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b) {
    return ggml_mul(ctx,
                    ggml_interpolate(ctx,
                                     a,
                                     a->ne[0] * b->ne[0],
                                     a->ne[1] * b->ne[1],
                                     a->ne[2] * b->ne[2],
                                     a->ne[3] * b->ne[3],
                                     GGML_SCALE_MODE_NEAREST),
                    b);
}

__STATIC_INLINE__ void ggml_ext_im_set_randn_f32(ggml_tensor* tensor, std::shared_ptr<RNG> rng) {
    uint32_t n                        = (uint32_t)ggml_nelements(tensor);
    std::vector<float> random_numbers = rng->randn(n);
    for (uint32_t i = 0; i < n; i++) {
        ggml_ext_im_set_f32_1d(tensor, i, random_numbers[i]);
    }
}

__STATIC_INLINE__ void ggml_ext_tensor_set_f32(ggml_tensor* tensor, float value, int64_t i0, int64_t i1 = 0, int64_t i2 = 0, int64_t i3 = 0) {
    GGML_ASSERT(tensor->nb[0] == sizeof(float));
    *(float*)((char*)(tensor->data) + i3 * tensor->nb[3] + i2 * tensor->nb[2] + i1 * tensor->nb[1] + i0 * tensor->nb[0]) = value;
}

__STATIC_INLINE__ float ggml_ext_tensor_get_f32(const ggml_tensor* tensor, int64_t i0, int64_t i1 = 0, int64_t i2 = 0, int64_t i3 = 0) {
    if (tensor->buffer != nullptr) {
        float value;
        ggml_backend_tensor_get(tensor, &value, i3 * tensor->nb[3] + i2 * tensor->nb[2] + i1 * tensor->nb[1] + i0 * tensor->nb[0], sizeof(float));
        return value;
    }
    GGML_ASSERT(tensor->nb[0] == sizeof(float));
    return *(float*)((char*)(tensor->data) + i3 * tensor->nb[3] + i2 * tensor->nb[2] + i1 * tensor->nb[1] + i0 * tensor->nb[0]);
}

__STATIC_INLINE__ int ggml_ext_tensor_get_i32(const ggml_tensor* tensor, int64_t i0, int64_t i1 = 0, int64_t i2 = 0, int64_t i3 = 0) {
    if (tensor->buffer != nullptr) {
        int value;
        ggml_backend_tensor_get(tensor, &value, i3 * tensor->nb[3] + i2 * tensor->nb[2] + i1 * tensor->nb[1] + i0 * tensor->nb[0], sizeof(int));
        return value;
    }
    GGML_ASSERT(tensor->nb[0] == sizeof(int));
    return *(int*)((char*)(tensor->data) + i3 * tensor->nb[3] + i2 * tensor->nb[2] + i1 * tensor->nb[1] + i0 * tensor->nb[0]);
}

__STATIC_INLINE__ ggml_fp16_t ggml_ext_tensor_get_f16(const ggml_tensor* tensor, int64_t i0, int64_t i1 = 0, int64_t i2 = 0, int64_t i3 = 0) {
    GGML_ASSERT(tensor->nb[0] == sizeof(ggml_fp16_t));
    return *(ggml_fp16_t*)((char*)(tensor->data) + i3 * tensor->nb[3] + i2 * tensor->nb[2] + i1 * tensor->nb[1] + i0 * tensor->nb[0]);
}

__STATIC_INLINE__ float sd_image_get_f32(sd_image_t image, int64_t iw, int64_t ih, int64_t ic, bool scale = true) {
    float value = *(image.data + ih * image.width * image.channel + iw * image.channel + ic);
    if (scale) {
        value /= 255.f;
    }
    return value;
}

__STATIC_INLINE__ void print_ggml_tensor(ggml_tensor* tensor, bool shape_only = false, const char* mark = "") {
    printf("%s (%s): shape(%zu, %zu, %zu, %zu)\n", mark, ggml_type_name(tensor->type), tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);
    fflush(stdout);
    if (shape_only) {
        return;
    }
    int range = 3;
    for (int i3 = 0; i3 < tensor->ne[3]; i3++) {
        if (i3 >= range && i3 + range < tensor->ne[3]) {
            continue;
        }
        for (int i2 = 0; i2 < tensor->ne[2]; i2++) {
            if (i2 >= range && i2 + range < tensor->ne[2]) {
                continue;
            }
            for (int i1 = 0; i1 < tensor->ne[1]; i1++) {
                if (i1 >= range && i1 + range < tensor->ne[1]) {
                    continue;
                }
                for (int i0 = 0; i0 < tensor->ne[0]; i0++) {
                    if (i0 >= range && i0 + range < tensor->ne[0]) {
                        continue;
                    }
                    if (tensor->type == GGML_TYPE_F32) {
                        printf("  [%d, %d, %d, %d] = %f\n", i3, i2, i1, i0, ggml_ext_tensor_get_f32(tensor, i0, i1, i2, i3));
                    } else if (tensor->type == GGML_TYPE_F16) {
                        printf("  [%d, %d, %d, %d] = %f\n", i3, i2, i1, i0, ggml_fp16_to_fp32(ggml_ext_tensor_get_f16(tensor, i0, i1, i2, i3)));
                    } else if (tensor->type == GGML_TYPE_I32) {
                        printf("  [%d, %d, %d, %d] = %i3\n", i3, i2, i1, i0, ggml_ext_tensor_get_i32(tensor, i0, i1, i2, i3));
                    }
                    fflush(stdout);
                }
            }
        }
    }
}

template <typename T>
__STATIC_INLINE__ void print_sd_tensor(const sd::Tensor<T>& tensor, bool shape_only = false, const char* mark = "") {
    printf("%s: shape(", mark);
    for (size_t i = 0; i < static_cast<size_t>(tensor.dim()); ++i) {
        printf("%s%lld", i == 0 ? "" : ", ", static_cast<long long>(tensor.shape()[i]));
    }
    printf(")\n");
    fflush(stdout);
    if (shape_only) {
        return;
    }
    if (tensor.empty()) {
        return;
    }
    int range                  = 3;
    std::vector<int64_t> shape = tensor.shape();
    while (shape.size() < 4) {
        shape.push_back(1);
    }
    for (int64_t i3 = 0; i3 < shape[3]; i3++) {
        if (i3 >= range && i3 + range < shape[3]) {
            continue;
        }
        for (int64_t i2 = 0; i2 < shape[2]; i2++) {
            if (i2 >= range && i2 + range < shape[2]) {
                continue;
            }
            for (int64_t i1 = 0; i1 < shape[1]; i1++) {
                if (i1 >= range && i1 + range < shape[1]) {
                    continue;
                }
                for (int64_t i0 = 0; i0 < shape[0]; i0++) {
                    if (i0 >= range && i0 + range < shape[0]) {
                        continue;
                    }
                    size_t offset = static_cast<size_t>(i0 + shape[0] * (i1 + shape[1] * (i2 + shape[2] * i3)));
                    printf("  [%lld, %lld, %lld, %lld] = ", static_cast<long long>(i3), static_cast<long long>(i2), static_cast<long long>(i1), static_cast<long long>(i0));
                    if constexpr (std::is_same_v<T, float>) {
                        printf("%f\n", tensor[static_cast<int64_t>(offset)]);
                    } else if constexpr (std::is_same_v<T, ggml_fp16_t>) {
                        printf("%f\n", ggml_fp16_to_fp32(tensor[static_cast<int64_t>(offset)]));
                    } else if constexpr (std::is_same_v<T, int32_t>) {
                        printf("%d\n", tensor[static_cast<int64_t>(offset)]);
                    } else if constexpr (std::is_same_v<T, int64_t>) {
                        printf("%lld\n", static_cast<long long>(tensor[static_cast<int64_t>(offset)]));
                    }
                    fflush(stdout);
                }
            }
        }
    }
}

__STATIC_INLINE__ void ggml_ext_tensor_iter(
    ggml_tensor* tensor,
    const std::function<void(ggml_tensor*, int64_t, int64_t, int64_t, int64_t)>& fn) {
    int64_t n0 = tensor->ne[0];
    int64_t n1 = tensor->ne[1];
    int64_t n2 = tensor->ne[2];
    int64_t n3 = tensor->ne[3];

    for (int64_t i3 = 0; i3 < n3; i3++) {
        for (int64_t i2 = 0; i2 < n2; i2++) {
            for (int64_t i1 = 0; i1 < n1; i1++) {
                for (int64_t i0 = 0; i0 < n0; i0++) {
                    fn(tensor, i0, i1, i2, i3);
                }
            }
        }
    }
}

__STATIC_INLINE__ void ggml_ext_tensor_iter(
    ggml_tensor* tensor,
    const std::function<void(ggml_tensor*, int64_t)>& fn) {
    int64_t n0 = tensor->ne[0];
    int64_t n1 = tensor->ne[1];
    int64_t n2 = tensor->ne[2];
    int64_t n3 = tensor->ne[3];

    for (int64_t i = 0; i < ggml_nelements(tensor); i++) {
        fn(tensor, i);
    }
}

__STATIC_INLINE__ void ggml_ext_tensor_diff(
    ggml_tensor* a,
    ggml_tensor* b,
    float gap = 0.1f) {
    GGML_ASSERT(ggml_nelements(a) == ggml_nelements(b));
    ggml_ext_tensor_iter(a, [&](ggml_tensor* a, int64_t i0, int64_t i1, int64_t i2, int64_t i3) {
        float a_value = ggml_ext_tensor_get_f32(a, i0, i1, i2, i3);
        float b_value = ggml_ext_tensor_get_f32(b, i0, i1, i2, i3);
        if (abs(a_value - b_value) > gap) {
            LOG_WARN("[%ld, %ld, %ld, %ld] %f %f", i3, i2, i1, i0, a_value, b_value);
        }
    });
}

__STATIC_INLINE__ ggml_tensor* load_tensor_from_file(ggml_context* ctx, const std::string& file_path) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        LOG_ERROR("failed to open '%s'", file_path.c_str());
        return nullptr;
    }
    int32_t n_dims;
    int32_t length;
    int32_t ttype;

    file.read(reinterpret_cast<char*>(&n_dims), sizeof(n_dims));
    file.read(reinterpret_cast<char*>(&length), sizeof(length));
    file.read(reinterpret_cast<char*>(&ttype), sizeof(ttype));

    LOG_DEBUG("load_tensor_from_file %d %d %d", n_dims, length, ttype);

    if (file.eof()) {
        LOG_ERROR("incomplete file '%s'", file_path.c_str());
        return nullptr;
    }

    int32_t nelements = 1;
    int32_t ne[4]     = {1, 1, 1, 1};
    for (int i = 0; i < n_dims; ++i) {
        file.read(reinterpret_cast<char*>(&ne[i]), sizeof(ne[i]));
        nelements *= ne[i];
    }
    std::string name(length, 0);
    file.read(&name[0], length);
    ggml_tensor* tensor = ggml_new_tensor_4d(ctx, (ggml_type)ttype, ne[0], ne[1], ne[2], ne[3]);
    const size_t bpe    = ggml_type_size(ggml_type(ttype));
    file.read(reinterpret_cast<char*>(tensor->data), ggml_nbytes(tensor));
    return tensor;
}

// __STATIC_INLINE__ void save_tensor_to_file(const std::string& file_name, ggml_tensor* tensor, const std::string & name) {
//     std::string file_name_ = file_name + ".tensor";
//     std::string name_ = name;
//     std::ofstream file("./" + file_name_, std::ios::binary);
//     file.write(reinterpret_cast<char*>(&tensor->n_dims), sizeof(tensor->n_dims));
//     int len = (int)name_.size();
//     file.write(reinterpret_cast<char*>(&len), sizeof(len));
//     int ttype = (int)tensor->type;
//     file.write(reinterpret_cast<char*>(&ttype), sizeof(ttype));
//     for (int i = 0; i < tensor->n_dims; ++i) {
//         int ne_ = (int) tensor->ne[i];
//         file.write(reinterpret_cast<char*>(&ne_), sizeof(ne_));
//     }
//     file.write(&name_[0], len);
//     char* data = nullptr;
//     file.write((char*)tensor->data, ggml_nbytes(tensor));
//     file.close();
// }

__STATIC_INLINE__ float sigmoid(float x) {
    return 1 / (1.0f + expf(-x));
}

// SPECIAL OPERATIONS WITH TENSORS

__STATIC_INLINE__ uint8_t* ggml_tensor_to_sd_image(ggml_tensor* input, uint8_t* image_data = nullptr) {
    int64_t width    = input->ne[0];
    int64_t height   = input->ne[1];
    int64_t channels = input->ne[2];
    GGML_ASSERT(channels == 3 && input->type == GGML_TYPE_F32);
    if (image_data == nullptr) {
        image_data = (uint8_t*)malloc(width * height * channels);
    }
    for (int iy = 0; iy < height; iy++) {
        for (int ix = 0; ix < width; ix++) {
            for (int k = 0; k < channels; k++) {
                float value                                               = ggml_ext_tensor_get_f32(input, ix, iy, k);
                *(image_data + iy * width * channels + ix * channels + k) = (uint8_t)(value * 255.0f);
            }
        }
    }
    return image_data;
}

__STATIC_INLINE__ uint8_t* ggml_tensor_to_sd_image(ggml_tensor* input, int idx, bool video = false) {
    int64_t width  = input->ne[0];
    int64_t height = input->ne[1];
    int64_t channels;
    if (video) {
        channels = input->ne[3];
    } else {
        channels = input->ne[2];
    }
    GGML_ASSERT(channels == 3 && input->type == GGML_TYPE_F32);
    uint8_t* image_data = (uint8_t*)malloc(width * height * channels);
    for (int ih = 0; ih < height; ih++) {
        for (int iw = 0; iw < width; iw++) {
            for (int ic = 0; ic < channels; ic++) {
                float value;
                if (video) {
                    value = ggml_ext_tensor_get_f32(input, iw, ih, idx, ic);
                } else {
                    value = ggml_ext_tensor_get_f32(input, iw, ih, ic, idx);
                }
                *(image_data + ih * width * channels + iw * channels + ic) = (uint8_t)(value * 255.0f);
            }
        }
    }
    return image_data;
}

__STATIC_INLINE__ void sd_image_to_ggml_tensor(sd_image_t image,
                                               ggml_tensor* tensor,
                                               bool scale = true) {
    GGML_ASSERT(image.width == tensor->ne[0]);
    GGML_ASSERT(image.height == tensor->ne[1]);
    GGML_ASSERT(image.channel == tensor->ne[2]);
    GGML_ASSERT(1 == tensor->ne[3]);
    GGML_ASSERT(tensor->type == GGML_TYPE_F32);
    ggml_ext_tensor_iter(tensor, [&](ggml_tensor* tensor, int64_t i0, int64_t i1, int64_t i2, int64_t i3) {
        float value = sd_image_get_f32(image, i0, i1, i2, scale);
        ggml_ext_tensor_set_f32(tensor, value, i0, i1, i2, i3);
    });
}

__STATIC_INLINE__ void ggml_ext_tensor_apply_mask(ggml_tensor* image_data,
                                                  ggml_tensor* mask,
                                                  ggml_tensor* output,
                                                  float masked_value = 0.5f) {
    int64_t width    = output->ne[0];
    int64_t height   = output->ne[1];
    int64_t channels = output->ne[2];
    float rescale_mx = 1.f * mask->ne[0] / output->ne[0];
    float rescale_my = 1.f * mask->ne[1] / output->ne[1];
    GGML_ASSERT(output->type == GGML_TYPE_F32);
    for (int ix = 0; ix < width; ix++) {
        for (int iy = 0; iy < height; iy++) {
            int mx  = (int)(ix * rescale_mx);
            int my  = (int)(iy * rescale_my);
            float m = ggml_ext_tensor_get_f32(mask, mx, my);
            m       = round(m);  // inpaint models need binary masks
            ggml_ext_tensor_set_f32(mask, m, mx, my);
            for (int k = 0; k < channels; k++) {
                float value = ggml_ext_tensor_get_f32(image_data, ix, iy, k);
                value       = (1 - m) * (value - masked_value) + masked_value;
                ggml_ext_tensor_set_f32(output, value, ix, iy, k);
            }
        }
    }
}

__STATIC_INLINE__ float ggml_ext_tensor_mean(ggml_tensor* src) {
    float mean        = 0.0f;
    int64_t nelements = ggml_nelements(src);
    float* data       = (float*)src->data;
    for (int i = 0; i < nelements; i++) {
        mean += data[i] / nelements * 1.0f;
    }
    return mean;
}

// a = a+b
__STATIC_INLINE__ void ggml_ext_tensor_add_inplace(ggml_tensor* a, ggml_tensor* b) {
    GGML_ASSERT(ggml_nelements(a) == ggml_nelements(b));
    int64_t nelements = ggml_nelements(a);
    float* vec_a      = (float*)a->data;
    float* vec_b      = (float*)b->data;
    for (int i = 0; i < nelements; i++) {
        vec_a[i] = vec_a[i] + vec_b[i];
    }
}

__STATIC_INLINE__ void ggml_ext_tensor_scale_inplace(ggml_tensor* src, float scale) {
    int64_t nelements = ggml_nelements(src);
    float* data       = (float*)src->data;
    for (int i = 0; i < nelements; i++) {
        data[i] = data[i] * scale;
    }
}

__STATIC_INLINE__ void ggml_ext_tensor_clamp_inplace(ggml_tensor* src, float min, float max) {
    int64_t nelements = ggml_nelements(src);
    float* data       = (float*)src->data;
    for (int i = 0; i < nelements; i++) {
        float val = data[i];
        data[i]   = val < min ? min : (val > max ? max : val);
    }
}

__STATIC_INLINE__ ggml_tensor* ggml_ext_tensor_concat(ggml_context* ctx,
                                                      ggml_tensor* a,
                                                      ggml_tensor* b,
                                                      int dim) {
    int64_t ne[GGML_MAX_DIMS];
    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
        if (d == dim) {
            ne[d] = a->ne[d] + b->ne[d];
            continue;
        }
        GGML_ASSERT(a->ne[d] == b->ne[d]);
        ne[d] = a->ne[d];
    }
    ggml_tensor* result = ggml_new_tensor(ctx, a->type, GGML_MAX_DIMS, ne);
    int64_t o[4]        = {0, 0, 0, 0};
    o[dim]              = a->ne[dim];

    float v;
    for (int i3 = 0; i3 < result->ne[3]; i3++) {
        for (int i2 = 0; i2 < result->ne[2]; i2++) {
            for (int i1 = 0; i1 < result->ne[1]; i1++) {
                for (int i0 = 0; i0 < result->ne[0]; i0++) {
                    if (i0 < a->ne[0] && i1 < a->ne[1] && i2 < a->ne[2] && i3 < a->ne[3]) {
                        v = ggml_ext_tensor_get_f32(a, i0, i1, i2, i3);
                    } else {
                        v = ggml_ext_tensor_get_f32(b, i0 - o[0], i1 - o[1], i2 - o[2], i3 - o[3]);
                    }

                    ggml_ext_tensor_set_f32(result, v, i0, i1, i2, i3);
                }
            }
        }
    }
    return result;
}

// convert values from [0, 1] to [-1, 1]
__STATIC_INLINE__ void scale_to_minus1_1(ggml_tensor* src) {
    int64_t nelements = ggml_nelements(src);
    float* data       = (float*)src->data;
    for (int i = 0; i < nelements; i++) {
        float val = data[i];
        data[i]   = val * 2.0f - 1.0f;
    }
}

// convert values from [-1, 1] to [0, 1]
__STATIC_INLINE__ void scale_to_0_1(ggml_tensor* src) {
    int64_t nelements = ggml_nelements(src);
    float* data       = (float*)src->data;
    for (int i = 0; i < nelements; i++) {
        float val = data[i];
        data[i]   = (val + 1.0f) * 0.5f;
    }
}

__STATIC_INLINE__ ggml_tensor* ggml_ext_cont(ggml_context* ctx,
                                             ggml_tensor* x) {
    if (ggml_is_contiguous(x)) {
        return x;
    }
    return ggml_cont(ctx, x);
}

// torch like permute
__STATIC_INLINE__ ggml_tensor* ggml_ext_torch_permute(ggml_context* ctx,
                                                      ggml_tensor* x,
                                                      int axis0,
                                                      int axis1,
                                                      int axis2,
                                                      int axis3) {
    int torch_axes[4] = {axis0, axis1, axis2, axis3};

    int ggml_axes[4] = {0};
    for (int i = 0; i < 4; ++i) {
        int found = 0;
        for (int j = 0; j < 4; ++j) {
            if (torch_axes[j] == i) {
                ggml_axes[i] = j;
                found        = 1;
                break;
            }
        }
        GGML_ASSERT(found && "Invalid permute input: must be a permutation of 0-3");
    }

    return ggml_permute(ctx, x, ggml_axes[0], ggml_axes[1], ggml_axes[2], ggml_axes[3]);
}

__STATIC_INLINE__ ggml_tensor* ggml_ext_slice(ggml_context* ctx,
                                              ggml_tensor* x,
                                              int dim,
                                              int64_t start,
                                              int64_t end,
                                              bool cont = true) {
    GGML_ASSERT(dim >= 0 && dim < 4);
    if (x->ne[dim] == 1) {
        return x;
    }
    while (start < 0) {
        start = x->ne[dim] + start;
    }
    while (end < 0) {
        end = x->ne[dim] + end;
    }
    GGML_ASSERT(end > start);
    GGML_ASSERT(start >= 0 && start < x->ne[dim]);
    GGML_ASSERT(end > start && end <= x->ne[dim]);

    int64_t slice_size  = end - start;
    int64_t slice_ne[4] = {x->ne[0], x->ne[1], x->ne[2], x->ne[3]};
    slice_ne[dim]       = slice_size;

    x = ggml_view_4d(ctx, x,
                     slice_ne[0], slice_ne[1], slice_ne[2], slice_ne[3],
                     x->nb[1], x->nb[2], x->nb[3], start * x->nb[dim]);

    if (cont) {
        x = ggml_cont(ctx, x);
    }

    return x;
}

// example: [N, 3*C, H, W] => ([N, C, H, W], [N, C, H, W], [N, C, H, W])
__STATIC_INLINE__ std::vector<ggml_tensor*> ggml_ext_chunk(ggml_context* ctx,
                                                           ggml_tensor* x,
                                                           int num,
                                                           int64_t dim,
                                                           bool cont = true) {
    GGML_ASSERT(dim >= 0 && dim < 4);
    GGML_ASSERT(x->ne[dim] % num == 0);

    std::vector<ggml_tensor*> chunks;
    int64_t chunk_size  = x->ne[dim] / num;
    int64_t stride      = chunk_size * x->nb[dim];
    int64_t chunk_ne[4] = {x->ne[0], x->ne[1], x->ne[2], x->ne[3]};
    chunk_ne[dim]       = chunk_size;
    for (int i = 0; i < num; i++) {
        auto chunk = ggml_view_4d(
            ctx, x,
            chunk_ne[0], chunk_ne[1], chunk_ne[2], chunk_ne[3],
            x->nb[1], x->nb[2], x->nb[3], stride * i);
        if (cont) {
            chunk = ggml_cont(ctx, chunk);
        }
        chunks.push_back(chunk);
    }

    return chunks;
}

__STATIC_INLINE__ ggml_tensor* ggml_ext_silu_act(ggml_context* ctx, ggml_tensor* x, bool gate_first = true) {
    // x: [ne3, ne2, ne1, ne0]
    // return: [ne3, ne2, ne1, ne0/2]

    auto x_vec = ggml_ext_chunk(ctx, x, 2, 0, false);
    ggml_tensor* gate;
    if (gate_first) {
        gate = x_vec[0];
        x    = x_vec[1];
    } else {
        x    = x_vec[0];
        gate = x_vec[1];
    }
    gate = ggml_cont(ctx, gate);
    gate = ggml_silu_inplace(ctx, gate);

    x = ggml_mul(ctx, x, gate);  // [ne3, ne2, ne1, ne0/2]

    return x;
}

typedef std::function<bool(ggml_tensor*, ggml_tensor*, bool)> on_tile_process;

__STATIC_INLINE__ void sd_tiling_calc_tiles(int& num_tiles_dim,
                                            float& tile_overlap_factor_dim,
                                            int small_dim,
                                            int tile_size,
                                            const float tile_overlap_factor,
                                            bool circular) {
    int tile_overlap     = static_cast<int>(tile_size * tile_overlap_factor);
    int non_tile_overlap = tile_size - tile_overlap;

    if (circular) {
        // circular means the last and first tile are overlapping (wraping around)
        num_tiles_dim = small_dim / non_tile_overlap;

        if (num_tiles_dim < 1) {
            num_tiles_dim = 1;
        }

        tile_overlap_factor_dim = (tile_size - small_dim / num_tiles_dim) / (float)tile_size;

        // if single tile and tile_overlap_factor is not 0, add one to ensure we have at least two overlapping tiles
        if (num_tiles_dim == 1 && tile_overlap_factor_dim > 0) {
            num_tiles_dim++;
            tile_overlap_factor_dim = 0.5;
        }

        return;
    }
    // else, non-circular means the last and first tile are not overlapping

    num_tiles_dim     = (small_dim - tile_overlap) / non_tile_overlap;
    int overshoot_dim = ((num_tiles_dim + 1) * non_tile_overlap + tile_overlap) % small_dim;

    if ((overshoot_dim != non_tile_overlap) && (overshoot_dim <= num_tiles_dim * (tile_size / 2 - tile_overlap))) {
        // if tiles don't fit perfectly using the desired overlap
        // and there is enough room to squeeze an extra tile without overlap becoming >0.5
        num_tiles_dim++;
    }

    tile_overlap_factor_dim = (float)(tile_size * num_tiles_dim - small_dim) / (float)(tile_size * (num_tiles_dim - 1));
    if (num_tiles_dim <= 2) {
        if (small_dim <= tile_size) {
            num_tiles_dim           = 1;
            tile_overlap_factor_dim = 0;
        } else {
            num_tiles_dim           = 2;
            tile_overlap_factor_dim = (2 * tile_size - small_dim) / (float)tile_size;
        }
    }
}

// Tiling

__STATIC_INLINE__ int64_t sd_tensor_plane_size(const sd::Tensor<float>& tensor) {
    GGML_ASSERT(tensor.dim() >= 2);
    return tensor.shape()[0] * tensor.shape()[1];
}

__STATIC_INLINE__ sd::Tensor<float> sd_tensor_split_2d(const sd::Tensor<float>& input, int width, int height, int x, int y) {
    GGML_ASSERT(input.dim() >= 4);
    std::vector<int64_t> output_shape = input.shape();
    output_shape[0]                   = width;
    output_shape[1]                   = height;
    sd::Tensor<float> output(std::move(output_shape));
    int64_t input_width  = input.shape()[0];
    int64_t input_height = input.shape()[1];
    int64_t input_plane  = sd_tensor_plane_size(input);
    int64_t output_plane = sd_tensor_plane_size(output);
    int64_t plane_count  = input.numel() / input_plane;
    for (int iy = 0; iy < height; iy++) {
        for (int ix = 0; ix < width; ix++) {
            int64_t src_xy = (ix + x) % input_width + input_width * ((iy + y) % input_height);
            int64_t dst_xy = ix + width * iy;
            for (int64_t plane = 0; plane < plane_count; ++plane) {
                output[plane * output_plane + dst_xy] = input[plane * input_plane + src_xy];
            }
        }
    }
    return output;
}

__STATIC_INLINE__ void sd_tensor_merge_2d(const sd::Tensor<float>& input,
                                          sd::Tensor<float>* output,
                                          int x,
                                          int y,
                                          int overlap_x,
                                          int overlap_y,
                                          bool circular_x,
                                          bool circular_y,
                                          int x_skip = 0,
                                          int y_skip = 0) {
    GGML_ASSERT(output != nullptr);
    int64_t width        = input.shape()[0];
    int64_t height       = input.shape()[1];
    int64_t img_width    = output->shape()[0];
    int64_t img_height   = output->shape()[1];
    int64_t input_plane  = sd_tensor_plane_size(input);
    int64_t output_plane = sd_tensor_plane_size(*output);
    int64_t plane_count  = input.numel() / input_plane;
    GGML_ASSERT(output->numel() / output_plane == plane_count);

    // unclamped -> expects x in the range [0-1]
    auto smootherstep_f32 = [](const float x) -> float {
        GGML_ASSERT(x >= 0.f && x <= 1.f);
        return x * x * x * (x * (6.0f * x - 15.0f) + 10.0f);
    };

    for (int iy = y_skip; iy < height; iy++) {
        for (int ix = x_skip; ix < width; ix++) {
            int64_t src_xy = ix + width * iy;
            int64_t ox     = (x + ix) % img_width;
            int64_t oy     = (y + iy) % img_height;
            int64_t dst_xy = ox + img_width * oy;
            for (int64_t plane = 0; plane < plane_count; ++plane) {
                float new_value = input[plane * input_plane + src_xy];
                if (overlap_x > 0 || overlap_y > 0) {
                    float old_value   = (*output)[plane * output_plane + dst_xy];
                    const float x_f_0 = (circular_x || (overlap_x > 0 && x > 0)) ? (ix - x_skip) / float(overlap_x) : 1.f;
                    const float x_f_1 = (circular_x || (overlap_x > 0 && x < (img_width - width))) ? (width - ix) / float(overlap_x) : 1.f;
                    const float y_f_0 = (circular_y || (overlap_y > 0 && y > 0)) ? (iy - y_skip) / float(overlap_y) : 1.f;
                    const float y_f_1 = (circular_y || (overlap_y > 0 && y < (img_height - height))) ? (height - iy) / float(overlap_y) : 1.f;
                    const float x_f   = std::min(std::min(x_f_0, x_f_1), 1.f);
                    const float y_f   = std::min(std::min(y_f_0, y_f_1), 1.f);
                    (*output)[plane * output_plane + dst_xy] =
                        old_value + new_value * smootherstep_f32(y_f) * smootherstep_f32(x_f);
                } else {
                    (*output)[plane * output_plane + dst_xy] = new_value;
                }
            }
        }
    }
}

template <typename Fn>
__STATIC_INLINE__ sd::Tensor<float> process_tiles_2d(const sd::Tensor<float>& input,
                                                     int output_width,
                                                     int output_height,
                                                     int scale,
                                                     int p_tile_size_x,
                                                     int p_tile_size_y,
                                                     float tile_overlap_factor,
                                                     bool circular_x,
                                                     bool circular_y,
                                                     Fn&& on_processing,
                                                     bool silent = false) {
    sd::Tensor<float> output;
    int input_width  = static_cast<int>(input.shape()[0]);
    int input_height = static_cast<int>(input.shape()[1]);

    GGML_ASSERT(((input_width / output_width) == (input_height / output_height)) &&
                ((output_width / input_width) == (output_height / input_height)));
    GGML_ASSERT(((input_width / output_width) == scale) ||
                ((output_width / input_width) == scale));

    int small_width  = output_width;
    int small_height = output_height;
    bool decode      = output_width > input_width;
    if (decode) {
        small_width  = input_width;
        small_height = input_height;
    }

    int num_tiles_x;
    float tile_overlap_factor_x;
    sd_tiling_calc_tiles(num_tiles_x, tile_overlap_factor_x, small_width, p_tile_size_x, tile_overlap_factor, circular_x);

    int num_tiles_y;
    float tile_overlap_factor_y;
    sd_tiling_calc_tiles(num_tiles_y, tile_overlap_factor_y, small_height, p_tile_size_y, tile_overlap_factor, circular_y);

    int tile_overlap_x     = static_cast<int32_t>(p_tile_size_x * tile_overlap_factor_x);
    int non_tile_overlap_x = p_tile_size_x - tile_overlap_x;
    int tile_overlap_y     = static_cast<int32_t>(p_tile_size_y * tile_overlap_factor_y);
    int non_tile_overlap_y = p_tile_size_y - tile_overlap_y;
    int tile_size_x        = p_tile_size_x < small_width ? p_tile_size_x : small_width;
    int tile_size_y        = p_tile_size_y < small_height ? p_tile_size_y : small_height;
    int input_tile_size_x  = tile_size_x;
    int input_tile_size_y  = tile_size_y;
    int output_tile_size_x = tile_size_x;
    int output_tile_size_y = tile_size_y;
    if (decode) {
        output_tile_size_x *= scale;
        output_tile_size_y *= scale;
    } else {
        input_tile_size_x *= scale;
        input_tile_size_y *= scale;
    }

    int num_tiles   = num_tiles_x * num_tiles_y;
    int tile_count  = 1;
    bool last_y     = false;
    bool last_x     = false;
    float last_time = 0.0f;
    if (!silent) {
        LOG_DEBUG("num tiles : %d, %d ", num_tiles_x, num_tiles_y);
        LOG_DEBUG("optimal overlap : %f, %f (targeting %f)", tile_overlap_factor_x, tile_overlap_factor_y, tile_overlap_factor);
        LOG_DEBUG("processing %i tiles", num_tiles);
        pretty_progress(0, num_tiles, 0.0f);
    }
    for (int y = 0; y < small_height && !last_y; y += non_tile_overlap_y) {
        int dy = 0;
        if (!circular_y && y + tile_size_y >= small_height) {
            int original_y = y;
            y              = small_height - tile_size_y;
            dy             = original_y - y;
            if (decode) {
                dy *= scale;
            }
            last_y = true;
        }
        for (int x = 0; x < small_width && !last_x; x += non_tile_overlap_x) {
            int dx = 0;
            if (!circular_x && x + tile_size_x >= small_width) {
                int original_x = x;
                x              = small_width - tile_size_x;
                dx             = original_x - x;
                if (decode) {
                    dx *= scale;
                }
                last_x = true;
            }

            int x_in  = decode ? x : scale * x;
            int y_in  = decode ? y : scale * y;
            int x_out = decode ? x * scale : x;
            int y_out = decode ? y * scale : y;

            int overlap_x_out = decode ? tile_overlap_x * scale : tile_overlap_x;
            int overlap_y_out = decode ? tile_overlap_y * scale : tile_overlap_y;

            int64_t t1       = ggml_time_ms();
            auto input_tile  = sd_tensor_split_2d(input, input_tile_size_x, input_tile_size_y, x_in, y_in);
            auto output_tile = on_processing(input_tile);
            if (output_tile.empty()) {
                return {};
            }
            GGML_ASSERT(output_tile.shape()[0] == output_tile_size_x && output_tile.shape()[1] == output_tile_size_y);
            if (output.empty()) {
                std::vector<int64_t> output_shape = output_tile.shape();
                output_shape[0]                   = output_width;
                output_shape[1]                   = output_height;
                output                            = sd::Tensor<float>::zeros(std::move(output_shape));
            }
            sd_tensor_merge_2d(output_tile, &output, x_out, y_out, overlap_x_out, overlap_y_out, circular_x, circular_y, dx, dy);

            if (!silent) {
                int64_t t2 = ggml_time_ms();
                last_time  = (t2 - t1) / 1000.0f;
                pretty_progress(tile_count, num_tiles, last_time);
            }
            tile_count++;
        }
        last_x = false;
    }
    if (!silent && tile_count < num_tiles) {
        pretty_progress(num_tiles, num_tiles, last_time);
    }
    if (output.empty()) {
        return {};
    }
    return output;
}

__STATIC_INLINE__ ggml_tensor* ggml_ext_group_norm_32(ggml_context* ctx,
                                                      ggml_tensor* a) {
    const float eps = 1e-6f;  // default eps parameter
    return ggml_group_norm(ctx, a, 32, eps);
}

__STATIC_INLINE__ bool ggml_ext_is_padded_1d(const ggml_tensor* x) {
    return x->nb[0] == ggml_type_size(x->type) &&
           x->nb[2] == x->nb[1] * x->ne[1] &&
           x->nb[3] == x->nb[2] * x->ne[2];
}

__STATIC_INLINE__ ggml_tensor* ggml_ext_scale(ggml_context* ctx,
                                              ggml_tensor* x,
                                              float factor,
                                              bool inplace = false) {
    if (!ggml_ext_is_padded_1d(x)) {
        x = ggml_cont(ctx, x);
    }
    if (inplace) {
        x = ggml_scale_inplace(ctx, x, factor);
    } else {
        x = ggml_scale(ctx, x, factor);
    }
    return x;
}

__STATIC_INLINE__ ggml_tensor* ggml_ext_gelu(ggml_context* ctx,
                                             ggml_tensor* x,
                                             bool inplace = false) {
    if (!ggml_is_contiguous(x)) {
        x = ggml_cont(ctx, x);
    }
    if (inplace) {
        x = ggml_gelu_inplace(ctx, x);
    } else {
        x = ggml_gelu(ctx, x);
    }
    return x;
}

__STATIC_INLINE__ ggml_tensor* ggml_ext_gelu_quick(ggml_context* ctx,
                                                   ggml_tensor* x,
                                                   bool inplace = false) {
    if (!ggml_is_contiguous(x)) {
        x = ggml_cont(ctx, x);
    }
    if (inplace) {
        x = ggml_gelu_quick_inplace(ctx, x);
    } else {
        x = ggml_gelu_quick(ctx, x);
    }
    return x;
}

__STATIC_INLINE__ ggml_tensor* ggml_ext_linear(ggml_context* ctx,
                                               ggml_tensor* x,
                                               ggml_tensor* w,
                                               ggml_tensor* b,
                                               bool force_prec_f32 = false,
                                               float scale         = 1.f) {
    if (scale != 1.f) {
        x = ggml_ext_scale(ctx, x, scale);
    }
    if (x->ne[2] * x->ne[3] > 1024) {
        // workaround: avoid ggml cuda error
        int64_t ne2 = x->ne[2];
        int64_t ne3 = x->ne[3];
        x           = ggml_reshape_2d(ctx, x, x->ne[0], x->ne[1] * x->ne[2] * x->ne[3]);
        x           = ggml_mul_mat(ctx, w, x);
        if (force_prec_f32) {
            ggml_mul_mat_set_prec(x, GGML_PREC_F32);
        }
        x = ggml_reshape_4d(ctx, x, x->ne[0], x->ne[1] / ne2 / ne3, ne2, ne3);
    } else {
        x = ggml_mul_mat(ctx, w, x);
        if (force_prec_f32) {
            ggml_mul_mat_set_prec(x, GGML_PREC_F32);
        }
    }
    if (scale != 1.f) {
        x = ggml_ext_scale(ctx, x, 1.f / scale);
    }
    if (b != nullptr) {
        x = ggml_add_inplace(ctx, x, b);
    }
    return x;
}

__STATIC_INLINE__ ggml_tensor* ggml_ext_pad_ext(ggml_context* ctx,
                                                ggml_tensor* x,
                                                int lp0,
                                                int rp0,
                                                int lp1,
                                                int rp1,
                                                int lp2,
                                                int rp2,
                                                int lp3,
                                                int rp3,
                                                bool circular_x = false,
                                                bool circular_y = false) {
    if (circular_x && circular_y) {
        return ggml_pad_ext_circular(ctx, x, lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3);
    }

    if (circular_x && (lp0 != 0 || rp0 != 0)) {
        x   = ggml_pad_ext_circular(ctx, x, lp0, rp0, 0, 0, 0, 0, 0, 0);
        lp0 = rp0 = 0;
    }
    if (circular_y && (lp1 != 0 || rp1 != 0)) {
        x   = ggml_pad_ext_circular(ctx, x, 0, 0, lp1, rp1, 0, 0, 0, 0);
        lp1 = rp1 = 0;
    }

    if (lp0 != 0 || rp0 != 0 || lp1 != 0 || rp1 != 0 || lp2 != 0 || rp2 != 0 || lp3 != 0 || rp3 != 0) {
        x = ggml_pad_ext(ctx, x, lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3);
    }
    return x;
}

__STATIC_INLINE__ ggml_tensor* ggml_ext_pad(ggml_context* ctx,
                                            ggml_tensor* x,
                                            int p0,
                                            int p1,
                                            int p2          = 0,
                                            int p3          = 0,
                                            bool circular_x = false,
                                            bool circular_y = false) {
    return ggml_ext_pad_ext(ctx, x, 0, p0, 0, p1, 0, p2, 0, p3, circular_x, circular_y);
}

// w: [OC，IC, KH, KW]
// x: [N, IC, IH, IW]
// b: [OC,]
// result: [N, OC, OH, OW]
__STATIC_INLINE__ ggml_tensor* ggml_ext_conv_2d(ggml_context* ctx,
                                                ggml_tensor* x,
                                                ggml_tensor* w,
                                                ggml_tensor* b,
                                                int s0          = 1,
                                                int s1          = 1,
                                                int p0          = 0,
                                                int p1          = 0,
                                                int d0          = 1,
                                                int d1          = 1,
                                                bool direct     = false,
                                                bool circular_x = false,
                                                bool circular_y = false,
                                                float scale     = 1.f) {
    if (scale != 1.f) {
        x = ggml_ext_scale(ctx, x, scale);
    }
    if (w->ne[2] != x->ne[2] && ggml_n_dims(w) == 2) {
        w = ggml_reshape_4d(ctx, w, 1, 1, w->ne[0], w->ne[1]);
    }

    if ((p0 != 0 || p1 != 0) && (circular_x || circular_y)) {
        x  = ggml_ext_pad_ext(ctx, x, p0, p0, p1, p1, 0, 0, 0, 0, circular_x, circular_y);
        p0 = 0;
        p1 = 0;
    }

    if (direct) {
        x = ggml_conv_2d_direct(ctx, w, x, s0, s1, p0, p1, d0, d1);
    } else {
        x = ggml_conv_2d(ctx, w, x, s0, s1, p0, p1, d0, d1);
    }
    if (scale != 1.f) {
        x = ggml_ext_scale(ctx, x, 1.f / scale);
    }
    if (b != nullptr) {
        b = ggml_reshape_4d(ctx, b, 1, 1, b->ne[0], 1);
        x = ggml_add_inplace(ctx, x, b);
    }
    return x;
}

// w: [OC，IC, KD, 1 * 1]
// x: [N, IC, IH, IW]
// b: [OC,]
// result: [N*OC, OD, OH, OW]
__STATIC_INLINE__ ggml_tensor* ggml_ext_conv_3d(ggml_context* ctx,
                                                ggml_tensor* x,
                                                ggml_tensor* w,
                                                ggml_tensor* b,
                                                int64_t IC,
                                                int s0              = 1,
                                                int s1              = 1,
                                                int s2              = 1,
                                                int p0              = 0,
                                                int p1              = 0,
                                                int p2              = 0,
                                                int d0              = 1,
                                                int d1              = 1,
                                                int d2              = 1,
                                                bool force_prec_f32 = false) {
    if (force_prec_f32) {
        ggml_tensor* im2col = ggml_im2col_3d(ctx, w, x, IC, s0, s1, s2, p0, p1, p2, d0, d1, d2, w->type);

        int64_t OC = w->ne[3] / IC;
        int64_t N  = x->ne[3] / IC;
        x          = ggml_mul_mat(ctx,
                                  ggml_reshape_2d(ctx, im2col, im2col->ne[0], im2col->ne[3] * im2col->ne[2] * im2col->ne[1]),
                                  ggml_reshape_2d(ctx, w, w->ne[0] * w->ne[1] * w->ne[2] * IC, OC));
        ggml_mul_mat_set_prec(x, GGML_PREC_F32);

        int64_t OD = im2col->ne[3] / N;
        x          = ggml_reshape_4d(ctx, x, im2col->ne[1] * im2col->ne[2], OD, N, OC);
        x          = ggml_cont(ctx, ggml_permute(ctx, x, 0, 1, 3, 2));
        x          = ggml_reshape_4d(ctx, x, im2col->ne[1], im2col->ne[2], OD, OC * N);
    } else {
        x = ggml_conv_3d(ctx, w, x, IC, s0, s1, s2, p0, p1, p2, d0, d1, d2);
    }

    if (b != nullptr) {
        b = ggml_reshape_4d(ctx, b, 1, 1, 1, b->ne[0]);  // [OC, 1, 1, 1]
        x = ggml_add_inplace(ctx, x, b);
    }
    return x;
}

// w: [OC，IC, KD, 1 * 1]
// x: [N, IC, ID, IH*IW]
// b: [OC,]
// result: [N, OC, OD, OH*OW]
__STATIC_INLINE__ ggml_tensor* ggml_ext_conv_3d_nx1x1(ggml_context* ctx,
                                                      ggml_tensor* x,
                                                      ggml_tensor* w,
                                                      ggml_tensor* b,
                                                      int s2 = 1,
                                                      int p2 = 1,
                                                      int d2 = 1) {
    x = ggml_conv_2d(ctx, w, x, 1, s2, 0, p2, 1, d2);  // [N, OC, T, OH * OW]
    if (b != nullptr) {
        b = ggml_reshape_4d(ctx, b, 1, 1, b->ne[0], 1);
        x = ggml_add(ctx, x, b);
    }
    return x;  // [N, OC, T, OH * OW]
}

// qkv: [N, L, 3*C]
// return: ([N, L, C], [N, L, C], [N, L, C])
__STATIC_INLINE__ std::vector<ggml_tensor*> split_qkv(ggml_context* ctx,
                                                      ggml_tensor* qkv) {
    qkv = ggml_reshape_4d(ctx, qkv, qkv->ne[0] / 3, 3, qkv->ne[1], qkv->ne[2]);  // [N, L, 3, C]
    qkv = ggml_cont(ctx, ggml_permute(ctx, qkv, 0, 3, 1, 2));                    // [3, N, L, C]

    int64_t offset = qkv->nb[2] * qkv->ne[2];
    auto q         = ggml_view_3d(ctx, qkv, qkv->ne[0], qkv->ne[1], qkv->ne[2], qkv->nb[1], qkv->nb[2], offset * 0);  // [N, L, C]
    auto k         = ggml_view_3d(ctx, qkv, qkv->ne[0], qkv->ne[1], qkv->ne[2], qkv->nb[1], qkv->nb[2], offset * 1);  // [N, L, C]
    auto v         = ggml_view_3d(ctx, qkv, qkv->ne[0], qkv->ne[1], qkv->ne[2], qkv->nb[1], qkv->nb[2], offset * 2);  // [N, L, C]
    return {q, k, v};
}

// qkv: [N, 3*C, H, W]
// return: ([N, C, H, W], [N, C, H, W], [N, C, H, W])
__STATIC_INLINE__ std::vector<ggml_tensor*> split_image_qkv(ggml_context* ctx,
                                                            ggml_tensor* qkv) {
    int64_t W   = qkv->ne[0];
    int64_t H   = qkv->ne[1];
    int64_t C   = qkv->ne[2] / 3;
    int64_t N   = qkv->ne[3];
    int64_t nb1 = qkv->nb[1];
    int64_t nb2 = qkv->nb[2];
    qkv         = ggml_reshape_4d(ctx, qkv, W * H, C, 3, N);                     // [N, 3, C, H*W]
    qkv         = ggml_cont(ctx, ggml_ext_torch_permute(ctx, qkv, 0, 1, 3, 2));  // [3, N, C, H*W]

    int64_t offset = qkv->nb[2] * qkv->ne[2];
    auto q         = ggml_view_4d(ctx, qkv, W, H, C, N, nb1, nb2, qkv->nb[3], offset * 0);  // [N, C, H, W]
    auto k         = ggml_view_4d(ctx, qkv, W, H, C, N, nb1, nb2, qkv->nb[3], offset * 1);  // [N, C, H, W]
    auto v         = ggml_view_4d(ctx, qkv, W, H, C, N, nb1, nb2, qkv->nb[3], offset * 2);  // [N, C, H, W]
    return {q, k, v};
}

__STATIC_INLINE__ ggml_tensor* ggml_ext_full(ggml_context* ctx,
                                             float value,
                                             int64_t ne0,
                                             int64_t ne1,
                                             int64_t ne2,
                                             int64_t ne3) {
    auto one = ggml_get_tensor(ctx, "ggml_runner_build_in_tensor:one");
    auto t   = ggml_ext_scale(ctx, one, value);             // [1,]
    t        = ggml_repeat_4d(ctx, t, ne0, ne1, ne2, ne3);  // [ne0, ne1, ne2, ne3]
    return t;
}

__STATIC_INLINE__ ggml_tensor* ggml_ext_zeros(ggml_context* ctx,
                                              int64_t ne0,
                                              int64_t ne1,
                                              int64_t ne2,
                                              int64_t ne3) {
    return ggml_ext_full(ctx, 0.f, ne0, ne1, ne2, ne3);
}

__STATIC_INLINE__ ggml_tensor* ggml_ext_zeros_like(ggml_context* ctx,
                                                   ggml_tensor* x) {
    return ggml_ext_zeros(ctx, x->ne[0], x->ne[1], x->ne[2], x->ne[3]);
}

__STATIC_INLINE__ ggml_tensor* ggml_ext_ones(ggml_context* ctx,
                                             int64_t ne0,
                                             int64_t ne1,
                                             int64_t ne2,
                                             int64_t ne3) {
    return ggml_ext_full(ctx, 1.f, ne0, ne1, ne2, ne3);
}

__STATIC_INLINE__ ggml_tensor* ggml_ext_ones_like(ggml_context* ctx,
                                                  ggml_tensor* x) {
    return ggml_ext_ones(ctx, x->ne[0], x->ne[1], x->ne[2], x->ne[3]);
}

__STATIC_INLINE__ ggml_tensor* ggml_ext_cast_f32(ggml_context* ctx, ggml_backend_t backend, ggml_tensor* a) {
    if (sd_backend_is(backend, "Vulkan")) {
        auto zero_index = ggml_get_tensor(ctx, "ggml_runner_build_in_tensor:zero_int");
        auto out        = ggml_reshape_1d(ctx, a, ggml_nelements(a));
        out             = ggml_get_rows(ctx, out, zero_index);
        out             = ggml_reshape(ctx, out, a);
        // auto out = ggml_cast(ctx, a, GGML_TYPE_F32);
        return out;
    } else {
        auto out         = ggml_reshape_2d(ctx, a, 1, ggml_nelements(a));
        ggml_tensor* one = ggml_ext_ones(ctx, 1, 1, 1, 1);  // [1,]
        if (ggml_is_transposed(out)) {
            out = ggml_mul_mat(ctx, one, out);
        } else {
            out = ggml_mul_mat(ctx, out, one);
        }
        out = ggml_reshape(ctx, out, a);
        return out;
    }
}

__STATIC_INLINE__ ggml_tensor* ggml_ext_block_sparse_mask_mean_topk_node(ggml_context* ctx,
                                                                         ggml_tensor* q,
                                                                         ggml_tensor* k,
                                                                         int block_q,
                                                                         int block_k,
                                                                         int top_k,
                                                                         bool causal = false);
__STATIC_INLINE__ ggml_tensor* ggml_ext_block_sparse_k_mean_node(ggml_context* ctx,
                                                                 ggml_tensor* k,
                                                                 int block_k);
__STATIC_INLINE__ ggml_tensor* ggml_ext_block_sparse_q_mean_node(ggml_context* ctx,
                                                                 ggml_tensor* q,
                                                                 int block_q);
__STATIC_INLINE__ ggml_tensor* ggml_ext_block_sparse_score_node(ggml_context* ctx,
                                                                ggml_tensor* q_mean,
                                                                ggml_tensor* k_mean,
                                                                int block_q,
                                                                int block_k,
                                                                bool causal);
__STATIC_INLINE__ ggml_tensor* ggml_ext_block_sparse_mask_from_score_node(ggml_context* ctx,
                                                                          ggml_tensor* score,
                                                                          int block_q,
                                                                          int block_k,
                                                                          bool causal,
                                                                          int protected_start = 0,
                                                                          int protected_len = 0,
                                                                          int n_heads = 1,
                                                                          int n_kv_heads = 1);
__STATIC_INLINE__ ggml_tensor* ggml_ext_block_sparse_index_from_score_node(ggml_context* ctx,
                                                                           ggml_tensor* score,
                                                                           int block_q,
                                                                           int block_k,
                                                                           bool causal,
                                                                           int protected_start = 0,
                                                                           int protected_len = 0,
                                                                           int n_heads = 1,
                                                                           int n_kv_heads = 1);
__STATIC_INLINE__ ggml_tensor* ggml_ext_block_sparse_index_32_from_16_score_node(ggml_context* ctx,
                                                                                 ggml_tensor* score,
                                                                                 bool causal,
                                                                                 int protected_start = 0,
                                                                                 int protected_len = 0,
                                                                                 int n_heads = 1,
                                                                                 int n_kv_heads = 1);
__STATIC_INLINE__ ggml_tensor* ggml_ext_block_sparse_full_keep_mask_node(ggml_context* ctx,
                                                                         int n_q_blocks,
                                                                         int n_k_blocks);
__STATIC_INLINE__ ggml_tensor* ggml_ext_block_sparse_full_mask_node(ggml_context* ctx,
                                                                    int n_q_blocks,
                                                                    int n_k_blocks);
__STATIC_INLINE__ ggml_tensor* ggml_ext_block_sparse_structural_index_32_node(ggml_context* ctx,
                                                                              int qo_len,
                                                                              int kv_len,
                                                                              int protected_start,
                                                                              int protected_len,
                                                                              int n_heads,
                                                                              int n_kv_heads,
                                                                              bool patch_halo,
                                                                              bool center_sink);
__STATIC_INLINE__ ggml_tensor* ggml_ext_block_sparse_direct_halo_index_32_node(ggml_context* ctx,
                                                                                int qo_len,
                                                                                int kv_len,
                                                                                int protected_start,
                                                                                int protected_len,
                                                                                int token_h,
                                                                                int token_w,
                                                                                int n_heads,
                                                                                int n_kv_heads,
                                                                                bool center_sink);
__STATIC_INLINE__ ggml_tensor* ggml_ext_attention_lse_node(ggml_context* ctx,
                                                           ggml_tensor* q,
                                                           ggml_tensor* k,
                                                           ggml_tensor* mask,
                                                           int n_heads,
                                                           float scale,
                                                           ggml_flash_attn_ext_mask_format mask_format =
                                                               GGML_FLASH_ATTN_EXT_MASK_DEFAULT);

__STATIC_INLINE__ void sd_htp_exact_rho_cache_cpu(
        ggml_tensor* dst, int ith, int nth, void* userdata) {
    SD_UNUSED(userdata);
    const ggml_tensor* dense = dst->src[0];
    const ggml_tensor* sparse = dst->src[1];
    GGML_ASSERT(dense != nullptr && sparse != nullptr);
    GGML_ASSERT(dense->type == GGML_TYPE_F32 && sparse->type == GGML_TYPE_F32);
    const int64_t head_dim = dense->ne[0] - 1;
    const int64_t rows = dense->ne[1] * dense->ne[2] * dense->ne[3];
    for (int64_t row = ith; row < rows; row += nth) {
        static_cast<float*>(dst->data)[row] = std::exp(
            static_cast<const float*>(sparse->data)[row * sparse->ne[0] + head_dim] -
            static_cast<const float*>(dense->data)[row * dense->ne[0] + head_dim]);
    }
}

__STATIC_INLINE__ void sd_htp_exact_residual_cache_cpu(
        ggml_tensor* dst, int ith, int nth, void* userdata) {
    SD_UNUSED(userdata);
    const ggml_tensor* dense = dst->src[0];
    const ggml_tensor* sparse = dst->src[1];
    const ggml_tensor* rho = dst->src[2];
    GGML_ASSERT(dense != nullptr && sparse != nullptr && rho != nullptr);
    GGML_ASSERT(dense->type == GGML_TYPE_F32 && sparse->type == GGML_TYPE_F32 &&
                rho->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16);
    const int64_t head_dim = dense->ne[0] - 1;
    const int64_t rows = dense->ne[1] * dense->ne[2] * dense->ne[3];
    for (int64_t row = ith; row < rows; row += nth) {
        const float scale = static_cast<const float*>(rho->data)[row];
        for (int64_t d = 0; d < head_dim; ++d) {
            const float value =
                static_cast<const float*>(dense->data)[row * dense->ne[0] + d] -
                static_cast<const float*>(sparse->data)[row * sparse->ne[0] + d] * scale;
            static_cast<ggml_fp16_t*>(dst->data)[row * head_dim + d] =
                ggml_fp32_to_fp16(value);
        }
    }
}

__STATIC_INLINE__ void sd_htp_fused_anchor_attention_cpu(
        ggml_tensor* dst, int ith, int nth, void* userdata) {
    SD_UNUSED(dst);
    SD_UNUSED(ith);
    SD_UNUSED(nth);
    SD_UNUSED(userdata);
    GGML_ABORT("htp_fused_anchor_attention was routed to the CPU");
}

__STATIC_INLINE__ bool sd_htp_opencl_residual_route_prerequisites() {
    const bool cache_prerequisites =
        sd_env_flag_enabled("SD_HTP_BLOCK_SPARSE_RESIDUAL_CACHE") &&
        sd_env_flag_enabled("SD_HTP_BLOCK_SPARSE_RESIDUAL_EXACT_RHO");
    if (sd_env_flag_enabled("SD_HTP_BLOCK_SPARSE_FUSED_ANCHOR")) {
        return cache_prerequisites &&
               sd_env_flag_enabled("SD_HTP_SPARSE_OPENCL_RESIDUAL");
    }
    return cache_prerequisites &&
           sd_env_flag_enabled("SD_HTP_SPARSE_OPENCL") &&
           sd_env_flag_enabled("SD_HTP_SPARSE_OPENCL_ASYNC") &&
           sd_env_flag_enabled("SD_HTP_SPARSE_OPENCL_EARLY_SUBMIT");
}

__STATIC_INLINE__ bool sd_htp_opencl_deferred_residual_enabled() {
    return sd_env_flag_enabled("SD_HTP_SPARSE_OPENCL_RESIDUAL") &&
           sd_env_flag_enabled_default(
               "SD_HTP_SPARSE_OPENCL_DEFER_RESIDUAL", true);
}

__STATIC_INLINE__ ggml_htp_opencl_lifetime_mode
sd_htp_opencl_lifetime_mode() {
    return ggml_htp_opencl_lifetime_mode_effective();
}

__STATIC_INLINE__ bool sd_htp_opencl_staging_slots(int* slots) {
    return ggml_htp_opencl_staging_slots_parse(
        std::getenv("SD_HTP_SPARSE_OPENCL_STAGING_SLOTS"), slots);
}

__STATIC_INLINE__ bool sd_htp_opencl_lifetime_mode_is_local() {
    const auto mode = sd_htp_opencl_lifetime_mode();
    return mode == GGML_HTP_OPENCL_LIFETIME_REUSE ||
           mode == GGML_HTP_OPENCL_LIFETIME_STRICT;
}

// OpenCL sparse is the only route whose sparse attention destination is
// produced externally.  Keep this predicate separate from the shared cache
// lifetime predicate so fused/CPU outputs remain real graph allocations.
__STATIC_INLINE__ bool sd_htp_opencl_sparse_local_lifetime_prerequisites() {
    int staging_slots = 0;
    return sd_htp_opencl_lifetime_mode_is_local() &&
           !sd_env_flag_enabled("SD_HTP_BLOCK_SPARSE_FUSED_ANCHOR") &&
           sd_htp_opencl_residual_route_prerequisites() &&
           sd_env_flag_enabled("SD_HTP_SPARSE_OPENCL") &&
           sd_env_flag_enabled("SD_HTP_SPARSE_OPENCL_RESIDUAL") &&
           sd_env_flag_enabled("SD_HTP_SPARSE_OPENCL_SUBMIT_THREAD") &&
           sd_env_flag_enabled("SD_HTP_SPARSE_OPENCL_DEFER_RESIDUAL") &&
           sd_htp_opencl_staging_slots(&staging_slots);
}

// All three anchor routes can use graph-order cache outputs and persistent
// cache bindings.  Only the route-specific producer ownership differs.
__STATIC_INLINE__ bool sd_htp_opencl_local_lifetime_prerequisites() {
    if (!sd_htp_opencl_lifetime_mode_is_local() ||
        !sd_env_flag_enabled("SD_HTP_BLOCK_SPARSE_RESIDUAL_CACHE") ||
        !sd_env_flag_enabled("SD_HTP_BLOCK_SPARSE_RESIDUAL_EXACT_RHO")) {
        return false;
    }
    if (sd_htp_opencl_sparse_local_lifetime_prerequisites()) {
        return true;
    }
    const bool fused = sd_env_flag_enabled("SD_HTP_BLOCK_SPARSE_FUSED_ANCHOR");
    const bool opencl_sparse = sd_env_flag_enabled("SD_HTP_SPARSE_OPENCL");
    const bool opencl_residual = sd_env_flag_enabled("SD_HTP_SPARSE_OPENCL_RESIDUAL");
    if (fused) {
        return !opencl_sparse && opencl_residual;
    }
    return !opencl_sparse && !opencl_residual;
}

__STATIC_INLINE__ bool sd_htp_opencl_background_residual_job_enabled() {
    const auto mode = sd_htp_opencl_lifetime_mode();
    if (mode == GGML_HTP_OPENCL_LIFETIME_INVALID ||
        !sd_htp_opencl_residual_route_prerequisites() ||
        !sd_htp_opencl_deferred_residual_enabled() ||
        !sd_env_flag_enabled("SD_HTP_SPARSE_OPENCL_SUBMIT_THREAD")) {
        return false;
    }
    if (mode == GGML_HTP_OPENCL_LIFETIME_DEFERRED) {
        return true;
    }
    return sd_htp_opencl_local_lifetime_prerequisites();
}

__STATIC_INLINE__ bool sd_htp_opencl_whole_arena_deferred_ownership() {
    if (!sd_htp_opencl_background_residual_job_enabled()) {
        return false;
    }
    return sd_htp_opencl_lifetime_mode() ==
           GGML_HTP_OPENCL_LIFETIME_DEFERRED;
}

__STATIC_INLINE__ bool sd_htp_opencl_persistent_cache_outputs_enabled() {
    return sd_htp_opencl_local_lifetime_prerequisites() ||
           sd_htp_opencl_whole_arena_deferred_ownership();
}

__STATIC_INLINE__ bool sd_htp_opencl_bind_cache_outputs_before_reserve() {
    return sd_htp_opencl_local_lifetime_prerequisites();
}

struct sd_htp_opencl_memory_plan {
    ggml_htp_opencl_lifetime_mode mode =
        GGML_HTP_OPENCL_LIFETIME_INVALID;
    size_t temporary_runtime_bytes = 0;
    size_t temporary_htp_bytes = 0;
    size_t temporary_qkv_bytes = 0;
    size_t host_cache_bytes = 0;
    size_t rpcmem_cache_logical_bytes = 0;
    size_t rpcmem_cache_allocation_bytes = 0;
    size_t sparse_destinations_in_gallocr = 0;
    size_t cache_outputs_in_gallocr = 0;
    size_t retained_roots = 0;
};

__STATIC_INLINE__ sd_htp_opencl_memory_plan sd_make_htp_opencl_memory_plan(
    ggml_htp_opencl_lifetime_mode mode,
    size_t temporary_runtime_bytes,
    size_t temporary_htp_bytes,
    size_t temporary_qkv_bytes,
    size_t host_cache_bytes,
    size_t rpcmem_cache_logical_bytes,
    size_t rpcmem_cache_allocation_bytes,
    size_t sparse_destination_count,
    size_t metadata_only_sparse_count,
    size_t cache_output_count,
    bool cache_outputs_prebound,
    size_t retained_roots) {
    sd_htp_opencl_memory_plan plan;
    plan.mode = mode;
    plan.temporary_runtime_bytes = temporary_runtime_bytes;
    plan.temporary_htp_bytes = temporary_htp_bytes;
    plan.temporary_qkv_bytes = temporary_qkv_bytes;
    plan.host_cache_bytes = host_cache_bytes;
    plan.rpcmem_cache_logical_bytes = rpcmem_cache_logical_bytes;
    plan.rpcmem_cache_allocation_bytes = rpcmem_cache_allocation_bytes;
    plan.sparse_destinations_in_gallocr =
        sparse_destination_count > metadata_only_sparse_count
            ? sparse_destination_count - metadata_only_sparse_count
            : 0;
    plan.cache_outputs_in_gallocr =
        cache_outputs_prebound ? 0 : cache_output_count;
    plan.retained_roots = retained_roots;
    return plan;
}

__STATIC_INLINE__ int sd_format_htp_opencl_memory_plan(
    char* buffer,
    size_t capacity,
    const sd_htp_opencl_memory_plan& plan) {
    if (buffer == nullptr || capacity == 0) {
        return -1;
    }
    return std::snprintf(
        buffer,
        capacity,
        "htp_opencl_memory_plan mode=%s "
        "temporary_runtime_bytes=%zu temporary_htp_bytes=%zu "
        "temporary_qkv_bytes=%zu host_cache_bytes=%zu "
        "rpcmem_cache_logical_bytes=%zu rpcmem_cache_allocation_bytes=%zu "
        "sparse_destinations_in_gallocr=%zu "
        "cache_outputs_in_gallocr=%zu retained_roots=%zu",
        ggml_htp_opencl_lifetime_mode_name(plan.mode),
        plan.temporary_runtime_bytes,
        plan.temporary_htp_bytes,
        plan.temporary_qkv_bytes,
        plan.host_cache_bytes,
        plan.rpcmem_cache_logical_bytes,
        plan.rpcmem_cache_allocation_bytes,
        plan.sparse_destinations_in_gallocr,
        plan.cache_outputs_in_gallocr,
        plan.retained_roots);
}

__STATIC_INLINE__ bool sd_htp_opencl_residual_route_has_deferred_job() {
    return sd_htp_opencl_background_residual_job_enabled();
}

__STATIC_INLINE__ bool sd_htp_release_compute_buffer_after_residual_anchor(
    int block_sparse_residual_mode,
    bool has_deferred_residual_job) {
    return block_sparse_residual_mode == SD_HTP_BLOCK_SPARSE_RESIDUAL_ANCHOR &&
           !has_deferred_residual_job;
}

enum sd_cache_storage_update_mode {
    SD_CACHE_STORAGE_MERGE = 0,
    SD_CACHE_STORAGE_APPEND = 1,
    SD_CACHE_STORAGE_REPLACE_SEGMENTS = 2,
};

struct sd_cache_storage_update_plan {
    sd_cache_storage_update_mode mode = SD_CACHE_STORAGE_MERGE;
    std::vector<size_t> replace_segment_indices;
};

__STATIC_INLINE__ sd_cache_storage_update_plan sd_plan_cache_storage_update(
    const std::vector<std::unordered_set<std::string>>& existing_segment_names,
    const std::unordered_set<std::string>& pending_names,
    bool has_keep_filter) {
    sd_cache_storage_update_plan plan;
    if (has_keep_filter || pending_names.empty()) {
        return plan;
    }
    for (size_t segment_idx = 0; segment_idx < existing_segment_names.size();
         ++segment_idx) {
        const auto& segment_names = existing_segment_names[segment_idx];
        size_t overlap_count = 0;
        for (const std::string& name : segment_names) {
            overlap_count += pending_names.count(name);
        }
        if (overlap_count == 0) {
            continue;
        }
        if (overlap_count != segment_names.size()) {
            return plan;
        }
        plan.replace_segment_indices.push_back(segment_idx);
    }
    plan.mode = plan.replace_segment_indices.empty()
                    ? SD_CACHE_STORAGE_APPEND
                    : SD_CACHE_STORAGE_REPLACE_SEGMENTS;
    return plan;
}

// q: [N, L_q, C(n_head*d_head)] or [N*n_head, L_q, d_head]
// k: [N, L_k, n_kv_head*d_head] or [N*n_kv_head, L_k, d_head]
// v: [N, L_k, n_kv_head*d_head] or [N, L_k, n_kv_head, d_head]
// mask: [N, L_q, L_k]
// return: [N, L_q, C]
__STATIC_INLINE__ ggml_tensor* ggml_ext_attention_ext(ggml_context* ctx,
                                                      ggml_backend_t backend,
                                                      ggml_tensor* q,
                                                      ggml_tensor* k,
                                                      ggml_tensor* v,
                                                      int64_t n_head,
                                                      ggml_tensor* mask = nullptr,
                                                      bool skip_reshape = false,
                                                      bool flash_attn   = false,
                                                      float kv_scale    = 1.0f,
                                                      bool block_sparse_attn = false,
                                                      int protected_start = 0,
                                                      int protected_len = 0,
                                                      int block_sparse_residual_mode = SD_HTP_BLOCK_SPARSE_RESIDUAL_OFF,
                                                      float block_sparse_residual_scale = 1.0f,
                                                      float block_sparse_residual_rho = 1.0f,
                                                      bool block_sparse_residual_exact_rho = false,
                                                      const char* block_sparse_residual_cache_name = nullptr,
                                                      ggml_tensor* block_sparse_cached_residual = nullptr,
                                                      const char* block_sparse_residual_rho_cache_name = nullptr,
                                                      ggml_tensor* block_sparse_cached_rho = nullptr,
                                                      const std::function<ggml_tensor*(const std::string&, ggml_tensor*)>* block_sparse_cache_tensor = nullptr,
                                                      int block_sparse_token_h = 0,
                                                      int block_sparse_token_w = 0,
                                                      bool block_sparse_force_full_keep = false) {  // avoid overflow
    auto packed_qkv_storage = [](ggml_tensor* tensor) -> ggml_tensor* {
        for (int depth = 0; tensor != nullptr && depth < 8; ++depth) {
            if (tensor->name[0] != '\0' &&
                (std::strcmp(tensor->name, "qkv_mul_rms_packed") == 0 ||
                 std::strcmp(tensor->name, "qkv_mul_rms_packed_dual") == 0 ||
                 std::strcmp(tensor->name, "qkv_mul_rms_packed_rope") == 0 ||
                 std::strcmp(tensor->name, "qkv_mul_rms_packed_dual_rope") == 0)) {
                return tensor;
            }
            tensor = tensor->view_src;
        }
        return nullptr;
    };
    ggml_tensor* packed_qkv = packed_qkv_storage(q);
    const bool has_packed_qkv_ancestor = packed_qkv != nullptr ||
                                         packed_qkv_storage(k) != nullptr ||
                                         packed_qkv_storage(v) != nullptr;
    const bool packed_layout = packed_qkv != nullptr && packed_qkv_storage(k) == packed_qkv &&
                               packed_qkv_storage(v) == packed_qkv &&
                               sd_env_flag_enabled_default("SD_HTP_PACKED_ATTENTION", true);
    if (sd_env_flag_enabled("SD_HTP_PACKED_FLUX2_TRACE") &&
        q->ne[0] == 128 && q->ne[1] == 768 && q->ne[2] == 24) {
        LOG_INFO("flux2_attention_layout packed=%d storage=%p qview=%p kview=%p vview=%p qtype=%s ktype=%s vtype=%s qviewsrc=%p",
                 packed_layout ? 1 : 0, (void *) packed_qkv, (void *) q, (void *) k, (void *) v,
                 ggml_type_name(q->type), ggml_type_name(k->type), ggml_type_name(v->type),
                 q->view_src);
    }
    int64_t L_q;
    int64_t L_k;
    int64_t C;
    int64_t N;
    int64_t d_head;
    int64_t n_kv_head;
    ggml_tensor* q_htp_flash = nullptr;
    ggml_tensor* k_htp_flash_base = nullptr;
    ggml_tensor* v_htp_flash_base = nullptr;
    const bool keep_for_htp_hybrid = sd_env_flag_enabled("SD_HTP_OPS") &&
                                     sd_env_flag_enabled("SD_HTP_COMPUTE");
    if (!skip_reshape && !packed_layout) {
        L_q       = q->ne[1];
        L_k       = k->ne[1];
        C         = q->ne[0];
        N         = q->ne[2];
        d_head    = C / n_head;
        n_kv_head = k->ne[0] / d_head;

        ggml_tensor* q_seq_major = ggml_reshape_4d(ctx, q, d_head, n_head, L_q, N);       // [N, L_q, n_head, d_head]
        ggml_tensor* k_seq_major = ggml_reshape_4d(ctx, k, d_head, n_kv_head, L_k, N);    // [N, L_k, n_kv_head, d_head]
        ggml_tensor* v_seq_major = ggml_reshape_4d(ctx, v, d_head, n_kv_head, L_k, N);    // [N, L_k, n_kv_head, d_head]

        if (N == 1) {
            ggml_tensor* q3 = ggml_reshape_3d(ctx, q, d_head, n_head, L_q);
            q_htp_flash = ggml_permute(ctx, q3, 0, 2, 1, 3);  // logical [d_head, L_q, n_head], backing [L_q, n_head, d_head]
            k_htp_flash_base = ggml_reshape_3d(ctx, k, d_head, n_kv_head, L_k);  // backing [L_k, n_kv_head, d_head]
            v_htp_flash_base = ggml_reshape_3d(ctx, v, d_head, n_kv_head, L_k);  // backing [L_k, n_kv_head, d_head]
        }

        q = q_seq_major;
        q = ggml_ext_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));  // [N, n_head, L_q, d_head]
        q = ggml_reshape_3d(ctx, q, d_head, L_q, n_head * N);      // [N * n_head, L_q, d_head]

        k = k_seq_major;
        k = ggml_ext_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));  // [N, n_kv_head, L_k, d_head]
        k = ggml_reshape_3d(ctx, k, d_head, L_k, n_kv_head * N);   // [N * n_kv_head, L_k, d_head]

        v = v_seq_major;
    } else {
        bool scalar_layout_resolved = false;
        if (!packed_layout && q->ne[3] == 1 && k->ne[3] == 1 && v->ne[3] == 1 &&
            q->ne[0] == k->ne[0] && q->ne[0] == v->ne[0]) {
            const bool q_seq_head = q->ne[2] == n_head && q->ne[1] != n_head;
            const bool q_head_seq = q->ne[1] == n_head && q->ne[2] != n_head;
            if (q_seq_head != q_head_seq) {
                const int64_t q_tokens = q_seq_head ? q->ne[1] : q->ne[2];
                struct kv_layout_candidate {
                    int64_t tokens;
                    int64_t heads;
                };
                const kv_layout_candidate k_candidates[] = {
                    {k->ne[1], k->ne[2]},
                    {k->ne[2], k->ne[1]},
                };
                const kv_layout_candidate v_candidates[] = {
                    {v->ne[1], v->ne[2]},
                    {v->ne[2], v->ne[1]},
                };
                int matches = 0;
                kv_layout_candidate match = {};
                for (const auto& k_candidate : k_candidates) {
                    for (const auto& v_candidate : v_candidates) {
                        if (k_candidate.tokens == v_candidate.tokens &&
                            k_candidate.heads == v_candidate.heads &&
                            k_candidate.tokens > 0 && k_candidate.heads > 0 &&
                            k_candidate.heads <= n_head &&
                            n_head % k_candidate.heads == 0) {
                            if (matches == 0 || match.tokens != k_candidate.tokens ||
                                match.heads != k_candidate.heads) {
                                match = k_candidate;
                                ++matches;
                            }
                        }
                    }
                }
                if (matches == 1) {
                    L_q = q_tokens;
                    L_k = match.tokens;
                    d_head = q->ne[0];
                    N = 1;
                    n_kv_head = match.heads;
                    scalar_layout_resolved = true;
                }
            }
        }
        if (!scalar_layout_resolved) {
            L_q       = q->ne[1];
            L_k       = k->ne[1];
            d_head    = v->ne[0];
            N         = v->ne[3];
            n_kv_head = k->ne[2] / N;
        }
        C         = d_head * n_head;

        if (skip_reshape && !packed_layout && N == 1 && keep_for_htp_hybrid &&
            sd_env_flag_enabled("SD_HTP_ATTENTION")) {
            q_htp_flash = q;
            k_htp_flash_base = k;
            v_htp_flash_base = v;
        }
    }

    float scale = (1.0f / sqrt((float)d_head));

    int kv_pad       = 0;
    ggml_tensor* kqv = nullptr;
    const bool use_htp_flash_layout = keep_for_htp_hybrid &&
                                      sd_env_flag_enabled("SD_HTP_ATTENTION") &&
                                      N == 1 &&
                                      q_htp_flash != nullptr &&
                                      k_htp_flash_base != nullptr &&
                                      v_htp_flash_base != nullptr;

    auto dense_mask_to_kq_layout = [&](ggml_tensor* mask_in) -> ggml_tensor* {
        if (mask_in == nullptr) {
            return nullptr;
        }
        if (mask_in->ne[0] == L_k && mask_in->ne[1] == L_q) {
            return mask_in;
        }
        if (mask_in->ne[0] == L_q && mask_in->ne[1] == L_k) {
            return ggml_transpose(ctx, mask_in);
        }
        return ggml_transpose(ctx, mask_in);
    };

    auto build_kqv = [&](ggml_tensor* q_in, ggml_tensor* k_in, ggml_tensor* v_in, ggml_tensor* mask_in, bool with_lse = false) -> ggml_tensor* {
        // htp_attention_layout removed - pack/unpack not used with diffusion_flash_attn=false
        if (kv_pad != 0) {
            k_in = ggml_pad(ctx, k_in, 0, kv_pad, 0, 0);
        }
        if (kv_scale != 1.0f) {
            k_in = ggml_ext_scale(ctx, k_in, kv_scale);
        }
        k_in = ggml_cast(ctx, k_in, GGML_TYPE_F16);

        v_in = ggml_ext_cont(ctx, ggml_permute(ctx, v_in, 0, 2, 1, 3));
        v_in = ggml_reshape_3d(ctx, v_in, d_head, L_k, n_kv_head * N);
        if (kv_pad != 0) {
            v_in = ggml_pad(ctx, v_in, 0, kv_pad, 0, 0);
        }
        if (kv_scale != 1.0f) {
            v_in = ggml_ext_scale(ctx, v_in, kv_scale);
        }
        v_in = ggml_cast(ctx, v_in, GGML_TYPE_F16);

        const bool block_sparse_mask = mask_in != nullptr && mask_in->type == GGML_TYPE_I32;
        if (mask_in != nullptr && !block_sparse_mask) {
            mask_in = dense_mask_to_kq_layout(mask_in);
            if (mask_in->type != GGML_TYPE_F16) {
                mask_in = ggml_cast(ctx, mask_in, GGML_TYPE_F16);
            }
        } else {
            if (mask_in == nullptr && (kv_pad > 0 || keep_for_htp_hybrid)) {
                mask_in = ggml_ext_zeros(ctx, L_k, L_q, 1, 1);
            }
            if (kv_pad > 0) {
                auto pad_tensor = ggml_ext_full(ctx, -INFINITY, kv_pad, L_q, 1, 1);
                mask_in         = ggml_concat(ctx, mask_in, pad_tensor, 0);
            }
        }

        if (mask_in != nullptr && !block_sparse_mask && mask_in->type != GGML_TYPE_F16) {
            // the need for padding got removed in ggml 4767bda
            // ensure we can still use the old version for now
#ifdef GGML_KQ_MASK_PAD
            int mask_pad = 0;
            if (mask_in->ne[1] % GGML_KQ_MASK_PAD != 0) {
                mask_pad = GGML_PAD(L_q, GGML_KQ_MASK_PAD) - mask_in->ne[1];
            }
            if (mask_pad > 0) {
                mask_in = ggml_pad(ctx, mask_in, 0, mask_pad, 0, 0);
            }
#endif
            if (mask_in->type != GGML_TYPE_F16) {
                mask_in = ggml_cast(ctx, mask_in, GGML_TYPE_F16);
            }
        }

        auto out = with_lse ? ggml_flash_attn_ext_with_lse(ctx, q_in, k_in, v_in, mask_in, scale / kv_scale, 0, 0)
                            : ggml_flash_attn_ext(ctx, q_in, k_in, v_in, mask_in, scale / kv_scale, 0, 0);
        ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
        if (!with_lse && kv_scale != 1.0f) {
            out = ggml_ext_scale(ctx, out, 1.0f / kv_scale);
        }
        return out;
    };

    auto build_packed_kqv = [&](ggml_tensor* mask_in, bool with_lse = false) -> ggml_tensor* {
        if (!packed_layout || q->type != GGML_TYPE_F16 || k->type != GGML_TYPE_F16 || v->type != GGML_TYPE_F16) {
            return nullptr;
        }
        if (mask_in != nullptr && mask_in->type != GGML_TYPE_I32) {
            return nullptr;
        }
        ggml_tensor* out = with_lse ? ggml_flash_attn_ext_with_lse(ctx, q, k, v, mask_in, scale, 0, 0)
                                    : ggml_flash_attn_ext(ctx, q, k, v, mask_in, scale, 0, 0);
        ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
        ggml_set_name(out, "packed_sparse_attention");
        return out;
    };

    struct htp_attention_inputs {
        ggml_tensor* q = nullptr;
        ggml_tensor* k = nullptr;
        ggml_tensor* v = nullptr;
        ggml_tensor* dense_mask = nullptr;
    };

    auto canonical_htp_token_major_base = [&](ggml_tensor* tensor,
                                               int64_t tokens,
                                               int64_t heads,
                                               ggml_type type) -> ggml_tensor* {
        if (tensor == nullptr || tensor->type != type || tensor->ne[3] != 1 ||
            tensor->ne[0] != d_head || tokens <= 0 || heads <= 0) {
            return nullptr;
        }
        const bool seq_head = tensor->ne[1] == tokens && tensor->ne[2] == heads;
        const bool head_seq = tensor->ne[1] == heads && tensor->ne[2] == tokens;
        if (seq_head == head_seq) {
            return nullptr;
        }
        if (head_seq) {
            return ggml_ext_cont(ctx, tensor);
        }
        return ggml_ext_cont(ctx, ggml_permute(ctx, tensor, 0, 2, 1, 3));
    };

    auto canonical_htp_qk = [&](ggml_tensor* tensor,
                                 int64_t tokens,
                                 int64_t heads,
                                 ggml_type type) -> ggml_tensor* {
        ggml_tensor* base = canonical_htp_token_major_base(tensor, tokens, heads, type);
        return base != nullptr ? ggml_permute(ctx, base, 0, 2, 1, 3) : nullptr;
    };

    auto canonical_htp_v = [&](ggml_tensor* tensor,
                                int64_t tokens,
                                int64_t heads,
                                ggml_type type) -> ggml_tensor* {
        return canonical_htp_token_major_base(tensor, tokens, heads, type);
    };

    auto prepare_htp_attention_inputs = [&](ggml_tensor* dense_mask_in,
                                            bool prepare_dense_mask,
                                            bool scale_v_for_legacy_output = true) -> htp_attention_inputs {
        htp_attention_inputs inputs;
        if (!use_htp_flash_layout) {
            return inputs;
        }

        ggml_tensor* q_in = canonical_htp_qk(
            q_htp_flash, L_q, n_head, GGML_TYPE_F32);
        if (q_in == nullptr) {
            return inputs;
        }
        if (kv_scale != 1.0f) {
            // The HTP dense kernel always applies 1/sqrt(head_dim) internally.
            // Balance the F16 K scaling on F32 Q instead of relying on the
            // ggml flash-attention scale parameter, which is not in its RPC ABI.
            // Scale the token-major base so materialization does not turn the
            // permuted attention view back into head-major backing storage.
            ggml_tensor* q_base = ggml_permute(ctx, q_in, 0, 2, 1, 3);
            q_base = ggml_ext_scale(ctx, q_base, 1.0f / kv_scale);
            q_in = ggml_permute(ctx, q_base, 0, 2, 1, 3);
        }

        ggml_tensor* k_base = k_htp_flash_base;
        ggml_tensor* v_base = v_htp_flash_base;
        if (kv_pad != 0) {
            k_base = ggml_pad(ctx, k_base, 0, 0, kv_pad, 0);
            v_base = ggml_pad(ctx, v_base, 0, 0, kv_pad, 0);
        }
        if (kv_scale != 1.0f) {
            k_base = ggml_ext_scale(ctx, k_base, kv_scale);
            if (scale_v_for_legacy_output) {
                v_base = ggml_ext_scale(ctx, v_base, kv_scale);
            }
        }
        if (k_base->type != GGML_TYPE_F16) {
            k_base = ggml_cast(ctx, k_base, GGML_TYPE_F16);
        }
        if (v_base->type != GGML_TYPE_F16) {
            v_base = ggml_cast(ctx, v_base, GGML_TYPE_F16);
        }

        ggml_tensor* k_in = canonical_htp_qk(
            k_base, L_k, n_kv_head, GGML_TYPE_F16);
        ggml_tensor* v_in = canonical_htp_v(
            v_base, L_k, n_kv_head, GGML_TYPE_F16);
        if (k_in == nullptr || v_in == nullptr) {
            return inputs;
        }

        if (prepare_dense_mask) {
            const int htp_dense_mask_pad =
                kv_pad == 0 ? (GGML_PAD(L_k, 64) - static_cast<int>(L_k)) : 0;
            if (dense_mask_in != nullptr) {
                dense_mask_in = dense_mask_to_kq_layout(dense_mask_in);
            }
            if (dense_mask_in == nullptr &&
                (kv_pad > 0 || htp_dense_mask_pad > 0 || keep_for_htp_hybrid)) {
                dense_mask_in = ggml_ext_zeros(ctx, L_k, L_q, 1, 1);
            }
            const int mask_pad = kv_pad > 0 ? kv_pad : htp_dense_mask_pad;
            if (dense_mask_in != nullptr && mask_pad > 0) {
                auto pad_tensor = ggml_ext_full(ctx, -INFINITY, mask_pad, L_q, 1, 1);
                dense_mask_in   = ggml_concat(ctx, dense_mask_in, pad_tensor, 0);
            }
            if (dense_mask_in != nullptr && dense_mask_in->type != GGML_TYPE_F16) {
                dense_mask_in = ggml_cast(ctx, dense_mask_in, GGML_TYPE_F16);
            }
        }

        inputs.q = q_in;
        inputs.k = k_in;
        inputs.v = v_in;
        inputs.dense_mask = dense_mask_in;
        return inputs;
    };

    auto build_htp_kqv = [&](ggml_tensor* mask_in, bool with_lse = false) -> ggml_tensor* {
        const bool block_sparse_mask = mask_in != nullptr && mask_in->type == GGML_TYPE_I32;
        // LSE producers share their O slabs with residual cache consumers,
        // which require final-scale values just like packed attention outputs.
        const htp_attention_inputs inputs =
            prepare_htp_attention_inputs(block_sparse_mask ? nullptr : mask_in,
                                         !block_sparse_mask,
                                         !with_lse);
        if (inputs.q == nullptr || inputs.k == nullptr || inputs.v == nullptr) {
            return nullptr;
        }
        ggml_tensor* attention_mask = block_sparse_mask ? mask_in : inputs.dense_mask;

        auto out = with_lse ? ggml_flash_attn_ext_with_lse(ctx, inputs.q, inputs.k, inputs.v, attention_mask, scale, 0, 0)
                            : ggml_flash_attn_ext(ctx, inputs.q, inputs.k, inputs.v, attention_mask, scale, 0, 0);
        ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
        if (!with_lse && kv_scale != 1.0f) {
            out = ggml_ext_scale(ctx, out, 1.0f / kv_scale);
        }
        return out;
    };

    auto flash_attn_o_view = [&](ggml_tensor* packed,
                                 bool with_lse = false,
                                 bool output_already_final_scale = false) -> ggml_tensor* {
        GGML_ASSERT(packed != nullptr);
        GGML_ASSERT(!with_lse || packed->ne[0] == d_head + 1);
        ggml_tensor* out = ggml_view_3d(ctx, packed, d_head, n_head, L_q, packed->nb[1], packed->nb[2], 0);
        // Packed QKV producers emit unscaled K/V, so their attention output
        // must not receive the non-packed path's inverse V-scale compensation.
        if (with_lse && kv_scale != 1.0f && !packed_layout &&
            !output_already_final_scale) {
            out = ggml_ext_scale(ctx, ggml_cont(ctx, out), 1.0f / kv_scale);
        }
        return out;
    };

    auto flash_attn_lse_view = [&](ggml_tensor* packed) -> ggml_tensor* {
        GGML_ASSERT(packed != nullptr);
        GGML_ASSERT(packed->ne[0] == d_head + 1);
        return ggml_view_3d(ctx, packed, 1, n_head, L_q, packed->nb[1], packed->nb[2], d_head * packed->nb[0]);
    };

    if (flash_attn) {
        // LOG_DEBUG("attention_ext L_q:%d L_k:%d n_head:%d C:%d d_head:%d N:%d", L_q, L_k, n_head, C, d_head, N);
        bool can_use_flash_attn = true;
        if (mask != nullptr) {
            // TODO: figure out if we can bend t5 to work too
            can_use_flash_attn = can_use_flash_attn && mask->ne[3] == 1;
        }

        const bool force_full_keep = block_sparse_force_full_keep ||
                                     sd_env_flag_enabled("SD_HTP_BLOCK_SPARSE_FORCE_FULL_KEEP");
        const bool block_sparse_host_index =
            sd_env_flag_enabled_default("SD_HTP_BLOCK_SPARSE_HOST_INDEX", true);
        if (can_use_flash_attn && mask == nullptr && block_sparse_attn) {
            const int block_q = SD_HTP_BLOCK_SPARSE_MASK_BLOCK;
            const int block_k = SD_HTP_BLOCK_SPARSE_MASK_BLOCK;
            const int n_k_blocks = n_kv_head * static_cast<int>((L_k + block_k - 1) / block_k);
            const int n_q_blocks = n_head * static_cast<int>((L_q + block_q - 1) / block_q);
            ggml_tensor* block_mask = nullptr;
            ggml_flash_attn_ext_mask_format block_mask_format = GGML_FLASH_ATTN_EXT_MASK_DEFAULT;
            const float block_sparse_keep_ratio = sd_htp_block_sparse_keep_ratio();
            const bool block_sparse_patch_halo = sd_env_flag_enabled("SD_HTP_BLOCK_SPARSE_PATCH_HALO");
            const bool block_sparse_center_sink = sd_env_flag_enabled("SD_HTP_BLOCK_SPARSE_CENTER_SINK");
            bool block_sparse_static_halo = false;
            if (force_full_keep || block_sparse_keep_ratio == 1.0f) {
                block_mask = ggml_ext_block_sparse_full_keep_mask_node(ctx, n_q_blocks, n_k_blocks);
                block_mask_format = GGML_FLASH_ATTN_EXT_MASK_BLOCK_MATRIX;
            } else if (sd_htp_block_sparse_direct_halo_enabled(block_sparse_keep_ratio,
                                                                block_sparse_patch_halo)) {
                block_mask = ggml_ext_block_sparse_direct_halo_index_32_node(ctx,
                                                                             static_cast<int>(L_q),
                                                                             static_cast<int>(L_k),
                                                                             protected_start,
                                                                             protected_len,
                                                                             block_sparse_token_h,
                                                                             block_sparse_token_w,
                                                                             static_cast<int>(n_head),
                                                                             static_cast<int>(n_kv_head),
                                                                             block_sparse_center_sink);
                block_mask_format = GGML_FLASH_ATTN_EXT_MASK_BLOCK_INDEX;
                const int64_t image_tokens = static_cast<int64_t>(block_sparse_token_h) *
                                             block_sparse_token_w;
                const bool image_only_halo = protected_start == 0 && protected_len == 0 &&
                                             L_q == image_tokens && L_k == image_tokens;
                const bool joint_halo = protected_start == image_tokens && protected_len > 0 &&
                                        L_q == protected_start + protected_len &&
                                        L_k == protected_start + protected_len &&
                                        protected_start % SD_HTP_BLOCK_SPARSE_MASK_BLOCK == 0;
                block_sparse_static_halo = !block_sparse_center_sink &&
                                           block_sparse_token_h > 0 &&
                                           block_sparse_token_w > 0 &&
                                           block_sparse_token_h % 4 == 0 &&
                                           block_sparse_token_w % 8 == 0 &&
                                           n_head > 0 &&
                                           n_kv_head > 0 &&
                                           n_head % n_kv_head == 0 &&
                                           (image_only_halo || joint_halo);
            } else if (!sd_htp_block_sparse_score_enabled(block_sparse_keep_ratio)) {
                if (block_sparse_center_sink) {
                    block_mask = ggml_ext_block_sparse_structural_index_32_node(ctx,
                                                                                static_cast<int>(L_q),
                                                                                static_cast<int>(L_k),
                                                                                protected_start,
                                                                                protected_len,
                                                                                static_cast<int>(n_head),
                                                                                static_cast<int>(n_kv_head),
                                                                                false,
                                                                                block_sparse_center_sink);
                    block_mask_format = GGML_FLASH_ATTN_EXT_MASK_BLOCK_INDEX;
                } else {
                    block_mask = ggml_ext_block_sparse_full_mask_node(ctx, n_q_blocks, n_k_blocks);
                    block_mask_format = GGML_FLASH_ATTN_EXT_MASK_BLOCK_MATRIX;
                }
            } else {
                if (block_sparse_host_index) {
                    const int score_block_q = SD_HTP_BLOCK_SPARSE_PATCH_BLOCK;
                    const int score_block_k = SD_HTP_BLOCK_SPARSE_PATCH_BLOCK;
                    ggml_tensor* q_mean = ggml_ext_block_sparse_q_mean_node(ctx, q, score_block_q);
                    ggml_tensor* k_mean = ggml_ext_block_sparse_k_mean_node(ctx, k, score_block_k);
                    ggml_tensor* score  = ggml_ext_block_sparse_score_node(ctx, q_mean, k_mean, score_block_q, score_block_k, false);
                    block_mask = ggml_ext_block_sparse_index_32_from_16_score_node(ctx,
                                                                                   score,
                                                                                   false,
                                                                                   protected_start,
                                                                                   protected_len,
                                                                                   n_head,
                                                                                   n_kv_head);
                    block_mask_format = GGML_FLASH_ATTN_EXT_MASK_BLOCK_INDEX;
                } else {
                    ggml_tensor* q_mean = ggml_ext_block_sparse_q_mean_node(ctx, q, block_q);
                    ggml_tensor* k_mean = ggml_ext_block_sparse_k_mean_node(ctx, k, block_k);
                    ggml_tensor* score  = ggml_ext_block_sparse_score_node(ctx, q_mean, k_mean, block_q, block_k, false);
                    block_mask = ggml_ext_block_sparse_mask_from_score_node(ctx,
                                                                            score,
                                                                            block_q,
                                                                            block_k,
                                                                            false,
                                                                            protected_start,
                                                                            protected_len,
                                                                            n_head,
                                                                            n_kv_head);
                    block_mask_format = GGML_FLASH_ATTN_EXT_MASK_BLOCK_MATRIX;
                }
            }
            const int saved_kv_pad  = kv_pad;
            kv_pad                  = 0;
            const bool can_cache_exact_rho =
                block_sparse_residual_exact_rho &&
                block_sparse_residual_rho_cache_name != nullptr &&
                block_sparse_cache_tensor != nullptr &&
                *block_sparse_cache_tensor;
            const bool anchor_can_cache_residual =
                block_sparse_residual_mode == SD_HTP_BLOCK_SPARSE_RESIDUAL_ANCHOR &&
                block_sparse_residual_cache_name != nullptr &&
                block_sparse_cache_tensor != nullptr &&
                *block_sparse_cache_tensor;
            // Keep the raw flash-attention node available until sparse metadata
            // has been attached. The non-LSE builders wrap kv-scaled outputs in
            // a scale op, which cannot be passed to the flash-attention setters.
            const bool sparse_with_lse = sd_htp_sparse_attention_needs_lse(
                packed_layout, block_sparse_residual_mode, kv_scale,
                anchor_can_cache_residual && can_cache_exact_rho);
            ggml_tensor* dense_candidate = nullptr;
            ggml_tensor* dense_lse = nullptr;
            bool dense_candidate_is_htp_layout = false;
            bool candidate_is_htp_layout = false;
            bool fused_anchor_used = false;
            ggml_tensor* candidate = nullptr;

            const bool fused_anchor_requested =
                block_sparse_residual_mode == SD_HTP_BLOCK_SPARSE_RESIDUAL_ANCHOR &&
                sd_env_flag_enabled("SD_HTP_BLOCK_SPARSE_FUSED_ANCHOR");
            if (fused_anchor_requested) {
                const char* fused_reject_reason = nullptr;
                htp_attention_inputs fused_inputs;
                auto qk_layout = [](const ggml_tensor* tensor,
                                    ggml_type type,
                                    int64_t token_count,
                                    int64_t head_count,
                                    int64_t dim) {
                    return tensor != nullptr && tensor->type == type &&
                           tensor->ne[0] == dim && tensor->ne[1] == token_count &&
                           tensor->ne[2] == head_count && tensor->ne[3] == 1;
                };
                auto v_layout = [](const ggml_tensor* tensor,
                                   ggml_type type,
                                   int64_t token_count,
                                   int64_t head_count,
                                   int64_t dim) {
                    return tensor != nullptr && tensor->type == type &&
                           tensor->ne[0] == dim && tensor->ne[1] == head_count &&
                           tensor->ne[2] == token_count && tensor->ne[3] == 1;
                };
                if (!anchor_can_cache_residual) {
                    fused_reject_reason = "residual cache output is unavailable";
                } else if (!can_cache_exact_rho) {
                    fused_reject_reason = "exact-rho cache output is unavailable";
                } else if (has_packed_qkv_ancestor || packed_layout) {
                    fused_reject_reason = "anchor Q/K/V has a packed producer ancestor";
                } else if (!use_htp_flash_layout) {
                    fused_reject_reason = "row-major HTP Q/K/V layout is unavailable";
                } else if (L_q != L_k) {
                    fused_reject_reason = "fused anchor requires self-attention";
                } else if (N != 1 || n_head != n_kv_head) {
                    fused_reject_reason = "batch or GQA is unsupported";
                } else if (d_head != 64 && d_head != 128) {
                    fused_reject_reason = "head dimension is unsupported";
                } else if (block_q != 32 || block_k != 32 ||
                           block_mask_format != GGML_FLASH_ATTN_EXT_MASK_BLOCK_INDEX) {
                    fused_reject_reason = "block-index 32x32 mask is required";
                } else if (!block_sparse_static_halo) {
                    fused_reject_reason = "static-halo sparse policy is required";
                } else {
                    // The fused op returns final-scale O directly. K still uses
                    // kv_scale with compensating Q scale, while V remains at
                    // model scale so neither output slab needs a post-op scale.
                    fused_inputs = prepare_htp_attention_inputs(nullptr, true, false);
                    if (fused_inputs.q == nullptr || fused_inputs.k == nullptr ||
                        fused_inputs.v == nullptr || fused_inputs.dense_mask == nullptr) {
                        fused_reject_reason = "failed to prepare fused HTP inputs";
                    } else if (!qk_layout(fused_inputs.q, GGML_TYPE_F32,
                                          L_q, n_head, d_head) ||
                               !qk_layout(fused_inputs.k, GGML_TYPE_F16,
                                          L_k, n_kv_head, d_head) ||
                               !v_layout(fused_inputs.v, GGML_TYPE_F16,
                                         L_k, n_kv_head, d_head)) {
                        fused_reject_reason = "Q/K/V dense HTP layouts are unsupported";
                    } else if (!ggml_is_contiguous(fused_inputs.dense_mask) ||
                               fused_inputs.dense_mask->type != GGML_TYPE_F16 ||
                               fused_inputs.dense_mask->ne[0] != GGML_PAD(L_k, 64) ||
                               fused_inputs.dense_mask->ne[1] != L_q) {
                        fused_reject_reason = "dense mask is not contiguous F16";
                    } else if (!ggml_is_contiguous(block_mask) ||
                               block_mask->type != GGML_TYPE_I32 ||
                               block_mask->ne[0] != (L_k + block_k - 1) / block_k ||
                               block_mask->ne[1] != n_q_blocks) {
                        fused_reject_reason = "block mask is not contiguous I32";
                    }
                }

                if (fused_reject_reason == nullptr) {
                    const int64_t slab_elements =
                        (d_head + 1) * n_head * L_q;
                    ggml_tensor* fused_args[] = {
                        fused_inputs.q,
                        fused_inputs.k,
                        fused_inputs.v,
                        fused_inputs.dense_mask,
                        block_mask,
                    };
                    auto* metadata = static_cast<ggml_htp_fused_anchor_metadata*>(
                        ggml_new_buffer(ctx, sizeof(ggml_htp_fused_anchor_metadata)));
                    metadata->qo_len = static_cast<int32_t>(L_q);
                    metadata->kv_len = static_cast<int32_t>(L_k);
                    metadata->n_heads = static_cast<int32_t>(n_head);
                    metadata->n_kv_heads = static_cast<int32_t>(n_kv_head);
                    metadata->head_dim = static_cast<int32_t>(d_head);
                    metadata->block_q = block_q;
                    metadata->block_k = block_k;
                    metadata->n_q_blocks = n_q_blocks;
                    metadata->n_k_blocks = n_k_blocks;
                    metadata->max_kept_blocks = static_cast<int32_t>(block_mask->ne[0]);
                    metadata->mask_format = 1;
                    metadata->sparse_policy = 1;
                    metadata->image_tokens = block_sparse_token_h * block_sparse_token_w;
                    metadata->text_tokens = protected_len > 0 ? protected_len : 0;
                    metadata->image_block_rows = block_sparse_token_h / 4;
                    metadata->image_block_cols = block_sparse_token_w / 8;

                    ggml_tensor* fused = ggml_custom_4d(
                        ctx, GGML_TYPE_F32, 2 * slab_elements, 1, 1, 1,
                        fused_args, 5, sd_htp_fused_anchor_attention_cpu,
                        1, metadata);
                    ggml_set_name(fused, "htp_fused_anchor_attention");
                    if (backend != nullptr && ggml_backend_is_htp(backend) &&
                        !ggml_backend_supports_op(backend, fused)) {
                        fused_reject_reason = "HTP backend rejected the fused custom op";
                    } else {
                        const size_t row_bytes =
                            static_cast<size_t>(d_head + 1) * sizeof(float);
                        const size_t token_bytes =
                            static_cast<size_t>(n_head) * row_bytes;
                        const size_t slab_bytes =
                            static_cast<size_t>(L_q) * token_bytes;
                        dense_candidate = ggml_view_4d(
                            ctx, fused, d_head + 1, n_head, L_q, 1,
                            row_bytes, token_bytes, slab_bytes, 0);
                        candidate = ggml_view_4d(
                            ctx, fused, d_head + 1, n_head, L_q, 1,
                            row_bytes, token_bytes, slab_bytes, slab_bytes);
                        ggml_set_name(dense_candidate, "htp_fused_anchor_dense_o_lse");
                        ggml_set_name(candidate, "htp_fused_anchor_sparse_o_lse");
                        candidate_is_htp_layout = true;
                        fused_anchor_used = true;
                    }
                }

                if (!fused_anchor_used) {
                    if (sd_env_flag_enabled("SD_HTP_SPARSE_OPENCL_STRICT")) {
                        GGML_ABORT("fused anchor admission failed: %s",
                                   fused_reject_reason != nullptr ? fused_reject_reason : "unknown reason");
                    }
                    static std::atomic<bool> fused_fallback_reported{false};
                    if (!fused_fallback_reported.exchange(true)) {
                        std::fprintf(stderr, "fused anchor fallback: %s\n",
                                     fused_reject_reason != nullptr ? fused_reject_reason : "unknown reason");
                    }
                }
            }

            // The legacy exact-rho anchor keeps two independent attention
            // producers and is used whenever fused admission is not selected.
            if (!fused_anchor_used && anchor_can_cache_residual && can_cache_exact_rho) {
                dense_candidate = build_htp_kqv(nullptr, true);
                dense_candidate_is_htp_layout = dense_candidate != nullptr;
                candidate_is_htp_layout = dense_candidate != nullptr;
                if (dense_candidate == nullptr) {
                    dense_candidate = build_kqv(q, k, v, nullptr, true);
                }
            }

            if (fused_anchor_used) {
                // The fused views already carry the HTP output contract.
            } else if (packed_layout) {
                candidate = build_packed_kqv(block_mask, sparse_with_lse);
                // Packed Q/K/V are already in the HMX layout consumed by the
                // HTP packed attention kernel. Treat this candidate as an HTP
                // layout so a CPU graph backend cannot reject it and trigger
                // the row-major fallback.
                candidate_is_htp_layout = candidate != nullptr;
            } else if (dense_candidate != nullptr && candidate_is_htp_layout) {
                candidate = ggml_flash_attn_ext_with_lse(
                    ctx,
                    dense_candidate->src[0],
                    dense_candidate->src[1],
                    dense_candidate->src[2],
                    block_mask,
                    scale,
                    0,
                    0);
                ggml_flash_attn_ext_set_prec(candidate, GGML_PREC_F32);
            } else {
                candidate = build_htp_kqv(block_mask, sparse_with_lse);
                candidate_is_htp_layout = candidate != nullptr;
                if (candidate == nullptr) {
                    candidate = build_kqv(q, k, v, block_mask, sparse_with_lse);
                }
            }
            if (!fused_anchor_used) {
                GGML_ASSERT(block_mask_format != GGML_FLASH_ATTN_EXT_MASK_DEFAULT);
                ggml_flash_attn_ext_set_block_sparse(candidate, block_q, block_k, block_mask_format);
                if (block_sparse_static_halo) {
                    const int image_tokens = block_sparse_token_h * block_sparse_token_w;
                    ggml_flash_attn_ext_set_static_halo_2d(candidate,
                                                           image_tokens,
                                                           protected_len > 0 ? protected_len : 0,
                                                           block_sparse_token_h / 4,
                                                           block_sparse_token_w / 8);
                }
                const sd_htp_route_mode route = sd_htp_route_mode_from_flags(
                    sd_env_flag_enabled("SD_HTP_BLOCK_SPARSE_FUSED_ANCHOR"),
                    sd_env_flag_enabled("SD_HTP_SPARSE_OPENCL"),
                    sd_env_flag_enabled("SD_HTP_SPARSE_OPENCL_RESIDUAL"));
                sd_htp_apply_sparse_halo_kv_load_override(
                    candidate, route, block_sparse_residual_mode);
                if (route == SD_HTP_ROUTE_HTP_CPU &&
                    block_sparse_residual_mode == SD_HTP_BLOCK_SPARSE_RESIDUAL_ANCHOR &&
                    block_sparse_static_halo) {
                    GGML_ASSERT(candidate != nullptr &&
                                candidate->op == GGML_OP_FLASH_ATTN_EXT &&
                                ggml_flash_attn_ext_get_halo_kv_load_mode(candidate) ==
                                    GGML_FLASH_ATTN_EXT_HALO_KV_LOAD_CHUNKED);
                }
            }
            if ((candidate_is_htp_layout && keep_for_htp_hybrid) || backend == nullptr || ggml_backend_supports_op(backend, candidate)) {
                ggml_tensor* sparse_out = flash_attn_o_view(
                    candidate, sparse_with_lse, fused_anchor_used || candidate_is_htp_layout);
                ggml_tensor* sparse_lse = sparse_with_lse ? flash_attn_lse_view(candidate) : nullptr;
                const bool residual_layout_trace =
                    sd_env_flag_enabled("SD_HTP_RESIDUAL_LAYOUT_TRACE");
                if (residual_layout_trace && anchor_can_cache_residual) {
                    std::fprintf(stderr,
                                 "residual_layout sparse_candidate name=%s type=%s ne=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu] sparse_out ne=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu] lse=%s\n",
                                 candidate->name, ggml_type_name(candidate->type),
                                 (long long) candidate->ne[0], (long long) candidate->ne[1],
                                 (long long) candidate->ne[2], (long long) candidate->ne[3],
                                 candidate->nb[0], candidate->nb[1], candidate->nb[2], candidate->nb[3],
                                 (long long) sparse_out->ne[0], (long long) sparse_out->ne[1],
                                 (long long) sparse_out->ne[2], (long long) sparse_out->ne[3],
                                 sparse_out->nb[0], sparse_out->nb[1], sparse_out->nb[2], sparse_out->nb[3],
                                 sparse_lse != nullptr ? "yes" : "no");
                }
                ggml_tensor* sparse_reshape_out = sparse_out;
                if (sd_htp_packed_residual_apply_needs_contiguous(
                        packed_layout, block_sparse_residual_mode)) {
                    sparse_reshape_out = ggml_cont(ctx, sparse_reshape_out);
                }
                const bool cached_exact_rho_valid =
                    block_sparse_residual_exact_rho &&
                    block_sparse_cached_rho != nullptr &&
                    block_sparse_cached_rho->ne[0] == 1 &&
                    block_sparse_cached_rho->ne[1] == sparse_out->ne[1] &&
                    block_sparse_cached_rho->ne[2] == sparse_out->ne[2] &&
                    block_sparse_cached_rho->ne[3] == 1;
                if (cached_exact_rho_valid) {
                    sparse_reshape_out = ggml_mul(ctx, sparse_reshape_out, block_sparse_cached_rho);
                } else if (block_sparse_residual_rho != 1.0f) {
                    sparse_reshape_out = ggml_ext_scale(ctx, sparse_reshape_out, block_sparse_residual_rho);
                }
                if (anchor_can_cache_residual) {
                    if (dense_candidate == nullptr && can_cache_exact_rho) {
                        dense_candidate = build_htp_kqv(nullptr, true);
                        dense_candidate_is_htp_layout = dense_candidate != nullptr;
                        if (dense_candidate == nullptr) {
                            dense_candidate = build_kqv(q, k, v, nullptr, true);
                        }
                    } else if (dense_candidate == nullptr) {
                        dense_candidate = build_htp_kqv(nullptr);
                        dense_candidate_is_htp_layout = dense_candidate != nullptr;
                        if (dense_candidate == nullptr) {
                            dense_candidate = build_kqv(q, k, v, nullptr);
                        }
                    }
                    ggml_tensor* dense_out = flash_attn_o_view(
                        dense_candidate, can_cache_exact_rho,
                        fused_anchor_used || dense_candidate_is_htp_layout);
                    if (can_cache_exact_rho) {
                        dense_lse = flash_attn_lse_view(dense_candidate);
                    }
                    if (residual_layout_trace) {
                        std::fprintf(stderr,
                                     "residual_layout dense_candidate name=%s type=%s ne=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu] dense_out ne=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu] dense_lse=%s\n",
                                     dense_candidate->name, ggml_type_name(dense_candidate->type),
                                     (long long) dense_candidate->ne[0], (long long) dense_candidate->ne[1],
                                     (long long) dense_candidate->ne[2], (long long) dense_candidate->ne[3],
                                     dense_candidate->nb[0], dense_candidate->nb[1], dense_candidate->nb[2], dense_candidate->nb[3],
                                     (long long) dense_out->ne[0], (long long) dense_out->ne[1],
                                     (long long) dense_out->ne[2], (long long) dense_out->ne[3],
                                     dense_out->nb[0], dense_out->nb[1], dense_out->nb[2], dense_out->nb[3],
                                     dense_lse != nullptr ? "yes" : "no");
                    }
                    ggml_tensor* rho = nullptr;
                    ggml_tensor* residual = nullptr;
                    if (can_cache_exact_rho) {
                        GGML_ASSERT(dense_lse != nullptr && sparse_lse != nullptr);
                        const bool residual_route_requested =
                            sd_env_flag_enabled("SD_HTP_SPARSE_OPENCL_RESIDUAL");
                        const bool residual_route_enabled =
                            residual_route_requested &&
                            sd_htp_opencl_residual_route_prerequisites();
                        if (residual_route_requested && !residual_route_enabled &&
                            sd_env_flag_enabled("SD_HTP_SPARSE_OPENCL_STRICT")) {
                            GGML_ABORT(
                                "SD_HTP_SPARSE_OPENCL_RESIDUAL prerequisites are not met for "
                                "the selected sparse or fused-anchor route");
                        }
                        if (residual_route_enabled) {
                            ggml_tensor* rho_args[] = { dense_candidate, candidate };
                            rho = ggml_custom_4d(
                                ctx, GGML_TYPE_F32, 1, n_head, L_q, N,
                                rho_args, 2, sd_htp_exact_rho_cache_cpu,
                                GGML_N_TASKS_MAX, nullptr);
                        } else {
                            ggml_tensor* rho_sub = ggml_sub(
                                ctx, sparse_lse, dense_lse);
                            ggml_set_name(
                                rho_sub, "htp_cpu_residual_rho_sub");
                            rho = ggml_exp(ctx, rho_sub);
                        }
                        ggml_set_name(rho, block_sparse_residual_rho_cache_name);
                        (*block_sparse_cache_tensor)(block_sparse_residual_rho_cache_name, rho);
                        if (residual_route_enabled) {
                            ggml_tensor* residual_args[] = {
                                dense_candidate, candidate, rho,
                            };
                            residual = ggml_custom_4d(
                                ctx, GGML_TYPE_F16, d_head, n_head, L_q, N,
                                residual_args, 3, sd_htp_exact_residual_cache_cpu,
                                GGML_N_TASKS_MAX, nullptr);
                        } else {
                            sparse_reshape_out = ggml_mul(ctx, sparse_reshape_out, rho);
                            ggml_set_name(
                                sparse_reshape_out,
                                "htp_cpu_residual_scale_mul");
                        }
                    }
                    if (residual == nullptr) {
                        residual = ggml_sub(ctx, dense_out, sparse_reshape_out);
                        residual = ggml_cast(ctx, residual, GGML_TYPE_F16);
                    }
                    if (residual_layout_trace) {
                        std::fprintf(stderr,
                                     "residual_layout cache residual name=%s type=%s ne=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu] rho=%s\n",
                                     residual->name, ggml_type_name(residual->type),
                                     (long long) residual->ne[0], (long long) residual->ne[1],
                                     (long long) residual->ne[2], (long long) residual->ne[3],
                                     residual->nb[0], residual->nb[1], residual->nb[2], residual->nb[3],
                                     rho != nullptr ? ggml_type_name(rho->type) : "none");
                    }
                    ggml_set_name(residual, block_sparse_residual_cache_name);
                    (*block_sparse_cache_tensor)(block_sparse_residual_cache_name, residual);
                    if (sd_env_flag_enabled("SD_HTP_BLOCK_SPARSE_MASK_TRACE")) {
                        std::fprintf(stderr,
                                     "block_sparse_residual anchor=1 exact_rho=%d rho=%.4f cache=%s type=%s bytes=%zu rho_cache=%s rho_type=%s rho_bytes=%zu shape=[%lld,%lld,%lld,%lld]\n",
                                     rho != nullptr ? 1 : 0,
                                     block_sparse_residual_rho,
                                     block_sparse_residual_cache_name,
                                     ggml_type_name(residual->type),
                                     ggml_nbytes(residual),
                                     block_sparse_residual_rho_cache_name != nullptr ? block_sparse_residual_rho_cache_name : "<unnamed>",
                                     rho != nullptr ? ggml_type_name(rho->type) : "none",
                                     rho != nullptr ? ggml_nbytes(rho) : 0,
                                     (long long) residual->ne[0],
                                     (long long) residual->ne[1],
                                     (long long) residual->ne[2],
                                     (long long) residual->ne[3]);
                    }
                    kqv = dense_out;
                } else if (block_sparse_residual_mode == SD_HTP_BLOCK_SPARSE_RESIDUAL_APPLY &&
                           block_sparse_cached_residual != nullptr &&
                           block_sparse_cached_residual->ne[0] == sparse_out->ne[0] &&
                           block_sparse_cached_residual->ne[1] == sparse_out->ne[1] &&
                           block_sparse_cached_residual->ne[2] == sparse_out->ne[2] &&
                           block_sparse_cached_residual->ne[3] == sparse_out->ne[3]) {
                    ggml_tensor* residual = block_sparse_cached_residual;
                    if (residual->type == GGML_TYPE_F16) {
                        residual = ggml_cast(ctx, residual, GGML_TYPE_F32);
                    }
                    if (block_sparse_residual_scale != 1.0f) {
                        residual = ggml_ext_scale(ctx, residual, block_sparse_residual_scale);
                    }
                    kqv = ggml_add(ctx, sparse_reshape_out, residual);
                    if (sd_env_flag_enabled("SD_HTP_RESIDUAL_VALUE_TRACE")) {
                        std::fprintf(stderr,
                                     "residual_value apply sparse_hash=%016" PRIx64
                                     " residual_hash=%016" PRIx64
                                     " sparse_data=%p residual_data=%p shape=[%lld,%lld,%lld,%lld]\n",
                                     sd_htp_residual_trace_hash(sparse_reshape_out),
                                     sd_htp_residual_trace_hash(residual),
                                     sparse_reshape_out->data, residual->data,
                                     (long long) sparse_reshape_out->ne[0],
                                     (long long) sparse_reshape_out->ne[1],
                                     (long long) sparse_reshape_out->ne[2],
                                     (long long) sparse_reshape_out->ne[3]);
                    }
                    if (residual_layout_trace) {
                        std::fprintf(stderr,
                                     "residual_layout apply cached_residual name=%s type=%s ne=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu] sparse_scaled ne=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu]\n",
                                     residual->name, ggml_type_name(residual->type),
                                     (long long) residual->ne[0], (long long) residual->ne[1],
                                     (long long) residual->ne[2], (long long) residual->ne[3],
                                     residual->nb[0], residual->nb[1], residual->nb[2], residual->nb[3],
                                     (long long) sparse_reshape_out->ne[0], (long long) sparse_reshape_out->ne[1],
                                     (long long) sparse_reshape_out->ne[2], (long long) sparse_reshape_out->ne[3],
                                     sparse_reshape_out->nb[0], sparse_reshape_out->nb[1],
                                     sparse_reshape_out->nb[2], sparse_reshape_out->nb[3]);
                    }
                    if (sd_env_flag_enabled("SD_HTP_BLOCK_SPARSE_MASK_TRACE")) {
                        std::fprintf(stderr,
                                     "block_sparse_residual apply=1 cache_hit=1 rho_cache_hit=%d scale=%.4f rho=%.4f cache=%s rho_cache=%s shape=[%lld,%lld,%lld,%lld]\n",
                                     cached_exact_rho_valid ? 1 : 0,
                                     block_sparse_residual_scale,
                                     block_sparse_residual_rho,
                                     block_sparse_residual_cache_name != nullptr ? block_sparse_residual_cache_name : "<unnamed>",
                                     block_sparse_residual_rho_cache_name != nullptr ? block_sparse_residual_rho_cache_name : "<unnamed>",
                                     (long long) sparse_out->ne[0],
                                     (long long) sparse_out->ne[1],
                                     (long long) sparse_out->ne[2],
                                     (long long) sparse_out->ne[3]);
                    }
                } else {
                    if (block_sparse_residual_mode == SD_HTP_BLOCK_SPARSE_RESIDUAL_APPLY &&
                        sd_env_flag_enabled("SD_HTP_BLOCK_SPARSE_MASK_TRACE")) {
                        std::fprintf(stderr,
                                     "block_sparse_residual apply=1 cache_hit=0 cache=%s\n",
                                     block_sparse_residual_cache_name != nullptr ? block_sparse_residual_cache_name : "<unnamed>");
                    }
                    kqv = sparse_out;
                }
            }
            kv_pad = saved_kv_pad;
        }

        if (kqv == nullptr &&
            can_use_flash_attn &&
            L_k % 256 != 0 &&
            !(use_htp_flash_layout && sd_env_flag_enabled("SD_HTP_ATTENTION"))) {
            kv_pad = GGML_PAD(L_k, 256) - static_cast<int>(L_k);
        }

        if (can_use_flash_attn) {
            if (kqv == nullptr) {
                const bool htp_dense_candidate = use_htp_flash_layout && sd_env_flag_enabled("SD_HTP_ATTENTION");
                kqv = htp_dense_candidate ? build_htp_kqv(mask) : nullptr;
                if (kqv == nullptr && packed_layout) {
                    kqv = build_packed_kqv(mask);
                }
                if (kqv == nullptr) {
                    kqv = build_kqv(q, k, v, mask);
                }
                if (!htp_dense_candidate && backend != nullptr && !ggml_backend_supports_op(backend, kqv)) {
                    kqv = nullptr;
                } else {
                    kqv = ggml_view_3d(ctx, kqv, d_head, n_head, L_q, kqv->nb[1], kqv->nb[2], 0);
                }
            }
        }
    }

    if (kqv == nullptr) {
        // if (flash_attn) {
        //     LOG_DEBUG("fallback to default attention, L_q:%d L_k:%d n_head:%d C:%d d_head:%d N:%d", L_q, L_k, n_head, C, d_head, N);
        // }
        v = ggml_ext_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3));  // [N, n_kv_head, d_head, L_k]
        v = ggml_reshape_3d(ctx, v, L_k, d_head, n_kv_head * N);   // [N * n_kv_head, d_head, L_k]

        auto kq = ggml_mul_mat(ctx, k, q);  // [N * n_head, L_q, L_k]
        ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
        kq = ggml_scale_inplace(ctx, kq, scale);
        if (mask) {
            kq = ggml_add_inplace(ctx, kq, mask);
        }
        kq = ggml_soft_max_inplace(ctx, kq);

        kqv = ggml_mul_mat(ctx, v, kq);  // [N * n_head, L_q, d_head]

        kqv = ggml_reshape_4d(ctx, kqv, d_head, L_q, n_head, N);  // [N, n_head, L_q, d_head]
        kqv = ggml_permute(ctx, kqv, 0, 2, 1, 3);                 // [N, L_q, n_head, d_head]
    }

    kqv = ggml_ext_cont(ctx, kqv);
    kqv = ggml_reshape_3d(ctx, kqv, d_head * n_head, L_q, N);  // [N, L_q, C]

    return kqv;
}

__STATIC_INLINE__ ggml_tensor* ggml_ext_layer_norm(ggml_context* ctx,
                                                   ggml_tensor* x,
                                                   ggml_tensor* w,
                                                   ggml_tensor* b,
                                                   float eps = EPS) {
    x = ggml_norm(ctx, x, eps);
    if (w != nullptr) {
        x = ggml_mul_inplace(ctx, x, w);
        if (b != nullptr) {
            x = ggml_add_inplace(ctx, x, b);
        }
    }
    return x;
}

__STATIC_INLINE__ ggml_tensor* ggml_ext_group_norm(ggml_context* ctx,
                                                   ggml_tensor* x,
                                                   ggml_tensor* w,
                                                   ggml_tensor* b,
                                                   int num_groups = 32) {
    if (ggml_n_dims(x) >= 3 && w != nullptr && b != nullptr) {
        w = ggml_reshape_4d(ctx, w, 1, 1, w->ne[0], 1);
        b = ggml_reshape_4d(ctx, b, 1, 1, b->ne[0], 1);
    }

    const float eps = 1e-6f;  // default eps parameter
    x               = ggml_group_norm(ctx, x, num_groups, eps);
    if (w != nullptr && b != nullptr) {
        x = ggml_mul_inplace(ctx, x, w);
        // b = ggml_repeat(ctx, b, x);
        x = ggml_add_inplace(ctx, x, b);
    }
    return x;
}

__STATIC_INLINE__ void ggml_ext_backend_tensor_get_and_sync(ggml_backend_t backend, const ggml_tensor* tensor, void* data, size_t offset, size_t size) {
    if ((sd_backend_is(backend, "ROCm") || sd_backend_is(backend, "CUDA") || sd_backend_is(backend, "SYCL")) &&
        !ggml_backend_is_cpu(backend)) {
        ggml_backend_tensor_get_async(backend, tensor, data, offset, size);
        ggml_backend_synchronize(backend);
        return;
    }

    ggml_backend_tensor_get(tensor, data, offset, size);
}

__STATIC_INLINE__ float ggml_ext_backend_tensor_get_f32(ggml_tensor* tensor) {
    GGML_ASSERT(tensor->type == GGML_TYPE_F32 || tensor->type == GGML_TYPE_F16 || tensor->type == GGML_TYPE_I32 || tensor->type == GGML_TYPE_BF16);
    float value;
    if (tensor->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(tensor, &value, 0, sizeof(value));
    } else if (tensor->type == GGML_TYPE_BF16) {
        ggml_bf16_t bf16_value;
        ggml_backend_tensor_get(tensor, &bf16_value, 0, sizeof(bf16_value));
        value = ggml_bf16_to_fp32(bf16_value);
    } else if (tensor->type == GGML_TYPE_F16) {
        ggml_fp16_t f16_value;
        ggml_backend_tensor_get(tensor, &f16_value, 0, sizeof(f16_value));
        value = ggml_fp16_to_fp32(f16_value);
    } else {  // GGML_TYPE_I32
        int int32_value;
        ggml_backend_tensor_get(tensor, &int32_value, 0, sizeof(int32_value));
        value = (float)int32_value;
    }
    return value;
}

__STATIC_INLINE__ float ggml_ext_tensor_get_scalar_f32(const ggml_tensor* tensor, int64_t i0, int64_t i1 = 0, int64_t i2 = 0, int64_t i3 = 0) {
    GGML_ASSERT(tensor != nullptr);
    GGML_ASSERT(i0 >= 0 && i0 < tensor->ne[0]);
    GGML_ASSERT(i1 >= 0 && i1 < tensor->ne[1]);
    GGML_ASSERT(i2 >= 0 && i2 < tensor->ne[2]);
    GGML_ASSERT(i3 >= 0 && i3 < tensor->ne[3]);

    const size_t offset = size_t(i3) * size_t(tensor->nb[3]) +
                          size_t(i2) * size_t(tensor->nb[2]) +
                          size_t(i1) * size_t(tensor->nb[1]) +
                          size_t(i0) * size_t(tensor->nb[0]);

    if (tensor->buffer != nullptr) {
        if (tensor->type == GGML_TYPE_F32) {
            float value;
            ggml_backend_tensor_get(tensor, &value, offset, sizeof(value));
            return value;
        }
        if (tensor->type == GGML_TYPE_F16) {
            ggml_fp16_t value;
            ggml_backend_tensor_get(tensor, &value, offset, sizeof(value));
            return ggml_fp16_to_fp32(value);
        }
        if (tensor->type == GGML_TYPE_BF16) {
            ggml_bf16_t value;
            ggml_backend_tensor_get(tensor, &value, offset, sizeof(value));
            return ggml_bf16_to_fp32(value);
        }
        if (tensor->type == GGML_TYPE_I32) {
            int32_t value;
            ggml_backend_tensor_get(tensor, &value, offset, sizeof(value));
            return (float)value;
        }
        GGML_ABORT("unsupported tensor type %s for scalar routing read", ggml_type_name(tensor->type));
    }

    const char* data = static_cast<const char*>(tensor->data);
    GGML_ASSERT(data != nullptr);
    if (tensor->type == GGML_TYPE_F32) {
        return *reinterpret_cast<const float*>(data + offset);
    }
    if (tensor->type == GGML_TYPE_F16) {
        return ggml_fp16_to_fp32(*reinterpret_cast<const ggml_fp16_t*>(data + offset));
    }
    if (tensor->type == GGML_TYPE_BF16) {
        return ggml_bf16_to_fp32(*reinterpret_cast<const ggml_bf16_t*>(data + offset));
    }
    if (tensor->type == GGML_TYPE_I32) {
        return (float)*reinterpret_cast<const int32_t*>(data + offset);
    }
    GGML_ABORT("unsupported tensor type %s for scalar routing read", ggml_type_name(tensor->type));
}

struct ggml_ext_attention_lse_params {
    int n_heads;
    int block_q;
    int block_k;
    ggml_flash_attn_ext_mask_format mask_format;
    float scale;
};

__STATIC_INLINE__ void ggml_ext_attention_lse_custom(ggml_tensor* dst, int ith, int nth, void* userdata) {
    const auto* params = static_cast<const ggml_ext_attention_lse_params*>(userdata);
    const ggml_tensor* q = dst->src[0];
    const ggml_tensor* k = dst->src[1];
    const ggml_tensor* mask = dst->src[2];
    GGML_ASSERT(params != nullptr && q != nullptr && k != nullptr);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(mask == nullptr || mask->type == GGML_TYPE_F32 || mask->type == GGML_TYPE_F16 || mask->type == GGML_TYPE_I32);

    const int head_dim = static_cast<int>(q->ne[0]);
    const int qo_len = static_cast<int>(q->ne[1]);
    const int n_heads = params->n_heads;
    const int kv_len = static_cast<int>(k->ne[1]);
    const int n_kv_heads = static_cast<int>(k->ne[2]);
    const int gqa = std::max(1, n_heads / std::max(1, n_kv_heads));
    const int q_blocks_per_head = (qo_len + params->block_q - 1) / params->block_q;
    const int k_blocks_per_head = (kv_len + params->block_k - 1) / params->block_k;
    const int n_q_blocks = n_heads * q_blocks_per_head;
    const int n_k_blocks = n_kv_heads * k_blocks_per_head;
    const ggml_flash_attn_ext_mask_format mask_format = params->mask_format;

    GGML_ASSERT(q->ne[2] == n_heads);
    GGML_ASSERT(k->ne[0] == head_dim);
    GGML_ASSERT(dst->ne[0] == 1 && dst->ne[1] == n_heads && dst->ne[2] == qo_len);
    if (mask != nullptr) {
        if (mask->type == GGML_TYPE_I32) {
            GGML_ASSERT(mask_format == GGML_FLASH_ATTN_EXT_MASK_BLOCK_MATRIX ||
                        mask_format == GGML_FLASH_ATTN_EXT_MASK_BLOCK_INDEX);
            GGML_ASSERT(mask->ne[1] == n_q_blocks);
            if (mask_format == GGML_FLASH_ATTN_EXT_MASK_BLOCK_MATRIX) {
                GGML_ASSERT(mask->ne[0] == n_k_blocks);
            } else {
                GGML_ASSERT(mask->ne[0] > 0);
            }
        } else {
            GGML_ASSERT(mask_format == GGML_FLASH_ATTN_EXT_MASK_DEFAULT);
            GGML_ASSERT(mask->ne[0] == kv_len && mask->ne[1] == qo_len);
            GGML_ASSERT(mask->ne[2] == 1 || mask->ne[2] == n_heads);
        }
    } else {
        GGML_ASSERT(mask_format == GGML_FLASH_ATTN_EXT_MASK_DEFAULT);
    }

    auto block_sparse_keeps = [&](int qt, int h, int kt, int kv_head) -> bool {
        GGML_ASSERT(mask != nullptr && mask->type == GGML_TYPE_I32);
        const int qb = h * q_blocks_per_head + qt / params->block_q;
        const int kb = kv_head * k_blocks_per_head + kt / params->block_k;
        if (mask_format == GGML_FLASH_ATTN_EXT_MASK_BLOCK_MATRIX) {
            return static_cast<int32_t>(ggml_ext_tensor_get_scalar_f32(mask, kb, qb)) == 0;
        }
        GGML_ASSERT(mask_format == GGML_FLASH_ATTN_EXT_MASK_BLOCK_INDEX);
        for (int i = 0; i < mask->ne[0]; ++i) {
            const int kept = static_cast<int32_t>(ggml_ext_tensor_get_scalar_f32(mask, i, qb));
            if (kept == kb) {
                return true;
            }
            if (kept < 0) {
                return false;
            }
        }
        return false;
    };

    auto* out = reinterpret_cast<float*>(dst->data);
    for (int row = ith; row < qo_len * n_heads; row += nth) {
        const int qt = row / n_heads;
        const int h = row % n_heads;
        const int kv_head = std::min(n_kv_heads - 1, h / gqa);
        const int mask_head = mask != nullptr && mask->ne[2] == n_heads ? h : 0;

        float max_score = -std::numeric_limits<float>::infinity();
        for (int kt = 0; kt < kv_len; ++kt) {
            float score = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                score += ggml_ext_tensor_get_scalar_f32(q, d, qt, h) *
                         ggml_ext_tensor_get_scalar_f32(k, d, kt, kv_head);
            }
            score *= params->scale;
            if (mask != nullptr) {
                if (mask->type == GGML_TYPE_I32) {
                    if (!block_sparse_keeps(qt, h, kt, kv_head)) {
                        score = -std::numeric_limits<float>::infinity();
                    }
                } else {
                    score += ggml_ext_tensor_get_scalar_f32(mask, kt, qt, mask_head);
                }
            }
            max_score = std::max(max_score, score);
        }

        double exp_sum = 0.0;
        if (std::isfinite(max_score)) {
            for (int kt = 0; kt < kv_len; ++kt) {
                float score = 0.0f;
                for (int d = 0; d < head_dim; ++d) {
                    score += ggml_ext_tensor_get_scalar_f32(q, d, qt, h) *
                             ggml_ext_tensor_get_scalar_f32(k, d, kt, kv_head);
                }
                score *= params->scale;
                if (mask != nullptr) {
                    if (mask->type == GGML_TYPE_I32) {
                        if (!block_sparse_keeps(qt, h, kt, kv_head)) {
                            score = -std::numeric_limits<float>::infinity();
                        }
                    } else {
                        score += ggml_ext_tensor_get_scalar_f32(mask, kt, qt, mask_head);
                    }
                }
                exp_sum += std::exp(double(score - max_score));
            }
        }

        const float lse = exp_sum > 0.0 ? max_score + float(std::log(exp_sum))
                                        : -std::numeric_limits<float>::infinity();
        out[size_t(qt) * size_t(n_heads) + size_t(h)] = lse;
    }
}

__STATIC_INLINE__ ggml_tensor* ggml_ext_attention_lse_node(ggml_context* ctx,
                                                           ggml_tensor* q,
                                                           ggml_tensor* k,
                                                           ggml_tensor* mask,
                                                           int n_heads,
                                                           float scale,
                                                           ggml_flash_attn_ext_mask_format mask_format) {
    GGML_ASSERT(q != nullptr && k != nullptr);
    GGML_ASSERT(n_heads > 0);
    GGML_ASSERT(q->ne[2] == n_heads);
    GGML_ASSERT(k->ne[0] == q->ne[0]);
    if (mask != nullptr) {
        GGML_ASSERT(mask->type == GGML_TYPE_F32 || mask->type == GGML_TYPE_F16 || mask->type == GGML_TYPE_I32);
    }
    if (mask != nullptr && mask->type == GGML_TYPE_I32) {
        GGML_ASSERT(mask_format == GGML_FLASH_ATTN_EXT_MASK_BLOCK_MATRIX ||
                    mask_format == GGML_FLASH_ATTN_EXT_MASK_BLOCK_INDEX);
    } else {
        GGML_ASSERT(mask_format == GGML_FLASH_ATTN_EXT_MASK_DEFAULT);
    }

    ggml_tensor* args[] = {q, k, mask};
    auto* params = static_cast<ggml_ext_attention_lse_params*>(
        ggml_new_buffer(ctx, sizeof(ggml_ext_attention_lse_params)));
    params->n_heads = n_heads;
    params->block_q = SD_HTP_BLOCK_SPARSE_MASK_BLOCK;
    params->block_k = SD_HTP_BLOCK_SPARSE_MASK_BLOCK;
    params->mask_format = mask_format;
    params->scale = scale;
    return ggml_custom_4d(ctx,
                          GGML_TYPE_F32,
                          1,
                          n_heads,
                          q->ne[1],
                          1,
                          args,
                          mask != nullptr ? 3 : 2,
                          ggml_ext_attention_lse_custom,
                          GGML_N_TASKS_MAX,
                          params);
}

__STATIC_INLINE__ std::vector<int32_t> ggml_ext_block_sparse_mask_mean_topk(const ggml_tensor* q,
                                                                            const ggml_tensor* k,
                                                                            int block_q,
                                                                            int block_k,
                                                                            int top_k,
                                                                            bool causal = false,
                                                                            int protected_prefix_len = 0) {
    GGML_ASSERT(q != nullptr && k != nullptr);
    GGML_ASSERT(block_q > 0 && block_k > 0);
    GGML_ASSERT(ggml_is_contiguous(q) && ggml_is_contiguous(k));
    GGML_ASSERT(q->type == GGML_TYPE_F32 || q->type == GGML_TYPE_F16);
    GGML_ASSERT(k->type == GGML_TYPE_F32 || k->type == GGML_TYPE_F16);
    GGML_ASSERT(q->ne[0] == k->ne[0]);

    const int head_dim   = static_cast<int>(q->ne[0]);
    const int qo_len     = static_cast<int>(q->ne[1]);
    const int kv_len     = static_cast<int>(k->ne[1]);
    const int n_heads    = static_cast<int>(q->ne[2]);
    const int n_kv_heads = static_cast<int>(k->ne[2]);
    const int n_q_blocks = (qo_len + block_q - 1) / block_q;
    const int n_k_blocks = (kv_len + block_k - 1) / block_k;
    const int gqa        = std::max(1, n_heads / std::max(1, n_kv_heads));

    std::vector<float> q_mean(size_t(n_q_blocks) * size_t(n_heads) * size_t(head_dim), 0.0f);
    std::vector<float> k_mean(size_t(n_k_blocks) * size_t(n_kv_heads) * size_t(head_dim), 0.0f);

    for (int qb = 0; qb < n_q_blocks; ++qb) {
        const int begin = qb * block_q;
        const int end   = std::min(qo_len, begin + block_q);
        const float inv = 1.0f / static_cast<float>(std::max<int64_t>(1, end - begin));
        for (int h = 0; h < n_heads; ++h) {
            for (int t = begin; t < end; ++t) {
                for (int d = 0; d < head_dim; ++d) {
                    q_mean[(size_t(qb) * n_heads + h) * head_dim + d] +=
                        ggml_ext_tensor_get_scalar_f32(q, d, t, h) * inv;
                }
            }
        }
    }

    for (int kb = 0; kb < n_k_blocks; ++kb) {
        const int begin = kb * block_k;
        const int end   = std::min(kv_len, begin + block_k);
        const float inv = 1.0f / static_cast<float>(std::max<int64_t>(1, end - begin));
        for (int h = 0; h < n_kv_heads; ++h) {
            for (int t = begin; t < end; ++t) {
                for (int d = 0; d < head_dim; ++d) {
                    k_mean[(size_t(kb) * n_kv_heads + h) * head_dim + d] +=
                        ggml_ext_tensor_get_scalar_f32(k, d, t, h) * inv;
                }
            }
        }
    }

    std::vector<ggml_fp16_t> scores(size_t(n_q_blocks) * size_t(n_k_blocks));
    for (int qb = 0; qb < n_q_blocks; ++qb) {
        for (int kb = 0; kb < n_k_blocks; ++kb) {
            float best = -std::numeric_limits<float>::infinity();
            for (int h = 0; h < n_heads; ++h) {
                const int kv_h = std::min(n_kv_heads - 1, h / gqa);
                float dot = 0.0f;
                for (int d = 0; d < head_dim; ++d) {
                    const float qv = q_mean[(size_t(qb) * n_heads + h) * head_dim + d];
                    const float kv = k_mean[(size_t(kb) * n_kv_heads + kv_h) * head_dim + d];
                    dot += qv * kv;
                }
                best = std::max(best, dot);
            }
            scores[size_t(qb) * size_t(n_k_blocks) + size_t(kb)] = ggml_fp32_to_fp16(best);
        }
    }

    std::vector<int32_t> block_mask(size_t(n_q_blocks) * size_t(n_k_blocks), 1);
    std::vector<std::pair<float, int>> candidates;
    candidates.reserve(size_t(n_k_blocks));

    protected_prefix_len = std::max(0, std::min<int>(protected_prefix_len, n_k_blocks * block_k));
    top_k = std::max(1, std::min(top_k, n_k_blocks));
    const float keep_ratio = sd_htp_block_sparse_keep_ratio();
    const int protected_q_blocks = std::min<int>(n_q_blocks, (protected_prefix_len + block_q - 1) / block_q);
    const int protected_k_blocks = std::min<int>(n_k_blocks, (protected_prefix_len + block_k - 1) / block_k);
    for (int qb = 0; qb < n_q_blocks; ++qb) {
        const bool protect_q = qb < protected_q_blocks;
        for (int kb = 0; kb < n_k_blocks; ++kb) {
            if (protect_q || kb < protected_k_blocks) {
                block_mask[size_t(qb) * size_t(n_k_blocks) + size_t(kb)] = 0;
            }
        }
    }
    for (int qb = 0; qb < n_q_blocks; ++qb) {
        const int q_begin = qb * block_q;
        if (q_begin < protected_prefix_len) {
            continue;
        }
        if (keep_ratio >= 1.0f) {
            for (int kb = 0; kb < n_k_blocks; ++kb) {
                const int k_begin = kb * block_k;
                if (k_begin < protected_prefix_len) {
                    continue;
                }
                if (causal && kb * block_k > (qb + 1) * block_q - 1) {
                    continue;
                }
                block_mask[size_t(qb) * size_t(n_k_blocks) + size_t(kb)] = 0;
            }
            continue;
        }
        candidates.clear();
        for (int kb = 0; kb < n_k_blocks; ++kb) {
            const int k_begin = kb * block_k;
            if (k_begin < protected_prefix_len) {
                continue;
            }
            if (causal && kb * block_k > (qb + 1) * block_q - 1) {
                continue;
            }
            const float s = ggml_fp16_to_fp32(scores[size_t(qb) * size_t(n_k_blocks) + size_t(kb)]);
            candidates.emplace_back(s, kb);
        }
        if (candidates.empty()) {
            continue;
        }
        std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
            if (a.first != b.first) {
                return a.first > b.first;
            }
            return a.second < b.second;
        });
        const float max_score = candidates.front().first;
        double total_weight = 0.0;
        for (const auto& cand : candidates) {
            total_weight += std::exp(double(cand.first - max_score));
        }
        if (total_weight <= 0.0) {
            continue;
        }
        double kept_weight = 0.0;
        for (size_t i = 0; i < candidates.size(); ++i) {
            kept_weight += std::exp(double(candidates[i].first - max_score));
            block_mask[size_t(qb) * size_t(n_k_blocks) + size_t(candidates[i].second)] = 0;
            if (kept_weight / total_weight >= keep_ratio) {
                break;
            }
        }
    }
    return block_mask;
}

struct ggml_ext_block_sparse_mean_params {
    int block;
};

__STATIC_INLINE__ void ggml_ext_block_sparse_q_mean_custom(ggml_tensor* dst, int ith, int nth, void* userdata) {
    const bool trace_mask = sd_env_flag_enabled("SD_HTP_BLOCK_SPARSE_MASK_TRACE");
    const int64_t trace_begin = trace_mask ? ggml_time_ms() : 0;
    const auto* params = static_cast<const ggml_ext_block_sparse_mean_params*>(userdata);
    const ggml_tensor* q = dst->src[0];
    GGML_ASSERT(q != nullptr && params != nullptr);
    const int head_dim   = static_cast<int>(q->ne[0]);
    const int qo_len     = static_cast<int>(q->ne[1]);
    const int n_heads    = static_cast<int>(q->ne[2]);
    const int n_q_blocks = static_cast<int>(dst->ne[1]);
    auto* out            = reinterpret_cast<ggml_fp16_t*>(dst->data);
    const int q_blocks_per_head = (qo_len + params->block - 1) / params->block;
    GGML_ASSERT(n_q_blocks == n_heads * q_blocks_per_head);
    thread_local std::vector<float> accum;
    accum.assign(static_cast<size_t>(head_dim), 0.0f);

    const float* q_f32 = q->type == GGML_TYPE_F32 ? reinterpret_cast<const float*>(q->data) : nullptr;
    const ggml_fp16_t* q_f16 = q->type == GGML_TYPE_F16 ? reinterpret_cast<const ggml_fp16_t*>(q->data) : nullptr;
    for (int qb = ith; qb < n_q_blocks; qb += nth) {
        const int h = qb / q_blocks_per_head;
        const int block_in_head = qb % q_blocks_per_head;
        const int begin = block_in_head * params->block;
        const int end   = std::min(qo_len, begin + params->block);
        const float inv = 1.0f / static_cast<float>(std::max(1, end - begin));
        std::fill(accum.begin(), accum.end(), 0.0f);
        for (int t = begin; t < end; ++t) {
            const size_t row_off = (size_t(h) * size_t(qo_len) + size_t(t)) * size_t(head_dim);
            if (q_f32 != nullptr) {
                const float* row_ptr = q_f32 + row_off;
                for (int d = 0; d < head_dim; ++d) {
                    accum[size_t(d)] += row_ptr[d];
                }
            } else if (q_f16 != nullptr) {
                const ggml_fp16_t* row_ptr = q_f16 + row_off;
                for (int d = 0; d < head_dim; ++d) {
                    accum[size_t(d)] += ggml_fp16_to_fp32(row_ptr[d]);
                }
            } else {
                for (int d = 0; d < head_dim; ++d) {
                    accum[size_t(d)] += ggml_ext_tensor_get_scalar_f32(q, d, t, h);
                }
            }
        }
        const size_t dst_off = size_t(qb) * size_t(head_dim);
        for (int d = 0; d < head_dim; ++d) {
            out[dst_off + size_t(d)] = ggml_fp32_to_fp16(accum[size_t(d)] * inv);
        }
    }
    if (trace_mask) {
        std::fprintf(stderr,
                     "block_sparse_mean_step q_flat_mean=%lldms ith=%d nth=%d n_q_blocks=%d heads=%d head_dim=%d block=%d\n",
                     static_cast<long long>(ggml_time_ms() - trace_begin),
                     ith,
                     nth,
                     n_q_blocks,
                     n_heads,
                     head_dim,
                     params->block);
    }
}

__STATIC_INLINE__ void ggml_ext_block_sparse_k_mean_custom(ggml_tensor* dst, int ith, int nth, void* userdata) {
    const bool trace_mask = sd_env_flag_enabled("SD_HTP_BLOCK_SPARSE_MASK_TRACE");
    const int64_t trace_begin = trace_mask ? ggml_time_ms() : 0;
    const auto* params = static_cast<const ggml_ext_block_sparse_mean_params*>(userdata);
    const ggml_tensor* k = dst->src[0];
    GGML_ASSERT(k != nullptr && params != nullptr);
    const int head_dim   = static_cast<int>(k->ne[0]);
    const int kv_len     = static_cast<int>(k->ne[1]);
    const int n_kv_heads = static_cast<int>(k->ne[2]);
    const int n_k_blocks = static_cast<int>(dst->ne[1]);
    auto* out            = reinterpret_cast<ggml_fp16_t*>(dst->data);
    const int k_blocks_per_head = (kv_len + params->block - 1) / params->block;
    GGML_ASSERT(n_k_blocks == n_kv_heads * k_blocks_per_head);
    thread_local std::vector<float> accum;
    accum.assign(static_cast<size_t>(head_dim), 0.0f);

    const float* k_f32 = k->type == GGML_TYPE_F32 ? reinterpret_cast<const float*>(k->data) : nullptr;
    const ggml_fp16_t* k_f16 = k->type == GGML_TYPE_F16 ? reinterpret_cast<const ggml_fp16_t*>(k->data) : nullptr;
    for (int kb = ith; kb < n_k_blocks; kb += nth) {
        const int h = kb / k_blocks_per_head;
        const int block_in_head = kb % k_blocks_per_head;
        const int begin = block_in_head * params->block;
        const int end   = std::min(kv_len, begin + params->block);
        const float inv = 1.0f / static_cast<float>(std::max(1, end - begin));
        std::fill(accum.begin(), accum.end(), 0.0f);
        for (int t = begin; t < end; ++t) {
            const size_t row_off = (size_t(h) * size_t(kv_len) + size_t(t)) * size_t(head_dim);
            if (k_f32 != nullptr) {
                const float* row_ptr = k_f32 + row_off;
                for (int d = 0; d < head_dim; ++d) {
                    accum[size_t(d)] += row_ptr[d];
                }
            } else if (k_f16 != nullptr) {
                const ggml_fp16_t* row_ptr = k_f16 + row_off;
                for (int d = 0; d < head_dim; ++d) {
                    accum[size_t(d)] += ggml_fp16_to_fp32(row_ptr[d]);
                }
            } else {
                for (int d = 0; d < head_dim; ++d) {
                    accum[size_t(d)] += ggml_ext_tensor_get_scalar_f32(k, d, t, h);
                }
            }
        }
        const size_t dst_off = size_t(kb) * size_t(head_dim);
        for (int d = 0; d < head_dim; ++d) {
            out[dst_off + size_t(d)] = ggml_fp32_to_fp16(accum[size_t(d)] * inv);
        }
    }
    if (trace_mask) {
        std::fprintf(stderr,
                     "block_sparse_mean_step k_flat_mean=%lldms ith=%d nth=%d n_k_blocks=%d heads=%d head_dim=%d block=%d\n",
                     static_cast<long long>(ggml_time_ms() - trace_begin),
                     ith,
                     nth,
                     n_k_blocks,
                     n_kv_heads,
                     head_dim,
                     params->block);
    }
}

__STATIC_INLINE__ ggml_tensor* ggml_ext_block_sparse_q_mean_node(ggml_context* ctx,
                                                                 ggml_tensor* q,
                                                                 int block_q) {
    GGML_ASSERT(q != nullptr && block_q > 0);
    const int n_q_blocks_per_head = static_cast<int>((q->ne[1] + block_q - 1) / block_q);
    const int n_q_blocks = static_cast<int>(q->ne[2]) * n_q_blocks_per_head;
    ggml_tensor* args[] = {q};
    auto* params = static_cast<ggml_ext_block_sparse_mean_params*>(
        ggml_new_buffer(ctx, sizeof(ggml_ext_block_sparse_mean_params)));
    params->block = block_q;
    return ggml_custom_4d(ctx,
                          GGML_TYPE_F16,
                          q->ne[0],
                          n_q_blocks,
                          1,
                          1,
                          args,
                          1,
                          ggml_ext_block_sparse_q_mean_custom,
                          GGML_N_TASKS_MAX,
                          params);
}

__STATIC_INLINE__ ggml_tensor* ggml_ext_block_sparse_k_mean_node(ggml_context* ctx,
                                                                 ggml_tensor* k,
                                                                 int block_k) {
    GGML_ASSERT(k != nullptr && block_k > 0);
    const int n_k_blocks_per_head = static_cast<int>((k->ne[1] + block_k - 1) / block_k);
    const int n_k_blocks = static_cast<int>(k->ne[2]) * n_k_blocks_per_head;
    ggml_tensor* args[] = {k};
    auto* params = static_cast<ggml_ext_block_sparse_mean_params*>(
        ggml_new_buffer(ctx, sizeof(ggml_ext_block_sparse_mean_params)));
    params->block = block_k;
    return ggml_custom_4d(ctx,
                          GGML_TYPE_F16,
                          k->ne[0],
                          n_k_blocks,
                          1,
                          1,
                          args,
                          1,
                          ggml_ext_block_sparse_k_mean_custom,
                          GGML_N_TASKS_MAX,
                          params);
}

struct ggml_ext_block_sparse_score_params {
    int block_q;
    int block_k;
    int causal;
};

__STATIC_INLINE__ void ggml_ext_block_sparse_score_custom(ggml_tensor* dst, int ith, int nth, void* userdata) {
    if (ith != 0) {
        return;
    }
    SD_UNUSED(nth);
    SD_UNUSED(userdata);
    const ggml_tensor* q_mean = dst->src[0];
    const ggml_tensor* k_mean = dst->src[1];
    GGML_ASSERT(q_mean != nullptr && k_mean != nullptr);
    const int head_dim   = static_cast<int>(q_mean->ne[0]);
    const int n_q_blocks = static_cast<int>(q_mean->ne[1]);
    const int n_k_blocks = static_cast<int>(k_mean->ne[1]);
    auto* out = reinterpret_cast<ggml_fp16_t*>(dst->data);
    for (int qb = 0; qb < n_q_blocks; ++qb) {
        for (int kb = 0; kb < n_k_blocks; ++kb) {
            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                const float qv = ggml_ext_tensor_get_scalar_f32(q_mean, d, qb);
                const float kv = ggml_ext_tensor_get_scalar_f32(k_mean, d, kb);
                dot += qv * kv;
            }
            out[size_t(qb) * n_k_blocks + kb] = ggml_fp32_to_fp16(dot);
        }
    }
}

__STATIC_INLINE__ ggml_tensor* ggml_ext_block_sparse_score_node(ggml_context* ctx,
                                                                ggml_tensor* q_mean,
                                                                ggml_tensor* k_mean,
                                                                int block_q,
                                                                int block_k,
                                                                bool causal) {
    GGML_ASSERT(q_mean != nullptr && k_mean != nullptr);
    const int n_q_blocks = static_cast<int>(q_mean->ne[1]);
    const int n_k_blocks = static_cast<int>(k_mean->ne[1]);
    ggml_tensor* args[] = {q_mean, k_mean};
    auto* params = static_cast<ggml_ext_block_sparse_score_params*>(
        ggml_new_buffer(ctx, sizeof(ggml_ext_block_sparse_score_params)));
    params->block_q = block_q;
    params->block_k = block_k;
    params->causal  = causal ? 1 : 0;
    ggml_tensor* result = ggml_custom_4d(ctx,
                                         GGML_TYPE_F16,
                                         n_k_blocks,
                                         n_q_blocks,
                                         1,
                                         1,
                                         args,
                                         2,
                                         ggml_ext_block_sparse_score_custom,
                                         1,
                                         params);
    ggml_set_name(result, "block_sparse_score");
    return result;
}

__STATIC_INLINE__ bool ggml_ext_block_overlaps_token_range(int block_index,
                                                          int block_size,
                                                          int protected_start,
                                                          int protected_len) {
    if (protected_len <= 0) {
        return false;
    }
    const int block_begin = block_index * block_size;
    const int block_end = block_begin + block_size;
    const int protected_end = protected_start + protected_len;
    return block_begin < protected_end && protected_start < block_end;
}

__STATIC_INLINE__ void ggml_ext_block_sparse_score_to_mask_rows(const ggml_tensor* score,
                                                                int32_t* block_mask,
                                                                int qb_start,
                                                                int qb_step,
                                                                int block_q,
                                                                int block_k,
                                                                bool causal = false,
                                                                int protected_start = 0,
                                                                int protected_len = 0,
                                                                int n_heads = 1,
                                                                int n_kv_heads = 1) {
    GGML_ASSERT(score != nullptr);
    GGML_ASSERT(score->type == GGML_TYPE_F16);
    GGML_ASSERT(block_mask != nullptr);
    GGML_ASSERT(qb_start >= 0);
    GGML_ASSERT(qb_step > 0);
    const int n_k_blocks = (int) score->ne[0];
    const int n_q_blocks = (int) score->ne[1];
    struct block_sparse_mask_candidate {
        float score;
        double weight;
        int kb;
    };
    std::vector<block_sparse_mask_candidate> candidates;
    candidates.reserve(size_t(n_k_blocks));

    n_heads = std::max(1, n_heads);
    n_kv_heads = std::max(1, n_kv_heads);
    protected_start = std::max(0, protected_start);
    protected_len = std::max(0, protected_len);
    const float keep_ratio = sd_htp_block_sparse_keep_ratio();
    const ggml_fp16_t* scores = reinterpret_cast<const ggml_fp16_t*>(score->data);
    GGML_ASSERT(n_heads % n_kv_heads == 0);
    const int gqa = std::max(1, n_heads / n_kv_heads);
    GGML_ASSERT(n_q_blocks % n_heads == 0);
    GGML_ASSERT(n_k_blocks % n_kv_heads == 0);
    const int q_blocks_per_head = n_q_blocks / n_heads;
    const int k_blocks_per_head = n_k_blocks / n_kv_heads;
    for (int qb = qb_start; qb < n_q_blocks; qb += qb_step) {
        const int q_head = qb / q_blocks_per_head;
        const int q_block_in_head = qb % q_blocks_per_head;
        const int kv_head = std::min(n_kv_heads - 1, q_head / gqa);
        const bool protect_q = ggml_ext_block_overlaps_token_range(q_block_in_head, block_q, protected_start, protected_len);
        for (int kb = 0; kb < n_k_blocks; ++kb) {
            const int k_head = kb / k_blocks_per_head;
            const int k_block_in_head = kb % k_blocks_per_head;
            const bool protect_k = ggml_ext_block_overlaps_token_range(k_block_in_head, block_k, protected_start, protected_len);
            if (k_head == kv_head && (protect_q || protect_k)) {
                block_mask[size_t(qb) * size_t(n_k_blocks) + size_t(kb)] = 0;
            } else {
                block_mask[size_t(qb) * size_t(n_k_blocks) + size_t(kb)] = 1;
            }
        }
    }
    for (int qb = qb_start; qb < n_q_blocks; qb += qb_step) {
        const int q_head = qb / q_blocks_per_head;
        const int q_block_in_head = qb % q_blocks_per_head;
        const int kv_head = std::min(n_kv_heads - 1, q_head / gqa);
        if (ggml_ext_block_overlaps_token_range(q_block_in_head, block_q, protected_start, protected_len)) {
            continue;
        }
        if (keep_ratio >= 1.0f) {
            for (int kb = 0; kb < n_k_blocks; ++kb) {
                const int k_head = kb / k_blocks_per_head;
                if (k_head != kv_head) {
                    continue;
                }
                const int k_block_in_head = kb % k_blocks_per_head;
                if (ggml_ext_block_overlaps_token_range(k_block_in_head, block_k, protected_start, protected_len)) {
                    continue;
                }
                if (causal && k_block_in_head * block_k > q_block_in_head * block_q + block_q - 1) {
                    continue;
                }
                block_mask[size_t(qb) * size_t(n_k_blocks) + size_t(kb)] = 0;
            }
            continue;
        }
        candidates.clear();
        for (int kb = 0; kb < n_k_blocks; ++kb) {
            const int k_head = kb / k_blocks_per_head;
            if (k_head != kv_head) {
                continue;
            }
            const int k_block_in_head = kb % k_blocks_per_head;
            if (ggml_ext_block_overlaps_token_range(k_block_in_head, block_k, protected_start, protected_len)) {
                continue;
            }
            if (causal && k_block_in_head * block_k > q_block_in_head * block_q + block_q - 1) {
                continue;
            }
            const float s = ggml_fp16_to_fp32(scores[size_t(qb) * size_t(n_k_blocks) + size_t(kb)]);
            candidates.push_back({s, 0.0, kb});
        }
        if (candidates.empty()) {
            continue;
        }
        std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
            if (a.score != b.score) {
                return a.score > b.score;
            }
            return a.kb < b.kb;
        });
        const float max_score = candidates.front().score;
        double total_weight = 0.0;
        for (auto& cand : candidates) {
            cand.weight = std::exp(double(cand.score - max_score));
            total_weight += cand.weight;
        }
        if (total_weight <= 0.0) {
            continue;
        }
        double kept_weight = 0.0;
        for (const auto& cand : candidates) {
            kept_weight += cand.weight;
            block_mask[size_t(qb) * size_t(n_k_blocks) + size_t(cand.kb)] = 0;
            if (kept_weight / total_weight >= keep_ratio) {
                break;
            }
        }
    }
}

__STATIC_INLINE__ std::vector<int32_t> ggml_ext_block_sparse_score_to_mask(const ggml_tensor* score,
                                                                           int block_q,
                                                                           int block_k,
                                                                           bool causal = false,
                                                                           int protected_start = 0,
                                                                           int protected_len = 0,
                                                                           int n_heads = 1,
                                                                           int n_kv_heads = 1) {
    GGML_ASSERT(score != nullptr);
    GGML_ASSERT(score->type == GGML_TYPE_F16);
    const int n_k_blocks = (int) score->ne[0];
    const int n_q_blocks = (int) score->ne[1];
    std::vector<int32_t> block_mask(size_t(n_q_blocks) * size_t(n_k_blocks), 1);
    ggml_ext_block_sparse_score_to_mask_rows(score,
                                             block_mask.data(),
                                             0,
                                             1,
                                             block_q,
                                             block_k,
                                             causal,
                                             protected_start,
                                             protected_len,
                                             n_heads,
                                             n_kv_heads);
    return block_mask;
}

__STATIC_INLINE__ int ggml_ext_block_sparse_max_kept_blocks(int n_q_blocks,
                                                            int n_k_blocks,
                                                            int n_heads,
                                                            int n_kv_heads) {
    n_heads = std::max(1, n_heads);
    n_kv_heads = std::max(1, n_kv_heads);
    GGML_ASSERT(n_q_blocks > 0 && n_k_blocks > 0);
    GGML_ASSERT(n_q_blocks % n_heads == 0);
    GGML_ASSERT(n_k_blocks % n_kv_heads == 0);
    return std::max(1, n_k_blocks / n_kv_heads);
}

__STATIC_INLINE__ void ggml_ext_block_sparse_score_to_index_rows(const ggml_tensor* score,
                                                                 int32_t* block_index,
                                                                 int max_kept,
                                                                 int qb_start,
                                                                 int qb_step,
                                                                 int block_q,
                                                                 int block_k,
                                                                 bool causal = false,
                                                                 int protected_start = 0,
                                                                 int protected_len = 0,
                                                                 int n_heads = 1,
                                                                 int n_kv_heads = 1) {
    GGML_ASSERT(score != nullptr);
    GGML_ASSERT(score->type == GGML_TYPE_F16);
    GGML_ASSERT(block_index != nullptr);
    GGML_ASSERT(max_kept > 0);
    GGML_ASSERT(qb_start >= 0);
    GGML_ASSERT(qb_step > 0);

    const int n_k_blocks = static_cast<int>(score->ne[0]);
    const int n_q_blocks = static_cast<int>(score->ne[1]);
    std::vector<int32_t> mask(size_t(n_q_blocks) * size_t(n_k_blocks), 1);
    ggml_ext_block_sparse_score_to_mask_rows(score,
                                             mask.data(),
                                             qb_start,
                                             qb_step,
                                             block_q,
                                             block_k,
                                             causal,
                                             protected_start,
                                             protected_len,
                                             n_heads,
                                             n_kv_heads);

    for (int qb = qb_start; qb < n_q_blocks; qb += qb_step) {
        int32_t* row = block_index + size_t(qb) * size_t(max_kept);
        std::fill(row, row + max_kept, -1);
        int kept = 0;
        for (int kb = 0; kb < n_k_blocks; ++kb) {
            if (mask[size_t(qb) * size_t(n_k_blocks) + size_t(kb)] != 0) {
                continue;
            }
            GGML_ASSERT(kept < max_kept);
            row[kept++] = kb;
        }
    }
}

__STATIC_INLINE__ void ggml_ext_block_sparse_trace_index_rows(const int32_t* block_index,
                                                              int max_kept,
                                                              int n_q_blocks,
                                                              int n_k_blocks) {
    const bool trace_rows = sd_env_flag_enabled("SD_HTP_BLOCK_SPARSE_MASK_TRACE");
    const bool trace_summary = trace_rows || sd_env_flag_enabled("SD_HTP_BLOCK_SPARSE_MASK_SUMMARY");
    if (!trace_summary) {
        return;
    }

    int min_kept = max_kept;
    int max_seen = 0;
    int64_t total_kept = 0;
    for (int qb = 0; qb < n_q_blocks; ++qb) {
        const int32_t* row = block_index + size_t(qb) * size_t(max_kept);
        int kept = 0;
        std::ostringstream kept_list;
        for (int i = 0; i < max_kept && row[i] >= 0; ++i) {
            if (trace_rows) {
                if (i != 0) {
                    kept_list << ',';
                }
                kept_list << row[i];
            }
            ++kept;
        }
        min_kept = std::min(min_kept, kept);
        max_seen = std::max(max_seen, kept);
        total_kept += kept;
        if (trace_rows) {
            std::fprintf(stderr,
                         "block_sparse_index_row qb=%d kept=%d/%d kept_k=[%s]\n",
                         qb,
                         kept,
                         n_k_blocks,
                         kept_list.str().c_str());
        }
    }

    const double avg_kept = n_q_blocks > 0 ? double(total_kept) / double(n_q_blocks) : 0.0;
    std::fprintf(stderr,
                 "block_sparse_index_rows end min_kept=%d max_kept=%d avg_kept=%.3f total_kept=%lld row_capacity=%d total_k_blocks=%d\n",
                 min_kept,
                 max_seen,
                 avg_kept,
                 static_cast<long long>(total_kept),
                 max_kept,
                 n_k_blocks);
}

__STATIC_INLINE__ void ggml_ext_block_sparse_trace_mask_rows(const std::vector<int32_t>& mask,
                                                            int n_q_blocks,
                                                            int n_k_blocks,
                                                            int block_q,
                                                            int block_k,
                                                            int protected_start,
                                                            int protected_len,
                                                            int n_heads,
                                                            int n_kv_heads,
                                                            float keep_ratio) {
    if (!sd_env_flag_enabled("SD_HTP_BLOCK_SPARSE_MASK_TRACE")) {
        return;
    }

    n_heads = std::max(1, n_heads);
    n_kv_heads = std::max(1, n_kv_heads);
    const int q_blocks_per_head = n_q_blocks / n_heads;
    const int k_blocks_per_head = n_k_blocks / n_kv_heads;
    protected_start = std::max(0, protected_start);
    protected_len = std::max(0, protected_len);

    int min_kept = n_k_blocks;
    int max_kept = 0;
    int64_t total_kept = 0;

    std::fprintf(stderr,
                 "block_sparse_mask_rows begin n_q_blocks=%d n_k_blocks=%d block_q=%d block_k=%d protected_start_tokens=%d protected_tokens=%d n_heads=%d n_kv_heads=%d keep_ratio=%.3f\n",
                 n_q_blocks,
                 n_k_blocks,
                 block_q,
                 block_k,
                 protected_start,
                 protected_len,
                 n_heads,
                 n_kv_heads,
                 keep_ratio);

    for (int qb = 0; qb < n_q_blocks; ++qb) {
        const int q_block_in_head = q_blocks_per_head > 0 ? qb % q_blocks_per_head : 0;
        const bool protect_q = ggml_ext_block_overlaps_token_range(q_block_in_head, block_q, protected_start, protected_len);
        int kept_count = 0;
        int protected_k_count = 0;
        std::ostringstream kept_list;
        bool first = true;
        for (int kb = 0; kb < n_k_blocks; ++kb) {
            const int k_block_in_head = k_blocks_per_head > 0 ? kb % k_blocks_per_head : 0;
            if (ggml_ext_block_overlaps_token_range(k_block_in_head, block_k, protected_start, protected_len)) {
                ++protected_k_count;
            }
            const int32_t v = mask[size_t(qb) * size_t(n_k_blocks) + size_t(kb)];
            if (v == 0) {
                ++kept_count;
                if (!first) {
                    kept_list << ',';
                }
                kept_list << kb;
                first = false;
            }
        }
        min_kept = std::min(min_kept, kept_count);
        max_kept = std::max(max_kept, kept_count);
        total_kept += kept_count;
        std::fprintf(stderr,
                     "block_sparse_mask_row qb=%d protect_q=%d kept=%d/%d protect_k=%d kept_k=[%s]\n",
                     qb,
                     protect_q ? 1 : 0,
                     kept_count,
                     n_k_blocks,
                     protected_k_count,
                     kept_list.str().c_str());
    }

    const double avg_kept = n_q_blocks > 0 ? double(total_kept) / double(n_q_blocks) : 0.0;
    std::fprintf(stderr,
                 "block_sparse_mask_rows end min_kept=%d max_kept=%d avg_kept=%.3f total_kept=%lld total=%lld\n",
                 min_kept,
                 max_kept,
                 avg_kept,
                 static_cast<long long>(total_kept),
                 static_cast<long long>(n_q_blocks) * static_cast<long long>(n_k_blocks));
}

struct ggml_ext_block_sparse_mask_from_score_params {
    int block_q;
    int block_k;
    int causal;
    int protected_start;
    int protected_len;
    int n_heads;
    int n_kv_heads;
};

struct ggml_ext_block_sparse_index_from_score_params {
    int block_q;
    int block_k;
    int causal;
    int protected_start;
    int protected_len;
    int n_heads;
    int n_kv_heads;
};

struct ggml_ext_block_sparse_index_32_from_16_score_params {
    int causal;
    int protected_start;
    int protected_len;
    int n_heads;
    int n_kv_heads;
};

struct ggml_ext_block_sparse_structural_index_32_params {
    int qo_len;
    int kv_len;
    int protected_start;
    int protected_len;
    int n_heads;
    int n_kv_heads;
    int patch_halo;
    int center_sink;
};

struct ggml_ext_block_sparse_direct_halo_index_32_params {
    int qo_len;
    int kv_len;
    int protected_start;
    int protected_len;
    int token_h;
    int token_w;
    int n_heads;
    int n_kv_heads;
    int center_sink;
};

__STATIC_INLINE__ void ggml_ext_block_sparse_full_keep_mask_custom(ggml_tensor* dst, int ith, int nth, void* userdata) {
    SD_UNUSED(nth);
    SD_UNUSED(userdata);
    if (ith != 0) {
        return;
    }
    GGML_ASSERT(dst != nullptr && dst->type == GGML_TYPE_I32);
    std::memset(dst->data, 0, static_cast<size_t>(ggml_nbytes(dst)));
    if (sd_env_flag_enabled("SD_HTP_BLOCK_SPARSE_MASK_TRACE")) {
        std::fprintf(stderr,
                     "block_sparse_mask_step full_keep=0ms n_q_blocks=%lld n_k_blocks=%lld\n",
                     static_cast<long long>(dst->ne[1]),
                     static_cast<long long>(dst->ne[0]));
    }
}

__STATIC_INLINE__ ggml_tensor* ggml_ext_block_sparse_full_keep_mask_node(ggml_context* ctx,
                                                                         int n_q_blocks,
                                                                         int n_k_blocks) {
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(n_q_blocks > 0 && n_k_blocks > 0);
    return ggml_custom_4d(ctx,
                          GGML_TYPE_I32,
                          n_k_blocks,
                          n_q_blocks,
                          1,
                          1,
                          nullptr,
                          0,
                          ggml_ext_block_sparse_full_keep_mask_custom,
                          1,
                          nullptr);
}

__STATIC_INLINE__ void ggml_ext_block_sparse_full_mask_custom(ggml_tensor* dst, int ith, int nth, void* userdata) {
    SD_UNUSED(nth);
    SD_UNUSED(userdata);
    if (ith != 0) {
        return;
    }
    GGML_ASSERT(dst != nullptr && dst->type == GGML_TYPE_I32);
    std::fill_n(static_cast<int32_t*>(dst->data), ggml_nelements(dst), 1);
    if (sd_env_flag_enabled("SD_HTP_BLOCK_SPARSE_MASK_TRACE")) {
        std::fprintf(stderr,
                     "block_sparse_mask_step full_mask=0ms n_q_blocks=%lld n_k_blocks=%lld\n",
                     static_cast<long long>(dst->ne[1]),
                     static_cast<long long>(dst->ne[0]));
    }
}

__STATIC_INLINE__ ggml_tensor* ggml_ext_block_sparse_full_mask_node(ggml_context* ctx,
                                                                    int n_q_blocks,
                                                                    int n_k_blocks) {
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(n_q_blocks > 0 && n_k_blocks > 0);
    return ggml_custom_4d(ctx,
                          GGML_TYPE_I32,
                          n_k_blocks,
                          n_q_blocks,
                          1,
                          1,
                          nullptr,
                          0,
                          ggml_ext_block_sparse_full_mask_custom,
                          1,
                          nullptr);
}

__STATIC_INLINE__ void ggml_ext_block_sparse_structural_index_32_custom(ggml_tensor* dst,
                                                                        int ith,
                                                                        int nth,
                                                                        void* userdata) {
    SD_UNUSED(nth);
    if (ith != 0) {
        return;
    }

    const auto* params = static_cast<const ggml_ext_block_sparse_structural_index_32_params*>(userdata);
    GGML_ASSERT(params != nullptr);
    GGML_ASSERT(dst != nullptr && dst->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->src[0] == nullptr);

    const int q16_per_head = (params->qo_len + SD_HTP_BLOCK_SPARSE_PATCH_BLOCK - 1) /
                             SD_HTP_BLOCK_SPARSE_PATCH_BLOCK;
    const int k16_per_head = (params->kv_len + SD_HTP_BLOCK_SPARSE_PATCH_BLOCK - 1) /
                             SD_HTP_BLOCK_SPARSE_PATCH_BLOCK;
    const int n_q16_blocks = params->n_heads * q16_per_head;
    const int n_k16_blocks = params->n_kv_heads * k16_per_head;
    const int image_q16_blocks = params->protected_len > 0
                                     ? std::min(q16_per_head,
                                                (params->protected_start + SD_HTP_BLOCK_SPARSE_PATCH_BLOCK - 1) /
                                                    SD_HTP_BLOCK_SPARSE_PATCH_BLOCK)
                                     : q16_per_head;
    const int image_k16_blocks = params->protected_len > 0
                                     ? std::min(k16_per_head,
                                                (params->protected_start + SD_HTP_BLOCK_SPARSE_PATCH_BLOCK - 1) /
                                                    SD_HTP_BLOCK_SPARSE_PATCH_BLOCK)
                                     : k16_per_head;

    std::vector<int32_t> mask16(size_t(n_q16_blocks) * size_t(n_k16_blocks), 1);
    if (params->patch_halo != 0) {
        ggml_ext_block_sparse_force_patch_halo_mask(mask16.data(),
                                                    n_q16_blocks,
                                                    n_k16_blocks,
                                                    image_q16_blocks,
                                                    image_k16_blocks,
                                                    params->n_heads,
                                                    params->n_kv_heads);
    }
    if (params->center_sink != 0) {
        ggml_ext_block_sparse_force_center_sink_mask(mask16.data(),
                                                     n_q16_blocks,
                                                     n_k16_blocks,
                                                     image_q16_blocks,
                                                     image_k16_blocks,
                                                     params->n_heads,
                                                     params->n_kv_heads);
    }

    ggml_ext_block_sparse_promote_16_mask_to_32_index(mask16.data(),
                                                      static_cast<int32_t*>(dst->data),
                                                      n_q16_blocks,
                                                      n_k16_blocks,
                                                      static_cast<int>(dst->ne[0]),
                                                      params->n_heads,
                                                      params->n_kv_heads);
}

__STATIC_INLINE__ ggml_tensor* ggml_ext_block_sparse_structural_index_32_node(ggml_context* ctx,
                                                                              int qo_len,
                                                                              int kv_len,
                                                                              int protected_start,
                                                                              int protected_len,
                                                                              int n_heads,
                                                                              int n_kv_heads,
                                                                              bool patch_halo,
                                                                              bool center_sink) {
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(qo_len > 0 && kv_len > 0);
    GGML_ASSERT(n_heads > 0 && n_kv_heads > 0);
    GGML_ASSERT(patch_halo || center_sink);

    const int q16_per_head = (qo_len + SD_HTP_BLOCK_SPARSE_PATCH_BLOCK - 1) /
                             SD_HTP_BLOCK_SPARSE_PATCH_BLOCK;
    const int k16_per_head = (kv_len + SD_HTP_BLOCK_SPARSE_PATCH_BLOCK - 1) /
                             SD_HTP_BLOCK_SPARSE_PATCH_BLOCK;
    const int n_q32_blocks = n_heads * ((q16_per_head + 1) / 2);
    const int max_kept32 = std::max(1, (k16_per_head + 1) / 2);

    auto* params = static_cast<ggml_ext_block_sparse_structural_index_32_params*>(
        ggml_new_buffer(ctx, sizeof(ggml_ext_block_sparse_structural_index_32_params)));
    params->qo_len = qo_len;
    params->kv_len = kv_len;
    params->protected_start = std::max(0, protected_start);
    params->protected_len = std::max(0, protected_len);
    params->n_heads = n_heads;
    params->n_kv_heads = n_kv_heads;
    params->patch_halo = patch_halo ? 1 : 0;
    params->center_sink = center_sink ? 1 : 0;

    ggml_tensor* result = ggml_custom_4d(ctx,
                                         GGML_TYPE_I32,
                                         max_kept32,
                                         n_q32_blocks,
                                         1,
                                         1,
                                         nullptr,
                                         0,
                                         ggml_ext_block_sparse_structural_index_32_custom,
                                         1,
                                         params);
    ggml_set_name(result, "block_sparse_index");
    return result;
}

__STATIC_INLINE__ void ggml_ext_block_sparse_direct_halo_index_32_custom(ggml_tensor* dst,
                                                                          int ith,
                                                                          int nth,
                                                                          void* userdata) {
    SD_UNUSED(nth);
    if (ith != 0) {
        return;
    }

    const auto* params = static_cast<const ggml_ext_block_sparse_direct_halo_index_32_params*>(userdata);
    GGML_ASSERT(params != nullptr);
    GGML_ASSERT(dst != nullptr && dst->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->src[0] == nullptr);
    const bool direct = ggml_ext_block_sparse_fill_direct_halo_index_32(
        static_cast<int32_t*>(dst->data),
        static_cast<int>(dst->ne[0]),
        params->qo_len,
        params->kv_len,
        params->protected_start,
        params->protected_len,
        params->token_h,
        params->token_w,
        params->n_heads,
        params->n_kv_heads,
        params->center_sink != 0);
    if (!direct) {
        ggml_ext_block_sparse_fill_full_index_32(static_cast<int32_t*>(dst->data),
                                                  static_cast<int>(dst->ne[0]),
                                                  params->qo_len,
                                                  params->kv_len,
                                                  params->n_heads,
                                                  params->n_kv_heads);
        if (sd_env_flag_enabled("SD_HTP_BLOCK_SPARSE_MASK_TRACE")) {
            std::fprintf(stderr,
                         "block_sparse_direct_halo fallback=full token_h=%d token_w=%d qo_len=%d kv_len=%d\n",
                         params->token_h,
                         params->token_w,
                         params->qo_len,
                         params->kv_len);
        }
    }
}

__STATIC_INLINE__ ggml_tensor* ggml_ext_block_sparse_direct_halo_index_32_node(ggml_context* ctx,
                                                                                int qo_len,
                                                                                int kv_len,
                                                                                int protected_start,
                                                                                int protected_len,
                                                                                int token_h,
                                                                                int token_w,
                                                                                int n_heads,
                                                                                int n_kv_heads,
                                                                                bool center_sink) {
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(qo_len > 0 && kv_len > 0);
    GGML_ASSERT(n_heads > 0 && n_kv_heads > 0);

    const int q_blocks_per_head = (qo_len + SD_HTP_BLOCK_SPARSE_MASK_BLOCK - 1) /
                                  SD_HTP_BLOCK_SPARSE_MASK_BLOCK;
    const int k_blocks_per_head = (kv_len + SD_HTP_BLOCK_SPARSE_MASK_BLOCK - 1) /
                                  SD_HTP_BLOCK_SPARSE_MASK_BLOCK;
    auto* params = static_cast<ggml_ext_block_sparse_direct_halo_index_32_params*>(
        ggml_new_buffer(ctx, sizeof(ggml_ext_block_sparse_direct_halo_index_32_params)));
    params->qo_len = qo_len;
    params->kv_len = kv_len;
    params->protected_start = std::max(0, protected_start);
    params->protected_len = std::max(0, protected_len);
    params->token_h = token_h;
    params->token_w = token_w;
    params->n_heads = n_heads;
    params->n_kv_heads = n_kv_heads;
    params->center_sink = center_sink ? 1 : 0;

    ggml_tensor* result = ggml_custom_4d(ctx,
                                         GGML_TYPE_I32,
                                         k_blocks_per_head,
                                         n_heads * q_blocks_per_head,
                                         1,
                                         1,
                                         nullptr,
                                         0,
                                         ggml_ext_block_sparse_direct_halo_index_32_custom,
                                         1,
                                         params);
    ggml_set_name(result, "block_sparse_index");
    return result;
}

__STATIC_INLINE__ void ggml_ext_block_sparse_mask_from_score_custom(ggml_tensor* dst, int ith, int nth, void* userdata) {
    const bool trace_mask = sd_env_flag_enabled("SD_HTP_BLOCK_SPARSE_MASK_TRACE");
    const int64_t trace_begin = trace_mask ? ggml_time_ms() : 0;
    const auto* params = static_cast<const ggml_ext_block_sparse_mask_from_score_params*>(userdata);
    GGML_ASSERT(params != nullptr);
    GGML_ASSERT(dst != nullptr && dst->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->src[0] != nullptr);

    ggml_ext_block_sparse_score_to_mask_rows(dst->src[0],
                                             static_cast<int32_t*>(dst->data),
                                             ith,
                                             nth,
                                             params->block_q,
                                             params->block_k,
                                             params->causal != 0,
                                             params->protected_start,
                                             params->protected_len,
                                             params->n_heads,
                                             params->n_kv_heads);
    if (trace_mask && ith == 0 && nth == 1) {
        std::vector<int32_t> mask(static_cast<int32_t*>(dst->data),
                                  static_cast<int32_t*>(dst->data) + ggml_nelements(dst));
        ggml_ext_block_sparse_trace_mask_rows(mask,
                                              static_cast<int>(dst->ne[1]),
                                              static_cast<int>(dst->ne[0]),
                                              params->block_q,
                                              params->block_k,
                                              params->protected_start,
                                              params->protected_len,
                                              params->n_heads,
                                              params->n_kv_heads,
                                              sd_htp_block_sparse_keep_ratio());
    }
    if (trace_mask && ith == 0) {
        std::fprintf(stderr,
                     "block_sparse_mask_step mask_from_score_thread0=%lldms n_q_blocks=%lld n_k_blocks=%lld block_q=%d block_k=%d protected_start_tokens=%d protected_tokens=%d n_heads=%d n_kv_heads=%d keep_ratio=%.3f nth=%d\n",
                     static_cast<long long>(ggml_time_ms() - trace_begin),
                     static_cast<long long>(dst->ne[1]),
                     static_cast<long long>(dst->ne[0]),
                     params->block_q,
                     params->block_k,
                     params->protected_start,
                     params->protected_len,
                     params->n_heads,
                     params->n_kv_heads,
                     sd_htp_block_sparse_keep_ratio(),
                     nth);
    }
}

__STATIC_INLINE__ ggml_tensor* ggml_ext_block_sparse_mask_from_score_node(ggml_context* ctx,
                                                                          ggml_tensor* score,
                                                                          int block_q,
                                                                          int block_k,
                                                                          bool causal,
                                                                          int protected_start,
                                                                          int protected_len,
                                                                          int n_heads,
                                                                          int n_kv_heads) {
    GGML_ASSERT(ctx != nullptr && score != nullptr);
    const int n_k_blocks = static_cast<int>(score->ne[0]);
    const int n_q_blocks = static_cast<int>(score->ne[1]);
    ggml_tensor* args[] = {score};
    auto* params = static_cast<ggml_ext_block_sparse_mask_from_score_params*>(
        ggml_new_buffer(ctx, sizeof(ggml_ext_block_sparse_mask_from_score_params)));
    params->block_q = block_q;
    params->block_k = block_k;
    params->causal  = causal ? 1 : 0;
    params->protected_start = std::max(0, protected_start);
    params->protected_len = std::max(0, protected_len);
    params->n_heads = std::max(1, n_heads);
    params->n_kv_heads = std::max(1, n_kv_heads);
    ggml_tensor* result = ggml_custom_4d(ctx,
                                         GGML_TYPE_I32,
                                         n_k_blocks,
                                         n_q_blocks,
                                         1,
                                         1,
                                         args,
                                         1,
                                         ggml_ext_block_sparse_mask_from_score_custom,
                                         GGML_N_TASKS_MAX,
                                         params);
    return result;
}

__STATIC_INLINE__ void ggml_ext_block_sparse_index_from_score_custom(ggml_tensor* dst,
                                                                     int ith,
                                                                     int nth,
                                                                     void* userdata) {
    const bool trace_index = sd_env_flag_enabled("SD_HTP_BLOCK_SPARSE_MASK_TRACE") ||
                             sd_env_flag_enabled("SD_HTP_BLOCK_SPARSE_MASK_SUMMARY");
    const int64_t trace_begin = trace_index ? ggml_time_ms() : 0;
    const auto* params = static_cast<const ggml_ext_block_sparse_index_from_score_params*>(userdata);
    GGML_ASSERT(params != nullptr);
    GGML_ASSERT(dst != nullptr && dst->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->src[0] != nullptr);

    ggml_ext_block_sparse_score_to_index_rows(dst->src[0],
                                              static_cast<int32_t*>(dst->data),
                                              static_cast<int>(dst->ne[0]),
                                              ith,
                                              nth,
                                              params->block_q,
                                              params->block_k,
                                              params->causal != 0,
                                              params->protected_start,
                                              params->protected_len,
                                              params->n_heads,
                                              params->n_kv_heads);

    if (trace_index && ith == 0) {
        std::fprintf(stderr,
                     "block_sparse_index_step index_from_score_thread0=%lldms n_q_blocks=%lld n_k_blocks=%lld max_kept=%lld block_q=%d block_k=%d protected_start_tokens=%d protected_tokens=%d n_heads=%d n_kv_heads=%d keep_ratio=%.3f nth=%d\n",
                     static_cast<long long>(ggml_time_ms() - trace_begin),
                     static_cast<long long>(dst->ne[1]),
                     static_cast<long long>(dst->src[0]->ne[0]),
                     static_cast<long long>(dst->ne[0]),
                     params->block_q,
                     params->block_k,
                     params->protected_start,
                     params->protected_len,
                     params->n_heads,
                     params->n_kv_heads,
                     sd_htp_block_sparse_keep_ratio(),
                     nth);
    }
    if (trace_index && ith == 0 && nth == 1) {
        ggml_ext_block_sparse_trace_index_rows(static_cast<const int32_t*>(dst->data),
                                               static_cast<int>(dst->ne[0]),
                                               static_cast<int>(dst->ne[1]),
                                               static_cast<int>(dst->src[0]->ne[0]));
    }
}

__STATIC_INLINE__ ggml_tensor* ggml_ext_block_sparse_index_from_score_node(ggml_context* ctx,
                                                                           ggml_tensor* score,
                                                                           int block_q,
                                                                           int block_k,
                                                                           bool causal,
                                                                           int protected_start,
                                                                           int protected_len,
                                                                           int n_heads,
                                                                           int n_kv_heads) {
    GGML_ASSERT(ctx != nullptr && score != nullptr);
    const int n_k_blocks = static_cast<int>(score->ne[0]);
    const int n_q_blocks = static_cast<int>(score->ne[1]);
    const int max_kept = ggml_ext_block_sparse_max_kept_blocks(n_q_blocks, n_k_blocks, n_heads, n_kv_heads);
    ggml_tensor* args[] = {score};
    auto* params = static_cast<ggml_ext_block_sparse_index_from_score_params*>(
        ggml_new_buffer(ctx, sizeof(ggml_ext_block_sparse_index_from_score_params)));
    params->block_q = block_q;
    params->block_k = block_k;
    params->causal  = causal ? 1 : 0;
    params->protected_start = std::max(0, protected_start);
    params->protected_len = std::max(0, protected_len);
    params->n_heads = std::max(1, n_heads);
    params->n_kv_heads = std::max(1, n_kv_heads);
    ggml_tensor* result = ggml_custom_4d(ctx,
                                         GGML_TYPE_I32,
                                         max_kept,
                                         n_q_blocks,
                                         1,
                                         1,
                                         args,
                                         1,
                                         ggml_ext_block_sparse_index_from_score_custom,
                                         GGML_N_TASKS_MAX,
                                         params);
    ggml_set_name(result, "block_sparse_index");
    return result;
}

__STATIC_INLINE__ void ggml_ext_block_sparse_index_32_from_16_score_custom(ggml_tensor* dst,
                                                                           int ith,
                                                                           int nth,
                                                                           void* userdata) {
    SD_UNUSED(nth);
    if (ith != 0) {
        return;
    }

    const bool trace_index = sd_env_flag_enabled("SD_HTP_BLOCK_SPARSE_MASK_TRACE") ||
                             sd_env_flag_enabled("SD_HTP_BLOCK_SPARSE_MASK_SUMMARY");
    const int64_t trace_begin = trace_index ? ggml_time_ms() : 0;
    const auto* params = static_cast<const ggml_ext_block_sparse_index_32_from_16_score_params*>(userdata);
    GGML_ASSERT(params != nullptr);
    GGML_ASSERT(dst != nullptr && dst->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->src[0] != nullptr);

    const ggml_tensor* score = dst->src[0];
    const int n_k16_blocks = static_cast<int>(score->ne[0]);
    const int n_q16_blocks = static_cast<int>(score->ne[1]);
    std::vector<int32_t> mask16(size_t(n_q16_blocks) * size_t(n_k16_blocks), 1);
    ggml_ext_block_sparse_score_to_mask_rows(score,
                                             mask16.data(),
                                             0,
                                             1,
                                             SD_HTP_BLOCK_SPARSE_PATCH_BLOCK,
                                             SD_HTP_BLOCK_SPARSE_PATCH_BLOCK,
                                             params->causal != 0,
                                             params->protected_start,
                                             params->protected_len,
                                             params->n_heads,
                                             params->n_kv_heads);
    const int q16_per_head = n_q16_blocks / std::max(1, params->n_heads);
    const int k16_per_head = n_k16_blocks / std::max(1, params->n_kv_heads);
    const int image_q16_blocks = params->protected_len > 0 ?
        std::min(q16_per_head, (params->protected_start + SD_HTP_BLOCK_SPARSE_PATCH_BLOCK - 1) / SD_HTP_BLOCK_SPARSE_PATCH_BLOCK) :
        q16_per_head;
    const int image_k16_blocks = params->protected_len > 0 ?
        std::min(k16_per_head, (params->protected_start + SD_HTP_BLOCK_SPARSE_PATCH_BLOCK - 1) / SD_HTP_BLOCK_SPARSE_PATCH_BLOCK) :
        k16_per_head;
    const bool patch_halo_enabled = sd_env_flag_enabled("SD_HTP_BLOCK_SPARSE_PATCH_HALO");
    if (patch_halo_enabled) {
        ggml_ext_block_sparse_force_patch_halo_mask(mask16.data(),
                                                    n_q16_blocks,
                                                    n_k16_blocks,
                                                    image_q16_blocks,
                                                    image_k16_blocks,
                                                    params->n_heads,
                                                    params->n_kv_heads);
    }
    const bool center_sink_enabled = sd_env_flag_enabled("SD_HTP_BLOCK_SPARSE_CENTER_SINK");
    if (center_sink_enabled) {
        ggml_ext_block_sparse_force_center_sink_mask(mask16.data(),
                                                     n_q16_blocks,
                                                     n_k16_blocks,
                                                     image_q16_blocks,
                                                     image_k16_blocks,
                                                     params->n_heads,
                                                     params->n_kv_heads);
    }

    ggml_ext_block_sparse_promote_16_mask_to_32_index(mask16.data(),
                                                      static_cast<int32_t*>(dst->data),
                                                      n_q16_blocks,
                                                      n_k16_blocks,
                                                      static_cast<int>(dst->ne[0]),
                                                      params->n_heads,
                                                      params->n_kv_heads);

    if (trace_index) {
        std::fprintf(stderr,
                     "block_sparse_index_step index_32_from_16_score=%lldms n_q16_blocks=%d n_k16_blocks=%d n_q32_blocks=%lld n_k32_blocks=%lld max_kept=%lld protected_start_tokens=%d protected_tokens=%d n_heads=%d n_kv_heads=%d keep_ratio=%.3f patch_halo=%d center_sink=%d\n",
                     static_cast<long long>(ggml_time_ms() - trace_begin),
                     n_q16_blocks,
                     n_k16_blocks,
                     static_cast<long long>(dst->ne[1]),
                     static_cast<long long>(params->n_kv_heads * ((n_k16_blocks / params->n_kv_heads + 1) / 2)),
                     static_cast<long long>(dst->ne[0]),
                     params->protected_start,
                     params->protected_len,
                     params->n_heads,
                     params->n_kv_heads,
                     sd_htp_block_sparse_keep_ratio(),
                     patch_halo_enabled ? 1 : 0,
                     center_sink_enabled ? 1 : 0);
        ggml_ext_block_sparse_trace_index_rows(static_cast<const int32_t*>(dst->data),
                                               static_cast<int>(dst->ne[0]),
                                               static_cast<int>(dst->ne[1]),
                                               static_cast<int>(params->n_kv_heads * ((n_k16_blocks / params->n_kv_heads + 1) / 2)));
    }
}

__STATIC_INLINE__ ggml_tensor* ggml_ext_block_sparse_index_32_from_16_score_node(ggml_context* ctx,
                                                                                 ggml_tensor* score,
                                                                                 bool causal,
                                                                                 int protected_start,
                                                                                 int protected_len,
                                                                                 int n_heads,
                                                                                 int n_kv_heads) {
    GGML_ASSERT(ctx != nullptr && score != nullptr);
    n_heads = std::max(1, n_heads);
    n_kv_heads = std::max(1, n_kv_heads);
    const int n_k16_blocks = static_cast<int>(score->ne[0]);
    const int n_q16_blocks = static_cast<int>(score->ne[1]);
    GGML_ASSERT(n_q16_blocks > 0 && n_k16_blocks > 0);
    GGML_ASSERT(n_q16_blocks % n_heads == 0);
    GGML_ASSERT(n_k16_blocks % n_kv_heads == 0);

    const int q16_per_head = n_q16_blocks / n_heads;
    const int k16_per_head = n_k16_blocks / n_kv_heads;
    const int q32_per_head = (q16_per_head + 1) / 2;
    const int k32_per_head = (k16_per_head + 1) / 2;
    const int n_q32_blocks = n_heads * q32_per_head;
    const int n_k32_blocks = n_kv_heads * k32_per_head;
    const int max_kept32 = std::max(1, k32_per_head);

    ggml_tensor* args[] = {score};
    auto* params = static_cast<ggml_ext_block_sparse_index_32_from_16_score_params*>(
        ggml_new_buffer(ctx, sizeof(ggml_ext_block_sparse_index_32_from_16_score_params)));
    params->causal = causal ? 1 : 0;
    params->protected_start = std::max(0, protected_start);
    params->protected_len = std::max(0, protected_len);
    params->n_heads = n_heads;
    params->n_kv_heads = n_kv_heads;

    ggml_tensor* result = ggml_custom_4d(ctx,
                                         GGML_TYPE_I32,
                                         max_kept32,
                                         n_q32_blocks,
                                         1,
                                         1,
                                         args,
                                         1,
                                         ggml_ext_block_sparse_index_32_from_16_score_custom,
                                         1,
                                         params);
    ggml_set_name(result, "block_sparse_index");
    return result;
}

__STATIC_INLINE__ ggml_tensor* ggml_ext_block_sparse_mask_mean_topk(ggml_context* ctx,
                                                                    const ggml_tensor* q,
                                                                    const ggml_tensor* k,
                                                                    int block_q,
                                                                    int block_k,
                                                                    int top_k,
                                                                    bool causal = false,
                                                                    int protected_prefix_len = 0) {
    std::vector<int32_t> mask =
        ggml_ext_block_sparse_mask_mean_topk(q, k, block_q, block_k, top_k, causal, protected_prefix_len);
    const int n_q_blocks       = static_cast<int>((q->ne[1] + block_q - 1) / block_q);
    const int n_k_blocks       = static_cast<int>((k->ne[1] + block_k - 1) / block_k);
    GGML_ASSERT(static_cast<int>(mask.size()) == n_q_blocks * n_k_blocks);

    ggml_tensor* result = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_k_blocks, n_q_blocks);
    if (result->data != nullptr) {
        std::memcpy(result->data, mask.data(), mask.size() * sizeof(int32_t));
    } else {
        ggml_backend_tensor_set(result, mask.data(), 0, mask.size() * sizeof(int32_t));
    }
    return result;
}

struct ggml_ext_block_sparse_mask_topk_params {
    int block_q;
    int block_k;
    int top_k;
    int causal;
};

__STATIC_INLINE__ void ggml_ext_block_sparse_mask_mean_topk_custom(ggml_tensor* dst, int ith, int nth, void* userdata) {
    SD_UNUSED(nth);
    if (ith != 0) {
        return;
    }

    const auto* params = static_cast<const ggml_ext_block_sparse_mask_topk_params*>(userdata);
    GGML_ASSERT(params != nullptr);
    GGML_ASSERT(dst != nullptr && dst->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->src[0] != nullptr && dst->src[1] != nullptr);

    std::vector<int32_t> mask = ggml_ext_block_sparse_mask_mean_topk(dst->src[0],
                                                                      dst->src[1],
                                                                      params->block_q,
                                                                      params->block_k,
                                                                      params->top_k,
                                                                      params->causal != 0);
    GGML_ASSERT(static_cast<int64_t>(mask.size()) == ggml_nelements(dst));
    std::memcpy(dst->data, mask.data(), mask.size() * sizeof(int32_t));
}

__STATIC_INLINE__ ggml_tensor* ggml_ext_block_sparse_mask_mean_topk_node(ggml_context* ctx,
                                                                         ggml_tensor* q,
                                                                         ggml_tensor* k,
                                                                         int block_q,
                                                                         int block_k,
                                                                         int top_k,
                                                                         bool causal) {
    GGML_ASSERT(ctx != nullptr && q != nullptr && k != nullptr);
    GGML_ASSERT(block_q > 0 && block_k > 0 && top_k > 0);

    const int n_q_blocks = static_cast<int>((q->ne[1] + block_q - 1) / block_q);
    const int n_k_blocks = static_cast<int>((k->ne[1] + block_k - 1) / block_k);
    GGML_ASSERT(n_q_blocks > 0 && n_k_blocks > 0);

    auto* params = static_cast<ggml_ext_block_sparse_mask_topk_params*>(
        ggml_new_buffer(ctx, sizeof(ggml_ext_block_sparse_mask_topk_params)));
    params->block_q = block_q;
    params->block_k = block_k;
    params->top_k   = std::max(1, std::min(top_k, n_k_blocks));
    params->causal  = causal ? 1 : 0;

    ggml_tensor* args[] = {q, k};
    ggml_tensor* result = ggml_custom_4d(ctx,
                                         GGML_TYPE_I32,
                                         n_k_blocks,
                                         n_q_blocks,
                                         1,
                                         1,
                                         args,
                                         2,
                                         ggml_ext_block_sparse_mask_mean_topk_custom,
                                         1,
                                         params);
    return result;
}

__STATIC_INLINE__ ggml_tensor* vector_to_ggml_tensor(ggml_context* ctx,
                                                     const std::vector<float>& vec) {
    ggml_tensor* t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, vec.size());
    memcpy(t->data, (const void*)vec.data(), ggml_nbytes(t));
    return t;
}

__STATIC_INLINE__ ggml_tensor* vector_to_ggml_tensor_i32(ggml_context* ctx,
                                                         const std::vector<int>& vec) {
    ggml_tensor* t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, vec.size());
    memcpy(t->data, (const void*)vec.data(), ggml_nbytes(t));
    return t;
}

__STATIC_INLINE__ std::vector<float> arange(float start, float end, float step = 1.f) {
    std::vector<float> result;

    for (float value = start; value < end; value += step) {
        result.push_back(value);
    }

    return result;
}

// Ref: https://github.com/CompVis/stable-diffusion/blob/main/ldm/modules/diffusionmodules/util.py#L151
__STATIC_INLINE__ std::vector<float> timestep_embedding(std::vector<float> timesteps,
                                                        int dim,
                                                        int max_period       = 10000,
                                                        bool flip_sin_to_cos = true,
                                                        float scale          = 1.f) {
    // timesteps: [N,]
    // embedding: [N, dim]
    size_t N = timesteps.size();
    std::vector<float> embedding(N * dim, 0.f);
    int half = dim / 2;
    std::vector<float> freqs(half);
    for (int i = 0; i < half; ++i) {
        freqs[i] = (float)std::exp(-std::log(max_period) * i / half);
    }
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < half; ++j) {
            float arg = timesteps[i] * freqs[j] * scale;
            if (flip_sin_to_cos) {
                embedding[i * dim + j]        = std::cos(arg);
                embedding[i * dim + j + half] = std::sin(arg);
            } else {
                embedding[i * dim + j]        = std::sin(arg);
                embedding[i * dim + j + half] = std::cos(arg);
            }
        }
    }
    return embedding;
}

__STATIC_INLINE__ void set_timestep_embedding(std::vector<float> timesteps,
                                              ggml_tensor* embedding,
                                              int dim,
                                              int max_period = 10000) {
    std::vector<float> embedding_vec = timestep_embedding(timesteps, dim, max_period);
    memcpy(((char*)embedding->data), ((char*)embedding_vec.data()), ggml_nbytes(embedding));
}

__STATIC_INLINE__ void set_timestep_embedding(std::vector<float> timesteps,
                                              sd::Tensor<float>* embedding,
                                              int dim,
                                              int max_period = 10000) {
    GGML_ASSERT(embedding != nullptr);
    std::vector<float> embedding_vec = timestep_embedding(timesteps, dim, max_period);
    if (embedding->numel() != static_cast<int64_t>(embedding_vec.size())) {
        embedding->resize({dim, static_cast<int64_t>(timesteps.size())});
    }
    std::copy(embedding_vec.begin(), embedding_vec.end(), embedding->values().begin());
}

__STATIC_INLINE__ ggml_tensor* new_timestep_embedding(ggml_context* ctx,
                                                      std::vector<float> timesteps,
                                                      int dim,
                                                      int max_period = 10000) {
    // timesteps: [N,]
    // embedding: [N, dim]
    std::vector<float> embedding_vec = timestep_embedding(timesteps, dim, max_period);
    ggml_tensor* embedding           = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, timesteps.size());
    if (embedding->data != nullptr) {
        memcpy(((char*)embedding->data), ((char*)embedding_vec.data()), ggml_nbytes(embedding));
    } else {
        ggml_backend_tensor_set(embedding, embedding_vec.data(), 0, ggml_nbytes(embedding));
    }
    return embedding;
}

__STATIC_INLINE__ ggml_tensor* ggml_ext_timestep_embedding(
    ggml_context* ctx,
    ggml_tensor* timesteps,
    int dim,
    int max_period    = 10000,
    float time_factor = 1.0f) {
    timesteps = ggml_ext_scale(ctx, timesteps, time_factor);
    return ggml_timestep_embedding(ctx, timesteps, dim, max_period);
}

__STATIC_INLINE__ size_t ggml_tensor_num(ggml_context* ctx) {
    size_t num = 0;
    for (ggml_tensor* t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
        num++;
    }
    return num;
}

__STATIC_INLINE__ ggml_tensor* ggml_ext_vec_concat(ggml_context* ctx,
                                                   std::vector<ggml_tensor*>& tensors,
                                                   int dim) {
    while (tensors.size() > 1) {
        std::vector<ggml_tensor*> next_level;
        for (size_t i = 0; i < tensors.size(); i += 2) {
            if (i + 1 < tensors.size()) {
                next_level.push_back(ggml_concat(ctx, tensors[i], tensors[i + 1], dim));
            } else {
                next_level.push_back(tensors[i]);
            }
        }
        tensors = std::move(next_level);
    }
    return tensors[0];
}

/* Reserve room for large model graphs. */
#define MAX_PARAMS_TENSOR_NUM 32768
#define MAX_GRAPH_SIZE 327680

struct WeightAdapter {
    struct ForwardParams {
        enum class op_type_t {
            OP_LINEAR,
            OP_CONV2D,
        } op_type;
        struct {
            bool force_prec_f32 = false;
            float scale         = 1.f;
        } linear;
        struct conv2d_params_t {
            int s0          = 1;
            int s1          = 1;
            int p0          = 0;
            int p1          = 0;
            int d0          = 1;
            int d1          = 1;
            bool direct     = false;
            bool circular_x = false;
            bool circular_y = false;
            float scale     = 1.f;
        } conv2d;
    };
    virtual ggml_tensor* patch_weight(ggml_context* ctx, ggml_backend_t backend, ggml_tensor* weight, const std::string& weight_name) = 0;
    virtual ggml_tensor* forward(ggml_context* ctx,
                                           ggml_backend_t backend,
                                           ggml_tensor* x,
                                           ggml_tensor* w,
                                           ggml_tensor* b,
                                           const std::string& prefix,
                                           ForwardParams forward_params)                                                              = 0;
    virtual size_t get_extra_graph_size()                                                                                             = 0;
};

__STATIC_INLINE__ ggml_tensor* sd_cache_tensor_and_expand_graph(
    ggml_cgraph* graph,
    const std::function<ggml_tensor*(const std::string&, ggml_tensor*)>& cache_tensor,
    const std::string& name,
    ggml_tensor* tensor) {
    if (!cache_tensor || tensor == nullptr) {
        return nullptr;
    }
    ggml_tensor* persisted = cache_tensor(name, tensor);
    if (graph != nullptr && persisted != nullptr) {
        ggml_set_output(persisted);
        ggml_build_forward_expand(graph, persisted);
    }
    return persisted;
}

__STATIC_INLINE__ ggml_tensor*
sd_cache_tensor_and_expand_local_residual_graph(
    ggml_cgraph* graph,
    const std::function<ggml_tensor*(const std::string&, ggml_tensor*)>& cache_tensor,
    const std::string& name,
    ggml_tensor* tensor) {
    if (!cache_tensor || tensor == nullptr) {
        return nullptr;
    }
    ggml_tensor* persisted = cache_tensor(name, tensor);
    if (graph == nullptr || persisted == nullptr) {
        return persisted;
    }

    ggml_tensor* sparse = persisted->src[1];
    if (sparse != nullptr && sparse->src[3] != nullptr) {
        ggml_build_forward_expand(graph, sparse->src[3]);
    }
    ggml_build_forward_expand(graph, persisted);
    return persisted;
}

__STATIC_INLINE__ bool sd_htp_tensor_reaches_through_views(
    const ggml_tensor* tensor,
    const ggml_tensor* target) {
    for (size_t depth = 0; tensor != nullptr && depth < 64; ++depth) {
        if (tensor == target) {
            return true;
        }
        if (tensor->op != GGML_OP_VIEW &&
            tensor->op != GGML_OP_RESHAPE &&
            tensor->op != GGML_OP_PERMUTE &&
            tensor->op != GGML_OP_TRANSPOSE) {
            return false;
        }
        tensor = tensor->src[0] != nullptr ? tensor->src[0] : tensor->view_src;
    }
    return false;
}

__STATIC_INLINE__ std::vector<const ggml_tensor*>
sd_collect_local_residual_owned_sparse_producers(ggml_cgraph* graph) {
    std::vector<const ggml_tensor*> producers;
    if (graph == nullptr) {
        return producers;
    }

    const int node_count = ggml_graph_n_nodes(graph);
    for (int i = 0; i < node_count; ++i) {
        const ggml_tensor* residual = ggml_graph_node(graph, i);
        if (residual == nullptr || residual->op != GGML_OP_CUSTOM ||
            std::strncmp(residual->name,
                         "htp_block_sparse_residual_b", 27) != 0) {
            continue;
        }
        const ggml_tensor* dense = residual->src[0];
        const ggml_tensor* sparse = residual->src[1];
        const ggml_tensor* rho = residual->src[2];
        if (dense == nullptr || sparse == nullptr || rho == nullptr ||
            rho->op != GGML_OP_CUSTOM ||
            std::strncmp(rho->name,
                         "htp_block_sparse_residual_rho_b", 31) != 0 ||
            rho->src[0] != dense || rho->src[1] != sparse) {
            continue;
        }

        bool has_ordinary_consumer = false;
        for (int consumer_index = 0;
             consumer_index < node_count && !has_ordinary_consumer;
             ++consumer_index) {
            const ggml_tensor* consumer =
                ggml_graph_node(graph, consumer_index);
            if (consumer == nullptr || consumer == rho || consumer == residual) {
                continue;
            }
            for (int source = 0; source < GGML_MAX_SRC; ++source) {
                if (sd_htp_tensor_reaches_through_views(
                        consumer->src[source], sparse)) {
                    has_ordinary_consumer = true;
                    break;
                }
            }
        }
        if (!has_ordinary_consumer &&
            std::find(producers.begin(), producers.end(), sparse) ==
                producers.end()) {
            producers.push_back(sparse);
        }
    }
    return producers;
}

enum sd_htp_residual_pair_role {
    SD_HTP_RESIDUAL_PAIR_DENSE = 0,
    SD_HTP_RESIDUAL_PAIR_SPARSE,
    SD_HTP_RESIDUAL_PAIR_RHO,
    SD_HTP_RESIDUAL_PAIR_RESIDUAL,
};

enum sd_htp_graph_tensor_location {
    SD_HTP_GRAPH_TENSOR_NODE = 0,
    SD_HTP_GRAPH_TENSOR_LEAF,
    SD_HTP_GRAPH_TENSOR_SOURCE_ONLY,
    SD_HTP_GRAPH_TENSOR_MISSING,
};

enum sd_htp_buffer_class {
    SD_HTP_BUFFER_NONE = 0,
    SD_HTP_BUFFER_RUNTIME,
    SD_HTP_BUFFER_HTP,
    SD_HTP_BUFFER_QKV,
    SD_HTP_BUFFER_PERSISTENT,
    SD_HTP_BUFFER_UNKNOWN,
};

struct sd_htp_residual_pair_allocation_record {
    size_t pair_id = 0;
    sd_htp_residual_pair_role role = SD_HTP_RESIDUAL_PAIR_DENSE;
    const ggml_tensor* tensor = nullptr;
    const ggml_tensor* view_root = nullptr;
    sd_htp_graph_tensor_location graph_location = SD_HTP_GRAPH_TENSOR_MISSING;
    int graph_index = -1;
    int planned_buffer_id = -1;
    int root_planned_buffer_id = -1;
    size_t ordinary_consumer_count = 0;
    bool no_alloc_direct = false;
    bool no_alloc_root = false;
    sd_htp_buffer_class actual_buffer_class = SD_HTP_BUFFER_NONE;
    void* data = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_backend_buffer_t resolved_backing = nullptr;
};

struct sd_htp_residual_pair_allocation_summary {
    size_t dense_outputs_expected = 0;
    size_t dense_outputs_with_backing = 0;
    size_t dense_outputs_no_alloc = 0;
    size_t metadata_only_sparse_without_backing = 0;
    size_t allocated_sparse_with_backing = 0;
    size_t persistent_outputs_expected = 0;
    size_t persistent_outputs_with_backing = 0;
};

__STATIC_INLINE__ const ggml_tensor* sd_htp_allocation_root(
    const ggml_tensor* tensor) {
    for (size_t depth = 0; tensor != nullptr && depth < 64; ++depth) {
        if (tensor->view_src != nullptr) {
            tensor = tensor->view_src;
            continue;
        }
        const bool view_only = tensor->op == GGML_OP_VIEW ||
                               tensor->op == GGML_OP_RESHAPE ||
                               tensor->op == GGML_OP_PERMUTE ||
                               tensor->op == GGML_OP_TRANSPOSE;
        if (view_only && tensor->src[0] != nullptr) {
            tensor = tensor->src[0];
            continue;
        }
        return tensor;
    }
    return tensor;
}

__STATIC_INLINE__ ggml_backend_buffer_t sd_htp_resolved_backing(
    const ggml_tensor* tensor) {
    for (size_t depth = 0; tensor != nullptr && depth < 64; ++depth) {
        if (tensor->buffer != nullptr) {
            return tensor->buffer;
        }
        if (tensor->view_src != nullptr) {
            tensor = tensor->view_src;
            continue;
        }
        const bool view_only = tensor->op == GGML_OP_VIEW ||
                               tensor->op == GGML_OP_RESHAPE ||
                               tensor->op == GGML_OP_PERMUTE ||
                               tensor->op == GGML_OP_TRANSPOSE;
        tensor = view_only ? tensor->src[0] : nullptr;
    }
    return nullptr;
}

__STATIC_INLINE__ sd_htp_graph_tensor_location
sd_htp_graph_tensor_location_of(
    ggml_cgraph* graph,
    const ggml_tensor* tensor,
    int* graph_index) {
    if (graph_index != nullptr) {
        *graph_index = -1;
    }
    if (graph == nullptr || tensor == nullptr) {
        return SD_HTP_GRAPH_TENSOR_MISSING;
    }
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        if (ggml_graph_node(graph, i) == tensor) {
            if (graph_index != nullptr) *graph_index = i;
            return SD_HTP_GRAPH_TENSOR_NODE;
        }
    }
    for (int i = 0; i < sd::ggml_graph_cut::leaf_count(graph); ++i) {
        if (sd::ggml_graph_cut::leaf_tensor(graph, i) == tensor) {
            if (graph_index != nullptr) *graph_index = i;
            return SD_HTP_GRAPH_TENSOR_LEAF;
        }
    }
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        const ggml_tensor* node = ggml_graph_node(graph, i);
        for (int source = 0; node != nullptr && source < GGML_MAX_SRC; ++source) {
            if (sd_htp_tensor_reaches_through_views(node->src[source], tensor)) {
                if (graph_index != nullptr) *graph_index = i;
                return SD_HTP_GRAPH_TENSOR_SOURCE_ONLY;
            }
        }
    }
    return SD_HTP_GRAPH_TENSOR_MISSING;
}

__STATIC_INLINE__ int sd_htp_planned_buffer_id(
    ggml_cgraph* graph,
    const ggml_tensor* tensor,
    const std::vector<int>& node_buffer_ids,
    const std::vector<int>& leaf_buffer_ids) {
    int index = -1;
    const sd_htp_graph_tensor_location location =
        sd_htp_graph_tensor_location_of(graph, tensor, &index);
    if (location == SD_HTP_GRAPH_TENSOR_NODE && index >= 0 &&
        static_cast<size_t>(index) < node_buffer_ids.size()) {
        return node_buffer_ids[index];
    }
    if (location == SD_HTP_GRAPH_TENSOR_LEAF && index >= 0 &&
        static_cast<size_t>(index) < leaf_buffer_ids.size()) {
        return leaf_buffer_ids[index];
    }
    return -1;
}

__STATIC_INLINE__ size_t sd_htp_ordinary_consumer_count(
    ggml_cgraph* graph,
    const ggml_tensor* tensor,
    const ggml_tensor* rho,
    const ggml_tensor* residual) {
    size_t count = 0;
    if (graph == nullptr || tensor == nullptr) {
        return count;
    }
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        const ggml_tensor* consumer = ggml_graph_node(graph, i);
        if (consumer == nullptr || consumer == rho || consumer == residual) {
            continue;
        }
        for (int source = 0; source < GGML_MAX_SRC; ++source) {
            if (sd_htp_tensor_reaches_through_views(
                    consumer->src[source], tensor)) {
                ++count;
                break;
            }
        }
    }
    return count;
}

__STATIC_INLINE__ sd_htp_buffer_class sd_htp_classify_buffer(
    ggml_backend_buffer_t backing,
    ggml_backend_buffer_type_t runtime_buft,
    ggml_backend_buffer_type_t htp_buft,
    ggml_backend_buffer_type_t qkv_buft,
    const std::unordered_set<ggml_backend_buffer_t>& persistent_backings) {
    if (backing == nullptr) {
        return SD_HTP_BUFFER_NONE;
    }
    if (persistent_backings.find(backing) != persistent_backings.end()) {
        return SD_HTP_BUFFER_PERSISTENT;
    }
    const ggml_backend_buffer_type_t buft =
        ggml_backend_buffer_get_type(backing);
    if (qkv_buft != nullptr && buft == qkv_buft) {
        return SD_HTP_BUFFER_QKV;
    }
    if (htp_buft != nullptr && buft == htp_buft) {
        return SD_HTP_BUFFER_HTP;
    }
    if (runtime_buft != nullptr && buft == runtime_buft) {
        return SD_HTP_BUFFER_RUNTIME;
    }
    return SD_HTP_BUFFER_UNKNOWN;
}

__STATIC_INLINE__ std::vector<sd_htp_residual_pair_allocation_record>
sd_collect_htp_residual_pair_allocation_records(
    ggml_cgraph* graph,
    const std::vector<const ggml_tensor*>& no_alloc_tensors,
    const std::vector<int>& node_buffer_ids,
    const std::vector<int>& leaf_buffer_ids,
    ggml_backend_buffer_type_t runtime_buft,
    ggml_backend_buffer_type_t htp_buft,
    ggml_backend_buffer_type_t qkv_buft,
    const std::unordered_set<ggml_backend_buffer_t>& persistent_backings) {
    std::vector<sd_htp_residual_pair_allocation_record> records;
    if (graph == nullptr) {
        return records;
    }
    size_t pair_id = 0;
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        const ggml_tensor* residual = ggml_graph_node(graph, i);
        if (residual == nullptr || residual->op != GGML_OP_CUSTOM ||
            std::strncmp(residual->name,
                         "htp_block_sparse_residual_b", 27) != 0) {
            continue;
        }
        const ggml_tensor* dense = residual->src[0];
        const ggml_tensor* sparse = residual->src[1];
        const ggml_tensor* rho = residual->src[2];
        if (dense == nullptr || sparse == nullptr || rho == nullptr ||
            rho->op != GGML_OP_CUSTOM ||
            std::strncmp(rho->name,
                         "htp_block_sparse_residual_rho_b", 31) != 0 ||
            rho->src[0] != dense || rho->src[1] != sparse) {
            continue;
        }
        ++pair_id;
        const std::pair<sd_htp_residual_pair_role, const ggml_tensor*> roles[] = {
            {SD_HTP_RESIDUAL_PAIR_DENSE, dense},
            {SD_HTP_RESIDUAL_PAIR_SPARSE, sparse},
            {SD_HTP_RESIDUAL_PAIR_RHO, rho},
            {SD_HTP_RESIDUAL_PAIR_RESIDUAL, residual},
        };
        for (const auto& role : roles) {
            sd_htp_residual_pair_allocation_record record;
            record.pair_id = pair_id;
            record.role = role.first;
            record.tensor = role.second;
            record.view_root = sd_htp_allocation_root(role.second);
            record.graph_location = sd_htp_graph_tensor_location_of(
                graph, role.second, &record.graph_index);
            record.planned_buffer_id = sd_htp_planned_buffer_id(
                graph, role.second, node_buffer_ids, leaf_buffer_ids);
            record.root_planned_buffer_id = sd_htp_planned_buffer_id(
                graph, record.view_root, node_buffer_ids, leaf_buffer_ids);
            record.ordinary_consumer_count = sd_htp_ordinary_consumer_count(
                graph, role.second, rho, residual);
            record.no_alloc_direct = std::find(
                no_alloc_tensors.begin(), no_alloc_tensors.end(), role.second) !=
                no_alloc_tensors.end();
            record.no_alloc_root = std::find(
                no_alloc_tensors.begin(), no_alloc_tensors.end(), record.view_root) !=
                no_alloc_tensors.end();
            record.data = role.second->data;
            record.buffer = role.second->buffer;
            record.resolved_backing = sd_htp_resolved_backing(role.second);
            record.actual_buffer_class = sd_htp_classify_buffer(
                record.resolved_backing, runtime_buft, htp_buft, qkv_buft,
                persistent_backings);
            records.push_back(record);
        }
    }
    return records;
}

__STATIC_INLINE__ sd_htp_residual_pair_allocation_summary
sd_summarize_htp_residual_pair_allocation_records(
    const std::vector<sd_htp_residual_pair_allocation_record>& records) {
    sd_htp_residual_pair_allocation_summary summary;
    for (const auto& record : records) {
        switch (record.role) {
            case SD_HTP_RESIDUAL_PAIR_DENSE:
                ++summary.dense_outputs_expected;
                summary.dense_outputs_with_backing +=
                    record.resolved_backing != nullptr ? 1 : 0;
                summary.dense_outputs_no_alloc +=
                    record.no_alloc_direct || record.no_alloc_root ? 1 : 0;
                break;
            case SD_HTP_RESIDUAL_PAIR_SPARSE:
                summary.metadata_only_sparse_without_backing +=
                    (record.no_alloc_direct || record.no_alloc_root) &&
                            record.resolved_backing == nullptr
                        ? 1 : 0;
                summary.allocated_sparse_with_backing +=
                    record.resolved_backing != nullptr ? 1 : 0;
                break;
            case SD_HTP_RESIDUAL_PAIR_RHO:
            case SD_HTP_RESIDUAL_PAIR_RESIDUAL:
                ++summary.persistent_outputs_expected;
                summary.persistent_outputs_with_backing +=
                    record.actual_buffer_class == SD_HTP_BUFFER_PERSISTENT
                        ? 1 : 0;
                break;
        }
    }
    return summary;
}

__STATIC_INLINE__ const char* sd_htp_residual_pair_role_name(
    sd_htp_residual_pair_role role) {
    switch (role) {
        case SD_HTP_RESIDUAL_PAIR_DENSE: return "dense";
        case SD_HTP_RESIDUAL_PAIR_SPARSE: return "sparse";
        case SD_HTP_RESIDUAL_PAIR_RHO: return "rho";
        case SD_HTP_RESIDUAL_PAIR_RESIDUAL: return "residual";
    }
    return "unknown";
}

__STATIC_INLINE__ const char* sd_htp_graph_tensor_location_name(
    sd_htp_graph_tensor_location location) {
    switch (location) {
        case SD_HTP_GRAPH_TENSOR_NODE: return "node";
        case SD_HTP_GRAPH_TENSOR_LEAF: return "leaf";
        case SD_HTP_GRAPH_TENSOR_SOURCE_ONLY: return "source-only";
        case SD_HTP_GRAPH_TENSOR_MISSING: return "missing";
    }
    return "missing";
}

__STATIC_INLINE__ const char* sd_htp_buffer_class_name(
    sd_htp_buffer_class buffer_class) {
    switch (buffer_class) {
        case SD_HTP_BUFFER_NONE: return "none";
        case SD_HTP_BUFFER_RUNTIME: return "runtime";
        case SD_HTP_BUFFER_HTP: return "htp";
        case SD_HTP_BUFFER_QKV: return "qkv";
        case SD_HTP_BUFFER_PERSISTENT: return "persistent";
        case SD_HTP_BUFFER_UNKNOWN: return "unknown";
    }
    return "unknown";
}

__STATIC_INLINE__ int sd_format_htp_residual_pair_allocation_record(
    char* buffer,
    size_t capacity,
    const char* checkpoint,
    const sd_htp_residual_pair_allocation_record& record) {
    if (buffer == nullptr || capacity == 0) {
        return -1;
    }
    return std::snprintf(
        buffer, capacity,
        "htp_opencl_pair_allocation checkpoint=%s pair_id=%zu role=%s "
        "tensor=%p name=%s op=%s graph_location=%s graph_index=%d "
        "view_src=%p view_root=%p ordinary_consumers=%zu "
        "no_alloc_direct=%d no_alloc_root=%d planned_buffer_id=%d "
        "root_planned_buffer_id=%d actual_buffer_class=%s data=%p "
        "buffer=%p resolved_backing=%p",
        checkpoint != nullptr ? checkpoint : "unknown", record.pair_id,
        sd_htp_residual_pair_role_name(record.role),
        static_cast<const void*>(record.tensor),
        record.tensor != nullptr && record.tensor->name[0] != '\0'
            ? record.tensor->name : "<unnamed>",
        record.tensor != nullptr ? ggml_op_name(record.tensor->op) : "<null>",
        sd_htp_graph_tensor_location_name(record.graph_location),
        record.graph_index,
        record.tensor != nullptr
            ? static_cast<const void*>(record.tensor->view_src) : nullptr,
        static_cast<const void*>(record.view_root),
        record.ordinary_consumer_count,
        record.no_alloc_direct ? 1 : 0,
        record.no_alloc_root ? 1 : 0,
        record.planned_buffer_id,
        record.root_planned_buffer_id,
        sd_htp_buffer_class_name(record.actual_buffer_class),
        record.data,
        static_cast<void*>(record.buffer),
        static_cast<void*>(record.resolved_backing));
}

__STATIC_INLINE__ int sd_format_htp_residual_pair_allocation_summary(
    char* buffer,
    size_t capacity,
    const char* checkpoint,
    const sd_htp_residual_pair_allocation_summary& summary) {
    if (buffer == nullptr || capacity == 0) {
        return -1;
    }
    return std::snprintf(
        buffer, capacity,
        "htp_opencl_memory_plan_contract checkpoint=%s "
        "dense_outputs_expected=%zu dense_outputs_with_backing=%zu "
        "dense_outputs_no_alloc=%zu "
        "metadata_only_sparse_without_backing=%zu "
        "allocated_sparse_with_backing=%zu "
        "persistent_outputs_expected=%zu "
        "persistent_outputs_with_backing=%zu",
        checkpoint != nullptr ? checkpoint : "unknown",
        summary.dense_outputs_expected,
        summary.dense_outputs_with_backing,
        summary.dense_outputs_no_alloc,
        summary.metadata_only_sparse_without_backing,
        summary.allocated_sparse_with_backing,
        summary.persistent_outputs_expected,
        summary.persistent_outputs_with_backing);
}

struct GGMLRunnerContext {
    ggml_backend_t backend                                           = nullptr;
    ggml_context* ggml_ctx                                           = nullptr;
    bool flash_attn_enabled                                          = false;
    bool block_sparse_attn_enabled                                   = false;
    bool htp_attention_enabled                                       = false;
    bool conv2d_direct_enabled                                       = false;
    bool circular_x_enabled                                          = false;
    bool circular_y_enabled                                          = false;
    int block_sparse_residual_mode                                   = SD_HTP_BLOCK_SPARSE_RESIDUAL_OFF;
    float block_sparse_residual_scale                                = 1.0f;
    float block_sparse_residual_rho                                  = 1.0f;
    bool block_sparse_residual_exact_rho                             = false;
    int block_sparse_residual_branch_index                           = 0;
    int block_sparse_residual_attention_index                        = 0;
    int block_sparse_token_h                                         = 0;
    int block_sparse_token_w                                         = 0;
    bool z_image_has_ref_latents                                     = false;
    std::shared_ptr<WeightAdapter> weight_adapter                    = nullptr;
    std::vector<std::pair<ggml_tensor*, std::string>>* debug_tensors = nullptr;
    std::function<ggml_tensor*(const std::string&)> get_cache_tensor;
    std::function<ggml_tensor*(const std::string&, ggml_tensor*)> cache_tensor;
    std::function<void(const std::string&, ggml_tensor*)> bind_cache_tensor;

    void capture_tensor(const std::string& name, ggml_tensor* tensor) {
        if (debug_tensors == nullptr || tensor == nullptr) {
            return;
        }
        ggml_tensor* snapshot = ggml_cont(ggml_ctx, tensor);
        ggml_tensor* dst      = ggml_dup_tensor(ggml_ctx, snapshot);
        snapshot              = ggml_cpy(ggml_ctx, snapshot, dst);
        ggml_set_output(snapshot);
        debug_tensors->push_back({snapshot, name});
    }

    ggml_tensor* load_cache_tensor(const std::string& name) const {
        if (!get_cache_tensor) {
            return nullptr;
        }
        return get_cache_tensor(name);
    }

    void persist_cache_tensor(const std::string& name, ggml_tensor* tensor) const {
        if (!cache_tensor || tensor == nullptr) {
            return;
        }
        cache_tensor(name, tensor);
    }

    ggml_tensor* cache_tensor_as_graph_input(const std::string& name) const {
        ggml_tensor* cached = load_cache_tensor(name);
        if (cached == nullptr || bind_cache_tensor == nullptr || ggml_ctx == nullptr) {
            return nullptr;
        }
        ggml_tensor* graph_tensor = ggml_dup_tensor(ggml_ctx, cached);
        ggml_set_name(graph_tensor, name.c_str());
        bind_cache_tensor(name, graph_tensor);
        return graph_tensor;
    }
};

// Local residual expansion is shared by fused, OpenCL sparse, and HTP/CPU
// anchor routes.  The route-specific producer ownership is handled later by
// the allocator/sidecar; graph ordering must stay common here.
__STATIC_INLINE__ void sd_install_local_residual_graph_expansion(
    ggml_cgraph* graph,
    int block_sparse_residual_mode,
    GGMLRunnerContext* runner_ctx) {
    if (graph == nullptr || runner_ctx == nullptr ||
        block_sparse_residual_mode != SD_HTP_BLOCK_SPARSE_RESIDUAL_ANCHOR ||
        !sd_htp_opencl_local_lifetime_prerequisites()) {
        return;
    }
    const auto cache_tensor = runner_ctx->cache_tensor;
    runner_ctx->cache_tensor = [graph, cache_tensor](
                                   const std::string& name,
                                   ggml_tensor* tensor) {
        return sd_cache_tensor_and_expand_local_residual_graph(
            graph, cache_tensor, name, tensor);
    };
}

__STATIC_INLINE__ sd_htp_residual_binding sd_htp_prepare_residual_binding(
    GGMLRunnerContext* ctx,
    const char* attention_kind,
    bool residual_eligible) {
    sd_htp_residual_binding binding;
    if (ctx == nullptr || !residual_eligible) {
        return binding;
    }

    binding.slot = ctx->block_sparse_residual_attention_index++;
    if (ctx->block_sparse_residual_mode == SD_HTP_BLOCK_SPARSE_RESIDUAL_OFF) {
        return binding;
    }

    binding.residual_key = sd_htp_block_sparse_residual_cache_key(
        ctx->block_sparse_residual_branch_index,
        attention_kind,
        binding.slot);
    if (ctx->block_sparse_residual_exact_rho) {
        binding.rho_key = sd_htp_block_sparse_residual_rho_cache_key(
            ctx->block_sparse_residual_branch_index,
            attention_kind,
            binding.slot);
    }
    if (ctx->block_sparse_residual_mode == SD_HTP_BLOCK_SPARSE_RESIDUAL_APPLY) {
        binding.cached_residual = ctx->cache_tensor_as_graph_input(binding.residual_key);
        if (!binding.rho_key.empty()) {
            binding.cached_rho = ctx->cache_tensor_as_graph_input(binding.rho_key);
        }
    }
    return binding;
}

struct GGMLRunner {
protected:
    typedef std::function<ggml_cgraph*()> get_graph_cb_t;
    using GraphCutSegment = sd::ggml_graph_cut::Segment;
    using GraphCutPlan    = sd::ggml_graph_cut::Plan;

    ggml_backend_t params_backend  = nullptr;
    ggml_backend_t runtime_backend = nullptr;
    // 只有 diffusion 推理明确注入 HTP backend 后，runner 才会启用 RPCMEM 放置和 hybrid compute 判定。
    ggml_backend_t htp_backend     = nullptr;
    std::unique_ptr<HtpPackedStagingPool> htp_packed_staging_pool_;
    HtpPackedStagingPool::TypedLoadFn htp_packed_staging_loader_;
    std::vector<HtpPackedStagingPool::Handle> htp_packed_staging_handles_;
    std::vector<ggml_tensor*> htp_packed_staging_tensors_;
    bool preserve_htp_packed_staging_ = false;
    int htp_packed_staging_owner_count_ = 1;
    int htp_packed_staging_pool_owner_count_ = 0;

    ggml_context* params_ctx                    = nullptr;
    ggml_backend_buffer_t params_buffer         = nullptr;
    HtpNonDeferredParamsAllocation non_deferred_params_allocation_;
    ggml_context* offload_ctx                   = nullptr;
    ggml_backend_buffer_t runtime_params_buffer = nullptr;
    bool params_on_runtime_backend              = false;
    // 普通 runtime offload 现在改为显式记录“哪些参数被临时换到 runtime backend”，这样就能跳过常驻 RPCMEM 的 HTP 权重。
    std::vector<std::pair<ggml_tensor*, ggml_tensor*>> runtime_offload_pairs_;
    // 参照 llama.cpp-npu，eligible diffusion 权重在参数分配阶段直接放进 RPCMEM，而不是图执行前再复制一份常驻权重池。
    ggml_context* htp_param_placement_ctx_            = nullptr;
    ggml_backend_buffer_t htp_param_placement_buffer_ = nullptr;
    std::vector<ggml_context*> htp_param_placement_ctxs_;
    std::vector<ggml_backend_buffer_t> htp_param_placement_buffers_;
    std::vector<std::pair<ggml_tensor*, ggml_tensor*>> htp_param_placement_pairs_;
    std::unordered_set<ggml_tensor*> htp_placed_param_tensors_;
    std::vector<ggml_tensor*> htp_packed_self_registered_weights_;

    struct CacheStorageSegment {
        ggml_context* ctx = nullptr;
        ggml_backend_buffer_t buffer = nullptr;
        std::unordered_set<std::string> names;
    };
    std::vector<CacheStorageSegment> cache_storage_segments_;

    struct AsyncResidualCacheGeneration {
        uint64_t generation = 0;
        ggml_backend_htp_async_cache_slot* slot = nullptr;
        size_t consumer_count = 0;
        bool producer_pending = false;
    };
    std::unordered_map<std::string, AsyncResidualCacheGeneration>
        async_residual_cache_slots_;
    std::map<ggml_tensor*, uint64_t> cache_backed_tensor_generation_map_;
    std::vector<std::pair<std::string, uint64_t>> active_async_cache_consumers_;
    uint64_t next_async_residual_cache_generation_ = 1;
    uint64_t active_async_residual_cache_generation_ = 0;

    struct AsyncResidualCacheOutputBinding {
        ggml_tensor* tensor = nullptr;
        ggml_backend_buffer_t buffer = nullptr;
        void* data = nullptr;
        void* extra = nullptr;
    };

    struct AsyncResidualCacheOutputTransaction {
        uint64_t generation = 0;
        std::vector<AsyncResidualCacheOutputBinding> bindings;
        bool active = false;
    };

    struct CacheInputBinding {
        ggml_tensor* tensor = nullptr;
        ggml_backend_buffer_t buffer = nullptr;
        void* data = nullptr;
        void* extra = nullptr;
        enum ggml_op op = GGML_OP_NONE;
        ggml_tensor* src[GGML_MAX_SRC] = {};
        std::string key;
        uint64_t generation = 0;
        bool consumer_acquired = false;
    };

    struct CacheInputBindingTransaction {
        std::vector<CacheInputBinding> bindings;
        bool active = false;
        bool committed = false;
    };

    enum class CacheInputAllocationFailurePoint {
        none,
        first_reserve,
        second_alloc,
    };

    ggml_context* compute_ctx    = nullptr;
    ggml_gallocr* compute_allocr = nullptr;
    struct RetiredComputeArenaGeneration {
        uint64_t generation = 0;
        ggml_gallocr* allocator = nullptr;
        size_t bytes = 0;
        bool qkv_released = false;
    };
    std::deque<RetiredComputeArenaGeneration> retired_compute_arenas_;
    uint64_t next_compute_arena_generation_ = 1;
    uint64_t active_compute_arena_generation_ = 0;

    ggml_context* partial_offload_ctx                   = nullptr;
    ggml_backend_buffer_t partial_runtime_params_buffer = nullptr;
    std::vector<std::pair<ggml_tensor*, ggml_tensor*>> partial_offload_pairs;
    size_t max_graph_vram_bytes = 0;

    std::shared_ptr<WeightAdapter> weight_adapter = nullptr;

    std::vector<float> one_vec = {1.f};
    ggml_tensor* one_tensor    = nullptr;

    std::vector<int> zero_int_vec = {0};
    ggml_tensor* zero_int_tensor  = nullptr;

    std::map<ggml_tensor*, const void*> backend_tensor_data_map;
    std::map<ggml_tensor*, std::string> cache_backed_tensor_name_map;
    std::map<std::string, ggml_tensor*> cache_tensor_map;  // name -> tensor
    std::vector<std::pair<ggml_tensor*, std::string>> debug_tensors;
    const std::string final_result_name = "ggml_runner_final_result_tensor";

    bool flash_attn_enabled    = false;
    bool block_sparse_attn_enabled = false;
    bool htp_attention_enabled  = false;
    bool conv2d_direct_enabled = false;
    bool circular_x_enabled    = false;
    bool circular_y_enabled    = false;

    sd::ggml_graph_cut::PlanCache graph_cut_plan_cache_;
    std::unordered_set<const ggml_tensor*> params_tensor_set_;
    std::vector<ggml_tensor*> params_buffer_tensors_;
    // 记录每个 node/leaf 应该落在哪个 buffer，0=默认 runtime buffer，1=HTP RPCMEM buffer。
    std::vector<int> compute_node_buffer_ids_;
    std::vector<int> compute_leaf_buffer_ids_;
    std::vector<const ggml_tensor*> compute_no_alloc_tensors_;

    template <typename T>
    static sd::Tensor<T> take_or_empty(std::optional<sd::Tensor<T>> tensor) {
        if (!tensor.has_value()) {
            return {};
        }
        return std::move(*tensor);
    }

    template <typename T>
    static sd::Tensor<T> restore_trailing_singleton_dims(std::optional<sd::Tensor<T>> tensor,
                                                         size_t expected_dim) {
        return restore_trailing_singleton_dims(take_or_empty(std::move(tensor)), expected_dim);
    }

    template <typename T>
    static sd::Tensor<T> restore_trailing_singleton_dims(sd::Tensor<T> tensor,
                                                         size_t expected_dim) {
        if (tensor.empty()) {
            return tensor;
        }
        while (static_cast<size_t>(tensor.dim()) < expected_dim) {
            tensor.unsqueeze_(tensor.dim());
        }
        return tensor;
    }

    void alloc_params_ctx() {
        ggml_init_params params;
        params.mem_size   = static_cast<size_t>(MAX_PARAMS_TENSOR_NUM * ggml_tensor_overhead());
        params.mem_buffer = nullptr;
        params.no_alloc   = true;

        params_ctx = ggml_init(params);
        GGML_ASSERT(params_ctx != nullptr);
        params_tensor_set_.clear();
        if (params_backend != runtime_backend) {
            offload_ctx = ggml_init(params);
            GGML_ASSERT(offload_ctx != nullptr);
        }
    }

    void free_params_ctx() {
        if (params_ctx != nullptr) {
            ggml_free(params_ctx);
            params_ctx = nullptr;
        }
        params_tensor_set_.clear();
        if (offload_ctx != nullptr) {
            ggml_free(offload_ctx);
            offload_ctx = nullptr;
        }
        runtime_offload_pairs_.clear();
        free_htp_param_placement();
        if (partial_offload_ctx != nullptr) {
            ggml_free(partial_offload_ctx);
            partial_offload_ctx = nullptr;
        }
    }

    ggml_context* new_cache_ctx() {
        ggml_init_params params;
        params.mem_size   = static_cast<size_t>(MAX_PARAMS_TENSOR_NUM * ggml_tensor_overhead());
        params.mem_buffer = nullptr;
        params.no_alloc   = true;

        ggml_context* ctx = ggml_init(params);
        GGML_ASSERT(ctx != nullptr);
        return ctx;
    }

    void free_cache_ctx() {
        for (CacheStorageSegment& segment : cache_storage_segments_) {
            if (segment.ctx != nullptr) {
                ggml_free(segment.ctx);
                segment.ctx = nullptr;
            }
        }
        cache_storage_segments_.clear();
    }

    void free_async_residual_cache_slots() {
        for (auto& keyed_slots : async_residual_cache_slots_) {
            sd_backend_htp_async_cache_slot_free(keyed_slots.second.slot);
            keyed_slots.second.slot = nullptr;
        }
        async_residual_cache_slots_.clear();
    }

    void trace_memory_ledger_teardown() const {
        if (!ggml_backend_htp_memory_ledger_trace_enabled() ||
            htp_backend == nullptr) {
            return;
        }
        ggml_htp_lifetime_snapshot lifetime = {};
        if (!sd_backend_htp_get_lifetime_snapshot(htp_backend, &lifetime)) {
            return;
        }
        ggml_htp_memory_ledger_snapshot snapshot = {};
        snapshot.requested_route = GGML_HTP_ROUTE_INVALID;
        snapshot.preflight_result = -1;
        snapshot.live_imports = lifetime.live_imports;
        snapshot.live_events = lifetime.live_events;
        snapshot.live_leases = lifetime.live_leases;
        snapshot.pending_jobs = lifetime.pending_jobs;
        snapshot.retired_arenas = std::max(
            retired_compute_arenas_.size(), lifetime.retired_arenas);
        char line[2048] = {};
        if (ggml_backend_htp_format_memory_ledger(
                line, sizeof(line), "teardown", &snapshot) > 0) {
            std::fprintf(stderr, "%s\n", line);
        }
    }

    AsyncResidualCacheGeneration* find_async_residual_cache_slot(
            const std::string& key,
            uint64_t generation) {
        auto found = async_residual_cache_slots_.find(key);
        if (found == async_residual_cache_slots_.end()) {
            return nullptr;
        }
        if (found->second.generation == generation) {
            return &found->second;
        }
        return nullptr;
    }

    AsyncResidualCacheGeneration* latest_async_residual_cache_slot(
            const std::string& key) {
        auto found = async_residual_cache_slots_.find(key);
        if (found == async_residual_cache_slots_.end() || found->second.slot == nullptr) {
            return nullptr;
        }
        return &found->second;
    }

    bool async_residual_cache_backed(const std::string& key) const {
        const auto found = async_residual_cache_slots_.find(key);
        return found != async_residual_cache_slots_.end() &&
               found->second.slot != nullptr;
    }

    bool all_cache_outputs_async_residual_backed() const {
        if (cache_tensor_map.empty()) {
            return false;
        }
        for (const auto& entry : cache_tensor_map) {
            if (!async_residual_cache_backed(entry.first)) {
                return false;
            }
        }
        return true;
    }

    void release_active_async_cache_consumers() {
        for (const auto& binding : active_async_cache_consumers_) {
            AsyncResidualCacheGeneration* generation =
                find_async_residual_cache_slot(binding.first, binding.second);
            if (generation == nullptr) {
                continue;
            }
            GGML_ASSERT(generation->consumer_count > 0);
            --generation->consumer_count;
            bool ready = false;
            char reason[512] = {};
            if (htp_backend != nullptr &&
                sd_backend_htp_opencl_cache_query(
                    htp_backend, binding.first.c_str(), binding.second,
                    &ready, reason, sizeof(reason)) && ready) {
                generation->producer_pending = false;
            }
        }
        active_async_cache_consumers_.clear();
    }

    AsyncResidualCacheGeneration* allocate_async_residual_cache_slot(
            const std::string& key,
            const ggml_tensor* metadata,
            uint64_t generation) {
        if (htp_backend == nullptr || metadata == nullptr || key.empty()) {
            return nullptr;
        }
        if (AsyncResidualCacheGeneration* existing =
                find_async_residual_cache_slot(key, generation)) {
            return existing;
        }
        auto existing = async_residual_cache_slots_.find(key);
        if (existing != async_residual_cache_slots_.end()) {
            LOG_ERROR("%s residual cache key still has a live generation: %s",
                      get_desc().c_str(), key.c_str());
            return nullptr;
        }

        char reason[512] = {};
        ggml_backend_htp_async_cache_slot* slot =
            sd_backend_htp_async_cache_slot_create(
                htp_backend,
                key.c_str(),
                generation,
                metadata->type,
                metadata->ne,
                reason,
                sizeof(reason));
        if (slot == nullptr) {
            LOG_ERROR("allocate async residual cache slot %s failed: %s",
                      key.c_str(), reason[0] != '\0' ? reason : "unknown error");
            return nullptr;
        }
        auto inserted = async_residual_cache_slots_.emplace(
            key, AsyncResidualCacheGeneration{ generation, slot, 0, false });
        if (!inserted.second) {
            sd_backend_htp_async_cache_slot_free(slot);
            LOG_ERROR("%s residual cache key insertion failed: %s",
                      get_desc().c_str(), key.c_str());
            return nullptr;
        }
        return &inserted.first->second;
    }

    bool replace_async_residual_cache_slots(
            const std::vector<std::pair<std::string, ggml_tensor*>>& outputs,
            uint64_t preserve_generation = 0) {
        std::vector<std::pair<std::string, uint64_t>> old_slots;
        old_slots.reserve(outputs.size());
        for (const auto& output : outputs) {
            auto found = async_residual_cache_slots_.find(output.first);
            if (found == async_residual_cache_slots_.end() ||
                found->second.slot == nullptr) {
                continue;
            }
            if (preserve_generation != 0 &&
                found->second.generation == preserve_generation) {
                continue;
            }
            old_slots.emplace_back(output.first, found->second.generation);
        }
        if (old_slots.empty()) {
            return true;
        }
        if (!active_async_cache_consumers_.empty()) {
            LOG_ERROR("%s residual cache replacement has active consumers",
                      get_desc().c_str());
            return false;
        }
        const bool trace = sd_env_flag_enabled("SD_HTP_SPARSE_OPENCL_TRACE");
        if (trace) {
            std::fprintf(stderr,
                         "htp_opencl_event name=gpu_residual_cache_replace_wait_begin "
                         "ts_us=%lld key_count=%zu\n",
                         (long long) ggml_time_us(), old_slots.size());
        }
        if (!reap_retired_compute_arenas(true)) {
            LOG_ERROR("%s residual cache replacement failed waiting deferred arenas",
                      get_desc().c_str());
            return false;
        }
        for (const auto& old : old_slots) {
            AsyncResidualCacheGeneration* old_generation =
                find_async_residual_cache_slot(old.first, old.second);
            const bool has_opencl_import = old_generation != nullptr &&
                sd_backend_htp_async_cache_slot_opencl_import_identity(
                    old_generation->slot) != 0;
            bool waited = false;
            char reason[512] = {};
            if (has_opencl_import && !sd_backend_htp_opencl_cache_wait(
                    htp_backend, old.first.c_str(), old.second,
                    &waited, reason, sizeof(reason))) {
                LOG_ERROR("%s residual cache wait failed: key=%s generation=%" PRIu64 " reason=%s",
                          get_desc().c_str(), old.first.c_str(), old.second,
                          reason[0] != '\0' ? reason : "unknown error");
                return false;
            }
            auto found = async_residual_cache_slots_.find(old.first);
            if (found != async_residual_cache_slots_.end() &&
                found->second.generation == old.second) {
                found->second.producer_pending = false;
            }
        }
        if (trace) {
            std::fprintf(stderr,
                         "htp_opencl_event name=gpu_residual_cache_replace_wait_end "
                         "ts_us=%lld key_count=%zu\n",
                         (long long) ggml_time_us(), old_slots.size());
        }
        for (const auto& old : old_slots) {
            auto found = async_residual_cache_slots_.find(old.first);
            if (found == async_residual_cache_slots_.end() ||
                found->second.generation != old.second) {
                LOG_ERROR("%s residual cache generation changed during replacement: key=%s",
                          get_desc().c_str(), old.first.c_str());
                return false;
            }
            if (found->second.consumer_count != 0 || found->second.producer_pending) {
                LOG_ERROR("%s residual cache generation is still busy: key=%s generation=%" PRIu64,
                          get_desc().c_str(), old.first.c_str(), old.second);
                return false;
            }
            sd_backend_htp_async_cache_slot_free(found->second.slot);
            async_residual_cache_slots_.erase(found);
        }
        return true;
    }

    void alloc_compute_ctx() {
        ggml_init_params params;
        params.mem_size   = static_cast<size_t>(ggml_tensor_overhead() * MAX_GRAPH_SIZE + ggml_graph_overhead());
        params.mem_buffer = nullptr;
        params.no_alloc   = true;

        compute_ctx = ggml_init(params);
        GGML_ASSERT(compute_ctx != nullptr);
    }

    bool reap_retired_compute_arenas(bool wait_all) {
        if (htp_backend == nullptr) {
            return retired_compute_arenas_.empty();
        }
        for (auto it = retired_compute_arenas_.begin();
             it != retired_compute_arenas_.end();) {
            if (!it->qkv_released) {
                bool qkv_released = false;
                char qkv_reason[512] = {};
                if (!ggml_backend_htp_opencl_release_qkv_generation(
                        htp_backend, it->generation, wait_all, &qkv_released,
                        qkv_reason, sizeof(qkv_reason))) {
                    LOG_ERROR(
                        "%s release deferred QKV generation=%" PRIu64 " failed: %s",
                        get_desc().c_str(), it->generation,
                        qkv_reason[0] != '\0' ? qkv_reason : "unknown error");
                    return false;
                }
                if (sd_env_flag_enabled("SD_HTP_QKV_RELEASE_TRACE")) {
                    LOG_INFO(
                        "qkv_release_poll generation=%" PRIu64 " wait=%d released=%d reason=%s",
                        it->generation, wait_all ? 1 : 0,
                        qkv_released ? 1 : 0,
                        qkv_reason[0] != '\0' ? qkv_reason : "ok");
                }
                if (qkv_released) {
                    const size_t qkv_bytes =
                        ggml_gallocr_get_buffer_size(it->allocator, 2);
                    if (!ggml_gallocr_free_buffer(it->allocator, 2, nullptr)) {
                        LOG_ERROR("%s free deferred QKV allocator buffer failed",
                                  get_desc().c_str());
                        return false;
                    }
                    it->qkv_released = true;
                    if (sd_env_flag_enabled("SD_HTP_SPARSE_OPENCL_TRACE")) {
                        std::fprintf(
                            stderr,
                            "htp_opencl_event name=gpu_qkv_release ts_us=%lld "
                            "generation=%" PRIu64 " bytes=%zu\n",
                            (long long) ggml_time_us(), it->generation, qkv_bytes);
                    }
                }
            }
            bool released = false;
            char reason[512] = {};
            if (!sd_backend_htp_opencl_release_arena_generation(
                    htp_backend, it->generation, wait_all, &released,
                    reason, sizeof(reason))) {
                LOG_ERROR(
                    "%s release deferred compute arena generation=%" PRIu64 " failed: %s",
                    get_desc().c_str(), it->generation,
                    reason[0] != '\0' ? reason : "unknown error");
                return false;
            }
            if (!released) {
                ++it;
                continue;
            }
            ggml_gallocr_free(it->allocator);
            it = retired_compute_arenas_.erase(it);
        }
        return true;
    }

    bool retire_current_compute_arena(uint64_t generation) {
        if (compute_allocr == nullptr || generation == 0) {
            return true;
        }
        if (!reap_retired_compute_arenas(false)) {
            return false;
        }
        while (retired_compute_arenas_.size() >= 2) {
            RetiredComputeArenaGeneration& oldest = retired_compute_arenas_.front();
            const int64_t wait_begin = ggml_time_us();
            const bool trace = sd_env_flag_enabled("SD_HTP_SPARSE_OPENCL_TRACE");
            if (trace) {
                std::fprintf(
                    stderr,
                    "htp_opencl_event name=gpu_residual_arena_pressure_wait_begin "
                    "ts_us=%lld generation=%" PRIu64 " bytes=%zu\n",
                    (long long) wait_begin, oldest.generation, oldest.bytes);
            }
            bool released = false;
            char reason[512] = {};
            if (!sd_backend_htp_opencl_release_arena_generation(
                    htp_backend, oldest.generation, true, &released,
                    reason, sizeof(reason)) || !released) {
                LOG_ERROR(
                    "%s pressure wait for compute arena generation=%" PRIu64
                    " failed: %s",
                    get_desc().c_str(), oldest.generation,
                    reason[0] != '\0' ? reason : "unknown error");
                return false;
            }
            if (trace) {
                std::fprintf(
                    stderr,
                    "htp_opencl_event name=gpu_residual_arena_pressure_wait_end "
                    "ts_us=%lld generation=%" PRIu64 "\n",
                    (long long) ggml_time_us(), oldest.generation);
            }
            ggml_gallocr_free(oldest.allocator);
            retired_compute_arenas_.pop_front();
        }

        size_t bytes = ggml_gallocr_get_buffer_size(compute_allocr, 0);
        if (!compute_node_buffer_ids_.empty()) {
            bytes += ggml_gallocr_get_buffer_size(compute_allocr, 1);
        }
        if (!compute_node_buffer_ids_.empty()) {
            bytes += ggml_gallocr_get_buffer_size(compute_allocr, 2);
        }
        retired_compute_arenas_.push_back({ generation, compute_allocr, bytes, false });
        if (sd_env_flag_enabled("SD_HTP_QKV_RELEASE_TRACE")) {
            LOG_INFO(
                "qkv_retire generation=%" PRIu64 " bytes=%zu",
                generation, bytes);
        }
        compute_allocr = nullptr;
        compute_node_buffer_ids_.clear();
        compute_leaf_buffer_ids_.clear();
        compute_no_alloc_tensors_.clear();
        // Do not probe the just-computed generation here. HTP dense has
        // returned, but its deferred OpenCL sparse work may still own QKV.
        // The next arena poll releases QKV after sparse completion.
        return true;
    }

    void free_compute_ctx() {
        debug_tensors.clear();
        if (compute_ctx != nullptr) {
            ggml_free(compute_ctx);
            compute_ctx = nullptr;
        }
        backend_tensor_data_map.clear();
        cache_backed_tensor_name_map.clear();
        cache_backed_tensor_generation_map_.clear();
    }

    void rebuild_params_tensor_set() {
        params_tensor_set_.clear();
        if (params_ctx == nullptr) {
            return;
        }
        for (ggml_tensor* t = ggml_get_first_tensor(params_ctx); t != nullptr; t = ggml_get_next_tensor(params_ctx, t)) {
            params_tensor_set_.insert(t);
        }
    }

    void prepare_build_in_tensor_before() {
        one_tensor = ggml_new_tensor_1d(compute_ctx, GGML_TYPE_F32, 1);
        ggml_set_name(one_tensor, "ggml_runner_build_in_tensor:one");
        set_backend_tensor_data(one_tensor, one_vec.data());

        zero_int_tensor = ggml_new_tensor_1d(compute_ctx, GGML_TYPE_I32, 1);
        ggml_set_name(zero_int_tensor, "ggml_runner_build_in_tensor:zero_int");
        set_backend_tensor_data(zero_int_tensor, zero_int_vec.data());
    }

    void prepare_build_in_tensor_after(ggml_cgraph* gf) {
        ggml_build_forward_expand(gf, one_tensor);
        ggml_build_forward_expand(gf, zero_int_tensor);
    }

    ggml_cgraph* new_graph_custom(size_t graph_size) {
        if (weight_adapter) {
            graph_size += weight_adapter->get_extra_graph_size();
        }
        return ggml_new_graph_custom(compute_ctx, graph_size, false);
    }

    ggml_cgraph* get_compute_graph(get_graph_cb_t get_graph) {
        prepare_build_in_tensor_before();
        ggml_cgraph* gf = get_graph();
        if (ggml_graph_n_nodes(gf) > 0) {
            auto result = ggml_graph_node(gf, -1);
            ggml_set_name(result, final_result_name.c_str());
        }
        for (const auto& entry : debug_tensors) {
            if (entry.first != nullptr) {
                ggml_build_forward_expand(gf, entry.first);
            }
        }
        const bool local_opencl_lifetime =
            sd_htp_opencl_local_lifetime_prerequisites();
        for (const auto& entry : cache_tensor_map) {
            if (entry.second != nullptr) {
                const bool local_residual_cache = local_opencl_lifetime &&
                    entry.first.rfind("htp_block_sparse_residual_", 0) == 0;
                if (!local_residual_cache) {
                    ggml_set_output(entry.second);
                }
                ggml_build_forward_expand(gf, entry.second);
            }
        }
        if (htp_backend != nullptr &&
            sd_htp_opencl_whole_arena_deferred_ownership()) {
            sd_backend_htp_opencl_retain_deferred_residual_inputs(gf);
        }
        prepare_build_in_tensor_after(gf);
        return gf;
    }

    bool prepare_compute_graph(get_graph_cb_t get_graph,
                               ggml_cgraph** gf_out) {
        GGML_ASSERT(gf_out != nullptr);

        reset_compute_ctx();
        ggml_cgraph* gf = get_compute_graph(get_graph);
        if (gf == nullptr) {
            free_compute_ctx();
            return false;
        }

        *gf_out = gf;
        return true;
    }

    void trace_residual_pair_allocation(
            const char* checkpoint,
            ggml_cgraph* gf) const {
        if (!sd_env_flag_enabled("SD_HTP_SPARSE_OPENCL_TRACE") ||
            gf == nullptr) {
            return;
        }
        std::unordered_set<ggml_backend_buffer_t> persistent_backings;
        for (const CacheStorageSegment& segment : cache_storage_segments_) {
            if (segment.buffer != nullptr) {
                persistent_backings.insert(segment.buffer);
            }
        }
        for (const auto& entry : async_residual_cache_slots_) {
            ggml_tensor* tensor = entry.second.slot != nullptr
                ? sd_backend_htp_async_cache_slot_tensor(entry.second.slot)
                : nullptr;
            if (tensor != nullptr && tensor->buffer != nullptr) {
                persistent_backings.insert(tensor->buffer);
            }
        }

        const ggml_backend_buffer_type_t runtime_buft =
            ggml_backend_get_default_buffer_type(runtime_backend);
        const ggml_backend_buffer_type_t htp_buft = htp_backend != nullptr
            ? ggml_backend_get_default_buffer_type(htp_backend) : nullptr;
        const ggml_backend_buffer_type_t qkv_buft = htp_backend != nullptr
            ? ggml_backend_htp_qkv_buffer_type() : nullptr;
        const auto records =
            sd_collect_htp_residual_pair_allocation_records(
                gf, compute_no_alloc_tensors_, compute_node_buffer_ids_,
                compute_leaf_buffer_ids_, runtime_buft, htp_buft, qkv_buft,
                persistent_backings);
        if (records.empty()) {
            return;
        }
        for (const auto& record : records) {
            char line[1024] = {};
            if (sd_format_htp_residual_pair_allocation_record(
                    line, sizeof(line), checkpoint, record) > 0) {
                std::fprintf(stderr, "%s\n", line);
            }
        }
        char summary_line[512] = {};
        if (sd_format_htp_residual_pair_allocation_summary(
                summary_line, sizeof(summary_line), checkpoint,
                sd_summarize_htp_residual_pair_allocation_records(records)) > 0) {
            std::fprintf(stderr, "%s\n", summary_line);
        }
    }

    bool alloc_compute_buffer(ggml_cgraph* gf) {
        if (!reap_retired_compute_arenas(false)) {
            return false;
        }
        ggml_backend_buffer_type_t runtime_buft = ggml_backend_get_default_buffer_type(runtime_backend);
        ggml_backend_buffer_type_t htp_buft     = htp_backend != nullptr ? ggml_backend_get_default_buffer_type(htp_backend) : nullptr;
        compute_node_buffer_ids_.clear();
        compute_leaf_buffer_ids_.clear();
        compute_no_alloc_tensors_.clear();

        ggml_backend_buffer_type_t qkv_buft = htp_backend != nullptr ? ggml_backend_htp_qkv_buffer_type() : nullptr;
        if (htp_op_routing_enabled() && htp_buft != nullptr) {
            // CPU、普通 HTP 以及可提前回收的 sparse QKV 使用三个独立 buffer。
            if (compute_allocr == nullptr) {
                ggml_backend_buffer_type_t bufts[] = {runtime_buft, htp_buft, qkv_buft};
                compute_allocr                    = ggml_gallocr_new_n(bufts, 3);
            }
            plan_compute_buffer_ids(gf, &compute_node_buffer_ids_, &compute_leaf_buffer_ids_);
        } else {
            if (compute_allocr == nullptr) {
                compute_allocr = ggml_gallocr_new(runtime_buft);
            }
        }

        compute_no_alloc_tensors_ =
            sd_htp_opencl_sparse_local_lifetime_prerequisites()
                ? sd_collect_local_residual_owned_sparse_producers(gf)
                : std::vector<const ggml_tensor*>{};
        if (!ggml_gallocr_set_no_alloc_tensors(
                compute_allocr,
                compute_no_alloc_tensors_.empty()
                    ? nullptr : compute_no_alloc_tensors_.data(),
                compute_no_alloc_tensors_.size())) {
            LOG_ERROR("%s: failed to configure metadata-only graph tensors\n",
                      get_desc().c_str());
            return false;
        }

        bool reserve_ok = false;
        if (!compute_node_buffer_ids_.empty() || !compute_leaf_buffer_ids_.empty()) {
            reserve_ok = ggml_gallocr_reserve_n(compute_allocr,
                                                gf,
                                                compute_node_buffer_ids_.data(),
                                                compute_leaf_buffer_ids_.data());
        } else {
            reserve_ok = ggml_gallocr_reserve(compute_allocr, gf);
        }

        if (!reserve_ok) {
            // failed to allocate the compute buffer
            LOG_ERROR("%s: failed to allocate the compute buffer\n", get_desc().c_str());
            free_compute_buffer();
            return false;
        }

        // compute the required memory
        size_t compute_buffer_size = ggml_gallocr_get_buffer_size(compute_allocr, 0);
        LOG_DEBUG("%s compute buffer size: %.2f MB(%s)",
                  get_desc().c_str(),
                  compute_buffer_size / 1024.0 / 1024.0,
                  ggml_backend_is_cpu(runtime_backend) ? "RAM" : "VRAM");
        if (compute_node_buffer_ids_.size() > 0) {
            const size_t rpcmem_buffer_size = ggml_gallocr_get_buffer_size(compute_allocr, 1);
            LOG_DEBUG("%s htp rpcmem buffer size: %.2f MB",
                      get_desc().c_str(),
                      rpcmem_buffer_size / 1024.0 / 1024.0);
        }
        if (compute_node_buffer_ids_.size() > 0 && qkv_buft != nullptr) {
            const size_t qkv_buffer_size = ggml_gallocr_get_buffer_size(compute_allocr, 2);
            LOG_DEBUG("%s sparse QKV buffer size: %.2f MB",
                      get_desc().c_str(), qkv_buffer_size / 1024.0 / 1024.0);
        }
        if (sd_htp_opencl_local_lifetime_prerequisites()) {
            const size_t runtime_buffer_bytes =
                ggml_gallocr_get_buffer_size(compute_allocr, 0);
            const size_t htp_buffer_bytes = compute_node_buffer_ids_.empty()
                ? 0 : ggml_gallocr_get_buffer_size(compute_allocr, 1);
            const size_t qkv_buffer_bytes =
                compute_node_buffer_ids_.empty() || qkv_buft == nullptr
                    ? 0 : ggml_gallocr_get_buffer_size(compute_allocr, 2);
            size_t host_cache_bytes = 0;
            for (const CacheStorageSegment& segment : cache_storage_segments_) {
                if (segment.buffer != nullptr) {
                    host_cache_bytes +=
                        ggml_backend_buffer_get_size(segment.buffer);
                }
            }
            size_t cache_output_count = 0;
            for (const auto& entry : cache_tensor_map) {
                if (entry.first.rfind("htp_block_sparse_residual_", 0) == 0) {
                    ++cache_output_count;
                }
            }

            ggml_htp_memory_ledger_snapshot ledger = {};
            ledger.requested_route = GGML_HTP_ROUTE_INVALID;
            ledger.preflight_result = -1;
            ledger.runtime_buffer_bytes = runtime_buffer_bytes;
            ledger.htp_buffer_bytes = htp_buffer_bytes;
            ledger.qkv_buffer_bytes = qkv_buffer_bytes;
            ledger.host_cache_bytes = host_cache_bytes;
            ledger.retired_arenas = retired_compute_arenas_.size();
            for (const auto& entry : async_residual_cache_slots_) {
                if (ledger.cache_slot_count >=
                    GGML_HTP_MEMORY_LEDGER_MAX_CACHE_SLOTS) {
                    break;
                }
                ggml_htp_cache_slot_snapshot slot = {};
                if (sd_backend_htp_async_cache_slot_snapshot(
                        entry.second.slot, &slot)) {
                    ledger.cache_slots[ledger.cache_slot_count++] = slot;
                }
            }
            ggml_backend_htp_memory_ledger_recalculate(&ledger);

            const sd_htp_opencl_memory_plan memory_plan =
                sd_make_htp_opencl_memory_plan(
                    sd_htp_opencl_lifetime_mode(),
                    runtime_buffer_bytes,
                    htp_buffer_bytes,
                    qkv_buffer_bytes,
                    host_cache_bytes,
                    ledger.rpcmem_cache_logical_bytes,
                    ledger.rpcmem_cache_allocation_bytes,
                    compute_no_alloc_tensors_.size(),
                    compute_no_alloc_tensors_.size(),
                    cache_output_count,
                    sd_htp_opencl_bind_cache_outputs_before_reserve(),
                    0);
            char summary[512] = {};
            if (sd_env_flag_enabled("SD_HTP_SPARSE_OPENCL_TRACE") &&
                sd_format_htp_opencl_memory_plan(
                    summary, sizeof(summary), memory_plan) > 0) {
                std::fprintf(stderr, "%s\n", summary);
            }
            if (htp_backend != nullptr &&
                !sd_backend_htp_set_graph_memory_ledger(
                    htp_backend, &ledger) &&
                ggml_backend_htp_memory_ledger_trace_enabled()) {
                std::fprintf(stderr,
                             "htp_memory_ledger status=error reason=set_failed\n");
            }
        }
        return true;
    }

    // 当前只允许 CPU runtime 走 HTP hybrid，因为未支持算子需要直接回退到 CPU。
    bool htp_op_routing_enabled() const {
        return htp_backend != nullptr && ggml_backend_is_cpu(runtime_backend);
    }

    // 统一收敛 diffusion 里 HTP F16 matmul 的基础 type/shape 判断；weight layout 资格由离线 packed flag 单独决定。
    bool is_htp_supported_f16_mul_mat(ggml_tensor* node) const {
        if (!htp_op_routing_enabled() || node == nullptr || node->op != GGML_OP_MUL_MAT) {
            return false;
        }
        if (!ggml_backend_supports_op(htp_backend, node)) {
            return false;
        }

        ggml_tensor* weight     = node->src[0];
        ggml_tensor* activation = node->src[1];
        if (weight == nullptr || activation == nullptr) {
            return false;
        }

        // 当前 HTP permuted F16 kernel 只按二维 [m,k] activation 和 [m,n] dst 连续布局处理，带 batch 维或 view-stride 的 MUL_MAT 先留在 CPU。
        const bool row_major_2d_layout_ok = ggml_nrows(node) == node->ne[1] &&
                                            ggml_nrows(activation) == activation->ne[1];

        return node->type == GGML_TYPE_F32 &&
               weight->type == GGML_TYPE_F16 &&
               activation->type == GGML_TYPE_F32 &&
               row_major_2d_layout_ok &&
               ggml_is_contiguous(weight) &&
               ggml_is_contiguous(activation) &&
               ggml_is_contiguous(node);
    }

    bool htp_packed_f16_weight_is_prepared(ggml_tensor* tensor) const {
        return sd_backend_htp_get_packed_f16_weight(tensor) != nullptr;
    }

    bool is_htp_supported_flash_attn(ggml_tensor* node) const {
        if (!htp_op_routing_enabled() || !htp_attention_enabled || node == nullptr || node->op != GGML_OP_FLASH_ATTN_EXT) {
            return false;
        }
        ggml_tensor* q    = node->src[0];
        ggml_tensor* k    = node->src[1];
        ggml_tensor* v    = node->src[2];
        ggml_tensor* mask = node->src[3];
        return q != nullptr &&
               k != nullptr &&
               v != nullptr &&
               mask != nullptr &&
               (mask->type == GGML_TYPE_F16 || mask->type == GGML_TYPE_I32) &&
               ggml_backend_supports_op(htp_backend, node);
    }

    bool is_htp_supported_block_sparse_score(ggml_tensor* node) const {
        if (!htp_op_routing_enabled() || !htp_attention_enabled || node == nullptr || node->op != GGML_OP_CUSTOM) {
            return false;
        }
        return ggml_backend_supports_op(htp_backend, node);
    }

    bool is_htp_supported_packed_qkv(ggml_tensor* node) const {
        if (!htp_op_routing_enabled() || !htp_attention_enabled || node == nullptr ||
            node->op != GGML_OP_CUSTOM ||
            !htp_packed_qkv_node_name_is_supported(node->name)) {
            return false;
        }
        return ggml_backend_supports_op(htp_backend, node);
    }

    // 当前阶段把 diffusion 里的 F16 matmul、score、以及 dense/block-sparse flash attention 纳入 HTP 自动路由。
    bool is_htp_target_op(ggml_tensor* node) const {
        if (!htp_op_routing_enabled() || node == nullptr) {
            return false;
        }
        if (is_htp_supported_packed_qkv(node)) {
            return true;
        }
        if (is_htp_supported_block_sparse_score(node)) {
            return true;
        }
        if (is_htp_supported_flash_attn(node)) {
            return true;
        }
        ggml_tensor* weight = node->src[0];
        // F16 permuted matmul 只能在 weight 已经是离线 packed layout 且完成注册时进入 HTP；否则必须走 CPU fallback。
        return is_htp_supported_f16_mul_mat(node) && htp_packed_f16_weight_is_prepared(weight);
    }

    bool htp_param_is_placed(ggml_tensor* tensor) const {
        ggml_tensor* root = get_htp_tensor_backing_root(tensor);
        if (root == nullptr || root->buffer == nullptr) {
            return false;
        }
        // view 权重本身不持有 buffer，但只要 backing root 已经按 llama.cpp-npu 形式直接放进 RPCMEM，就可以用 view offset 分发给 HTP。
        return (htp_placed_param_tensors_.find(root) != htp_placed_param_tensors_.end() ||
                std::find(htp_packed_staging_tensors_.begin(), htp_packed_staging_tensors_.end(), root) != htp_packed_staging_tensors_.end()) &&
               ggml_backend_buft_is_rpcmem(ggml_backend_buffer_get_type(root->buffer));
    }

    static bool is_htp_packed_qkv_param_name(const char* name) {
        if (name == nullptr || name[0] == '\0') {
            return false;
        }
        return std::strstr(name, ".attn.qkv.bias") != nullptr ||
               std::strstr(name, ".attn.ln_q.weight") != nullptr ||
               std::strstr(name, ".attn.ln_k.weight") != nullptr ||
               std::strstr(name, ".attn2.qkv.bias") != nullptr ||
               std::strstr(name, ".attn2.ln_q.weight") != nullptr ||
               std::strstr(name, ".attn2.ln_k.weight") != nullptr ||
               std::strstr(name, ".attention.q_norm.weight") != nullptr ||
               std::strstr(name, ".attention.k_norm.weight") != nullptr ||
               std::strstr(name, ".query_norm.weight") != nullptr ||
               std::strstr(name, ".key_norm.weight") != nullptr;
    }

    bool should_place_param_in_htp_rpcmem(ggml_tensor* tensor) const {
        if (!htp_op_routing_enabled() || !sd_env_flag_enabled("SD_HTP_COMPUTE") || htp_backend == nullptr || tensor == nullptr) {
            return false;
        }
        if ((tensor->flags & SD_GGML_TENSOR_FLAG_HTP_PACKED_QKV_PARAM) != 0 ||
            is_htp_packed_qkv_param_name(tensor->name)) {
            return tensor->view_src == nullptr && tensor->type == GGML_TYPE_F32 && ggml_is_contiguous(tensor);
        }
        // packed GGUF 里这类权重已经是 HTP kernel 需要的 W16A32 布局，不能留在普通 CPU buffer；
        // 直接放进 RPCMEM 后，HTP matmul 可以把原 tensor 本体当作 packed weight 使用。
        return tensor->view_src == nullptr &&
               (tensor->flags & SD_GGML_TENSOR_FLAG_HTP_PACKED_W16A32) != 0 &&
               tensor->type == GGML_TYPE_F16 &&
               ggml_n_dims(tensor) == 2 &&
               ggml_is_contiguous(tensor);
    }

    bool should_prepare_htp_packed_f16_weight(ggml_tensor* tensor) const {
        if (!htp_op_routing_enabled() || !sd_env_flag_enabled("SD_HTP_COMPUTE") || tensor == nullptr) {
            return false;
        }
        ggml_tensor* root = get_htp_tensor_backing_root(tensor);
        // 只接受离线 packed 的直接参数。普通 row-major F16 权重不再在推理前在线重排。
        return root != nullptr &&
               root == tensor &&
               (root->flags & SD_GGML_TENSOR_FLAG_HTP_MUL_MAT_WEIGHT) != 0 &&
               (root->flags & SD_GGML_TENSOR_FLAG_HTP_PACKED_W16A32) != 0 &&
               tensor->view_src == nullptr &&
               tensor->type == GGML_TYPE_F16 &&
               ggml_n_dims(tensor) == 2 &&
               ggml_is_contiguous(tensor) &&
               tensor->ne[0] % 32 == 0 &&
               tensor->ne[1] % 32 == 0;
    }

    std::vector<ggml_tensor*> collect_htp_placeable_params() const {
        std::vector<ggml_tensor*> tensors;
        if (params_ctx == nullptr) {
            return tensors;
        }
        for (ggml_tensor* tensor = ggml_get_first_tensor(params_ctx); tensor != nullptr; tensor = ggml_get_next_tensor(params_ctx, tensor)) {
            if (should_place_param_in_htp_rpcmem(tensor)) {
                tensors.push_back(tensor);
            }
        }
        return tensors;
    }

    std::vector<ggml_tensor*> collect_htp_packable_f16_weights() const {
        std::vector<ggml_tensor*> tensors;
        if (params_ctx == nullptr) {
            return tensors;
        }
        for (ggml_tensor* tensor = ggml_get_first_tensor(params_ctx); tensor != nullptr; tensor = ggml_get_next_tensor(params_ctx, tensor)) {
            if (should_prepare_htp_packed_f16_weight(tensor)) {
                tensors.push_back(tensor);
            }
        }
        return tensors;
    }

    std::vector<ggml_tensor*> collect_htp_packable_f16_weights(ggml_cgraph* gf) const {
        if (gf == nullptr) {
            return collect_htp_packable_f16_weights();
        }

        // 按当前图收集会被 HTP matmul 使用的离线 packed 权重；普通 row-major 权重即使形状满足，
        // 也不会在这里被在线重排或注册。
        std::vector<ggml_tensor*> tensors;
        std::unordered_set<ggml_tensor*> seen;
        for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
            ggml_tensor* node = ggml_graph_node(gf, i);
            if (is_htp_supported_f16_mul_mat(node)) {
                ggml_tensor* root = get_htp_tensor_backing_root(node->src[0]);
                if (should_prepare_htp_packed_f16_weight(root) && seen.insert(root).second) {
                    tensors.push_back(root);
                }
                continue;
            }
            if (!is_htp_supported_packed_qkv(node)) {
                continue;
            }
            for (int src_index : htp_packed_qkv_weight_src_indices(node->name)) {
                ggml_tensor* root = get_htp_tensor_backing_root(node->src[src_index]);
                if (should_prepare_htp_packed_f16_weight(root) && seen.insert(root).second) {
                    tensors.push_back(root);
                }
            }
            for (int src_index : htp_packed_qkv_staging_src_indices(node->name)) {
                ggml_tensor* root = get_htp_tensor_backing_root(node->src[src_index]);
                if (should_place_param_in_htp_rpcmem(root) && seen.insert(root).second) {
                    tensors.push_back(root);
                }
            }
        }
        return tensors;
    }

    uint64_t hash_tensor_bytes_for_debug(ggml_tensor* tensor) const {
        GGML_ASSERT(tensor != nullptr);
        static constexpr uint64_t FNV1A64_OFFSET = UINT64_C(14695981039346656037);
        uint64_t hash = FNV1A64_OFFSET;
        const size_t nbytes = ggml_nbytes(tensor);
        std::vector<uint8_t> buffer(std::min<size_t>(nbytes, 1024 * 1024));
        for (size_t offset = 0; offset < nbytes;) {
            const size_t chunk = std::min(buffer.size(), nbytes - offset);
            ggml_backend_tensor_get(tensor, buffer.data(), offset, chunk);
            hash = sd_fnv1a64_update(hash, buffer.data(), chunk);
            offset += chunk;
        }
        return hash;
    }

    void log_graph_cut_cache_hash(const char* phase,
                                  const std::string& name,
                                  ggml_tensor* tensor,
                                  uint64_t hash) const {
        if (!sd_env_flag_enabled("SD_HTP_GRAPH_CUT_CACHE_HASH") ||
            tensor == nullptr) {
            return;
        }
        std::fprintf(stderr,
                     "htp_graph_cut_cache_hash phase=%s name=%s bytes=%zu hash=%016" PRIx64 "\n",
                     phase != nullptr ? phase : "unknown",
                     name.c_str(), ggml_nbytes(tensor), hash);
    }

    void log_htp_packed_f16_weight_hashes_for_debug(const std::vector<ggml_tensor*>& tensors) {
        if (!sd_env_flag_enabled("SD_HTP_DEBUG_PACKED_HASH")) {
            return;
        }

        const char* filter = sd_env_value("SD_HTP_DEBUG_PACKED_FILTER");
        const char* limit_value = sd_env_value("SD_HTP_DEBUG_PACKED_LOG_LIMIT");
        int log_limit = limit_value != nullptr ? std::atoi(limit_value) : 8;
        if (log_limit < 0) {
            log_limit = 0;
        }

        static constexpr uint64_t FNV1A64_OFFSET = UINT64_C(14695981039346656037);
        uint64_t aggregate = FNV1A64_OFFSET;
        size_t hashed_count = 0;
        size_t hashed_bytes = 0;
        int logged = 0;

        for (ggml_tensor* original : tensors) {
            if (original == nullptr) {
                continue;
            }
            const char* name = original->name[0] != '\0' ? original->name : "<unnamed>";
            if (filter != nullptr && std::strstr(name, filter) == nullptr) {
                continue;
            }

            ggml_tensor* packed = sd_backend_htp_get_packed_f16_weight(original);
            if (packed == nullptr) {
                LOG_WARN("%s HTP packed hash debug skipped unregistered weight '%s'",
                         get_desc().c_str(),
                         name);
                continue;
            }

            const uint64_t tensor_hash = hash_tensor_bytes_for_debug(packed);
            const size_t nbytes = ggml_nbytes(packed);
            aggregate = sd_fnv1a64_update(aggregate, &tensor_hash, sizeof(tensor_hash));
            aggregate = sd_fnv1a64_update(aggregate, &nbytes, sizeof(nbytes));
            aggregate = sd_fnv1a64_update(aggregate, name, std::strlen(name));
            hashed_count += 1;
            hashed_bytes += nbytes;

            if (logged < log_limit) {
                LOG_INFO("%s HTP packed hash[%zu] name='%s' packed='%s' ne=[%lld,%lld,%lld,%lld] bytes=%zu hash=%016" PRIx64,
                         get_desc().c_str(),
                         hashed_count - 1,
                         name,
                         packed->name[0] != '\0' ? packed->name : "<unnamed>",
                         (long long)packed->ne[0],
                         (long long)packed->ne[1],
                         (long long)packed->ne[2],
                         (long long)packed->ne[3],
                         nbytes,
                         tensor_hash);
                logged += 1;
            }
        }

        LOG_INFO("%s HTP packed hash aggregate tensors=%zu bytes=%zu hash=%016" PRIx64 "%s%s",
                 get_desc().c_str(),
                 hashed_count,
                 hashed_bytes,
                 aggregate,
                 filter != nullptr ? " filter=" : "",
                 filter != nullptr ? filter : "");
    }

    void bind_tensor_allocation(ggml_tensor* dst, ggml_tensor* src) {
        GGML_ASSERT(dst != nullptr);
        GGML_ASSERT(src != nullptr);
        dst->buffer = src->buffer;
        dst->data   = src->data;
        dst->extra  = src->extra;
    }

    bool allocate_htp_placed_params() {
        if (!htp_op_routing_enabled() || htp_param_placement_buffer_ != nullptr ||
            sd_env_flag_enabled("SD_HTP_PACKED_STAGING")) {
            return true;
        }

        // 参数分配阶段只搬运离线 packed 的权重。这里不创建第二份 packed shadow，
        // 而是把原 tensor 的 buffer/data 绑定到 RPCMEM 分配出的镜像 tensor 上。
        std::vector<ggml_tensor*> tensors = collect_htp_placeable_params();
        if (tensors.empty()) {
            return true;
        }

        ggml_init_params params;
        params.mem_size   = std::max<size_t>(1, tensors.size()) * ggml_tensor_overhead();
        params.mem_buffer = nullptr;
        params.no_alloc   = true;

        htp_param_placement_ctx_ = ggml_init(params);
        GGML_ASSERT(htp_param_placement_ctx_ != nullptr);

        htp_param_placement_pairs_.clear();
        htp_param_placement_pairs_.reserve(tensors.size());
        htp_placed_param_tensors_.clear();
        htp_placed_param_tensors_.reserve(tensors.size());

        ggml_backend_buffer_type_t htp_buft = ggml_backend_get_default_buffer_type(htp_backend);
        const int session_count = ggml_backend_htp_session_count(htp_backend);
        const bool owner_aware = sd_env_flag_enabled("SD_HTP_MULTI_SESSION") &&
            session_count > 1 && !sd_env_flag_enabled("SD_HTP_PACKED_STAGING");

        // Duplicate metadata once, then bind each duplicate into an owner-
        // specific RPCMEM chunk.  The ordinary path below remains unchanged
        // for single-session and staging execution.
        std::vector<ggml_tensor*> htp_tensors;
        htp_tensors.reserve(tensors.size());
        for (ggml_tensor* tensor : tensors) {
            ggml_tensor* htp_tensor = ggml_dup_tensor(htp_param_placement_ctx_, tensor);
            ggml_set_name(htp_tensor, tensor->name);
            htp_tensors.push_back(htp_tensor);
            htp_param_placement_pairs_.push_back({tensor, htp_tensor});
        }

        if (owner_aware) {
            std::vector<size_t> tensor_sizes;
            tensor_sizes.reserve(tensors.size());
            const size_t alignment = ggml_backend_buft_get_alignment(htp_buft);
            for (ggml_tensor * tensor : tensors) {
                tensor_sizes.push_back(GGML_PAD(
                    ggml_backend_buft_get_alloc_size(htp_buft, tensor), alignment));
            }
            const std::vector<size_t> chunk_sizes = htp_per_tensor_chunk_sizes(
                tensor_sizes, ggml_backend_buft_get_max_size(htp_buft));
            if (chunk_sizes.size() != tensors.size()) {
                LOG_ERROR("%s invalid HTP per-tensor parameter allocation plan: tensors=%zu max=%zu",
                          get_desc().c_str(), tensors.size(),
                          ggml_backend_buft_get_max_size(htp_buft));
                free_htp_param_placement();
                return false;
            }

            for (size_t index = 0; index < tensors.size(); ++index) {
                // The graph is not available during model loading. Allocate
                // through session 0 as the canonical creation route only;
                // the tensor remains eligible for mapping in any consuming
                // session and is never assigned by parameter ordinal.
                ggml_backend_buffer_t tensor_buffer =
                    ggml_backend_htp_alloc_buffer_for_session(htp_backend, 0, chunk_sizes[index]);
                if (tensor_buffer == nullptr) {
                    LOG_ERROR("%s alloc HTP per-tensor parameter buffer failed: tensor=%s bytes=%zu",
                              get_desc().c_str(), tensors[index]->name, chunk_sizes[index]);
                    free_htp_param_placement();
                    return false;
                }
                ggml_backend_buffer_set_usage(tensor_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
                ggml_backend_htp_buffer_set_session_index(tensor_buffer, -1);
                htp_param_placement_buffers_.push_back(tensor_buffer);
                if (std::getenv("SD_HTP_MULTI_SESSION_TRACE") != nullptr) {
                    ggml_backend_htp_rpcmem_info info = {};
                    const bool have_info = ggml_backend_htp_get_rpcmem_info(tensor_buffer, &info);
                    LOG_INFO("%s HTP tensor allocation tensor=%s bytes=%zu fd=%d logical=%zu allocation=%zu",
                             get_desc().c_str(), tensors[index]->name, chunk_sizes[index],
                             have_info ? info.fd : -1,
                             have_info ? info.logical_size : 0,
                             have_info ? info.allocation_size : 0);
                }
                if (htp_param_placement_buffer_ == nullptr) {
                    htp_param_placement_buffer_ = tensor_buffer;
                }
                ggml_tensor * htp_tensor = htp_tensors[index];
                htp_tensor->buffer = tensor_buffer;
                htp_tensor->data = ggml_backend_buffer_get_base(tensor_buffer);
                htp_tensor->extra = nullptr;
            }
            LOG_INFO("%s per-tensor non-staging HTP params: tensors=%zu allocations=%zu sessions=%d",
                     get_desc().c_str(), tensors.size(), htp_param_placement_buffers_.size(), session_count);
        } else {
            htp_param_placement_buffer_ = ggml_backend_alloc_ctx_tensors_from_buft(htp_param_placement_ctx_, htp_buft);
        }
        if (htp_param_placement_buffer_ == nullptr) {
            LOG_ERROR("%s alloc HTP packed-only params buffer failed, num_tensors = %zu",
                      get_desc().c_str(),
                      htp_param_placement_pairs_.size());
            htp_param_placement_pairs_.clear();
            htp_placed_param_tensors_.clear();
            ggml_free(htp_param_placement_ctx_);
            htp_param_placement_ctx_ = nullptr;
            return false;
        }
        if (!owner_aware) {
            ggml_backend_buffer_set_usage(htp_param_placement_buffer_, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        } else {
            for (ggml_backend_buffer_t buffer : htp_param_placement_buffers_) {
                ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            }
        }

        size_t placed_bytes = 0;
        for (auto& pair : htp_param_placement_pairs_) {
            ggml_tensor* tensor     = pair.first;
            ggml_tensor* htp_tensor = pair.second;
            if (!owner_aware) {
                bind_tensor_allocation(tensor, htp_tensor);
            } else {
                tensor->buffer = htp_tensor->buffer;
                tensor->data = htp_tensor->data;
                tensor->extra = htp_tensor->extra;
            }
            htp_placed_param_tensors_.insert(tensor);
            placed_bytes += ggml_nbytes(tensor);
        }

        LOG_INFO("%s placed packed-only HTP F16 params in RPCMEM (%6.2f MB, %zu tensors)",
                 get_desc().c_str(),
                 placed_bytes / (1024.f * 1024.f),
                 htp_placed_param_tensors_.size());
        return true;
    }

    bool prepare_htp_packed_f16_weights(ggml_cgraph* gf = nullptr) {
        if (!htp_op_routing_enabled() || htp_backend == nullptr) {
            return true;
        }

        // 推理前只做“自注册”：original tensor 和 packed tensor 是同一个对象。
        // 这样运行时没有在线重排、没有额外常驻 packed buffer，也不会保留 row-major 副本。
        std::vector<ggml_tensor*> tensors = collect_htp_packable_f16_weights(gf);
        if (tensors.empty()) {
            return true;
        }
        size_t self_registered_bytes = 0;
        for (ggml_tensor* tensor : tensors) {
            if (!htp_packed_f16_weight_is_prepared(tensor)) {
                sd_backend_htp_register_packed_f16_weight(tensor, tensor);
                htp_packed_self_registered_weights_.push_back(tensor);
                self_registered_bytes += ggml_nbytes(tensor);
            }
        }

        if (self_registered_bytes > 0) {
            LOG_INFO("%s registered offline packed HTP F16 weights (%6.2f MB, %zu tensors)",
                     get_desc().c_str(),
                     self_registered_bytes / (1024.f * 1024.f),
                     htp_packed_self_registered_weights_.size());
        }
        log_htp_packed_f16_weight_hashes_for_debug(tensors);
        return true;
    }

    void free_htp_packed_weights() {
        for (ggml_tensor* tensor : htp_packed_self_registered_weights_) {
            sd_backend_htp_unregister_packed_f16_weight(tensor);
        }
        htp_packed_self_registered_weights_.clear();
    }

    void free_htp_param_placement() {
        free_htp_packed_weights();
        for (auto& pair : htp_param_placement_pairs_) {
            ggml_tensor* tensor = pair.first;
            if (tensor != nullptr) {
                tensor->buffer = nullptr;
                tensor->data   = nullptr;
                tensor->extra  = nullptr;
            }
        }
        htp_param_placement_pairs_.clear();
        htp_placed_param_tensors_.clear();

        if (htp_param_placement_buffer_ != nullptr &&
            std::find(htp_param_placement_buffers_.begin(), htp_param_placement_buffers_.end(),
                      htp_param_placement_buffer_) == htp_param_placement_buffers_.end()) {
            ggml_backend_buffer_free(htp_param_placement_buffer_);
        }
        for (ggml_backend_buffer_t buffer : htp_param_placement_buffers_) {
            if (buffer != nullptr) {
                ggml_backend_buffer_free(buffer);
            }
        }
        htp_param_placement_buffers_.clear();
        for (ggml_context* ctx : htp_param_placement_ctxs_) {
            if (ctx != nullptr) {
                ggml_free(ctx);
            }
        }
        htp_param_placement_ctxs_.clear();
        htp_param_placement_buffer_ = nullptr;
        htp_param_placement_ctx_ = nullptr;
    }

    // 参数张量做局部 offload 时必须落到根张量，partial offload 当前不接受 view 张量本身。
    static ggml_tensor* get_htp_tensor_backing_root(ggml_tensor* tensor) {
        while (tensor != nullptr && tensor->view_src != nullptr) {
            tensor = tensor->view_src;
        }
        return tensor;
    }

    // 只有直接参与 HTP matmul 的 node/weight/activation 需要进入 RPCMEM，不能把 view 的 backing root 整块规划进去。
    bool tensor_directly_used_by_htp_target_op(ggml_tensor* tensor, const std::vector<ggml_tensor*>& target_nodes) const {
        if (!htp_op_routing_enabled() || tensor == nullptr) {
            return false;
        }
        for (ggml_tensor* node : target_nodes) {
            if (tensor == node) {
                return true;
            }
            if (node->op == GGML_OP_MUL_MAT &&
                node->src[0] != nullptr &&
                tensor == node->src[0] &&
                htp_packed_f16_weight_is_prepared(node->src[0])) {
                continue;
            }
            for (int j = 0; j < GGML_MAX_SRC; ++j) {
                ggml_tensor* src = node->src[j];
                if (tensor == src) {
                    return true;
                }
                if (j == 0 && node->op == GGML_OP_MUL_MAT) {
                    continue;
                }
                // HTP attention inputs are commonly PERMUTE/RESHAPE views. The
                // allocator cannot assign a buffer to the view itself, so the
                // backing root must use the same RPCMEM placement as the
                // attention node. Fused anchor follows the original dense
                // FLASH_ATTN_EXT input contract.
                const bool htp_attention_input =
                    node->op == GGML_OP_FLASH_ATTN_EXT ||
                    (node->op == GGML_OP_CUSTOM &&
                     std::strcmp(node->name, "htp_fused_anchor_attention") == 0);
                if (htp_attention_input && src != nullptr &&
                    tensor == get_htp_tensor_backing_root(src)) {
                    return true;
                }
                if (src != nullptr && ggml_is_contiguous(src) && tensor == get_htp_tensor_backing_root(src)) {
                    return true;
                }
            }
        }
        return false;
    }

    // 全量收集符合条件且权重已直接放置到 RPCMEM 的 diffusion F16 matmul。
    std::vector<ggml_tensor*> collect_planned_htp_target_ops(ggml_cgraph* gf) const {
        std::vector<ggml_tensor*> selected_nodes;
        if (!htp_op_routing_enabled() || gf == nullptr) {
            return selected_nodes;
        }

        for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
            ggml_tensor* node = ggml_graph_node(gf, i);
            if (!is_htp_target_op(node)) {
                continue;
            }
            selected_nodes.push_back(node);
        }

        return selected_nodes;
    }

    bool validate_htp_packed_only_consumers(ggml_cgraph* gf) {
        if (gf == nullptr) {
            return true;
        }

        // packed-only tensor 的内存布局已经不是 CPU matmul 可解释的 row-major。
        // 如果图里任何非 HTP matmul 消费它，必须 fail fast，不能静默 CPU fallback。
        bool valid = true;
        for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
            ggml_tensor* node = ggml_graph_node(gf, i);
            if (node == nullptr) {
                continue;
            }
            for (int j = 0; j < GGML_MAX_SRC; ++j) {
                ggml_tensor* src = node->src[j];
                ggml_tensor* root = get_htp_tensor_backing_root(src);
                if (root == nullptr || (root->flags & SD_GGML_TENSOR_FLAG_HTP_PACKED_W16A32) == 0) {
                    continue;
                }

                const bool routed_to_htp_matmul = j == 0 &&
                                                   node->op == GGML_OP_MUL_MAT &&
                                                   src == node->src[0] &&
                                                   is_htp_target_op(node);
                if (j == 0 && node->op == GGML_OP_MUL_MAT && !routed_to_htp_matmul &&
                    sd_env_flag_enabled("SD_HTP_PACKED_FLUX2_TRACE")) {
                    LOG_INFO("flux2_packed_matmul_skip weight_contig=%d activation_contig=%d output_contig=%d backend_support=%d prepared=%d weight=[%lld,%lld] activation=[%lld,%lld,%lld,%lld]",
                             ggml_is_contiguous(node->src[0]) ? 1 : 0,
                             node->src[1] != nullptr && ggml_is_contiguous(node->src[1]) ? 1 : 0,
                             ggml_is_contiguous(node) ? 1 : 0,
                             ggml_backend_supports_op(htp_backend, node) ? 1 : 0,
                             htp_packed_f16_weight_is_prepared(node->src[0]) ? 1 : 0,
                             (long long) node->src[0]->ne[0], (long long) node->src[0]->ne[1],
                             node->src[1] != nullptr ? (long long) node->src[1]->ne[0] : 0,
                             node->src[1] != nullptr ? (long long) node->src[1]->ne[1] : 0,
                             node->src[1] != nullptr ? (long long) node->src[1]->ne[2] : 0,
                             node->src[1] != nullptr ? (long long) node->src[1]->ne[3] : 0);
                }
                // The dual packed QKV custom node consumes two offline-packed
                // weights at src[1] and src[6]. They are HTP inputs, not
                // row-major CPU consumers, so keep them in the packed-only
                // placement contract.
                const bool routed_to_htp_packed_qkv =
                    node->op == GGML_OP_CUSTOM &&
                    (( (std::strcmp(node->name, "qkv_mul_rms_packed_dual") == 0 ||
                        std::strcmp(node->name, "qkv_mul_rms_packed_dual_rope") == 0) && (j == 1 || j == 6)) ||
                     ((std::strcmp(node->name, "qkv_mul_rms_packed") == 0 ||
                       std::strcmp(node->name, "qkv_mul_rms_packed_rope") == 0) && j == 1)) &&
                    is_htp_target_op(node);
                const bool routed_packed_view = j == 0 && node->op == GGML_OP_VIEW &&
                                                (node->flags & SD_GGML_TENSOR_FLAG_HTP_PACKED_W16A32) != 0 &&
                                                ggml_is_contiguous(node);
                const bool routed_to_htp = routed_to_htp_matmul || routed_to_htp_packed_qkv || routed_packed_view;
                if (routed_to_htp) {
                    continue;
                }

                LOG_ERROR("%s packed-only HTP tensor '%s' is consumed by unsupported op '%s' node '%s' src[%d]; packed layout cannot fall back to CPU; weight=[%lld,%lld,%lld,%lld] activation=[%lld,%lld,%lld,%lld] dst=[%lld,%lld,%lld,%lld]",
                          get_desc().c_str(),
                          root->name,
                          ggml_op_name(node->op),
                          node->name,
                          j,
                          (long long) root->ne[0],
                          (long long) root->ne[1],
                          (long long) root->ne[2],
                          (long long) root->ne[3],
                          node->src[1] != nullptr ? (long long) node->src[1]->ne[0] : 0,
                          node->src[1] != nullptr ? (long long) node->src[1]->ne[1] : 0,
                          node->src[1] != nullptr ? (long long) node->src[1]->ne[2] : 0,
                          node->src[1] != nullptr ? (long long) node->src[1]->ne[3] : 0,
                          (long long) node->ne[0],
                          (long long) node->ne[1],
                          (long long) node->ne[2],
                          (long long) node->ne[3]);
                valid = false;
            }
        }
        return valid;
    }

    void plan_compute_buffer_ids(ggml_cgraph* gf,
                                 std::vector<int>* node_buffer_ids,
                                 std::vector<int>* leaf_buffer_ids) const {
        GGML_ASSERT(gf != nullptr);
        GGML_ASSERT(node_buffer_ids != nullptr);
        GGML_ASSERT(leaf_buffer_ids != nullptr);

        node_buffer_ids->assign(ggml_graph_n_nodes(gf), 0);
        leaf_buffer_ids->assign(sd::ggml_graph_cut::leaf_count(gf), 0);
        std::vector<ggml_tensor*> target_nodes = collect_planned_htp_target_ops(gf);

        // 只把 RMS_NORM / MUL_MAT 及其直接输入输出放到 RPCMEM，其他张量维持原 runtime buffer。
        for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
            ggml_tensor* node = ggml_graph_node(gf, i);
            if (tensor_directly_used_by_htp_target_op(node, target_nodes)) {
                (*node_buffer_ids)[i] = 1;
            }
        }
        for (int i = 0; i < sd::ggml_graph_cut::leaf_count(gf); ++i) {
            ggml_tensor* leaf = sd::ggml_graph_cut::leaf_tensor(gf, i);
            if (tensor_directly_used_by_htp_target_op(leaf, target_nodes)) {
                (*leaf_buffer_ids)[i] = 1;
            }
        }

        // Residual-owned OpenCL sparse attention gets a separate arena.  The
        // sparse node itself is placed there as well, which propagates the
        // allocation choice to its Q/K/V roots through gallocr. The mask is
        // intentionally kept in the normal arena because it may have a
        // separate preparation/staging consumer.
        if (!sd_env_flag_enabled("SD_HTP_SPARSE_OPENCL")) {
            return;
        }
        auto sparse_candidate = [](const ggml_tensor * node) {
            if (node == nullptr || node->op != GGML_OP_FLASH_ATTN_EXT ||
                node->src[0] == nullptr || node->src[1] == nullptr ||
                node->src[2] == nullptr || node->src[3] == nullptr ||
                node->src[4] != nullptr ||
                node->src[1]->type != GGML_TYPE_F16 || node->src[2]->type != GGML_TYPE_F16 ||
                node->src[3]->type != GGML_TYPE_I32 ||
                (node->type != GGML_TYPE_F32 && node->type != GGML_TYPE_F16) ||
                (node->src[0]->type != GGML_TYPE_F32 && node->src[0]->type != GGML_TYPE_F16)) {
                return false;
            }
            int32_t block_q = 0;
            int32_t block_k = 0;
            int32_t with_lse = 0;
            std::memcpy(&block_q, node->op_params + 4, sizeof(block_q));
            std::memcpy(&block_k, node->op_params + 5, sizeof(block_k));
            std::memcpy(&with_lse, node->op_params + 6, sizeof(with_lse));
            return (node->src[0]->ne[0] == 64 || node->src[0]->ne[0] == 128) &&
                   block_q == 32 && block_k == 32 && with_lse == 1;
        };
        auto root_of = [](ggml_tensor * tensor) {
            while (tensor != nullptr && tensor->view_src != nullptr) {
                tensor = tensor->view_src;
            }
            return tensor;
        };
        auto view_only = [](const ggml_tensor * tensor) {
            return tensor != nullptr &&
                (tensor->op == GGML_OP_VIEW || tensor->op == GGML_OP_RESHAPE ||
                 tensor->op == GGML_OP_PERMUTE || tensor->op == GGML_OP_TRANSPOSE);
        };
        std::unordered_set<ggml_tensor *> sparse_roots;
        for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
            ggml_tensor * node = ggml_graph_node(gf, i);
            if (sd_env_flag_enabled("SD_HTP_QKV_RELEASE_TRACE") &&
                node != nullptr && node->op == GGML_OP_FLASH_ATTN_EXT &&
                node->src[3] != nullptr && node->src[3]->type == GGML_TYPE_I32) {
                int32_t bq = 0, bk = 0, lse = 0;
                std::memcpy(&bq, node->op_params + 4, sizeof(bq));
                std::memcpy(&bk, node->op_params + 5, sizeof(bk));
                std::memcpy(&lse, node->op_params + 6, sizeof(lse));
                LOG_INFO("qkv_release_candidate node=%s qtype=%s ktype=%s vtype=%s qdim=%lld block=%d/%d lse=%d match=%d",
                         node->name, ggml_type_name(node->src[0]->type),
                         ggml_type_name(node->src[1]->type), ggml_type_name(node->src[2]->type),
                         (long long) node->src[0]->ne[0], bq, bk, lse,
                         sparse_candidate(node) ? 1 : 0);
            }
            if (!sparse_candidate(node)) {
                continue;
            }
            bool conflict = false;
            for (int j = 0; j < ggml_graph_n_nodes(gf) && !conflict; ++j) {
                ggml_tensor * consumer = ggml_graph_node(gf, j);
                if (consumer == node || view_only(consumer)) {
                    continue;
                }
                for (int s = 0; s < GGML_MAX_SRC; ++s) {
                    ggml_tensor * src = consumer->src[s];
                    for (int q = 0; q < 4; ++q) {
                        ggml_tensor * sparse_root = root_of(node->src[q]);
                        // The allocating root's producer is not a second
                        // consumer. CPY commonly carries its destination as
                        // src[1], which otherwise looks self-referential here.
                        if (consumer == sparse_root) {
                            continue;
                        }
                        if (sparse_root == root_of(src) &&
                            !sparse_candidate(consumer) &&
                            !(q < 3 &&
                              sd_htp_anchor_dense_consumer_matches_qkv(
                                  node, consumer, root_of) &&
                              is_htp_supported_flash_attn(consumer))) {
                            if (sd_env_flag_enabled("SD_HTP_QKV_RELEASE_TRACE")) {
                                LOG_INFO("qkv_release_conflict sparse=%s root=%s consumer=%s op=%s src=%d",
                                         node->name, root_of(node->src[q])->name,
                                         consumer->name, ggml_op_name(consumer->op), s);
                            }
                            conflict = true;
                            break;
                        }
                    }
                    if (conflict) {
                        break;
                    }
                }
            }
            if (!conflict) {
                // Only Q/K/V are shared with the OpenCL sidecar and can be
                // reclaimed as soon as dense and sparse attention finish.
                // Masks remain in the normal arena because their graph-side
                // preparation/staging may have independent consumers.
                for (int q = 0; q < 3; ++q) {
                    if (ggml_tensor * root = root_of(node->src[q])) {
                        sparse_roots.insert(root);
                    }
                }
            }
        }
        if (!sparse_roots.empty()) {
            for (int i = 0; i < sd::ggml_graph_cut::leaf_count(gf); ++i) {
                if (sparse_roots.count(root_of(sd::ggml_graph_cut::leaf_tensor(gf, i)))) {
                    (*leaf_buffer_ids)[i] = 2;
                }
            }
            for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
                ggml_tensor * node = ggml_graph_node(gf, i);
                if (sparse_roots.count(root_of(node))) {
                    (*node_buffer_ids)[i] = 2;
                }
            }
        }
    }

    void free_cache_buffer() {
        for (CacheStorageSegment& segment : cache_storage_segments_) {
            if (segment.buffer != nullptr) {
                ggml_backend_buffer_free(segment.buffer);
                segment.buffer = nullptr;
            }
        }
    }

    void free_cache_storage_segment(CacheStorageSegment& segment) {
        if (segment.buffer != nullptr) {
            ggml_backend_buffer_free(segment.buffer);
            segment.buffer = nullptr;
        }
        if (segment.ctx != nullptr) {
            ggml_free(segment.ctx);
            segment.ctx = nullptr;
        }
        segment.names.clear();
    }

    bool allocate_cache_storage_segment(
            const std::map<std::string, ggml_tensor*>& sources,
            CacheStorageSegment* segment_out) {
        GGML_ASSERT(segment_out != nullptr);
        if (htp_backend != nullptr && htp_backend != runtime_backend) {
            ggml_backend_synchronize(htp_backend);
        }
        CacheStorageSegment segment;
        segment.ctx = new_cache_ctx();
        std::vector<std::pair<ggml_tensor*, ggml_tensor*>> source_to_cache_tensors;
        source_to_cache_tensors.reserve(sources.size());
        for (const auto& kv : sources) {
            ggml_tensor* source_tensor = sd::ggml_graph_cut::cache_source_tensor(kv.second);
            ggml_tensor* cache_tensor = ggml_dup_tensor(segment.ctx, source_tensor);
            ggml_set_name(cache_tensor, kv.first.c_str());
            source_to_cache_tensors.push_back({source_tensor, cache_tensor});
            segment.names.insert(kv.first);
        }
        const size_t num_tensors = ggml_tensor_num(segment.ctx);
        segment.buffer = ggml_backend_alloc_ctx_tensors(segment.ctx, runtime_backend);
        if (segment.buffer == nullptr) {
            LOG_ERROR("%s alloc cache backend buffer failed, num_tensors=%zu",
                      get_desc().c_str(), num_tensors);
            free_cache_storage_segment(segment);
            return false;
        }
        for (const auto& kv : source_to_cache_tensors) {
            ggml_tensor* src              = kv.first;
            ggml_tensor* dst              = kv.second;
            ggml_backend_buffer_t src_buf = sd::ggml_graph_cut::tensor_buffer(src);
            ggml_backend_buffer_t dst_buf = sd::ggml_graph_cut::tensor_buffer(dst);
            if (src_buf == nullptr || dst_buf == nullptr) {
                LOG_ERROR("%s cache copy tensor buffer missing: name=%s op=%s src0=%p src0_name=%s src0_buffer=%p src_buffer=%p src_view_src=%p src_view_src_buffer=%p dst_buffer=%p",
                          get_desc().c_str(),
                          src && src->name[0] != '\0' ? src->name : "<unnamed>",
                          src ? ggml_op_name(src->op) : "<null>",
                          src ? src->src[0] : nullptr,
                          (src && src->src[0] && src->src[0]->name[0] != '\0') ? src->src[0]->name : "<unnamed>",
                          (src && src->src[0]) ? sd::ggml_graph_cut::tensor_buffer(src->src[0]) : nullptr,
                          src ? src->buffer : nullptr,
                          src ? src->view_src : nullptr,
                          (src && src->view_src) ? src->view_src->buffer : nullptr,
                          dst ? dst->buffer : nullptr);
                free_cache_storage_segment(segment);
                return false;
            }
            uint64_t source_hash = 0;
            if (sd_env_flag_enabled("SD_HTP_GRAPH_CUT_CACHE_HASH")) {
                source_hash = hash_tensor_bytes_for_debug(src);
                log_graph_cut_cache_hash("source", kv.second->name, src, source_hash);
            }
            const bool use_staging_copy = src->view_src != nullptr ||
                                          !ggml_is_contiguous(src) ||
                                          src->buffer == nullptr;
            if (use_staging_copy) {
                std::vector<uint8_t> host_data(ggml_nbytes(src));
                ggml_backend_tensor_get(src, host_data.data(), 0, host_data.size());
                ggml_backend_tensor_set(dst, host_data.data(), 0, host_data.size());
            } else {
                ggml_backend_tensor_copy(src, dst);
            }
            if (sd_env_flag_enabled("SD_HTP_GRAPH_CUT_CACHE_HASH")) {
                const uint64_t destination_hash = hash_tensor_bytes_for_debug(dst);
                log_graph_cut_cache_hash("destination", kv.second->name, dst, destination_hash);
                std::fprintf(stderr,
                             "htp_graph_cut_cache_hash_compare name=%s bytes=%zu source=%016" PRIx64
                             " destination=%016" PRIx64 " match=%d\n",
                             kv.second->name,
                             ggml_nbytes(src), source_hash, destination_hash,
                             source_hash == destination_hash ? 1 : 0);
            }
        }
        ggml_backend_synchronize(runtime_backend);
        const size_t cache_buffer_size = ggml_backend_buffer_get_size(segment.buffer);
        LOG_DEBUG("%s cache backend buffer size = % 6.2f MB(%s) (%zu tensors)",
                  get_desc().c_str(),
                  cache_buffer_size / (1024.f * 1024.f),
                  ggml_backend_is_cpu(runtime_backend) ? "RAM" : "VRAM",
                  num_tensors);
        *segment_out = std::move(segment);
        return true;
    }

    bool copy_cache_tensors_to_cache_buffer(
            const std::unordered_set<std::string>* cache_keep_names = nullptr,
            ggml_cgraph* gf = nullptr) {
        std::unordered_set<const ggml_tensor*> graph_nodes;
        if (gf != nullptr) {
            for (int node_idx = 0; node_idx < ggml_graph_n_nodes(gf); ++node_idx) {
                graph_nodes.insert(ggml_graph_node(gf, node_idx));
            }
        }
        std::map<std::string, ggml_tensor*> pending_sources;
        for (auto entry = cache_tensor_map.begin(); entry != cache_tensor_map.end();) {
            if (gf != nullptr && graph_nodes.count(entry->second) == 0) {
                ++entry;
                continue;
            }
            const auto kv = *entry;
            entry = cache_tensor_map.erase(entry);
            if (cache_keep_names != nullptr &&
                cache_keep_names->find(kv.first) == cache_keep_names->end()) {
                continue;
            }
            if (!async_residual_cache_backed(kv.first)) {
                pending_sources[kv.first] = kv.second;
            }
        }

        if (cache_keep_names == nullptr && pending_sources.empty()) {
            return true;
        }

        std::vector<std::unordered_set<std::string>> existing_segment_names;
        existing_segment_names.reserve(cache_storage_segments_.size());
        for (const CacheStorageSegment& segment : cache_storage_segments_) {
            existing_segment_names.push_back(segment.names);
        }
        std::unordered_set<std::string> pending_names;
        pending_names.reserve(pending_sources.size());
        for (const auto& pending : pending_sources) {
            pending_names.insert(pending.first);
        }
        const sd_cache_storage_update_plan update_plan =
            sd_plan_cache_storage_update(existing_segment_names,
                                         pending_names,
                                         cache_keep_names != nullptr);

        if (update_plan.mode != SD_CACHE_STORAGE_MERGE) {
            const size_t segments_before = cache_storage_segments_.size();
            for (auto it = update_plan.replace_segment_indices.rbegin();
                 it != update_plan.replace_segment_indices.rend(); ++it) {
                const size_t segment_idx = *it;
                GGML_ASSERT(segment_idx < cache_storage_segments_.size());
                free_cache_storage_segment(cache_storage_segments_[segment_idx]);
                cache_storage_segments_.erase(
                    cache_storage_segments_.begin() + segment_idx);
            }
            CacheStorageSegment new_segment;
            if (!allocate_cache_storage_segment(pending_sources, &new_segment)) {
                return false;
            }
            const size_t new_segment_bytes =
                ggml_backend_buffer_get_size(new_segment.buffer);
            cache_storage_segments_.push_back(std::move(new_segment));
            if (sd_env_flag_enabled("SD_HTP_SPARSE_OPENCL_TRACE")) {
                LOG_INFO("cache_storage_update mode=%s segments_before=%zu replaced=%zu segments_after=%zu new_bytes=%zu",
                         update_plan.mode == SD_CACHE_STORAGE_APPEND ? "append" : "replace",
                         segments_before,
                         update_plan.replace_segment_indices.size(),
                         cache_storage_segments_.size(),
                         new_segment_bytes);
            }
            return true;
        }

        std::map<std::string, ggml_tensor*> merged_cache_sources;
        for (const CacheStorageSegment& segment : cache_storage_segments_) {
            if (segment.ctx == nullptr) {
                continue;
            }
            for (ggml_tensor* tensor = ggml_get_first_tensor(segment.ctx);
                 tensor != nullptr;
                 tensor = ggml_get_next_tensor(segment.ctx, tensor)) {
                if (cache_keep_names != nullptr &&
                    cache_keep_names->find(tensor->name) == cache_keep_names->end()) {
                    continue;
                }
                if (!async_residual_cache_backed(tensor->name)) {
                    merged_cache_sources[tensor->name] = tensor;
                }
            }
        }
        for (const auto& pending : pending_sources) {
            merged_cache_sources[pending.first] = pending.second;
        }
        if (merged_cache_sources.empty()) {
            free_cache_buffer();
            free_cache_ctx();
            return true;
        }

        CacheStorageSegment merged_segment;
        if (!allocate_cache_storage_segment(merged_cache_sources, &merged_segment)) {
            return false;
        }
        free_cache_buffer();
        free_cache_ctx();
        cache_storage_segments_.push_back(std::move(merged_segment));
        return true;
    }

    template <typename T>
    std::optional<sd::Tensor<T>> read_graph_tensor(ggml_tensor* tensor, const char* label) {
        if (tensor == nullptr) {
            LOG_ERROR("%s %s tensor is null", get_desc().c_str(), label);
            return std::nullopt;
        }
        if (tensor->type != sd::GGMLTypeTraits<T>::type) {
            LOG_ERROR("%s %s tensor type mismatch: got %s",
                      get_desc().c_str(),
                      label,
                      ggml_type_name(tensor->type));
            return std::nullopt;
        }
        ggml_backend_buffer_t buf = sd::ggml_graph_cut::tensor_buffer(tensor);
        if (buf == nullptr) {
            LOG_ERROR("%s %s tensor buffer missing: name=%s op=%s buffer=%p view_src=%p view_src_buffer=%p data=%p",
                      get_desc().c_str(),
                      label,
                      tensor->name[0] != '\0' ? tensor->name : "<unnamed>",
                      ggml_op_name(tensor->op),
                      tensor->buffer,
                      tensor->view_src,
                      tensor->view_src ? tensor->view_src->buffer : nullptr,
                      tensor->data);
            return std::nullopt;
        }

        sd::Tensor<T> result(sd::shape_from_ggml(tensor));
        if (tensor->view_src != nullptr || !ggml_is_contiguous(tensor) || tensor->buffer == nullptr) {
            ggml_backend_tensor_get(tensor, result.data(), 0, ggml_nbytes(tensor));
        } else {
            ggml_backend_tensor_get(tensor, result.data(), 0, ggml_nbytes(tensor));
        }
        return result;
    }

    void copy_data_to_backend_tensor(ggml_cgraph* gf, bool clear_after_copy = true) {
        GGML_ASSERT(gf != nullptr);
        std::unordered_set<const ggml_tensor*> graph_tensor_set;
        const int n_leafs = sd::ggml_graph_cut::leaf_count(gf);
        const int n_nodes = ggml_graph_n_nodes(gf);
        graph_tensor_set.reserve(static_cast<size_t>(n_leafs + n_nodes));
        for (int i = 0; i < n_leafs; ++i) {
            graph_tensor_set.insert(sd::ggml_graph_cut::leaf_tensor(gf, i));
        }
        for (int i = 0; i < n_nodes; ++i) {
            graph_tensor_set.insert(ggml_graph_node(gf, i));
        }

        for (auto& kv : backend_tensor_data_map) {
            auto tensor = kv.first;
            auto data   = kv.second;
            if (tensor == nullptr || data == nullptr) {
                continue;
            }
            const char* name = ggml_get_name(tensor);
            if (graph_tensor_set.find(tensor) == graph_tensor_set.end()) {
                continue;
            }
            if (tensor->buffer == nullptr) {
                LOG_WARN("%s skip backend tensor copy: tensor buffer not set, name='%s', ne=[%lld,%lld,%lld,%lld], type=%s",
                         get_desc().c_str(),
                         name != nullptr ? name : "",
                         (long long)tensor->ne[0],
                         (long long)tensor->ne[1],
                         (long long)tensor->ne[2],
                         (long long)tensor->ne[3],
                         ggml_type_name(tensor->type));
                continue;
            }

            ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
            if (buf == nullptr) {
                LOG_WARN("%s graph exec skip tensor copy: name=%s op=%s reason=buffer_not_set data=%p view_src=%p view_src_buffer=%p",
                         get_desc().c_str(),
                         tensor && tensor->name[0] != '\0' ? tensor->name : "<unnamed>",
                         tensor ? ggml_op_name(tensor->op) : "<null>",
                         data,
                         tensor ? tensor->view_src : nullptr,
                         (tensor && tensor->view_src) ? tensor->view_src->buffer : nullptr);
                continue;
            }

            ggml_backend_tensor_set(tensor, data, 0, ggml_nbytes(tensor));
        }

        if (clear_after_copy) {
            backend_tensor_data_map.clear();
        }
    }

    bool bind_cache_backed_tensors(
            CacheInputBindingTransaction* transaction,
            ggml_cgraph* gf) {
        if (transaction == nullptr || transaction->active) {
            return false;
        }
        GGML_ASSERT(active_async_cache_consumers_.empty());
        transaction->bindings.clear();
        transaction->bindings.reserve(cache_backed_tensor_name_map.size());
        transaction->active = true;
        transaction->committed = false;
        std::unordered_set<const ggml_tensor*> graph_tensors;
        for (int node_idx = 0; node_idx < ggml_graph_n_nodes(gf); ++node_idx) {
            graph_tensors.insert(ggml_graph_node(gf, node_idx));
        }
        for (int leaf_idx = 0; leaf_idx < sd::ggml_graph_cut::leaf_count(gf); ++leaf_idx) {
            graph_tensors.insert(sd::ggml_graph_cut::leaf_tensor(gf, leaf_idx));
        }
        for (const auto& kv : cache_backed_tensor_name_map) {
            if (graph_tensors.count(kv.first) == 0) {
                continue;
            }
            ggml_tensor* graph_tensor = kv.first;
            ggml_tensor* cache_tensor = nullptr;
            uint64_t async_generation = 0;
            const auto generation_it =
                cache_backed_tensor_generation_map_.find(graph_tensor);
            if (generation_it != cache_backed_tensor_generation_map_.end()) {
                async_generation = generation_it->second;
                AsyncResidualCacheGeneration* cached =
                    find_async_residual_cache_slot(kv.second, async_generation);
                if (cached != nullptr && cached->slot != nullptr) {
                    cache_tensor =
                        sd_backend_htp_async_cache_slot_tensor(cached->slot);
                }
            } else {
                cache_tensor = get_cache_tensor_by_name(kv.second);
            }
            if (graph_tensor == nullptr || cache_tensor == nullptr) {
                LOG_WARN("%s block sparse residual cache miss while binding: %s",
                         get_desc().c_str(),
                         kv.second.c_str());
                return false;
            }
            if (graph_tensor->type != cache_tensor->type) {
                LOG_WARN("%s block sparse residual cache type mismatch while binding: %s graph=%s cache=%s",
                         get_desc().c_str(), kv.second.c_str(),
                         ggml_type_name(graph_tensor->type),
                         ggml_type_name(cache_tensor->type));
                return false;
            }
            for (int i = 0; i < GGML_MAX_DIMS; ++i) {
                if (graph_tensor->ne[i] != cache_tensor->ne[i]) {
                    LOG_WARN("%s block sparse residual cache shape mismatch while binding: %s graph=[%lld,%lld,%lld,%lld] cache=[%lld,%lld,%lld,%lld]",
                             get_desc().c_str(),
                             kv.second.c_str(),
                             (long long) graph_tensor->ne[0],
                             (long long) graph_tensor->ne[1],
                             (long long) graph_tensor->ne[2],
                             (long long) graph_tensor->ne[3],
                             (long long) cache_tensor->ne[0],
                             (long long) cache_tensor->ne[1],
                             (long long) cache_tensor->ne[2],
                             (long long) cache_tensor->ne[3]);
                    return false;
                }
            }
            CacheInputBinding binding;
            binding.tensor = graph_tensor;
            binding.buffer = graph_tensor->buffer;
            binding.data = graph_tensor->data;
            binding.extra = graph_tensor->extra;
            binding.op = graph_tensor->op;
            binding.key = kv.second;
            binding.generation = async_generation;
            for (int src_idx = 0; src_idx < GGML_MAX_SRC; ++src_idx) {
                binding.src[src_idx] = graph_tensor->src[src_idx];
            }
            transaction->bindings.push_back(std::move(binding));
            CacheInputBinding& active_binding = transaction->bindings.back();
            if (async_generation != 0) {
                char reason[512] = {};
                if (htp_backend == nullptr ||
                    !sd_backend_htp_opencl_bind_cache_input(
                        htp_backend, graph_tensor, kv.second.c_str(),
                        async_generation, reason, sizeof(reason))) {
                    LOG_ERROR(
                        "%s bind async residual cache input failed: key=%s "
                        "generation=%" PRIu64 " reason=%s",
                        get_desc().c_str(), kv.second.c_str(), async_generation,
                        reason[0] != '\0' ? reason : "unknown error");
                    return false;
                }
                AsyncResidualCacheGeneration* cached =
                    find_async_residual_cache_slot(kv.second, async_generation);
                GGML_ASSERT(cached != nullptr);
                ++cached->consumer_count;
                active_async_cache_consumers_.push_back(
                    { kv.second, async_generation });
                active_binding.consumer_acquired = true;
            }
            graph_tensor->buffer = cache_tensor->buffer;
            graph_tensor->data   = cache_tensor->data;
            graph_tensor->extra  = cache_tensor->extra;
            for (int src_idx = 0; src_idx < GGML_MAX_SRC; ++src_idx) {
                graph_tensor->src[src_idx] = nullptr;
            }
            graph_tensor->op = GGML_OP_NONE;
        }
        return true;
    }

    void restore_cache_input_bindings(
            CacheInputBindingTransaction* transaction) {
        if (transaction == nullptr || !transaction->active) {
            return;
        }
        release_active_async_cache_consumers();
        for (auto it = transaction->bindings.rbegin();
             it != transaction->bindings.rend(); ++it) {
            CacheInputBinding& binding = *it;
            binding.tensor->buffer = binding.buffer;
            binding.tensor->data = binding.data;
            binding.tensor->extra = binding.extra;
            binding.tensor->op = binding.op;
            for (int src_idx = 0; src_idx < GGML_MAX_SRC; ++src_idx) {
                binding.tensor->src[src_idx] = binding.src[src_idx];
            }
        }
        transaction->bindings.clear();
        transaction->active = false;
    }

    void rollback_cache_input_bindings(
            CacheInputBindingTransaction* transaction) {
        if (transaction == nullptr || transaction->committed) {
            return;
        }
        restore_cache_input_bindings(transaction);
    }

    void commit_cache_input_bindings(
            CacheInputBindingTransaction* transaction) {
        if (transaction == nullptr || !transaction->active) {
            return;
        }
        restore_cache_input_bindings(transaction);
        transaction->committed = true;
    }

    void finalize_cache_input_bindings(
            CacheInputBindingTransaction* transaction,
            ggml_status status) {
        if (status == GGML_STATUS_SUCCESS) {
            commit_cache_input_bindings(transaction);
        } else {
            restore_cache_input_bindings(transaction);
        }
    }

    bool allocate_compute_graph_with_cache_inputs(
            ggml_cgraph* gf,
            CacheInputBindingTransaction* transaction,
            CacheInputAllocationFailurePoint failure_point =
                CacheInputAllocationFailurePoint::none) {
        if (!bind_cache_backed_tensors(transaction, gf)) {
            return false;
        }
        if (failure_point ==
            CacheInputAllocationFailurePoint::first_reserve) {
            return false;
        }
        if (!alloc_compute_buffer(gf)) {
            return false;
        }
        if (failure_point ==
            CacheInputAllocationFailurePoint::second_alloc) {
            return false;
        }
        if (!compute_node_buffer_ids_.empty() ||
            !compute_leaf_buffer_ids_.empty()) {
            return ggml_gallocr_reserve_n(
                       compute_allocr, gf, compute_node_buffer_ids_.data(),
                       compute_leaf_buffer_ids_.data()) &&
                   ggml_gallocr_alloc_graph(compute_allocr, gf);
        }
        return ggml_gallocr_alloc_graph(compute_allocr, gf);
    }

    void rollback_async_residual_cache_outputs(
            AsyncResidualCacheOutputTransaction* transaction) {
        if (transaction == nullptr || !transaction->active) {
            return;
        }
        for (const AsyncResidualCacheOutputBinding& binding :
             transaction->bindings) {
            binding.tensor->buffer = binding.buffer;
            binding.tensor->data = binding.data;
            binding.tensor->extra = binding.extra;
        }
        for (auto it = async_residual_cache_slots_.begin();
             it != async_residual_cache_slots_.end();) {
            if (it->second.generation != transaction->generation) {
                ++it;
                continue;
            }
            sd_backend_htp_async_cache_slot_free(it->second.slot);
            it = async_residual_cache_slots_.erase(it);
        }
        transaction->bindings.clear();
        transaction->generation = 0;
        transaction->active = false;
    }

    void commit_async_residual_cache_outputs(
            AsyncResidualCacheOutputTransaction* transaction) {
        if (transaction == nullptr) {
            return;
        }
        transaction->bindings.clear();
        transaction->generation = 0;
        transaction->active = false;
    }

    bool prepare_async_residual_cache_outputs(
            AsyncResidualCacheOutputTransaction* transaction,
            ggml_cgraph* gf) {
        GGML_ASSERT(transaction != nullptr);
        GGML_ASSERT(!transaction->active);
        if (!sd_htp_opencl_persistent_cache_outputs_enabled()) {
            return true;
        }
        std::vector<std::pair<std::string, ggml_tensor*>> outputs;
        std::unordered_set<const ggml_tensor*> graph_nodes;
        for (int node_idx = 0; node_idx < ggml_graph_n_nodes(gf); ++node_idx) {
            graph_nodes.insert(ggml_graph_node(gf, node_idx));
        }
        for (const auto& entry : cache_tensor_map) {
            ggml_tensor* tensor = entry.second;
            if (graph_nodes.count(tensor) == 0 || tensor == nullptr || tensor->op != GGML_OP_CUSTOM ||
                entry.first.rfind("htp_block_sparse_residual_", 0) != 0) {
                continue;
            }
            outputs.push_back(entry);
        }
        if (outputs.empty()) {
            return true;
        }

        const uint64_t generation = active_async_residual_cache_generation_ != 0
            ? active_async_residual_cache_generation_
            : next_async_residual_cache_generation_++;
        if (!replace_async_residual_cache_slots(outputs, generation)) {
            return false;
        }

        const bool trace = sd_env_flag_enabled("SD_HTP_SPARSE_OPENCL_TRACE");
        bool allocation_failed = false;
        transaction->generation = generation;
        transaction->bindings.reserve(outputs.size());
        transaction->active = true;
        for (const auto& output : outputs) {
            AsyncResidualCacheGeneration* cached =
                allocate_async_residual_cache_slot(
                    output.first, output.second, generation);
            if (cached == nullptr || cached->slot == nullptr) {
                allocation_failed = true;
                break;
            }
            ggml_tensor* slot_tensor =
                sd_backend_htp_async_cache_slot_tensor(cached->slot);
            GGML_ASSERT(slot_tensor != nullptr);
            transaction->bindings.push_back({
                output.second,
                output.second->buffer,
                output.second->data,
                output.second->extra,
            });
            output.second->buffer = slot_tensor->buffer;
            output.second->data = slot_tensor->data;
            output.second->extra = slot_tensor->extra;
            cached->producer_pending = true;
            if (trace) {
                std::fprintf(
                    stderr,
                    "gpu_residual_cache_generation key=%s new=%" PRIu64 " live=1\n",
                    output.first.c_str(), generation);
            }
        }
        if (allocation_failed) {
            rollback_async_residual_cache_outputs(transaction);
            return !sd_htp_opencl_bind_cache_outputs_before_reserve() &&
                   !sd_env_flag_enabled("SD_HTP_SPARSE_OPENCL_STRICT");
        }
        return true;
    }

    bool offload_all_params() {
        restore_partial_params();
        if (params_backend == runtime_backend) {
            return true;
        }
        if (params_on_runtime_backend) {
            return true;
        }
        GGML_ASSERT(runtime_params_buffer == nullptr);
        int64_t t0         = ggml_time_ms();
        if (runtime_offload_pairs_.empty()) {
            for (ggml_tensor* t = ggml_get_first_tensor(params_ctx); t != nullptr; t = ggml_get_next_tensor(params_ctx, t)) {
                GGML_ASSERT(t->view_src == nullptr);
                if (htp_placed_param_tensors_.find(t) != htp_placed_param_tensors_.end()) {
                    continue;
                }
                if (htp_packed_staging_contains(htp_packed_staging_tensors_, t)) {
                    continue;
                }
                ggml_tensor* offload_t = ggml_dup_tensor(offload_ctx, t);
                ggml_set_name(offload_t, t->name);
                runtime_offload_pairs_.push_back({t, offload_t});
            }
        }
        size_t num_tensors = runtime_offload_pairs_.size();
        if (num_tensors == 0) {
            return true;
        }

        runtime_params_buffer = ggml_backend_alloc_ctx_tensors(offload_ctx, runtime_backend);

        if (runtime_params_buffer == nullptr) {
            LOG_ERROR("%s alloc runtime params backend buffer failed, num_tensors = %i",
                      get_desc().c_str(),
                      num_tensors);
            return false;
        }
        ggml_backend_buffer_set_usage(runtime_params_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        for (auto& pair : runtime_offload_pairs_) {
            ggml_tensor* t         = pair.first;
            ggml_tensor* offload_t = pair.second;
            ggml_backend_tensor_copy(t, offload_t);
            std::swap(t->buffer, offload_t->buffer);
            std::swap(t->data, offload_t->data);
            std::swap(t->extra, offload_t->extra);
        }

        int64_t t1 = ggml_time_ms();

        size_t params_buffer_size = ggml_backend_buffer_get_size(runtime_params_buffer);
        LOG_INFO("%s offload params (%6.2f MB, %i tensors) to runtime backend (%s), taking %.2fs",
                 get_desc().c_str(),
                 params_buffer_size / (1024.f * 1024.f),
                 num_tensors,
                 ggml_backend_name(runtime_backend),
                 (t1 - t0) * 1.0f / 1000);

        params_on_runtime_backend = true;

        return true;
    }

    bool offload_partial_params(const std::vector<ggml_tensor*>& tensors) {
        restore_partial_params();
        if (params_backend == runtime_backend) {
            return true;
        }
        if (tensors.empty()) {
            return true;
        }
        GGML_ASSERT(!params_on_runtime_backend);
        GGML_ASSERT(partial_runtime_params_buffer == nullptr);

        std::vector<ggml_tensor*> unique_tensors;
        std::unordered_set<ggml_tensor*> seen_tensors;
        unique_tensors.reserve(tensors.size());
        seen_tensors.reserve(tensors.size());
        for (ggml_tensor* tensor : tensors) {
            if (tensor == nullptr) {
                continue;
            }
            if (htp_placed_param_tensors_.find(tensor) != htp_placed_param_tensors_.end()) {
                continue;
            }
            if (htp_packed_staging_contains(htp_packed_staging_tensors_, tensor)) {
                continue;
            }
            if (seen_tensors.insert(tensor).second) {
                unique_tensors.push_back(tensor);
            }
        }
        if (unique_tensors.empty()) {
            return true;
        }

        ggml_init_params params;
        params.mem_size   = std::max<size_t>(1, unique_tensors.size()) * ggml_tensor_overhead();
        params.mem_buffer = nullptr;
        params.no_alloc   = true;

        partial_offload_ctx = ggml_init(params);
        GGML_ASSERT(partial_offload_ctx != nullptr);

        partial_offload_pairs.clear();
        partial_offload_pairs.reserve(unique_tensors.size());

        for (ggml_tensor* tensor : unique_tensors) {
            GGML_ASSERT(tensor->view_src == nullptr);
            ggml_tensor* offload_tensor = ggml_dup_tensor(partial_offload_ctx, tensor);
            ggml_set_name(offload_tensor, tensor->name);
            partial_offload_pairs.push_back({tensor, offload_tensor});
        }

        partial_runtime_params_buffer = ggml_backend_alloc_ctx_tensors(partial_offload_ctx, runtime_backend);
        if (partial_runtime_params_buffer == nullptr) {
            LOG_ERROR("%s alloc partial runtime params backend buffer failed, num_tensors = %zu",
                      get_desc().c_str(),
                      partial_offload_pairs.size());
            ggml_free(partial_offload_ctx);
            partial_offload_ctx = nullptr;
            partial_offload_pairs.clear();
            return false;
        }
        ggml_backend_buffer_set_usage(partial_runtime_params_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        for (auto& pair : partial_offload_pairs) {
            ggml_tensor* tensor         = pair.first;
            ggml_tensor* offload_tensor = pair.second;

            ggml_backend_tensor_copy(tensor, offload_tensor);
            std::swap(tensor->buffer, offload_tensor->buffer);
            std::swap(tensor->data, offload_tensor->data);
            std::swap(tensor->extra, offload_tensor->extra);
        }

        size_t params_buffer_size = ggml_backend_buffer_get_size(partial_runtime_params_buffer);
        LOG_DEBUG("%s offload partial params (%6.2f MB, %zu tensors) to runtime backend (%s)",
                  get_desc().c_str(),
                  params_buffer_size / (1024.f * 1024.f),
                  partial_offload_pairs.size(),
                  ggml_backend_name(runtime_backend));

        return true;
    }

    void restore_all_params() {
        if (!preserve_htp_packed_staging_) {
            release_htp_packed_staging();
        }
        restore_partial_params();
        if (!params_on_runtime_backend) {
            return;
        }
        for (auto& pair : runtime_offload_pairs_) {
            ggml_tensor* t         = pair.first;
            ggml_tensor* offload_t = pair.second;
            t->buffer         = offload_t->buffer;
            t->data           = offload_t->data;
            t->extra          = offload_t->extra;
            offload_t->buffer = nullptr;
            offload_t->data   = nullptr;
            offload_t->extra  = nullptr;
        }

        if (runtime_params_buffer != nullptr) {
            ggml_backend_buffer_free(runtime_params_buffer);
            runtime_params_buffer = nullptr;
        }
        params_on_runtime_backend = false;
    }

    void restore_partial_params() {
        if (!preserve_htp_packed_staging_) {
            release_htp_packed_staging();
        }
        if (partial_offload_pairs.empty()) {
            if (partial_runtime_params_buffer != nullptr) {
                ggml_backend_buffer_free(partial_runtime_params_buffer);
                partial_runtime_params_buffer = nullptr;
            }
            if (partial_offload_ctx != nullptr) {
                ggml_free(partial_offload_ctx);
                partial_offload_ctx = nullptr;
            }
            return;
        }

        for (auto& pair : partial_offload_pairs) {
            ggml_tensor* tensor         = pair.first;
            ggml_tensor* offload_tensor = pair.second;

            tensor->buffer         = offload_tensor->buffer;
            tensor->data           = offload_tensor->data;
            tensor->extra          = offload_tensor->extra;
            offload_tensor->buffer = nullptr;
            offload_tensor->data   = nullptr;
            offload_tensor->extra  = nullptr;
        }

        if (partial_runtime_params_buffer != nullptr) {
            ggml_backend_buffer_free(partial_runtime_params_buffer);
            partial_runtime_params_buffer = nullptr;
        }
        partial_offload_pairs.clear();

        if (partial_offload_ctx != nullptr) {
            ggml_free(partial_offload_ctx);
            partial_offload_ctx = nullptr;
        }
    }

    bool should_use_graph_cut_segmented_compute(const GraphCutPlan& plan) {
        // Packed staging and graph-cut execution are one opt-in experiment.
        // With the flag unset, preserve the original full-graph path.
        return sd_env_flag_enabled("SD_HTP_PACKED_STAGING") &&
               plan.has_cuts &&
               plan.valid &&
               max_graph_vram_bytes > 0 &&
               plan.segments.size() > 1;
    }

    bool can_attempt_graph_cut_segmented_compute() const {
        return sd_env_flag_enabled("SD_HTP_PACKED_STAGING") &&
               max_graph_vram_bytes > 0;
    }

    bool resolve_graph_cut_plan(ggml_cgraph* gf,
                                GraphCutPlan* plan_out) {
        GGML_ASSERT(plan_out != nullptr);
        GGML_ASSERT(gf != nullptr);
        int planner_session_count = 1;
        const bool multi_session_enabled = sd_env_flag_enabled("SD_HTP_MULTI_SESSION");
        if (multi_session_enabled && htp_backend != nullptr) {
            const int actual_session_count = ggml_backend_htp_session_count(htp_backend);
            if (actual_session_count > 0) {
                planner_session_count = actual_session_count;
            }
        }
        const bool shared_param_backing = sd_env_flag_enabled("SD_HTP_PACKED_STAGING");
        *plan_out = sd::ggml_graph_cut::resolve_plan(runtime_backend,
                                                     gf,
                                                     &graph_cut_plan_cache_,
                                                     max_graph_vram_bytes,
                                                     params_tensor_set_,
                                                     get_desc().c_str(),
                                                     planner_session_count,
                                                     shared_param_backing);
        if (multi_session_enabled) {
            LOG_INFO("%s graph-cut planner sessions=%d shared_params=%d segments=%zu max_vram=%zu",
                     get_desc().c_str(), planner_session_count,
                     shared_param_backing ? 1 : 0,
                     plan_out->segments.size(), max_graph_vram_bytes);
        }
        return true;
    }

    void reset_segment_runtime_tensors(const GraphCutSegment& segment,
                                       ggml_cgraph* gf) {
        GGML_ASSERT(gf != nullptr);

        for (const auto& input : segment.input_refs) {
            ggml_tensor* input_tensor = sd::ggml_graph_cut::input_tensor(gf, input);
            if (input_tensor == nullptr) {
                continue;
            }
            switch (input.type) {
                case GraphCutSegment::INPUT_PREVIOUS_CUT:
                case GraphCutSegment::INPUT_EXTERNAL:
                    input_tensor->buffer = nullptr;
                    input_tensor->data   = nullptr;
                    input_tensor->extra  = nullptr;
                    break;
                case GraphCutSegment::INPUT_PARAM:
                    break;
            }
        }

        for (int node_idx : segment.internal_node_indices) {
            ggml_tensor* node = ggml_graph_node(gf, node_idx);
            if (node == nullptr) {
                continue;
            }
            node->buffer = nullptr;
            node->data   = nullptr;
            node->extra  = nullptr;
        }
    }

    bool bind_segment_cached_inputs(ggml_cgraph* gf,
                                    const GraphCutSegment& segment) {
        GGML_ASSERT(gf != nullptr);
        // Graph-cut has no segment-level owner.  The common HTP dispatcher
        // selects the consuming session per operation and maps this canonical
        // backing tensor in that session on demand.
        for (const auto& input : segment.input_refs) {
            ggml_tensor* input_tensor = sd::ggml_graph_cut::input_tensor(gf, input);
            if (input_tensor == nullptr) {
                continue;
            }
            switch (input.type) {
                case GraphCutSegment::INPUT_PREVIOUS_CUT: {
                    ggml_tensor* cache_tensor = get_cache_tensor_by_name(input.display_name);
                    if (cache_tensor == nullptr) {
                        LOG_ERROR("%s missing graph cut cache tensor: %s",
                                  get_desc().c_str(),
                                  input.display_name.c_str());
                        return false;
                    }
                    ggml_tensor* bound_cache_tensor = cache_tensor;
                    if (input_tensor->view_src != nullptr) {
                        input_tensor->view_src = bound_cache_tensor;
                        input_tensor->buffer   = nullptr;
                        input_tensor->data     = bound_cache_tensor->data == nullptr
                                                     ? nullptr
                                                     : static_cast<void*>(static_cast<char*>(bound_cache_tensor->data) + input_tensor->view_offs);
                        input_tensor->extra    = bound_cache_tensor->extra;
                    } else {
                        input_tensor->buffer = bound_cache_tensor->buffer;
                        input_tensor->data   = bound_cache_tensor->data;
                        input_tensor->extra  = bound_cache_tensor->extra;
                    }
                    for (int src_idx = 0; src_idx < GGML_MAX_SRC; ++src_idx) {
                        input_tensor->src[src_idx] = nullptr;
                    }
                    input_tensor->op = GGML_OP_NONE;
                    break;
                }
                case GraphCutSegment::INPUT_EXTERNAL:
                case GraphCutSegment::INPUT_PARAM:
                    break;
            }
        }
        return true;
    }

    bool htp_packed_staging_enabled() const {
        return sd_env_flag_enabled("SD_HTP_PACKED_STAGING") &&
               htp_backend != nullptr && static_cast<bool>(htp_packed_staging_loader_);
    }

    bool ensure_htp_packed_staging_pool() {
        if (!htp_packed_staging_enabled()) {
            return true;
        }
        if (htp_packed_staging_pool_ != nullptr) {
            if (htp_packed_staging_pool_owner_count_ == htp_packed_staging_owner_count_) {
                return true;
            }
            if (!htp_packed_staging_handles_.empty()) {
                LOG_ERROR("%s cannot change packed staging owners with active slots",
                          get_desc().c_str());
                return false;
            }
            htp_packed_staging_pool_.reset();
            htp_packed_staging_pool_owner_count_ = 0;
        }
        const size_t budget_mb = static_cast<size_t>(std::max(1, sd_env_int_value("SD_HTP_PACKED_STAGING_BUDGET_MB", 512)));
        const size_t slot_mb = static_cast<size_t>(std::max(1, sd_env_int_value("SD_HTP_PACKED_STAGING_SLOT_MB", 256)));
        const size_t budget_bytes = budget_mb * 1024ull * 1024ull;
        const size_t slot_bytes = slot_mb * 1024ull * 1024ull;
        if (slot_bytes > budget_bytes) {
            LOG_ERROR("%s packed staging slot (%zu MiB) exceeds budget (%zu MiB)",
                      get_desc().c_str(), slot_mb, budget_mb);
            return false;
        }
        ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(htp_backend);
        HtpPackedStagingPool::DiagnosticFn diagnostic;
        if (sd_env_flag_enabled("SD_HTP_PACKED_STAGING_DIAGNOSTICS")) {
            diagnostic = [this](const HtpPackedStagingPool::DiagnosticEvent& event) {
                const char* kind = "pool_created";
                switch (event.kind) {
                    case HtpPackedStagingPool::DiagnosticKind::PoolCreated:
                        break;
                    case HtpPackedStagingPool::DiagnosticKind::SlotAcquired:
                        kind = "slot_acquired";
                        break;
                    case HtpPackedStagingPool::DiagnosticKind::TensorPlaced:
                        kind = "tensor_placed";
                        break;
                    case HtpPackedStagingPool::DiagnosticKind::SlotReleased:
                        kind = "slot_released";
                        break;
                }
                LOG_INFO("%s htp_staging event=%s requested=%zu slot=%zu session=%d domain=%d session_allocated=%zu total_allocated=%zu reused=%d tensor=%s",
                         get_desc().c_str(), kind, event.requested_bytes,
                         event.slot_index, event.logical_session_index, event.domain,
                         event.session_allocated_bytes, event.total_allocated_bytes,
                         event.reused ? 1 : 0, event.tensor_name.c_str());
            };
        }
        const auto free_allocation = [](const HtpPackedStagingPool::Allocation& allocation) {
            if (allocation.buffer != nullptr) {
                ggml_backend_buffer_free(allocation.buffer);
            }
        };
        // Keep one persistent arena, but allocate chunks with a stable
        // session/domain affinity.  This target rejects a live 256 MiB fd
        // being mapped in two FastRPC PDs, so a chunk must never contain
        // weights consumed by another session.  The arena remains shared at
        // runner level and chunks are retained until backend teardown.
        if (htp_packed_staging_owner_count_ > 1) {
            htp_packed_staging_pool_ = std::make_unique<HtpPackedStagingPool>(
                budget_bytes,
                slot_bytes,
                htp_packed_staging_owner_count_,
                [backend = htp_backend](size_t bytes, int logical_session_index) {
                    ggml_backend_buffer_t buffer = ggml_backend_htp_alloc_buffer_for_session(
                        backend, logical_session_index, bytes);
                    if (buffer == nullptr) {
                        return HtpPackedStagingPool::Allocation{};
                    }
                    ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
                    ggml_backend_htp_buffer_set_persistent_staging(buffer, true);
                    return HtpPackedStagingPool::Allocation{
                        ggml_backend_buffer_get_base(buffer), buffer,
                        logical_session_index,
                        ggml_backend_htp_session_domain(backend, logical_session_index)};
                },
                free_allocation,
                std::move(diagnostic));
        } else {
            htp_packed_staging_pool_ = std::make_unique<HtpPackedStagingPool>(
                budget_bytes,
                slot_bytes,
                [buft, backend = htp_backend](size_t bytes) {
                    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, bytes);
                    if (buffer == nullptr) {
                        return HtpPackedStagingPool::Allocation{};
                    }
                    ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
                    ggml_backend_htp_buffer_set_persistent_staging(buffer, true);
                    return HtpPackedStagingPool::Allocation{
                        ggml_backend_buffer_get_base(buffer), buffer, 0,
                        ggml_backend_htp_session_domain(backend, 0)};
                },
                free_allocation,
                std::move(diagnostic));
        }
        htp_packed_staging_pool_owner_count_ = htp_packed_staging_owner_count_;
        LOG_INFO("%s HTP packed staging enabled: budget=%zu MiB slot=%zu MiB owners=%d",
                 get_desc().c_str(), budget_mb, slot_mb,
                 htp_packed_staging_pool_owner_count_);
        return true;
    }

    bool stage_htp_packed_weights(ggml_cgraph* gf) {
        if (!htp_packed_staging_enabled()) {
            return true;
        }
        if (!ensure_htp_packed_staging_pool() || gf == nullptr) {
            return false;
        }
        std::vector<ggml_tensor*> tensors = collect_htp_packable_f16_weights(gf);
        const auto validate_deferred_qkv_staging = [&]() {
            for (int node_index = 0; node_index < ggml_graph_n_nodes(gf); ++node_index) {
                ggml_tensor* node = ggml_graph_node(gf, node_index);
                if (!is_htp_supported_packed_qkv(node)) {
                    continue;
                }
                for (int src_index : htp_packed_qkv_staging_src_indices(node->name)) {
                    ggml_tensor* param = get_htp_tensor_backing_root(node->src[src_index]);
                    if (param != nullptr &&
                        (((param->flags & SD_GGML_TENSOR_FLAG_HTP_DEFERRED_PACKED_PARAM) != 0) ||
                         should_place_param_in_htp_rpcmem(param)) &&
                        !htp_packed_staging_contains(htp_packed_staging_tensors_, param)) {
                        LOG_ERROR("%s HTP QKV parameter is not staged: node=%s src[%d] tensor=%s bytes=%zu",
                                  get_desc().c_str(), node->name, src_index,
                                  param->name, ggml_nbytes(param));
                        return false;
                    }
                }
            }
            return true;
        };
        if (tensors.empty()) {
            return validate_deferred_qkv_staging();
        }
        std::unordered_set<ggml_tensor*> staged_lookup(tensors.begin(), tensors.end());
        std::unordered_map<ggml_tensor*, int> tensor_sessions;
        const int owner_count = std::max(1, htp_packed_staging_owner_count_);
        int htp_node_count = 0;
        for (int node_index = 0; node_index < ggml_graph_n_nodes(gf); ++node_index) {
            ggml_tensor* node = ggml_graph_node(gf, node_index);
            if (node != nullptr && (is_htp_supported_f16_mul_mat(node) ||
                                    is_htp_supported_packed_qkv(node))) {
                ++htp_node_count;
            }
        }
        int htp_ordinal = 0;
        for (int node_index = 0; node_index < ggml_graph_n_nodes(gf); ++node_index) {
            ggml_tensor* node = ggml_graph_node(gf, node_index);
            if (node == nullptr || (!is_htp_supported_f16_mul_mat(node) &&
                                    !is_htp_supported_packed_qkv(node))) {
                continue;
            }
            const int owner = htp_session_owner_for_ordinal(
                htp_ordinal++, htp_node_count, owner_count);
            std::vector<int> weight_src_indices;
            if (is_htp_supported_f16_mul_mat(node)) {
                weight_src_indices.push_back(0);
            } else {
                weight_src_indices = htp_packed_qkv_staging_src_indices(node->name);
            }
            for (int src_index : weight_src_indices) {
                ggml_tensor* src = node->src[src_index];
                while (src != nullptr && src->view_src != nullptr) {
                    src = src->view_src;
                }
                if (src != nullptr) {
                    if (staged_lookup.count(src) != 0) {
                        tensor_sessions.emplace(src, owner);
                    } else {
                        ggml_tensor * packed_src = ggml_backend_htp_get_packed_f16_weight(src);
                        if (packed_src != nullptr && staged_lookup.count(packed_src) != 0) {
                            tensor_sessions.emplace(packed_src, owner);
                        }
                    }
                }
            }
        }
        if (!htp_packed_staging_handles_.empty()) {
            // The graph-cut executor may already own the canonical staged
            // slots from the complete graph. Rebind the tensors after a
            // per-segment parameter swap without reloading their bytes.
            for (ggml_tensor* tensor : tensors) {
                bool rebound = false;
                for (auto handle : htp_packed_staging_handles_) {
                    if (htp_packed_staging_pool_->rebind_tensor(handle, tensor)) {
                        rebound = true;
                        break;
                    }
                }
                if (!rebound) {
                    return false;
                }
            }
            return validate_deferred_qkv_staging();
        }
        size_t required_bytes = 0;
        for (ggml_tensor* tensor : tensors) {
            required_bytes = GGML_PAD(required_bytes, 128) + ggml_nbytes(tensor);
        }
        htp_packed_staging_handles_.clear();
        htp_packed_staging_tensors_.clear();
        for (ggml_tensor* tensor : tensors) {
            const auto owner_it = tensor_sessions.find(tensor);
            const int owner = owner_it != tensor_sessions.end() ? owner_it->second : 0;
            bool placed = false;
            for (auto handle : htp_packed_staging_handles_) {
                if (htp_packed_staging_pool_->session_for_handle(handle) != owner) {
                    continue;
                }
                if (htp_packed_staging_pool_->place_tensor(
                        htp_packed_staging_loader_, handle, tensor->name, tensor)) {
                    placed = true;
                    break;
                }
            }
            if (!placed) {
                HtpPackedStagingPool::Handle handle;
                if (!htp_packed_staging_pool_->acquire_for_session(
                        ggml_nbytes(tensor), owner, &handle) ||
                    !htp_packed_staging_pool_->place_tensor(
                        htp_packed_staging_loader_, handle, tensor->name, tensor)) {
                    if (handle.valid()) {
                        htp_packed_staging_pool_->release(&handle);
                    }
                    LOG_ERROR("%s packed staging pool exhausted while placing '%s' (segment=%zu bytes, budget=%zu bytes)",
                              get_desc().c_str(), tensor->name, required_bytes,
                              htp_packed_staging_pool_->budget_bytes());
                    release_htp_packed_staging();
                    // Diagnostic-only escape hatch: graph-cut/session routing
                    // can still be exercised with the original packed weights.
                    // Keep the default behavior strict so production callers
                    // never silently lose staging protection.
                    if (sd_env_flag_enabled("SD_HTP_PACKED_STAGING_ALLOW_FALLBACK")) {
                        LOG_INFO("%s packed staging fallback: continuing without staged weights",
                                 get_desc().c_str());
                        return true;
                    }
                    return false;
                }
                htp_packed_staging_handles_.push_back(handle);
            }
            if (!placed && htp_packed_staging_handles_.empty()) {
                LOG_ERROR("%s failed to stage packed tensor '%s'", get_desc().c_str(), tensor->name);
                release_htp_packed_staging();
                return false;
            }
            htp_packed_staging_tensors_.push_back(tensor);
        }
        if (!validate_deferred_qkv_staging()) {
            release_htp_packed_staging();
            return false;
        }
        const size_t reclaimed_slots = htp_packed_staging_pool_->reclaim_unleased();
        if (reclaimed_slots > 0) {
            LOG_INFO("%s reclaimed %zu unused HTP packed staging slots",
                     get_desc().c_str(), reclaimed_slots);
        }
        std::string owner_bytes;
        for (int session = 0; session < htp_packed_staging_pool_owner_count_; ++session) {
            if (!owner_bytes.empty()) {
                owner_bytes += ",";
            }
            owner_bytes += std::to_string(session) + ":" +
                           std::to_string(htp_packed_staging_pool_->session_allocated_bytes(session));
        }
        LOG_INFO("%s staged packed HTP weights: %zu tensors, %zu bytes, live_slots=%zu owner_bytes=[%s]",
                 get_desc().c_str(), htp_packed_staging_tensors_.size(), required_bytes,
                 htp_packed_staging_pool_->active_slots(), owner_bytes.c_str());
        return true;
    }

    void release_htp_packed_staging() {
        if (htp_packed_staging_pool_ == nullptr) {
            return;
        }
        for (auto handle : htp_packed_staging_handles_) {
            for (auto it = htp_packed_staging_tensors_.rbegin();
                 it != htp_packed_staging_tensors_.rend(); ++it) {
                htp_packed_staging_pool_->unbind_tensor(handle, *it);
            }
            htp_packed_staging_pool_->release(&handle);
        }
        htp_packed_staging_handles_.clear();
        htp_packed_staging_tensors_.clear();
    }

    template <typename T>
    std::optional<sd::Tensor<T>> execute_graph(ggml_cgraph* gf,
                                               int n_threads,
                                               bool free_compute_buffer_immediately,
                                               const std::vector<ggml_tensor*>& runtime_param_tensors,
                                               bool preserve_backend_tensor_data_map,
                                               bool no_return                                          = false,
                                               const std::unordered_set<std::string>* cache_keep_names = nullptr) {
        int64_t t_execute_begin              = ggml_time_ms();
        const bool use_htp_compute           = htp_op_routing_enabled() &&
                                              sd_env_flag_enabled("SD_HTP_COMPUTE");
        const bool use_partial_param_offload = !runtime_param_tensors.empty();
        const bool use_packed_staging = use_htp_compute && htp_packed_staging_enabled();
        const bool preserve_staging_before_call = preserve_htp_packed_staging_;
        auto staging_guard_fn = [this, use_packed_staging, preserve_staging_before_call](int*) {
            if (use_packed_staging) {
                release_htp_packed_staging();
                preserve_htp_packed_staging_ = preserve_staging_before_call;
            } else if (!preserve_htp_packed_staging_) {
                release_htp_packed_staging();
            }
        };
        int staging_guard_token = 0;
        std::unique_ptr<int, decltype(staging_guard_fn)> staging_guard(
            &staging_guard_token, staging_guard_fn);

        // Stage the packed weights before parameter offload.  offload_* uses
        // htp_packed_staging_tensors_ to exclude these tensors; doing this in
        // the opposite order duplicates the complete packed model in HTP
        // before the segment-sized staging allocation is attempted.
        if (use_packed_staging) {
            preserve_htp_packed_staging_ = true;
            if (!stage_htp_packed_weights(gf)) {
                return std::nullopt;
            }
        }

        int64_t t_offload_begin              = ggml_time_ms();
        if (use_partial_param_offload) {
            if (!offload_partial_params(runtime_param_tensors)) {
                LOG_ERROR("%s offload partial params to runtime backend failed", get_desc().c_str());
                return std::nullopt;
            }
        } else {
            if (!offload_all_params()) {
                LOG_ERROR("%s offload params to runtime backend failed", get_desc().c_str());
                return std::nullopt;
            }
        }
        int64_t t_offload_end = ggml_time_ms();

        if (use_htp_compute && !prepare_htp_packed_f16_weights(gf)) {
            LOG_ERROR("%s prepare HTP packed F16 weights failed", get_desc().c_str());
            if (use_partial_param_offload) {
                restore_partial_params();
            }
            return std::nullopt;
        }
        if (use_htp_compute && !validate_htp_packed_only_consumers(gf)) {
            if (use_partial_param_offload) {
                restore_partial_params();
            }
            return std::nullopt;
        }

        AsyncResidualCacheOutputTransaction cache_output_transaction;
        auto cache_output_rollback = std::unique_ptr<
            AsyncResidualCacheOutputTransaction,
            std::function<void(AsyncResidualCacheOutputTransaction*)>>(
                &cache_output_transaction,
                [this](AsyncResidualCacheOutputTransaction* transaction) {
                    rollback_async_residual_cache_outputs(transaction);
                });
        CacheInputBindingTransaction cache_input_transaction;
        auto cache_input_rollback = std::unique_ptr<
            CacheInputBindingTransaction,
            std::function<void(CacheInputBindingTransaction*)>>(
                &cache_input_transaction,
                [this](CacheInputBindingTransaction* transaction) {
                    rollback_cache_input_bindings(transaction);
                });
        const bool bind_cache_outputs_before_reserve =
            sd_htp_opencl_bind_cache_outputs_before_reserve();
        if (bind_cache_outputs_before_reserve &&
            !prepare_async_residual_cache_outputs(
                &cache_output_transaction, gf)) {
            LOG_ERROR("%s prepare async residual cache outputs failed",
                      get_desc().c_str());
            if (use_partial_param_offload) {
                restore_partial_params();
            }
            return std::nullopt;
        }

        int64_t t_alloc_begin = ggml_time_ms();
        if (!allocate_compute_graph_with_cache_inputs(
                gf, &cache_input_transaction)) {
            LOG_ERROR("%s alloc compute graph failed", get_desc().c_str());
            if (free_compute_buffer_immediately) {
                free_compute_buffer();
            } else if (use_partial_param_offload) {
                restore_partial_params();
            }
            return std::nullopt;
        }
        trace_residual_pair_allocation("post_gallocr", gf);
        int64_t t_alloc_end = ggml_time_ms();

        int64_t t_copy_begin = ggml_time_ms();
        if (!bind_cache_outputs_before_reserve &&
            !prepare_async_residual_cache_outputs(
                &cache_output_transaction, gf)) {
            LOG_ERROR("%s prepare async residual cache outputs failed", get_desc().c_str());
            if (free_compute_buffer_immediately) {
                free_compute_buffer();
            } else if (use_partial_param_offload) {
                restore_partial_params();
            }
            return std::nullopt;
        }
        copy_data_to_backend_tensor(gf, !preserve_backend_tensor_data_map);
        trace_residual_pair_allocation("pre_htp_compute", gf);
        int64_t t_copy_end = ggml_time_ms();
        if (ggml_backend_is_cpu(runtime_backend)) {
            ggml_backend_cpu_set_n_threads(runtime_backend, n_threads);
        }

        int64_t t_compute_begin = ggml_time_ms();
        if (use_htp_compute) {
            // 只有显式打开 SD_HTP_COMPUTE 时才让整张图走 HTP hybrid，未支持算子仍回退到 CPU。
            sd_backend_htp_set_n_threads(htp_backend, n_threads);
            sd_backend_htp_set_hybrid_enabled(true);
        }
        uint64_t arena_generation = 0;
        if (use_htp_compute &&
            sd_htp_opencl_whole_arena_deferred_ownership()) {
            arena_generation = next_compute_arena_generation_++;
            active_compute_arena_generation_ = arena_generation;
            sd_backend_htp_opencl_set_compute_arena_generation(
                htp_backend, arena_generation);
        }
        ggml_status status      = ggml_backend_graph_compute(use_htp_compute ? htp_backend : runtime_backend, gf);
        finalize_cache_input_bindings(&cache_input_transaction, status);
        if (status == GGML_STATUS_SUCCESS) {
            cache_input_rollback.release();
        }
        if (use_htp_compute) {
            sd_backend_htp_set_hybrid_enabled(false);
        }
        const bool arena_pending = arena_generation != 0 &&
            sd_backend_htp_opencl_arena_generation_pending(
                htp_backend, arena_generation);
        if (sd_env_flag_enabled("SD_HTP_QKV_RELEASE_TRACE")) {
            LOG_INFO(
                "qkv_arena_state generation=%" PRIu64 " pending=%d",
                arena_generation, arena_pending ? 1 : 0);
        }
        if (arena_pending) {
            if (!retire_current_compute_arena(arena_generation)) {
                status = GGML_STATUS_FAILED;
            }
        }
        active_compute_arena_generation_ = 0;
        if (!preserve_htp_packed_staging_) {
            release_htp_packed_staging();
        }
        int64_t t_compute_end   = ggml_time_ms();
        if (status != GGML_STATUS_SUCCESS) {
            LOG_ERROR("%s compute failed: %s", get_desc().c_str(), ggml_status_to_string(status));
            if (free_compute_buffer_immediately) {
                free_compute_buffer();
            } else if (use_partial_param_offload) {
                restore_partial_params();
            }
            return std::nullopt;
        }
        commit_async_residual_cache_outputs(&cache_output_transaction);
        cache_output_rollback.release();

        if (ggml_htp_opencl_can_release_transient_buffers_post_compute(
                sd_htp_opencl_lifetime_mode(),
                use_htp_compute,
                free_compute_buffer_immediately,
                all_cache_outputs_async_residual_backed(),
                !debug_tensors.empty())) {
            if (!ggml_gallocr_free_buffer(compute_allocr, 2, nullptr) ||
                !ggml_gallocr_free_buffer(compute_allocr, 1, nullptr)) {
                LOG_ERROR("%s release post-compute HTP buffers failed",
                          get_desc().c_str());
                free_compute_buffer();
                return std::nullopt;
            }
        }

        std::unordered_set<const ggml_tensor*> debug_graph_tensor_set;
        const int n_debug_leafs = sd::ggml_graph_cut::leaf_count(gf);
        const int n_debug_nodes = ggml_graph_n_nodes(gf);
        debug_graph_tensor_set.reserve(static_cast<size_t>(n_debug_leafs + n_debug_nodes));
        for (int i = 0; i < n_debug_leafs; ++i) {
            debug_graph_tensor_set.insert(sd::ggml_graph_cut::leaf_tensor(gf, i));
        }
        for (int i = 0; i < n_debug_nodes; ++i) {
            debug_graph_tensor_set.insert(ggml_graph_node(gf, i));
        }

        for (const auto& entry : debug_tensors) {
            auto tensor = entry.first;
            if (tensor == nullptr) {
                continue;
            }
            if (debug_graph_tensor_set.find(tensor) == debug_graph_tensor_set.end()) {
                continue;
            }
            ggml_backend_buffer_t tensor_buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
            if (tensor_buf == nullptr) {
                LOG_WARN("%s skip debug tensor '%s': tensor buffer not set",
                         get_desc().c_str(),
                         entry.second.c_str());
                continue;
            }
            if (tensor->type != GGML_TYPE_F32) {
                LOG_WARN("%s skip debug tensor '%s': only GGML_TYPE_F32 is supported, got %s",
                         get_desc().c_str(),
                         entry.second.c_str(),
                         ggml_type_name(tensor->type));
                continue;
            }
            auto debug_tensor = sd::make_sd_tensor_from_ggml<float>(tensor);
            print_sd_tensor(debug_tensor, false, entry.second.c_str());
        }

        bool trace_residual_cache_copy = false;
        if (sd_env_flag_enabled("SD_HTP_SPARSE_OPENCL_TRACE")) {
            for (const auto& entry : cache_tensor_map) {
                if (entry.first.rfind("htp_block_sparse_residual_", 0) == 0 &&
                    !async_residual_cache_backed(entry.first)) {
                    trace_residual_cache_copy = true;
                    break;
                }
            }
        }
        if (trace_residual_cache_copy) {
            std::fprintf(stderr,
                         "htp_residual_cache_copy_begin ts_us=%lld\n",
                         (long long) ggml_time_us());
        }
        int64_t t_cache_begin = ggml_time_ms();
        if (!copy_cache_tensors_to_cache_buffer(cache_keep_names, gf)) {
            if (free_compute_buffer_immediately) {
                free_compute_buffer();
            } else if (use_partial_param_offload) {
                restore_partial_params();
            }
            return std::nullopt;
        }
        if (trace_residual_cache_copy) {
            std::fprintf(stderr,
                         "htp_residual_cache_copy_end ts_us=%lld\n",
                         (long long) ggml_time_us());
        }
        int64_t t_cache_end = ggml_time_ms();
        auto result         = ggml_get_tensor(compute_ctx, final_result_name.c_str());
        std::optional<sd::Tensor<T>> output;
        if (!no_return) {
            output = read_graph_tensor<T>(result, "output");
            if (!output.has_value()) {
                if (free_compute_buffer_immediately) {
                    free_compute_buffer();
                } else if (use_partial_param_offload) {
                    restore_partial_params();
                }
                return std::nullopt;
            }
        } else {
            output = sd::Tensor<T>();
        }

        if (free_compute_buffer_immediately) {
            free_compute_buffer();
        } else if (use_partial_param_offload) {
            restore_partial_params();
        }
        if (use_partial_param_offload) {
            LOG_DEBUG("%s execute_graph timing: offload=%lld ms alloc=%lld ms copy_in=%lld ms compute=%lld ms cache=%lld ms total=%lld ms",
                      get_desc().c_str(),
                      t_offload_end - t_offload_begin,
                      t_alloc_end - t_alloc_begin,
                      t_copy_end - t_copy_begin,
                      t_compute_end - t_compute_begin,
                      t_cache_end - t_cache_begin,
                      ggml_time_ms() - t_execute_begin);
        }
        return output;
    }

    template <typename T>
    std::optional<sd::Tensor<T>> compute_with_graph_cuts(ggml_cgraph* gf,
                                                         const GraphCutPlan& plan,
                                                         int n_threads,
                                                         bool free_compute_buffer_immediately,
                                                         bool no_return = false) {
        GGML_ASSERT(gf != nullptr);

        GGML_ASSERT(active_async_residual_cache_generation_ == 0);
        struct AsyncResidualCacheGenerationScope {
            uint64_t* active_generation;
            AsyncResidualCacheGenerationScope(
                    uint64_t* active_generation,
                    uint64_t generation)
                : active_generation(active_generation) {
                *this->active_generation = generation;
            }
            ~AsyncResidualCacheGenerationScope() {
                *active_generation = 0;
            }
        } async_cache_generation_scope(
            &active_async_residual_cache_generation_,
            next_async_residual_cache_generation_++);

        struct StagingOwnerCountGuard {
            int* value;
            int previous;
            ~StagingOwnerCountGuard() { *value = previous; }
        } staging_owner_guard{
            &htp_packed_staging_owner_count_, htp_packed_staging_owner_count_};
        const int initialized_session_count = htp_backend != nullptr
            ? ggml_backend_htp_session_count(htp_backend) : 1;
        htp_packed_staging_owner_count_ = ::htp_packed_staging_owner_count(
            true,
            sd_env_flag_enabled("SD_HTP_MULTI_SESSION"),
            initialized_session_count);

        free_compute_buffer();
        free_cache_ctx_and_buffer();

        std::optional<sd::Tensor<T>> output = sd::Tensor<T>();
        // Staging is scoped to one graph view (segment), not the complete
        // graph.  A complete MMDiT graph can exceed the staging budget by an
        // order of magnitude; execute_graph() stages only the weights needed
        // by its segment and releases them after that view completes.  The
        // allocation remains shared by every session dispatch within the
        // segment, so session switches do not reload the same weight.
        preserve_htp_packed_staging_ = false;
        release_htp_packed_staging();
        for (size_t seg_idx = 0; seg_idx < plan.segments.size(); ++seg_idx) {
            int64_t t_segment_begin = ggml_time_ms();
            const auto& segment     = plan.segments[seg_idx];
            auto future_cut_names   = sd::ggml_graph_cut::collect_future_input_names(
                gf, plan, seg_idx);
            for (const auto& entry : cache_tensor_map) {
                if (entry.first.rfind("htp_block_sparse_residual_", 0) == 0) {
                    future_cut_names.insert(entry.first);
                }
            }
            LOG_DEBUG("%s graph cut executing segment %zu/%zu: %s",
                      get_desc().c_str(), seg_idx + 1,
                      plan.segments.size(), segment.group_name.c_str());

            // A segment is only a graph view.  Its HTP nodes are assigned by
            // the same ordinal dispatcher used by full-graph execution; do
            // not select an owner or reacquire a lease for the whole segment.
            // Cached tensors retain one canonical backing allocation and are
            // mapped lazily by whichever session consumes them.
            reset_segment_runtime_tensors(segment, gf);
            if (!bind_segment_cached_inputs(gf, segment)) {
                free_cache_ctx_and_buffer();
                free_compute_buffer();
                free_compute_ctx();
                return std::nullopt;
            }

            const bool is_last_segment = seg_idx + 1 == plan.segments.size();
            if (!is_last_segment) {
                for (size_t output_idx = 0; output_idx < segment.output_node_indices.size(); ++output_idx) {
                    ggml_tensor* output_tensor = sd::ggml_graph_cut::output_tensor(gf, segment, output_idx);
                    const bool is_residual_output = output_tensor != nullptr &&
                        output_tensor->op == GGML_OP_CUSTOM &&
                        std::strncmp(output_tensor->name,
                                     "htp_block_sparse_residual_b", 27) == 0;
                    if (output_tensor != nullptr &&
                        (is_residual_output ||
                         (sd::ggml_graph_cut::is_graph_cut_tensor(output_tensor) &&
                          future_cut_names.find(output_tensor->name) != future_cut_names.end()))) {
                        cache(output_tensor->name, output_tensor);
                    }
                }
            }

            ggml_context* segment_graph_ctx = nullptr;
            ggml_cgraph* segment_graph      = sd::ggml_graph_cut::build_segment_graph(gf, segment, &segment_graph_ctx);
            auto segment_output             = execute_graph<T>(segment_graph,
                                                   n_threads,
                                                   true,
                                                   sd::ggml_graph_cut::runtime_param_tensors(gf, segment, get_desc().c_str()),
                                                   true,
                                                   !is_last_segment || no_return,
                                                   &future_cut_names);
            ggml_free(segment_graph_ctx);
            if (!segment_output.has_value()) {
                free_cache_ctx_and_buffer();
                free_compute_buffer();
                free_compute_ctx();
                return std::nullopt;
            }
            output = std::move(segment_output);
        }

        backend_tensor_data_map.clear();
        free_cache_ctx_and_buffer();
        free_compute_ctx();
        return output;
    }

public:
    virtual std::string get_desc() = 0;

    GGMLRunner(ggml_backend_t backend, ggml_backend_t params_backend)
        : params_backend(params_backend),
          runtime_backend(backend) {
        GGML_ASSERT(runtime_backend != nullptr);
        GGML_ASSERT(params_backend != nullptr);
        alloc_params_ctx();
    }

    virtual ~GGMLRunner() {
        free_compute_buffer();
        reap_retired_compute_arenas(true);
        free_params_buffer();
        free_params_ctx();
        free_compute_ctx();
        free_cache_ctx_and_buffer();
        free_async_residual_cache_slots();
        trace_memory_ledger_teardown();
    }

    virtual GGMLRunnerContext get_context() {
        GGMLRunnerContext runner_ctx;
        runner_ctx.ggml_ctx              = compute_ctx;
        runner_ctx.backend               = runtime_backend;
        runner_ctx.flash_attn_enabled    = flash_attn_enabled;
        runner_ctx.block_sparse_attn_enabled = block_sparse_attn_enabled;
        runner_ctx.htp_attention_enabled  = htp_attention_enabled;
        runner_ctx.conv2d_direct_enabled = conv2d_direct_enabled;
        runner_ctx.circular_x_enabled    = circular_x_enabled;
        runner_ctx.circular_y_enabled    = circular_y_enabled;
        runner_ctx.weight_adapter        = weight_adapter;
        runner_ctx.debug_tensors         = &debug_tensors;
        runner_ctx.get_cache_tensor      = [this](const std::string& name) {
            return this->get_cache_tensor_by_name(name);
        };
        runner_ctx.cache_tensor = [this](const std::string& name, ggml_tensor* tensor) {
            return this->cache(name, tensor);
        };
        runner_ctx.bind_cache_tensor = [this](const std::string& name, ggml_tensor* tensor) {
            if (tensor != nullptr) {
                this->cache_backed_tensor_name_map[tensor] = name;
                AsyncResidualCacheGeneration* generation =
                    this->latest_async_residual_cache_slot(name);
                if (sd_env_flag_enabled("SD_HTP_SPARSE_OPENCL") &&
                    generation != nullptr && generation->slot != nullptr &&
                    sd_backend_htp_async_cache_slot_opencl_import_identity(
                        generation->slot) != 0) {
                    this->cache_backed_tensor_generation_map_[tensor] =
                        generation->generation;
                }
            }
        };
        return runner_ctx;
    }

    void reset_compute_ctx() {
        free_compute_ctx();
        alloc_compute_ctx();
    }

    bool alloc_params_buffer() {
        params_buffer_tensors_.clear();
        size_t num_tensors = ggml_tensor_num(params_ctx);
        if (num_tensors > 0) {
            // ggml_backend_alloc_ctx_tensors fails when all tensors are already allocated
            // (typical for memory-mapped weights). See ggml-alloc.c n_buffers==0 branch.
            // 先按 HTP buffer type 放置可执行的 F16 二维权重，后续普通 params 分配会自动跳过这些已分配 tensor。
            if (!allocate_htp_placed_params()) {
                return false;
            }

            size_t deferred_tensor_count = 0;
            for (ggml_tensor* tensor = ggml_get_first_tensor(params_ctx);
                 tensor != nullptr;
                 tensor = ggml_get_next_tensor(params_ctx, tensor)) {
                tensor->flags &= ~SD_GGML_TENSOR_FLAG_HTP_DEFERRED_PACKED_PARAM;
                if (htp_should_defer_packed_param(
                        htp_op_routing_enabled(),
                        sd_env_flag_enabled("SD_HTP_PACKED_STAGING"),
                        max_graph_vram_bytes,
                        tensor)) {
                    tensor->flags |= SD_GGML_TENSOR_FLAG_HTP_DEFERRED_PACKED_PARAM;
                    ++deferred_tensor_count;
                }
            }
            if (deferred_tensor_count > 0) {
                if (!non_deferred_params_allocation_.allocate(params_ctx, params_backend)) {
                    LOG_ERROR("%s alloc non-deferred params backend buffer failed, num_tensors = %i",
                              get_desc().c_str(), num_tensors);
                    return false;
                }
                params_buffer = nullptr;
                rebuild_params_tensor_set();
                LOG_INFO("%s deferred packed params allocation: logical=%zu bytes host_backed=%zu bytes deferred_packed=%zu bytes deferred_tensors=%zu",
                         get_desc().c_str(),
                         non_deferred_params_allocation_.logical_bytes(),
                         non_deferred_params_allocation_.host_backed_bytes(),
                         non_deferred_params_allocation_.deferred_bytes(),
                         deferred_tensor_count);
                LOG_DEBUG("%s params backend buffer size = % 6.2f MB(%s) (%i tensors)",
                          get_desc().c_str(),
                          non_deferred_params_allocation_.buffer_size() / (1024.f * 1024.f),
                          ggml_backend_is_cpu(params_backend) ? "RAM" : "VRAM",
                          num_tensors);
                return true;
            }
            bool all_have_data = true;
            for (ggml_tensor* t = ggml_get_first_tensor(params_ctx); t != nullptr; t = ggml_get_next_tensor(params_ctx, t)) {
                if (t->data == nullptr) {
                    all_have_data = false;
                    break;
                }
            }
            if (all_have_data) {
                LOG_DEBUG("%s all params already mmap-allocated (no separate buffer needed)", get_desc().c_str());
                params_buffer = nullptr;
                rebuild_params_tensor_set();
                return true;
            }
        } else {
            LOG_DEBUG("%s skipping params allocation (no tensors)", get_desc().c_str());
            return true;
        }
        for (ggml_tensor* t = ggml_get_first_tensor(params_ctx); t != nullptr; t = ggml_get_next_tensor(params_ctx, t)) {
            if (t->data == nullptr) {
                params_buffer_tensors_.push_back(t);
            }
        }
        params_buffer = ggml_backend_alloc_ctx_tensors(params_ctx, params_backend);
        if (params_buffer == nullptr) {
            params_buffer_tensors_.clear();
            LOG_ERROR("%s alloc params backend buffer failed, num_tensors = %i",
                      get_desc().c_str(),
                      num_tensors);
            return false;
        }
        rebuild_params_tensor_set();
        ggml_backend_buffer_set_usage(params_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        size_t params_buffer_size = ggml_backend_buffer_get_size(params_buffer);
        LOG_DEBUG("%s params backend buffer size = % 6.2f MB(%s) (%i tensors)",
                  get_desc().c_str(),
                  params_buffer_size / (1024.f * 1024.f),
                  ggml_backend_is_cpu(params_backend) ? "RAM" : "VRAM",
                  num_tensors);
        return true;
    }

    void free_params_buffer() {
        free_htp_param_placement();
        non_deferred_params_allocation_.clear();
        if (params_buffer != nullptr) {
            ggml_backend_buffer_free(params_buffer);
            params_buffer = nullptr;
        }
        for (ggml_tensor* tensor : params_buffer_tensors_) {
            tensor->buffer = nullptr;
            tensor->data   = nullptr;
            tensor->extra  = nullptr;
        }
        params_buffer_tensors_.clear();
    }

    size_t get_params_buffer_size() {
        size_t size = 0;
        if (params_buffer != nullptr) {
            size += ggml_backend_buffer_get_size(params_buffer);
        }
        if (htp_param_placement_buffer_ != nullptr) {
            size += ggml_backend_buffer_get_size(htp_param_placement_buffer_);
        }
        for (ggml_backend_buffer_t buffer : htp_param_placement_buffers_) {
            if (buffer != nullptr && buffer != htp_param_placement_buffer_) {
                size += ggml_backend_buffer_get_size(buffer);
            }
        }
        size += non_deferred_params_allocation_.buffer_size();
        return size;
    }

    void free_cache_ctx_and_buffer() {
        free_cache_buffer();
        free_cache_ctx();
    }

    void free_compute_buffer() {
        if (!preserve_htp_packed_staging_) {
            release_htp_packed_staging();
        }
        reap_retired_compute_arenas(false);
        if (compute_allocr != nullptr) {
            ggml_gallocr_free(compute_allocr);
            compute_allocr = nullptr;
        }
        compute_node_buffer_ids_.clear();
        compute_leaf_buffer_ids_.clear();
        compute_no_alloc_tensors_.clear();
        restore_partial_params();
        restore_all_params();
    }

    // do copy after alloc graph
    void set_backend_tensor_data(ggml_tensor* tensor, const void* data) {
        backend_tensor_data_map[tensor] = data;
    }

    template <typename T>
    ggml_tensor* make_input(const sd::Tensor<T>& tensor) {
        ggml_tensor* input = sd::make_ggml_tensor(compute_ctx, tensor, false);
        set_backend_tensor_data(input, tensor.data());
        return input;
    }

    template <typename T>
    ggml_tensor* make_optional_input(const sd::Tensor<T>& tensor) {
        if (tensor.empty()) {
            return nullptr;
        }
        return make_input(tensor);
    }

    template <typename T>
    ggml_tensor* make_optional_input(const sd::Tensor<T>* tensor) {
        if (tensor == nullptr) {
            return nullptr;
        }
        return make_input(*tensor);
    }

    ggml_tensor* to_backend(ggml_tensor* tensor) {
        GGML_ASSERT(compute_ctx != nullptr);
        if (tensor == nullptr) {
            return nullptr;
        }
        // it's performing a compute, check if backend isn't cpu
        if (!ggml_backend_is_cpu(runtime_backend) && (tensor->buffer == nullptr || ggml_backend_buffer_is_host(tensor->buffer))) {
            // pass input tensors to gpu memory
            auto backend_tensor = ggml_dup_tensor(compute_ctx, tensor);

            set_backend_tensor_data(backend_tensor, tensor->data);
            return backend_tensor;
        } else {
            return tensor;
        }
    }

    ggml_tensor* cache(const std::string name, ggml_tensor* tensor) {
        if (tensor != nullptr && tensor->view_src != nullptr) {
            tensor = ggml_cont(compute_ctx, tensor);
        }
        cache_tensor_map[name] = tensor;
        // Cache entries are backing allocations, not session-owned segments.
        // The consuming HTP operation creates/uses its session-local mapper
        // view through the common dispatch path.
        return tensor;
    }

    ggml_tensor* get_cache_tensor_by_name(const std::string& name) {
        AsyncResidualCacheGeneration* generation =
            latest_async_residual_cache_slot(name);
        if (generation != nullptr && generation->slot != nullptr) {
            return sd_backend_htp_async_cache_slot_tensor(generation->slot);
        }
        for (auto it = cache_storage_segments_.rbegin();
             it != cache_storage_segments_.rend(); ++it) {
            if (it->ctx == nullptr || it->names.count(name) == 0) {
                continue;
            }
            if (ggml_tensor* tensor = ggml_get_tensor(it->ctx, name.c_str())) {
                return tensor;
            }
        }
        return nullptr;
    }

    template <typename T>
    std::optional<sd::Tensor<T>> compute(get_graph_cb_t get_graph,
                                         int n_threads,
                                         bool free_compute_buffer_immediately,
                                         bool no_return = false) {
        ggml_cgraph* gf = nullptr;
        if (!prepare_compute_graph(get_graph, &gf)) {
            return std::nullopt;
        }
        GGML_ASSERT(gf != nullptr);

        const bool use_htp_compute = htp_op_routing_enabled() &&
                                     sd_env_flag_enabled("SD_HTP_COMPUTE");
        if (use_htp_compute && !prepare_htp_packed_f16_weights(gf)) {
            LOG_ERROR("%s prepare HTP packed F16 weights failed", get_desc().c_str());
            free_compute_ctx();
            return std::nullopt;
        }
        if (use_htp_compute && !validate_htp_packed_only_consumers(gf)) {
            free_compute_ctx();
            return std::nullopt;
        }

        const bool uses_cross_step_cache =
            !cache_tensor_map.empty() || !cache_backed_tensor_name_map.empty();
        // Packed staging is segment-scoped, so its unified opt-in must also
        // allow graph-cut when cross-step cache tensors are present. Multi-
        // session only changes segment ownership and transfer routing.
        const bool allow_packed_staging_graph_cut =
            sd_env_flag_enabled("SD_HTP_PACKED_STAGING");
        if ((!uses_cross_step_cache || allow_packed_staging_graph_cut) &&
            can_attempt_graph_cut_segmented_compute()) {
            if (allow_packed_staging_graph_cut && uses_cross_step_cache) {
                LOG_INFO("%s packed-staging graph-cut override: cache tensors=%zu cache bindings=%zu",
                         get_desc().c_str(), cache_tensor_map.size(),
                         cache_backed_tensor_name_map.size());
            }
            GraphCutPlan plan;
            if (!resolve_graph_cut_plan(gf, &plan)) {
                free_compute_ctx();
                return std::nullopt;
            }
            if (sd_env_flag_enabled("SD_HTP_MULTI_SESSION")) {
                LOG_INFO("%s multi-session graph plan: has_cuts=%d valid=%d segments=%zu max_vram=%zu",
                         get_desc().c_str(), plan.has_cuts ? 1 : 0, plan.valid ? 1 : 0,
                         plan.segments.size(), max_graph_vram_bytes);
            }
            if (should_use_graph_cut_segmented_compute(plan)) {
                return compute_with_graph_cuts<T>(gf,
                                                  plan,
                                                  n_threads,
                                                  free_compute_buffer_immediately,
                                                  no_return);
            }
        }
        return execute_graph<T>(gf,
                                n_threads,
                                free_compute_buffer_immediately,
                                {},
                                false,
                                no_return);
    }

    void set_flash_attention_enabled(bool enabled) {
        flash_attn_enabled = enabled;
    }

    void set_block_sparse_attention_enabled(bool enabled) {
        if (block_sparse_attn_enabled == enabled) {
            return;
        }
        block_sparse_attn_enabled = enabled;
        free_compute_buffer();
        graph_cut_plan_cache_ = sd::ggml_graph_cut::PlanCache{};
    }

    bool is_block_sparse_attention_enabled() const {
        return block_sparse_attn_enabled;
    }

    void set_htp_attention_enabled(bool enabled) {
        htp_attention_enabled = enabled;
    }

    void set_conv2d_direct_enabled(bool enabled) {
        conv2d_direct_enabled = enabled;
    }

    void set_circular_axes(bool circular_x, bool circular_y) {
        circular_x_enabled = circular_x;
        circular_y_enabled = circular_y;
    }

    void set_weight_adapter(const std::shared_ptr<WeightAdapter>& adapter) {
        weight_adapter = adapter;
    }

    // 由上层在模型初始化完成后注入 HTP backend，避免默认路径无意打开 HTP 逻辑。
    void set_htp_backend(ggml_backend_t backend) {
        htp_backend = backend;
    }

    void set_htp_packed_staging_loader(ModelLoader* loader) {
        if (loader == nullptr) {
            htp_packed_staging_loader_ = {};
            return;
        }
        htp_packed_staging_loader_ = [loader](const std::string& name,
                                              void* dst,
                                              size_t bytes,
                                              ggml_type dst_type) {
            return loader->reload_tensor_bytes(name, dst, bytes, dst_type);
        };
    }

    void set_max_graph_vram_bytes(size_t max_vram_bytes) {
        max_graph_vram_bytes = max_vram_bytes;
    }

    ggml_backend_t get_runtime_backend() {
        return runtime_backend;
    }

    ggml_backend_t get_params_backend() {
        return params_backend;
    }
};

class GGMLBlock {
protected:
    typedef std::unordered_map<std::string, ggml_tensor*> ParameterMap;
    typedef std::unordered_map<std::string, std::shared_ptr<GGMLBlock>> GGMLBlockMap;
    GGMLBlockMap blocks;
    ParameterMap params;

    ggml_type get_type(const std::string& name, const String2TensorStorage& tensor_storage_map, ggml_type default_type) {
        ggml_type wtype = default_type;
        auto iter       = tensor_storage_map.find(name);
        if (iter != tensor_storage_map.end()) {
            const TensorStorage& tensor_storage = iter->second;
            if (tensor_storage.expected_type != GGML_TYPE_COUNT) {
                wtype = tensor_storage.expected_type;
            } else {
                wtype = tensor_storage.type;
            }
        }
        return wtype;
    }

    TensorDataLayout get_data_layout(const std::string& name, const String2TensorStorage& tensor_storage_map) {
        auto iter = tensor_storage_map.find(name);
        if (iter == tensor_storage_map.end()) {
            return TensorDataLayout::REFERENCE;
        }
        return iter->second.layout;
    }

    void init_blocks(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") {
        for (auto& pair : blocks) {
            auto& block = pair.second;
            block->init(ctx, tensor_storage_map, prefix + pair.first);
        }
    }

    virtual void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") {}

public:
    ggml_tensor* get_param(const std::string& name) const {
        auto it = params.find(name);
        return it != params.end() ? it->second : nullptr;
    }

    void init(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, std::string prefix = "") {
        if (prefix.size() > 0) {
            prefix = prefix + ".";
        }
        init_params(ctx, tensor_storage_map, prefix);
        // Mark packed QKV's F32 bias and RMSNorm scales before parameter
        // placement, so they are allocated in RPCMEM with the packed weights.
        const bool packed_qkv_param_prefix = prefix.find("attn.qkv.") != std::string::npos ||
                                             prefix.find("attn.ln_q.") != std::string::npos ||
                                             prefix.find("attn.ln_k.") != std::string::npos ||
                                             prefix.find("attn2.qkv.") != std::string::npos ||
                                             prefix.find("attn2.ln_q.") != std::string::npos ||
                                             prefix.find("attn2.ln_k.") != std::string::npos ||
                                             prefix.find("attention.q_norm.") != std::string::npos ||
                                             prefix.find("attention.k_norm.") != std::string::npos ||
                                             prefix.find("norm.query_norm.") != std::string::npos ||
                                             prefix.find("norm.key_norm.") != std::string::npos;
        if (packed_qkv_param_prefix) {
            for (auto& pair : params) {
                if (pair.second != nullptr && pair.second->type == GGML_TYPE_F32) {
                    pair.second->flags |= SD_GGML_TENSOR_FLAG_HTP_PACKED_QKV_PARAM;
                }
            }
        }
        init_blocks(ctx, tensor_storage_map, prefix);
    }

    size_t get_params_num() {
        size_t num_tensors = params.size();
        for (auto& pair : blocks) {
            auto& block = pair.second;

            num_tensors += block->get_params_num();
        }
        return num_tensors;
    };

    size_t get_params_mem_size() {
        size_t mem_size = 0;
        for (auto& pair : blocks) {
            auto& block = pair.second;

            mem_size += block->get_params_mem_size();
        }

        for (auto& pair : params) {
            mem_size += ggml_nbytes(pair.second);
        }

        return mem_size;
    }

    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors, std::string prefix = "") {
        if (prefix.size() > 0) {
            prefix = prefix + ".";
        }
        for (auto& pair : blocks) {
            auto& block = pair.second;
            block->get_param_tensors(tensors, prefix + pair.first);
        }

        for (auto& pair : params) {
            ggml_tensor* param       = pair.second;
            std::string tensor_name  = prefix + pair.first;
            tensors[tensor_name]     = param;
            ggml_set_name(param, tensor_name.c_str());
        }
    }

    virtual std::string get_desc() {
        return "GGMLBlock";
    }

    void get_all_blocks(std::vector<GGMLBlock*>& result) {
        result.push_back(this);
        for (auto& block_iter : blocks) {
            if (block_iter.second) {
                block_iter.second->get_all_blocks(result);
            }
        }
    }
};

class UnaryBlock : public GGMLBlock {
public:
    virtual ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) = 0;
};

class Identity : public UnaryBlock {
public:
    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
        return x;
    }
};

class Linear : public UnaryBlock {
protected:
    int64_t in_features;
    int64_t out_features;
    bool bias;
    bool force_f32;
    bool force_prec_f32;
    float scale;
    std::string prefix;

    void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") override {
        this->prefix         = prefix;
        enum ggml_type wtype = get_type(prefix + "weight", tensor_storage_map, GGML_TYPE_F32);
        if (in_features % ggml_blck_size(wtype) != 0 || force_f32) {
            wtype = GGML_TYPE_F32;
        }
        params["weight"] = ggml_new_tensor_2d(ctx, wtype, in_features, out_features);
        // Linear weight 的唯一前向用途是 MUL_MAT，参数分配阶段可据此选择 HTP RPCMEM buffer type。
        params["weight"]->flags |= SD_GGML_TENSOR_FLAG_HTP_MUL_MAT_WEIGHT;
        if (get_data_layout(prefix + "weight", tensor_storage_map) == TensorDataLayout::HTP_PERMUTED_W16A32) {
            params["weight"]->flags |= SD_GGML_TENSOR_FLAG_HTP_PACKED_W16A32;
        }
        if (bias) {
            enum ggml_type wtype = GGML_TYPE_F32;
            params["bias"]       = ggml_new_tensor_1d(ctx, wtype, out_features);
        }
    }

public:
    Linear(int64_t in_features,
           int64_t out_features,
           bool bias           = true,
           bool force_f32      = false,
           bool force_prec_f32 = false,
           float scale         = 1.f)
        : in_features(in_features),
          out_features(out_features),
          bias(bias),
          force_f32(force_f32),
          force_prec_f32(force_prec_f32),
          scale(scale) {}

    void set_scale(float scale_) {
        scale = scale_;
    }

    void set_force_prec_f32(bool force_prec_f32_) {
        force_prec_f32 = force_prec_f32_;
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
        ggml_tensor* w = params["weight"];
        ggml_tensor* b = nullptr;
        if (bias) {
            b = params["bias"];
        }
        if (ctx->weight_adapter) {
            WeightAdapter::ForwardParams forward_params;
            forward_params.op_type               = WeightAdapter::ForwardParams::op_type_t::OP_LINEAR;
            forward_params.linear.force_prec_f32 = force_prec_f32;
            forward_params.linear.scale          = scale;
            return ctx->weight_adapter->forward(ctx->ggml_ctx, ctx->backend, x, w, b, prefix, forward_params);
        }
        return ggml_ext_linear(ctx->ggml_ctx, x, w, b, force_prec_f32, scale);
    }
};

__STATIC_INLINE__ bool support_get_rows(ggml_type wtype) {
    std::set<ggml_type> allow_types = {GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_Q5_1, GGML_TYPE_Q5_0, GGML_TYPE_Q4_1, GGML_TYPE_Q4_0};
    if (allow_types.find(wtype) != allow_types.end()) {
        return true;
    }
    return false;
}

class Embedding : public UnaryBlock {
protected:
    int64_t embedding_dim;
    int64_t num_embeddings;
    void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map, const std::string prefix = "") override {
        enum ggml_type wtype = get_type(prefix + "weight", tensor_storage_map, GGML_TYPE_F32);
        if (!support_get_rows(wtype)) {
            wtype = GGML_TYPE_F32;
        }
        params["weight"] = ggml_new_tensor_2d(ctx, wtype, embedding_dim, num_embeddings);
    }

public:
    Embedding(int64_t num_embeddings, int64_t embedding_dim)
        : embedding_dim(embedding_dim),
          num_embeddings(num_embeddings) {
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx,
                         ggml_tensor* input_ids) {
        // input_ids: [N, n_token]
        auto weight = params["weight"];

        // There are issues with ggml batch inference, so we are expanding it here first.
        // TODO: fix ggml batch inference
        int64_t n = input_ids->ne[1];
        input_ids = ggml_reshape_1d(ctx->ggml_ctx, input_ids, input_ids->ne[0] * input_ids->ne[1]);

        input_ids      = ggml_reshape_3d(ctx->ggml_ctx, input_ids, input_ids->ne[0], 1, input_ids->ne[1]);
        auto embedding = ggml_get_rows(ctx->ggml_ctx, weight, input_ids);
        embedding      = ggml_reshape_3d(ctx->ggml_ctx, embedding, embedding->ne[0], embedding->ne[1] / n, n);

        // [N, n_token, embedding_dim]
        return embedding;
    }
};

class Conv2d : public UnaryBlock {
protected:
    int64_t in_channels;
    int64_t out_channels;
    std::pair<int, int> kernel_size;
    std::pair<int, int> stride;
    std::pair<int, int> padding;
    std::pair<int, int> dilation;
    bool bias;
    float scale = 1.f;
    std::string prefix;

    void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map, const std::string prefix = "") override {
        this->prefix         = prefix;
        enum ggml_type wtype = GGML_TYPE_F16;
        params["weight"]     = ggml_new_tensor_4d(ctx, wtype, kernel_size.second, kernel_size.first, in_channels, out_channels);
        if (bias) {
            enum ggml_type wtype = GGML_TYPE_F32;
            params["bias"]       = ggml_new_tensor_1d(ctx, wtype, out_channels);
        }
    }

public:
    Conv2d(int64_t in_channels,
           int64_t out_channels,
           std::pair<int, int> kernel_size,
           std::pair<int, int> stride   = {1, 1},
           std::pair<int, int> padding  = {0, 0},
           std::pair<int, int> dilation = {1, 1},
           bool bias                    = true)
        : in_channels(in_channels),
          out_channels(out_channels),
          kernel_size(kernel_size),
          stride(stride),
          padding(padding),
          dilation(dilation),
          bias(bias) {}

    void set_scale(float scale_value) {
        scale = scale_value;
    }

    std::string get_desc() {
        return "Conv2d";
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
        ggml_tensor* w = params["weight"];
        ggml_tensor* b = nullptr;
        if (bias) {
            b = params["bias"];
        }
        if (ctx->weight_adapter) {
            WeightAdapter::ForwardParams forward_params;
            forward_params.op_type           = WeightAdapter::ForwardParams::op_type_t::OP_CONV2D;
            forward_params.conv2d.s0         = stride.second;
            forward_params.conv2d.s1         = stride.first;
            forward_params.conv2d.p0         = padding.second;
            forward_params.conv2d.p1         = padding.first;
            forward_params.conv2d.d0         = dilation.second;
            forward_params.conv2d.d1         = dilation.first;
            forward_params.conv2d.direct     = ctx->conv2d_direct_enabled;
            forward_params.conv2d.circular_x = ctx->circular_x_enabled;
            forward_params.conv2d.circular_y = ctx->circular_y_enabled;
            forward_params.conv2d.scale      = scale;
            return ctx->weight_adapter->forward(ctx->ggml_ctx, ctx->backend, x, w, b, prefix, forward_params);
        }
        return ggml_ext_conv_2d(ctx->ggml_ctx,
                                x,
                                w,
                                b,
                                stride.second,
                                stride.first,
                                padding.second,
                                padding.first,
                                dilation.second,
                                dilation.first,
                                ctx->conv2d_direct_enabled,
                                ctx->circular_x_enabled,
                                ctx->circular_y_enabled,
                                scale);
    }
};

class Conv2d_grouped : public UnaryBlock {
protected:
    int64_t in_channels;
    int64_t out_channels;
    int groups;
    std::pair<int, int> kernel_size;
    std::pair<int, int> stride;
    std::pair<int, int> padding;
    std::pair<int, int> dilation;
    bool bias;
    float scale = 1.f;
    std::string prefix;

    void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map, const std::string prefix = "") override {
        this->prefix         = prefix;
        enum ggml_type wtype = GGML_TYPE_F16;
        params["weight"]     = ggml_new_tensor_4d(ctx, wtype, kernel_size.second, kernel_size.first, in_channels / groups, out_channels);
        if (bias) {
            enum ggml_type wtype = GGML_TYPE_F32;
            params["bias"]       = ggml_new_tensor_1d(ctx, wtype, out_channels);
        }
    }

public:
    Conv2d_grouped(int64_t in_channels,
                   int64_t out_channels,
                   int groups,
                   std::pair<int, int> kernel_size,
                   std::pair<int, int> stride   = {1, 1},
                   std::pair<int, int> padding  = {0, 0},
                   std::pair<int, int> dilation = {1, 1},
                   bool bias                    = true)
        : in_channels(in_channels),
          out_channels(out_channels),
          groups(groups),
          kernel_size(kernel_size),
          stride(stride),
          padding(padding),
          dilation(dilation),
          bias(bias) {}

    void set_scale(float scale_value) {
        scale = scale_value;
    }

    std::string get_desc() {
        return "Conv2d_grouped";
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
        ggml_tensor* w = params["weight"];
        ggml_tensor* b = nullptr;
        if (bias) {
            b = params["bias"];
        }

        if (groups == 1) {
            if (ctx->weight_adapter) {
                WeightAdapter::ForwardParams forward_params;
                forward_params.op_type           = WeightAdapter::ForwardParams::op_type_t::OP_CONV2D;
                forward_params.conv2d.s0         = stride.second;
                forward_params.conv2d.s1         = stride.first;
                forward_params.conv2d.p0         = padding.second;
                forward_params.conv2d.p1         = padding.first;
                forward_params.conv2d.d0         = dilation.second;
                forward_params.conv2d.d1         = dilation.first;
                forward_params.conv2d.direct     = ctx->conv2d_direct_enabled;
                forward_params.conv2d.circular_x = ctx->circular_x_enabled;
                forward_params.conv2d.circular_y = ctx->circular_y_enabled;
                forward_params.conv2d.scale      = scale;
                return ctx->weight_adapter->forward(ctx->ggml_ctx, ctx->backend, x, w, b, prefix, forward_params);
            }
            return ggml_ext_conv_2d(ctx->ggml_ctx, x, w, b,
                                    stride.second, stride.first,
                                    padding.second, padding.first,
                                    dilation.second, dilation.first,
                                    ctx->conv2d_direct_enabled,
                                    ctx->circular_x_enabled,
                                    ctx->circular_y_enabled,
                                    scale);
        }

        if (groups == in_channels && groups == out_channels) {
            ggml_tensor* res;
            if (ctx->conv2d_direct_enabled) {
                res = ggml_conv_2d_dw_direct(ctx->ggml_ctx, x, w,
                                             stride.second, stride.first,
                                             padding.second, padding.first,
                                             dilation.second, dilation.first);
            } else {
                res = ggml_conv_2d_dw(ctx->ggml_ctx, x, w,
                                      stride.second, stride.first,
                                      padding.second, padding.first,
                                      dilation.second, dilation.first);
            }
            if (b) {
                res = ggml_add(ctx->ggml_ctx, res, b);
            }
            return res;
        }

        int64_t ic_g = in_channels / groups;
        int64_t oc_g = out_channels / groups;

        std::vector<ggml_tensor*> out_slices(groups);

        for (int i = 0; i < groups; ++i) {
            size_t x_offset  = i * ic_g * x->nb[2];
            ggml_tensor* x_i = ggml_view_4d(ctx->ggml_ctx, x,
                                            x->ne[0], x->ne[1], ic_g, x->ne[3],
                                            x->nb[1], x->nb[2], x->nb[3],
                                            x_offset);

            size_t w_offset  = i * oc_g * w->nb[3];
            ggml_tensor* w_i = ggml_view_4d(ctx->ggml_ctx, w,
                                            w->ne[0], w->ne[1], w->ne[2], oc_g,
                                            w->nb[1], w->nb[2], w->nb[3],
                                            w_offset);

            ggml_tensor* b_i = nullptr;
            if (b) {
                size_t b_offset = i * oc_g * b->nb[0];
                b_i             = ggml_view_1d(ctx->ggml_ctx, b, oc_g, b_offset);
            }

            if (ctx->weight_adapter) {
                WeightAdapter::ForwardParams forward_params;
                forward_params.op_type           = WeightAdapter::ForwardParams::op_type_t::OP_CONV2D;
                forward_params.conv2d.s0         = stride.second;
                forward_params.conv2d.s1         = stride.first;
                forward_params.conv2d.p0         = padding.second;
                forward_params.conv2d.p1         = padding.first;
                forward_params.conv2d.d0         = dilation.second;
                forward_params.conv2d.d1         = dilation.first;
                forward_params.conv2d.direct     = ctx->conv2d_direct_enabled;
                forward_params.conv2d.circular_x = ctx->circular_x_enabled;
                forward_params.conv2d.circular_y = ctx->circular_y_enabled;
                forward_params.conv2d.scale      = scale;
                out_slices[i]                    = ctx->weight_adapter->forward(ctx->ggml_ctx, ctx->backend, x_i, w_i, b_i, prefix, forward_params);
            } else {
                out_slices[i] = ggml_ext_conv_2d(ctx->ggml_ctx, x_i, w_i, b_i,
                                                 stride.second, stride.first,
                                                 padding.second, padding.first,
                                                 dilation.second, dilation.first,
                                                 ctx->conv2d_direct_enabled,
                                                 ctx->circular_x_enabled,
                                                 ctx->circular_y_enabled,
                                                 scale);
            }
        }

        ggml_tensor* out = ggml_ext_vec_concat(ctx->ggml_ctx, out_slices, 2);

        return out;
    }
};

class Conv3d : public UnaryBlock {
protected:
    int64_t in_channels;
    int64_t out_channels;
    std::tuple<int, int, int> kernel_size;
    std::tuple<int, int, int> stride;
    std::tuple<int, int, int> padding;
    std::tuple<int, int, int> dilation;
    bool bias;
    bool force_prec_f32;
    std::string prefix;

    void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map, const std::string prefix = "") override {
        this->prefix         = prefix;
        enum ggml_type wtype = GGML_TYPE_F16;
        params["weight"]     = ggml_new_tensor_4d(ctx,
                                                  wtype,
                                                  std::get<2>(kernel_size),
                                                  std::get<1>(kernel_size),
                                                  std::get<0>(kernel_size),
                                                  in_channels * out_channels);
        if (bias) {
            params["bias"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_channels);
        }
    }

public:
    Conv3d(int64_t in_channels,
           int64_t out_channels,
           std::tuple<int, int, int> kernel_size,
           std::tuple<int, int, int> stride   = {1, 1, 1},
           std::tuple<int, int, int> padding  = {0, 0, 0},
           std::tuple<int, int, int> dilation = {1, 1, 1},
           bool bias                          = true,
           bool force_prec_f32                = false)
        : in_channels(in_channels),
          out_channels(out_channels),
          kernel_size(kernel_size),
          stride(stride),
          padding(padding),
          dilation(dilation),
          bias(bias),
          force_prec_f32(force_prec_f32) {}

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
        ggml_tensor* w = params["weight"];
        ggml_tensor* b = nullptr;
        if (ctx->weight_adapter) {
            w = ctx->weight_adapter->patch_weight(ctx->ggml_ctx, ctx->backend, w, prefix + "weight");
            if (w->type != GGML_TYPE_F16) {
                w = ggml_cast(ctx->ggml_ctx, w, GGML_TYPE_F16);
            }
        }
        if (bias) {
            b = params["bias"];
            if (ctx->weight_adapter) {
                b = ctx->weight_adapter->patch_weight(ctx->ggml_ctx, ctx->backend, b, prefix + "bias");
            }
        }
        return ggml_ext_conv_3d(ctx->ggml_ctx, x, w, b, in_channels,
                                std::get<2>(stride), std::get<1>(stride), std::get<0>(stride),
                                std::get<2>(padding), std::get<1>(padding), std::get<0>(padding),
                                std::get<2>(dilation), std::get<1>(dilation), std::get<0>(dilation),
                                force_prec_f32);
    }
};

class LayerNorm : public UnaryBlock {
protected:
    int64_t normalized_shape;
    float eps;
    bool elementwise_affine;
    bool bias;
    std::string prefix;

    void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") override {
        this->prefix = prefix;
        if (elementwise_affine) {
            enum ggml_type wtype = GGML_TYPE_F32;
            params["weight"]     = ggml_new_tensor_1d(ctx, wtype, normalized_shape);
            if (bias) {
                enum ggml_type wtype = GGML_TYPE_F32;
                params["bias"]       = ggml_new_tensor_1d(ctx, wtype, normalized_shape);
            }
        }
    }

public:
    LayerNorm(int64_t normalized_shape,
              float eps               = 1e-05f,
              bool elementwise_affine = true,
              bool bias               = true)
        : normalized_shape(normalized_shape),
          eps(eps),
          elementwise_affine(elementwise_affine),
          bias(bias) {}

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
        ggml_tensor* w = nullptr;
        ggml_tensor* b = nullptr;

        if (elementwise_affine) {
            w = params["weight"];
            if (ctx->weight_adapter) {
                w = ctx->weight_adapter->patch_weight(ctx->ggml_ctx, ctx->backend, w, prefix + "weight");
            }
            if (bias) {
                b = params["bias"];
                if (ctx->weight_adapter) {
                    b = ctx->weight_adapter->patch_weight(ctx->ggml_ctx, ctx->backend, b, prefix + "bias");
                }
            }
        }
        return ggml_ext_layer_norm(ctx->ggml_ctx, x, w, b, eps);
    }
};

class GroupNorm : public GGMLBlock {
protected:
    int num_groups;
    int64_t num_channels;
    float eps;
    bool affine;
    std::string prefix;

    void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") override {
        this->prefix = prefix;
        if (affine) {
            enum ggml_type wtype      = GGML_TYPE_F32;
            enum ggml_type bias_wtype = GGML_TYPE_F32;
            params["weight"]          = ggml_new_tensor_1d(ctx, wtype, num_channels);
            params["bias"]            = ggml_new_tensor_1d(ctx, bias_wtype, num_channels);
        }
    }

public:
    GroupNorm(int num_groups,
              int64_t num_channels,
              float eps   = 1e-05f,
              bool affine = true)
        : num_groups(num_groups),
          num_channels(num_channels),
          eps(eps),
          affine(affine) {}

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
        ggml_tensor* w = nullptr;
        ggml_tensor* b = nullptr;
        if (affine) {
            w = params["weight"];
            b = params["bias"];
            if (ctx->weight_adapter) {
                w = ctx->weight_adapter->patch_weight(ctx->ggml_ctx, ctx->backend, w, prefix + "weight");
                b = ctx->weight_adapter->patch_weight(ctx->ggml_ctx, ctx->backend, b, prefix + "bias");
            }
        }
        return ggml_ext_group_norm(ctx->ggml_ctx, x, w, b, num_groups);
    }
};

class GroupNorm32 : public GroupNorm {
public:
    GroupNorm32(int64_t num_channels)
        : GroupNorm(32, num_channels, 1e-06f) {}
};

class RMSNorm : public UnaryBlock {
protected:
    int64_t hidden_size;
    float eps;
    std::string prefix;

    void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, std::string prefix = "") override {
        this->prefix         = prefix;
        enum ggml_type wtype = GGML_TYPE_F32;
        params["weight"]     = ggml_new_tensor_1d(ctx, wtype, hidden_size);
    }

public:
    RMSNorm(int64_t hidden_size,
            float eps = 1e-06f)
        : hidden_size(hidden_size),
          eps(eps) {}

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
        ggml_tensor* w = params["weight"];
        if (ctx->weight_adapter) {
            w = ctx->weight_adapter->patch_weight(ctx->ggml_ctx, ctx->backend, w, prefix + "weight");
        }
        x = ggml_rms_norm(ctx->ggml_ctx, x, eps);
        x = ggml_mul_inplace(ctx->ggml_ctx, x, w);
        return x;
    }
};

class MultiheadAttention : public GGMLBlock {
protected:
    int64_t embed_dim;
    int64_t n_head;
    bool proj_in;
    std::string q_proj_name;
    std::string k_proj_name;
    std::string v_proj_name;
    std::string in_proj_name;
    std::string out_proj_name;

public:
    MultiheadAttention(int64_t embed_dim,
                       int64_t n_head,
                       bool qkv_proj_bias        = true,
                       bool out_proj_bias        = true,
                       bool proj_in              = false,
                       std::string q_proj_name   = "q_proj",
                       std::string k_proj_name   = "k_proj",
                       std::string v_proj_name   = "v_proj",
                       std::string in_proj_name  = "in_proj",
                       std::string out_proj_name = "out_proj")
        : embed_dim(embed_dim),
          n_head(n_head),
          proj_in(proj_in),
          q_proj_name(q_proj_name),
          k_proj_name(k_proj_name),
          v_proj_name(v_proj_name),
          in_proj_name(in_proj_name),
          out_proj_name(out_proj_name) {
        if (proj_in) {
            blocks[in_proj_name] = std::shared_ptr<GGMLBlock>(new Linear(embed_dim, embed_dim * 3, qkv_proj_bias));
        } else {
            blocks[q_proj_name] = std::shared_ptr<GGMLBlock>(new Linear(embed_dim, embed_dim, qkv_proj_bias));
            blocks[k_proj_name] = std::shared_ptr<GGMLBlock>(new Linear(embed_dim, embed_dim, qkv_proj_bias));
            blocks[v_proj_name] = std::shared_ptr<GGMLBlock>(new Linear(embed_dim, embed_dim, qkv_proj_bias));
        }
        blocks[out_proj_name] = std::shared_ptr<GGMLBlock>(new Linear(embed_dim, embed_dim, out_proj_bias));
    }

    // x: [N, n_token, embed_dim]
    ggml_tensor* forward(GGMLRunnerContext* ctx,
                         ggml_tensor* x,
                         ggml_tensor* mask = nullptr) {
        auto out_proj = std::dynamic_pointer_cast<Linear>(blocks[out_proj_name]);

        ggml_tensor* q;
        ggml_tensor* k;
        ggml_tensor* v;
        if (proj_in) {
            auto in_proj = std::dynamic_pointer_cast<Linear>(blocks[in_proj_name]);
            auto qkv     = in_proj->forward(ctx, x);
            auto qkv_vec = split_qkv(ctx->ggml_ctx, qkv);
            q            = qkv_vec[0];
            k            = qkv_vec[1];
            v            = qkv_vec[2];
        } else {
            auto q_proj = std::dynamic_pointer_cast<Linear>(blocks[q_proj_name]);
            auto k_proj = std::dynamic_pointer_cast<Linear>(blocks[k_proj_name]);
            auto v_proj = std::dynamic_pointer_cast<Linear>(blocks[v_proj_name]);

            q = q_proj->forward(ctx, x);
            k = k_proj->forward(ctx, x);
            v = v_proj->forward(ctx, x);
        }

        x = ggml_ext_attention_ext(ctx->ggml_ctx, ctx->backend, q, k, v, n_head, mask, false);  // [N, n_token, embed_dim]

        x = out_proj->forward(ctx, x);  // [N, n_token, embed_dim]
        return x;
    }
};

#endif  // __GGML_EXTEND__HPP__
