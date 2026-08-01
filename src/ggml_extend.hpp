#ifndef __GGML_EXTEND_HPP__
#define __GGML_EXTEND_HPP__

#include <assert.h>
#include <inttypes.h>
#include <stdarg.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
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

#include "htp_tensor_flags.h"
#include "model.h"
#include "tensor.hpp"

#include "rng.hpp"
#include "tensor_ggml.hpp"
#include "util.h"

#define EPS 1e-05f

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

__STATIC_INLINE__ bool sd_htp_patch_major_layout_enabled(bool sparse_attn_enabled,
                                                          int residual_mode) {
    return sparse_attn_enabled || residual_mode == SD_HTP_BLOCK_SPARSE_RESIDUAL_ANCHOR;
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

__STATIC_INLINE__ ggml_tensor* ggml_ext_merge_lora(ggml_context* ctx,
                                                   ggml_tensor* lora_down,
                                                   ggml_tensor* lora_up,
                                                   ggml_tensor* lora_mid = nullptr) {
    ggml_tensor* updown;
    // flat lora tensors to multiply it
    int64_t lora_up_rows  = lora_up->ne[ggml_n_dims(lora_up) - 1];
    lora_up               = ggml_reshape_2d(ctx, lora_up, ggml_nelements(lora_up) / lora_up_rows, lora_up_rows);
    auto lora_down_n_dims = ggml_n_dims(lora_down);
    // assume n_dims should always be a multiple of 2 (otherwise rank 1 doesn't work)
    lora_down_n_dims       = (lora_down_n_dims + lora_down_n_dims % 2);
    int64_t lora_down_rows = lora_down->ne[lora_down_n_dims - 1];
    lora_down              = ggml_reshape_2d(ctx, lora_down, ggml_nelements(lora_down) / lora_down_rows, lora_down_rows);

    // ggml_mul_mat requires tensor b transposed
    lora_down = ggml_cont(ctx, ggml_transpose(ctx, lora_down));
    if (lora_mid == nullptr) {
        updown = ggml_mul_mat(ctx, lora_up, lora_down);
        updown = ggml_cont(ctx, ggml_transpose(ctx, updown));
    } else {
        // undoing tucker decomposition for conv layers.
        // lora_mid  has shape (3,    3,   Rank, Rank)
        // lora_down has shape (Rank, In,  1,    1)
        // lora_up   has shape (Rank, Out, 1,    1)
        // conv layer shape is (3,    3,   Out,  In)
        updown = ggml_ext_mul_n_mode(ctx, ggml_ext_mul_n_mode(ctx, lora_mid, lora_down, 3), lora_up, 2);
        updown = ggml_cont(ctx, updown);
    }
    return updown;
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
                rho->type == GGML_TYPE_F32);
    const int64_t head_dim = dense->ne[0] - 1;
    const int64_t rows = dense->ne[1] * dense->ne[2] * dense->ne[3];
    for (int64_t row = ith; row < rows; row += nth) {
        const float scale = static_cast<const float*>(rho->data)[row];
        for (int64_t d = 0; d < head_dim; ++d) {
            static_cast<float*>(dst->data)[row * head_dim + d] =
                static_cast<const float*>(dense->data)[row * dense->ne[0] + d] -
                static_cast<const float*>(sparse->data)[row * sparse->ne[0] + d] * scale;
        }
    }
}

__STATIC_INLINE__ bool sd_htp_opencl_residual_route_prerequisites() {
    return sd_env_flag_enabled("SD_HTP_SPARSE_OPENCL") &&
           sd_env_flag_enabled("SD_HTP_SPARSE_OPENCL_ASYNC") &&
           sd_env_flag_enabled("SD_HTP_SPARSE_OPENCL_EARLY_SUBMIT") &&
           sd_env_flag_enabled("SD_HTP_BLOCK_SPARSE_RESIDUAL_CACHE") &&
           sd_env_flag_enabled("SD_HTP_BLOCK_SPARSE_RESIDUAL_EXACT_RHO");
}

__STATIC_INLINE__ bool sd_htp_opencl_deferred_residual_enabled() {
    return sd_env_flag_enabled("SD_HTP_SPARSE_OPENCL_RESIDUAL") &&
           sd_env_flag_enabled_default(
               "SD_HTP_SPARSE_OPENCL_DEFER_RESIDUAL", true);
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
                                                      const std::function<void(const std::string&, ggml_tensor*)>* block_sparse_cache_tensor = nullptr,
                                                      int block_sparse_token_h = 0,
                                                      int block_sparse_token_w = 0) {  // avoid overflow
    auto packed_qkv_storage = [](ggml_tensor* tensor) -> ggml_tensor* {
        for (int depth = 0; tensor != nullptr && depth < 8; ++depth) {
            if (tensor->name[0] != '\0' && std::strcmp(tensor->name, "qkv_mul_rms_packed_dual") == 0) {
                return tensor;
            }
            tensor = tensor->view_src;
        }
        return nullptr;
    };
    ggml_tensor* packed_qkv = packed_qkv_storage(q);
    const bool packed_layout = packed_qkv != nullptr && packed_qkv_storage(k) == packed_qkv &&
                               packed_qkv_storage(v) == packed_qkv &&
                               sd_env_flag_enabled_default("SD_HTP_PACKED_ATTENTION", true);
    int64_t L_q;
    int64_t L_k;
    int64_t C;
    int64_t N;
    int64_t d_head;
    int64_t n_kv_head;
    ggml_tensor* q_htp_flash = nullptr;
    ggml_tensor* k_htp_flash_base = nullptr;
    ggml_tensor* v_htp_flash_base = nullptr;
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
        L_q       = q->ne[1];
        L_k       = k->ne[1];
        d_head    = v->ne[0];
        N         = v->ne[3];
        n_kv_head = k->ne[2] / N;
        C         = d_head * n_head;
    }

    float scale = (1.0f / sqrt((float)d_head));

    int kv_pad       = 0;
    ggml_tensor* kqv = nullptr;
    const bool keep_for_htp_hybrid = sd_env_flag_enabled("SD_HTP_OPS") &&
                                     sd_env_flag_enabled("SD_HTP_COMPUTE");
    const bool use_htp_flash_layout = keep_for_htp_hybrid &&
                                      sd_env_flag_enabled("SD_HTP_ATTENTION") &&
                                      N == 1 &&
                                      kv_scale == 1.0f &&
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

    auto build_htp_kqv = [&](ggml_tensor* mask_in, bool with_lse = false) -> ggml_tensor* {
        const bool block_sparse_mask = mask_in != nullptr && mask_in->type == GGML_TYPE_I32;
        if (!use_htp_flash_layout) {
            return nullptr;
        }

        ggml_tensor* q_in = q_htp_flash;
        if (q_in->type != GGML_TYPE_F32) {
            return nullptr;
        }

        ggml_tensor* k_base = k_htp_flash_base;
        ggml_tensor* v_base = v_htp_flash_base;
        if (kv_pad != 0) {
            k_base = ggml_pad(ctx, k_base, 0, 0, kv_pad, 0);
            v_base = ggml_pad(ctx, v_base, 0, 0, kv_pad, 0);
        }
        if (kv_scale != 1.0f) {
            k_base = ggml_ext_scale(ctx, k_base, kv_scale);
            v_base = ggml_ext_scale(ctx, v_base, kv_scale);
        }
        if (k_base->type != GGML_TYPE_F16) {
            k_base = ggml_cast(ctx, k_base, GGML_TYPE_F16);
        }
        if (v_base->type != GGML_TYPE_F16) {
            v_base = ggml_cast(ctx, v_base, GGML_TYPE_F16);
        }

        ggml_tensor* k_in = ggml_permute(ctx, k_base, 0, 2, 1, 3);
        ggml_tensor* v_in = ggml_permute(ctx, v_base, 0, 2, 1, 3);

        if (!block_sparse_mask) {
            const int htp_dense_mask_pad =
                kv_pad == 0 ? (GGML_PAD(L_k, 64) - static_cast<int>(L_k)) : 0;
            if (mask_in != nullptr) {
                mask_in = dense_mask_to_kq_layout(mask_in);
            }
            if (mask_in == nullptr && (kv_pad > 0 || htp_dense_mask_pad > 0 || keep_for_htp_hybrid)) {
                mask_in = ggml_ext_zeros(ctx, L_k, L_q, 1, 1);
            }
            const int mask_pad = kv_pad > 0 ? kv_pad : htp_dense_mask_pad;
            if (mask_in != nullptr && mask_pad > 0) {
                auto pad_tensor = ggml_ext_full(ctx, -INFINITY, mask_pad, L_q, 1, 1);
                mask_in         = ggml_concat(ctx, mask_in, pad_tensor, 0);
            }
            if (mask_in != nullptr && mask_in->type != GGML_TYPE_F16) {
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

    auto flash_attn_o_view = [&](ggml_tensor* packed, bool with_lse = false) -> ggml_tensor* {
        GGML_ASSERT(packed != nullptr);
        GGML_ASSERT(!with_lse || packed->ne[0] == d_head + 1);
        ggml_tensor* out = ggml_view_3d(ctx, packed, d_head, n_head, L_q, packed->nb[1], packed->nb[2], 0);
        if (with_lse && kv_scale != 1.0f) {
            out = ggml_ext_scale(ctx, out, 1.0f / kv_scale);
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

        const bool block_sparse_force_full_keep =
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
            if (block_sparse_force_full_keep || block_sparse_keep_ratio == 1.0f) {
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
                block_sparse_static_halo = !block_sparse_center_sink &&
                                           block_sparse_token_h > 0 &&
                                           block_sparse_token_w > 0 &&
                                           block_sparse_token_h % 4 == 0 &&
                                           block_sparse_token_w % 8 == 0 &&
                                           n_head > 0 &&
                                           n_kv_head > 0 &&
                                           n_head % n_kv_head == 0 &&
                                           protected_start == static_cast<int64_t>(block_sparse_token_h) *
                                                                  block_sparse_token_w &&
                                           protected_len > 0 &&
                                           L_q == protected_start + protected_len &&
                                           L_k == protected_start + protected_len &&
                                           protected_start > 0 &&
                                           protected_start % SD_HTP_BLOCK_SPARSE_MASK_BLOCK == 0;
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
            const bool sparse_with_lse = packed_layout || (anchor_can_cache_residual && can_cache_exact_rho);
            ggml_tensor* dense_candidate = nullptr;
            ggml_tensor* dense_lse = nullptr;
            bool candidate_is_htp_layout = false;

            // An exact-rho anchor needs two attention operators with identical
            // Q/K/V. Reusing the dense operator's prepared inputs keeps those
            // copies on the main graph path, so the sparse sidecar can launch
            // while HTP computes the dense operator instead of after cache-tail
            // Q/K/V preparation has completed.
            if (anchor_can_cache_residual && can_cache_exact_rho) {
                dense_candidate = build_htp_kqv(nullptr, true);
                candidate_is_htp_layout = dense_candidate != nullptr;
                if (dense_candidate == nullptr) {
                    dense_candidate = build_kqv(q, k, v, nullptr, true);
                }
            }

            ggml_tensor* candidate = nullptr;
            if (packed_layout) {
                candidate = build_packed_kqv(block_mask, sparse_with_lse);
                candidate_is_htp_layout = false;
            } else if (dense_candidate != nullptr && candidate_is_htp_layout) {
                candidate = ggml_flash_attn_ext_with_lse(
                    ctx,
                    dense_candidate->src[0],
                    dense_candidate->src[1],
                    dense_candidate->src[2],
                    block_mask,
                    scale / kv_scale,
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
            GGML_ASSERT(block_mask_format != GGML_FLASH_ATTN_EXT_MASK_DEFAULT);
            ggml_flash_attn_ext_set_block_sparse(candidate, block_q, block_k, block_mask_format);
            if (block_sparse_static_halo) {
                ggml_flash_attn_ext_set_static_halo_2d(candidate,
                                                       protected_start,
                                                       protected_len,
                                                       block_sparse_token_h / 4,
                                                       block_sparse_token_w / 8);
            }
            if ((candidate_is_htp_layout && keep_for_htp_hybrid) || backend == nullptr || ggml_backend_supports_op(backend, candidate)) {
                ggml_tensor* sparse_out = flash_attn_o_view(candidate, sparse_with_lse);
                ggml_tensor* sparse_lse = sparse_with_lse ? flash_attn_lse_view(candidate) : nullptr;
                ggml_tensor* sparse_reshape_out = sparse_out;
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
                        if (dense_candidate == nullptr) {
                            dense_candidate = build_kqv(q, k, v, nullptr, true);
                        }
                    } else if (dense_candidate == nullptr) {
                        dense_candidate = build_htp_kqv(nullptr);
                        if (dense_candidate == nullptr) {
                            dense_candidate = build_kqv(q, k, v, nullptr);
                        }
                    }
                    ggml_tensor* dense_out = flash_attn_o_view(dense_candidate, can_cache_exact_rho);
                    if (can_cache_exact_rho) {
                        dense_lse = flash_attn_lse_view(dense_candidate);
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
                                "SD_HTP_SPARSE_OPENCL_RESIDUAL requires OpenCL sparse, "
                                "async, early-submit, residual cache, and exact-rho");
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
                                ctx, GGML_TYPE_F32, d_head, n_head, L_q, N,
                                residual_args, 3, sd_htp_exact_residual_cache_cpu,
                                GGML_N_TASKS_MAX, nullptr);
                        } else {
                            sparse_reshape_out = ggml_mul(ctx, sparse_out, rho);
                            ggml_set_name(
                                sparse_reshape_out,
                                "htp_cpu_residual_scale_mul");
                        }
                    }
                    if (residual == nullptr) {
                        residual = ggml_sub(ctx, dense_out, sparse_reshape_out);
                    }
                    ggml_set_name(residual, block_sparse_residual_cache_name);
                    (*block_sparse_cache_tensor)(block_sparse_residual_cache_name, residual);
                    if (sd_env_flag_enabled("SD_HTP_BLOCK_SPARSE_MASK_TRACE")) {
                        std::fprintf(stderr,
                                     "block_sparse_residual anchor=1 exact_rho=%d rho=%.4f cache=%s rho_cache=%s shape=[%lld,%lld,%lld,%lld]\n",
                                     rho != nullptr ? 1 : 0,
                                     block_sparse_residual_rho,
                                     block_sparse_residual_cache_name,
                                     block_sparse_residual_rho_cache_name != nullptr ? block_sparse_residual_rho_cache_name : "<unnamed>",
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
                    if (block_sparse_residual_scale != 1.0f) {
                        residual = ggml_ext_scale(ctx, residual, block_sparse_residual_scale);
                    }
                    kqv = ggml_add(ctx, sparse_reshape_out, residual);
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

/* SDXL with LoRA requires more space */
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
    virtual ggml_tensor* forward_with_lora(ggml_context* ctx,
                                           ggml_backend_t backend,
                                           ggml_tensor* x,
                                           ggml_tensor* w,
                                           ggml_tensor* b,
                                           const std::string& prefix,
                                           ForwardParams forward_params)                                                              = 0;
    virtual size_t get_extra_graph_size()                                                                                             = 0;
};

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
    std::shared_ptr<WeightAdapter> weight_adapter                    = nullptr;
    std::vector<std::pair<ggml_tensor*, std::string>>* debug_tensors = nullptr;
    std::function<ggml_tensor*(const std::string&)> get_cache_tensor;
    std::function<void(const std::string&, ggml_tensor*)> cache_tensor;
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

struct GGMLRunner {
protected:
    typedef std::function<ggml_cgraph*()> get_graph_cb_t;
    using GraphCutSegment = sd::ggml_graph_cut::Segment;
    using GraphCutPlan    = sd::ggml_graph_cut::Plan;

    ggml_backend_t params_backend  = nullptr;
    ggml_backend_t runtime_backend = nullptr;
    // 只有 diffusion 推理明确注入 HTP backend 后，runner 才会启用 RPCMEM 放置和 hybrid compute 判定。
    ggml_backend_t htp_backend     = nullptr;

    ggml_context* params_ctx                    = nullptr;
    ggml_backend_buffer_t params_buffer         = nullptr;
    ggml_context* offload_ctx                   = nullptr;
    ggml_backend_buffer_t runtime_params_buffer = nullptr;
    bool params_on_runtime_backend              = false;
    // 普通 runtime offload 现在改为显式记录“哪些参数被临时换到 runtime backend”，这样就能跳过常驻 RPCMEM 的 HTP 权重。
    std::vector<std::pair<ggml_tensor*, ggml_tensor*>> runtime_offload_pairs_;
    // 参照 llama.cpp-npu，eligible diffusion 权重在参数分配阶段直接放进 RPCMEM，而不是图执行前再复制一份常驻权重池。
    ggml_context* htp_param_placement_ctx_            = nullptr;
    ggml_backend_buffer_t htp_param_placement_buffer_ = nullptr;
    std::vector<std::pair<ggml_tensor*, ggml_tensor*>> htp_param_placement_pairs_;
    std::unordered_set<ggml_tensor*> htp_placed_param_tensors_;
    std::vector<ggml_tensor*> htp_packed_self_registered_weights_;

    ggml_context* cache_ctx            = nullptr;
    ggml_backend_buffer_t cache_buffer = nullptr;

    struct AsyncResidualCacheGeneration {
        uint64_t generation = 0;
        ggml_backend_htp_async_cache_slot* slot = nullptr;
        size_t consumer_count = 0;
        bool producer_pending = false;
    };
    std::unordered_map<std::string, std::deque<AsyncResidualCacheGeneration>>
        async_residual_cache_slots_;
    std::map<ggml_tensor*, uint64_t> cache_backed_tensor_generation_map_;
    std::vector<std::pair<std::string, uint64_t>> active_async_cache_consumers_;
    uint64_t next_async_residual_cache_generation_ = 1;

    ggml_context* compute_ctx    = nullptr;
    ggml_gallocr* compute_allocr = nullptr;
    struct RetiredComputeArenaGeneration {
        uint64_t generation = 0;
        ggml_gallocr* allocator = nullptr;
        size_t bytes = 0;
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
    // 记录每个 node/leaf 应该落在哪个 buffer，0=默认 runtime buffer，1=HTP RPCMEM buffer。
    std::vector<int> compute_node_buffer_ids_;
    std::vector<int> compute_leaf_buffer_ids_;

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
        if (htp_param_placement_ctx_ != nullptr) {
            ggml_free(htp_param_placement_ctx_);
            htp_param_placement_ctx_ = nullptr;
        }
        htp_param_placement_pairs_.clear();
        htp_placed_param_tensors_.clear();
        if (partial_offload_ctx != nullptr) {
            ggml_free(partial_offload_ctx);
            partial_offload_ctx = nullptr;
        }
    }

    void alloc_cache_ctx() {
        ggml_init_params params;
        params.mem_size   = static_cast<size_t>(MAX_PARAMS_TENSOR_NUM * ggml_tensor_overhead());
        params.mem_buffer = nullptr;
        params.no_alloc   = true;

        cache_ctx = ggml_init(params);
        GGML_ASSERT(cache_ctx != nullptr);
    }

    void free_cache_ctx() {
        if (cache_ctx != nullptr) {
            ggml_free(cache_ctx);
            cache_ctx = nullptr;
        }
    }

    void free_async_residual_cache_slots() {
        for (auto& keyed_slots : async_residual_cache_slots_) {
            for (AsyncResidualCacheGeneration& generation : keyed_slots.second) {
                ggml_backend_htp_async_cache_slot_free(generation.slot);
                generation.slot = nullptr;
            }
        }
        async_residual_cache_slots_.clear();
    }

    AsyncResidualCacheGeneration* find_async_residual_cache_slot(
            const std::string& key,
            uint64_t generation) {
        auto found = async_residual_cache_slots_.find(key);
        if (found == async_residual_cache_slots_.end()) {
            return nullptr;
        }
        for (AsyncResidualCacheGeneration& slot : found->second) {
            if (slot.generation == generation) {
                return &slot;
            }
        }
        return nullptr;
    }

    AsyncResidualCacheGeneration* latest_async_residual_cache_slot(
            const std::string& key) {
        auto found = async_residual_cache_slots_.find(key);
        if (found == async_residual_cache_slots_.end() || found->second.empty()) {
            return nullptr;
        }
        return &found->second.back();
    }

    bool async_residual_cache_backed(const std::string& key) const {
        const auto found = async_residual_cache_slots_.find(key);
        return found != async_residual_cache_slots_.end() &&
               !found->second.empty() && found->second.back().slot != nullptr;
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
                ggml_backend_htp_opencl_cache_query(
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
        std::deque<AsyncResidualCacheGeneration>& generations =
            async_residual_cache_slots_[key];
        if (generations.size() >= 2) {
            AsyncResidualCacheGeneration& oldest = generations.front();
            if (oldest.producer_pending || oldest.consumer_count != 0) {
                LOG_DEBUG(
                    "gpu_residual_cache_pressure_wait key=%s generation=%" PRIu64,
                    key.c_str(), oldest.generation);
                return nullptr;
            }
            ggml_backend_htp_async_cache_slot_free(oldest.slot);
            generations.pop_front();
        }

        char reason[512] = {};
        ggml_backend_htp_async_cache_slot* slot =
            ggml_backend_htp_async_cache_slot_create(
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
            if (generations.empty()) {
                async_residual_cache_slots_.erase(key);
            }
            return nullptr;
        }
        generations.push_back({ generation, slot, 0, false });
        return &generations.back();
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
            bool released = false;
            char reason[512] = {};
            if (!ggml_backend_htp_opencl_release_arena_generation(
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
            std::fprintf(
                stderr,
                "htp_opencl_event name=gpu_residual_arena_pressure_wait_begin "
                "ts_us=%lld generation=%" PRIu64 " bytes=%zu\n",
                (long long) wait_begin, oldest.generation, oldest.bytes);
            bool released = false;
            char reason[512] = {};
            if (!ggml_backend_htp_opencl_release_arena_generation(
                    htp_backend, oldest.generation, true, &released,
                    reason, sizeof(reason)) || !released) {
                LOG_ERROR(
                    "%s pressure wait for compute arena generation=%" PRIu64
                    " failed: %s",
                    get_desc().c_str(), oldest.generation,
                    reason[0] != '\0' ? reason : "unknown error");
                return false;
            }
            std::fprintf(
                stderr,
                "htp_opencl_event name=gpu_residual_arena_pressure_wait_end "
                "ts_us=%lld generation=%" PRIu64 "\n",
                (long long) ggml_time_us(), oldest.generation);
            ggml_gallocr_free(oldest.allocator);
            retired_compute_arenas_.pop_front();
        }

        size_t bytes = ggml_gallocr_get_buffer_size(compute_allocr, 0);
        if (!compute_node_buffer_ids_.empty()) {
            bytes += ggml_gallocr_get_buffer_size(compute_allocr, 1);
        }
        retired_compute_arenas_.push_back({ generation, compute_allocr, bytes });
        compute_allocr = nullptr;
        compute_node_buffer_ids_.clear();
        compute_leaf_buffer_ids_.clear();
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
        for (const auto& entry : cache_tensor_map) {
            if (entry.second != nullptr) {
                ggml_set_output(entry.second);
                ggml_build_forward_expand(gf, entry.second);
            }
        }
        if (htp_backend != nullptr) {
            ggml_backend_htp_opencl_retain_deferred_residual_inputs(gf);
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

    bool alloc_compute_buffer(ggml_cgraph* gf) {
        if (!reap_retired_compute_arenas(false)) {
            return false;
        }
        ggml_backend_buffer_type_t runtime_buft = ggml_backend_get_default_buffer_type(runtime_backend);
        ggml_backend_buffer_type_t htp_buft     = htp_backend != nullptr ? ggml_backend_get_default_buffer_type(htp_backend) : nullptr;
        compute_node_buffer_ids_.clear();
        compute_leaf_buffer_ids_.clear();

        if (htp_op_routing_enabled() && htp_buft != nullptr) {
            // 为 CPU 与 HTP RPCMEM 同时创建 allocator，后续按节点/叶子打到不同 buffer。
            if (compute_allocr == nullptr) {
                ggml_backend_buffer_type_t bufts[] = {runtime_buft, htp_buft};
                compute_allocr                    = ggml_gallocr_new_n(bufts, 2);
            }
            plan_compute_buffer_ids(gf, &compute_node_buffer_ids_, &compute_leaf_buffer_ids_);
        } else {
            if (compute_allocr == nullptr) {
                compute_allocr = ggml_gallocr_new(runtime_buft);
            }
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
            std::strcmp(node->name, "qkv_mul_rms_packed_dual") != 0) {
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
        return htp_placed_param_tensors_.find(root) != htp_placed_param_tensors_.end() &&
               ggml_backend_buft_is_rpcmem(ggml_backend_buffer_get_type(root->buffer));
    }

    bool should_place_param_in_htp_rpcmem(ggml_tensor* tensor) const {
        if (!htp_op_routing_enabled() || !sd_env_flag_enabled("SD_HTP_COMPUTE") || htp_backend == nullptr || tensor == nullptr) {
            return false;
        }
        if ((tensor->flags & SD_GGML_TENSOR_FLAG_HTP_PACKED_QKV_PARAM) != 0) {
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
            if (!is_htp_supported_f16_mul_mat(node)) {
                continue;
            }
            ggml_tensor* weight = node->src[0];
            if (should_prepare_htp_packed_f16_weight(weight) && seen.insert(weight).second) {
                tensors.push_back(weight);
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
        if (!htp_op_routing_enabled() || htp_param_placement_buffer_ != nullptr) {
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

        for (ggml_tensor* tensor : tensors) {
            ggml_tensor* htp_tensor = ggml_dup_tensor(htp_param_placement_ctx_, tensor);
            ggml_set_name(htp_tensor, tensor->name);
            htp_param_placement_pairs_.push_back({tensor, htp_tensor});
        }

        ggml_backend_buffer_type_t htp_buft = ggml_backend_get_default_buffer_type(htp_backend);
        htp_param_placement_buffer_ = ggml_backend_alloc_ctx_tensors_from_buft(htp_param_placement_ctx_, htp_buft);
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
        ggml_backend_buffer_set_usage(htp_param_placement_buffer_, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        size_t placed_bytes = 0;
        for (auto& pair : htp_param_placement_pairs_) {
            ggml_tensor* tensor     = pair.first;
            ggml_tensor* htp_tensor = pair.second;
            bind_tensor_allocation(tensor, htp_tensor);
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

        if (htp_param_placement_buffer_ != nullptr) {
            ggml_backend_buffer_free(htp_param_placement_buffer_);
            htp_param_placement_buffer_ = nullptr;
        }
        if (htp_param_placement_ctx_ != nullptr) {
            ggml_free(htp_param_placement_ctx_);
            htp_param_placement_ctx_ = nullptr;
        }
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
                // FLASH_ATTN_EXT 的 HTP 输入通常是 PERMUTE/RESHAPE view；allocator 不会给 view
                // 单独分配 buffer，必须把真正持有 data 的 backing root 放进 RPCMEM。
                if (node->op == GGML_OP_FLASH_ATTN_EXT && src != nullptr && tensor == get_htp_tensor_backing_root(src)) {
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
                // The dual packed QKV custom node consumes two offline-packed
                // weights at src[1] and src[6]. They are HTP inputs, not
                // row-major CPU consumers, so keep them in the packed-only
                // placement contract.
                const bool routed_to_htp_dual_qkv =
                    node->op == GGML_OP_CUSTOM &&
                    std::strcmp(node->name, "qkv_mul_rms_packed_dual") == 0 &&
                    (j == 1 || j == 6) &&
                    is_htp_target_op(node);
                const bool routed_to_htp = routed_to_htp_matmul || routed_to_htp_dual_qkv;
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
    }

    void free_cache_buffer() {
        if (cache_buffer != nullptr) {
            ggml_backend_buffer_free(cache_buffer);
            cache_buffer = nullptr;
        }
    }

    bool copy_cache_tensors_to_cache_buffer(const std::unordered_set<std::string>* cache_keep_names = nullptr) {
        ggml_context* old_cache_ctx            = cache_ctx;
        ggml_backend_buffer_t old_cache_buffer = cache_buffer;
        cache_ctx                              = nullptr;
        cache_buffer                           = nullptr;
        std::map<std::string, ggml_tensor*> merged_cache_sources;
        if (old_cache_ctx != nullptr) {
            for (ggml_tensor* tensor = ggml_get_first_tensor(old_cache_ctx); tensor != nullptr; tensor = ggml_get_next_tensor(old_cache_ctx, tensor)) {
                if (cache_keep_names != nullptr && cache_keep_names->find(tensor->name) == cache_keep_names->end()) {
                    continue;
                }
                if (async_residual_cache_backed(tensor->name)) {
                    continue;
                }
                merged_cache_sources[tensor->name] = tensor;
            }
        }
        for (const auto& kv : cache_tensor_map) {
            if (cache_keep_names != nullptr && cache_keep_names->find(kv.first) == cache_keep_names->end()) {
                continue;
            }
            if (async_residual_cache_backed(kv.first)) {
                continue;
            }
            merged_cache_sources[kv.first] = kv.second;
        }
        cache_tensor_map.clear();
        if (merged_cache_sources.empty()) {
            if (old_cache_buffer != nullptr) {
                ggml_backend_buffer_free(old_cache_buffer);
            }
            if (old_cache_ctx != nullptr) {
                ggml_free(old_cache_ctx);
            }
            return true;
        }

        alloc_cache_ctx();
        std::vector<std::pair<ggml_tensor*, ggml_tensor*>> source_to_cache_tensors;
        source_to_cache_tensors.reserve(merged_cache_sources.size());
        for (const auto& kv : merged_cache_sources) {
            ggml_tensor* source_tensor = sd::ggml_graph_cut::cache_source_tensor(kv.second);
            auto cache_tensor          = ggml_dup_tensor(cache_ctx, source_tensor);
            ggml_set_name(cache_tensor, kv.first.c_str());
            source_to_cache_tensors.push_back({source_tensor, cache_tensor});
        }
        size_t num_tensors = ggml_tensor_num(cache_ctx);
        cache_buffer       = ggml_backend_alloc_ctx_tensors(cache_ctx, runtime_backend);
        GGML_ASSERT(cache_buffer != nullptr);
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
                return false;
            }
            const bool use_staging_copy = src->view_src != nullptr || !ggml_is_contiguous(src) || src->buffer == nullptr;
            if (use_staging_copy) {
                std::vector<uint8_t> host_data(ggml_nbytes(src));
                ggml_backend_tensor_get(src, host_data.data(), 0, host_data.size());
                ggml_backend_tensor_set(dst, host_data.data(), 0, host_data.size());
            } else {
                ggml_backend_tensor_copy(src, dst);
            }
        }
        ggml_backend_synchronize(runtime_backend);
        size_t cache_buffer_size = ggml_backend_buffer_get_size(cache_buffer);
        LOG_DEBUG("%s cache backend buffer size = % 6.2f MB(%s) (%i tensors)",
                  get_desc().c_str(),
                  cache_buffer_size / (1024.f * 1024.f),
                  ggml_backend_is_cpu(runtime_backend) ? "RAM" : "VRAM",
                  num_tensors);
        if (old_cache_buffer != nullptr) {
            ggml_backend_buffer_free(old_cache_buffer);
        }
        if (old_cache_ctx != nullptr) {
            ggml_free(old_cache_ctx);
        }
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

    bool bind_cache_backed_tensors() {
        GGML_ASSERT(active_async_cache_consumers_.empty());
        for (const auto& kv : cache_backed_tensor_name_map) {
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
                        ggml_backend_htp_async_cache_slot_tensor(cached->slot);
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
            if (async_generation != 0) {
                char reason[512] = {};
                if (htp_backend == nullptr ||
                    !ggml_backend_htp_opencl_bind_cache_input(
                        htp_backend, graph_tensor, kv.second.c_str(),
                        async_generation, reason, sizeof(reason))) {
                    LOG_ERROR(
                        "%s bind async residual cache input failed: key=%s "
                        "generation=%" PRIu64 " reason=%s",
                        get_desc().c_str(), kv.second.c_str(), async_generation,
                        reason[0] != '\0' ? reason : "unknown error");
                    release_active_async_cache_consumers();
                    return false;
                }
                AsyncResidualCacheGeneration* cached =
                    find_async_residual_cache_slot(kv.second, async_generation);
                GGML_ASSERT(cached != nullptr);
                ++cached->consumer_count;
                active_async_cache_consumers_.push_back(
                    { kv.second, async_generation });
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

    bool bind_async_residual_cache_outputs() {
        if (!sd_env_flag_enabled("SD_HTP_SPARSE_OPENCL_RESIDUAL")) {
            return true;
        }
        std::vector<std::pair<std::string, ggml_tensor*>> outputs;
        for (const auto& entry : cache_tensor_map) {
            ggml_tensor* tensor = entry.second;
            if (tensor == nullptr || tensor->op != GGML_OP_CUSTOM ||
                entry.first.rfind("htp_block_sparse_residual_", 0) != 0) {
                continue;
            }
            outputs.push_back(entry);
        }
        if (outputs.empty()) {
            return true;
        }

        const uint64_t generation = next_async_residual_cache_generation_++;
        for (const auto& output : outputs) {
            AsyncResidualCacheGeneration* cached =
                allocate_async_residual_cache_slot(
                    output.first, output.second, generation);
            if (cached == nullptr || cached->slot == nullptr) {
                if (sd_env_flag_enabled("SD_HTP_SPARSE_OPENCL_STRICT")) {
                    return false;
                }
                continue;
            }
            ggml_tensor* slot_tensor =
                ggml_backend_htp_async_cache_slot_tensor(cached->slot);
            GGML_ASSERT(slot_tensor != nullptr);
            output.second->buffer = slot_tensor->buffer;
            output.second->data = slot_tensor->data;
            output.second->extra = slot_tensor->extra;
            cached->producer_pending = true;
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
        return plan.has_cuts &&
               plan.valid &&
               max_graph_vram_bytes > 0 &&
               plan.segments.size() > 1 &&
               !ggml_backend_is_cpu(runtime_backend);
    }

    bool can_attempt_graph_cut_segmented_compute() const {
        return max_graph_vram_bytes > 0 &&
               !ggml_backend_is_cpu(runtime_backend);
    }

    bool resolve_graph_cut_plan(ggml_cgraph* gf,
                                GraphCutPlan* plan_out) {
        GGML_ASSERT(plan_out != nullptr);
        GGML_ASSERT(gf != nullptr);
        *plan_out = sd::ggml_graph_cut::resolve_plan(runtime_backend,
                                                     gf,
                                                     &graph_cut_plan_cache_,
                                                     max_graph_vram_bytes,
                                                     params_tensor_set_,
                                                     get_desc().c_str());
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

    bool bind_segment_cached_inputs(ggml_cgraph* gf, const GraphCutSegment& segment) {
        GGML_ASSERT(gf != nullptr);
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
                    if (input_tensor->view_src != nullptr) {
                        input_tensor->view_src = cache_tensor;
                        input_tensor->buffer   = nullptr;
                        input_tensor->data     = cache_tensor->data == nullptr
                                                     ? nullptr
                                                     : static_cast<void*>(static_cast<char*>(cache_tensor->data) + input_tensor->view_offs);
                        input_tensor->extra    = cache_tensor->extra;
                    } else {
                        input_tensor->buffer = cache_tensor->buffer;
                        input_tensor->data   = cache_tensor->data;
                        input_tensor->extra  = cache_tensor->extra;
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

        int64_t t_alloc_begin = ggml_time_ms();
        if (!alloc_compute_buffer(gf)) {
            LOG_ERROR("%s alloc compute buffer failed", get_desc().c_str());
            if (use_partial_param_offload) {
                restore_partial_params();
            }
            return std::nullopt;
        }

        bool alloc_ok = false;
        if (!compute_node_buffer_ids_.empty() || !compute_leaf_buffer_ids_.empty()) {
            alloc_ok = ggml_gallocr_reserve_n(compute_allocr,
                                              gf,
                                              compute_node_buffer_ids_.data(),
                                              compute_leaf_buffer_ids_.data()) &&
                       ggml_gallocr_alloc_graph(compute_allocr, gf);
        } else {
            alloc_ok = ggml_gallocr_alloc_graph(compute_allocr, gf);
        }

        if (!alloc_ok) {
            LOG_ERROR("%s alloc compute graph failed", get_desc().c_str());
            if (free_compute_buffer_immediately) {
                free_compute_buffer();
            } else if (use_partial_param_offload) {
                restore_partial_params();
            }
            return std::nullopt;
        }
        int64_t t_alloc_end = ggml_time_ms();

        int64_t t_copy_begin = ggml_time_ms();
        if (!bind_async_residual_cache_outputs()) {
            LOG_ERROR("%s bind async residual cache outputs failed", get_desc().c_str());
            if (free_compute_buffer_immediately) {
                free_compute_buffer();
            } else if (use_partial_param_offload) {
                restore_partial_params();
            }
            return std::nullopt;
        }
        if (!bind_cache_backed_tensors()) {
            if (free_compute_buffer_immediately) {
                free_compute_buffer();
            } else if (use_partial_param_offload) {
                restore_partial_params();
            }
            return std::nullopt;
        }
        copy_data_to_backend_tensor(gf, !preserve_backend_tensor_data_map);
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
        if (use_htp_compute && sd_htp_opencl_deferred_residual_enabled()) {
            arena_generation = next_compute_arena_generation_++;
            active_compute_arena_generation_ = arena_generation;
            ggml_backend_htp_opencl_set_compute_arena_generation(
                htp_backend, arena_generation);
        }
        ggml_status status      = ggml_backend_graph_compute(use_htp_compute ? htp_backend : runtime_backend, gf);
        release_active_async_cache_consumers();
        if (use_htp_compute) {
            sd_backend_htp_set_hybrid_enabled(false);
        }
        if (arena_generation != 0 &&
            ggml_backend_htp_opencl_arena_generation_pending(
                htp_backend, arena_generation)) {
            if (!retire_current_compute_arena(arena_generation)) {
                status = GGML_STATUS_FAILED;
            }
        }
        active_compute_arena_generation_ = 0;
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
        if (!copy_cache_tensors_to_cache_buffer(cache_keep_names)) {
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

        free_compute_buffer();
        free_cache_ctx_and_buffer();

        std::optional<sd::Tensor<T>> output = sd::Tensor<T>();
        for (size_t seg_idx = 0; seg_idx < plan.segments.size(); ++seg_idx) {
            int64_t t_segment_begin = ggml_time_ms();
            const auto& segment     = plan.segments[seg_idx];
            auto future_cut_names   = sd::ggml_graph_cut::collect_future_input_names(gf, plan, seg_idx);
            LOG_DEBUG("%s graph cut executing segment %zu/%zu: %s",
                      get_desc().c_str(),
                      seg_idx + 1,
                      plan.segments.size(),
                      segment.group_name.c_str());

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
                    if (output_tensor != nullptr &&
                        sd::ggml_graph_cut::is_graph_cut_tensor(output_tensor) &&
                        future_cut_names.find(output_tensor->name) != future_cut_names.end()) {
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
            this->cache(name, tensor);
        };
        runner_ctx.bind_cache_tensor = [this](const std::string& name, ggml_tensor* tensor) {
            if (tensor != nullptr) {
                this->cache_backed_tensor_name_map[tensor] = name;
                if (sd_env_flag_enabled("SD_HTP_SPARSE_OPENCL_RESIDUAL")) {
                    AsyncResidualCacheGeneration* generation =
                        this->latest_async_residual_cache_slot(name);
                    if (generation != nullptr) {
                        this->cache_backed_tensor_generation_map_[tensor] =
                            generation->generation;
                    }
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
        size_t num_tensors = ggml_tensor_num(params_ctx);
        if (num_tensors > 0) {
            // ggml_backend_alloc_ctx_tensors fails when all tensors are already allocated
            // (typical for memory-mapped weights). See ggml-alloc.c n_buffers==0 branch.
            // 先按 HTP buffer type 放置可执行的 F16 二维权重，后续普通 params 分配会自动跳过这些已分配 tensor。
            if (!allocate_htp_placed_params()) {
                return false;
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
        params_buffer = ggml_backend_alloc_ctx_tensors(params_ctx, params_backend);
        if (params_buffer == nullptr) {
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
        if (params_buffer != nullptr) {
            ggml_backend_buffer_free(params_buffer);
            params_buffer = nullptr;
        }
    }

    size_t get_params_buffer_size() {
        size_t size = 0;
        if (params_buffer != nullptr) {
            size += ggml_backend_buffer_get_size(params_buffer);
        }
        if (htp_param_placement_buffer_ != nullptr) {
            size += ggml_backend_buffer_get_size(htp_param_placement_buffer_);
        }
        return size;
    }

    void free_cache_ctx_and_buffer() {
        free_cache_buffer();
        free_cache_ctx();
    }

    void free_compute_buffer() {
        reap_retired_compute_arenas(false);
        if (compute_allocr != nullptr) {
            ggml_gallocr_free(compute_allocr);
            compute_allocr = nullptr;
        }
        compute_node_buffer_ids_.clear();
        compute_leaf_buffer_ids_.clear();
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

    void cache(const std::string name, ggml_tensor* tensor) {
        if (tensor != nullptr && tensor->view_src != nullptr) {
            tensor = ggml_cont(compute_ctx, tensor);
        }
        cache_tensor_map[name] = tensor;
    }

    ggml_tensor* get_cache_tensor_by_name(const std::string& name) {
        if (sd_env_flag_enabled("SD_HTP_SPARSE_OPENCL_RESIDUAL")) {
            AsyncResidualCacheGeneration* generation =
                latest_async_residual_cache_slot(name);
            if (generation != nullptr && generation->slot != nullptr) {
                return ggml_backend_htp_async_cache_slot_tensor(generation->slot);
            }
        }
        if (cache_ctx == nullptr) {
            return nullptr;
        }
        return ggml_get_tensor(cache_ctx, name.c_str());
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
        if (!uses_cross_step_cache && can_attempt_graph_cut_segmented_compute()) {
            GraphCutPlan plan;
            if (!resolve_graph_cut_plan(gf, &plan)) {
                free_compute_ctx();
                return std::nullopt;
            }
            if (should_use_graph_cut_segmented_compute(plan)) {
                return compute_with_graph_cuts<T>(gf,
                                                  plan,
                                                  n_threads,
                                                  free_compute_buffer_immediately,
                                                  no_return);
            }
        }
        if (!alloc_compute_buffer(gf)) {
            LOG_ERROR("%s alloc compute buffer failed", get_desc().c_str());
            return std::nullopt;
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
        // Mark dual packed QKV's F32 bias and RMSNorm scales before parameter
        // placement, so they are allocated in RPCMEM with the packed weights.
        const bool packed_qkv_param_prefix = prefix.find("attn.qkv.") != std::string::npos ||
                                             prefix.find("attn.ln_q.") != std::string::npos ||
                                             prefix.find("attn.ln_k.") != std::string::npos;
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
            return ctx->weight_adapter->forward_with_lora(ctx->ggml_ctx, ctx->backend, x, w, b, prefix, forward_params);
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
            return ctx->weight_adapter->forward_with_lora(ctx->ggml_ctx, ctx->backend, x, w, b, prefix, forward_params);
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
                return ctx->weight_adapter->forward_with_lora(ctx->ggml_ctx, ctx->backend, x, w, b, prefix, forward_params);
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
                out_slices[i]                    = ctx->weight_adapter->forward_with_lora(ctx->ggml_ctx, ctx->backend, x_i, w_i, b_i, prefix, forward_params);
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

__STATIC_INLINE__ ggml_tensor* ggml_ext_lokr_forward(
    ggml_context* ctx,
    ggml_backend_t backend,
    ggml_tensor* h,    // Input: [q, batch] or [W, H, q, batch]
    ggml_tensor* w1,   // Outer C (Full rank)
    ggml_tensor* w1a,  // Outer A (Low rank part 1)
    ggml_tensor* w1b,  // Outer B (Low rank part 2)
    ggml_tensor* w2,   // Inner BA (Full rank)
    ggml_tensor* w2a,  // Inner A (Low rank part 1)
    ggml_tensor* w2b,  // Inner B (Low rank part 2)
    bool is_conv,
    WeightAdapter::ForwardParams::conv2d_params_t conv_params,
    float scale) {
    GGML_ASSERT((w1 != nullptr || (w1a != nullptr && w1b != nullptr)));
    GGML_ASSERT((w2 != nullptr || (w2a != nullptr && w2b != nullptr)));

    int uq = (w1 != nullptr) ? (int)w1->ne[0] : (int)w1a->ne[0];
    int up = (w1 != nullptr) ? (int)w1->ne[1] : (int)w1b->ne[1];

    int q_actual = is_conv ? (int)h->ne[2] : (int)h->ne[0];
    int vq       = q_actual / uq;

    int vp = (w2 != nullptr) ? (is_conv ? (int)w2->ne[3] : (int)w2->ne[1])
                             : (int)w2a->ne[1];
    GGML_ASSERT(q_actual == (uq * vq) && "Input dimension mismatch for LoKR split");

    ggml_tensor* hb;

    if (!is_conv) {
        int batch          = (int)h->ne[1];
        int merge_batch_uq = batch;
        int merge_batch_vp = batch;

        if (sd_backend_is(backend, "Vulkan")) {
            if (batch > 1) {
                // no access to backend here, worst case is slightly worse perfs for other backends when built alongside Vulkan backend
                int max_batch    = 65535;
                int max_batch_uq = max_batch / uq;
                merge_batch_uq   = 1;
                for (int i = max_batch_uq; i > 0; i--) {
                    if (batch % i == 0) {
                        merge_batch_uq = i;
                        break;
                    }
                }

                int max_batch_vp = max_batch / vp;
                merge_batch_vp   = 1;
                for (int i = max_batch_vp; i > 0; i--) {
                    if (batch % i == 0) {
                        merge_batch_vp = i;
                        break;
                    }
                }
            }
        }

        ggml_tensor* h_split = ggml_reshape_3d(ctx, h, vq, uq * merge_batch_uq, batch / merge_batch_uq);
        if (w2 != nullptr) {
            hb = ggml_mul_mat(ctx, w2, h_split);
        } else {
            hb = ggml_mul_mat(ctx, w2b, ggml_mul_mat(ctx, w2a, h_split));
        }

        if (batch > 1) {
            hb = ggml_reshape_3d(ctx, hb, vp, uq, batch);
        }
        ggml_tensor* hb_t = ggml_cont(ctx, ggml_transpose(ctx, hb));
        hb_t              = ggml_reshape_3d(ctx, hb_t, uq, vp * merge_batch_vp, batch / merge_batch_vp);

        ggml_tensor* hc_t;
        if (w1 != nullptr) {
            hc_t = ggml_mul_mat(ctx, w1, hb_t);
        } else {
            hc_t = ggml_mul_mat(ctx, w1b, ggml_mul_mat(ctx, w1a, hb_t));
        }

        if (batch > 1) {
            hc_t = ggml_reshape_3d(ctx, hc_t, up, vp, batch);
        }

        ggml_tensor* hc  = ggml_transpose(ctx, hc_t);
        ggml_tensor* out = ggml_reshape_2d(ctx, ggml_cont(ctx, hc), up * vp, batch);
        return ggml_ext_scale(ctx, out, scale);
    } else {
        int batch = (int)h->ne[3];
        // 1. Reshape input: [W, H, vq*uq, batch] -> [W, H, vq, uq * batch]
        ggml_tensor* h_split = ggml_reshape_4d(ctx, h, h->ne[0], h->ne[1], vq, uq * batch);

        if (w2 != nullptr) {
            hb = ggml_ext_conv_2d(ctx, h_split, w2, nullptr,
                                  conv_params.s0,
                                  conv_params.s1,
                                  conv_params.p0,
                                  conv_params.p1,
                                  conv_params.d0,
                                  conv_params.d1,
                                  conv_params.direct,
                                  conv_params.circular_x,
                                  conv_params.circular_y,
                                  conv_params.scale);
        } else {
            // swap a and b order for conv lora
            ggml_tensor* a = w2b;
            ggml_tensor* b = w2a;

            // unpack conv2d weights if needed
            if (ggml_n_dims(a) < 4) {
                int k = (int)sqrt(a->ne[0] / h_split->ne[2]);
                GGML_ASSERT(k * k * h_split->ne[2] == a->ne[0]);
                a = ggml_reshape_4d(ctx, a, k, k, a->ne[0] / (k * k), a->ne[1]);
            } else if (a->ne[2] != h_split->ne[2]) {
                int k = (int)sqrt(a->ne[2] / h_split->ne[2]);
                GGML_ASSERT(k * k * h_split->ne[2] == a->ne[2]);
                a = ggml_reshape_4d(ctx, a, a->ne[0] * k, a->ne[1] * k, a->ne[2] / (k * k), a->ne[3]);
            }
            ggml_tensor* ha = ggml_ext_conv_2d(ctx, h_split, a, nullptr,
                                               conv_params.s0,
                                               conv_params.s1,
                                               conv_params.p0,
                                               conv_params.p1,
                                               conv_params.d0,
                                               conv_params.d1,
                                               conv_params.direct,
                                               conv_params.circular_x,
                                               conv_params.circular_y,
                                               conv_params.scale);

            // not supporting lora_mid here
            hb = ggml_ext_conv_2d(ctx,
                                  ha,
                                  b,
                                  nullptr,
                                  1,
                                  1,
                                  0,
                                  0,
                                  1,
                                  1,
                                  conv_params.direct,
                                  conv_params.circular_x,
                                  conv_params.circular_y,
                                  conv_params.scale);
        }

        // Current hb shape: [W_out, H_out, vp, uq * batch]
        int w_out = (int)hb->ne[0];
        int h_out = (int)hb->ne[1];

        // ggml_tensor* hb_cat = ggml_reshape_4d(ctx, hb, w_out , h_out , vp * uq, batch);
        // [W_out, H_out, vp * uq,  batch]
        // Now left to compute (W1 kr Id) * hb_cat == (W1 kr W2) cv h

        // merge the uq groups of size vp*w_out*h_out
        ggml_tensor* hb_merged = ggml_reshape_2d(ctx, hb, w_out * h_out * vp, uq * batch);
        ggml_tensor* hc_t;
        ggml_tensor* hb_merged_t = ggml_cont(ctx, ggml_transpose(ctx, hb_merged));
        if (w1 != nullptr) {
            // Would be great to be able to transpose w1 instead to avoid transposing both hb and hc
            hc_t = ggml_mul_mat(ctx, w1, hb_merged_t);
        } else {
            hc_t = ggml_mul_mat(ctx, w1b, ggml_mul_mat(ctx, w1a, hb_merged_t));
        }
        ggml_tensor* hc = ggml_transpose(ctx, hc_t);
        // ungroup
        ggml_tensor* out = ggml_reshape_4d(ctx, ggml_cont(ctx, hc), w_out, h_out, up * vp, batch);
        return ggml_ext_scale(ctx, out, scale);
    }
}

#endif  // __GGML_EXTEND__HPP__
