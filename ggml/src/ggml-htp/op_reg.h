// HTP 算子注册表。定义所有支持算子的枚举索引（HtpOpsIndex），
// 以及每种算子对应的参数结构体（算子类型 + 张量维度 + 各张量的 RpcmemBufAddr）。
// 目前支持 RMS_NORM_F32、permuted F16 MAT_MUL、三种量化 MAT_MUL 以及 FLASH_ATTN。

#pragma once

#include <stddef.h>
#include <stdint.h>

enum HtpSparsePolicy {
    HTP_SPARSE_POLICY_NONE        = 0,
    HTP_SPARSE_POLICY_STATIC_HALO = 1,
};

enum HtpHaloKvLoadMode {
    HTP_HALO_KV_LOAD_BASELINE    = 0,
    HTP_HALO_KV_LOAD_L2_PREFETCH = 1,
    HTP_HALO_KV_LOAD_USERDMA_AB  = 2,
    HTP_HALO_KV_LOAD_CHUNKED     = 3,
};

// Packed HMX kernels intentionally support only the contracts exercised by
// the deployed 8 Gen 3 binaries. Keep this predicate C-compatible so the
// ggml host and DSP executor cannot drift apart.
static inline int htp_packed_head_count_supported(int n_heads) {
    return n_heads == 24 || n_heads == 30 || n_heads == 38;
}

static inline int htp_packed_head_dim_supported(int head_dim) {
    return head_dim == 64 || head_dim == 128;
}

static inline int htp_packed_shape_supported(int n_heads, int n_kv_heads,
                                              int head_dim, int require_equal_kv) {
    return n_heads > 0 && n_kv_heads > 0 && n_heads % n_kv_heads == 0 &&
           htp_packed_head_count_supported(n_heads) &&
           htp_packed_head_dim_supported(head_dim) &&
           (n_heads != 38 || head_dim == 64) &&
           (!require_equal_kv || (n_heads == n_kv_heads &&
                                  htp_packed_head_count_supported(n_kv_heads)));
}

enum HtpOpsIndex {
    HTP_OPS_RMS_NORM_F32,
    HTP_OPS_MAT_MUL_PERMUTED_W16A32,
    HTP_OPS_MAT_MUL_PERMUTED_W4D16A32,
    HTP_OPS_MAT_MUL_PERMUTED_W8D16A32,
    HTP_OPS_MAT_MUL_PERMUTED_W4D16A32_IQ4_NL,
    HTP_OPS_FLASH_ATTN_QO_F32_KV_F16,
    HTP_OPS_BLOCK_SPARSE_ATTN_QO_F32_KV_F16,
    HTP_OPS_BLOCK_SPARSE_SCORE_QO_F32_KM_F16,
    HTP_OPS_QKV_MUL_RMS_PACKED_W16A32,
  HTP_OPS_BLOCK_SPARSE_ATTN_QKV_PACKED_F16,
  HTP_OPS_QKV_MUL_RMS_PACKED_DUAL_W16A32,
  HTP_OPS_QKV_MUL_RMS_PACKED_ROPE_W16A32,
  HTP_OPS_QKV_MUL_RMS_PACKED_DUAL_ROPE_W16A32,
  HTP_OPS_FUSED_ANCHOR_ATTN_QO_F32_KV_F16,
  HTP_OPS_ATTN_REARRANGE_BENCH,
  HTP_OPS_MAT_MUL_NATIVE_W8A8,
  HTP_OPS_MAT_MUL_F16_EPILOGUE,
  HTP_OPS_CONV2D_NCHW_F16,
  HTP_OPS_ATTENTION_F32,
  HTP_OPS_MODEL_AUX_F32,
  HTP_OPS_COUNT,
};

struct RpcmemBufAddr {
    int32_t fd;
    int32_t offset;
} __attribute__((packed));

struct RmsNormF32Params {
    struct RpcmemBufAddr dst;
    struct RpcmemBufAddr src;
    int32_t ne0;
    int32_t ne1;
} __attribute__((packed));

struct W8A8Header {
    uint32_t magic, version, k, n;
    uint32_t weight_offset, scale_offset, sum_offset, inverse_offset, bytes, reserved;
    uint64_t original_f16_signature;
    uint8_t padding[80];
};
static_assert(sizeof(W8A8Header) == 128);

struct NativeW8A8Params {
    struct RpcmemBufAddr output, activation, constants;
    int32_t m, k, n, elapsed_us;
} __attribute__((packed));
static_assert(sizeof(NativeW8A8Params) == 40);

// Optional diagnostic trailer, matching the DSP ABI. Mode 3 stores A64/W_N64
// capacities and a table-hit bit in reserved, weight load count/bytes in the
// mac_ticks/readout_ticks fields; the five complete phase timings stay valid.
struct NativeW8A8Profile {
    uint32_t mode, reserved;
    uint64_t setup_ticks, activation_ticks, weight_ticks, hmx_ticks, output_ticks;
    uint64_t mac_ticks, readout_ticks;
};
struct NativeW8A8ProfileParams {
    NativeW8A8Params base;
    NativeW8A8Profile profile;
} __attribute__((packed));
// Opt-in CPU epilogue experiment. Raw planes occupy the normal output tensor;
// only progress descriptors and activation scales need an auxiliary buffer.
#define W8A8_CPU_MAX_BATCHES 64
#define W8A8_CONTROL_WEIGHT_FIRST 1u
struct W8A8CpuBatch {
  int32_t m0, rows, n0, columns;
};
struct W8A8CpuControl {
  uint32_t produced;
  uint32_t flags;
  uint32_t vtcm_bytes, activation_rows, weight_columns, weight_loads;
  uint64_t weight_loaded_bytes;
  uint32_t resident_batch_rows; // Requested M batch size, clamped to VTCM capacity.
  uint8_t padding[92];
  float scales[4096];
  struct W8A8CpuBatch batches[W8A8_CPU_MAX_BATCHES];
};
struct NativeW8A8CpuParams {
  struct NativeW8A8Params base;
  struct RpcmemBufAddr control;
} __attribute__((packed));
static_assert(sizeof(NativeW8A8CpuParams) == 48);
static_assert(sizeof(W8A8CpuControl) == 17536);
static_assert(offsetof(W8A8CpuControl, scales) == 128);
static_assert(offsetof(W8A8CpuControl, batches) == 16512);

static_assert(sizeof(NativeW8A8Profile) == 64);
static_assert(sizeof(NativeW8A8ProfileParams) == 104);
static_assert(HTP_OPS_MAT_MUL_NATIVE_W8A8 == 15);

struct MatMulParams {
    struct RpcmemBufAddr output;
    struct RpcmemBufAddr activation;
    struct RpcmemBufAddr weight;
    int32_t m;
    int32_t k;
    int32_t n;
    // These fields are part of the wire ABI even when host-side tracing is
    // disabled.  The DSP skeleton may enable tracing independently.
    uint64_t debug_vtcm_base;
    uint32_t debug_vtcm_size;
    int32_t  debug_vtcm_ctx;
    int32_t  debug_hmx_ctx;
    uint64_t debug_worker_pool;
    int32_t  debug_hmx_lock_result;
    uint64_t debug_dsp_activation_hash;
    uint64_t debug_dsp_output_hash;
    uint64_t debug_dsp_output_hash_after_invalidate;
} __attribute__((packed));

static_assert(sizeof(MatMulParams) == 92);
// Optional profiling trailer. The original 92-byte request remains supported.
struct MatMulProfileParams {
    MatMulParams matmul;
    int32_t elapsed_us;
} __attribute__((packed));
static_assert(sizeof(MatMulProfileParams) == 96);

struct QkvMulRmsPackedParams {
    struct RpcmemBufAddr q;
    struct RpcmemBufAddr k_out;
    struct RpcmemBufAddr v;
    struct RpcmemBufAddr activation;
    struct RpcmemBufAddr weight;
    struct RpcmemBufAddr bias;
    struct RpcmemBufAddr q_norm;
    struct RpcmemBufAddr k_norm;
    int32_t tokens;
    int32_t input_dim;
    int32_t n_heads;
    int32_t head_dim;
    int32_t elapsed_us;
} __attribute__((packed));

struct QkvMulRmsPackedDualParams {
    struct RpcmemBufAddr q;
    struct RpcmemBufAddr k_out;
    struct RpcmemBufAddr v;
    struct RpcmemBufAddr image_activation;
    struct RpcmemBufAddr image_weight;
    struct RpcmemBufAddr image_bias;
    struct RpcmemBufAddr image_q_norm;
    struct RpcmemBufAddr image_k_norm;
    struct RpcmemBufAddr text_activation;
    struct RpcmemBufAddr text_weight;
    struct RpcmemBufAddr text_bias;
    struct RpcmemBufAddr text_q_norm;
    struct RpcmemBufAddr text_k_norm;
    int32_t image_tokens;
    int32_t text_tokens;
    int32_t input_dim;
    int32_t n_heads;
    int32_t head_dim;
    int32_t elapsed_us;
} __attribute__((packed));

struct QkvMulRmsPackedBufferSizes {
    size_t combined_tokens;
    size_t token_tiles;
    size_t qkv_bytes;
    size_t activation_bytes;
    size_t weight_bytes;
    size_t bias_bytes;
    size_t norm_bytes;
};

static_assert(sizeof(QkvMulRmsPackedParams) == 84);
static_assert(sizeof(QkvMulRmsPackedDualParams) == 128);

struct QkvMulRmsPackedRopeParams {
    struct RpcmemBufAddr q;
    struct RpcmemBufAddr k_out;
    struct RpcmemBufAddr v;
    struct RpcmemBufAddr activation;
    struct RpcmemBufAddr weight;
    struct RpcmemBufAddr bias;
    struct RpcmemBufAddr q_norm;
    struct RpcmemBufAddr k_norm;
    struct RpcmemBufAddr rope_pe;
    int32_t tokens;
    int32_t input_dim;
    int32_t n_heads;
    int32_t head_dim;
    int32_t rope_token_offset;
    int32_t rope_token_count;
    int32_t elapsed_us;
} __attribute__((packed));

struct QkvMulRmsPackedDualRopeParams {
    struct RpcmemBufAddr q;
    struct RpcmemBufAddr k_out;
    struct RpcmemBufAddr v;
    struct RpcmemBufAddr image_activation;
    struct RpcmemBufAddr image_weight;
    struct RpcmemBufAddr image_bias;
    struct RpcmemBufAddr image_q_norm;
    struct RpcmemBufAddr image_k_norm;
    struct RpcmemBufAddr text_activation;
    struct RpcmemBufAddr text_weight;
    struct RpcmemBufAddr text_bias;
    struct RpcmemBufAddr text_q_norm;
    struct RpcmemBufAddr text_k_norm;
    struct RpcmemBufAddr rope_pe;
    int32_t image_tokens;
    int32_t text_tokens;
    int32_t input_dim;
    int32_t n_heads;
    int32_t head_dim;
    int32_t image_rope_token_offset;
    int32_t text_rope_token_offset;
    int32_t elapsed_us;
} __attribute__((packed));

static_assert(sizeof(QkvMulRmsPackedRopeParams) == 100);
static_assert(sizeof(QkvMulRmsPackedDualRopeParams) == 144);

struct FlashAttnParams {
    struct RpcmemBufAddr o;
    struct RpcmemBufAddr q;
    struct RpcmemBufAddr k;
    struct RpcmemBufAddr v;
    struct RpcmemBufAddr mask;
    int32_t qo_len;
    int32_t kv_len;
    int32_t n_heads;
    int32_t n_kv_heads;
    int32_t head_dim;
    int32_t impl;
    int32_t with_lse;
    int32_t elapsed_us;
} __attribute__((packed));

struct FusedAnchorAttnParams {
    struct RpcmemBufAddr output;
    struct RpcmemBufAddr q;
    struct RpcmemBufAddr k;
    struct RpcmemBufAddr v;
    struct RpcmemBufAddr dense_mask;
    struct RpcmemBufAddr block_mask;
    int32_t qo_len;
    int32_t kv_len;
    int32_t n_heads;
    int32_t n_kv_heads;
    int32_t head_dim;
    int32_t block_q;
    int32_t block_k;
    int32_t n_q_blocks;
    int32_t n_k_blocks;
    int32_t max_kept_blocks;
    int32_t mask_format;
    int32_t sparse_policy;
    int32_t image_tokens;
    int32_t text_tokens;
    int32_t image_block_rows;
    int32_t image_block_cols;
    int32_t elapsed_us;
} __attribute__((packed));

static_assert(offsetof(FusedAnchorAttnParams, elapsed_us) == 112);
static_assert(sizeof(FusedAnchorAttnParams) == 116);

struct BlockSparseAttnParams {
    struct RpcmemBufAddr o;
    struct RpcmemBufAddr q;
    struct RpcmemBufAddr k;
    struct RpcmemBufAddr v;
    struct RpcmemBufAddr block_mask;
    int32_t qo_len;
    int32_t kv_len;
    int32_t n_heads;
    int32_t n_kv_heads;
    int32_t head_dim;
    int32_t block_q;
    int32_t block_k;
    int32_t n_q_blocks;
    int32_t n_k_blocks;
    int32_t mask_format;
    int32_t max_kept_blocks;
    int32_t with_lse;
    int32_t elapsed_us;
    int32_t sparse_policy;
    int32_t image_tokens;
    int32_t text_tokens;
    int32_t halo_kv_load_mode;
    int32_t halo_trace;
    int32_t image_block_rows;
    int32_t image_block_cols;
    int32_t halo_exec_mode;
    int32_t halo_image_rows_updated;
    int32_t halo_text_q_image_row_updates;
    int32_t halo_text_q_image_block_updates;
    int32_t halo_image_q_batch_updates;
    int32_t halo_image_q_segment_updates;
} __attribute__((packed));

enum HtpHaloExecTraceMode {
    HTP_HALO_EXEC_TRACE_L2_PREFETCH = 1,
    HTP_HALO_EXEC_TRACE_USERDMA_ROWS = 2,
    HTP_HALO_EXEC_TRACE_CHUNKED_FALLBACK = 3,
};

static_assert(offsetof(BlockSparseAttnParams, elapsed_us) == 88);
static_assert(offsetof(BlockSparseAttnParams, sparse_policy) == 92);
static_assert(offsetof(BlockSparseAttnParams, image_block_rows) == 112);
static_assert(offsetof(BlockSparseAttnParams, image_block_cols) == 116);
static_assert(offsetof(BlockSparseAttnParams, halo_exec_mode) == 120);
static_assert(offsetof(BlockSparseAttnParams, halo_image_q_batch_updates) == 136);
static_assert(offsetof(BlockSparseAttnParams, halo_image_q_segment_updates) == 140);
static_assert(sizeof(BlockSparseAttnParams) == 144);

struct BlockSparseScoreParams {
    struct RpcmemBufAddr score;
    struct RpcmemBufAddr q_mean;
    struct RpcmemBufAddr k_mean;
    int32_t head_dim;
    int32_t n_q_blocks;
    int32_t n_k_blocks;
} __attribute__((packed));

// Separate opcode: old skeletons reject the request instead of misreading it.
struct F16EpilogueParams {
    struct RpcmemBufAddr output, activation, weight, bias;
    int32_t m, k, n, channels, mode, elapsed_us;
} __attribute__((packed));
// mode: 0=bias, 1=bias+GELU, 2=bias+QuickGELU, 3=bias+NCHW store.
#ifdef __cplusplus
static_assert(sizeof(F16EpilogueParams) == 56);
static_assert(HTP_OPS_MAT_MUL_F16_EPILOGUE == 16);
#else
_Static_assert(sizeof(struct F16EpilogueParams) == 56, "F16 epilogue ABI changed");
_Static_assert(HTP_OPS_MAT_MUL_F16_EPILOGUE == 16, "F16 epilogue opcode changed");
#endif

struct NativeConvParams {
    struct RpcmemBufAddr output, activation, weight, bias;
    int32_t iw, ih, ic, oc, batch, ow, oh, kw, kh, sx, sy, px, py, dx, dy, kp, np, upscale;
    int32_t elapsed_us, gather_us, weight_us, hmx_us, store_us;
} __attribute__((packed));
struct NativeAttentionParams {
    struct RpcmemBufAddr output, q, k, v, mask;
    int32_t queries, keys, heads, dim, batch, mask_heads, mask_batch, mask_queries;
    float scale;
    int32_t elapsed_us;
} __attribute__((packed));

struct NativeAuxParams {
 struct RpcmemBufAddr output, input, weight, bias;
 int32_t mode, count, width, channels, spatial, groups, batch, other_count, activation;
 float eps;
 int32_t elapsed_us;
} __attribute__((packed));

#ifdef __cplusplus
static_assert(sizeof(NativeConvParams)==124 && HTP_OPS_CONV2D_NCHW_F16==17);
static_assert(sizeof(NativeAttentionParams)==80 && HTP_OPS_ATTENTION_F32==18);
static_assert(sizeof(NativeAuxParams)==76 && HTP_OPS_MODEL_AUX_F32==19);
#else
_Static_assert(sizeof(struct NativeConvParams)==124 && HTP_OPS_CONV2D_NCHW_F16==17,"native convolution ABI");
_Static_assert(sizeof(struct NativeAttentionParams)==80 && HTP_OPS_ATTENTION_F32==18,"native attention ABI");
_Static_assert(sizeof(struct NativeAuxParams)==76 && HTP_OPS_MODEL_AUX_F32==19,"native auxiliary ABI");
#endif
