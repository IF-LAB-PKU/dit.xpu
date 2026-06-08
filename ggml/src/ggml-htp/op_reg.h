// HTP 算子注册表。定义所有支持算子的枚举索引（HtpOpsIndex），
// 以及每种算子对应的参数结构体（算子类型 + 张量维度 + 各张量的 RpcmemBufAddr）。
// 目前支持 RMS_NORM_F32、permuted F16 MAT_MUL、三种量化 MAT_MUL 以及 FLASH_ATTN。

#pragma once

#include <stdint.h>

enum HtpOpsIndex {
    HTP_OPS_RMS_NORM_F32,
    HTP_OPS_MAT_MUL_PERMUTED_W16A32,
    HTP_OPS_MAT_MUL_PERMUTED_W4D16A32,
    HTP_OPS_MAT_MUL_PERMUTED_W8D16A32,
    HTP_OPS_MAT_MUL_PERMUTED_W4D16A32_IQ4_NL,
    HTP_OPS_FLASH_ATTN_QO_F32_KV_F16,
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

struct MatMulParams {
    struct RpcmemBufAddr output;
    struct RpcmemBufAddr activation;
    struct RpcmemBufAddr weight;
    int32_t m;
    int32_t k;
    int32_t n;
} __attribute__((packed));

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
} __attribute__((packed));
