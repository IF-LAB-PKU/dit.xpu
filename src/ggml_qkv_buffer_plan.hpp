#ifndef SD_GGML_QKV_BUFFER_PLAN_HPP
#define SD_GGML_QKV_BUFFER_PLAN_HPP

#include "ggml.h"

using sd_htp_qkv_root_fn = ggml_tensor * (*)(ggml_tensor *);

inline bool sd_htp_anchor_dense_consumer_matches_qkv(
        const ggml_tensor * sparse,
        const ggml_tensor * consumer,
        sd_htp_qkv_root_fn root_of) {
    if (sparse == nullptr || consumer == nullptr || root_of == nullptr ||
        sparse->op != GGML_OP_FLASH_ATTN_EXT ||
        consumer->op != GGML_OP_FLASH_ATTN_EXT ||
        sparse->src[0] == nullptr || sparse->src[1] == nullptr ||
        sparse->src[2] == nullptr || sparse->src[3] == nullptr ||
        consumer->src[0] == nullptr || consumer->src[1] == nullptr ||
        consumer->src[2] == nullptr || consumer->src[3] == nullptr) {
        return false;
    }

    // The OpenCL sidecar only consumes block-index/matrix I32 masks. The
    // paired dense HTP node must use the normal F16 mask path.
    if (sparse->src[3]->type != GGML_TYPE_I32 ||
        consumer->src[3]->type != GGML_TYPE_F16) {
        return false;
    }

    for (int i = 0; i < 3; ++i) {
        if (root_of(const_cast<ggml_tensor *>(sparse->src[i])) !=
            root_of(const_cast<ggml_tensor *>(consumer->src[i]))) {
            return false;
        }
    }
    return true;
}

#endif
