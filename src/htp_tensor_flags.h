#pragma once

#include <cstdint>

static constexpr int32_t SD_GGML_TENSOR_FLAG_HTP_MUL_MAT_WEIGHT = 1 << 24;
static constexpr int32_t SD_GGML_TENSOR_FLAG_HTP_PACKED_W16A32  = 1 << 25;
static constexpr int32_t SD_GGML_TENSOR_FLAG_HTP_PACKED_QKV_PARAM = 1 << 26;
static constexpr int32_t SD_GGML_TENSOR_FLAG_HTP_DEFERRED_PACKED_PARAM = 1 << 27;
// Logical shape only; storage is the bound native INT8 constant tensor.
static constexpr int32_t SD_GGML_TENSOR_FLAG_HTP_W8A8_REPLACED = 1 << 28;
