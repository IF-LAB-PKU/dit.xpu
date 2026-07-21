// HTP 算子分发层的实现。
// 对每个支持的 GGML 算子（RMS_NORM / MUL_MAT / FLASH_ATTN_EXT）：
// 1) 验证张量数据类型、形状、buffer 类型是否兼容 HTP 执行；
// 2) 将所有输入/输出张量解析为 (fd, offset) 形式的 RPC 内存地址；
// 3) 将算子参数打包到共享内存消息缓冲区（MessageHeader 环形缓冲）；
// 4) 通过原子标志位通知 DSP 侧开始计算，阻塞等待完成；
// 5) 可选地批量处理延迟的 fastrpc_munmap 请求以降低开销。

#include "htp-ops.h"

#include <cerrno>
#include <cinttypes>
#include <cstdlib>
#include <cstdio>
#include <dlfcn.h>
#include <unistd.h>

#include <atomic>
#include <cstring>
#include <memory>
#include <tuple>
#include <unordered_set>
#include <vector>

#include "ggml-backend-impl.h"
#include "ggml-htp-impl.h"
#include "ggml-htp.h"
#include "ggml.h"
#include "message.h"
#include "op_reg.h"

#ifdef GGML_HTP_OPENCL_SIDECAR
#include "ggml-opencl-htp.h"
#endif

namespace {

// HTP hybrid 路径里部分张量可能还是 view，这里统一回退到真正持有 buffer 的 backing tensor。
const ggml_tensor * get_tensor_with_backing_buffer(const ggml_tensor * tensor) {
    while (tensor != nullptr && tensor->buffer == nullptr && tensor->view_src != nullptr) {
        tensor = tensor->view_src;
    }
    return tensor;
}

const ggml_tensor * get_tensor_backing_root(const ggml_tensor * tensor) {
    while (tensor != nullptr && tensor->view_src != nullptr) {
        tensor = tensor->view_src;
    }
    return tensor;
}

// hybrid 调度阶段只有真正已经落到 RPCMEM 且可直接寻址的张量，才允许进入 HTP 分发；否则必须立刻回退到 CPU，避免“支持 shape 但没做 placement”的节点撞到映射断言。
bool tensor_ready_for_htp_dispatch(const ggml_tensor * tensor) {
    const ggml_tensor * backing = get_tensor_with_backing_buffer(tensor);
    if (tensor == nullptr || backing == nullptr || backing->buffer == nullptr || !ggml_is_contiguous(tensor)) {
        return false;
    }
    return ggml_backend_buft_is_rpcmem(backing->buffer->buft);
}

bool tensor_ready_for_htp_dispatch_view_backed(const ggml_tensor * tensor) {
    const ggml_tensor * backing = get_tensor_backing_root(tensor);
    if (tensor == nullptr || backing == nullptr || backing->buffer == nullptr) {
        return false;
    }
    if (!ggml_backend_buft_is_rpcmem(backing->buffer->buft)) {
        return false;
    }
    return tensor->view_src != nullptr ? ggml_is_contiguous(backing) : ggml_is_contiguous(tensor);
}

// 出现映射数量不足时，把 tensor 本体与 backing root 的 buffer 状态都打出来，便于确认哪一路没有真正落到 RPCMEM。
void dump_tensor_rpcmem_state(const char * label, const ggml_tensor * tensor) {
    if (tensor == nullptr) {
        std::fprintf(stderr, "[ggml-htp-trace] %s: <null>\n", label);
        return;
    }

    const ggml_tensor * backing = get_tensor_with_backing_buffer(tensor);
    ggml_backend_buffer * tensor_buffer   = tensor->buffer;
    ggml_backend_buffer * backing_buffer  = backing != nullptr ? backing->buffer : nullptr;
    const bool tensor_is_rpcmem  = tensor_buffer != nullptr && ggml_backend_buft_is_rpcmem(tensor_buffer->buft);
    const bool backing_is_rpcmem = backing_buffer != nullptr && ggml_backend_buft_is_rpcmem(backing_buffer->buft);
    void * tensor_base  = tensor_buffer != nullptr ? ggml_backend_buffer_get_base(tensor_buffer) : nullptr;
    void * backing_base = backing_buffer != nullptr ? ggml_backend_buffer_get_base(backing_buffer) : nullptr;

    std::fprintf(stderr,
                 "[ggml-htp-trace] %s: name=%s op=%s type=%s data=%p buffer=%p tensor_rpcmem=%d view_src=%p "
                 "backing=%p backing_name=%s backing_buffer=%p backing_rpcmem=%d tensor_base=%p backing_base=%p "
                 "shape=[%lld,%lld,%lld,%lld]\n",
                 label,
                 tensor->name[0] != '\0' ? tensor->name : "<unnamed>",
                 ggml_op_name(tensor->op),
                 ggml_type_name(tensor->type),
                 tensor->data,
                 tensor_buffer,
                 tensor_is_rpcmem ? 1 : 0,
                 tensor->view_src,
                 backing,
                 (backing != nullptr && backing->name[0] != '\0') ? backing->name : "<unnamed>",
                 backing_buffer,
                 backing_is_rpcmem ? 1 : 0,
                 tensor_base,
                 backing_base,
                 (long long) tensor->ne[0],
                 (long long) tensor->ne[1],
                 (long long) tensor->ne[2],
                 (long long) tensor->ne[3]);
}

// 只在映射不完整时输出详细诊断，避免正常推理日志过于嘈杂。
void dump_missing_mapping_diagnostics(const ggml_tensor * dst,
                                      const std::vector<std::pair<int, ssize_t>> & mappings) {
    std::fprintf(stderr,
                 "[ggml-htp-trace] incomplete rpcmem mappings for dst=%s op=%s count=%zu\n",
                 dst != nullptr && dst->name[0] != '\0' ? dst->name : "<unnamed>",
                 dst != nullptr ? ggml_op_name(dst->op) : "<null>",
                 mappings.size());
    dump_tensor_rpcmem_state("dst", dst);
    if (dst != nullptr) {
        dump_tensor_rpcmem_state("src0", dst->src[0]);
        dump_tensor_rpcmem_state("src1", dst->src[1]);
        dump_tensor_rpcmem_state("src2", dst->src[2]);
    }
}

auto get_all_rpcmem_mappings(const ggml_tensor * dst) {
    const auto & mapper = ggml_backend_htp_context::instance()->mapper;

    std::vector<std::pair<int, ssize_t>> mappings;
    const ggml_tensor * mapped_dst = get_tensor_with_backing_buffer(dst);
    if (mapped_dst != nullptr && mapped_dst->buffer != nullptr && ggml_backend_buft_is_rpcmem(mapped_dst->buffer->buft)) {
        mappings.push_back(mapper.get_tensor_mapping(dst));
    }
    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        auto * src        = dst->src[i];
        auto * mapped_src = get_tensor_with_backing_buffer(src);
        if (mapped_src != nullptr && mapped_src->buffer != nullptr && ggml_backend_buft_is_rpcmem(mapped_src->buffer->buft)) {
            mappings.push_back(mapper.get_tensor_mapping(src));
        }
    }
    return mappings;
}

template <typename T>
void write_buf(uint8_t *& p, const T & v) {
    *reinterpret_cast<T *>(p) = v;
    p += sizeof(v);
}

void write_buf(uint8_t *& p, void * src, size_t size) {
    std::memcpy((void *) p, src, size);
    p += size;
}

uint8_t param_buf[4096];

// HTP message channel 是同步等待 DSP 回写完成标志，这里给等待加默认超时，避免某个算子落入异常慢路径后整次推理无声卡死。
int htp_dispatch_timeout_ms() {
    const char * env = std::getenv("GGML_HTP_OP_TIMEOUT_MS");
    if (env == nullptr || env[0] == '\0') {
        return 120000;
    }
    char * end = nullptr;
    long value = std::strtol(env, &end, 10);
    if (end == env || *end != '\0' || value < 0) {
        return 120000;
    }
    return static_cast<int>(value);
}

// 按需打开逐算子 dispatch 日志；超时时即使未打开 trace 也会打印同一组关键信息。
bool htp_trace_ops_enabled() {
    const char * env = std::getenv("GGML_HTP_TRACE_OPS");
    return env != nullptr && env[0] != '\0' && std::strcmp(env, "0") != 0;
}

bool env_flag_enabled(const char * name) {
    const char * env = std::getenv(name);
    return env != nullptr && env[0] != '\0' && std::strcmp(env, "0") != 0;
}

bool htp_debug_hash_enabled() {
    const char * env = std::getenv("GGML_HTP_DEBUG_HASH");
    return env != nullptr && env[0] != '\0' && std::strcmp(env, "0") != 0;
}

int htp_debug_hash_limit() {
    const char * env = std::getenv("GGML_HTP_DEBUG_HASH_LIMIT");
    if (env == nullptr || env[0] == '\0') {
        return 16;
    }
    char * end = nullptr;
    long value = std::strtol(env, &end, 10);
    if (end == env || *end != '\0' || value < 0) {
        return 16;
    }
    return static_cast<int>(value);
}

bool htp_debug_hash_filter_matches(const ggml_tensor * weight) {
    const char * filter = std::getenv("GGML_HTP_DEBUG_HASH_FILTER");
    if (filter == nullptr || filter[0] == '\0') {
        return true;
    }
    const char * name = weight != nullptr && weight->name[0] != '\0' ? weight->name : "";
    return std::strstr(name, filter) != nullptr;
}

uint64_t htp_fnv1a64_update(uint64_t hash, const void * data, size_t size) {
    const auto * bytes = static_cast<const uint8_t *>(data);
    for (size_t i = 0; i < size; ++i) {
        hash ^= bytes[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

uint64_t htp_hash_tensor_bytes(const ggml_tensor * tensor) {
    static constexpr uint64_t FNV1A64_OFFSET = UINT64_C(14695981039346656037);
    if (tensor == nullptr || tensor->data == nullptr) {
        return 0;
    }
    return htp_fnv1a64_update(FNV1A64_OFFSET, tensor->data, ggml_nbytes(tensor));
}

std::atomic<int> htp_debug_hash_counter{0};

bool is_block_sparse_score_custom(const ggml_tensor * dst) {
    if (dst == nullptr || dst->op != GGML_OP_CUSTOM || std::strcmp(dst->name, "block_sparse_score") != 0) {
        return false;
    }

    const ggml_tensor * q_mean = dst->src[0];
    const ggml_tensor * k_mean = dst->src[1];
    if (q_mean == nullptr || k_mean == nullptr || dst->src[2] != nullptr) {
        return false;
    }

    return dst->type == GGML_TYPE_F16 &&
           q_mean->type == GGML_TYPE_F16 &&
           k_mean->type == GGML_TYPE_F16 &&
           q_mean->ne[0] == k_mean->ne[0] &&
           q_mean->ne[2] == 1 &&
           k_mean->ne[2] == 1 &&
           dst->ne[0] == k_mean->ne[1] &&
           dst->ne[1] == q_mean->ne[1] &&
           ggml_is_contiguous(dst) &&
           ggml_is_contiguous(q_mean) &&
           ggml_is_contiguous(k_mean);
}

bool htp_attention_impl_is_normal() {
    const char * v = std::getenv("SD_HTP_ATTENTION_IMPL");
    return v != nullptr && std::strcmp(v, "normal") == 0;
}

const char * htp_op_index_name(int op_index) {
    switch (op_index) {
        case HTP_OPS_RMS_NORM_F32:
            return "HTP_OPS_RMS_NORM_F32";
        case HTP_OPS_MAT_MUL_PERMUTED_W16A32:
            return "HTP_OPS_MAT_MUL_PERMUTED_W16A32";
        case HTP_OPS_MAT_MUL_PERMUTED_W4D16A32:
            return "HTP_OPS_MAT_MUL_PERMUTED_W4D16A32";
        case HTP_OPS_MAT_MUL_PERMUTED_W8D16A32:
            return "HTP_OPS_MAT_MUL_PERMUTED_W8D16A32";
        case HTP_OPS_MAT_MUL_PERMUTED_W4D16A32_IQ4_NL:
            return "HTP_OPS_MAT_MUL_PERMUTED_W4D16A32_IQ4_NL";
        case HTP_OPS_FLASH_ATTN_QO_F32_KV_F16:
            return "HTP_OPS_FLASH_ATTN_QO_F32_KV_F16";
        case HTP_OPS_BLOCK_SPARSE_ATTN_QO_F32_KV_F16:
            return "HTP_OPS_BLOCK_SPARSE_ATTN_QO_F32_KV_F16";
        case HTP_OPS_BLOCK_SPARSE_SCORE_QO_F32_KM_F16:
            return "HTP_OPS_BLOCK_SPARSE_SCORE_QO_F32_KM_F16";
        default:
            return "HTP_OPS_UNKNOWN";
    }
}

void dump_htp_dispatch_state(const char * prefix,
                             const ggml_tensor * dst,
                             int op_index,
                             int dispatch_m,
                             int dispatch_k,
                             int dispatch_n,
                             int wait_ms) {
    std::fprintf(stderr,
                 "[ggml-htp-trace] %s op_index=%d(%s) ggml_op=%s tensor=%s type=%s "
                 "dispatch_mkn=[%d,%d,%d] waited_ms=%d shape=[%lld,%lld,%lld,%lld]\n",
                 prefix,
                 op_index,
                 htp_op_index_name(op_index),
                 dst != nullptr ? ggml_op_name(dst->op) : "<null>",
                 dst != nullptr && dst->name[0] != '\0' ? dst->name : "<unnamed>",
                 dst != nullptr ? ggml_type_name(dst->type) : "<null>",
                 dispatch_m,
                 dispatch_k,
                 dispatch_n,
                 wait_ms,
                 dst != nullptr ? (long long) dst->ne[0] : 0LL,
                 dst != nullptr ? (long long) dst->ne[1] : 0LL,
                 dst != nullptr ? (long long) dst->ne[2] : 0LL,
                 dst != nullptr ? (long long) dst->ne[3] : 0LL);
    if (dst != nullptr && dst->op == GGML_OP_MUL_MAT) {
        dump_tensor_rpcmem_state("dispatch_dst", dst);
        dump_tensor_rpcmem_state("dispatch_weight", dst->src[0]);
        dump_tensor_rpcmem_state("dispatch_activation", dst->src[1]);
    }
}

} // namespace

extern "C" {

bool htp_ops_support_op(const struct ggml_tensor * dst) {
    auto * ctx = ggml_backend_htp_context::instance();
    if (ctx->skip_htp_ops || !ctx->ops_backend_initialized) {
        return false;
    }

    void * ops_dl_handle = ctx->ops_dl_handle;
    GGML_ASSERT(ops_dl_handle != nullptr);

    switch (dst->op) {
        case GGML_OP_RMS_NORM:
            if (dst->type == GGML_TYPE_F32 && dst->src[0] != nullptr && dst->src[0]->type == GGML_TYPE_F32) {
                return dlsym(ops_dl_handle, "htp_ops_rpc_rms_norm_f32") != nullptr;
            }
            return false;
        case GGML_OP_MUL_MAT:
            {
                auto * weight     = dst->src[0];
                auto * activation = dst->src[1];
                if (weight == nullptr || activation == nullptr) {
                    return false;
                }

                size_t k = weight->ne[0];
                size_t n = weight->ne[1];

                // 当前 permuted matmul DSP kernel 只按二维 [m,k] activation 和 [m,n] dst 线性写入，带 batch 维的 ggml MUL_MAT 先不分发到 HTP。
                bool row_major_2d_layout_ok = ggml_nrows(dst) == dst->ne[1] &&
                                              ggml_nrows(activation) == activation->ne[1];
                bool tiled_shape_ok = k % 32 == 0 && n % 32 == 0 && row_major_2d_layout_ok;

                if (dst->type == GGML_TYPE_F32 && weight->type == GGML_TYPE_F16 && activation->type == GGML_TYPE_F32) {
                    return tiled_shape_ok &&
                           dlsym(ops_dl_handle, "htp_ops_rpc_mat_mul_permuted_w16a32") != nullptr;
                }
                if (dst->type == GGML_TYPE_F32 && weight->type == GGML_TYPE_Q4_0 && activation->type == GGML_TYPE_F32) {
                    return tiled_shape_ok;
                }
                if (dst->type == GGML_TYPE_F32 && weight->type == GGML_TYPE_Q8_0 && activation->type == GGML_TYPE_F32) {
                    return tiled_shape_ok;
                }
                if (dst->type == GGML_TYPE_F32 && weight->type == GGML_TYPE_IQ4_NL &&
                    activation->type == GGML_TYPE_F32) {
                    return tiled_shape_ok;
                }
                return false;
            }
        case GGML_OP_FLASH_ATTN_EXT:
            {
                float max_bias      = *reinterpret_cast<const float *>(&dst->op_params[1]);
                float logit_softcap = *reinterpret_cast<const float *>(&dst->op_params[2]);

                auto * q    = dst->src[0];
                auto * k    = dst->src[1];
                auto * v    = dst->src[2];
                auto * mask = dst->src[3];
                if (q == nullptr || k == nullptr || v == nullptr || mask == nullptr) {
                    return false;
                }

                const bool with_lse = ggml_get_op_params_i32(dst, 6) != 0;
                const ggml_flash_attn_ext_mask_format mask_format =
                    ggml_flash_attn_ext_get_mask_format(dst);
                const bool mask_format_ok =
                    (mask->type == GGML_TYPE_F16 && mask_format == GGML_FLASH_ATTN_EXT_MASK_DEFAULT) ||
                    (mask->type == GGML_TYPE_I32 &&
                     (mask_format == GGML_FLASH_ATTN_EXT_MASK_BLOCK_MATRIX ||
                      mask_format == GGML_FLASH_ATTN_EXT_MASK_BLOCK_INDEX));
                const bool shape_ok =
                    q->ne[0] == k->ne[0] &&
                    q->ne[0] == v->ne[0] &&
                    dst->ne[0] == v->ne[0] + (with_lse ? 1 : 0) &&
                    dst->ne[1] == q->ne[2] &&
                    dst->ne[2] == q->ne[1] &&
                    dst->ne[3] == q->ne[3];

                return dst->type == GGML_TYPE_F32 && q->type == GGML_TYPE_F32 && k->type == GGML_TYPE_F16 &&
                       v->type == GGML_TYPE_F16 && (mask->type == GGML_TYPE_F16 || mask->type == GGML_TYPE_I32) &&
                       mask_format_ok && shape_ok && max_bias == 0 && logit_softcap == 0;
            }
        case GGML_OP_CUSTOM:
            return is_block_sparse_score_custom(dst);
        default:
            return false;
    }
}

bool htp_ops_support_op_runtime(const struct ggml_tensor * dst) {
    if (!htp_ops_support_op(dst) || !tensor_ready_for_htp_dispatch(dst)) {
        return false;
    }

    if (dst->op == GGML_OP_FLASH_ATTN_EXT) {
        const ggml_tensor * mask = dst->src[3];
        const bool sparse_block_mask = mask != nullptr && mask->type == GGML_TYPE_I32;
        const bool dense_f16_mask = mask != nullptr && mask->type == GGML_TYPE_F16;
        if (sparse_block_mask || dense_f16_mask) {
            for (int i = 0; i < GGML_MAX_SRC; ++i) {
                const ggml_tensor * src = dst->src[i];
                if (src == nullptr) {
                    continue;
                }
                if (!tensor_ready_for_htp_dispatch_view_backed(src)) {
                    return false;
                }
            }
            return true;
        }
    }

    if (dst->op == GGML_OP_MUL_MAT) {
        const ggml_tensor * weight     = dst->src[0];
        const ggml_tensor * activation = dst->src[1];
        if (weight != nullptr && activation != nullptr &&
            dst->type == GGML_TYPE_F32 &&
            weight->type == GGML_TYPE_F16 &&
            activation->type == GGML_TYPE_F32) {
            ggml_tensor * packed_weight = ggml_backend_htp_get_packed_f16_weight(weight);
            return tensor_ready_for_htp_dispatch(activation) &&
                   tensor_ready_for_htp_dispatch(packed_weight);
        }
    }

    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        const ggml_tensor * src = dst->src[i];
        if (src == nullptr) {
            continue;
        }
        if (!tensor_ready_for_htp_dispatch(src)) {
            return false;
        }
    }

    return true;
}

bool htp_ops_opencl_routes_node(const struct ggml_tensor * dst) {
#ifdef GGML_HTP_OPENCL_SIDECAR
    const auto * ctx = ggml_backend_htp_context::instance();
    return dst != nullptr && ctx->sparse_opencl_routed_nodes.find(dst) !=
        ctx->sparse_opencl_routed_nodes.end();
#else
    GGML_UNUSED(dst);
    return false;
#endif
}

bool htp_ops_opencl_async_enabled(void) {
#ifdef GGML_HTP_OPENCL_SIDECAR
    const auto * ctx = ggml_backend_htp_context::instance();
    const char * value = std::getenv("SD_HTP_SPARSE_OPENCL_ASYNC");
    return ctx->sparse_opencl_graph != nullptr &&
           value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
#else
    return false;
#endif
}

#ifdef GGML_HTP_OPENCL_SIDECAR
struct htp_ops_opencl_early_submit_candidate {
    const ggml_tensor * node = nullptr;
    std::vector<const ggml_tensor *> producer_roots;
    std::unordered_set<const ggml_tensor *> completed_roots;
    bool submitted = false;
};

struct htp_ops_opencl_early_submit_state {
    std::vector<htp_ops_opencl_early_submit_candidate> candidates;
    std::unordered_set<const ggml_tensor *> prerequisites;
};

static std::unique_ptr<htp_ops_opencl_early_submit_state> g_opencl_early_submit;

static bool htp_ops_opencl_early_submit_enabled(void) {
    const char * value = std::getenv("SD_HTP_SPARSE_OPENCL_EARLY_SUBMIT");
    return htp_ops_opencl_async_enabled() && value != nullptr && value[0] != '\0' &&
           std::strcmp(value, "0") != 0;
}

static bool htp_ops_opencl_is_view_only(const ggml_tensor * tensor) {
    if (tensor == nullptr) {
        return false;
    }
    return tensor->op == GGML_OP_VIEW || tensor->op == GGML_OP_RESHAPE ||
           tensor->op == GGML_OP_PERMUTE || tensor->op == GGML_OP_TRANSPOSE;
}

static const ggml_tensor * htp_ops_opencl_early_submit_input_root(const ggml_tensor * tensor) {
    for (size_t depth = 0; tensor != nullptr && depth < 64; ++depth) {
        if (!htp_ops_opencl_is_view_only(tensor)) {
            return tensor;
        }
        tensor = tensor->src[0] != nullptr ? tensor->src[0] : tensor->view_src;
    }
    return nullptr;
}

static bool htp_ops_opencl_early_submit_candidate_ready(
        const htp_ops_opencl_early_submit_candidate & candidate) {
    return candidate.completed_roots.size() == candidate.producer_roots.size();
}

static bool htp_ops_opencl_early_submit_is_inputless_single_task_custom(
        const ggml_tensor * tensor) {
    if (tensor == nullptr || tensor->op != GGML_OP_CUSTOM) {
        return false;
    }
    for (int source = 0; source < GGML_MAX_SRC; ++source) {
        if (tensor->src[source] != nullptr) {
            return false;
        }
    }
    ggml_custom_op_params params = {};
    std::memcpy(&params, tensor->op_params, sizeof(params));
    return params.n_tasks == 1;
}

static void htp_ops_opencl_trace_early_submit(
        const ggml_tensor * node,
        const ggml_tensor * completed_node) {
    const char * trace = std::getenv("SD_HTP_SPARSE_OPENCL_TRACE");
    if (trace == nullptr || trace[0] == '\0' || std::strcmp(trace, "0") == 0) {
        return;
    }
    std::fprintf(
        stderr,
        "htp_opencl_event name=gpu_sparse_early_enqueue ts_us=%lld producer=%s ready_after=%s\n",
        (long long) ggml_time_us(),
        node != nullptr && node->name[0] != '\0' ? node->name : "<unnamed>",
        completed_node != nullptr && completed_node->name[0] != '\0'
            ? completed_node->name : "<graph-input>");
}

static int htp_ops_opencl_early_submit_ready_candidates(const ggml_tensor * completed_node) {
    GGML_ASSERT(g_opencl_early_submit != nullptr);
    for (htp_ops_opencl_early_submit_candidate & candidate : g_opencl_early_submit->candidates) {
        if (candidate.submitted || !htp_ops_opencl_early_submit_candidate_ready(candidate)) {
            continue;
        }
        htp_ops_opencl_trace_early_submit(candidate.node, completed_node);
        const int err = htp_ops_opencl_compute_node(const_cast<ggml_tensor *>(candidate.node));
        if (err != 0) {
            return err;
        }
        candidate.submitted = true;
    }
    return 0;
}
#endif

int htp_ops_opencl_early_submit_begin(const struct ggml_cgraph * graph) {
#ifdef GGML_HTP_OPENCL_SIDECAR
    if (!htp_ops_opencl_early_submit_enabled()) {
        return 0;
    }
    if (graph == nullptr || g_opencl_early_submit != nullptr) {
        return -EINVAL;
    }

    auto state = std::make_unique<htp_ops_opencl_early_submit_state>();
    const auto * ctx = ggml_backend_htp_context::instance();
    state->candidates.reserve(ctx->sparse_opencl_routed_nodes.size());
    for (int node_index = 0; node_index < graph->n_nodes; ++node_index) {
        const ggml_tensor * node = graph->nodes[node_index];
        if (node == nullptr || ctx->sparse_opencl_routed_nodes.find(node) ==
                                   ctx->sparse_opencl_routed_nodes.end()) {
            continue;
        }

        htp_ops_opencl_early_submit_candidate candidate = {};
        candidate.node = node;
        for (int source = 0; source < GGML_MAX_SRC; ++source) {
            const ggml_tensor * root =
                htp_ops_opencl_early_submit_input_root(node->src[source]);
            if (root == nullptr || root->op == GGML_OP_NONE) {
                continue;
            }
            bool duplicate = false;
            for (const ggml_tensor * existing : candidate.producer_roots) {
                duplicate = duplicate || existing == root;
            }
            if (!duplicate) {
                candidate.producer_roots.push_back(root);
                if (htp_ops_opencl_early_submit_is_inputless_single_task_custom(root)) {
                    state->prerequisites.insert(root);
                }
            }
        }
        state->candidates.push_back(std::move(candidate));
    }

    g_opencl_early_submit = std::move(state);
    const int err = htp_ops_opencl_early_submit_ready_candidates(nullptr);
    if (err != 0) {
        g_opencl_early_submit.reset();
    }
    return err;
#else
    GGML_UNUSED(graph);
    return 0;
#endif
}

void htp_ops_opencl_early_submit_end(void) {
#ifdef GGML_HTP_OPENCL_SIDECAR
    g_opencl_early_submit.reset();
#endif
}

bool htp_ops_opencl_early_submit_tracks_node(const struct ggml_tensor * node) {
#ifdef GGML_HTP_OPENCL_SIDECAR
    if (g_opencl_early_submit == nullptr || node == nullptr) {
        return false;
    }
    for (const htp_ops_opencl_early_submit_candidate & candidate :
             g_opencl_early_submit->candidates) {
        if (candidate.submitted) {
            continue;
        }
        for (const ggml_tensor * root : candidate.producer_roots) {
            if (root == node) {
                return true;
            }
        }
    }
#else
    GGML_UNUSED(node);
#endif
    return false;
}

bool htp_ops_opencl_early_submit_is_prerequisite(const struct ggml_tensor * node) {
#ifdef GGML_HTP_OPENCL_SIDECAR
    return g_opencl_early_submit != nullptr && node != nullptr &&
           g_opencl_early_submit->prerequisites.find(node) !=
               g_opencl_early_submit->prerequisites.end();
#else
    GGML_UNUSED(node);
    return false;
#endif
}

int htp_ops_opencl_early_submit_node_completed(const struct ggml_tensor * node) {
#ifdef GGML_HTP_OPENCL_SIDECAR
    if (g_opencl_early_submit == nullptr || node == nullptr) {
        return 0;
    }
    for (htp_ops_opencl_early_submit_candidate & candidate :
             g_opencl_early_submit->candidates) {
        if (candidate.submitted) {
            continue;
        }
        for (const ggml_tensor * root : candidate.producer_roots) {
            if (root == node) {
                candidate.completed_roots.insert(root);
            }
        }
    }
    return htp_ops_opencl_early_submit_ready_candidates(node);
#else
    GGML_UNUSED(node);
    return 0;
#endif
}

bool htp_ops_opencl_early_submit_was_submitted(const struct ggml_tensor * node) {
#ifdef GGML_HTP_OPENCL_SIDECAR
    if (g_opencl_early_submit == nullptr || node == nullptr) {
        return false;
    }
    for (const htp_ops_opencl_early_submit_candidate & candidate :
             g_opencl_early_submit->candidates) {
        if (candidate.node == node) {
            return candidate.submitted;
        }
    }
#else
    GGML_UNUSED(node);
#endif
    return false;
}

void htp_ops_opencl_early_submit_mark_submitted(const struct ggml_tensor * node) {
#ifdef GGML_HTP_OPENCL_SIDECAR
    if (g_opencl_early_submit == nullptr || node == nullptr) {
        return;
    }
    for (htp_ops_opencl_early_submit_candidate & candidate :
             g_opencl_early_submit->candidates) {
        if (candidate.node == node) {
            candidate.submitted = true;
            return;
        }
    }
#else
    GGML_UNUSED(node);
#endif
}

#ifdef GGML_HTP_OPENCL_SIDECAR

static const ggml_tensor * htp_ops_opencl_routed_producer(
        const ggml_backend_htp_context * ctx,
        const ggml_tensor * tensor) {
    for (size_t depth = 0; tensor != nullptr && depth < 64; ++depth) {
        if (ctx->sparse_opencl_routed_nodes.find(tensor) !=
            ctx->sparse_opencl_routed_nodes.end()) {
            return tensor;
        }
        if (!htp_ops_opencl_is_view_only(tensor)) {
            return nullptr;
        }
        tensor = tensor->src[0] != nullptr ? tensor->src[0] : tensor->view_src;
    }
    return nullptr;
}

static size_t htp_ops_opencl_collect_producers(
        const ggml_tensor * dst,
        const ggml_tensor ** producers,
        size_t capacity) {
    if (dst == nullptr || htp_ops_opencl_is_view_only(dst)) {
        return 0;
    }
    const auto * ctx = ggml_backend_htp_context::instance();
    size_t count = 0;
    for (int source = 0; source < GGML_MAX_SRC; ++source) {
        const ggml_tensor * producer = htp_ops_opencl_routed_producer(ctx, dst->src[source]);
        if (producer == nullptr) {
            continue;
        }
        bool duplicate = false;
        for (size_t i = 0; i < count; ++i) {
            duplicate = duplicate || producers[i] == producer;
        }
        if (!duplicate && count < capacity) {
            producers[count++] = producer;
        }
    }
    return count;
}
#endif

bool htp_ops_opencl_node_needs_wait(const struct ggml_tensor * dst) {
#ifdef GGML_HTP_OPENCL_SIDECAR
    if (!htp_ops_opencl_async_enabled()) {
        return false;
    }
    const ggml_tensor * producers[GGML_MAX_SRC] = {};
    return htp_ops_opencl_collect_producers(dst, producers, GGML_MAX_SRC) != 0;
#else
    GGML_UNUSED(dst);
    return false;
#endif
}

int htp_ops_opencl_wait_node(const struct ggml_tensor * dst) {
#ifdef GGML_HTP_OPENCL_SIDECAR
    auto * ctx = ggml_backend_htp_context::instance();
    if (!htp_ops_opencl_async_enabled() || dst == nullptr) {
        return 0;
    }

    const ggml_tensor * producers[GGML_MAX_SRC] = {};
    const size_t producer_count =
        htp_ops_opencl_collect_producers(dst, producers, GGML_MAX_SRC);
    bool waited_any = false;
    for (size_t i = 0; i < producer_count; ++i) {
        bool waited = false;
        char reason[1024] = {};
        const ggml_opencl_htp_result result =
            ggml_backend_opencl_htp_graph_wait_for_tensor(
                ctx->sparse_opencl_graph, producers[i], &waited,
                reason, sizeof(reason));
        if (result != GGML_OPENCL_HTP_OK) {
            std::fprintf(
                stderr,
                "ggml-htp: OpenCL sparse dependency wait failed result=%d reason=%s\n",
                (int) result, reason[0] != '\0' ? reason : "unknown error");
            return -EIO;
        }
        waited_any = waited_any || waited;
    }

    GGML_UNUSED(waited_any);
    return 0;
#else
    GGML_UNUSED(dst);
    return -ENOSYS;
#endif
}

int htp_ops_opencl_compute_node(struct ggml_tensor * dst) {
#ifdef GGML_HTP_OPENCL_SIDECAR
    auto * ctx = ggml_backend_htp_context::instance();
    if (ctx->sparse_opencl_graph == nullptr || dst == nullptr ||
        ctx->sparse_opencl_routed_nodes.find(dst) == ctx->sparse_opencl_routed_nodes.end()) {
        return -EINVAL;
    }

    const int64_t begin_ms = ggml_time_ms();
    char reason[1024] = {};
    const bool async = htp_ops_opencl_async_enabled();
    const ggml_opencl_htp_result result = async
        ? ggml_backend_opencl_htp_graph_compute_block_sparse_attn_async(
              ctx->sparse_opencl_graph, dst, reason, sizeof(reason))
        : ggml_backend_opencl_htp_graph_compute_block_sparse_attn_sync(
              ctx->sparse_opencl_graph, dst, reason, sizeof(reason));
    const char * trace = std::getenv("SD_HTP_SPARSE_OPENCL_TRACE");
    if (trace != nullptr && trace[0] != '\0' && std::strcmp(trace, "0") != 0) {
        std::fprintf(
            stderr,
            "attention_timing route=opencl mask_type=i32 total=%lldms qo=%lld kv=%lld "
            "heads=%lld d=%lld with_lse=%d out_d=%lld block_sparse=1 async=%d result=%d\n",
            (long long) (ggml_time_ms() - begin_ms),
            (long long) dst->src[0]->ne[1],
            (long long) dst->src[1]->ne[1],
            (long long) dst->src[0]->ne[2],
            (long long) dst->src[0]->ne[0],
            ggml_get_op_params_i32(dst, 6) != 0 ? 1 : 0,
            (long long) dst->ne[0],
            async ? 1 : 0,
            (int) result);
    }
    if (result != GGML_OPENCL_HTP_OK) {
        std::fprintf(
            stderr, "ggml-htp: OpenCL sparse attention failed result=%d reason=%s\n",
            (int) result, reason[0] != '\0' ? reason : "unknown error");
        return -EIO;
    }
    return 0;
#else
    GGML_UNUSED(dst);
    return -ENOSYS;
#endif
}

int htp_ops_compute_op(struct ggml_compute_params * params, struct ggml_tensor * dst) {
    if (params->ith != 0) {
        return 0;
    }

    auto * ctx           = ggml_backend_htp_context::instance();
    void * ops_dl_handle = ctx->ops_dl_handle;
    GGML_ASSERT(ops_dl_handle != nullptr);

    int op_index  = -1;
    int args_size = 0;
    int dispatch_m = 0;
    int dispatch_k = 0;
    int dispatch_n = 0;
    const ggml_tensor * debug_weight = nullptr;
    const ggml_tensor * debug_activation = nullptr;
    const ggml_tensor * debug_packed_weight = nullptr;
    bool debug_f16_matmul = false;

    switch (dst->op) {
        case GGML_OP_RMS_NORM:
            {
                prepare_tensor_rpcmem_mapping(dst);

                auto mappings = get_all_rpcmem_mappings(dst);
                GGML_ASSERT(mappings.size() == 2);

                auto [dst_fd, dst_offset] = mappings[0];
                auto [src_fd, src_offset] = mappings[1];

                RmsNormF32Params op_params{
                    .dst = { dst_fd, (int32_t) dst_offset },
                    .src = { src_fd, (int32_t) src_offset },
                    .ne0 = (int32_t) dst->ne[0],
                    .ne1 = (int32_t) ggml_nrows(dst),
                };
                *reinterpret_cast<RmsNormF32Params *>(param_buf) = op_params;

                op_index  = HTP_OPS_RMS_NORM_F32;
                args_size = sizeof(RmsNormF32Params);
                dispatch_m = op_params.ne1;
                dispatch_k = op_params.ne0;
                dispatch_n = 1;
            }
            break;

        case GGML_OP_MUL_MAT:
            {
                auto * weight     = dst->src[0];
                auto * activation = dst->src[1];
                ggml_tensor * packed_weight = nullptr;

                const bool is_f16 = dst->type == GGML_TYPE_F32 &&
                                    weight != nullptr && weight->type == GGML_TYPE_F16 &&
                                    activation != nullptr && activation->type == GGML_TYPE_F32;

                if (is_f16) {
                    // F16 路径的 DSP kernel 只接受 permuted W16A32 权重。src[0] 可能是普通 ggml weight，
                    // 因此必须从注册表取出“已确认 packed”的 tensor；离线 packed GGUF 下这里会返回 weight 本体。
                    packed_weight = ggml_backend_htp_get_packed_f16_weight(weight);
                    if (packed_weight == nullptr) {
                        std::fprintf(stderr,
                                     "[ggml-htp-trace] missing packed F16 weight for tensor=%s\n",
                                     weight != nullptr && weight->name[0] != '\0' ? weight->name : "<unnamed>");
                        return -EINVAL;
                    }
                    prepare_tensor_rpcmem_mapping_with_extra(dst, packed_weight);
                } else {
                    prepare_tensor_rpcmem_mapping(dst);
                }

                int output_fd = -1;
                ssize_t output_offset = 0;
                int weight_fd = -1;
                ssize_t weight_offset = 0;
                int activation_fd = -1;
                ssize_t activation_offset = 0;

                if (is_f16) {
                    auto & mapper = ctx->mapper;
                    std::tie(output_fd, output_offset)         = mapper.get_tensor_mapping(dst);
                    std::tie(weight_fd, weight_offset)         = mapper.get_tensor_mapping(packed_weight);
                    std::tie(activation_fd, activation_offset) = mapper.get_tensor_mapping(activation);
                } else {
                    auto mappings = get_all_rpcmem_mappings(dst);
                    if (mappings.size() != 3) {
                        dump_missing_mapping_diagnostics(dst, mappings);
                    }
                    GGML_ASSERT(mappings.size() == 3);

                    std::tie(output_fd, output_offset)         = mappings[0];
                    std::tie(weight_fd, weight_offset)         = mappings[1];
                    std::tie(activation_fd, activation_offset) = mappings[2];
                }

                MatMulParams op_params{
                    .output     = { output_fd,     (int32_t) output_offset     },
                    .activation = { activation_fd, (int32_t) activation_offset },
                    .weight     = { weight_fd,     (int32_t) weight_offset     },
                    .m          = (int32_t) ggml_nrows(activation),
                    .k          = (int32_t) weight->ne[0],
                    .n          = (int32_t) weight->ne[1],
                };
                *reinterpret_cast<MatMulParams *>(param_buf) = op_params;

                args_size = sizeof(MatMulParams);
                dispatch_m = op_params.m;
                dispatch_k = op_params.k;
                dispatch_n = op_params.n;

                if (dst->type == GGML_TYPE_F32 && weight->type == GGML_TYPE_F16 && activation->type == GGML_TYPE_F32) {
                    op_index = HTP_OPS_MAT_MUL_PERMUTED_W16A32;
                    debug_f16_matmul = true;
                    debug_weight = weight;
                    debug_activation = activation;
                    debug_packed_weight = packed_weight;
                } else if (dst->type == GGML_TYPE_F32 && weight->type == GGML_TYPE_Q4_0 && activation->type == GGML_TYPE_F32) {
                    op_index = HTP_OPS_MAT_MUL_PERMUTED_W4D16A32;
                } else if (dst->type == GGML_TYPE_F32 && weight->type == GGML_TYPE_Q8_0 && activation->type == GGML_TYPE_F32) {
                    op_index = HTP_OPS_MAT_MUL_PERMUTED_W8D16A32;
                } else if (dst->type == GGML_TYPE_F32 && weight->type == GGML_TYPE_IQ4_NL && activation->type == GGML_TYPE_F32) {
                    op_index = HTP_OPS_MAT_MUL_PERMUTED_W4D16A32_IQ4_NL;
                } else {
                    GGML_ASSERT(false && "not implemented");
                }
            }
            break;

        case GGML_OP_FLASH_ATTN_EXT:
            {
                prepare_tensor_rpcmem_mapping(dst);

                auto * q = dst->src[0];
                auto * k = dst->src[1];

                auto mappings = get_all_rpcmem_mappings(dst);
                GGML_ASSERT(mappings.size() == 5);

                auto [o_fd, o_offset]       = mappings[0];
                auto [q_fd, q_offset]       = mappings[1];
                auto [k_fd, k_offset]       = mappings[2];
                auto [v_fd, v_offset]       = mappings[3];
                auto [mask_fd, mask_offset] = mappings[4];

                auto * mask = dst->src[3];
                const bool with_lse = ggml_get_op_params_i32(dst, 6) != 0;
                if (mask->type == GGML_TYPE_I32) {
                    const int32_t block_q = ggml_get_op_params_i32(dst, 4);
                    const int32_t block_k = ggml_get_op_params_i32(dst, 5);
                    GGML_ASSERT(block_q > 0 && block_k > 0);
                    const ggml_flash_attn_ext_mask_format mask_format =
                        ggml_flash_attn_ext_get_mask_format(dst);
                    GGML_ASSERT(mask_format == GGML_FLASH_ATTN_EXT_MASK_BLOCK_MATRIX ||
                                mask_format == GGML_FLASH_ATTN_EXT_MASK_BLOCK_INDEX);
                    const bool is_index = mask_format == GGML_FLASH_ATTN_EXT_MASK_BLOCK_INDEX;
                    const int32_t computed_n_q_blocks =
                        (int32_t) q->ne[2] * (int32_t) ((q->ne[1] + block_q - 1) / block_q);
                    const int32_t computed_n_k_blocks =
                        (int32_t) k->ne[2] * (int32_t) ((k->ne[1] + block_k - 1) / block_k);
                    GGML_ASSERT((int32_t) mask->ne[1] == computed_n_q_blocks);
                    if (!is_index) {
                        GGML_ASSERT((int32_t) mask->ne[0] == computed_n_k_blocks);
                    }

                    BlockSparseAttnParams op_params{
                        .o          = { o_fd,    (int32_t) o_offset    },
                        .q          = { q_fd,    (int32_t) q_offset    },
                        .k          = { k_fd,    (int32_t) k_offset    },
                        .v          = { v_fd,    (int32_t) v_offset    },
                        .block_mask = { mask_fd, (int32_t) mask_offset },
                        .qo_len     = (int32_t) q->ne[1],
                        .kv_len     = (int32_t) k->ne[1],
                        .n_heads    = (int32_t) q->ne[2],
                        .n_kv_heads = (int32_t) k->ne[2],
                        .head_dim   = (int32_t) q->ne[0],
                        .block_q    = block_q,
                        .block_k    = block_k,
                        .n_q_blocks = is_index ? computed_n_q_blocks : (int32_t) mask->ne[1],
                        .n_k_blocks = computed_n_k_blocks,
                        .mask_format = is_index ? 1 : 0,
                        .max_kept_blocks = is_index ? (int32_t) mask->ne[0] : 0,
                        .with_lse = with_lse ? 1 : 0,
                        .elapsed_us = 0,
                    };
                    *reinterpret_cast<BlockSparseAttnParams *>(param_buf) = op_params;

                    op_index   = HTP_OPS_BLOCK_SPARSE_ATTN_QO_F32_KV_F16;
                    args_size  = sizeof(BlockSparseAttnParams);
                    dispatch_m = op_params.qo_len;
                    dispatch_k = op_params.kv_len;
                    dispatch_n = op_params.n_heads;
                } else {
                    FlashAttnParams op_params{
                        .o          = { o_fd,    (int32_t) o_offset    },
                        .q          = { q_fd,    (int32_t) q_offset    },
                        .k          = { k_fd,    (int32_t) k_offset    },
                        .v          = { v_fd,    (int32_t) v_offset    },
                        .mask       = { mask_fd, (int32_t) mask_offset },
                        .qo_len     = (int32_t) q->ne[1],
                        .kv_len     = (int32_t) k->ne[1],
                        .n_heads    = (int32_t) q->ne[2],
                        .n_kv_heads = (int32_t) k->ne[2],
                        .head_dim   = (int32_t) q->ne[0],
                        .impl       = htp_attention_impl_is_normal() ? 1 : 0,
                        .with_lse   = with_lse ? 1 : 0,
                        .elapsed_us = 0,
                    };
                    *reinterpret_cast<FlashAttnParams *>(param_buf) = op_params;

                    op_index   = HTP_OPS_FLASH_ATTN_QO_F32_KV_F16;
                    args_size  = sizeof(FlashAttnParams);
                    dispatch_m = op_params.qo_len;
                    dispatch_k = op_params.kv_len;
                    dispatch_n = op_params.n_heads;
                }
            }
            break;

        case GGML_OP_CUSTOM:
            {
                if (!is_block_sparse_score_custom(dst)) {
                    break;
                }

                prepare_tensor_rpcmem_mapping(dst);

                auto * q_mean = dst->src[0];
                auto * k_mean = dst->src[1];

                auto mappings = get_all_rpcmem_mappings(dst);
                if (mappings.size() != 3) {
                    dump_missing_mapping_diagnostics(dst, mappings);
                }
                GGML_ASSERT(mappings.size() == 3);

                auto [score_fd, score_offset] = mappings[0];
                auto [q_fd, q_offset]         = mappings[1];
                auto [k_fd, k_offset]         = mappings[2];

                BlockSparseScoreParams op_params{
                    .score      = { score_fd, (int32_t) score_offset },
                    .q_mean     = { q_fd,     (int32_t) q_offset     },
                    .k_mean     = { k_fd,     (int32_t) k_offset     },
                    .head_dim   = (int32_t) q_mean->ne[0],
                    .n_q_blocks = (int32_t) q_mean->ne[1],
                    .n_k_blocks = (int32_t) k_mean->ne[1],
                };
                *reinterpret_cast<BlockSparseScoreParams *>(param_buf) = op_params;

                op_index   = HTP_OPS_BLOCK_SPARSE_SCORE_QO_F32_KM_F16;
                args_size  = sizeof(BlockSparseScoreParams);
                dispatch_m = op_params.n_q_blocks;
                dispatch_k = op_params.n_k_blocks;
                dispatch_n = op_params.head_dim;
            }
            break;

        default:
            break;
    }

    // 理论上 htp_ops_compute_op 只会被已支持算子调用；若前面没有成功选出 HTP op，直接返回错误而不是发一个无效 op 到 DSP。
    if (op_index < 0 || args_size <= 0) {
        std::fprintf(stderr,
                     "[ggml-htp-trace] unsupported dispatch op=%s tensor=%s\n",
                     ggml_op_name(dst->op),
                     dst->name[0] != '\0' ? dst->name : "<unnamed>");
        return -EINVAL;
    }

    int  n_reqs         = 1;
    int  n_unmap_fds    = (int) ctx->mapper.get_pending_unmap_reqs().size();
    bool has_unmap_reqs = n_unmap_fds > 0;
    if (has_unmap_reqs) {
        ++n_reqs;
    }

    size_t op_req_size = sizeof(RequestHeader) + sizeof(OpComputeRequest) + args_size;
    auto * msg_hdr = reinterpret_cast<MessageHeader *>(ctx->ops_msg_chan);

    auto * d_ptr = reinterpret_cast<volatile std::atomic<uint64_t> *>(&(msg_hdr->state.d));
    std::atomic_store(d_ptr, uint64_t{ 0 });

    msg_hdr->n_reqs         = n_reqs;
    msg_hdr->req_offsets[0] = message_header_size(msg_hdr);
    msg_hdr->req_offsets[1] = msg_hdr->req_offsets[0] + op_req_size;

    {
        RequestHeader req_hdr{
            .state = 0,
            .type  = REQUEST_TYPE_OP_COMPUTE,
        };
        OpComputeRequest op_req{
            .op = (uint32_t) op_index,
        };

        auto * p = reinterpret_cast<uint8_t *>(message_header_get_request_ptr(msg_hdr, 0));
        write_buf(p, req_hdr);
        write_buf(p, op_req);
        write_buf(p, param_buf, args_size);
    }

    if (has_unmap_reqs) {
        size_t map_req_size     = sizeof(RequestHeader) + sizeof(RpcmemMapRequest) + n_unmap_fds * sizeof(int32_t);
        msg_hdr->req_offsets[2] = msg_hdr->req_offsets[1] + map_req_size;

        RequestHeader req_hdr{
            .state = 0,
            .type  = REQUEST_TYPE_RPCMEM_MAP,
        };
        RpcmemMapRequest map_req{
            .n_puts = n_unmap_fds,
            .n_gets = 0,
        };

        auto * p = reinterpret_cast<uint8_t *>(message_header_get_request_ptr(msg_hdr, 1));
        write_buf(p, req_hdr);
        write_buf(p, map_req);
        for (const auto & req : ctx->mapper.get_pending_unmap_reqs()) {
            write_buf(p, req.fd);
        }
    }

    {
        uint32_t sum = 0;
        uint32_t * begin = ((uint32_t *) msg_hdr) + 3;
        uint32_t * end   = ((uint32_t *) msg_hdr) + ggml_backend_htp_context::MAX_MSG_SIZE / 4;
        for (auto * p = begin; p < end; ++p) {
            sum += *p;
        }
        sum += 0x00000001 + 0x00000000;
        msg_hdr->checksum = -sum;
    }

    auto * v0_ptr = reinterpret_cast<volatile std::atomic<uint8_t> *>(&(msg_hdr->state.v[0]));
    auto * v1_ptr = reinterpret_cast<volatile std::atomic<uint8_t> *>(&(msg_hdr->state.v[1]));
    if (htp_trace_ops_enabled()) {
        dump_htp_dispatch_state("dispatch-begin", dst, op_index, dispatch_m, dispatch_k, dispatch_n, 0);
    }

    int debug_hash_index = -1;
    uint64_t debug_activation_hash = 0;
    if (debug_f16_matmul && htp_debug_hash_enabled() && htp_debug_hash_filter_matches(debug_weight)) {
        const int next_index = htp_debug_hash_counter.fetch_add(1);
        if (next_index < htp_debug_hash_limit()) {
            debug_hash_index = next_index;
            debug_activation_hash = htp_hash_tensor_bytes(debug_activation);
        }
    }

    std::atomic_store_explicit(v0_ptr, uint8_t{ 1 }, std::memory_order_release);

    const int timeout_ms = htp_dispatch_timeout_ms();
    const int64_t wait_begin_ms = ggml_time_ms();
    while (std::atomic_load_explicit(v1_ptr, std::memory_order_acquire) == 0) {
        const int waited_ms = static_cast<int>(ggml_time_ms() - wait_begin_ms);
        if (timeout_ms > 0 && waited_ms >= timeout_ms) {
            dump_htp_dispatch_state("dispatch-timeout", dst, op_index, dispatch_m, dispatch_k, dispatch_n, waited_ms);
            d_ptr->store(0, std::memory_order_relaxed);
            if (has_unmap_reqs) {
                ctx->mapper.unmap_all_pending_buffers();
            }
            return -ETIMEDOUT;
        }
        usleep(1000);
    }
    d_ptr->store(0, std::memory_order_relaxed);

    if (has_unmap_reqs) {
        ctx->mapper.unmap_all_pending_buffers();
    }

    std::atomic_thread_fence(std::memory_order_acquire);
    auto * completed_request = message_header_get_request_ptr(msg_hdr, 0);
    const int ret = completed_request->state;
    const char * internal_trace = std::getenv("SD_HTP_ATTN_INTERNAL_TRACE");
    if (internal_trace != nullptr && internal_trace[0] != '\0' && std::strcmp(internal_trace, "0") != 0 &&
        (op_index == HTP_OPS_FLASH_ATTN_QO_F32_KV_F16 ||
         op_index == HTP_OPS_BLOCK_SPARSE_ATTN_QO_F32_KV_F16)) {
        auto * compute_request = reinterpret_cast<OpComputeRequest *>(completed_request->data);
        const int32_t elapsed_us = op_index == HTP_OPS_FLASH_ATTN_QO_F32_KV_F16
            ? reinterpret_cast<const FlashAttnParams *>(compute_request->payload)->elapsed_us
            : reinterpret_cast<const BlockSparseAttnParams *>(compute_request->payload)->elapsed_us;
        std::fprintf(
            stderr,
            "htp_attention_internal kind=%s elapsed_us=%d qo=%d kv=%d heads=%d with_lse=%d\n",
            op_index == HTP_OPS_FLASH_ATTN_QO_F32_KV_F16 ? "dense" : "block_sparse",
            elapsed_us,
            dispatch_m,
            dispatch_k,
            dispatch_n,
            ggml_get_op_params_i32(dst, 6) != 0 ? 1 : 0);
    }
    if (debug_hash_index >= 0) {
        const uint64_t output_hash = htp_hash_tensor_bytes(dst);
        std::fprintf(stderr,
                     "[ggml-htp-hash] idx=%d ret=%d weight=%s packed=%s self_packed=%d mkn=[%d,%d,%d] "
                     "act_bytes=%zu act_hash=%016" PRIx64 " out_bytes=%zu out_hash=%016" PRIx64 "\n",
                     debug_hash_index,
                     ret,
                     debug_weight != nullptr && debug_weight->name[0] != '\0' ? debug_weight->name : "<unnamed>",
                     debug_packed_weight != nullptr && debug_packed_weight->name[0] != '\0' ? debug_packed_weight->name : "<unnamed>",
                     debug_weight == debug_packed_weight ? 1 : 0,
                     dispatch_m,
                     dispatch_k,
                     dispatch_n,
                     debug_activation != nullptr ? ggml_nbytes(debug_activation) : 0,
                     debug_activation_hash,
                     dst != nullptr ? ggml_nbytes(dst) : 0,
                     output_hash);
    }
    if (op_index == HTP_OPS_BLOCK_SPARSE_SCORE_QO_F32_KM_F16) {
        const char * trace = std::getenv("SD_HTP_BLOCK_SPARSE_MASK_TRACE");
        if (trace != nullptr && trace[0] != '\0' && std::strcmp(trace, "0") != 0) {
            std::fprintf(stderr,
                         "block_sparse_score_step score=%dms n_q_blocks=%d n_k_blocks=%d head_dim=%d ret=%d\n",
                         static_cast<int>(ggml_time_ms() - wait_begin_ms),
                         dispatch_m,
                         dispatch_k,
                         dispatch_n,
                         ret);
        }
    }
    if (ret != 0 || htp_trace_ops_enabled()) {
        dump_htp_dispatch_state(ret == 0 ? "dispatch-end" : "dispatch-error",
                                dst,
                                op_index,
                                dispatch_m,
                                dispatch_k,
                                dispatch_n,
                                static_cast<int>(ggml_time_ms() - wait_begin_ms));
    }
    return ret;
}

}
