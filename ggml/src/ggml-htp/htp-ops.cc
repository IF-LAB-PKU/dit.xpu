// HTP 算子分发层的实现。
// 对每个支持的 GGML 算子（RMS_NORM / MUL_MAT / FLASH_ATTN_EXT）：
// 1) 验证张量数据类型、形状、buffer 类型是否兼容 HTP 执行；
// 2) 将所有输入/输出张量解析为 (fd, offset) 形式的 RPC 内存地址；
// 3) 将算子参数打包到共享内存消息缓冲区（MessageHeader 环形缓冲）；
// 4) 通过原子标志位通知 DSP 侧开始计算，阻塞等待完成；
// 5) 可选地批量处理延迟的 fastrpc_munmap 请求以降低开销。

#include "htp-ops.h"

#include <cerrno>
#include <cstdlib>
#include <cstdio>
#include <dlfcn.h>
#include <unistd.h>

#include <atomic>
#include <cstring>
#include <vector>

#include "ggml-backend-impl.h"
#include "ggml-htp-impl.h"
#include "ggml-htp.h"
#include "ggml.h"
#include "message.h"
#include "op_reg.h"

namespace {

// HTP hybrid 路径里部分张量可能还是 view，这里统一回退到真正持有 buffer 的 backing tensor。
const ggml_tensor * get_tensor_with_backing_buffer(const ggml_tensor * tensor) {
    while (tensor != nullptr && tensor->buffer == nullptr && tensor->view_src != nullptr) {
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

const char * htp_op_index_name(int op_index) {
    switch (op_index) {
        case HTP_OPS_RMS_NORM_F32:
            return "HTP_OPS_RMS_NORM_F32";
        case HTP_OPS_MAT_MUL_ROW_MAJOR_W16A32:
            return "HTP_OPS_MAT_MUL_ROW_MAJOR_W16A32";
        case HTP_OPS_MAT_MUL_PERMUTED_W4D16A32:
            return "HTP_OPS_MAT_MUL_PERMUTED_W4D16A32";
        case HTP_OPS_MAT_MUL_PERMUTED_W8D16A32:
            return "HTP_OPS_MAT_MUL_PERMUTED_W8D16A32";
        case HTP_OPS_MAT_MUL_PERMUTED_W4D16A32_IQ4_NL:
            return "HTP_OPS_MAT_MUL_PERMUTED_W4D16A32_IQ4_NL";
        case HTP_OPS_FLASH_ATTN_QO_F32_KV_F16:
            return "HTP_OPS_FLASH_ATTN_QO_F32_KV_F16";
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

                // 当前 row-major matmul DSP kernel 只按二维 [m,k] activation 和 [m,n] dst 线性写入，带 batch 维的 ggml MUL_MAT 先不分发到 HTP。
                bool row_major_2d_layout_ok = ggml_nrows(dst) == dst->ne[1] &&
                                              ggml_nrows(activation) == activation->ne[1];
                bool tiled_shape_ok = k % 32 == 0 && n % 32 == 0 && row_major_2d_layout_ok;

                if (dst->type == GGML_TYPE_F32 && weight->type == GGML_TYPE_F16 && activation->type == GGML_TYPE_F32) {
                    return row_major_2d_layout_ok &&
                           dlsym(ops_dl_handle, "htp_ops_rpc_mat_mul_row_major_w16a32") != nullptr;
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

                return dst->type == GGML_TYPE_F32 && q->type == GGML_TYPE_F32 && k->type == GGML_TYPE_F16 &&
                       v->type == GGML_TYPE_F16 && mask->type == GGML_TYPE_F16 &&
                       max_bias == 0 && logit_softcap == 0;
            }
        default:
            return false;
    }
}

bool htp_ops_support_op_runtime(const struct ggml_tensor * dst) {
    if (!htp_ops_support_op(dst) || !tensor_ready_for_htp_dispatch(dst)) {
        return false;
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

int htp_ops_compute_op(struct ggml_compute_params * params, struct ggml_tensor * dst) {
    if (params->ith != 0) {
        return 0;
    }

    prepare_tensor_rpcmem_mapping(dst);

    auto * ctx           = ggml_backend_htp_context::instance();
    void * ops_dl_handle = ctx->ops_dl_handle;
    GGML_ASSERT(ops_dl_handle != nullptr);

    int op_index  = -1;
    int args_size = 0;
    int dispatch_m = 0;
    int dispatch_k = 0;
    int dispatch_n = 0;

    switch (dst->op) {
        case GGML_OP_RMS_NORM:
            {
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

                auto mappings = get_all_rpcmem_mappings(dst);
                if (mappings.size() != 3) {
                    dump_missing_mapping_diagnostics(dst, mappings);
                }
                GGML_ASSERT(mappings.size() == 3);

                auto [output_fd, output_offset]         = mappings[0];
                auto [weight_fd, weight_offset]         = mappings[1];
                auto [activation_fd, activation_offset] = mappings[2];

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
                    // sd.cpp 当前接入的 F16 权重仍是普通 row-major 存储，不能仅因为 k/n 是 32 的倍数就误走 permuted 权重内核，否则结果会直接失真成噪声图。
                    op_index = HTP_OPS_MAT_MUL_ROW_MAJOR_W16A32;
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
                auto * q = dst->src[0];
                auto * k = dst->src[1];

                auto mappings = get_all_rpcmem_mappings(dst);
                GGML_ASSERT(mappings.size() == 5);

                auto [o_fd, o_offset]       = mappings[0];
                auto [q_fd, q_offset]       = mappings[1];
                auto [k_fd, k_offset]       = mappings[2];
                auto [v_fd, v_offset]       = mappings[3];
                auto [mask_fd, mask_offset] = mappings[4];

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
                };
                *reinterpret_cast<FlashAttnParams *>(param_buf) = op_params;

                op_index  = HTP_OPS_FLASH_ATTN_QO_F32_KV_F16;
                args_size = sizeof(FlashAttnParams);
                dispatch_m = op_params.qo_len;
                dispatch_k = op_params.kv_len;
                dispatch_n = op_params.n_heads;
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
    const int ret = message_header_get_request_ptr(msg_hdr, 0)->state;
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
