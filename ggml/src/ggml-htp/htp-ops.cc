// HTP 算子分发层的实现。
// 对每个支持的 GGML 算子（RMS_NORM / MUL_MAT / FLASH_ATTN_EXT）：
// 1) 验证张量数据类型、形状、buffer 类型是否兼容 HTP 执行；
// 2) 将所有输入/输出张量解析为 (fd, offset) 形式的 RPC 内存地址；
// 3) 将算子参数打包到共享内存消息缓冲区（MessageHeader 环形缓冲）；
// 4) 通过原子标志位通知 DSP 侧开始计算，阻塞等待完成；
// 5) 可选地批量处理延迟的 DSP put + FastRPC munmap 请求。

#include "htp-ops.h"

#include <cerrno>
#include <cinttypes>
#include <cstdlib>
#include <cstdio>
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <linux/dma-buf.h>
#include <sys/mman.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cstring>
#include <limits>
#include <memory>
#include <tuple>
#include <unordered_set>
#include <vector>

#include "ggml-backend-impl.h"
#include "ggml-htp-impl.h"
#include "ggml-htp.h"
#include "ggml.h"
#include "dsprpc_interface.h"
#include "message.h"
#include "op_reg.h"
#include "rpcmem_session_route.h"

#ifdef GGML_HTP_OPENCL_SIDECAR
#include "ggml-opencl-htp.h"
#endif

namespace {

bool packed_head_dim_supported(int head_dim) {
    return htp_packed_head_dim_supported(head_dim) != 0;
}

bool packed_qkv_weight_shape(const ggml_tensor * weight, int * n_heads, int * head_dim) {
    if (weight == nullptr || n_heads == nullptr || head_dim == nullptr) {
        return false;
    }
    for (const int heads : {24, 30, 38}) {
        for (const int dim : {64, 128}) {
            if (weight->ne[1] == 3 * heads * dim) {
                *n_heads = heads;
                *head_dim = dim;
                return true;
            }
        }
    }
    return false;
}

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
    const auto & mapper = ggml_backend_htp_context::instance()->dispatch_mapper();

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

// Diagnostic boundary probe: this calls the existing htp-ops-lib DSP kernel
// through its standard FastRPC entry point, bypassing only the custom message
// channel. It is intentionally restricted to the reserved session so the
// legacy domain-3 path remains byte-for-byte unchanged.
using direct_f16_matmul_trace_session_fn = int(
    htp_ops_session *, int, int, int, int, int, int, int, int, int, uint64_t *, uint64_t *, uint64_t *);

bool htp_direct_reserved_matmul_enabled(const ggml_backend_htp_context * ctx) {
    return ctx != nullptr && ctx->reserved_session != nullptr &&
           ctx->reserved_domain_id >= 0 &&
           ctx->dispatch_mapper().domain() == ctx->reserved_domain_id &&
           env_flag_enabled("SD_HTP_MULTI_SESSION_DIRECT");
}

bool htp_host_dma_buf_sync_read(int fd) {
    if (fd < 0 || !env_flag_enabled("SD_HTP_HOST_DMA_BUF_SYNC")) {
        return true;
    }
    dma_buf_sync sync = { DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ };
    if (ioctl(fd, DMA_BUF_IOCTL_SYNC, &sync) != 0) {
        std::fprintf(stderr, "[ggml-htp] DMA_BUF_SYNC START failed fd=%d errno=%d\n", fd, errno);
        return false;
    }
    sync.flags = DMA_BUF_SYNC_END | DMA_BUF_SYNC_READ;
    if (ioctl(fd, DMA_BUF_IOCTL_SYNC, &sync) != 0) {
        std::fprintf(stderr, "[ggml-htp] DMA_BUF_SYNC END failed fd=%d errno=%d\n", fd, errno);
        return false;
    }
    return true;
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

// This exists only for the reserved-session output-alias diagnostic. It keeps
// tensor placement unchanged and is instantiated only when explicitly enabled.
struct htp_independent_output {
    void * ptr = nullptr;
    int fd = -1;
    size_t size = 0;
    int domain = -1;
    bool mapped = false;

    bool ensure(int mapping_domain, size_t output_size) {
        if (ptr != nullptr && mapped && domain == mapping_domain && size >= output_size) {
            return true;
        }
        if (mapped) {
            const int ret = fastrpc_munmap(domain, fd, ptr, size);
            if (ret != 0) {
                std::fprintf(stderr,
                             "[ggml-htp] independent output fastrpc_munmap failed domain=%d fd=%d size=%zu ret=%d\n",
                             domain, fd, size, ret);
                return false;
            }
            mapped = false;
        }
        if (ptr != nullptr) {
            rpcmem_free(ptr);
            ptr = nullptr;
            fd = -1;
        }
        return init(mapping_domain, output_size);
    }

    bool init(int mapping_domain, size_t output_size) {
        if (output_size == 0 || output_size > static_cast<size_t>(std::numeric_limits<int>::max())) {
            return false;
        }

        domain = mapping_domain;
        size = output_size;
        ptr = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_FLAG_UNCACHED,
                           static_cast<int>(output_size));
        if (ptr == nullptr) {
            return false;
        }
        fd = rpcmem_to_fd(ptr);
        if (fd < 0) {
            rpcmem_free(ptr);
            ptr = nullptr;
            return false;
        }
        const int ret = fastrpc_mmap(domain, fd, ptr, 0, size, FASTRPC_MAP_FD);
        if (ret != 0) {
            std::fprintf(stderr,
                         "[ggml-htp] independent output fastrpc_mmap failed domain=%d fd=%d size=%zu ret=%d\n",
                         domain, fd, size, ret);
            rpcmem_free(ptr);
            ptr = nullptr;
            fd = -1;
            return false;
        }
        mapped = true;
        std::memset(ptr, 0, size);
        return true;
    }

    ~htp_independent_output() {
        if (mapped) {
            const int ret = fastrpc_munmap(domain, fd, ptr, size);
            if (ret != 0) {
                std::fprintf(stderr,
                             "[ggml-htp] independent output fastrpc_munmap failed domain=%d fd=%d size=%zu ret=%d\n",
                             domain, fd, size, ret);
                return;
            }
        }
        if (ptr != nullptr) {
            rpcmem_free(ptr);
        }
    }
};

// Reserved-PD diagnostic: stage an activation in a dedicated RPCMEM
// allocation before dispatch. This isolates the DSP input visibility path
// from the graph's shared fd/offset placement.
struct htp_independent_activation {
    void * ptr = nullptr;
    int fd = -1;
    size_t size = 0;
    int domain = -1;
    bool mapped = false;

    bool init(int mapping_domain, const void * source, size_t activation_size) {
        if (source == nullptr || activation_size == 0 ||
            activation_size > static_cast<size_t>(std::numeric_limits<int>::max())) {
            return false;
        }

        domain = mapping_domain;
        size = activation_size;
        ptr = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_FLAG_UNCACHED,
                           static_cast<int>(activation_size));
        if (ptr == nullptr) {
            return false;
        }
        fd = rpcmem_to_fd(ptr);
        if (fd < 0) {
            rpcmem_free(ptr);
            ptr = nullptr;
            return false;
        }
        const int ret = fastrpc_mmap(domain, fd, ptr, 0, size, FASTRPC_MAP_FD);
        if (ret != 0) {
            std::fprintf(stderr,
                         "[ggml-htp] independent activation fastrpc_mmap failed domain=%d fd=%d size=%zu ret=%d\n",
                         domain, fd, size, ret);
            rpcmem_free(ptr);
            ptr = nullptr;
            fd = -1;
            return false;
        }
        mapped = true;
        std::memcpy(ptr, source, size);
        return true;
    }

    ~htp_independent_activation() {
        if (mapped) {
            const int ret = fastrpc_munmap(domain, fd, ptr, size);
            if (ret != 0) {
                std::fprintf(stderr,
                             "[ggml-htp] independent activation fastrpc_munmap failed domain=%d fd=%d size=%zu ret=%d\n",
                             domain, fd, size, ret);
                return;
            }
        }
        if (ptr != nullptr) {
            rpcmem_free(ptr);
        }
    }
};

// Diagnostic output isolation for the custom channel. Most HTP parameter
// structs put the destination descriptor first; packed QKV ops expose three
// destinations sharing the same backing tensor and are handled together.
struct htp_isolated_dispatch_output {
    htp_independent_output * allocation = nullptr;
    void * destination = nullptr;
    size_t size = 0;
    bool active = false;

    bool init(int domain, ggml_tensor * dst) {
        if (dst == nullptr || dst->data == nullptr) {
            return false;
        }
        destination = dst->data;
        size = ggml_nbytes(dst);
        static htp_independent_output pool;
        allocation = &pool;
        // Reserve one reusable slot large enough for every SD3.5 HTP output;
        // resizing would require a DSP put before munmap and would recreate
        // the FastRPC VA accumulation this diagnostic is meant to avoid.
        active = allocation->ensure(domain, std::max<size_t>(size, 16u * 1024u * 1024u));
        return active;
    }

    bool bind(int op_index, void * params, int original_fd, ssize_t original_offset) {
        if (!active || params == nullptr || original_fd < 0 || original_offset < 0) {
            return false;
        }

        auto bind_descriptor = [&](RpcmemBufAddr & descriptor) {
            if (descriptor.fd != original_fd || descriptor.offset < original_offset) {
                return false;
            }
            const size_t relative = static_cast<size_t>(descriptor.offset - original_offset);
            if (relative >= size) {
                return false;
            }
            descriptor.fd = allocation->fd;
            descriptor.offset = static_cast<int32_t>(relative);
            return true;
        };

        auto * first = static_cast<RpcmemBufAddr *>(params);
        if (!bind_descriptor(*first)) {
            return false;
        }
        switch (op_index) {
            case HTP_OPS_QKV_MUL_RMS_PACKED_W16A32:
            {
                auto * p = static_cast<QkvMulRmsPackedParams *>(params);
                if (!bind_descriptor(p->k_out) || !bind_descriptor(p->v)) {
                    return false;
                }
                break;
            }
            case HTP_OPS_QKV_MUL_RMS_PACKED_DUAL_W16A32:
            {
                auto * p = static_cast<QkvMulRmsPackedDualParams *>(params);
                if (!bind_descriptor(p->k_out) || !bind_descriptor(p->v)) {
                    return false;
                }
                break;
            }
            case HTP_OPS_QKV_MUL_RMS_PACKED_ROPE_W16A32:
            {
                auto * p = static_cast<QkvMulRmsPackedRopeParams *>(params);
                if (!bind_descriptor(p->k_out) || !bind_descriptor(p->v)) {
                    return false;
                }
                break;
            }
            case HTP_OPS_QKV_MUL_RMS_PACKED_DUAL_ROPE_W16A32:
            {
                auto * p = static_cast<QkvMulRmsPackedDualRopeParams *>(params);
                if (!bind_descriptor(p->k_out) || !bind_descriptor(p->v)) {
                    return false;
                }
                break;
            }
            default:
                break;
        }
        return true;
    }

    void copy_back() const {
        if (active && destination != nullptr && allocation != nullptr && allocation->ptr != nullptr) {
            std::memcpy(destination, allocation->ptr, size);
        }
    }
};

// Probe whether a completed RPCMEM write becomes visible through the existing
// host mapping, after msync, or only through a fresh mapping of the same fd.
void htp_host_cache_probe(int fd, ssize_t offset, const void * data, size_t size) {
    if (fd < 0 || data == nullptr || size == 0 || !env_flag_enabled("SD_HTP_HOST_CACHE_PROBE")) {
        return;
    }

    const size_t page_size = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    const uintptr_t data_addr = reinterpret_cast<uintptr_t>(data);
    const uintptr_t page_addr = data_addr & ~(static_cast<uintptr_t>(page_size) - 1);
    const size_t page_delta = static_cast<size_t>(data_addr - page_addr);
    const size_t sync_size = page_delta + size;
    const bool msync_ok = msync(reinterpret_cast<void *>(page_addr), sync_size, MS_SYNC | MS_INVALIDATE) == 0;
    const uint64_t existing_hash = htp_fnv1a64_update(UINT64_C(14695981039346656037), data, size);
    const uint64_t msync_hash = htp_fnv1a64_update(UINT64_C(14695981039346656037), data, size);

    const ssize_t page_offset = offset & -static_cast<ssize_t>(page_size);
    const size_t fd_delta = static_cast<size_t>(offset - page_offset);
    const size_t map_size = fd_delta + size;
    void * fresh_map = mmap(nullptr, map_size, PROT_READ, MAP_SHARED, fd, page_offset);
    const bool fresh_ok = fresh_map != MAP_FAILED;
    uint64_t fresh_hash = 0;
    if (fresh_ok) {
        fresh_hash = htp_fnv1a64_update(UINT64_C(14695981039346656037),
                                        static_cast<const uint8_t *>(fresh_map) + fd_delta, size);
        munmap(fresh_map, map_size);
    }

    std::fprintf(stderr,
                 "[ggml-htp-cache] fd=%d offset=%zd size=%zu msync=%d fresh_mmap=%d "
                 "existing=%016" PRIx64 " after_msync=%016" PRIx64 " fresh=%016" PRIx64 " errno=%d\n",
                 fd, offset, size, msync_ok ? 1 : 0, fresh_ok ? 1 : 0,
                 existing_hash, msync_hash, fresh_hash, errno);
}

std::atomic<int> htp_debug_hash_counter{0};
std::atomic<bool> htp_opencl_residual_job_just_queued{false};
std::atomic<bool> htp_opencl_next_node_attempt_traced{false};

bool htp_checked_mul_size(size_t lhs, size_t rhs, size_t * result) {
    if (result == nullptr || (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs)) {
        return false;
    }
    *result = lhs * rhs;
    return true;
}

const ggml_htp_fused_anchor_metadata * fused_anchor_metadata(const ggml_tensor * dst) {
    if (dst == nullptr || dst->op != GGML_OP_CUSTOM) {
        return nullptr;
    }
    ggml_custom_op_params custom = {};
    std::memcpy(&custom, dst->op_params, sizeof(custom));
    return static_cast<const ggml_htp_fused_anchor_metadata *>(custom.userdata);
}

bool fused_anchor_qk_layout(
        const ggml_tensor * tensor, ggml_type type,
        int32_t tokens, int32_t heads, int32_t head_dim) {
    return tensor != nullptr && tensor->type == type && tokens > 0 && heads > 0 && head_dim > 0 &&
           tensor->ne[0] == head_dim && tensor->ne[1] == tokens && tensor->ne[2] == heads &&
           tensor->ne[3] == 1;
}

bool fused_anchor_v_layout(
        const ggml_tensor * tensor, ggml_type type,
        int32_t tokens, int32_t heads, int32_t head_dim) {
    return tensor != nullptr && tensor->type == type && tokens > 0 && heads > 0 && head_dim > 0 &&
           tensor->ne[0] == head_dim && tensor->ne[1] == heads && tensor->ne[2] == tokens &&
           tensor->ne[3] == 1;
}

bool fused_anchor_expected_sizes(
        const ggml_htp_fused_anchor_metadata * params,
        size_t * output_bytes, size_t * slab_bytes, size_t * q_bytes,
        size_t * kv_bytes, size_t * dense_mask_bytes, size_t * block_mask_bytes) {
    if (params == nullptr || output_bytes == nullptr || slab_bytes == nullptr || q_bytes == nullptr ||
        kv_bytes == nullptr || dense_mask_bytes == nullptr || block_mask_bytes == nullptr) {
        return false;
    }
    size_t rows = 0;
    size_t elements = 0;
    size_t dense_stride = 0;
    if (!htp_checked_mul_size(static_cast<size_t>(params->qo_len),
                              static_cast<size_t>(params->n_heads), &rows) ||
        !htp_checked_mul_size(rows, static_cast<size_t>(params->head_dim + 1), &elements) ||
        !htp_checked_mul_size(elements, sizeof(float), slab_bytes) ||
        !htp_checked_mul_size(*slab_bytes, 2, output_bytes) ||
        !htp_checked_mul_size(rows, static_cast<size_t>(params->head_dim), &elements) ||
        !htp_checked_mul_size(elements, sizeof(float), q_bytes) ||
        !htp_checked_mul_size(static_cast<size_t>(params->kv_len),
                              static_cast<size_t>(params->n_kv_heads), &elements) ||
        !htp_checked_mul_size(elements, static_cast<size_t>(params->head_dim), &elements) ||
        !htp_checked_mul_size(elements, sizeof(ggml_fp16_t), kv_bytes)) {
        return false;
    }
    if (static_cast<size_t>(params->kv_len) > std::numeric_limits<size_t>::max() - 63u) {
        return false;
    }
    dense_stride = (static_cast<size_t>(params->kv_len) + 63u) & ~size_t(63u);
    return htp_checked_mul_size(static_cast<size_t>(params->qo_len), dense_stride, &elements) &&
           htp_checked_mul_size(elements, sizeof(ggml_fp16_t), dense_mask_bytes) &&
           htp_checked_mul_size(static_cast<size_t>(params->n_q_blocks),
                                static_cast<size_t>(params->max_kept_blocks), &elements) &&
           htp_checked_mul_size(elements, sizeof(int32_t), block_mask_bytes);
}

bool is_fused_anchor_attention_custom(const ggml_tensor * dst) {
    const ggml_htp_fused_anchor_metadata * params = fused_anchor_metadata(dst);
    if (params == nullptr || std::strcmp(dst->name, "htp_fused_anchor_attention") != 0 ||
        dst->type != GGML_TYPE_F32 || dst->ne[1] != 1 || dst->ne[2] != 1 || dst->ne[3] != 1 ||
        dst->src[0] == nullptr || dst->src[1] == nullptr || dst->src[2] == nullptr ||
        dst->src[3] == nullptr || dst->src[4] == nullptr || dst->src[5] != nullptr ||
        params->qo_len <= 0 || params->qo_len != params->kv_len || params->n_heads <= 0 ||
        params->n_heads != params->n_kv_heads ||
        (params->head_dim != 64 && params->head_dim != 128) ||
        params->block_q != 32 || params->block_k != 32 || params->mask_format != 1 ||
        params->sparse_policy != HTP_SPARSE_POLICY_STATIC_HALO ||
        params->image_tokens <= 0 || params->text_tokens < 0 ||
        params->image_tokens + params->text_tokens != params->qo_len ||
        params->image_tokens % params->block_q != 0 ||
        params->image_block_rows <= 0 || params->image_block_cols <= 0 ||
        params->image_block_rows > std::numeric_limits<int32_t>::max() / params->image_block_cols ||
        params->image_block_rows * params->image_block_cols != params->image_tokens / params->block_q) {
        return false;
    }
    const int32_t q_blocks_per_head = (params->qo_len + params->block_q - 1) / params->block_q;
    const int32_t k_blocks_per_head = (params->kv_len + params->block_k - 1) / params->block_k;
    if (params->n_heads > std::numeric_limits<int32_t>::max() / q_blocks_per_head ||
        params->n_kv_heads > std::numeric_limits<int32_t>::max() / k_blocks_per_head ||
        params->n_q_blocks != params->n_heads * q_blocks_per_head ||
        params->n_k_blocks != params->n_kv_heads * k_blocks_per_head ||
        params->max_kept_blocks != k_blocks_per_head) {
        return false;
    }

    size_t output_bytes = 0, slab_bytes = 0, q_bytes = 0, kv_bytes = 0;
    size_t dense_mask_bytes = 0, block_mask_bytes = 0;
    if (!fused_anchor_expected_sizes(params, &output_bytes, &slab_bytes, &q_bytes, &kv_bytes,
                                     &dense_mask_bytes, &block_mask_bytes)) {
        return false;
    }
    const size_t dense_stride = (static_cast<size_t>(params->kv_len) + 63u) & ~size_t(63u);
    return static_cast<size_t>(dst->ne[0]) == output_bytes / sizeof(float) &&
           fused_anchor_qk_layout(dst->src[0], GGML_TYPE_F32, params->qo_len,
                                  params->n_heads, params->head_dim) &&
           fused_anchor_qk_layout(dst->src[1], GGML_TYPE_F16, params->kv_len,
                                  params->n_kv_heads, params->head_dim) &&
           fused_anchor_v_layout(dst->src[2], GGML_TYPE_F16, params->kv_len,
                                 params->n_kv_heads, params->head_dim) &&
           dst->src[3]->type == GGML_TYPE_F16 && dst->src[3]->ne[0] == static_cast<int64_t>(dense_stride) &&
           dst->src[3]->ne[1] == params->qo_len && dst->src[3]->ne[2] == 1 && dst->src[3]->ne[3] == 1 &&
           ggml_is_contiguous(dst->src[3]) &&
           dst->src[4]->type == GGML_TYPE_I32 && dst->src[4]->ne[0] == params->max_kept_blocks &&
           dst->src[4]->ne[1] == params->n_q_blocks && dst->src[4]->ne[2] == 1 && dst->src[4]->ne[3] == 1 &&
           ggml_is_contiguous(dst->src[4]) && ggml_is_contiguous(dst);
}

struct htp_tensor_range {
    ggml_backend_buffer_t backing = nullptr;
    size_t offset = 0;
    size_t size = 0;
};

bool htp_tensor_range_for_bytes(const ggml_tensor * tensor, size_t size, htp_tensor_range * range) {
    const ggml_tensor * backing_tensor = get_tensor_with_backing_buffer(tensor);
    if (tensor == nullptr || range == nullptr || backing_tensor == nullptr || backing_tensor->buffer == nullptr ||
        tensor->data == nullptr) {
        return false;
    }
    void * base = ggml_backend_buffer_get_base(backing_tensor->buffer);
    const size_t backing_size = ggml_backend_buffer_get_size(backing_tensor->buffer);
    const uintptr_t base_address = reinterpret_cast<uintptr_t>(base);
    const uintptr_t data_address = reinterpret_cast<uintptr_t>(tensor->data);
    if (base == nullptr || data_address < base_address || data_address - base_address > backing_size) {
        return false;
    }
    const size_t offset = static_cast<size_t>(data_address - base_address);
    if (size > backing_size - offset) {
        return false;
    }
    *range = {backing_tensor->buffer, offset, size};
    return true;
}

bool htp_ranges_overlap(const htp_tensor_range & lhs, const htp_tensor_range & rhs) {
    return lhs.backing == rhs.backing && lhs.offset < rhs.offset + rhs.size &&
           rhs.offset < lhs.offset + lhs.size;
}

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

bool is_qkv_mul_rms_packed_custom(const ggml_tensor * dst) {
    return dst != nullptr && dst->op == GGML_OP_CUSTOM &&
           std::strcmp(dst->name, "qkv_mul_rms_packed") == 0 &&
           dst->src[0] != nullptr && dst->src[1] != nullptr &&
           dst->src[2] != nullptr && dst->src[3] != nullptr && dst->src[4] != nullptr &&
           dst->src[0]->type == GGML_TYPE_F32 && dst->src[1]->type == GGML_TYPE_F16 &&
           dst->src[2]->type == GGML_TYPE_F32 && dst->src[3]->type == GGML_TYPE_F32 &&
           dst->src[4]->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16 &&
           dst->ne[1] == 1 && dst->ne[2] == 1 && dst->ne[3] == 1;
}

bool is_qkv_mul_rms_packed_dual_custom(const ggml_tensor * dst) {
    if (dst == nullptr || dst->op != GGML_OP_CUSTOM ||
        std::strcmp(dst->name, "qkv_mul_rms_packed_dual") != 0 ||
        dst->type != GGML_TYPE_F16 || dst->ne[1] != 1 || dst->ne[2] != 1 || dst->ne[3] != 1) {
        return false;
    }
    for (int i = 0; i < 10; ++i) {
        if (dst->src[i] == nullptr && i != 2 && i != 7) return false;
    }
    for (int i = 0; i < 10; ++i) {
        const ggml_tensor * src = dst->src[i];
        if (src == nullptr) continue;
        if ((i == 1 || i == 6) && src->type != GGML_TYPE_F16) return false;
        if (i != 1 && i != 6 && src->type != GGML_TYPE_F32) return false;
    }
    return true;
}

bool is_qkv_mul_rms_packed_rope_custom(const ggml_tensor * dst) {
    if (dst == nullptr || dst->op != GGML_OP_CUSTOM ||
        std::strcmp(dst->name, "qkv_mul_rms_packed_rope") != 0 ||
        dst->type != GGML_TYPE_F16 || dst->ne[1] != 1 || dst->ne[2] != 1 || dst->ne[3] != 1) {
        return false;
    }
    for (int i = 0; i <= 5; ++i) {
        if (dst->src[i] == nullptr && i != 2) return false;
    }
    return dst->src[0]->type == GGML_TYPE_F32 && dst->src[1]->type == GGML_TYPE_F16 &&
           (dst->src[2] == nullptr || dst->src[2]->type == GGML_TYPE_F32) &&
           dst->src[3]->type == GGML_TYPE_F32 && dst->src[4]->type == GGML_TYPE_F32 &&
           dst->src[5]->type == GGML_TYPE_F32;
}

bool is_qkv_mul_rms_packed_dual_rope_custom(const ggml_tensor * dst) {
    if (dst == nullptr || dst->op != GGML_OP_CUSTOM ||
        std::strcmp(dst->name, "qkv_mul_rms_packed_dual_rope") != 0 ||
        dst->type != GGML_TYPE_F16 || dst->ne[1] != 1 || dst->ne[2] != 1 || dst->ne[3] != 1) {
        return false;
    }
    for (int i = 0; i <= 10; ++i) {
        if (dst->src[i] == nullptr && i != 2 && i != 7) return false;
    }
    for (int i = 0; i <= 10; ++i) {
        const ggml_tensor * src = dst->src[i];
        if (src == nullptr) continue;
        if ((i == 1 || i == 6) && src->type != GGML_TYPE_F16) return false;
        if (i != 1 && i != 6 && src->type != GGML_TYPE_F32) return false;
    }
    return true;
}

bool is_packed_sparse_attention(const ggml_tensor * dst) {
    return dst != nullptr && dst->op == GGML_OP_FLASH_ATTN_EXT &&
           std::strcmp(dst->name, "packed_sparse_attention") == 0;
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
        case HTP_OPS_QKV_MUL_RMS_PACKED_W16A32:
            return "HTP_OPS_QKV_MUL_RMS_PACKED_W16A32";
        case HTP_OPS_QKV_MUL_RMS_PACKED_DUAL_W16A32:
            return "HTP_OPS_QKV_MUL_RMS_PACKED_DUAL_W16A32";
        case HTP_OPS_QKV_MUL_RMS_PACKED_ROPE_W16A32:
            return "HTP_OPS_QKV_MUL_RMS_PACKED_ROPE_W16A32";
        case HTP_OPS_QKV_MUL_RMS_PACKED_DUAL_ROPE_W16A32:
            return "HTP_OPS_QKV_MUL_RMS_PACKED_DUAL_ROPE_W16A32";
        case HTP_OPS_FUSED_ANCHOR_ATTN_QO_F32_KV_F16:
            return "HTP_OPS_FUSED_ANCHOR_ATTN_QO_F32_KV_F16";
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
        case HTP_OPS_BLOCK_SPARSE_ATTN_QKV_PACKED_F16:
            return "HTP_OPS_BLOCK_SPARSE_ATTN_QKV_PACKED_F16";
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

bool htp_ops_is_fused_anchor_attention(const struct ggml_tensor * dst) {
    return is_fused_anchor_attention_custom(dst);
}

int htp_ops_resolve_halo_kv_load_mode(void) {
    const char * value = std::getenv("SD_HTP_HALO_KV_LOAD");
    if (value == nullptr || value[0] == '\0' || std::strcmp(value, "l2fetch") == 0) {
        return HTP_HALO_KV_LOAD_L2_PREFETCH;
    }
    if (std::strcmp(value, "baseline") == 0) {
        return HTP_HALO_KV_LOAD_BASELINE;
    }
    if (std::strcmp(value, "userdma") == 0) {
        return HTP_HALO_KV_LOAD_USERDMA_AB;
    }
    if (std::strcmp(value, "chunked") == 0) {
        return HTP_HALO_KV_LOAD_CHUNKED;
    }

    std::fprintf(stderr,
                 "[ggml-htp] invalid SD_HTP_HALO_KV_LOAD=%s (expected baseline, l2fetch, userdma, or chunked)\n",
                 value);
    return -EINVAL;
}

int htp_ops_resolve_halo_kv_load_mode_for_node(
        const struct ggml_tensor * node, bool * used_node_override) {
    static_assert(static_cast<int>(GGML_FLASH_ATTN_EXT_HALO_KV_LOAD_BASELINE) == HTP_HALO_KV_LOAD_BASELINE);
    static_assert(static_cast<int>(GGML_FLASH_ATTN_EXT_HALO_KV_LOAD_L2_PREFETCH) == HTP_HALO_KV_LOAD_L2_PREFETCH);
    static_assert(static_cast<int>(GGML_FLASH_ATTN_EXT_HALO_KV_LOAD_USERDMA_AB) == HTP_HALO_KV_LOAD_USERDMA_AB);
    static_assert(static_cast<int>(GGML_FLASH_ATTN_EXT_HALO_KV_LOAD_CHUNKED) == HTP_HALO_KV_LOAD_CHUNKED);

    if (used_node_override != nullptr) {
        *used_node_override = false;
    }
    if (node != nullptr && node->op == GGML_OP_FLASH_ATTN_EXT) {
        const auto mode = ggml_flash_attn_ext_get_halo_kv_load_mode(node);
        if (mode != GGML_FLASH_ATTN_EXT_HALO_KV_LOAD_INHERIT) {
            if (used_node_override != nullptr) {
                *used_node_override = true;
            }
            return static_cast<int>(mode);
        }
    }
    return htp_ops_resolve_halo_kv_load_mode();
}

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

                if (is_packed_sparse_attention(dst)) {
                    const bool full_keep_packed =
                        mask_format == GGML_FLASH_ATTN_EXT_MASK_BLOCK_MATRIX &&
                        ggml_flash_attn_ext_get_sparse_policy(dst) == GGML_FLASH_ATTN_EXT_SPARSE_POLICY_NONE;
                    const bool static_halo_packed =
                        mask_format == GGML_FLASH_ATTN_EXT_MASK_BLOCK_INDEX &&
                        ggml_flash_attn_ext_get_sparse_policy(dst) == GGML_FLASH_ATTN_EXT_SPARSE_POLICY_STATIC_HALO;
                    return dst->type == GGML_TYPE_F32 && q->type == GGML_TYPE_F16 &&
                           k->type == GGML_TYPE_F16 && v->type == GGML_TYPE_F16 &&
                           mask->type == GGML_TYPE_I32 && (full_keep_packed || static_halo_packed) &&
                           shape_ok && q->ne[1] == k->ne[1] && k->ne[2] == q->ne[2] && q->ne[3] == 1 &&
                           htp_packed_shape_supported((int) q->ne[2], (int) k->ne[2], (int) q->ne[0], 1);
                }

                return dst->type == GGML_TYPE_F32 && q->type == GGML_TYPE_F32 && k->type == GGML_TYPE_F16 &&
                       v->type == GGML_TYPE_F16 && (mask->type == GGML_TYPE_F16 || mask->type == GGML_TYPE_I32) &&
                       packed_head_dim_supported((int) q->ne[0]) && mask_format_ok && shape_ok &&
                       max_bias == 0 && logit_softcap == 0;
            }
        case GGML_OP_CUSTOM:
            if (is_fused_anchor_attention_custom(dst)) {
                return true;
            }
            if (is_qkv_mul_rms_packed_dual_rope_custom(dst)) {
                const ggml_tensor * image_activation = dst->src[0];
                const ggml_tensor * text_activation = dst->src[5];
                const ggml_tensor * image_weight = dst->src[1];
                const ggml_tensor * text_weight = dst->src[6];
                int image_heads = 0, image_head_dim = 0;
                int text_heads = 0, text_head_dim = 0;
                const bool image_shape_ok = packed_qkv_weight_shape(image_weight, &image_heads, &image_head_dim);
                const bool text_shape_ok = packed_qkv_weight_shape(text_weight, &text_heads, &text_head_dim);
                return image_activation->ne[0] == text_activation->ne[0] &&
                       image_activation->ne[0] % 32 == 0 && image_shape_ok && text_shape_ok &&
                       htp_packed_shape_supported(image_heads, text_heads, image_head_dim, 1) &&
                       image_head_dim == text_head_dim &&
                       image_weight->ne[0] == image_activation->ne[0] &&
                       text_weight->ne[0] == text_activation->ne[0] &&
                       image_weight->ne[1] == text_weight->ne[1] &&
                       dst->src[3]->ne[0] == image_head_dim && dst->src[4]->ne[0] == image_head_dim &&
                       dst->src[8]->ne[0] == text_head_dim && dst->src[9]->ne[0] == text_head_dim;
            }
            if (is_qkv_mul_rms_packed_rope_custom(dst)) {
                const ggml_tensor * activation = dst->src[0];
                const ggml_tensor * weight = dst->src[1];
                int n_heads = 0, head_dim = 0;
                const bool weight_shape_ok = packed_qkv_weight_shape(weight, &n_heads, &head_dim);
                return activation->ne[0] > 0 && activation->ne[1] > 0 &&
                       activation->ne[0] % 32 == 0 && weight->ne[0] == activation->ne[0] &&
                       weight_shape_ok && htp_packed_shape_supported(n_heads, n_heads, head_dim, 1) &&
                       dst->src[3]->ne[0] == head_dim && dst->src[4]->ne[0] == head_dim;
            }
            if (is_qkv_mul_rms_packed_dual_custom(dst)) {
                const ggml_tensor * image_activation = dst->src[0];
                const ggml_tensor * text_activation = dst->src[5];
                const ggml_tensor * image_weight = dst->src[1];
                const ggml_tensor * text_weight = dst->src[6];
                int image_heads = 0, image_head_dim = 0;
                int text_heads = 0, text_head_dim = 0;
                const bool image_shape_ok = packed_qkv_weight_shape(image_weight, &image_heads, &image_head_dim);
                const bool text_shape_ok = packed_qkv_weight_shape(text_weight, &text_heads, &text_head_dim);
                return image_activation->ne[0] == text_activation->ne[0] &&
                       image_activation->ne[0] % 32 == 0 &&
                       image_weight->ne[0] == image_activation->ne[0] &&
                       text_weight->ne[0] == text_activation->ne[0] &&
                       image_shape_ok && text_shape_ok && image_head_dim == text_head_dim &&
                       htp_packed_shape_supported(image_heads, text_heads, image_head_dim, 1) &&
                       dst->src[3]->ne[0] == image_head_dim && dst->src[4]->ne[0] == image_head_dim &&
                       dst->src[8]->ne[0] == text_head_dim && dst->src[9]->ne[0] == text_head_dim &&
                       dst->ne[0] > 0;
            }
            if (is_qkv_mul_rms_packed_custom(dst)) {
                const ggml_tensor * activation = dst->src[0];
                const ggml_tensor * weight = dst->src[1];
                int n_heads = 0, head_dim = 0;
                const bool weight_shape_ok = packed_qkv_weight_shape(weight, &n_heads, &head_dim);
                return activation->ne[0] > 0 && activation->ne[1] > 0 &&
                       activation->ne[0] % 32 == 0 && weight->ne[0] == activation->ne[0] &&
                       weight_shape_ok && htp_packed_shape_supported(n_heads, n_heads, head_dim, 1) &&
                       dst->src[3]->ne[0] == head_dim && dst->src[4]->ne[0] == head_dim;
            }
            return is_block_sparse_score_custom(dst);
        default:
            return false;
    }
}

bool htp_ops_support_op_runtime(const struct ggml_tensor * dst) {
    const bool fused_anchor_named = dst != nullptr && dst->op == GGML_OP_CUSTOM &&
        std::strcmp(dst->name, "htp_fused_anchor_attention") == 0;
    const char * fused_trace_env = std::getenv("SD_HTP_BLOCK_SPARSE_MASK_TRACE");
    const bool trace_fused_anchor = fused_anchor_named && fused_trace_env != nullptr &&
        fused_trace_env[0] != '\0' && std::strcmp(fused_trace_env, "0") != 0;
    if (!htp_ops_support_op(dst)) {
        if (trace_fused_anchor) {
            std::fprintf(stderr, "fused_anchor_runtime_support rejected stage=static_support\n");
        }
        return false;
    }
    if (!tensor_ready_for_htp_dispatch(dst)) {
        if (trace_fused_anchor) {
            std::fprintf(stderr, "fused_anchor_runtime_support rejected stage=dst_rpcmem\n");
            dump_tensor_rpcmem_state("fused_dst", dst);
        }
        return false;
    }

    if (is_fused_anchor_attention_custom(dst)) {
        const ggml_htp_fused_anchor_metadata * metadata = fused_anchor_metadata(dst);
        size_t output_bytes = 0, slab_bytes = 0, q_bytes = 0, kv_bytes = 0;
        size_t dense_mask_bytes = 0, block_mask_bytes = 0;
        if (!fused_anchor_expected_sizes(metadata, &output_bytes, &slab_bytes, &q_bytes, &kv_bytes,
                                         &dense_mask_bytes, &block_mask_bytes)) {
            if (trace_fused_anchor) {
                std::fprintf(stderr, "fused_anchor_runtime_support rejected stage=expected_sizes\n");
            }
            return false;
        }
        for (int i = 0; i < 5; ++i) {
            if (!tensor_ready_for_htp_dispatch_view_backed(dst->src[i])) {
                if (trace_fused_anchor) {
                    std::fprintf(stderr,
                                 "fused_anchor_runtime_support rejected stage=src_rpcmem src=%d\n", i);
                    dump_tensor_rpcmem_state("fused_src", dst->src[i]);
                }
                return false;
            }
        }
        htp_tensor_range ranges[6] = {};
        const size_t sizes[6] = {output_bytes, q_bytes, kv_bytes, kv_bytes, dense_mask_bytes, block_mask_bytes};
        const ggml_tensor * tensors[6] = {dst, dst->src[0], dst->src[1], dst->src[2], dst->src[3], dst->src[4]};
        for (int i = 0; i < 6; ++i) {
            if (!htp_tensor_range_for_bytes(tensors[i], sizes[i], &ranges[i]) ||
                ranges[i].offset > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
                if (trace_fused_anchor) {
                    std::fprintf(stderr,
                                 "fused_anchor_runtime_support rejected stage=range tensor=%d bytes=%zu\n",
                                 i, sizes[i]);
                    dump_tensor_rpcmem_state("fused_range", tensors[i]);
                }
                return false;
            }
        }
        for (int i = 1; i < 6; ++i) {
            if (htp_ranges_overlap(ranges[0], ranges[i])) {
                if (trace_fused_anchor) {
                    std::fprintf(stderr,
                                 "fused_anchor_runtime_support rejected stage=output_overlap src=%d "
                                 "output_offset=%zu output_bytes=%zu src_offset=%zu src_bytes=%zu\n",
                                 i, ranges[0].offset, ranges[0].size,
                                 ranges[i].offset, ranges[i].size);
                }
                return false;
            }
        }
        return true;
    }

    if (is_qkv_mul_rms_packed_custom(dst)) {
        for (int i = 0; i < 5; ++i) {
            if (!tensor_ready_for_htp_dispatch_view_backed(dst->src[i])) return false;
        }
        return tensor_ready_for_htp_dispatch_view_backed(dst);
    }

    if (is_qkv_mul_rms_packed_rope_custom(dst)) {
        for (int i = 0; i <= 5; ++i) {
            if (dst->src[i] != nullptr && !tensor_ready_for_htp_dispatch_view_backed(dst->src[i])) return false;
        }
        return tensor_ready_for_htp_dispatch_view_backed(dst);
    }

    if (is_qkv_mul_rms_packed_dual_custom(dst)) {
        for (int i = 0; i < 10; ++i) {
            if (dst->src[i] == nullptr) continue;
            if (!tensor_ready_for_htp_dispatch_view_backed(dst->src[i])) {
                return false;
            }
        }
        if (!tensor_ready_for_htp_dispatch_view_backed(dst)) {
            return false;
        }
        return true;
    }

    if (is_qkv_mul_rms_packed_dual_rope_custom(dst)) {
        for (int i = 0; i <= 10; ++i) {
            if (dst->src[i] != nullptr && !tensor_ready_for_htp_dispatch_view_backed(dst->src[i])) return false;
        }
        return tensor_ready_for_htp_dispatch_view_backed(dst);
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

bool htp_ops_opencl_defer_submission_wait_requested(void) {
    const char * value = std::getenv(
        "SD_HTP_SPARSE_OPENCL_DEFER_SUBMISSION_WAIT");
    return value != nullptr && value[0] != '\0' &&
           std::strcmp(value, "0") != 0;
}

#ifdef GGML_HTP_OPENCL_SIDECAR
static int htp_ops_opencl_compute_node_impl(struct ggml_tensor * dst, bool early_submit);

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

static bool htp_ops_opencl_submission_thread_active(void) {
    const auto * ctx = ggml_backend_htp_context::instance();
    return htp_ops_opencl_early_submit_enabled() && ctx != nullptr &&
           ctx->sparse_opencl_submission_thread_active;
}

static bool htp_ops_opencl_defer_submission_wait_enabled(void) {
    return htp_ops_opencl_submission_thread_active() &&
           htp_ops_opencl_defer_submission_wait_requested();
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

static void htp_ops_opencl_trace_submit_wait_event(
        const char * event,
        const ggml_tensor * node) {
    const char * trace = std::getenv("SD_HTP_SPARSE_OPENCL_TRACE");
    if (trace == nullptr || trace[0] == '\0' || std::strcmp(trace, "0") == 0) {
        return;
    }
    std::fprintf(
        stderr,
        "htp_opencl_event name=%s pair=%llu ts_us=%lld producer=%s\n",
        event,
        static_cast<unsigned long long>(htp_ops_opencl_pair_id_for_tensor(node)),
        (long long) ggml_time_us(),
        node != nullptr && node->name[0] != '\0' ? node->name : "<unnamed>");
}

static int htp_ops_opencl_early_submit_ready_candidates(const ggml_tensor * completed_node) {
    GGML_ASSERT(g_opencl_early_submit != nullptr);
    for (htp_ops_opencl_early_submit_candidate & candidate : g_opencl_early_submit->candidates) {
        if (candidate.submitted || !htp_ops_opencl_early_submit_candidate_ready(candidate)) {
            continue;
        }
        const ggml_htp_opencl_lifetime_mode lifetime_mode =
            ggml_htp_opencl_lifetime_mode_effective();
        if (lifetime_mode == GGML_HTP_OPENCL_LIFETIME_REUSE ||
            lifetime_mode == GGML_HTP_OPENCL_LIFETIME_STRICT) {
            auto * ctx = ggml_backend_htp_context::instance();
            char stage_reason[512] = {};
            const ggml_opencl_htp_result stage_result =
                ggml_backend_opencl_htp_graph_stage_block_sparse_attn_mask(
                    ctx->sparse_opencl_graph, candidate.node,
                    stage_reason, sizeof(stage_reason));
            if (stage_result != GGML_OPENCL_HTP_OK) {
                std::fprintf(
                    stderr,
                    "ggml-htp: failed to stage local OpenCL sparse mask result=%d reason=%s\n",
                    (int) stage_result,
                    stage_reason[0] != '\0' ? stage_reason : "unknown error");
                return -EIO;
            }
        }
        htp_ops_opencl_trace_early_submit(candidate.node, completed_node);
        if (htp_ops_opencl_submission_thread_active()) {
            auto * ctx = ggml_backend_htp_context::instance();
            const int64_t begin_ms = ggml_time_ms();
            char reason[512] = {};
            const ggml_opencl_htp_result result =
                ggml_backend_opencl_htp_graph_queue_block_sparse_attn(
                    ctx->sparse_opencl_graph, candidate.node, reason, sizeof(reason));
            const char * trace = std::getenv("SD_HTP_SPARSE_OPENCL_TRACE");
            if (trace != nullptr && trace[0] != '\0' && std::strcmp(trace, "0") != 0) {
                const ggml_tensor * node = candidate.node;
                std::fprintf(
                    stderr,
                    "attention_timing route=opencl mask_type=i32 total=%lldms qo=%lld kv=%lld "
                    "heads=%lld d=%lld with_lse=%d out_d=%lld block_sparse=1 async=1 queued=1 result=%d\n",
                    (long long) (ggml_time_ms() - begin_ms),
                    (long long) node->src[0]->ne[1],
                    (long long) node->src[1]->ne[1],
                    (long long) node->src[0]->ne[2],
                    (long long) node->src[0]->ne[0],
                    ggml_get_op_params_i32(node, 6) != 0 ? 1 : 0,
                    (long long) node->ne[0],
                    (int) result);
            }
            if (result != GGML_OPENCL_HTP_OK) {
                std::fprintf(
                    stderr,
                    "ggml-htp: failed to queue OpenCL sparse attention result=%d reason=%s\n",
                    (int) result, reason[0] != '\0' ? reason : "unknown error");
                return -EIO;
            }
            if (htp_ops_opencl_defer_submission_wait_enabled()) {
                htp_ops_opencl_trace_submit_wait_event(
                    "gpu_sparse_submit_wait_deferred", candidate.node);
            } else {
                htp_ops_opencl_trace_submit_wait_event(
                    "gpu_sparse_submit_wait_legacy_begin", candidate.node);
                const ggml_opencl_htp_result committed =
                    ggml_backend_opencl_htp_graph_wait_for_sparse_submission(
                        ctx->sparse_opencl_graph, candidate.node,
                        reason, sizeof(reason));
                htp_ops_opencl_trace_submit_wait_event(
                    "gpu_sparse_submit_wait_legacy_end", candidate.node);
                if (committed != GGML_OPENCL_HTP_OK) {
                    std::fprintf(
                        stderr,
                        "ggml-htp: OpenCL sparse submission commit failed result=%d reason=%s\n",
                        (int) committed,
                        reason[0] != '\0' ? reason : "unknown error");
                    return -EIO;
                }
            }
        } else {
            const int err = htp_ops_opencl_compute_node_impl(
                const_cast<ggml_tensor *>(candidate.node), true);
            if (err != 0) {
                return err;
            }
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
                                   ctx->sparse_opencl_routed_nodes.end() ||
            node->op != GGML_OP_FLASH_ATTN_EXT) {
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
                const ggml_tensor * mask_root =
                    htp_ops_opencl_early_submit_input_root(candidate.node->src[3]);
                if (root == mask_root &&
                    htp_ops_opencl_early_submit_is_inputless_single_task_custom(root) &&
                    candidate.completed_roots.find(root) == candidate.completed_roots.end()) {
                    auto * ctx = ggml_backend_htp_context::instance();
                    char reason[512] = {};
                    const ggml_opencl_htp_result result =
                        ggml_backend_opencl_htp_graph_stage_block_sparse_attn_mask(
                            ctx->sparse_opencl_graph,
                            candidate.node,
                            reason,
                            sizeof(reason));
                    if (result != GGML_OPENCL_HTP_OK) {
                        std::fprintf(
                            stderr,
                            "ggml-htp: failed to stage OpenCL sparse mask "
                            "result=%d reason=%s\n",
                            (int) result,
                            reason[0] != '\0' ? reason : "unknown error");
                        return -EIO;
                    }
                }
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

static const ggml_tensor * htp_ops_opencl_bound_cache_input(
        const ggml_backend_htp_context * ctx,
        const ggml_tensor * tensor) {
    for (size_t depth = 0; tensor != nullptr && depth < 64; ++depth) {
        if (ctx->sparse_opencl_cache_input_bindings.find(tensor) !=
            ctx->sparse_opencl_cache_input_bindings.end()) {
            return tensor;
        }
        if (!htp_ops_opencl_is_view_only(tensor)) {
            return nullptr;
        }
        tensor = tensor->src[0] != nullptr ? tensor->src[0] : tensor->view_src;
    }
    return nullptr;
}

static size_t htp_ops_opencl_collect_cache_inputs(
        const ggml_tensor * dst,
        const ggml_tensor ** inputs,
        size_t capacity) {
    if (dst == nullptr || htp_ops_opencl_is_view_only(dst)) {
        return 0;
    }
    const auto * ctx = ggml_backend_htp_context::instance();
    size_t count = 0;
    for (int source = 0; source < GGML_MAX_SRC; ++source) {
        const ggml_tensor * input =
            htp_ops_opencl_bound_cache_input(ctx, dst->src[source]);
        if (input == nullptr) {
            continue;
        }
        bool duplicate = false;
        for (size_t i = 0; i < count; ++i) {
            duplicate = duplicate || inputs[i] == input;
        }
        if (!duplicate && count < capacity) {
            inputs[count++] = input;
        }
    }
    return count;
}

static int htp_ops_opencl_wait_cache_inputs(const ggml_tensor * dst) {
    auto * ctx = ggml_backend_htp_context::instance();
    const ggml_tensor * inputs[GGML_MAX_SRC] = {};
    const size_t input_count =
        htp_ops_opencl_collect_cache_inputs(dst, inputs, GGML_MAX_SRC);
    for (size_t i = 0; i < input_count; ++i) {
        auto found = ctx->sparse_opencl_cache_input_bindings.find(inputs[i]);
        if (found == ctx->sparse_opencl_cache_input_bindings.end()) {
            continue;
        }
        const ggml_backend_htp_context::cache_input_binding binding = found->second;
        char reason[1024] = {};
        ggml_opencl_htp_cache_state state = GGML_OPENCL_HTP_CACHE_EMPTY;
        ggml_opencl_htp_result result = ggml_backend_opencl_htp_cache_query(
            ctx->sparse_opencl_sidecar,
            binding.cache_key.c_str(), binding.generation,
            &state, reason, sizeof(reason));
        if (result != GGML_OPENCL_HTP_OK) {
            std::fprintf(
                stderr,
                "ggml-htp: OpenCL residual cache query failed result=%d reason=%s\n",
                (int) result, reason[0] != '\0' ? reason : "unknown error");
            return -EIO;
        }
        if (state != GGML_OPENCL_HTP_CACHE_READY) {
            std::fprintf(
                stderr,
                "htp_opencl_event name=gpu_residual_cache_wait_begin "
                "ts_us=%lld key=%s generation=%" PRIu64 "\n",
                (long long) ggml_time_us(), binding.cache_key.c_str(),
                binding.generation);
            bool waited = false;
            result = ggml_backend_opencl_htp_cache_wait(
                ctx->sparse_opencl_sidecar,
                binding.cache_key.c_str(), binding.generation,
                &waited, reason, sizeof(reason));
            std::fprintf(
                stderr,
                "htp_opencl_event name=gpu_residual_cache_wait_end "
                "ts_us=%lld key=%s generation=%" PRIu64 " waited=%d\n",
                (long long) ggml_time_us(), binding.cache_key.c_str(),
                binding.generation, waited ? 1 : 0);
            if (result != GGML_OPENCL_HTP_OK) {
                std::fprintf(
                    stderr,
                    "ggml-htp: OpenCL residual cache wait failed result=%d reason=%s\n",
                    (int) result, reason[0] != '\0' ? reason : "unknown error");
                return -EIO;
            }
        }
    }
    return 0;
}
#endif

bool htp_ops_opencl_node_needs_wait(const struct ggml_tensor * dst) {
#ifdef GGML_HTP_OPENCL_SIDECAR
    const ggml_tensor * cache_inputs[GGML_MAX_SRC] = {};
    if (htp_ops_opencl_collect_cache_inputs(dst, cache_inputs, GGML_MAX_SRC) != 0) {
        return true;
    }
    if (!htp_ops_opencl_async_enabled()) {
        return false;
    }
    if (dst != nullptr && dst->op == GGML_OP_CUSTOM &&
        std::strncmp(dst->name, "htp_block_sparse_residual_", 26) == 0) {
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
    if (dst == nullptr) {
        return 0;
    }
    const int cache_wait_err = htp_ops_opencl_wait_cache_inputs(dst);
    if (cache_wait_err != 0 || !htp_ops_opencl_async_enabled()) {
        return cache_wait_err;
    }
    if (dst->op == GGML_OP_CUSTOM &&
        std::strncmp(dst->name, "htp_block_sparse_residual_", 26) == 0) {
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

void htp_ops_opencl_wait_node_completed(const struct ggml_tensor * dst) {
#ifdef GGML_HTP_OPENCL_SIDECAR
    auto * ctx = ggml_backend_htp_context::instance();
    const ggml_tensor * inputs[GGML_MAX_SRC] = {};
    const size_t input_count =
        htp_ops_opencl_collect_cache_inputs(dst, inputs, GGML_MAX_SRC);
    for (size_t i = 0; i < input_count; ++i) {
        ctx->sparse_opencl_cache_input_bindings.erase(inputs[i]);
    }
#else
    GGML_UNUSED(dst);
#endif
}

int htp_ops_opencl_wait_producer(const struct ggml_tensor * producer) {
#ifdef GGML_HTP_OPENCL_SIDECAR
    auto * ctx = ggml_backend_htp_context::instance();
    if (!htp_ops_opencl_async_enabled() || producer == nullptr ||
        ctx->sparse_opencl_routed_nodes.find(producer) ==
            ctx->sparse_opencl_routed_nodes.end()) {
        return -EINVAL;
    }

    bool waited = false;
    char reason[1024] = {};
    const ggml_opencl_htp_result result =
        ggml_backend_opencl_htp_graph_wait_for_tensor(
            ctx->sparse_opencl_graph, producer, &waited,
            reason, sizeof(reason));
    if (result != GGML_OPENCL_HTP_OK) {
        std::fprintf(
            stderr,
            "ggml-htp: OpenCL sparse producer wait failed result=%d reason=%s\n",
            (int) result, reason[0] != '\0' ? reason : "unknown error");
        return -EIO;
    }
    return 0;
#else
    GGML_UNUSED(producer);
    return -ENOSYS;
#endif
}

bool htp_ops_opencl_sparse_is_residual_owned(
        const struct ggml_tensor * producer) {
#ifdef GGML_HTP_OPENCL_SIDECAR
    const auto * ctx = ggml_backend_htp_context::instance();
    return ctx->sparse_opencl_graph != nullptr &&
        ggml_backend_opencl_htp_graph_sparse_output_is_residual_owned(
            ctx->sparse_opencl_graph, producer);
#else
    GGML_UNUSED(producer);
    return false;
#endif
}

int htp_ops_opencl_wait_sparse_input_lifetime(
        const struct ggml_tensor * producer) {
#ifdef GGML_HTP_OPENCL_SIDECAR
    auto * ctx = ggml_backend_htp_context::instance();
    if (ctx->sparse_opencl_graph == nullptr || producer == nullptr) {
        return -EINVAL;
    }
    bool waited = false;
    char reason[1024] = {};
    const ggml_opencl_htp_result result =
        ggml_backend_opencl_htp_graph_wait_for_sparse_input_lifetime(
            ctx->sparse_opencl_graph, producer, &waited,
            reason, sizeof(reason));
    if (result != GGML_OPENCL_HTP_OK) {
        std::fprintf(
            stderr,
            "ggml-htp: OpenCL sparse input lifetime wait failed result=%d reason=%s\n",
            (int) result, reason[0] != '\0' ? reason : "unknown error");
        return -EIO;
    }
    return 0;
#else
    GGML_UNUSED(producer);
    return -ENOSYS;
#endif
}

int htp_ops_opencl_wait_sparse_input_lifetime_after_dense(
        const struct ggml_tensor * dense) {
#ifdef GGML_HTP_OPENCL_SIDECAR
    auto * ctx = ggml_backend_htp_context::instance();
    if (ctx->sparse_opencl_graph == nullptr || dense == nullptr) {
        return -EINVAL;
    }
    bool waited = false;
    char reason[1024] = {};
    const ggml_opencl_htp_result result =
        ggml_backend_opencl_htp_graph_wait_for_sparse_input_lifetime_after_dense(
            ctx->sparse_opencl_graph, dense, &waited,
            reason, sizeof(reason));
    if (result == GGML_OPENCL_HTP_UNSUPPORTED) {
        return 0;
    }
    if (result != GGML_OPENCL_HTP_OK) {
        std::fprintf(
            stderr,
            "ggml-htp: OpenCL strict sparse wait failed result=%d reason=%s\n",
            (int) result, reason[0] != '\0' ? reason : "unknown error");
        return -EIO;
    }
    return 0;
#else
    GGML_UNUSED(dense);
    return -ENOSYS;
#endif
}

int htp_ops_opencl_wait_overlapping_output_leases(
        const struct ggml_tensor * writer) {
#ifdef GGML_HTP_OPENCL_SIDECAR
    const ggml_htp_opencl_lifetime_mode mode =
        ggml_htp_opencl_lifetime_mode_effective();
    if ((mode != GGML_HTP_OPENCL_LIFETIME_REUSE &&
         mode != GGML_HTP_OPENCL_LIFETIME_STRICT) || writer == nullptr) {
        return 0;
    }
    auto * ctx = ggml_backend_htp_context::instance();
    if (ctx->sparse_opencl_graph == nullptr) {
        return 0;
    }
    bool waited = false;
    char reason[1024] = {};
    const ggml_opencl_htp_result result =
        ggml_backend_opencl_htp_graph_wait_for_overlapping_output_leases(
            ctx->sparse_opencl_graph, writer, &waited,
            reason, sizeof(reason));
    if (result != GGML_OPENCL_HTP_OK) {
        std::fprintf(
            stderr,
            "ggml-htp: OpenCL output lease guard failed result=%d reason=%s\n",
            (int) result, reason[0] != '\0' ? reason : "unknown error");
        return -EIO;
    }
    return 0;
#else
    GGML_UNUSED(writer);
    return 0;
#endif
}

uint64_t htp_ops_opencl_pair_id_for_tensor(const struct ggml_tensor * tensor) {
#ifdef GGML_HTP_OPENCL_SIDECAR
    auto * ctx = ggml_backend_htp_context::instance();
    return ctx->sparse_opencl_graph != nullptr
        ? ggml_backend_opencl_htp_graph_pair_id_for_tensor(
              ctx->sparse_opencl_graph, tensor)
        : 0;
#else
    GGML_UNUSED(tensor);
    return 0;
#endif
}

static int htp_ops_opencl_compute_node_impl(struct ggml_tensor * dst, bool early_submit) {
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
        ? (early_submit
              ? ggml_backend_opencl_htp_graph_compute_block_sparse_attn_early_async(
                    ctx->sparse_opencl_graph, dst, reason, sizeof(reason))
              : ggml_backend_opencl_htp_graph_compute_block_sparse_attn_async(
                    ctx->sparse_opencl_graph, dst, reason, sizeof(reason)))
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
    GGML_UNUSED(early_submit);
    return -ENOSYS;
#endif
}

int htp_ops_opencl_compute_node(struct ggml_tensor * dst) {
    return htp_ops_opencl_compute_node_impl(dst, false);
}

int htp_ops_opencl_compute_residual_node(struct ggml_tensor * dst) {
#ifdef GGML_HTP_OPENCL_SIDECAR
    auto * ctx = ggml_backend_htp_context::instance();
    if (ctx->sparse_opencl_graph == nullptr || dst == nullptr ||
        ctx->sparse_opencl_routed_nodes.find(dst) ==
            ctx->sparse_opencl_routed_nodes.end()) {
        return -EINVAL;
    }
    char reason[1024] = {};
    ggml_opencl_htp_result result = GGML_OPENCL_HTP_ERROR;
    if (ctx->sparse_opencl_submission_thread_active) {
        // The submission thread already owns this cache-only residual job.
        // Its completion is synchronized once at graph finish, after ordinary
        // graph nodes have been allowed to continue.
        return 0;
    } else {
        result = ggml_backend_opencl_htp_graph_compute_exact_rho_residual_sync(
            ctx->sparse_opencl_graph, dst, reason, sizeof(reason));
    }
    if (result != GGML_OPENCL_HTP_OK) {
        std::fprintf(stderr,
                     "ggml-htp: OpenCL exact-rho residual failed result=%d reason=%s\n",
                     (int) result, reason[0] != '\0' ? reason : "unknown error");
        return -EIO;
    }
    return 0;
#else
    GGML_UNUSED(dst);
    return -ENOSYS;
#endif
}

int htp_ops_opencl_queue_residual_after_dense(struct ggml_tensor * dense) {
#ifdef GGML_HTP_OPENCL_SIDECAR
    auto * ctx = ggml_backend_htp_context::instance();
    if (ctx->sparse_opencl_graph == nullptr || dense == nullptr ||
        !ctx->sparse_opencl_submission_thread_active) {
        return 0;
    }
    char reason[1024] = {};
    const ggml_opencl_htp_result result =
        ggml_backend_opencl_htp_graph_queue_exact_rho_residual_after_dense(
            ctx->sparse_opencl_graph, dense, reason, sizeof(reason));
    if (result == GGML_OPENCL_HTP_UNSUPPORTED) {
        return 0;
    }
    if (result != GGML_OPENCL_HTP_OK) {
        std::fprintf(stderr,
                     "ggml-htp: queue OpenCL residual after dense failed result=%d reason=%s\n",
                     (int) result, reason[0] != '\0' ? reason : "unknown error");
        return -EIO;
    }
    htp_opencl_residual_job_just_queued.store(true, std::memory_order_release);
    return 0;
#else
    GGML_UNUSED(dense);
    return 0;
#endif
}

void htp_ops_opencl_trace_next_node_begin(const struct ggml_tensor * node) {
#ifdef GGML_HTP_OPENCL_SIDECAR
    if (!htp_opencl_residual_job_just_queued.exchange(
            false, std::memory_order_acq_rel)) {
        return;
    }
    htp_opencl_next_node_attempt_traced.store(true, std::memory_order_release);
    const char * trace = std::getenv("SD_HTP_SPARSE_OPENCL_TRACE");
    if (trace != nullptr && trace[0] != '\0' && std::strcmp(trace, "0") != 0) {
        const uint64_t pair_id = htp_ops_opencl_pair_id_for_tensor(node);
        std::fprintf(
            stderr,
            "htp_opencl_event name=next_htp_node_attempt pair=%llu ts_us=%lld producer=%s\n",
            static_cast<unsigned long long>(pair_id),
            (long long) ggml_time_us(),
            node != nullptr && node->name[0] != '\0' ? node->name : "<unnamed>");
    }
#else
    GGML_UNUSED(node);
#endif
}

void htp_ops_opencl_trace_next_node_admitted(const struct ggml_tensor * node) {
#ifdef GGML_HTP_OPENCL_SIDECAR
    if (!htp_opencl_next_node_attempt_traced.exchange(
            false, std::memory_order_acq_rel)) {
        return;
    }
    const char * trace = std::getenv("SD_HTP_SPARSE_OPENCL_TRACE");
    if (trace != nullptr && trace[0] != '\0' && std::strcmp(trace, "0") != 0) {
        const uint64_t pair_id = htp_ops_opencl_pair_id_for_tensor(node);
        std::fprintf(
            stderr,
            "htp_opencl_event name=next_htp_node_admitted pair=%llu ts_us=%lld producer=%s\n",
            static_cast<unsigned long long>(pair_id),
            (long long) ggml_time_us(),
            node != nullptr && node->name[0] != '\0' ? node->name : "<unnamed>");
    }
#else
    GGML_UNUSED(node);
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
    int debug_output_fd = -1;
    int debug_weight_fd = -1;
    int debug_activation_fd = -1;
    ssize_t debug_output_offset = 0;
    ssize_t debug_weight_offset = 0;
    ssize_t debug_activation_offset = 0;
#ifdef HTP_OPS_OP_TRACE
    uint64_t debug_vtcm_base = 0;
    uint32_t debug_vtcm_size = 0;
    int32_t debug_vtcm_ctx = 0;
    int32_t debug_hmx_ctx = 0;
    uint64_t debug_worker_pool = 0;
    int32_t debug_hmx_lock_result = 0;
    uint64_t debug_dsp_activation_hash = 0;
    uint64_t debug_dsp_output_hash = 0;
    uint64_t debug_dsp_output_hash_after_invalidate = 0;
#endif

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
                    auto & mapper = ctx->dispatch_mapper();
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
                    debug_output_fd = output_fd;
                    debug_weight_fd = weight_fd;
                    debug_activation_fd = activation_fd;
                    debug_output_offset = output_offset;
                    debug_weight_offset = weight_offset;
                    debug_activation_offset = activation_offset;
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
                auto * v = dst->src[2];

                if (env_flag_enabled("SD_HTP_PACKED_FLUX2_TRACE") &&
                    q->ne[0] == 128 && q->ne[1] == 768 && q->ne[2] == 24) {
                    std::fprintf(stderr,
                                 "flux2_attn_boundary_meta q=%s k=%s v=%s nb=[%zu,%zu,%zu] k_nb=[%zu,%zu,%zu] v_nb=[%zu,%zu,%zu]\n",
                                 ggml_type_name(q->type), ggml_type_name(k->type), ggml_type_name(v->type),
                                 q->nb[0], q->nb[1], q->nb[2], k->nb[0], k->nb[1], k->nb[2],
                                 v->nb[0], v->nb[1], v->nb[2]);
                }

                // Flux2 diagnostic: compare the first logical token/head values at the
                // attention boundary. Packed Q/K/V are HMX tiled, while dense inputs
                // are ordinary row-major tensors.
                if (env_flag_enabled("SD_HTP_PACKED_FLUX2_TRACE") &&
                    q->ne[0] == 128 && q->ne[1] == 768 && q->ne[2] == 24 &&
                    q->type == GGML_TYPE_F16 && k->type == GGML_TYPE_F16 &&
                    v->type == GGML_TYPE_F16) {
                    const size_t token_tiles = (size_t) (q->ne[1] + 31) / 32;
                    const size_t tile_elms = 1024;
                    const size_t head_stride = token_tiles * 4 * tile_elms;
                    const auto * qv = static_cast<const ggml_fp16_t *>(q->data);
                    const auto * kv = static_cast<const ggml_fp16_t *>(k->data);
                    const auto * vv = static_cast<const ggml_fp16_t *>(v->data);
                    std::fprintf(stderr, "flux2_attn_packed_boundary q=");
                    for (int i = 0; i < 8; ++i) {
                        std::fprintf(stderr, "%s%.8g", i == 0 ? "" : ",",
                                     ggml_fp16_to_fp32(qv[i]));
                    }
                    std::fprintf(stderr, " k=");
                    for (int i = 0; i < 8; ++i) {
                        std::fprintf(stderr, "%s%.8g", i == 0 ? "" : ",",
                                     ggml_fp16_to_fp32(kv[i]));
                    }
                    std::fprintf(stderr, " v=");
                    for (int i = 0; i < 8; ++i) {
                        const size_t dim_tile = (size_t) i / 32;
                        const size_t elem = dim_tile * token_tiles * tile_elms + (size_t) (i % 32);
                        std::fprintf(stderr, "%s%.8g", i == 0 ? "" : ",",
                                     ggml_fp16_to_fp32(vv[elem]));
                    }
                    std::fprintf(stderr, " stride=%zu\n", head_stride);
                } else if (env_flag_enabled("SD_HTP_PACKED_FLUX2_TRACE") &&
                           q->ne[0] == 128 && q->ne[1] == 768 && q->ne[2] == 24 &&
                           q->type == GGML_TYPE_F32 && k->type == GGML_TYPE_F32 &&
                           v->type == GGML_TYPE_F32) {
                    const auto * qv = static_cast<const float *>(q->data);
                    const auto * kv = static_cast<const float *>(k->data);
                    const auto * vv = static_cast<const float *>(v->data);
                    std::fprintf(stderr, "flux2_attn_dense_boundary q=");
                    for (int i = 0; i < 8; ++i) {
                        std::fprintf(stderr, "%s%.8g", i == 0 ? "" : ",", qv[i]);
                    }
                    std::fprintf(stderr, " k=");
                    for (int i = 0; i < 8; ++i) {
                        std::fprintf(stderr, "%s%.8g", i == 0 ? "" : ",", kv[i]);
                    }
                    std::fprintf(stderr, " v=");
                    for (int i = 0; i < 8; ++i) {
                        std::fprintf(stderr, "%s%.8g", i == 0 ? "" : ",",
                                     vv[i]);
                    }
                    std::fprintf(stderr, "\n");
                }

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

                    const ggml_flash_attn_ext_sparse_policy ggml_sparse_policy =
                        ggml_flash_attn_ext_get_sparse_policy(dst);
                    int32_t sparse_policy = HTP_SPARSE_POLICY_NONE;
                    int32_t image_tokens = 0;
                    int32_t text_tokens = 0;
                    int32_t image_block_rows = 0;
                    int32_t image_block_cols = 0;
                    int32_t halo_kv_load_mode = HTP_HALO_KV_LOAD_BASELINE;
                    bool halo_kv_load_node_override = false;
                    const int32_t halo_trace = env_flag_enabled("SD_HTP_HALO_KV_TRACE") ? 1 : 0;
                    if (ggml_sparse_policy == GGML_FLASH_ATTN_EXT_SPARSE_POLICY_STATIC_HALO) {
                        const int resolved_mode = htp_ops_resolve_halo_kv_load_mode_for_node(
                            dst, &halo_kv_load_node_override);
                        if (resolved_mode < 0) {
                            return resolved_mode;
                        }
                        sparse_policy = HTP_SPARSE_POLICY_STATIC_HALO;
                        image_tokens = ggml_flash_attn_ext_get_image_tokens(dst);
                        text_tokens = ggml_flash_attn_ext_get_text_tokens(dst);
                        image_block_rows = ggml_flash_attn_ext_get_image_block_rows(dst);
                        image_block_cols = ggml_flash_attn_ext_get_image_block_cols(dst);
                        halo_kv_load_mode = resolved_mode;
                        if (halo_trace != 0) {
                            std::fprintf(stderr,
                                         "htp_halo_kv_dispatch policy=%d image_tokens=%d text_tokens=%d "
                                         "image_block_rows=%d image_block_cols=%d loader=%d source=%s\n",
                                         sparse_policy,
                                         image_tokens,
                                         text_tokens,
                                         image_block_rows,
                                         image_block_cols,
                                         halo_kv_load_mode,
                                         halo_kv_load_node_override ? "node" : "env");
                        }
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
                        .sparse_policy = sparse_policy,
                        .image_tokens = image_tokens,
                        .text_tokens = text_tokens,
                        .halo_kv_load_mode = halo_kv_load_mode,
                        .halo_trace = halo_trace,
                        .image_block_rows = image_block_rows,
                        .image_block_cols = image_block_cols,
                        .halo_exec_mode = -1,
                        .halo_image_rows_updated = 0,
                        .halo_text_q_image_row_updates = 0,
                        .halo_text_q_image_block_updates = 0,
                        .halo_image_q_batch_updates = 0,
                        .halo_image_q_segment_updates = 0,
                    };
                    *reinterpret_cast<BlockSparseAttnParams *>(param_buf) = op_params;

                    op_index   = is_packed_sparse_attention(dst) ?
                        HTP_OPS_BLOCK_SPARSE_ATTN_QKV_PACKED_F16 :
                        HTP_OPS_BLOCK_SPARSE_ATTN_QO_F32_KV_F16;
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
                if (is_fused_anchor_attention_custom(dst)) {
                    prepare_tensor_rpcmem_mapping(dst);
                    auto mappings = get_all_rpcmem_mappings(dst);
                    if (mappings.size() != 6) {
                        dump_missing_mapping_diagnostics(dst, mappings);
                        return -EINVAL;
                    }
                    const ggml_htp_fused_anchor_metadata * metadata = fused_anchor_metadata(dst);
                    FusedAnchorAttnParams op_params = {};
                    op_params.output = {mappings[0].first, static_cast<int32_t>(mappings[0].second)};
                    op_params.q = {mappings[1].first, static_cast<int32_t>(mappings[1].second)};
                    op_params.k = {mappings[2].first, static_cast<int32_t>(mappings[2].second)};
                    op_params.v = {mappings[3].first, static_cast<int32_t>(mappings[3].second)};
                    op_params.dense_mask = {mappings[4].first, static_cast<int32_t>(mappings[4].second)};
                    op_params.block_mask = {mappings[5].first, static_cast<int32_t>(mappings[5].second)};
                    op_params.qo_len = metadata->qo_len;
                    op_params.kv_len = metadata->kv_len;
                    op_params.n_heads = metadata->n_heads;
                    op_params.n_kv_heads = metadata->n_kv_heads;
                    op_params.head_dim = metadata->head_dim;
                    op_params.block_q = metadata->block_q;
                    op_params.block_k = metadata->block_k;
                    op_params.n_q_blocks = metadata->n_q_blocks;
                    op_params.n_k_blocks = metadata->n_k_blocks;
                    op_params.max_kept_blocks = metadata->max_kept_blocks;
                    op_params.mask_format = metadata->mask_format;
                    op_params.sparse_policy = metadata->sparse_policy;
                    op_params.image_tokens = metadata->image_tokens;
                    op_params.text_tokens = metadata->text_tokens;
                    op_params.image_block_rows = metadata->image_block_rows;
                    op_params.image_block_cols = metadata->image_block_cols;
                    op_params.elapsed_us = 0;
                    *reinterpret_cast<FusedAnchorAttnParams *>(param_buf) = op_params;
                    op_index = HTP_OPS_FUSED_ANCHOR_ATTN_QO_F32_KV_F16;
                    args_size = sizeof(op_params);
                    dispatch_m = op_params.qo_len;
                    dispatch_k = op_params.kv_len;
                    dispatch_n = op_params.n_heads;
                    break;
                }
                if (is_qkv_mul_rms_packed_dual_rope_custom(dst)) {
                    prepare_tensor_rpcmem_mapping_with_extra(dst, dst->src[0]);
                    const auto & mapper = ggml_backend_htp_context::instance()->dispatch_mapper();
                    std::vector<std::pair<int, ssize_t>> mappings;
                    mappings.push_back(mapper.get_tensor_mapping(dst));
                    for (int i = 0; i <= 10; ++i) {
                        mappings.push_back(dst->src[i] != nullptr ? mapper.get_tensor_mapping(dst->src[i]) : std::make_pair(-1, (ssize_t) 0));
                    }
                    const int image_tokens = (int) dst->src[0]->ne[1];
                    const int text_tokens = (int) dst->src[5]->ne[1];
                    const int input_dim = (int) dst->src[0]->ne[0];
                    const int head_dim = (int) dst->src[3]->ne[0];
                    int n_heads = 0, weight_head_dim = 0;
                    if (!packed_qkv_weight_shape(dst->src[1], &n_heads, &weight_head_dim) ||
                        weight_head_dim != head_dim) {
                        return -EINVAL;
                    }
                    const size_t token_tiles = (size_t) (image_tokens + text_tokens + 31) / 32;
                    const size_t qkv_size = (size_t) n_heads * token_tiles * (size_t) (head_dim / 32) * 1024 * sizeof(uint16_t);
                    auto [packed_fd, packed_offset] = mappings[0];
                    QkvMulRmsPackedDualRopeParams op_params{
                        .q = { packed_fd, (int32_t) packed_offset },
                        .k_out = { packed_fd, (int32_t) (packed_offset + qkv_size) },
                        .v = { packed_fd, (int32_t) (packed_offset + 2 * qkv_size) },
                        .image_activation = { mappings[1].first, (int32_t) mappings[1].second },
                        .image_weight = { mappings[2].first, (int32_t) mappings[2].second },
                        .image_bias = { mappings[3].first, (int32_t) mappings[3].second },
                        .image_q_norm = { mappings[4].first, (int32_t) mappings[4].second },
                        .image_k_norm = { mappings[5].first, (int32_t) mappings[5].second },
                        .text_activation = { mappings[6].first, (int32_t) mappings[6].second },
                        .text_weight = { mappings[7].first, (int32_t) mappings[7].second },
                        .text_bias = { mappings[8].first, (int32_t) mappings[8].second },
                        .text_q_norm = { mappings[9].first, (int32_t) mappings[9].second },
                        .text_k_norm = { mappings[10].first, (int32_t) mappings[10].second },
                        .rope_pe = { mappings[11].first, (int32_t) mappings[11].second },
                        .image_tokens = image_tokens, .text_tokens = text_tokens,
                        .input_dim = input_dim, .n_heads = n_heads, .head_dim = head_dim,
                        .image_rope_token_offset = 0,
                        .text_rope_token_offset = image_tokens,
                        .elapsed_us = 0,
                    };
                    *reinterpret_cast<QkvMulRmsPackedDualRopeParams *>(param_buf) = op_params;
                    op_index = HTP_OPS_QKV_MUL_RMS_PACKED_DUAL_ROPE_W16A32;
                    args_size = sizeof(op_params);
                    dispatch_m = image_tokens + text_tokens;
                    dispatch_k = input_dim;
                    dispatch_n = 3 * n_heads * head_dim;
                    if (env_flag_enabled("SD_HTP_PACKED_FLUX2_TRACE")) {
                        std::fprintf(stderr,
                                     "[ggml-htp-trace] dispatch qkv_mul_rms_packed_dual_rope image=%d text=%d input=%d head_dim=%d\n",
                                     image_tokens, text_tokens, input_dim, head_dim);
                    }
                    break;
                }
                if (is_qkv_mul_rms_packed_rope_custom(dst)) {
                    prepare_tensor_rpcmem_mapping_with_extra(dst, dst->src[1]);
                    const auto & mapper = ggml_backend_htp_context::instance()->dispatch_mapper();
                    std::vector<std::pair<int, ssize_t>> mappings;
                    mappings.push_back(mapper.get_tensor_mapping(dst));
                    for (int i = 0; i <= 5; ++i) {
                        mappings.push_back(dst->src[i] != nullptr ? mapper.get_tensor_mapping(dst->src[i]) : std::make_pair(-1, (ssize_t) 0));
                    }
                    const int tokens = (int) dst->src[0]->ne[1];
                    const int input_dim = (int) dst->src[0]->ne[0];
                    const int head_dim = (int) dst->src[3]->ne[0];
                    int n_heads = 0, weight_head_dim = 0;
                    if (!packed_qkv_weight_shape(dst->src[1], &n_heads, &weight_head_dim) ||
                        weight_head_dim != head_dim) {
                        return -EINVAL;
                    }
                    const size_t token_tiles = (size_t) (tokens + 31) / 32;
                    const size_t qkv_size = (size_t) n_heads * token_tiles * (size_t) (head_dim / 32) * 1024 * sizeof(uint16_t);
                    auto [packed_fd, packed_offset] = mappings[0];
                    QkvMulRmsPackedRopeParams op_params{
                        .q = { packed_fd, (int32_t) packed_offset },
                        .k_out = { packed_fd, (int32_t) (packed_offset + qkv_size) },
                        .v = { packed_fd, (int32_t) (packed_offset + 2 * qkv_size) },
                        .activation = { mappings[1].first, (int32_t) mappings[1].second },
                        .weight = { mappings[2].first, (int32_t) mappings[2].second },
                        .bias = { mappings[3].first, (int32_t) mappings[3].second },
                        .q_norm = { mappings[4].first, (int32_t) mappings[4].second },
                        .k_norm = { mappings[5].first, (int32_t) mappings[5].second },
                        .rope_pe = { mappings[6].first, (int32_t) mappings[6].second },
                        .tokens = tokens, .input_dim = input_dim,
                        .n_heads = n_heads, .head_dim = head_dim,
                        .rope_token_offset = 0, .rope_token_count = tokens,
                        .elapsed_us = 0,
                    };
                    *reinterpret_cast<QkvMulRmsPackedRopeParams *>(param_buf) = op_params;
                    op_index = HTP_OPS_QKV_MUL_RMS_PACKED_ROPE_W16A32;
                    args_size = sizeof(op_params);
                    dispatch_m = tokens;
                    dispatch_k = input_dim;
                    dispatch_n = 3 * n_heads * head_dim;
                    if (env_flag_enabled("SD_HTP_PACKED_FLUX2_TRACE")) {
                        std::fprintf(stderr,
                                     "[ggml-htp-trace] dispatch qkv_mul_rms_packed_rope tokens=%d input=%d head_dim=%d\n",
                                     tokens, input_dim, head_dim);
                    }
                    break;
                }
                if (is_qkv_mul_rms_packed_dual_custom(dst)) {
                    prepare_tensor_rpcmem_mapping_with_extra(dst, dst->src[0]);
                    const auto & mapper = ggml_backend_htp_context::instance()->dispatch_mapper();
                    std::vector<std::pair<int, ssize_t>> mappings;
                    mappings.push_back(mapper.get_tensor_mapping(dst));
                    for (int i = 0; i < 10; ++i) {
                        mappings.push_back(dst->src[i] != nullptr ? mapper.get_tensor_mapping(dst->src[i]) : std::make_pair(-1, (ssize_t) 0));
                    }
                    const int image_tokens = (int) dst->src[0]->ne[1];
                    const int text_tokens = (int) dst->src[5]->ne[1];
                    const int input_dim = (int) dst->src[0]->ne[0];
                    const int head_dim = (int) dst->src[3]->ne[0];
                    int n_heads = 0, weight_head_dim = 0;
                    if (!packed_qkv_weight_shape(dst->src[1], &n_heads, &weight_head_dim) ||
                        weight_head_dim != head_dim) {
                        return -EINVAL;
                    }
                    const size_t token_tiles = (size_t) (image_tokens + text_tokens + 31) / 32;
                    const size_t qkv_size = (size_t) n_heads * token_tiles * (size_t) (head_dim / 32) * 1024 * sizeof(uint16_t);
                    auto [packed_fd, packed_offset] = mappings[0];
                    QkvMulRmsPackedDualParams op_params{
                        .q = { packed_fd, (int32_t) packed_offset },
                        .k_out = { packed_fd, (int32_t) (packed_offset + qkv_size) },
                        .v = { packed_fd, (int32_t) (packed_offset + 2 * qkv_size) },
                        .image_activation = { mappings[1].first, (int32_t) mappings[1].second },
                        .image_weight = { mappings[2].first, (int32_t) mappings[2].second },
                        .image_bias = { mappings[3].first, (int32_t) mappings[3].second },
                        .image_q_norm = { mappings[4].first, (int32_t) mappings[4].second },
                        .image_k_norm = { mappings[5].first, (int32_t) mappings[5].second },
                        .text_activation = { mappings[6].first, (int32_t) mappings[6].second },
                        .text_weight = { mappings[7].first, (int32_t) mappings[7].second },
                        .text_bias = { mappings[8].first, (int32_t) mappings[8].second },
                        .text_q_norm = { mappings[9].first, (int32_t) mappings[9].second },
                        .text_k_norm = { mappings[10].first, (int32_t) mappings[10].second },
                        .image_tokens = image_tokens, .text_tokens = text_tokens,
                        .input_dim = input_dim, .n_heads = n_heads, .head_dim = head_dim, .elapsed_us = 0,
                    };
                    *reinterpret_cast<QkvMulRmsPackedDualParams *>(param_buf) = op_params;
                    op_index = HTP_OPS_QKV_MUL_RMS_PACKED_DUAL_W16A32;
                    args_size = sizeof(op_params);
                    dispatch_m = image_tokens + text_tokens;
                    dispatch_k = input_dim;
                    dispatch_n = 3 * n_heads * head_dim;
                    break;
                }
                if (is_qkv_mul_rms_packed_custom(dst)) {
                    prepare_tensor_rpcmem_mapping_with_extra(dst, dst->src[1]);
                    auto mappings = get_all_rpcmem_mappings(dst);
                    if (mappings.size() != 6) {
                        dump_missing_mapping_diagnostics(dst, mappings);
                        return -EINVAL;
                    }
                    const int tokens = (int) dst->src[0]->ne[1];
                    const int input_dim = (int) dst->src[0]->ne[0];
                    const int head_dim = (int) dst->src[3]->ne[0];
                    int n_heads = 0, weight_head_dim = 0;
                    if (!packed_qkv_weight_shape(dst->src[1], &n_heads, &weight_head_dim) ||
                        weight_head_dim != head_dim) {
                        return -EINVAL;
                    }
                    const size_t token_tiles = (tokens + 31) / 32;
                    const size_t qkv_size = (size_t) n_heads * token_tiles * (size_t) (head_dim / 32) * 1024 * sizeof(uint16_t);
                    auto [packed_fd, packed_offset] = mappings[0];
                    auto [activation_fd, activation_offset] = mappings[1];
                    auto [weight_fd, weight_offset] = mappings[2];
                    auto [bias_fd, bias_offset] = mappings[3];
                    auto [qnorm_fd, qnorm_offset] = mappings[4];
                    auto [knorm_fd, knorm_offset] = mappings[5];
                    QkvMulRmsPackedParams op_params{
                        .q = { packed_fd, (int32_t) packed_offset },
                        .k_out = { packed_fd, (int32_t) (packed_offset + qkv_size) },
                        .v = { packed_fd, (int32_t) (packed_offset + 2 * qkv_size) },
                        .activation = { activation_fd, (int32_t) activation_offset },
                        .weight = { weight_fd, (int32_t) weight_offset },
                        .bias = { bias_fd, (int32_t) bias_offset },
                        .q_norm = { qnorm_fd, (int32_t) qnorm_offset },
                        .k_norm = { knorm_fd, (int32_t) knorm_offset },
                        .tokens = tokens, .input_dim = input_dim,
                        .n_heads = n_heads, .head_dim = head_dim, .elapsed_us = 0,
                    };
                    *reinterpret_cast<QkvMulRmsPackedParams *>(param_buf) = op_params;
                    op_index = HTP_OPS_QKV_MUL_RMS_PACKED_W16A32;
                    args_size = sizeof(op_params);
                    dispatch_m = tokens; dispatch_k = input_dim; dispatch_n = 3 * n_heads * head_dim;
                    break;
                }
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

    const bool isolate_all_outputs =
        env_flag_enabled("SD_HTP_MULTI_SESSION_INDEPENDENT_ALL_OUTPUT") &&
        ctx->reserved_session != nullptr && ctx->dispatch_mapper().domain() == ctx->reserved_domain_id;
    htp_isolated_dispatch_output isolated_dispatch_output;
    if (isolate_all_outputs) {
        auto * output_descriptor = reinterpret_cast<RpcmemBufAddr *>(param_buf);
        const int original_output_fd = output_descriptor->fd;
        const ssize_t original_output_offset = output_descriptor->offset;
        if (!isolated_dispatch_output.init(ctx->dispatch_mapper().domain(), dst) ||
            !isolated_dispatch_output.bind(op_index, param_buf, original_output_fd,
                                           original_output_offset)) {
            std::fprintf(stderr,
                         "[ggml-htp] independent all-output isolation failed op=%d tensor=%s\n",
                         op_index, dst->name[0] != '\0' ? dst->name : "<unnamed>");
            return -ENOMEM;
        }
    }

    int  n_reqs         = 1;
    int  n_unmap_fds    = (int) ctx->dispatch_mapper().get_pending_unmap_reqs().size();
    bool has_unmap_reqs = n_unmap_fds > 0;
    if (has_unmap_reqs) {
        ++n_reqs;
    }

    size_t op_req_size = sizeof(RequestHeader) + sizeof(OpComputeRequest) + args_size;
    if (has_unmap_reqs) {
        const int mapper_domain = ctx->dispatch_mapper().domain();
        const int mapper_session = htp_rpcmem_session_index_for_domain(ctx->sessions, mapper_domain);
        const auto * runtime = ctx->session_runtime(mapper_session);
        const bool use_v2_map = ctx->multi_session_enabled && runtime != nullptr;
        const size_t entry_size = use_v2_map ? sizeof(RpcmemMapEntryV2) : sizeof(int32_t);
        const size_t map_req_size = sizeof(RequestHeader) +
            (use_v2_map ? sizeof(RpcmemMapRequestV2) + static_cast<size_t>(n_unmap_fds) * entry_size
                        : sizeof(RpcmemMapRequest) + static_cast<size_t>(n_unmap_fds) * entry_size);
        const size_t total_size = sizeof(MessageHeader) + 3 * sizeof(int32_t) +
            op_req_size + map_req_size;
        if (total_size > ggml_backend_htp_context::MAX_MSG_SIZE) {
            if (!ctx->dispatch_mapper().flush_pending_unmaps()) {
                std::fprintf(stderr,
                             "ggml-htp: failed to flush %d pending RPCMEM puts before oversized op message\n",
                             n_unmap_fds);
                return -ENOMEM;
            }
            n_unmap_fds = 0;
            has_unmap_reqs = false;
            n_reqs = 1;
        }
    }
    auto * msg_hdr = reinterpret_cast<MessageHeader *>(ctx->message_channel());

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
        auto & dispatch_mapper = ctx->dispatch_mapper();
        const int mapper_domain = dispatch_mapper.domain();
        const int mapper_session = htp_rpcmem_session_index_for_domain(ctx->sessions, mapper_domain);
        auto * runtime = ctx->session_runtime(mapper_session);
        if (env_flag_enabled("SD_HTP_RPCMEM_TRACE")) {
            std::fprintf(stderr,
                         "[rpcmem-trace] op_pending_put mapper_session=%d mapper_domain=%d "
                         "active_session=%d active_domain=%d runtime_domain=%d channel=%p pending=%d\n",
                         mapper_session, mapper_domain, ctx->active_session_index,
                         ctx->active_mapper().domain(), runtime != nullptr ? static_cast<int>(runtime->domain) : -1,
                         ctx->message_channel(), n_unmap_fds);
        }
        const bool use_v2_map = ctx->multi_session_enabled && runtime != nullptr;
        size_t map_req_size = sizeof(RequestHeader) +
            (use_v2_map ? sizeof(RpcmemMapRequestV2) + n_unmap_fds * sizeof(RpcmemMapEntryV2)
                        : sizeof(RpcmemMapRequest) + n_unmap_fds * sizeof(int32_t));
        msg_hdr->req_offsets[2] = msg_hdr->req_offsets[1] + map_req_size;

        RequestHeader req_hdr{
            .state = 0,
            .type  = REQUEST_TYPE_RPCMEM_MAP,
        };
        auto * p = reinterpret_cast<uint8_t *>(message_header_get_request_ptr(msg_hdr, 1));
        write_buf(p, req_hdr);
        if (use_v2_map) {
            RpcmemMapRequestV2 map_req{
                .magic = RPCMEM_MAP_REQUEST_SESSION_MAGIC,
                .version = RPCMEM_MAP_REQUEST_VERSION_SESSION,
                .header_size = sizeof(RpcmemMapRequestV2),
                .session_token = runtime->id == 0 ? 0 :
                    ((static_cast<uint64_t>(runtime->id) << 32) | runtime->domain),
                .n_puts = n_unmap_fds,
                .n_gets = 0,
            };
            write_buf(p, map_req);
            for (const auto & req : dispatch_mapper.get_pending_unmap_reqs()) {
                RpcmemMapEntryV2 entry{req.fd, req.role, req.offset, req.buf_size, 0};
                write_buf(p, entry);
            }
        } else {
            RpcmemMapRequest map_req{n_unmap_fds, 0};
            write_buf(p, map_req);
            for (const auto & req : dispatch_mapper.get_pending_unmap_reqs()) write_buf(p, req.fd);
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

    if (htp_direct_reserved_matmul_enabled(ctx) && debug_f16_matmul) {
        auto direct_matmul = reinterpret_cast<direct_f16_matmul_trace_session_fn *>(
            dlsym(ops_dl_handle, "htp_ops_rpc_mat_mul_permuted_w16a32_trace_session"));
        if (direct_matmul == nullptr) {
            std::fprintf(stderr,
                         "[ggml-htp] direct reserved matmul requested but libhtp_ops.so lacks "
                         "htp_ops_rpc_mat_mul_permuted_w16a32_trace_session\n");
            return -ENOSYS;
        }

        // Pending puts are encoded on the custom channel. Drain them before
        // entering direct FastRPC so the direct call sees the same mapper
        // state as the normal channel path.
        if (!ctx->dispatch_mapper().flush_pending_unmaps()) {
            return -EIO;
        }
        uint64_t direct_dsp_activation_hash = 0;
        uint64_t direct_dsp_output_hash = 0;
        uint64_t direct_dsp_output_hash_after_remap = 0;
        const bool use_independent_output =
            env_flag_enabled("SD_HTP_MULTI_SESSION_INDEPENDENT_OUTPUT") &&
            htp_debug_hash_filter_matches(debug_weight);
        const bool use_independent_activation =
            env_flag_enabled("SD_HTP_MULTI_SESSION_INDEPENDENT_ACTIVATION") &&
            htp_debug_hash_filter_matches(debug_weight);
        htp_independent_output independent_output;
        htp_independent_activation independent_activation;
        int direct_output_fd = debug_output_fd;
        int direct_output_offset = static_cast<int>(debug_output_offset);
        int direct_activation_fd = debug_activation_fd;
        int direct_activation_offset = static_cast<int>(debug_activation_offset);
        if (use_independent_output) {
            if (!independent_output.init(ctx->dispatch_mapper().domain(), ggml_nbytes(dst))) {
                std::fprintf(stderr,
                             "[ggml-htp] independent output allocation failed for weight=%s bytes=%zu\n",
                             debug_weight != nullptr && debug_weight->name[0] != '\0'
                                 ? debug_weight->name : "<unnamed>",
                             ggml_nbytes(dst));
                return -ENOMEM;
            }
            direct_output_fd = independent_output.fd;
            direct_output_offset = 0;
        }
        if (use_independent_activation) {
            if (!independent_activation.init(ctx->dispatch_mapper().domain(),
                                             debug_activation->data,
                                             ggml_nbytes(debug_activation))) {
                std::fprintf(stderr,
                             "[ggml-htp] independent activation allocation failed for weight=%s bytes=%zu\n",
                             debug_weight != nullptr && debug_weight->name[0] != '\0'
                                 ? debug_weight->name : "<unnamed>",
                             ggml_nbytes(debug_activation));
                return -ENOMEM;
            }
            direct_activation_fd = independent_activation.fd;
            direct_activation_offset = 0;
        }
        const int direct_ret = direct_matmul(
            ctx->reserved_session,
            direct_output_fd,
            direct_output_offset,
            direct_activation_fd,
            direct_activation_offset,
            debug_weight_fd,
            static_cast<int>(debug_weight_offset),
            dispatch_m,
            dispatch_k,
            dispatch_n,
            &direct_dsp_activation_hash,
            &direct_dsp_output_hash,
            &direct_dsp_output_hash_after_remap);

        if (direct_ret == 0) {
            (void) htp_host_dma_buf_sync_read(direct_output_fd);
        }
        const uint64_t direct_independent_output_hash = use_independent_output
            ? htp_fnv1a64_update(UINT64_C(14695981039346656037),
                                  independent_output.ptr, independent_output.size)
            : 0;
        if (direct_ret == 0 && use_independent_output) {
            // Preserve the graph's normal tensor ownership after observing the
            // isolated Host readback, so only this output allocation differs.
            std::memcpy(dst->data, independent_output.ptr, independent_output.size);
        }
        const uint64_t direct_weight_hash = htp_hash_tensor_bytes(debug_weight);
        const uint64_t direct_packed_weight_hash = htp_hash_tensor_bytes(debug_packed_weight);
        const uint64_t direct_output_hash = htp_hash_tensor_bytes(dst);
        std::fprintf(stderr,
                     "[ggml-htp-hash] idx=%d ret=%d direct=1 weight=%s packed=%s self_packed=%d "
                     "mkn=[%d,%d,%d] domain=%d weight_hash=%016" PRIx64 " packed_hash=%016" PRIx64 " "
                     "fds=[%d,%d,%d] offsets=[%zd,%zd,%zd] act_bytes=%zu act_hash=%016" PRIx64 " "
                     "dsp_act_hash=%016" PRIx64 " dsp_out_hash=%016" PRIx64 " "
                     "dsp_out_hash_after_remap=%016" PRIx64 " "
                     "isolated_output=%d isolated_fd=%d isolated_host_hash=%016" PRIx64 " "
                     "isolated_activation=%d isolated_activation_fd=%d "
                     "out_bytes=%zu out_hash=%016" PRIx64 "\n",
                     debug_hash_index,
                     direct_ret,
                     debug_weight != nullptr && debug_weight->name[0] != '\0' ? debug_weight->name : "<unnamed>",
                     debug_packed_weight != nullptr && debug_packed_weight->name[0] != '\0' ? debug_packed_weight->name : "<unnamed>",
                     debug_weight == debug_packed_weight ? 1 : 0,
                     dispatch_m,
                     dispatch_k,
                     dispatch_n,
                     ctx->dispatch_mapper().domain(),
                     direct_weight_hash,
                     direct_packed_weight_hash,
                     debug_output_fd,
                     debug_weight_fd,
                     debug_activation_fd,
                     debug_output_offset,
                     debug_weight_offset,
                     debug_activation_offset,
                     debug_activation != nullptr ? ggml_nbytes(debug_activation) : 0,
                     debug_activation_hash,
                     direct_dsp_activation_hash,
                     direct_dsp_output_hash,
                     direct_dsp_output_hash_after_remap,
                     use_independent_output ? 1 : 0,
                     use_independent_output ? independent_output.fd : -1,
                     direct_independent_output_hash,
                     use_independent_activation ? 1 : 0,
                     use_independent_activation ? independent_activation.fd : -1,
                     dst != nullptr ? ggml_nbytes(dst) : 0,
                     direct_output_hash);
        return direct_ret;
    }

    // The doorbell closes the host submit window: the request is written, the
    // checksum is in place, and the DSP is told to pick the request up.  The
    // timestamp is what lets a caller split the host window of the node into a
    // submit part (in front of the doorbell) and a wait part (behind it).
    ctx->last_dispatch_doorbell_us.store(ggml_time_us(), std::memory_order_release);
    std::atomic_store_explicit(v0_ptr, uint8_t{ 1 }, std::memory_order_release);

    const int timeout_ms = htp_dispatch_timeout_ms();
    const int64_t wait_begin_ms = ggml_time_ms();
    while (std::atomic_load_explicit(v1_ptr, std::memory_order_acquire) == 0) {
        const int waited_ms = static_cast<int>(ggml_time_ms() - wait_begin_ms);
        if (timeout_ms > 0 && waited_ms >= timeout_ms) {
            dump_htp_dispatch_state("dispatch-timeout", dst, op_index, dispatch_m, dispatch_k, dispatch_n, waited_ms);
            d_ptr->store(0, std::memory_order_relaxed);
            if (has_unmap_reqs) {
                ctx->dispatch_mapper().flush_pending_unmaps();
            }
            return -ETIMEDOUT;
        }
        usleep(1000);
    }
    d_ptr->store(0, std::memory_order_relaxed);

    if (has_unmap_reqs) {
        if (!ctx->dispatch_mapper().unmap_all_pending_buffers()) {
            return -EIO;
        }
    }

    isolated_dispatch_output.copy_back();

    std::atomic_thread_fence(std::memory_order_acquire);
    auto * completed_request = message_header_get_request_ptr(msg_hdr, 0);
    int ret = completed_request->state;
    auto * compute_request = reinterpret_cast<OpComputeRequest *>(completed_request->data);
    if (ret == 0 && debug_f16_matmul) {
        (void) htp_host_dma_buf_sync_read(debug_output_fd);
        htp_host_cache_probe(debug_output_fd, debug_output_offset, dst->data, ggml_nbytes(dst));
    }
    if (env_flag_enabled("SD_HTP_PACKED_FLUX2_TRACE") && debug_f16_matmul && dst != nullptr &&
        dst->type == GGML_TYPE_F32 && dispatch_m == 768 && dispatch_k == 3072 &&
        (dispatch_n == 9216 || dispatch_n == 27648)) {
        const float * values = static_cast<const float *>(dst->data);
        std::fprintf(stderr, "flux2_dense_matmul_debug m=%d n=%d values=", dispatch_m, dispatch_n);
        for (int i = 0; i < 8; ++i) {
            std::fprintf(stderr, "%s%.8g", i == 0 ? "" : ",", values[i]);
        }
        std::fprintf(stderr, "\n");
    }
    const auto * block_sparse_params =
        (op_index == HTP_OPS_BLOCK_SPARSE_ATTN_QO_F32_KV_F16 ||
         op_index == HTP_OPS_BLOCK_SPARSE_ATTN_QKV_PACKED_F16)
        ? reinterpret_cast<const BlockSparseAttnParams *>(compute_request->payload)
        : nullptr;
    const auto * fused_anchor_params = op_index == HTP_OPS_FUSED_ANCHOR_ATTN_QO_F32_KV_F16
        ? reinterpret_cast<const FusedAnchorAttnParams *>(compute_request->payload)
        : nullptr;
    const int32_t attention_elapsed_us = op_index == HTP_OPS_FLASH_ATTN_QO_F32_KV_F16
        ? reinterpret_cast<const FlashAttnParams *>(compute_request->payload)->elapsed_us
        : (block_sparse_params != nullptr ? block_sparse_params->elapsed_us
                                          : (fused_anchor_params != nullptr ? fused_anchor_params->elapsed_us : -1));
    if (ret == 0 && attention_elapsed_us >= 0) {
        ctx->last_attention_elapsed_us.store(attention_elapsed_us, std::memory_order_release);
    }
    const char * internal_trace = std::getenv("SD_HTP_ATTN_INTERNAL_TRACE");
    if (internal_trace != nullptr && internal_trace[0] != '\0' && std::strcmp(internal_trace, "0") != 0 &&
        (op_index == HTP_OPS_FLASH_ATTN_QO_F32_KV_F16 || block_sparse_params != nullptr ||
         fused_anchor_params != nullptr)) {
        std::fprintf(
            stderr,
            "htp_attention_internal kind=%s elapsed_us=%d qo=%d kv=%d heads=%d with_lse=%d "
            "halo_exec_mode=%d image_rows=%d text_q_rows=%d text_q_block_updates=%d "
            "image_q_batches=%d image_q_segments=%d\n",
            op_index == HTP_OPS_FLASH_ATTN_QO_F32_KV_F16 ? "dense" :
                (fused_anchor_params != nullptr ? "fused_anchor" : "block_sparse"),
            attention_elapsed_us,
            dispatch_m,
            dispatch_k,
            dispatch_n,
            ggml_get_op_params_i32(dst, 6) != 0 ? 1 : 0,
            block_sparse_params != nullptr ? block_sparse_params->halo_exec_mode : -1,
            block_sparse_params != nullptr ? block_sparse_params->halo_image_rows_updated : 0,
            block_sparse_params != nullptr ? block_sparse_params->halo_text_q_image_row_updates : 0,
            block_sparse_params != nullptr ? block_sparse_params->halo_text_q_image_block_updates : 0,
            block_sparse_params != nullptr ? block_sparse_params->halo_image_q_batch_updates : 0,
            block_sparse_params != nullptr ? block_sparse_params->halo_image_q_segment_updates : 0);
    }
    if (block_sparse_params != nullptr && env_flag_enabled("SD_HTP_HALO_REQUIRE_USERDMA_TRACE")) {
        const int text_q_blocks = (block_sparse_params->text_tokens + 31) / 32;
        const int expected_image_rows = block_sparse_params->text_tokens > 0
            ? block_sparse_params->image_block_rows
            : 0;
        const int expected_row_updates = block_sparse_params->image_block_rows * text_q_blocks;
        const int expected_block_updates = expected_row_updates * block_sparse_params->image_block_cols;
        const int expected_image_q_batches =
            block_sparse_params->image_block_rows * block_sparse_params->image_block_cols;
        const int expected_image_q_segments =
            block_sparse_params->image_block_cols *
            (3 * block_sparse_params->image_block_rows - 2 +
             (block_sparse_params->text_tokens > 0 ? block_sparse_params->image_block_rows : 0));
        if (ret != 0 || block_sparse_params->halo_exec_mode != HTP_HALO_EXEC_TRACE_USERDMA_ROWS ||
            block_sparse_params->halo_image_rows_updated != expected_image_rows ||
            block_sparse_params->halo_text_q_image_row_updates != expected_row_updates ||
            block_sparse_params->halo_text_q_image_block_updates != expected_block_updates ||
            block_sparse_params->halo_image_q_batch_updates != expected_image_q_batches ||
            block_sparse_params->halo_image_q_segment_updates != expected_image_q_segments) {
            std::fprintf(
                stderr,
                "htp_halo_userdma_trace_mismatch ret=%d mode=%d rows=%d/%d text_q_rows=%d/%d "
                "text_q_blocks=%d/%d image_q_batches=%d/%d image_q_segments=%d/%d\n",
                ret,
                block_sparse_params->halo_exec_mode,
                block_sparse_params->halo_image_rows_updated,
                expected_image_rows,
                block_sparse_params->halo_text_q_image_row_updates,
                expected_row_updates,
                block_sparse_params->halo_text_q_image_block_updates,
                expected_block_updates,
                block_sparse_params->halo_image_q_batch_updates,
                expected_image_q_batches,
                block_sparse_params->halo_image_q_segment_updates,
                expected_image_q_segments);
            ret = -EIO;
        }
    }
    if (block_sparse_params != nullptr && env_flag_enabled("SD_HTP_HALO_REQUIRE_L2_TRACE")) {
        if (ret != 0 || block_sparse_params->halo_exec_mode != HTP_HALO_EXEC_TRACE_L2_PREFETCH) {
            std::fprintf(stderr,
                         "htp_halo_l2_trace_mismatch ret=%d mode=%d expected=%d\n",
                         ret, block_sparse_params->halo_exec_mode,
                         HTP_HALO_EXEC_TRACE_L2_PREFETCH);
            ret = -EIO;
        }
    }
    if (env_flag_enabled("SD_HTP_PACKED_FLUX2_TRACE") && dst != nullptr &&
        (std::strcmp(dst->name, "qkv_mul_rms_packed_rope") == 0 ||
         std::strcmp(dst->name, "flux2_single_dense_qkv_debug") == 0)) {
        std::fprintf(stderr, "flux2_qkv_debug name=%s type=%s bytes=%zu values=",
                     dst->name, ggml_type_name(dst->type), ggml_nbytes(dst));
        const int count = 8;
        if (dst->type == GGML_TYPE_F16) {
            const ggml_fp16_t * values = static_cast<const ggml_fp16_t *>(dst->data);
            for (int i = 0; i < count; ++i) {
                std::fprintf(stderr, "%s%.8g", i == 0 ? "" : ",", ggml_fp16_to_fp32(values[i]));
            }
        } else if (dst->type == GGML_TYPE_F32) {
            const float * values = static_cast<const float *>(dst->data);
            for (int i = 0; i < count; ++i) {
                std::fprintf(stderr, "%s%.8g", i == 0 ? "" : ",", values[i]);
            }
        }
        std::fprintf(stderr, "\n");
    }
    if (debug_hash_index >= 0) {
#ifdef HTP_OPS_OP_TRACE
        const auto * trace_params = reinterpret_cast<const MatMulParams *>(compute_request->payload);
        debug_vtcm_base = trace_params->debug_vtcm_base;
        debug_vtcm_size = trace_params->debug_vtcm_size;
        debug_vtcm_ctx = trace_params->debug_vtcm_ctx;
        debug_hmx_ctx = trace_params->debug_hmx_ctx;
        debug_worker_pool = trace_params->debug_worker_pool;
        debug_hmx_lock_result = trace_params->debug_hmx_lock_result;
        debug_dsp_activation_hash = trace_params->debug_dsp_activation_hash;
        debug_dsp_output_hash = trace_params->debug_dsp_output_hash;
        debug_dsp_output_hash_after_invalidate = trace_params->debug_dsp_output_hash_after_invalidate;
#endif
        const uint64_t weight_hash = htp_hash_tensor_bytes(debug_weight);
        const uint64_t packed_weight_hash = htp_hash_tensor_bytes(debug_packed_weight);
        const uint64_t output_hash = htp_hash_tensor_bytes(dst);
        std::fprintf(stderr,
                     "[ggml-htp-hash] idx=%d ret=%d weight=%s packed=%s self_packed=%d mkn=[%d,%d,%d] "
                     "domain=%d weight_hash=%016" PRIx64 " packed_hash=%016" PRIx64 " "
                     "fds=[%d,%d,%d] offsets=[%zd,%zd,%zd] "
#ifdef HTP_OPS_OP_TRACE
                     "vtcm=[%016" PRIx64 ",%u,ctx=%d] hmx_ctx=%d worker=%016" PRIx64 " lock=%d "
                     "dsp_act_hash=%016" PRIx64 " dsp_out_hash=%016" PRIx64
                     " dsp_out_hash_after_invalidate=%016" PRIx64 " "
#endif
                     "act_bytes=%zu act_hash=%016" PRIx64 " out_bytes=%zu out_hash=%016" PRIx64 "\n",
                     debug_hash_index,
                     ret,
                     debug_weight != nullptr && debug_weight->name[0] != '\0' ? debug_weight->name : "<unnamed>",
                     debug_packed_weight != nullptr && debug_packed_weight->name[0] != '\0' ? debug_packed_weight->name : "<unnamed>",
                     debug_weight == debug_packed_weight ? 1 : 0,
                     dispatch_m,
                     dispatch_k,
                     dispatch_n,
                     ctx->dispatch_mapper().domain(),
                     weight_hash,
                     packed_weight_hash,
                     debug_output_fd,
                     debug_weight_fd,
                     debug_activation_fd,
                     debug_output_offset,
                     debug_weight_offset,
                     debug_activation_offset,
#ifdef HTP_OPS_OP_TRACE
                     debug_vtcm_base,
                     debug_vtcm_size,
                     debug_vtcm_ctx,
                     debug_hmx_ctx,
                     debug_worker_pool,
                     debug_hmx_lock_result,
                     debug_dsp_activation_hash,
                     debug_dsp_output_hash,
                     debug_dsp_output_hash_after_invalidate,
#endif
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
