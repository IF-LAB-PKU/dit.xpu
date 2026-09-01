// HTP 后端主入口。向 GGML 后端注册系统注册 HTP 后端：
// 1) 初始化 DSP 会话并加载 libhtp_ops.so；
// 2) 实现基于 RPCMEM（ION 共享内存）的缓冲区分配/释放；
// 3) 实现后端设备接口：设备属性、算力查询、图计算调度（逐节点 HTP 分发或混合模式）。
// 支持的缓冲区类型为 "RPCMEM"，默认按 256 MiB 分块，128 字节对齐；必要时可用环境变量调整切分粒度。

#include "ggml-htp.h"

#include <dlfcn.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "dsprpc_interface.h"
#include "ggml-cpu/ggml-cpu-impl.h"
#include "ggml-cpu.h"
#include "ggml-backend-impl.h"
#include "ggml-htp-impl.h"
#include "htp-ops.h"
#include "rpcmem_session_route.h"
#include "session_placement.h"

#ifdef GGML_HTP_OPENCL_SIDECAR
#include "ggml-opencl-htp.h"
#endif

extern "C" {
struct ggml_threadpool;
enum ggml_status ggml_graph_compute_htp_hybrid(struct ggml_cgraph * cgraph, struct ggml_cplan * cplan);
struct ggml_cplan ggml_graph_plan(const struct ggml_cgraph * cgraph, int n_threads, struct ggml_threadpool * threadpool);
}

struct ggml_backend_htp_buffer_context {
    void * data;
    int fd;
    size_t logical_size;
    size_t allocation_size;
    bool is_rpcmem;
    bool persistent_staging = false;
    int logical_session_index;
    int domain;
    ggml_backend_htp_context * owner;
};

struct ggml_backend_htp_async_cache_slot {
    ggml_context * metadata_ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_tensor * tensor = nullptr;
    ggml_backend_htp_context * owner = nullptr;
    std::string cache_key;
    uint64_t generation = 0;
    uint64_t opencl_import_identity = 0;
};

static ggml_backend_buffer_t ggml_backend_htp_get_backing_buffer(const ggml_tensor * tensor);

using htp_ops_session_open_fn = int(htp_ops_session **, int, const char *, int);
using htp_ops_session_init_fn = int(htp_ops_session *);
using htp_ops_session_acquire_fn = int(htp_ops_session *);
using htp_ops_session_release_fn = int(htp_ops_session *);
using htp_ops_session_create_channel_fn = int(htp_ops_session *, int, unsigned int);
using htp_ops_session_destroy_channel_fn = int(htp_ops_session *);
using htp_ops_session_domain_fn = unsigned int(const htp_ops_session *);
using htp_ops_session_close_fn = void(htp_ops_session *);

static bool ggml_backend_htp_env_enabled(const char * name) {
    const char * value = std::getenv(name);
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

static int ggml_backend_htp_multi_session_count() {
    const char * value = std::getenv("SD_HTP_MULTI_SESSION_COUNT");
    if (value == nullptr || value[0] == '\0') {
        return 2;
    }
    char * end = nullptr;
    const long requested = std::strtol(value, &end, 10);
    if (end == value || end == nullptr || *end != '\0') {
        return 2;
    }
    return static_cast<int>(std::max(1L, std::min(16L, requested)));
}

static void * ggml_backend_htp_aligned_alloc(size_t size) {
    constexpr size_t alignment = 128;
#ifdef _WIN32
    return _aligned_malloc(size, alignment);
#else
    void * data = nullptr;
    if (posix_memalign(&data, alignment, size) != 0) {
        return nullptr;
    }
    return data;
#endif
}

static void ggml_backend_htp_aligned_free(void * data) {
#ifdef _WIN32
    _aligned_free(data);
#else
    std::free(data);
#endif
}

// 把 hybrid 模式做成进程内显式开关，便于 sd.cpp 在单次推理里临时打开 HTP 混合执行。
static std::atomic<bool> g_htp_hybrid_enabled{false};

void ggml_backend_htp_set_hybrid_enabled(bool enabled) {
    g_htp_hybrid_enabled.store(enabled, std::memory_order_relaxed);
}

bool ggml_backend_htp_get_hybrid_enabled(void) {
    return g_htp_hybrid_enabled.load(std::memory_order_relaxed);
}

static bool ggml_backend_htp_is_view_only(const ggml_tensor * tensor) {
    return tensor != nullptr &&
        (tensor->op == GGML_OP_VIEW || tensor->op == GGML_OP_RESHAPE ||
         tensor->op == GGML_OP_PERMUTE || tensor->op == GGML_OP_TRANSPOSE);
}

static ggml_tensor * ggml_backend_htp_allocating_root(ggml_tensor * tensor) {
    for (size_t depth = 0; tensor != nullptr && depth < 64; ++depth) {
        if (tensor->view_src != nullptr) {
            tensor = tensor->view_src;
            continue;
        }
        if (ggml_backend_htp_is_view_only(tensor) && tensor->src[0] != nullptr) {
            tensor = tensor->src[0];
            continue;
        }
        return tensor;
    }
    return nullptr;
}

size_t ggml_backend_htp_opencl_retain_deferred_residual_inputs(
        struct ggml_cgraph * graph) {
    if (graph == nullptr) {
        return 0;
    }
    static constexpr char rho_prefix[] =
        "htp_block_sparse_residual_rho_b";
    std::vector<ggml_tensor *> retained_roots;
    auto retain_root = [&retained_roots](ggml_tensor * tensor) {
        ggml_tensor * root = ggml_backend_htp_allocating_root(tensor);
        if (root == nullptr ||
            std::find(retained_roots.begin(), retained_roots.end(), root) !=
                retained_roots.end()) {
            return;
        }
        ggml_set_output(root);
        retained_roots.push_back(root);
    };
    for (int i = 0; i < graph->n_nodes; ++i) {
        ggml_tensor * node = graph->nodes[i];
        if (node == nullptr || node->op != GGML_OP_CUSTOM ||
            std::strncmp(node->name, rho_prefix, sizeof(rho_prefix) - 1) != 0 ||
            node->src[0] == nullptr || node->src[1] == nullptr) {
            continue;
        }
        retain_root(node->src[0]);
        ggml_tensor * sparse = node->src[1];
        if (sparse->op != GGML_OP_FLASH_ATTN_EXT) {
            continue;
        }
        for (int source = 0; source < 4; ++source) {
            retain_root(sparse->src[source]);
        }
    }
    return retained_roots.size();
}

static bool ggml_backend_htp_use_hybrid_graph_compute() {
    return ggml_backend_htp_get_hybrid_enabled();
}

static const char * ggml_backend_htp_route_mode_name(enum ggml_htp_route_mode mode) {
    switch (mode) {
        case GGML_HTP_ROUTE_FUSED_OPENCL_RESIDUAL: return "fused";
        case GGML_HTP_ROUTE_OPENCL_ANCHOR:         return "opencl";
        case GGML_HTP_ROUTE_HTP_CPU:               return "htp-cpu";
        case GGML_HTP_ROUTE_DISABLED:              return "disabled";
        case GGML_HTP_ROUTE_INVALID:               return "invalid";
    }
    return "invalid";
}

static enum ggml_htp_route_mode ggml_backend_htp_requested_route_mode() {
    if (!ggml_backend_htp_env_enabled("SD_HTP_BLOCK_SPARSE_RESIDUAL_CACHE")) {
        return GGML_HTP_ROUTE_DISABLED;
    }
    const bool fused =
        ggml_backend_htp_env_enabled("SD_HTP_BLOCK_SPARSE_FUSED_ANCHOR");
    const bool opencl_sparse =
        ggml_backend_htp_env_enabled("SD_HTP_SPARSE_OPENCL");
    const bool opencl_residual =
        ggml_backend_htp_env_enabled("SD_HTP_SPARSE_OPENCL_RESIDUAL");
    if (fused && !opencl_sparse && opencl_residual) {
        return GGML_HTP_ROUTE_FUSED_OPENCL_RESIDUAL;
    }
    if (!fused && opencl_sparse && opencl_residual) {
        return GGML_HTP_ROUTE_OPENCL_ANCHOR;
    }
    if (!fused && !opencl_sparse && !opencl_residual) {
        return GGML_HTP_ROUTE_HTP_CPU;
    }
    return GGML_HTP_ROUTE_INVALID;
}

bool ggml_backend_htp_route_summary_validate(
        const struct ggml_htp_route_summary * summary,
        char * reason,
        size_t reason_capacity) {
    auto fail = [&](const char * message) {
        if (reason != nullptr && reason_capacity > 0) {
            std::snprintf(reason, reason_capacity, "%s", message);
        }
        return false;
    };
    if (summary == nullptr) {
        return fail("route summary must not be null");
    }
    if (summary->mode == GGML_HTP_ROUTE_DISABLED) {
        if (reason != nullptr && reason_capacity > 0) {
            std::snprintf(reason, reason_capacity, "disabled");
        }
        return true;
    }
    if (summary->mode == GGML_HTP_ROUTE_INVALID) {
        return fail("requested route flags are invalid");
    }
    if (summary->cpu_attention_fallback != 0) {
        return fail("eligible attention fell back to CPU");
    }
    if (summary->anchor) {
        if (summary->cache_pairs <= 0) {
            // Warmup and dense-tail graphs can contain only dense attention and
            // intentionally have no residual cache pair. They are outside the
            // anchor ownership contract and must not be rejected as anchors.
            if (summary->htp_dense > 0 && summary->htp_sparse == 0 &&
                summary->opencl_sparse == 0 && summary->fused_anchor == 0 &&
                summary->opencl_residual == 0 && summary->cpu_residual == 0) {
                return true;
            }
            return fail("anchor graph has no residual cache pairs");
        }
        switch (summary->mode) {
            case GGML_HTP_ROUTE_FUSED_OPENCL_RESIDUAL:
                if (summary->fused_anchor <= 0 ||
                    summary->opencl_residual != summary->cache_pairs ||
                    summary->htp_dense != 0 || summary->htp_sparse != 0 ||
                    summary->opencl_sparse != 0 || summary->cpu_residual != 0) {
                    return fail("fused anchor ownership does not match the requested route");
                }
                break;
            case GGML_HTP_ROUTE_OPENCL_ANCHOR:
                if (summary->fused_anchor != 0 || summary->htp_dense <= 0 ||
                    summary->opencl_sparse <= 0 ||
                    summary->opencl_residual != summary->cache_pairs ||
                    summary->htp_sparse != 0 || summary->cpu_residual != 0) {
                    return fail("OpenCL anchor ownership does not match the requested route");
                }
                break;
            case GGML_HTP_ROUTE_HTP_CPU:
                if (summary->fused_anchor != 0 || summary->htp_dense <= 0 ||
                    summary->htp_sparse <= 0 ||
                    summary->cpu_residual != summary->cache_pairs ||
                    summary->opencl_sparse != 0 || summary->opencl_residual != 0) {
                    return fail("HTP/CPU anchor ownership does not match the requested route");
                }
                break;
            default:
                return fail("unsupported route mode");
        }
    } else if (summary->htp_sparse <= 0 || summary->opencl_sparse != 0 ||
               summary->fused_anchor != 0 || summary->htp_dense != 0 ||
               summary->opencl_residual != 0 || summary->cpu_residual != 0 ||
               summary->cache_pairs != 0) {
        return fail("sparse apply graph is not exclusively HTP sparse owned");
    }
    if (reason != nullptr && reason_capacity > 0) {
        std::snprintf(reason, reason_capacity, "ok");
    }
    return true;
}

bool ggml_backend_htp_get_last_route_summary(
        ggml_backend_t backend,
        struct ggml_htp_route_summary * summary) {
    if (backend == nullptr || summary == nullptr || !ggml_backend_is_htp(backend)) {
        return false;
    }
    auto * ctx = static_cast<ggml_backend_htp_context *>(backend->context);
    if (ctx == nullptr || !ctx->has_last_route_summary) {
        return false;
    }
    *summary = ctx->last_route_summary;
    return true;
}

static const ggml_tensor * ggml_backend_htp_producer_through_views(
        const ggml_tensor * tensor) {
    for (size_t depth = 0; tensor != nullptr && depth < 64; ++depth) {
        if (!ggml_backend_htp_is_view_only(tensor)) {
            return tensor;
        }
        tensor = tensor->src[0] != nullptr ? tensor->src[0] : tensor->view_src;
    }
    return nullptr;
}

static bool ggml_backend_htp_is_sparse_attention(const ggml_tensor * tensor) {
    return tensor != nullptr && tensor->op == GGML_OP_FLASH_ATTN_EXT &&
           tensor->src[3] != nullptr && tensor->src[3]->type == GGML_TYPE_I32;
}

static bool ggml_backend_htp_is_residual_cache_output(
        const ggml_tensor * tensor) {
    static constexpr char prefix[] = "htp_block_sparse_residual_b";
    return tensor != nullptr &&
           std::strncmp(tensor->name, prefix, sizeof(prefix) - 1) == 0;
}

static bool ggml_backend_htp_is_residual_rho_output(
        const ggml_tensor * tensor) {
    static constexpr char prefix[] = "htp_block_sparse_residual_rho_b";
    return tensor != nullptr &&
           std::strncmp(tensor->name, prefix, sizeof(prefix) - 1) == 0;
}

static void ggml_backend_htp_collect_attention_ancestors(
        const ggml_tensor * tensor,
        std::unordered_set<const ggml_tensor *> * attention,
        std::unordered_set<const ggml_tensor *> * visited) {
    const ggml_tensor * producer =
        ggml_backend_htp_producer_through_views(tensor);
    if (producer == nullptr || !visited->insert(producer).second) {
        return;
    }
    if (producer->op == GGML_OP_FLASH_ATTN_EXT ||
        (producer->op == GGML_OP_CUSTOM &&
         std::strcmp(producer->name, "htp_fused_anchor_attention") == 0)) {
        attention->insert(producer);
        return;
    }
    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        if (producer->src[i] != nullptr) {
            ggml_backend_htp_collect_attention_ancestors(
                producer->src[i], attention, visited);
        }
    }
}

static bool ggml_backend_htp_collect_route_summary_impl(
        ggml_backend_htp_context * ctx,
        const ggml_cgraph * cgraph,
        struct ggml_htp_route_summary * summary) {
    *summary = {};
    summary->mode = ggml_backend_htp_requested_route_mode();
    if (summary->mode == GGML_HTP_ROUTE_DISABLED) {
        return false;
    }

    std::unordered_set<const ggml_tensor *> anchor_attention;
    std::unordered_set<const ggml_tensor *> fused_producers;
    std::vector<const ggml_tensor *> dense_attention;
    std::vector<const ggml_tensor *> sparse_attention;
    bool has_attention = false;
    const int node_count = ggml_graph_n_nodes(const_cast<ggml_cgraph *>(cgraph));
    for (int i = 0; i < node_count; ++i) {
        const ggml_tensor * node =
            ggml_graph_node(const_cast<ggml_cgraph *>(cgraph), i);
        if (node == nullptr) {
            continue;
        }
        if (node->op == GGML_OP_FLASH_ATTN_EXT) {
            has_attention = true;
            if (ggml_backend_htp_is_sparse_attention(node)) {
                sparse_attention.push_back(node);
            } else {
                dense_attention.push_back(node);
            }
        }
        if (node->op == GGML_OP_CUSTOM &&
            std::strcmp(node->name, "htp_fused_anchor_attention") == 0) {
            has_attention = true;
        }
        if (!ggml_backend_htp_is_residual_cache_output(node)) {
            continue;
        }
        std::unordered_set<const ggml_tensor *> cache_attention;
        std::unordered_set<const ggml_tensor *> visited;
        ggml_backend_htp_collect_attention_ancestors(
            node, &cache_attention, &visited);
        if (cache_attention.empty()) {
            continue;
        }
        ++summary->cache_pairs;
        anchor_attention.insert(cache_attention.begin(), cache_attention.end());
#ifdef GGML_HTP_OPENCL_SIDECAR
        const bool residual_opencl =
            ctx != nullptr && node->op == GGML_OP_CUSTOM &&
            ctx->sparse_opencl_routed_nodes.find(node) !=
                ctx->sparse_opencl_routed_nodes.end() &&
            node->src[2] != nullptr &&
            ctx->sparse_opencl_routed_nodes.find(node->src[2]) !=
                ctx->sparse_opencl_routed_nodes.end();
#else
        const bool residual_opencl = false;
#endif
        if (residual_opencl) {
            ++summary->opencl_residual;
        } else {
            ++summary->cpu_residual;
        }
    }

    summary->anchor = summary->cache_pairs > 0 || sparse_attention.empty();
    if (!has_attention && summary->cache_pairs == 0) {
        return false;
    }
    const auto count_attention = [&](const ggml_tensor * attention) {
        const ggml_tensor * producer =
            ggml_backend_htp_producer_through_views(attention);
        if (producer != nullptr && producer->op == GGML_OP_CUSTOM &&
            std::strcmp(producer->name, "htp_fused_anchor_attention") == 0) {
            if (fused_producers.insert(producer).second) {
                if (htp_ops_support_op_runtime(producer)) {
                    ++summary->fused_anchor;
                } else {
                    ++summary->cpu_attention_fallback;
                }
            }
            return;
        }
#ifdef GGML_HTP_OPENCL_SIDECAR
        const bool opencl = ctx != nullptr &&
                            ctx->sparse_opencl_routed_nodes.find(attention) !=
                            ctx->sparse_opencl_routed_nodes.end();
#else
        const bool opencl = false;
#endif
        if (opencl) {
            if (ggml_backend_htp_is_sparse_attention(attention)) {
                ++summary->opencl_sparse;
            }
        } else if (htp_ops_support_op_runtime(attention)) {
            if (ggml_backend_htp_is_sparse_attention(attention)) {
                ++summary->htp_sparse;
            } else {
                ++summary->htp_dense;
            }
        } else {
            ++summary->cpu_attention_fallback;
        }
    };
    if (summary->anchor) {
        if (summary->cache_pairs > 0) {
            for (const ggml_tensor * attention : anchor_attention) {
                count_attention(attention);
            }
        } else {
            for (const ggml_tensor * attention : dense_attention) {
                count_attention(attention);
            }
        }
    } else {
        for (const ggml_tensor * attention : sparse_attention) {
            count_attention(attention);
        }
    }
    return true;
}

bool ggml_backend_htp_collect_route_summary(
        ggml_backend_t backend,
        const struct ggml_cgraph * cgraph,
        struct ggml_htp_route_summary * summary) {
    if (cgraph == nullptr || summary == nullptr) {
        return false;
    }
    ggml_backend_htp_context * ctx = nullptr;
    if (backend != nullptr && ggml_backend_is_htp(backend)) {
        ctx = static_cast<ggml_backend_htp_context *>(backend->context);
    }
    return ggml_backend_htp_collect_route_summary_impl(ctx, cgraph, summary);
}

static void ggml_backend_htp_print_route_summary(
        const struct ggml_htp_route_summary & summary) {
    std::fprintf(
        stderr,
        "htp_route_summary mode=%s anchor=%d fused=%d htp_dense=%d "
        "htp_sparse=%d opencl_sparse=%d opencl_residual=%d cpu_residual=%d "
        "cache_pairs=%d cpu_attention_fallback=%d status=%s\n",
        ggml_backend_htp_route_mode_name(summary.mode), summary.anchor,
        summary.fused_anchor, summary.htp_dense, summary.htp_sparse,
        summary.opencl_sparse, summary.opencl_residual, summary.cpu_residual,
        summary.cache_pairs, summary.cpu_attention_fallback,
        summary.status_ok ? "ok" : "error");
}

// Qualcomm documents a 2 GiB DSP VA limit on older Snapdragon devices and a
// 3 GiB limit on newer devices such as Snapdragon 8 Gen 3. Keep the default
// at the newer-device limit; deployments with an older target can override it
// explicitly with GGML_HTP_MAX_ACTIVE_MAP_MB.
static size_t ggml_backend_htp_get_max_active_map_size() {
    const char * env = std::getenv("GGML_HTP_MAX_ACTIVE_MAP_MB");
    if (env != nullptr && env[0] != '\0') {
        char * end = nullptr;
        unsigned long long mb = std::strtoull(env, &end, 10);
        if (end != env && end != nullptr && *end == '\0' && mb > 0) {
            return static_cast<size_t>(mb) * 1024ULL * 1024ULL;
        }
    }
    return 3072ULL * 1024ULL * 1024ULL;
}

// HTP 参数 offload 的 buffer 切分粒度也做成环境变量，便于在真机上验证是“映射窗口不够”还是“切成太多独立 fd”导致的 fastrpc_mmap 失败。
static size_t ggml_backend_htp_get_buffer_max_size() {
    const char * env = std::getenv("GGML_HTP_MAX_BUFFER_MB");
    if (env != nullptr && env[0] != '\0') {
        char * end = nullptr;
        unsigned long long mb = std::strtoull(env, &end, 10);
        if (end != env && end != nullptr && *end == '\0' && mb > 0) {
            return static_cast<size_t>(mb) * 1024ULL * 1024ULL;
        }
    }
    return 256ULL * 1024ULL * 1024ULL;
}

static bool ggml_backend_htp_defer_unmap_enabled() {
    const char * env = std::getenv("SD_HTP_DEFER_UNMAP");
    return env == nullptr || env[0] == '\0' || std::strcmp(env, "0") != 0;
}

static bool ggml_backend_htp_ensure_work_data(ggml_backend_htp_context * ctx, size_t work_size) {
    if (work_size == 0) {
        return true;
    }

    if (ctx->work_data != nullptr && ctx->work_size >= work_size) {
        return true;
    }

    void * new_work_data = ggml_backend_htp_aligned_alloc(work_size);
    if (new_work_data == nullptr) {
        return false;
    }

    ggml_backend_htp_aligned_free(ctx->work_data);
    ctx->work_data = static_cast<uint8_t *>(new_work_data);
    ctx->work_size = work_size;
    return true;
}

ggml_backend_htp_context::ggml_backend_htp_context()
    : mapper(ggml_backend_htp_get_max_active_map_size(), ggml_backend_htp_defer_unmap_enabled()) {
    if (!dsprpc_is_available()) {
        const char * err = dsprpc_last_error();
        runtime_error    = err ? err : "libcdsprpc.so is unavailable";
        return;
    }

    rpcmem_init();

    ops_dl_handle = dlopen(HTP_OPS_DL_PATH, RTLD_LAZY | RTLD_LOCAL);
    if (ops_dl_handle == nullptr) {
        const char * err = dlerror();
        runtime_error    = err ? err : "unable to load libhtp_ops.so";
        return;
    }

    using open_session_fn_type = int(int, int);
    using init_htp_ops_fn_type = void();

    auto open_session = reinterpret_cast<open_session_fn_type *>(dlsym(ops_dl_handle, "open_dsp_session"));
    auto init_htp_ops = reinterpret_cast<init_htp_ops_fn_type *>(dlsym(ops_dl_handle, "init_htp_backend"));
    if (open_session == nullptr || init_htp_ops == nullptr) {
        runtime_error = "libhtp_ops.so is missing open_dsp_session/init_htp_backend";
        return;
    }

    int err = open_session(CDSP_DOMAIN_ID, 1);
    if (err != 0) {
        runtime_error = "failed to open remote HTP session";
        return;
    }

    init_htp_ops();

    if (init_message_channel() != 0) {
        runtime_error = "failed to initialize HTP message channel";
        return;
    }
    active_ops_msg_chan = ops_msg_chan;

    using default_session_fn_type = htp_ops_session *();
    auto get_default_session = reinterpret_cast<default_session_fn_type *>(
        dlsym(ops_dl_handle, "htp_ops_default_session"));
    HtpSessionRuntime default_runtime(0, CDSP_DOMAIN_ID);
    default_runtime.native_session = get_default_session != nullptr ? get_default_session() : nullptr;
    default_runtime.mapper = &mapper;
    default_runtime.channel_initialized = true;
    default_runtime.message_channel = ops_msg_chan;
    default_runtime.message_channel_fd = msg_chan_fd;
    default_runtime.lease_active = true;
    sessions.push_back(default_runtime);

    if (ggml_backend_htp_env_enabled("SD_HTP_MULTI_SESSION")) {
        auto open_reserved = reinterpret_cast<htp_ops_session_open_fn *>(
            dlsym(ops_dl_handle, "htp_ops_session_open"));
        auto get_domain = reinterpret_cast<htp_ops_session_domain_fn *>(
            dlsym(ops_dl_handle, "htp_ops_session_domain"));
        auto init_reserved = reinterpret_cast<htp_ops_session_init_fn *>(
            dlsym(ops_dl_handle, "htp_ops_session_init_backend"));
        auto create_reserved_channel = reinterpret_cast<htp_ops_session_create_channel_fn *>(
            dlsym(ops_dl_handle, "htp_ops_session_create_channel"));
        auto close_reserved = reinterpret_cast<htp_ops_session_close_fn *>(
            dlsym(ops_dl_handle, "htp_ops_session_close"));
        if (open_reserved == nullptr || get_domain == nullptr || init_reserved == nullptr ||
            create_reserved_channel == nullptr || close_reserved == nullptr) {
            runtime_error = "libhtp_ops.so lacks multi-session API";
            return;
        }

        reserved_session_open_fn = reinterpret_cast<void *>(open_reserved);
        reserved_session_init_fn = reinterpret_cast<void *>(init_reserved);
        reserved_session_create_channel_fn = reinterpret_cast<void *>(create_reserved_channel);
        reserved_session_domain_fn = reinterpret_cast<void *>(get_domain);
        reserved_session_close_fn = reinterpret_cast<void *>(close_reserved);
        reserved_session_acquire_fn = reinterpret_cast<void *>(
            dlsym(ops_dl_handle, "htp_ops_session_acquire_backend"));
        reserved_session_release_fn = reinterpret_cast<void *>(
            dlsym(ops_dl_handle, "htp_ops_session_release_backend"));
        if (reserved_session_acquire_fn == nullptr || reserved_session_release_fn == nullptr) {
            runtime_error = "libhtp_ops.so lacks multi-session lease API";
            return;
        }
        const int requested_sessions = ggml_backend_htp_multi_session_count();
        for (int session_index = 1; session_index < requested_sessions; ++session_index) {
            htp_ops_session * reserved = nullptr;
            if (open_reserved(&reserved, CDSP_DOMAIN_ID, "", 1) != 0 || reserved == nullptr) {
                std::fprintf(stderr,
                             "[ggml-htp] reserve session index=%d failed; keeping %zu initialized sessions\n",
                             session_index, sessions.size());
                break;
            }
            const int reserved_domain = static_cast<int>(get_domain(reserved));
            auto mapper_instance = std::make_unique<RpcMemMapper>(
                ggml_backend_htp_get_max_active_map_size(),
                ggml_backend_htp_defer_unmap_enabled(), reserved_domain);
            HtpSessionRuntime runtime(static_cast<uint32_t>(session_index),
                                      static_cast<uint32_t>(reserved_domain));
            runtime.native_session = reserved;
            runtime.reserved = true;
            runtime.mapper = mapper_instance.get();
            session_mappers.push_back(std::move(mapper_instance));

            std::fprintf(stderr, "[ggml-htp] session[%d] backend init begin domain=%d ptr=%p\n",
                         session_index, reserved_domain, static_cast<void *>(reserved));
            const int init_err = init_reserved(reserved);
            if (init_err != 0) {
                std::fprintf(stderr, "[ggml-htp] session[%d] backend init failed rc=%d\n",
                             session_index, init_err);
                close_reserved(reserved);
                session_mappers.pop_back();
                break;
            }
            sessions.push_back(runtime);
            HtpSessionRuntime & initialized = sessions.back();
            if (init_session_message_channel(initialized) != 0) {
                std::fprintf(stderr, "[ggml-htp] session[%d] channel init failed\n", session_index);
                close_reserved(reserved);
                sessions.pop_back();
                session_mappers.pop_back();
                break;
            }
            if (session_index == 1) {
                reserved_session = reserved;
                reserved_domain_id = reserved_domain;
                reserved_mapper = std::unique_ptr<RpcMemMapper>(
                    session_mappers.back().release());
                initialized.mapper = reserved_mapper.get();
                reserved_ops_msg_chan = initialized.message_channel;
                reserved_msg_chan_fd = initialized.message_channel_fd;
                reserved_msg_chan_mapped = true;
                // Keep ownership in the legacy field for session 1. Extra
                // sessions remain owned by session_mappers.
            }
            multi_session_enabled = true;
            std::fprintf(stderr, "[ggml-htp] session[%d] ready domain=%d channel_fd=%d\n",
                         session_index, reserved_domain, initialized.message_channel_fd);
        }
        if (sessions.size() <= 1) {
            if (requested_sessions <= 1) {
                // An explicit count of one is a useful way to exercise the
                // compatibility path while keeping the opt-in environment
                // present. Do not fail initialization in that case.
                multi_session_enabled = false;
            } else {
                runtime_error = "no reserved HTP FastRPC session could be initialized";
                return;
            }
        }
        if (ggml_backend_htp_env_enabled("SD_HTP_MULTI_SESSION_USE_RESERVED")) {
            // The reserved session is opened mapping-only. Promote it to the
            // active compute owner here so direct diagnostics and custom
            // channel dispatches cannot accidentally run while its DSP
            // backend is still in mapping-only state.
            if (set_active_session(1) != 0) {
                runtime_error = "failed to activate reserved HTP session";
                return;
            }
            std::fprintf(stderr,
                         "[ggml-htp] diagnostic routing: all HTP mappings/ops use reserved domain=%d\n",
                         reserved_domain_id);
        }
    }

    ops_backend_initialized = true;

#ifdef GGML_HTP_OPENCL_SIDECAR
    if (ggml_backend_htp_env_enabled("SD_HTP_SPARSE_OPENCL") ||
        ggml_backend_htp_env_enabled("SD_HTP_SPARSE_OPENCL_RESIDUAL")) {
        ggml_opencl_qcom_requirements requirements = {};
        char reason[512] = {};
        sparse_opencl_sidecar =
            ggml_backend_opencl_htp_sidecar_create(&requirements, reason, sizeof(reason));
        if (sparse_opencl_sidecar != nullptr) {
            sparse_opencl_page_size = requirements.page_size;
            sparse_opencl_ext_mem_padding = requirements.ext_mem_padding;
        } else {
            sparse_opencl_error = reason[0] != '\0' ? reason : "unknown OpenCL sidecar error";
            if (ggml_backend_htp_env_enabled("SD_HTP_SPARSE_OPENCL_STRICT")) {
                runtime_error = "OpenCL sidecar initialization failed: " + sparse_opencl_error;
                return;
            }
        }
    }
#endif

    runtime_available       = true;

    if (std::getenv("SKIP_HTP_OPS") != nullptr) {
        skip_htp_ops = true;
    }
}

int ggml_backend_htp_context::set_active_session(int session_index) {
    if (!multi_session_enabled) {
        return session_index == 0 ? 0 : -1;
    }
    HtpSessionRuntime * target = session_runtime(session_index);
    HtpSessionRuntime * current = session_runtime(active_session_index);
    if (target == nullptr || current == nullptr || target->native_session == nullptr) {
        return -1;
    }
    if (session_index == active_session_index) {
        return 0;
    }

    if (ggml_backend_htp_env_enabled("SD_HTP_MULTI_SESSION_UNMAP_ON_SWITCH")) {
        // A diagnostic isolation mode: do not leave the same RPCMEM fd mapped
        // in both FastRPC PDs while ownership moves to the next session.
        // Buffers remain allocated and are remapped lazily by the destination
        // mapper, so this does not change tensor contents or the legacy path.
        active_mapper().release_all_mappings();
        std::fprintf(stderr, "[ggml-htp] released all mappings before session switch from=%d to=%d\n",
                     active_session_index, session_index);
    }

    using release_fn_type = int(htp_ops_session *);
    using acquire_fn_type = int(htp_ops_session *);
    using init_fn_type = void();
    using set_active_fn_type = int(htp_ops_session *);
    using set_default_fn_type = int();

    if (current->reserved && current->lease_active) {
        auto release = reinterpret_cast<release_fn_type *>(reserved_session_release_fn);
        if (release == nullptr || release(current->native_session) != 0) {
            std::fprintf(stderr, "[ggml-htp] session[%d] release failed\n", active_session_index);
            return -1;
        }
        current->lease_active = false;
    } else if (!current->reserved && current->lease_active) {
        auto release = reinterpret_cast<int (*)()>(dlsym(ops_dl_handle, "release_htp_backend"));
        if (release == nullptr || release() != 0) {
            std::fprintf(stderr, "[ggml-htp] default session release failed\n");
            return -1;
        }
        current->lease_active = false;
    }

    if (target->reserved) {
        auto acquire = reinterpret_cast<acquire_fn_type *>(reserved_session_acquire_fn);
        if (acquire == nullptr || acquire(target->native_session) != 0) {
            std::fprintf(stderr, "[ggml-htp] session[%d] acquire failed\n", session_index);
            return -1;
        }
        target->lease_active = true;
    } else {
        auto init_default = reinterpret_cast<init_fn_type *>(dlsym(ops_dl_handle, "init_htp_backend"));
        if (init_default == nullptr) {
            return -1;
        }
        init_default();
        target->lease_active = true;
    }

    auto set_active = reinterpret_cast<set_active_fn_type *>(
        dlsym(ops_dl_handle, "htp_ops_set_active_session"));
    auto set_default = reinterpret_cast<set_default_fn_type *>(
        dlsym(ops_dl_handle, "htp_ops_set_default_session"));
    const int select_err = target->reserved
        ? (set_active == nullptr ? -1 : set_active(target->native_session))
        : (set_default == nullptr ? -1 : set_default());
    if (select_err != 0) {
        std::fprintf(stderr, "[ggml-htp] session[%d] handle selection failed\n", session_index);
        return -1;
    }

    active_ops_msg_chan = target->message_channel;
    active_session_index = session_index;
    std::fprintf(stderr, "[ggml-htp] active session=%d domain=%u\n",
                 session_index, target->domain);
    return 0;
}

namespace {
std::mutex g_ctx_mutex;
std::unique_ptr<ggml_backend_htp_context> g_ctx_ptr;
int g_ctx_ref_count = 0;
} // namespace

ggml_backend_htp_context::~ggml_backend_htp_context() {
    std::lock_guard<std::mutex> compute_lock(compute_mutex);
#ifdef GGML_HTP_OPENCL_SIDECAR
    if (sparse_opencl_graph != nullptr) {
        char reason[512] = {};
        ggml_backend_opencl_htp_graph_finish(sparse_opencl_graph, reason, sizeof(reason));
        ggml_backend_opencl_htp_graph_free(&sparse_opencl_graph);
    }
    for (deferred_opencl_graph & deferred : deferred_opencl_graphs) {
        if (deferred.graph == nullptr) {
            continue;
        }
        char reason[512] = {};
        ggml_backend_opencl_htp_graph_wait_deferred(
            deferred.graph, reason, sizeof(reason));
        ggml_backend_opencl_htp_graph_finish(
            deferred.graph, reason, sizeof(reason));
        ggml_backend_opencl_htp_graph_free(&deferred.graph);
    }
    deferred_opencl_graphs.clear();
    active_compute_arena_generation = 0;
    sparse_opencl_submission_thread_active = false;
    sparse_opencl_routed_nodes.clear();
    ggml_backend_opencl_htp_sidecar_free(sparse_opencl_sidecar);
    sparse_opencl_sidecar = nullptr;
#endif

    ggml_backend_htp_aligned_free(work_data);

    if (ops_dl_handle != nullptr) {
        if (ops_backend_initialized) {
            using close_session_fn_type = void();
            using destroy_channel_fn_type = int();

            auto close_session = reinterpret_cast<close_session_fn_type *>(dlsym(ops_dl_handle, "close_dsp_session"));
            auto destroy_channel = reinterpret_cast<destroy_channel_fn_type *>(
                dlsym(ops_dl_handle, "destroy_htp_message_channel"));
            if (ops_msg_chan != nullptr && msg_chan_fd >= 0 && close_session != nullptr) {
                const bool released = htp_teardown_rpcmem_channel(
                    [&]() { return destroy_channel != nullptr ? destroy_channel() : -1; },
                    [&]() { return fastrpc_munmap(
                        CDSP_DOMAIN_ID, msg_chan_fd, ops_msg_chan, MAX_MSG_SIZE); },
                    [&]() {
                        rpcmem_free(ops_msg_chan);
                        ops_msg_chan = nullptr;
                        msg_chan_fd = -1;
                    },
                    [&]() { close_session(); });
                if (!released) {
                    std::fprintf(stderr, "[ggml-htp] default message channel teardown failed\n");
                }
            } else if (close_session != nullptr) {
                close_session();
            }

            ops_backend_initialized = false;
        }

        auto close_reserved = reinterpret_cast<htp_ops_session_close_fn *>(reserved_session_close_fn);
        auto destroy_reserved = reinterpret_cast<htp_ops_session_destroy_channel_fn *>(
            dlsym(ops_dl_handle, "htp_ops_session_destroy_channel"));
        if (close_reserved != nullptr) {
            for (size_t i = sessions.size(); i-- > 1;) {
                HtpSessionRuntime & session = sessions[i];
                if (session.message_channel != nullptr && session.message_channel_fd >= 0 &&
                    session.native_session != nullptr) {
                    const bool released = htp_teardown_rpcmem_channel(
                        [&]() {
                            return destroy_reserved != nullptr
                                ? destroy_reserved(session.native_session) : -1;
                        },
                        [&]() {
                            return fastrpc_munmap(
                                static_cast<int>(session.domain), session.message_channel_fd,
                                session.message_channel, MAX_MSG_SIZE);
                        },
                        [&]() {
                            rpcmem_free(session.message_channel);
                            session.message_channel = nullptr;
                            session.message_channel_fd = -1;
                            session.channel_initialized = false;
                        },
                        [&]() {
                            close_reserved(session.native_session);
                            session.native_session = nullptr;
                        });
                    if (!released) {
                        std::fprintf(stderr,
                                     "[ggml-htp] session[%zu] message channel teardown failed\n", i);
                    }
                } else if (session.native_session != nullptr) {
                    close_reserved(session.native_session);
                    session.native_session = nullptr;
                }
            }
        }
        reserved_session = nullptr;
        reserved_domain_id = -1;
        reserved_ops_msg_chan = nullptr;
        reserved_msg_chan_fd = -1;
        reserved_msg_chan_mapped = false;
        multi_session_enabled = false;

        dlclose(ops_dl_handle);
    }

    if (dsprpc_is_available()) {
        rpcmem_deinit();
    }
}

int ggml_backend_htp_context::init_message_channel() {
    using create_msg_channel_fn_type = int(int, unsigned int);

    auto create_msg_channel =
        reinterpret_cast<create_msg_channel_fn_type *>(dlsym(ops_dl_handle, "create_htp_message_channel"));
    if (create_msg_channel == nullptr) {
        return -1;
    }

    ops_msg_chan = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_FLAG_UNCACHED, MAX_MSG_SIZE);
    if (ops_msg_chan == nullptr) {
        return -1;
    }

    msg_chan_fd = rpcmem_to_fd(ops_msg_chan);
    if (msg_chan_fd < 0) {
        return -1;
    }

    int err = fastrpc_mmap(CDSP_DOMAIN_ID, msg_chan_fd, ops_msg_chan, 0, MAX_MSG_SIZE, FASTRPC_MAP_FD);
    if (err != 0) {
        return -1;
    }

    return create_msg_channel(msg_chan_fd, MAX_MSG_SIZE);
}

int ggml_backend_htp_context::init_session_message_channel(HtpSessionRuntime & session) {
    if (session.native_session == nullptr || session.domain == 0) {
        return -1;
    }

    void * channel = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_FLAG_UNCACHED, MAX_MSG_SIZE);
    if (channel == nullptr) {
        return -1;
    }
    const int channel_fd = rpcmem_to_fd(channel);
    if (channel_fd < 0) {
        rpcmem_free(channel);
        return -1;
    }
    int err = fastrpc_mmap(static_cast<int>(session.domain), channel_fd, channel,
                           0, MAX_MSG_SIZE, FASTRPC_MAP_FD);
    if (err != 0) {
        rpcmem_free(channel);
        return -1;
    }

    auto create_channel = reinterpret_cast<htp_ops_session_create_channel_fn *>(
        dlsym(ops_dl_handle, "htp_ops_session_create_channel_with_token"));
    if (create_channel == nullptr) {
        create_channel = reinterpret_cast<htp_ops_session_create_channel_fn *>(
            dlsym(ops_dl_handle, "htp_ops_session_create_channel"));
    }
    if (create_channel == nullptr || create_channel(session.native_session, channel_fd, MAX_MSG_SIZE) != 0) {
        fastrpc_munmap(static_cast<int>(session.domain), channel_fd, channel, MAX_MSG_SIZE);
        rpcmem_free(channel);
        return -1;
    }
    session.message_channel = channel;
    session.message_channel_fd = channel_fd;
    session.channel_initialized = true;
    return 0;
}

int ggml_backend_htp_context::init_reserved_message_channel() {
    HtpSessionRuntime * runtime = session_runtime(1);
    return runtime != nullptr ? init_session_message_channel(*runtime) : -1;
}

ggml_backend_htp_context * ggml_backend_htp_context::instance() {
    std::lock_guard<std::mutex> lock(g_ctx_mutex);
    if (g_ctx_ptr == nullptr) {
        g_ctx_ptr.reset(new ggml_backend_htp_context());
    }
    return g_ctx_ptr.get();
}

bool ggml_backend_htp_set_active_session(ggml_backend_t backend, int session_index) {
    if (backend == nullptr || !ggml_backend_is_htp(backend)) {
        return false;
    }
    auto * ctx = static_cast<ggml_backend_htp_context *>(backend->context);
    std::lock_guard<std::mutex> lock(ctx->compute_mutex);
    return ctx->set_active_session(session_index) == 0;
}

bool ggml_backend_htp_select_session_for_compute(int session_index) {
    auto * ctx = ggml_backend_htp_context::instance();
    return ctx != nullptr && ctx->set_active_session(session_index) == 0;
}

int ggml_backend_htp_session_count_for_compute(void) {
    auto * ctx = ggml_backend_htp_context::instance();
    return ctx != nullptr ? static_cast<int>(ctx->session_count()) : 0;
}

int ggml_backend_htp_active_session(ggml_backend_t backend) {
    if (backend == nullptr || !ggml_backend_is_htp(backend)) {
        return -1;
    }
    auto * ctx = static_cast<ggml_backend_htp_context *>(backend->context);
    return ctx->active_session_index;
}

int ggml_backend_htp_active_domain(ggml_backend_t backend) {
    if (backend == nullptr || !ggml_backend_is_htp(backend)) {
        return -1;
    }
    auto * ctx = static_cast<ggml_backend_htp_context *>(backend->context);
    return ctx->active_mapper().domain();
}

int ggml_backend_htp_session_count(ggml_backend_t backend) {
    if (backend == nullptr || !ggml_backend_is_htp(backend)) {
        return 0;
    }
    auto * ctx = static_cast<ggml_backend_htp_context *>(backend->context);
    return ctx != nullptr ? static_cast<int>(ctx->session_count()) : 0;
}

int ggml_backend_htp_session_domain(ggml_backend_t backend, int session_index) {
    if (backend == nullptr || !ggml_backend_is_htp(backend)) {
        return -1;
    }
    auto * ctx = static_cast<ggml_backend_htp_context *>(backend->context);
    const HtpSessionRuntime * runtime = ctx != nullptr ? ctx->session_runtime(session_index) : nullptr;
    return runtime != nullptr ? static_cast<int>(runtime->domain) : -1;
}

ggml_backend_htp_context * ggml_backend_htp_context::acquire_instance() {
    std::lock_guard<std::mutex> lock(g_ctx_mutex);
    if (g_ctx_ptr == nullptr) {
        g_ctx_ptr.reset(new ggml_backend_htp_context());
    }
    ++g_ctx_ref_count;
    return g_ctx_ptr.get();
}

void ggml_backend_htp_context::release_instance() {
    std::lock_guard<std::mutex> lock(g_ctx_mutex);
    GGML_ASSERT(g_ctx_ref_count > 0);
    if (--g_ctx_ref_count == 0) {
        g_ctx_ptr.reset();
    }
}

static ggml_guid_t ggml_backend_htp_guid(void) {
    static ggml_guid guid = { 0x9b, 0x58, 0x4a, 0x17, 0x11, 0x82, 0x4a, 0x39,
                              0xa0, 0x7e, 0x61, 0x42, 0x93, 0xc4, 0x51, 0x2d };
    return &guid;
}

static void * ggml_backend_htp_buffer_get_base(ggml_backend_buffer_t buffer) {
    auto * ctx = static_cast<ggml_backend_htp_buffer_context *>(buffer->context);
    return ctx->data;
}

static void ggml_backend_htp_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    auto * ctx = static_cast<ggml_backend_htp_buffer_context *>(buffer->context);
    if (ctx->is_rpcmem) {
        bool diagnostic_freed = false;
        auto mappers = htp_unique_rpcmem_session_mappers(ctx->owner->sessions);
        if (std::find(mappers.begin(), mappers.end(), &ctx->owner->mapper) == mappers.end()) {
            mappers.push_back(&ctx->owner->mapper);
        }
        for (RpcMemMapper * mapper : mappers) {
            diagnostic_freed = diagnostic_freed || mapper->was_freed_by_diagnostic(ctx->data);
            if (!mapper->release_buffer(ctx->data)) {
                GGML_ABORT("failed to release RPCMEM buffer mapping domain=%d ptr=%p\n",
                           mapper->domain(), ctx->data);
            }
        }
        if (!diagnostic_freed) {
            rpcmem_free(ctx->data);
        }
    } else {
        ggml_backend_htp_aligned_free(ctx->data);
    }
    delete ctx;
    ggml_backend_htp_context::release_instance();
}

static void ggml_backend_htp_buffer_memset_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor,
                                                  uint8_t value, size_t offset, size_t size) {
    GGML_UNUSED(buffer);
    std::memset(static_cast<char *>(tensor->data) + offset, value, size);
}

static void ggml_backend_htp_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor,
                                               const void * data, size_t offset, size_t size) {
    GGML_UNUSED(buffer);
    std::memcpy(static_cast<char *>(tensor->data) + offset, data, size);
}

static void ggml_backend_htp_buffer_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor,
                                               void * data, size_t offset, size_t size) {
    GGML_UNUSED(buffer);
    std::memcpy(data, static_cast<const char *>(tensor->data) + offset, size);
}

static bool ggml_backend_htp_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * src,
                                               struct ggml_tensor * dst) {
    GGML_UNUSED(buffer);
    if (ggml_nbytes(src) != ggml_nbytes(dst)) {
        return false;
    }

    std::memcpy(dst->data, src->data, ggml_nbytes(src));
    return true;
}

static void ggml_backend_htp_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    auto * ctx = static_cast<ggml_backend_htp_buffer_context *>(buffer->context);
    std::memset(ctx->data, value, buffer->size);
}

static const struct ggml_backend_buffer_i ggml_backend_htp_buffer_i = {
    /* .free_buffer   = */ ggml_backend_htp_buffer_free_buffer,
    /* .get_base      = */ ggml_backend_htp_buffer_get_base,
    /* .init_tensor   = */ nullptr,
    /* .memset_tensor = */ ggml_backend_htp_buffer_memset_tensor,
    /* .set_tensor    = */ ggml_backend_htp_buffer_set_tensor,
    /* .get_tensor    = */ ggml_backend_htp_buffer_get_tensor,
    /* .set_tensor_2d = */ nullptr,
    /* .get_tensor_2d = */ nullptr,
    /* .cpy_tensor    = */ ggml_backend_htp_buffer_cpy_tensor,
    /* .clear         = */ ggml_backend_htp_buffer_clear,
    /* .reset         = */ nullptr,
};

static const char * ggml_backend_htp_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return "RPCMEM";
}

static const char * ggml_backend_htp_qkv_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return "RPCMEM-QKV";
}

static ggml_backend_buffer_t ggml_backend_htp_buffer_type_alloc_buffer_for_session(
        ggml_backend_buffer_type_t buft,
        size_t size,
        int logical_session_index,
        int domain) {
    auto * ctx            = new ggml_backend_htp_buffer_context();
    auto * htp_ctx        = ggml_backend_htp_context::acquire_instance();
    ctx->owner            = htp_ctx;
    ctx->is_rpcmem        = htp_ctx->runtime_available;
    ctx->fd               = -1;
    ctx->logical_size     = size;
    ctx->allocation_size  = size;
    ctx->logical_session_index = logical_session_index;
    ctx->domain = domain;

#ifdef GGML_HTP_OPENCL_SIDECAR
    const size_t ext_mem_padding = htp_ctx->sparse_opencl_sidecar != nullptr
        ? htp_ctx->sparse_opencl_ext_mem_padding
        : 0;
#else
    const size_t ext_mem_padding = 0;
#endif

    if (ctx->is_rpcmem) {
        if (size > std::numeric_limits<size_t>::max() - ext_mem_padding) {
            delete ctx;
            ggml_backend_htp_context::release_instance();
            return nullptr;
        }
        ctx->allocation_size = size + ext_mem_padding;
        if (ctx->allocation_size > static_cast<size_t>(std::numeric_limits<int>::max())) {
            delete ctx;
            ggml_backend_htp_context::release_instance();
            return nullptr;
        }
        ctx->data = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_FLAG_UNCACHED,
                                 static_cast<int>(ctx->allocation_size));
        if (ctx->data != nullptr) {
            ctx->fd = rpcmem_to_fd(ctx->data);
            if (ctx->fd < 0) {
                rpcmem_free(ctx->data);
                ctx->data = nullptr;
            }
        }
    } else {
        ctx->data = ggml_backend_htp_aligned_alloc(size);
    }

    if (ctx->data == nullptr) {
        delete ctx;
        ggml_backend_htp_context::release_instance();
        return nullptr;
    }

    return ggml_backend_buffer_init(buft, ggml_backend_htp_buffer_i, ctx, size);
}

static ggml_backend_buffer_t ggml_backend_htp_buffer_type_alloc_buffer(
        ggml_backend_buffer_type_t buft,
        size_t size) {
    auto * htp_ctx = ggml_backend_htp_context::instance();
    const HtpSessionRuntime * runtime = htp_ctx != nullptr ? htp_ctx->session_runtime(0) : nullptr;
    return ggml_backend_htp_buffer_type_alloc_buffer_for_session(
        buft, size, 0, runtime != nullptr ? static_cast<int>(runtime->domain) : CDSP_DOMAIN_ID);
}

static size_t ggml_backend_htp_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return 128;
}

static size_t ggml_backend_htp_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return ggml_backend_htp_get_buffer_max_size();
}

static bool ggml_backend_htp_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return true;
}

static size_t ggml_backend_htp_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft,
                                                          const struct ggml_tensor * tensor) {
    const size_t alignment = ggml_backend_htp_buffer_type_get_alignment(buft);
    const size_t size      = ggml_nbytes(tensor);
    return GGML_PAD(size, alignment);
}

static ggml_backend_buffer_type_t ggml_backend_htp_buffer_type(void) {
    static struct ggml_backend_buffer_type buft = {
        /* .iface   = */ {
            /* .get_name       = */ ggml_backend_htp_buffer_type_get_name,
            /* .alloc_buffer   = */ ggml_backend_htp_buffer_type_alloc_buffer,
            /* .get_alignment  = */ ggml_backend_htp_buffer_type_get_alignment,
            /* .get_max_size   = */ ggml_backend_htp_buffer_type_get_max_size,
            /* .get_alloc_size = */ ggml_backend_htp_buffer_type_get_alloc_size,
            /* .is_host        = */ ggml_backend_htp_buffer_type_is_host,
        },
        /* .device  = */ nullptr,
        /* .context = */ nullptr,
    };

    buft.device = ggml_backend_reg_dev_get(ggml_backend_htp_reg(), 0);
    return &buft;
}

ggml_backend_buffer_type_t ggml_backend_htp_qkv_buffer_type(void) {
    static struct ggml_backend_buffer_type buft = {
        /* .iface   = */ {
            /* .get_name       = */ ggml_backend_htp_qkv_buffer_type_get_name,
            /* .alloc_buffer   = */ ggml_backend_htp_buffer_type_alloc_buffer,
            /* .get_alignment  = */ ggml_backend_htp_buffer_type_get_alignment,
            /* .get_max_size   = */ ggml_backend_htp_buffer_type_get_max_size,
            /* .get_alloc_size = */ ggml_backend_htp_buffer_type_get_alloc_size,
            /* .is_host        = */ ggml_backend_htp_buffer_type_is_host,
        },
        /* .device  = */ nullptr,
        /* .context = */ nullptr,
    };
    buft.device = ggml_backend_reg_dev_get(ggml_backend_htp_reg(), 0);
    return &buft;
}

bool ggml_backend_buft_is_rpcmem(ggml_backend_buffer_type_t buft) {
    return buft != nullptr && (buft->iface.get_name == ggml_backend_htp_buffer_type_get_name ||
                               buft->iface.get_name == ggml_backend_htp_qkv_buffer_type_get_name);
}

ggml_backend_buffer_t ggml_backend_htp_alloc_buffer_for_session(
        ggml_backend_t backend,
        int session_index,
        size_t size) {
    if (backend == nullptr || !ggml_backend_is_htp(backend) || size == 0) {
        return nullptr;
    }
    auto * ctx = static_cast<ggml_backend_htp_context *>(backend->context);
    const HtpSessionRuntime * runtime =
        ctx != nullptr ? ctx->session_runtime(session_index) : nullptr;
    if (runtime == nullptr || !runtime->initialized || runtime->mapper == nullptr) {
        return nullptr;
    }
    return ggml_backend_htp_buffer_type_alloc_buffer_for_session(
        ggml_backend_htp_buffer_type(), size, session_index,
        static_cast<int>(runtime->domain));
}

bool ggml_backend_htp_get_rpcmem_info(
        ggml_backend_buffer_t buffer,
        struct ggml_backend_htp_rpcmem_info * info) {
    if (info == nullptr) {
        return false;
    }

    *info = { nullptr, -1, 0, 0 };
    if (buffer == nullptr ||
        !ggml_backend_buft_is_rpcmem(ggml_backend_buffer_get_type(buffer)) ||
        buffer->context == nullptr) {
        return false;
    }

    auto * ctx = static_cast<ggml_backend_htp_buffer_context *>(buffer->context);
    if (!ctx->is_rpcmem || ctx->data == nullptr || ctx->fd < 0) {
        return false;
    }

    info->base = ctx->data;
    info->fd = ctx->fd;
    info->logical_size = ctx->logical_size;
    info->allocation_size = ctx->allocation_size;
    return true;
}

void ggml_backend_htp_buffer_set_persistent_staging(
        ggml_backend_buffer_t buffer, bool persistent) {
    if (buffer == nullptr || buffer->context == nullptr) {
        return;
    }
    auto * ctx = static_cast<ggml_backend_htp_buffer_context *>(buffer->context);
    ctx->persistent_staging = persistent;
}

bool ggml_backend_htp_buffer_is_persistent_staging(ggml_backend_buffer_t buffer) {
    if (buffer == nullptr || buffer->context == nullptr) {
        return false;
    }
    return static_cast<ggml_backend_htp_buffer_context *>(buffer->context)->persistent_staging;
}

int ggml_backend_htp_buffer_session_index(ggml_backend_buffer_t buffer) {
    if (buffer == nullptr || buffer->context == nullptr ||
        !ggml_backend_buft_is_rpcmem(ggml_backend_buffer_get_type(buffer))) {
        return -1;
    }
    return static_cast<ggml_backend_htp_buffer_context *>(buffer->context)->logical_session_index;
}

void ggml_backend_htp_buffer_set_session_index(ggml_backend_buffer_t buffer, int session_index) {
    if (buffer != nullptr && buffer->context != nullptr &&
        ggml_backend_buft_is_rpcmem(ggml_backend_buffer_get_type(buffer))) {
        static_cast<ggml_backend_htp_buffer_context *>(buffer->context)->logical_session_index = session_index;
    }
}

int ggml_backend_htp_node_session_owner(const struct ggml_tensor * node) {
    if (node == nullptr) return -1;
    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        const ggml_backend_buffer_t buffer = ggml_backend_htp_get_backing_buffer(node->src[i]);
        if (buffer != nullptr && ggml_backend_buffer_get_usage(buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
            const int owner = ggml_backend_htp_buffer_session_index(buffer);
            if (owner >= 0) return owner;
        }
    }
    return -1;
}

ggml_backend_htp_async_cache_slot * ggml_backend_htp_async_cache_slot_create(
        ggml_backend_t backend,
        const char * cache_key,
        uint64_t generation,
        enum ggml_type type,
        const int64_t ne[GGML_MAX_DIMS],
        char * reason,
        size_t reason_capacity) {
    auto set_slot_reason = [reason, reason_capacity](const char * message) {
        if (reason != nullptr && reason_capacity > 0) {
            std::snprintf(reason, reason_capacity, "%s", message);
            reason[reason_capacity - 1] = '\0';
        }
    };
    if (backend == nullptr || !ggml_backend_is_htp(backend) || cache_key == nullptr ||
        cache_key[0] == '\0' || ne == nullptr || type < 0 || type >= GGML_TYPE_COUNT) {
        set_slot_reason("backend, cache key, type, and shape must be valid");
        return nullptr;
    }
    for (size_t i = 0; i < GGML_MAX_DIMS; ++i) {
        if (ne[i] <= 0) {
            set_slot_reason("cache slot dimensions must be positive");
            return nullptr;
        }
    }
    if (ggml_backend_htp_env_enabled(
            "SD_HTP_SPARSE_OPENCL_TEST_FAIL_CACHE_ALLOC")) {
        set_slot_reason("injected cache allocation failure");
        return nullptr;
    }

    std::unique_ptr<ggml_backend_htp_async_cache_slot> slot(
        new (std::nothrow) ggml_backend_htp_async_cache_slot());
    if (!slot) {
        set_slot_reason("failed to allocate async cache slot state");
        return nullptr;
    }
    const ggml_init_params params = {
        /* .mem_size   = */ 16 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    slot->metadata_ctx = ggml_init(params);
    if (slot->metadata_ctx == nullptr) {
        set_slot_reason("failed to allocate async cache metadata context");
        return nullptr;
    }
    slot->tensor = ggml_new_tensor_4d(
        slot->metadata_ctx, type, ne[0], ne[1], ne[2], ne[3]);
    ggml_set_name(slot->tensor, cache_key);
    slot->buffer = ggml_backend_alloc_ctx_tensors(slot->metadata_ctx, backend);
    if (slot->buffer == nullptr) {
        ggml_free(slot->metadata_ctx);
        slot->metadata_ctx = nullptr;
        set_slot_reason("failed to allocate RPCMEM async cache backing");
        return nullptr;
    }

    slot->owner = ggml_backend_htp_context::instance();
    slot->cache_key = cache_key;
    slot->generation = generation;
#ifdef GGML_HTP_OPENCL_SIDECAR
    if (slot->owner->sparse_opencl_sidecar != nullptr &&
        ggml_backend_htp_opencl_persistent_import_required()) {
        ggml_backend_htp_rpcmem_info info = {};
        GGML_ASSERT(ggml_backend_htp_get_rpcmem_info(slot->buffer, &info));
        const ggml_opencl_htp_import_descriptor descriptor = {
            slot->buffer,
            info.base,
            info.fd,
            info.logical_size,
            info.allocation_size,
        };
        if (ggml_backend_htp_env_enabled(
                "SD_HTP_SPARSE_OPENCL_TEST_FAIL_CACHE_IMPORT")) {
            ggml_backend_buffer_free(slot->buffer);
            ggml_free(slot->metadata_ctx);
            slot->buffer = nullptr;
            slot->metadata_ctx = nullptr;
            set_slot_reason("injected cache import failure");
            return nullptr;
        }
        const ggml_opencl_htp_result import_result =
            ggml_backend_opencl_htp_persistent_import_acquire(
                slot->owner->sparse_opencl_sidecar,
                &descriptor,
                cache_key,
                generation,
                &slot->opencl_import_identity,
                reason,
                reason_capacity);
        if (import_result != GGML_OPENCL_HTP_OK) {
            ggml_backend_buffer_free(slot->buffer);
            ggml_free(slot->metadata_ctx);
            slot->buffer = nullptr;
            slot->metadata_ctx = nullptr;
            return nullptr;
        }
    }
#else
    // Without the sidecar this is still a valid persistent HTP buffer.  The
    // CPU residual route uses it as a cross-step cache and never imports it
    // into OpenCL.
#endif
    set_slot_reason("ok");
    return slot.release();
}

ggml_tensor * ggml_backend_htp_async_cache_slot_tensor(
        ggml_backend_htp_async_cache_slot * slot) {
    return slot != nullptr ? slot->tensor : nullptr;
}

bool ggml_backend_htp_async_cache_slot_rpcmem_info(
        const ggml_backend_htp_async_cache_slot * slot,
        ggml_backend_htp_rpcmem_info * info) {
    return slot != nullptr && ggml_backend_htp_get_rpcmem_info(slot->buffer, info);
}

uint64_t ggml_backend_htp_async_cache_slot_opencl_import_identity(
        const ggml_backend_htp_async_cache_slot * slot) {
    return slot != nullptr ? slot->opencl_import_identity : 0;
}

static int32_t ggml_backend_htp_cache_key_branch(const char * cache_key) {
    if (cache_key == nullptr) {
        return -1;
    }
    const char * marker = cache_key;
    while ((marker = std::strstr(marker, "_b")) != nullptr) {
        marker += 2;
        if (*marker < '0' || *marker > '9') {
            continue;
        }
        char * end = nullptr;
        const long branch = std::strtol(marker, &end, 10);
        if (end != marker && branch >= 0 &&
            branch <= std::numeric_limits<int32_t>::max()) {
            return static_cast<int32_t>(branch);
        }
    }
    return -1;
}

bool ggml_backend_htp_async_cache_slot_snapshot(
        const ggml_backend_htp_async_cache_slot * slot,
        ggml_htp_cache_slot_snapshot * snapshot) {
    if (slot == nullptr || snapshot == nullptr) {
        return false;
    }
    ggml_backend_htp_rpcmem_info info = {};
    if (!ggml_backend_htp_async_cache_slot_rpcmem_info(slot, &info)) {
        return false;
    }
    *snapshot = {};
    std::snprintf(snapshot->cache_key, sizeof(snapshot->cache_key), "%s",
                  slot->cache_key.c_str());
    snapshot->branch = ggml_backend_htp_cache_key_branch(slot->cache_key.c_str());
    snapshot->generation = slot->generation;
    snapshot->fd = info.fd;
    snapshot->logical_size = info.logical_size;
    snapshot->allocation_size = info.allocation_size;
    snapshot->opencl_import_identity = slot->opencl_import_identity;
    return true;
}

bool ggml_backend_htp_opencl_persistent_import_required(void) {
    const enum ggml_htp_route_mode route =
        ggml_backend_htp_requested_route_mode();
    return route == GGML_HTP_ROUTE_FUSED_OPENCL_RESIDUAL ||
           route == GGML_HTP_ROUTE_OPENCL_ANCHOR;
}

void ggml_backend_htp_async_cache_slot_free(ggml_backend_htp_async_cache_slot * slot) {
    if (slot == nullptr) {
        return;
    }
#ifdef GGML_HTP_OPENCL_SIDECAR
    if (slot->opencl_import_identity != 0 && slot->owner != nullptr &&
        slot->owner->sparse_opencl_sidecar != nullptr) {
        char reason[256] = {};
        const ggml_opencl_htp_result release_result =
            ggml_backend_opencl_htp_persistent_import_release(
                slot->owner->sparse_opencl_sidecar,
                slot->cache_key.c_str(),
                slot->generation,
                reason,
                sizeof(reason));
        if (release_result != GGML_OPENCL_HTP_OK) {
            std::fprintf(stderr,
                         "ggml-htp: persistent cache import release failed: %s\n",
                         reason[0] != '\0' ? reason : "unknown error");
        }
    }
#endif
    if (slot->buffer != nullptr) {
        ggml_backend_buffer_free(slot->buffer);
    }
    if (slot->metadata_ctx != nullptr) {
        ggml_free(slot->metadata_ctx);
    }
    delete slot;
}

void ggml_backend_htp_pack_f16_weight(ggml_fp16_t * dst, const ggml_fp16_t * src, int64_t k, int64_t n) {
    GGML_ASSERT(dst != nullptr);
    GGML_ASSERT(src != nullptr);
    GGML_ASSERT(k % 32 == 0);
    GGML_ASSERT(n % 32 == 0);

    // 离线 HTP packed GGUF 复用这个布局转换：把 ggml Linear 的 row-major [n, k]
    // 权重重排成 DSP HMX matmul 直接读取的 32x32 tile 顺序。
    // src is ggml Linear weight laid out as row-major [n, k].
    // dst is HMX 32x32 tile order for logical W[k, n].
    for (int64_t i = 0; i < k; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            const int64_t i0 = i / 32;
            const int64_t i1 = i % 32;
            const int64_t j0 = j / 32;
            const int64_t j1 = j % 32;

            const int64_t tile_idx = j0 * (k / 32) + i0;
            ggml_fp16_t * tile = dst + tile_idx * 32 * 32;
            tile[(i1 & ~int64_t{1}) * 32 + j1 * 2 + (i1 & int64_t{1})] = src[j * k + i];
        }
    }
}

void ggml_backend_htp_register_packed_f16_weight(const ggml_tensor * original, ggml_tensor * packed) {
    if (original == nullptr || packed == nullptr) {
        return;
    }
    ggml_backend_htp_context::instance()->packed_f16_weights[original] = packed;
}

void ggml_backend_htp_unregister_packed_f16_weight(const ggml_tensor * original) {
    if (original == nullptr) {
        return;
    }
    ggml_backend_htp_context::instance()->packed_f16_weights.erase(original);
}

ggml_tensor * ggml_backend_htp_get_packed_f16_weight(const ggml_tensor * original) {
    if (original == nullptr) {
        return nullptr;
    }
    auto & packed = ggml_backend_htp_context::instance()->packed_f16_weights;
    auto it = packed.find(original);
    if (it != packed.end()) {
        return it->second;
    }

    // Offline-packed column-range views share the root tensor's W16A32
    // storage. Preserve the view's byte offset for dispatch while inheriting
    // the root's registration; downstream shape checks still reject invalid
    // or non-contiguous views.
    const ggml_tensor * root = original;
    while (root->view_src != nullptr) {
        root = root->view_src;
    }
    it = packed.find(root);
    if (root != original && it != packed.end() && it->second == root && ggml_is_contiguous(original)) {
        return const_cast<ggml_tensor *>(original);
    }
    return nullptr;
}

bool ggml_backend_htp_runtime_available(void) {
    return ggml_backend_htp_context::instance()->runtime_available;
}

const char * ggml_backend_htp_runtime_error(void) {
    const auto & error = ggml_backend_htp_context::instance()->runtime_error;
    return error.empty() ? nullptr : error.c_str();
}

static const char * ggml_backend_htp_get_name(ggml_backend_t backend) {
    GGML_UNUSED(backend);
    return "HTP";
}

static void ggml_backend_htp_free(ggml_backend_t backend) {
    delete backend;
    ggml_backend_htp_context::release_instance();
}

// 调度阶段优先取真正持有共享内存的 backing buffer，避免 view 张量因为 buffer 还没回填而被误判成非 RPCMEM。
static ggml_backend_buffer_t ggml_backend_htp_get_backing_buffer(const ggml_tensor * tensor) {
    while (tensor != nullptr) {
        if (tensor->buffer != nullptr) {
            return tensor->buffer;
        }
        tensor = tensor->view_src;
    }
    return nullptr;
}

#ifdef GGML_HTP_OPENCL_SIDECAR
static std::vector<ggml_opencl_htp_import_descriptor>
ggml_backend_htp_collect_opencl_descriptors(const ggml_cgraph * cgraph) {
    std::vector<ggml_opencl_htp_import_descriptor> descriptors;
    auto collect_tensor = [&](const ggml_tensor * tensor) {
        ggml_backend_buffer_t backing = ggml_backend_htp_get_backing_buffer(tensor);
        if (backing == nullptr ||
            std::any_of(
                descriptors.begin(), descriptors.end(),
                [backing](const ggml_opencl_htp_import_descriptor & descriptor) {
                    return descriptor.backing == backing;
                })) {
            return;
        }

        ggml_backend_htp_rpcmem_info info = {};
        if (!ggml_backend_htp_get_rpcmem_info(backing, &info)) {
            return;
        }
        descriptors.push_back({
            /* .backing         = */ backing,
            /* .base            = */ info.base,
            /* .fd              = */ info.fd,
            /* .logical_size    = */ info.logical_size,
            /* .allocation_size = */ info.allocation_size,
        });
    };

    const int node_count = ggml_graph_n_nodes(const_cast<ggml_cgraph *>(cgraph));
    for (int i = 0; i < node_count; ++i) {
        const ggml_tensor * node = ggml_graph_node(const_cast<ggml_cgraph *>(cgraph), i);
        collect_tensor(node);
        if (node == nullptr) {
            continue;
        }
        for (int source = 0; source < GGML_MAX_SRC; ++source) {
            collect_tensor(node->src[source]);
        }
    }
    return descriptors;
}

static ggml_opencl_htp_result ggml_backend_htp_opencl_preflight(
        ggml_backend_htp_context * ctx,
        const ggml_cgraph * cgraph,
        char * reason,
        size_t reason_capacity) {
    GGML_ASSERT(ctx->sparse_opencl_graph == nullptr);
    GGML_ASSERT(!ctx->sparse_opencl_submission_thread_active);
    GGML_ASSERT(ctx->sparse_opencl_routed_nodes.empty());
    if (ctx->sparse_opencl_sidecar == nullptr) {
        std::snprintf(reason, reason_capacity, "OpenCL sidecar is unavailable");
        return GGML_OPENCL_HTP_UNSUPPORTED;
    }
    if (ggml_backend_htp_env_enabled("SD_HTP_SPARSE_OPENCL_TEST_REJECT_PREFLIGHT")) {
        std::snprintf(reason, reason_capacity, "injected OpenCL preflight rejection");
        return GGML_OPENCL_HTP_UNSUPPORTED;
    }

    std::vector<ggml_opencl_htp_import_descriptor> descriptors;
    try {
        descriptors = ggml_backend_htp_collect_opencl_descriptors(cgraph);
    } catch (...) {
        std::snprintf(reason, reason_capacity, "failed to collect HTP RPCMEM descriptors");
        return GGML_OPENCL_HTP_ERROR;
    }
    ggml_opencl_htp_graph * graph = nullptr;
    const ggml_opencl_htp_result result = ggml_backend_opencl_htp_graph_preflight(
        ctx->sparse_opencl_sidecar,
        cgraph,
        descriptors.data(),
        descriptors.size(),
        &graph,
        reason,
        reason_capacity);
    if (result == GGML_OPENCL_HTP_OK) {
        GGML_ASSERT(graph != nullptr);
        std::unordered_set<const ggml_tensor *> routed_nodes;
        try {
            const int node_count = ggml_graph_n_nodes(const_cast<ggml_cgraph *>(cgraph));
            routed_nodes.reserve(ggml_backend_opencl_htp_graph_routed_node_count(graph));
            for (int i = 0; i < node_count; ++i) {
                const ggml_tensor * node =
                    ggml_graph_node(const_cast<ggml_cgraph *>(cgraph), i);
                if (ggml_backend_opencl_htp_graph_routes_node(graph, node)) {
                    routed_nodes.insert(node);
                }
            }
        } catch (...) {
            char cleanup_reason[512] = {};
            ggml_backend_opencl_htp_graph_finish(graph, cleanup_reason, sizeof(cleanup_reason));
            ggml_backend_opencl_htp_graph_free(&graph);
            std::snprintf(
                reason, reason_capacity,
                "failed to freeze routed OpenCL tensor set before worker launch");
            return GGML_OPENCL_HTP_ERROR;
        }
        if (routed_nodes.empty()) {
            char cleanup_reason[512] = {};
            ggml_backend_opencl_htp_graph_finish(graph, cleanup_reason, sizeof(cleanup_reason));
            ggml_backend_opencl_htp_graph_free(&graph);
            std::snprintf(reason, reason_capacity, "OpenCL preflight returned no frozen routed tensors");
            return GGML_OPENCL_HTP_ERROR;
        }

        // No worker exists yet. Publish the complete frozen set and active graph as one session.
        if (ggml_backend_htp_env_enabled("SD_HTP_SPARSE_OPENCL_TEST_FAIL_ENQUEUE")) {
            ggml_backend_opencl_htp_graph_test_fail_next_sparse_enqueue(graph);
        }
        if (ggml_backend_htp_env_enabled("SD_HTP_SPARSE_OPENCL_TEST_FAIL_WAIT")) {
            ggml_backend_opencl_htp_graph_test_fail_next_sparse_wait(graph);
        }
        if (ggml_backend_htp_env_enabled(
                "SD_HTP_SPARSE_OPENCL_TEST_FAIL_RHO_ENQUEUE")) {
            ggml_backend_opencl_htp_graph_test_fail_next_rho_enqueue(graph);
        }
        if (ggml_backend_htp_env_enabled(
                "SD_HTP_SPARSE_OPENCL_TEST_FAIL_RESIDUAL_ENQUEUE")) {
            ggml_backend_opencl_htp_graph_test_fail_next_residual_enqueue(graph);
        }
        if (ggml_backend_htp_env_enabled(
                "SD_HTP_SPARSE_OPENCL_TEST_FAIL_RESIDUAL_WAIT")) {
            ggml_backend_opencl_htp_graph_test_fail_next_residual_wait(graph);
        }
        ctx->sparse_opencl_routed_nodes.swap(routed_nodes);
        ctx->sparse_opencl_graph = graph;
    } else {
        GGML_ASSERT(graph == nullptr);
    }
    return result;
}

static bool ggml_backend_htp_opencl_finish(ggml_backend_htp_context * ctx) {
    if (ctx->sparse_opencl_graph == nullptr) {
        ctx->sparse_opencl_submission_thread_active = false;
        ctx->sparse_opencl_routed_nodes.clear();
        ctx->sparse_opencl_cache_input_bindings.clear();
        ctx->active_compute_arena_generation = 0;
        return true;
    }

    ggml_opencl_htp_graph * graph = ctx->sparse_opencl_graph;
    ctx->sparse_opencl_graph = nullptr;
    ctx->sparse_opencl_submission_thread_active = false;
    char reason[512] = {};
    ggml_opencl_htp_result result = GGML_OPENCL_HTP_OK;
    bool deferred = false;
    if (ctx->active_compute_arena_generation != 0) {
        bool pending = false;
        result = ggml_backend_opencl_htp_graph_defer(
            graph, &pending, reason, sizeof(reason));
        if (result == GGML_OPENCL_HTP_OK && pending) {
            try {
                ctx->deferred_opencl_graphs.push_back({
                    ctx->active_compute_arena_generation, graph,
                });
                deferred = true;
                graph = nullptr;
            } catch (...) {
                result = GGML_OPENCL_HTP_ERROR;
                std::snprintf(reason, sizeof(reason),
                              "failed to retain deferred OpenCL graph");
            }
        }
    }
    ctx->active_compute_arena_generation = 0;
    if (!deferred) {
        const ggml_opencl_htp_result finish =
            ggml_backend_opencl_htp_graph_finish(graph, reason, sizeof(reason));
        if (result == GGML_OPENCL_HTP_OK) {
            result = finish;
        }
    }
    ggml_backend_opencl_htp_graph_free(&graph);
    // Hybrid compute has joined every worker before this function is called.
    ctx->sparse_opencl_routed_nodes.clear();
    ctx->sparse_opencl_cache_input_bindings.clear();
    if (result != GGML_OPENCL_HTP_OK) {
        std::fprintf(
            stderr, "ggml-htp: OpenCL graph finish failed result=%d reason=%s\n",
            (int) result, reason[0] != '\0' ? reason : "unknown error");
        return false;
    }
    return true;
}

static void ggml_backend_htp_collect_lifetime_snapshot_locked(
        ggml_backend_htp_context * ctx,
        ggml_htp_lifetime_snapshot * snapshot) {
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(snapshot != nullptr);
    *snapshot = {};
    snapshot->live_imports =
        ggml_backend_opencl_htp_sidecar_live_persistent_import_count(
            ctx->sparse_opencl_sidecar);
    snapshot->live_events =
        ggml_backend_opencl_htp_sidecar_live_cache_event_count(
            ctx->sparse_opencl_sidecar);
    auto add_graph = [snapshot](const ggml_opencl_htp_graph * graph) {
        ggml_opencl_htp_graph_lifetime_snapshot graph_snapshot = {};
        if (ggml_backend_opencl_htp_graph_get_lifetime_snapshot(
                graph, &graph_snapshot)) {
            snapshot->live_leases += graph_snapshot.live_leases;
            snapshot->pending_jobs += graph_snapshot.pending_jobs;
        }
    };
    add_graph(ctx->sparse_opencl_graph);
    for (const auto & deferred : ctx->deferred_opencl_graphs) {
        add_graph(deferred.graph);
    }
    snapshot->retired_arenas = ctx->deferred_opencl_graphs.size();
}

static void ggml_backend_htp_capture_memory_ledger_locked(
        ggml_backend_htp_context * ctx,
        enum ggml_htp_route_mode requested_route,
        ggml_opencl_htp_result preflight,
        const ggml_htp_route_summary * route_summary) {
    GGML_ASSERT(ctx != nullptr);
    ggml_htp_memory_ledger_snapshot snapshot =
        ctx->has_pending_memory_ledger
            ? ctx->pending_memory_ledger
            : ggml_htp_memory_ledger_snapshot{};
    snapshot.requested_route = requested_route;
    snapshot.preflight_result = static_cast<int32_t>(preflight);
    snapshot.routed_sparse_count = 0;
    snapshot.routed_rho_count = 0;
    snapshot.routed_residual_count = 0;
    for (const ggml_tensor * node : ctx->sparse_opencl_routed_nodes) {
        if (ggml_backend_htp_is_sparse_attention(node)) {
            ++snapshot.routed_sparse_count;
        } else if (ggml_backend_htp_is_residual_rho_output(node)) {
            ++snapshot.routed_rho_count;
        } else if (ggml_backend_htp_is_residual_cache_output(node)) {
            ++snapshot.routed_residual_count;
        }
    }
    snapshot.reported_opencl_residual =
        route_summary != nullptr && route_summary->opencl_residual > 0
            ? static_cast<size_t>(route_summary->opencl_residual)
            : 0;
    ggml_htp_lifetime_snapshot lifetime = {};
    ggml_backend_htp_collect_lifetime_snapshot_locked(ctx, &lifetime);
    snapshot.live_imports = lifetime.live_imports;
    snapshot.live_events = lifetime.live_events;
    snapshot.live_leases = lifetime.live_leases;
    snapshot.pending_jobs = lifetime.pending_jobs;
    snapshot.retired_arenas = std::max(
        snapshot.retired_arenas, lifetime.retired_arenas);
    if (!ggml_backend_htp_memory_ledger_recalculate(&snapshot)) {
        snapshot.cache_slot_count = 0;
        snapshot.rpcmem_cache_logical_bytes = 0;
        snapshot.rpcmem_cache_allocation_bytes = 0;
    }
    ctx->last_memory_ledger = snapshot;
    ctx->has_last_memory_ledger = true;
    ctx->pending_memory_ledger = {};
    ctx->has_pending_memory_ledger = false;

    if (!ggml_backend_htp_memory_ledger_trace_enabled()) {
        return;
    }
    char line[2048] = {};
    if (ggml_backend_htp_format_memory_ledger(
            line, sizeof(line), "pre_compute", &snapshot) > 0) {
        std::fprintf(stderr, "%s\n", line);
    }
    for (size_t i = 0; i < snapshot.cache_slot_count; ++i) {
        char slot_line[512] = {};
        if (ggml_backend_htp_format_memory_ledger_slot(
                slot_line, sizeof(slot_line), &snapshot.cache_slots[i]) > 0) {
            std::fprintf(stderr, "%s\n", slot_line);
        }
    }
}

static bool ggml_backend_htp_test_abort_after_sparse_enqueue(void * data) {
    auto * ctx = static_cast<ggml_backend_htp_context *>(data);
    return ctx != nullptr && ctx->sparse_opencl_graph != nullptr &&
        (ggml_backend_opencl_htp_graph_pending_sparse_event_count(
             ctx->sparse_opencl_graph) > 0 ||
         ggml_backend_opencl_htp_graph_submitted_sparse_count(
             ctx->sparse_opencl_graph) > 0);
}

static bool ggml_backend_htp_test_abort_after_sparse_queue(void * data) {
    auto * ctx = static_cast<ggml_backend_htp_context *>(data);
    return ctx != nullptr && ctx->sparse_opencl_graph != nullptr &&
        ggml_backend_opencl_htp_graph_pending_sparse_submission_count(
            ctx->sparse_opencl_graph) > 0;
}

static bool ggml_backend_htp_test_abort_after_residual_queue(void * data) {
    auto * ctx = static_cast<ggml_backend_htp_context *>(data);
    return ctx != nullptr && ctx->sparse_opencl_graph != nullptr &&
        ggml_backend_opencl_htp_graph_pending_residual_submission_count(
            ctx->sparse_opencl_graph) > 0;
}

static bool ggml_backend_htp_test_abort_after_residual_enqueue(void * data) {
    auto * ctx = static_cast<ggml_backend_htp_context *>(data);
    return ctx != nullptr && ctx->sparse_opencl_graph != nullptr &&
        ggml_backend_opencl_htp_graph_pending_residual_event_count(
            ctx->sparse_opencl_graph) > 0;
}
#endif

static bool ggml_backend_htp_node_has_rpcmem_buffers(const ggml_tensor * node) {
    ggml_backend_buffer_t node_buffer = ggml_backend_htp_get_backing_buffer(node);
    if (node == nullptr || node_buffer == nullptr || !ggml_is_contiguous(node)) {
        return false;
    }

    if (!ggml_backend_buft_is_rpcmem(ggml_backend_buffer_get_type(node_buffer))) {
        return false;
    }

    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        const ggml_tensor * src = node->src[i];
        if (src == nullptr) {
            continue;
        }

        ggml_backend_buffer_t src_buffer = ggml_backend_htp_get_backing_buffer(src);
        if (src_buffer == nullptr || !ggml_is_contiguous(src)) {
            return false;
        }

        if (!ggml_backend_buft_is_rpcmem(ggml_backend_buffer_get_type(src_buffer))) {
            return false;
        }
    }

    return true;
}

static enum ggml_status ggml_backend_htp_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    auto * ctx = static_cast<ggml_backend_htp_context *>(backend->context);
    GGML_ASSERT(ctx != nullptr);
    std::lock_guard<std::mutex> compute_lock(ctx->compute_mutex);

    if (!ctx->runtime_available || !ctx->ops_backend_initialized) {
        std::fprintf(stderr, "ggml-htp: runtime unavailable or ops backend not initialized\n");
        return GGML_STATUS_FAILED;
    }

    if (ggml_graph_n_nodes(cgraph) == 0) {
        return GGML_STATUS_SUCCESS;
    }

    if (ggml_backend_htp_use_hybrid_graph_compute()) {
        struct ggml_cplan cplan = ggml_graph_plan(cgraph, ctx->n_threads > 0 ? ctx->n_threads : 1, nullptr);
        if (!ggml_backend_htp_ensure_work_data(ctx, cplan.work_size)) {
            std::fprintf(stderr, "ggml-htp: failed to allocate hybrid work buffer (%zu bytes)\n", cplan.work_size);
            return GGML_STATUS_FAILED;
        }
        cplan.work_data = ctx->work_data;
        ggml_htp_route_summary route_summary = {};
        bool has_route_summary = false;
        bool route_summary_valid = true;
#ifdef GGML_HTP_OPENCL_SIDECAR
        char preflight_reason[1024] = {};
        const enum ggml_htp_route_mode requested_route =
            ggml_backend_htp_requested_route_mode();
        ggml_opencl_htp_result preflight = GGML_OPENCL_HTP_NO_WORK;
        if (requested_route == GGML_HTP_ROUTE_FUSED_OPENCL_RESIDUAL ||
            requested_route == GGML_HTP_ROUTE_OPENCL_ANCHOR) {
            preflight = ggml_backend_htp_opencl_preflight(
                ctx, cgraph, preflight_reason, sizeof(preflight_reason));
        }
        const char * trace = std::getenv("SD_HTP_SPARSE_OPENCL_TRACE");
        if (trace != nullptr && trace[0] != '\0' && std::strcmp(trace, "0") != 0) {
            std::fprintf(
                stderr,
                "ggml-htp: OpenCL sidecar preflight result=%d routed_nodes=%zu reason=%s\n",
                (int) preflight,
                ctx->sparse_opencl_routed_nodes.size(),
                preflight_reason[0] != '\0' ? preflight_reason : "unknown");
        }
        const bool route_requires_opencl =
            requested_route == GGML_HTP_ROUTE_FUSED_OPENCL_RESIDUAL ||
            requested_route == GGML_HTP_ROUTE_OPENCL_ANCHOR;
        char route_reason[512] = {};
        has_route_summary = ggml_backend_htp_collect_route_summary_impl(
            ctx, cgraph, &route_summary);
        if (has_route_summary) {
            route_summary_valid = ggml_backend_htp_route_summary_validate(
                &route_summary, route_reason, sizeof(route_reason));
            route_summary.status_ok = route_summary_valid ? 1 : 0;
            ctx->last_route_summary = route_summary;
            ctx->has_last_route_summary = true;
        }
        const bool route_requires_opencl_work =
            route_requires_opencl && has_route_summary &&
            route_summary.anchor != 0 && route_summary.cache_pairs > 0;
        ggml_backend_htp_capture_memory_ledger_locked(
            ctx, requested_route, preflight,
            has_route_summary ? &route_summary : nullptr);
        if (!route_summary_valid &&
            ggml_backend_htp_env_enabled("SD_HTP_ROUTE_STRICT")) {
            if (ctx->sparse_opencl_graph != nullptr) {
                ggml_backend_htp_opencl_finish(ctx);
            }
            ggml_backend_htp_print_route_summary(route_summary);
            std::fprintf(stderr, "ggml-htp: route strict validation failed: %s\n",
                         route_reason[0] != '\0' ? route_reason : "unknown");
            return GGML_STATUS_FAILED;
        }
        if (preflight == GGML_OPENCL_HTP_ERROR ||
            (preflight != GGML_OPENCL_HTP_OK &&
             (route_requires_opencl_work ||
              (preflight != GGML_OPENCL_HTP_NO_WORK &&
               (ggml_backend_htp_env_enabled("SD_HTP_SPARSE_OPENCL_STRICT") ||
                (route_requires_opencl &&
                 ggml_backend_htp_env_enabled("SD_HTP_ROUTE_STRICT"))))))) {
            std::fprintf(
                stderr,
                "ggml-htp: OpenCL sidecar preflight rejected in %s mode: %s\n",
                preflight == GGML_OPENCL_HTP_ERROR ? "error" : "strict",
                preflight_reason[0] != '\0' ? preflight_reason : "unknown error");
            return GGML_STATUS_FAILED;
        }
        if (preflight == GGML_OPENCL_HTP_OK &&
            ggml_backend_htp_env_enabled("SD_HTP_SPARSE_OPENCL") &&
            ggml_backend_htp_env_enabled("SD_HTP_SPARSE_OPENCL_ASYNC") &&
            ggml_backend_htp_env_enabled("SD_HTP_SPARSE_OPENCL_EARLY_SUBMIT") &&
            ggml_backend_htp_env_enabled("SD_HTP_SPARSE_OPENCL_SUBMIT_THREAD")) {
            char start_reason[512] = {};
            const ggml_opencl_htp_result start_result =
                ggml_backend_opencl_htp_graph_start_submission_thread(
                    ctx->sparse_opencl_graph, start_reason, sizeof(start_reason));
            if (start_result != GGML_OPENCL_HTP_OK) {
                std::fprintf(
                    stderr,
                    "ggml-htp: OpenCL submission thread start failed result=%d reason=%s; "
                    "falling back to inline submission\n",
                    (int) start_result,
                    start_reason[0] != '\0' ? start_reason : "unknown error");
            } else {
                ctx->sparse_opencl_submission_thread_active = true;
            }
        }
        if (preflight == GGML_OPENCL_HTP_OK &&
            ggml_backend_htp_env_enabled("SD_HTP_SPARSE_OPENCL_TEST_ABORT_AFTER_QUEUE")) {
            cplan.abort_callback = ggml_backend_htp_test_abort_after_sparse_queue;
            cplan.abort_callback_data = ctx;
        }
        if (preflight == GGML_OPENCL_HTP_OK &&
            ggml_backend_htp_env_enabled("SD_HTP_SPARSE_OPENCL_TEST_ABORT_AFTER_ENQUEUE")) {
            cplan.abort_callback = ggml_backend_htp_test_abort_after_sparse_enqueue;
            cplan.abort_callback_data = ctx;
        }
        if (preflight == GGML_OPENCL_HTP_OK &&
            ggml_backend_htp_env_enabled(
                "SD_HTP_SPARSE_OPENCL_TEST_ABORT_AFTER_RESIDUAL_QUEUE")) {
            cplan.abort_callback = ggml_backend_htp_test_abort_after_residual_queue;
            cplan.abort_callback_data = ctx;
        }
        if (preflight == GGML_OPENCL_HTP_OK &&
            ggml_backend_htp_env_enabled(
                "SD_HTP_SPARSE_OPENCL_TEST_ABORT_AFTER_RESIDUAL_ENQUEUE")) {
            cplan.abort_callback = ggml_backend_htp_test_abort_after_residual_enqueue;
            cplan.abort_callback_data = ctx;
        }
#endif
        enum ggml_status status = GGML_STATUS_SUCCESS;
        const int early_submit_err = htp_ops_opencl_early_submit_begin(cgraph);
        if (early_submit_err != 0) {
            std::fprintf(stderr, "ggml-htp: OpenCL early sparse submit setup failed: %d\n", early_submit_err);
            status = GGML_STATUS_FAILED;
        } else {
            status = ggml_graph_compute_htp_hybrid(cgraph, &cplan);
        }
        htp_ops_opencl_early_submit_end();
#ifdef GGML_HTP_OPENCL_SIDECAR
        if (status != GGML_STATUS_SUCCESS && ctx->sparse_opencl_graph != nullptr) {
            char cancel_reason[512] = {};
            const ggml_opencl_htp_result cancel_result =
                ggml_backend_opencl_htp_graph_cancel_submission_thread(
                    ctx->sparse_opencl_graph, cancel_reason, sizeof(cancel_reason));
            if (cancel_result != GGML_OPENCL_HTP_OK) {
                std::fprintf(
                    stderr,
                    "ggml-htp: OpenCL submission cancellation failed result=%d reason=%s\n",
                    (int) cancel_result,
                    cancel_reason[0] != '\0' ? cancel_reason : "unknown error");
            }
        }
        if (!ggml_backend_htp_opencl_finish(ctx) && status == GGML_STATUS_SUCCESS) {
            status = GGML_STATUS_FAILED;
        }
#endif
        if (has_route_summary) {
            route_summary.status_ok =
                route_summary_valid && status == GGML_STATUS_SUCCESS ? 1 : 0;
            ctx->last_route_summary = route_summary;
            ctx->has_last_route_summary = true;
            if (ggml_backend_htp_env_enabled("SD_HTP_ROUTE_TRACE")) {
                ggml_backend_htp_print_route_summary(route_summary);
            }
        }
        return status;
    }

    ggml_compute_params params = {
        /* .ith        = */ 0,
        /* .nth        = */ 1,
        /* .wsize      = */ 0,
        /* .wdata      = */ nullptr,
        /* .threadpool = */ nullptr,
        /* .use_ref    = */ false,
    };

    const bool multi_session_enabled =
        ctx->multi_session_enabled && ctx->session_count() > 1;
    const bool use_dispatch_context = multi_session_enabled &&
        ggml_backend_htp_env_enabled("SD_HTP_PACKED_STAGING");
    if (use_dispatch_context) {
        // The direct HTP path uses the same ordinal policy as the hybrid
        // worker and graph-cut graph views.  Only HTP operations consume an
        // ordinal; there is no second graph-cut owner rule.
        ggml_backend_htp_dispatch_begin(ggml_graph_n_nodes(cgraph));
    }

    for (int i = 0; i < ggml_graph_n_nodes(cgraph); ++i) {
        ggml_tensor * node = ggml_graph_node(cgraph, i);
        if (!htp_ops_support_op(node)) {
            std::fprintf(stderr, "ggml-htp: unsupported node[%d] op=%s type=%s\n",
                         i,
                         node ? ggml_op_name(node->op) : "(null)",
                         node ? ggml_type_name(node->type) : "(null)");
            if (use_dispatch_context) {
                ggml_backend_htp_dispatch_end();
            }
            return GGML_STATUS_FAILED;
        }

        if (!ggml_backend_htp_node_has_rpcmem_buffers(node)) {
            std::fprintf(stderr, "ggml-htp: node[%d] missing rpcmem/contiguous buffers for op=%s\n",
                         i, node ? ggml_op_name(node->op) : "(null)");
            if (use_dispatch_context) {
                ggml_backend_htp_dispatch_end();
            }
            return GGML_STATUS_FAILED;
        }

        if (multi_session_enabled) {
            int owner_session = htp_session_owner_for_ordinal(
                i, ggml_graph_n_nodes(cgraph), static_cast<int>(ctx->session_count()));
            bool selected = false;
            if (use_dispatch_context) {
                const int ordinal_owner = ggml_backend_htp_dispatch_next_owner();
                const int node_owner = ggml_backend_htp_node_session_owner(node);
                owner_session = node_owner >= 0 ? node_owner : ordinal_owner;
                selected = ggml_backend_htp_dispatch_acquire(owner_session);
            } else {
                selected = ctx->set_active_session(owner_session) == 0;
            }
            if (!selected) {
                std::fprintf(stderr,
                             "ggml-htp: failed to select owner session[%d] for node[%d] op=%s\n",
                             owner_session, i, node ? ggml_op_name(node->op) : "(null)");
                if (use_dispatch_context) {
                    ggml_backend_htp_dispatch_end();
                }
                return GGML_STATUS_FAILED;
            }
            if (ggml_backend_htp_env_enabled("SD_HTP_MULTI_SESSION_TRACE")) {
                std::fprintf(stderr,
                             "[ggml-htp] placement node=%d htp_ordinal=%d/%d owner_session=%d domain=%d op=%s\n",
                             i, i, ggml_graph_n_nodes(cgraph), owner_session,
                             ctx->active_mapper().domain(),
                             node ? ggml_op_name(node->op) : "(null)");
            }
        }

        const int err = htp_ops_compute_op(&params, node);
        if (err != 0) {
            std::fprintf(stderr, "ggml-htp: node[%d] compute failed err=0x%x op=%s\n",
                         i, err, node ? ggml_op_name(node->op) : "(null)");
            if (use_dispatch_context) {
                ggml_backend_htp_dispatch_end();
            }
            return GGML_STATUS_FAILED;
        }
    }

    if (use_dispatch_context) {
        ggml_backend_htp_dispatch_end();
    }

    return GGML_STATUS_SUCCESS;
}

static const struct ggml_backend_i ggml_backend_htp_i = {
    /* .get_name             = */ ggml_backend_htp_get_name,
    /* .free                 = */ ggml_backend_htp_free,
    /* .set_tensor_async     = */ nullptr,
    /* .get_tensor_async     = */ nullptr,
    /* .set_tensor_2d_async  = */ nullptr,
    /* .get_tensor_2d_async  = */ nullptr,
    /* .cpy_tensor_async     = */ nullptr,
    /* .synchronize          = */ nullptr,
    /* .graph_plan_create    = */ nullptr,
    /* .graph_plan_free      = */ nullptr,
    /* .graph_plan_update    = */ nullptr,
    /* .graph_plan_compute   = */ nullptr,
    /* .graph_compute        = */ ggml_backend_htp_graph_compute,
    /* .event_record         = */ nullptr,
    /* .event_wait           = */ nullptr,
    /* .graph_optimize       = */ nullptr,
};

ggml_backend_t ggml_backend_htp_init(void) {
    auto * ctx = ggml_backend_htp_context::acquire_instance();
    if (ctx->n_threads <= 0) {
        // Keep hybrid execution single-threaded for now. The HTP op bridge is
        // correctness-first at this stage and still needs dedicated validation
        // before enabling multi-threaded CPU fallback around NPU kernels.
        ctx->n_threads = 1;
    }

    auto * backend = new ggml_backend {
        /* .guid    = */ ggml_backend_htp_guid(),
        /* .iface   = */ ggml_backend_htp_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_htp_reg(), 0),
        /* .context = */ ctx,
    };

    return backend;
}

bool ggml_backend_is_htp(ggml_backend_t backend) {
    return backend != nullptr && ggml_guid_matches(backend->guid, ggml_backend_htp_guid());
}

void ggml_backend_htp_reset_attention_profile(ggml_backend_t backend) {
    if (!ggml_backend_is_htp(backend)) {
        return;
    }
    auto * ctx = static_cast<ggml_backend_htp_context *>(backend->context);
    ctx->last_attention_elapsed_us.store(-1, std::memory_order_release);
}

int64_t ggml_backend_htp_get_last_attention_elapsed_us(ggml_backend_t backend) {
    if (!ggml_backend_is_htp(backend)) {
        return -1;
    }
    const auto * ctx = static_cast<const ggml_backend_htp_context *>(backend->context);
    return ctx->last_attention_elapsed_us.load(std::memory_order_acquire);
}

void ggml_backend_htp_opencl_set_compute_arena_generation(
        ggml_backend_t backend,
        uint64_t generation) {
#ifdef GGML_HTP_OPENCL_SIDECAR
    if (!ggml_backend_is_htp(backend)) {
        return;
    }
    auto * ctx = static_cast<ggml_backend_htp_context *>(backend->context);
    std::lock_guard<std::mutex> lock(ctx->compute_mutex);
    ctx->active_compute_arena_generation = generation;
#else
    GGML_UNUSED(backend);
    GGML_UNUSED(generation);
#endif
}

bool ggml_backend_htp_opencl_arena_generation_pending(
        ggml_backend_t backend,
        uint64_t generation) {
#ifdef GGML_HTP_OPENCL_SIDECAR
    if (!ggml_backend_is_htp(backend) || generation == 0) {
        return false;
    }
    auto * ctx = static_cast<ggml_backend_htp_context *>(backend->context);
    std::lock_guard<std::mutex> lock(ctx->compute_mutex);
    return std::find_if(
        ctx->deferred_opencl_graphs.begin(),
        ctx->deferred_opencl_graphs.end(),
        [generation](const ggml_backend_htp_context::deferred_opencl_graph & item) {
            return item.arena_generation == generation;
        }) != ctx->deferred_opencl_graphs.end();
#else
    GGML_UNUSED(backend);
    GGML_UNUSED(generation);
    return false;
#endif
}

bool ggml_backend_htp_opencl_release_qkv_generation(
        ggml_backend_t backend,
        uint64_t generation,
        bool wait_for_sparse,
        bool * released,
        char * reason,
        size_t reason_capacity) {
#ifdef GGML_HTP_OPENCL_SIDECAR
    if (released != nullptr) {
        *released = false;
    }
    if (!ggml_backend_is_htp(backend) || generation == 0 || released == nullptr) {
        if (reason != nullptr && reason_capacity > 0) {
            std::snprintf(reason, reason_capacity, "invalid backend, generation, or output");
        }
        return false;
    }
    auto * ctx = static_cast<ggml_backend_htp_context *>(backend->context);
    std::lock_guard<std::mutex> lock(ctx->compute_mutex);
    auto found = std::find_if(
        ctx->deferred_opencl_graphs.begin(), ctx->deferred_opencl_graphs.end(),
        [generation](const ggml_backend_htp_context::deferred_opencl_graph & item) {
            return item.arena_generation == generation;
        });
    if (found == ctx->deferred_opencl_graphs.end()) {
        if (released != nullptr) {
            // The deferred graph already completed and released its aliases.
            *released = true;
        }
        if (reason != nullptr && reason_capacity > 0) {
            std::snprintf(reason, reason_capacity, "generation is not deferred");
        }
        return true;
    }
    return ggml_backend_opencl_htp_graph_reap_qkv_aliases(
               found->graph, wait_for_sparse, released,
               reason, reason_capacity) == GGML_OPENCL_HTP_OK;
#else
    GGML_UNUSED(backend);
    GGML_UNUSED(generation);
    if (released != nullptr) {
        *released = false;
    }
    if (reason != nullptr && reason_capacity > 0) {
        std::snprintf(reason, reason_capacity, "OpenCL sidecar is unavailable");
    }
    return false;
#endif
}

bool ggml_backend_htp_opencl_bind_cache_input(
        ggml_backend_t backend,
        const struct ggml_tensor * input,
        const char * cache_key,
        uint64_t generation,
        char * reason,
        size_t reason_capacity) {
#ifdef GGML_HTP_OPENCL_SIDECAR
    if (!ggml_backend_is_htp(backend) || input == nullptr || cache_key == nullptr ||
        cache_key[0] == '\0' || generation == 0) {
        std::snprintf(reason, reason_capacity, "invalid cache input binding");
        return false;
    }
    auto * ctx = static_cast<ggml_backend_htp_context *>(backend->context);
    std::lock_guard<std::mutex> lock(ctx->compute_mutex);
    if (ctx->sparse_opencl_sidecar == nullptr) {
        std::snprintf(reason, reason_capacity, "OpenCL sidecar is unavailable");
        return false;
    }
    ggml_opencl_htp_cache_state state = GGML_OPENCL_HTP_CACHE_EMPTY;
    const ggml_opencl_htp_result result = ggml_backend_opencl_htp_cache_query(
        ctx->sparse_opencl_sidecar, cache_key, generation, &state,
        reason, reason_capacity);
    if (result != GGML_OPENCL_HTP_OK) {
        return false;
    }
    try {
        ctx->sparse_opencl_cache_input_bindings[input] = { cache_key, generation };
    } catch (...) {
        std::snprintf(reason, reason_capacity, "failed to retain cache input binding");
        return false;
    }
    std::snprintf(reason, reason_capacity, "ok");
    return true;
#else
    GGML_UNUSED(backend);
    GGML_UNUSED(input);
    GGML_UNUSED(cache_key);
    GGML_UNUSED(generation);
    std::snprintf(reason, reason_capacity, "OpenCL sidecar support is unavailable");
    return false;
#endif
}

bool ggml_backend_htp_opencl_cache_query(
        ggml_backend_t backend,
        const char * cache_key,
        uint64_t generation,
        bool * ready,
        char * reason,
        size_t reason_capacity) {
    if (ready != nullptr) {
        *ready = false;
    }
#ifdef GGML_HTP_OPENCL_SIDECAR
    if (!ggml_backend_is_htp(backend) || cache_key == nullptr ||
        cache_key[0] == '\0' || generation == 0 || ready == nullptr) {
        std::snprintf(reason, reason_capacity, "invalid cache query");
        return false;
    }
    auto * ctx = static_cast<ggml_backend_htp_context *>(backend->context);
    std::lock_guard<std::mutex> lock(ctx->compute_mutex);
    if (ctx->sparse_opencl_sidecar == nullptr) {
        std::snprintf(reason, reason_capacity, "OpenCL sidecar is unavailable");
        return false;
    }
    ggml_opencl_htp_cache_state state = GGML_OPENCL_HTP_CACHE_EMPTY;
    const ggml_opencl_htp_result result = ggml_backend_opencl_htp_cache_query(
        ctx->sparse_opencl_sidecar, cache_key, generation, &state,
        reason, reason_capacity);
    if (result != GGML_OPENCL_HTP_OK) {
        return false;
    }
    *ready = state == GGML_OPENCL_HTP_CACHE_READY;
    return true;
#else
    GGML_UNUSED(backend);
    GGML_UNUSED(cache_key);
    GGML_UNUSED(generation);
    GGML_UNUSED(ready);
    std::snprintf(reason, reason_capacity, "OpenCL sidecar support is unavailable");
    return false;
#endif
}

bool ggml_backend_htp_opencl_cache_wait(
        ggml_backend_t backend,
        const char * cache_key,
        uint64_t generation,
        bool * waited,
        char * reason,
        size_t reason_capacity) {
    if (waited != nullptr) {
        *waited = false;
    }
#ifdef GGML_HTP_OPENCL_SIDECAR
    if (!ggml_backend_is_htp(backend) || cache_key == nullptr ||
        cache_key[0] == '\0' || generation == 0) {
        std::snprintf(reason, reason_capacity, "invalid cache wait");
        return false;
    }
    auto * ctx = static_cast<ggml_backend_htp_context *>(backend->context);
    std::lock_guard<std::mutex> lock(ctx->compute_mutex);
    if (ctx->sparse_opencl_sidecar == nullptr) {
        std::snprintf(reason, reason_capacity, "OpenCL sidecar is unavailable");
        return false;
    }
    return ggml_backend_opencl_htp_cache_wait(
               ctx->sparse_opencl_sidecar, cache_key, generation, waited,
               reason, reason_capacity) == GGML_OPENCL_HTP_OK;
#else
    GGML_UNUSED(backend);
    GGML_UNUSED(cache_key);
    GGML_UNUSED(generation);
    GGML_UNUSED(waited);
    std::snprintf(reason, reason_capacity, "OpenCL sidecar support is unavailable");
    return false;
#endif
}

void ggml_backend_htp_opencl_test_fail_next_cache_wait(
        ggml_backend_t backend) {
#ifdef GGML_HTP_OPENCL_SIDECAR
    if (!ggml_backend_is_htp(backend)) {
        return;
    }
    auto * ctx = static_cast<ggml_backend_htp_context *>(backend->context);
    std::lock_guard<std::mutex> lock(ctx->compute_mutex);
    ggml_backend_opencl_htp_sidecar_test_fail_next_residual_wait(
        ctx->sparse_opencl_sidecar);
#else
    GGML_UNUSED(backend);
#endif
}

size_t ggml_backend_htp_opencl_cache_input_binding_count(
        ggml_backend_t backend) {
#ifdef GGML_HTP_OPENCL_SIDECAR
    if (!ggml_backend_is_htp(backend)) {
        return 0;
    }
    auto * ctx = static_cast<ggml_backend_htp_context *>(backend->context);
    std::lock_guard<std::mutex> lock(ctx->compute_mutex);
    return ctx->sparse_opencl_cache_input_bindings.size();
#else
    GGML_UNUSED(backend);
    return 0;
#endif
}

bool ggml_backend_htp_set_graph_memory_ledger(
        ggml_backend_t backend,
        const ggml_htp_memory_ledger_snapshot * snapshot) {
    if (!ggml_backend_is_htp(backend) || snapshot == nullptr ||
        snapshot->cache_slot_count > GGML_HTP_MEMORY_LEDGER_MAX_CACHE_SLOTS) {
        return false;
    }
    ggml_htp_memory_ledger_snapshot copy = *snapshot;
    if (!ggml_backend_htp_memory_ledger_recalculate(&copy)) {
        return false;
    }
    auto * ctx = static_cast<ggml_backend_htp_context *>(backend->context);
    std::lock_guard<std::mutex> lock(ctx->compute_mutex);
    ctx->pending_memory_ledger = copy;
    ctx->has_pending_memory_ledger = true;
    return true;
}

bool ggml_backend_htp_get_last_memory_ledger(
        ggml_backend_t backend,
        ggml_htp_memory_ledger_snapshot * snapshot) {
    if (!ggml_backend_is_htp(backend) || snapshot == nullptr) {
        return false;
    }
    auto * ctx = static_cast<ggml_backend_htp_context *>(backend->context);
    std::lock_guard<std::mutex> lock(ctx->compute_mutex);
    if (!ctx->has_last_memory_ledger) {
        return false;
    }
    *snapshot = ctx->last_memory_ledger;
    return true;
}

bool ggml_backend_htp_get_lifetime_snapshot(
        ggml_backend_t backend,
        ggml_htp_lifetime_snapshot * snapshot) {
    if (!ggml_backend_is_htp(backend) || snapshot == nullptr) {
        return false;
    }
    auto * ctx = static_cast<ggml_backend_htp_context *>(backend->context);
    std::lock_guard<std::mutex> lock(ctx->compute_mutex);
#ifdef GGML_HTP_OPENCL_SIDECAR
    ggml_backend_htp_collect_lifetime_snapshot_locked(ctx, snapshot);
#else
    *snapshot = {};
#endif
    return true;
}

size_t ggml_backend_htp_opencl_live_cache_import_count(
        ggml_backend_t backend) {
#ifdef GGML_HTP_OPENCL_SIDECAR
    if (!ggml_backend_is_htp(backend)) {
        return 0;
    }
    auto * ctx = static_cast<ggml_backend_htp_context *>(backend->context);
    std::lock_guard<std::mutex> lock(ctx->compute_mutex);
    return ggml_backend_opencl_htp_sidecar_live_persistent_import_count(
        ctx->sparse_opencl_sidecar);
#else
    GGML_UNUSED(backend);
    return 0;
#endif
}

size_t ggml_backend_htp_opencl_live_cache_event_count(
        ggml_backend_t backend) {
#ifdef GGML_HTP_OPENCL_SIDECAR
    if (!ggml_backend_is_htp(backend)) {
        return 0;
    }
    auto * ctx = static_cast<ggml_backend_htp_context *>(backend->context);
    std::lock_guard<std::mutex> lock(ctx->compute_mutex);
    return ggml_backend_opencl_htp_sidecar_live_cache_event_count(
        ctx->sparse_opencl_sidecar);
#else
    GGML_UNUSED(backend);
    return 0;
#endif
}

bool ggml_backend_htp_opencl_release_arena_generation(
        ggml_backend_t backend,
        uint64_t generation,
        bool wait,
        bool * released,
        char * reason,
        size_t reason_capacity) {
    if (released != nullptr) {
        *released = false;
    }
#ifdef GGML_HTP_OPENCL_SIDECAR
    if (!ggml_backend_is_htp(backend) || generation == 0) {
        std::snprintf(reason, reason_capacity, "invalid HTP backend or arena generation");
        return false;
    }
    auto * ctx = static_cast<ggml_backend_htp_context *>(backend->context);
    std::lock_guard<std::mutex> lock(ctx->compute_mutex);
    const auto found = std::find_if(
        ctx->deferred_opencl_graphs.begin(),
        ctx->deferred_opencl_graphs.end(),
        [generation](const ggml_backend_htp_context::deferred_opencl_graph & item) {
            return item.arena_generation == generation;
        });
    if (found == ctx->deferred_opencl_graphs.end()) {
        if (released != nullptr) {
            *released = true;
        }
        std::snprintf(reason, reason_capacity, "ok");
        return true;
    }

    ggml_opencl_htp_result result = GGML_OPENCL_HTP_OK;
    bool ready = false;
    if (wait) {
        result = ggml_backend_opencl_htp_graph_wait_deferred(
            found->graph, reason, reason_capacity);
        ready = result == GGML_OPENCL_HTP_OK;
    } else {
        result = ggml_backend_opencl_htp_graph_deferred_ready(
            found->graph, &ready, reason, reason_capacity);
    }
    if (result != GGML_OPENCL_HTP_OK) {
        return false;
    }
    if (!ready) {
        std::snprintf(reason, reason_capacity, "deferred arena is still pending");
        return true;
    }
    result = ggml_backend_opencl_htp_graph_reap_deferred(
        found->graph, reason, reason_capacity);
    if (result != GGML_OPENCL_HTP_OK) {
        return false;
    }
    ggml_opencl_htp_graph * graph = found->graph;
    ctx->deferred_opencl_graphs.erase(found);
    ggml_backend_opencl_htp_graph_free(&graph);
    if (released != nullptr) {
        *released = true;
    }
    std::snprintf(reason, reason_capacity, "ok");
    return true;
#else
    GGML_UNUSED(backend);
    GGML_UNUSED(generation);
    GGML_UNUSED(wait);
    if (released != nullptr) {
        *released = true;
    }
    if (reason != nullptr && reason_capacity > 0) {
        std::snprintf(reason, reason_capacity, "ok");
    }
    return true;
#endif
}

size_t ggml_backend_htp_opencl_deferred_arena_count(ggml_backend_t backend) {
#ifdef GGML_HTP_OPENCL_SIDECAR
    if (!ggml_backend_is_htp(backend)) {
        return 0;
    }
    auto * ctx = static_cast<ggml_backend_htp_context *>(backend->context);
    std::lock_guard<std::mutex> lock(ctx->compute_mutex);
    return ctx->deferred_opencl_graphs.size();
#else
    GGML_UNUSED(backend);
    return 0;
#endif
}

void ggml_backend_htp_set_n_threads(ggml_backend_t backend, int n_threads) {
    if (!ggml_backend_is_htp(backend)) {
        return;
    }

    auto * ctx = static_cast<ggml_backend_htp_context *>(backend->context);
    if (ctx != nullptr) {
        ctx->n_threads = n_threads > 0 ? n_threads : 1;
    }
}

static const char * ggml_backend_htp_device_get_name(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return "HTP";
}

static const char * ggml_backend_htp_device_get_description(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    if (ggml_backend_htp_runtime_available()) {
        return "Qualcomm HTP backend";
    }
    return "Qualcomm HTP backend (runtime unavailable)";
}

static void ggml_backend_htp_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    GGML_UNUSED(dev);
    *free  = 0;
    *total = 0;
}

static enum ggml_backend_dev_type ggml_backend_htp_device_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;
}

static void ggml_backend_htp_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name         = ggml_backend_htp_device_get_name(dev);
    props->description  = ggml_backend_htp_device_get_description(dev);
    props->memory_free  = 0;
    props->memory_total = 0;
    props->type         = ggml_backend_htp_device_get_type(dev);
    props->device_id    = nullptr;
    props->caps = {
        /* .async                = */ false,
        /* .host_buffer          = */ false,
        /* .buffer_from_host_ptr = */ false,
        /* .events               = */ false,
    };
}

static ggml_backend_t ggml_backend_htp_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(dev);
    GGML_UNUSED(params);
    return ggml_backend_htp_init();
}

static ggml_backend_buffer_type_t ggml_backend_htp_device_get_buffer_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return ggml_backend_htp_buffer_type();
}

static bool ggml_backend_htp_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    GGML_UNUSED(dev);
    return htp_ops_support_op(op);
}

static bool ggml_backend_htp_device_offload_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    GGML_UNUSED(dev);
    return htp_ops_support_op(op);
}

static bool ggml_backend_htp_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(dev);
    return ggml_backend_buft_is_rpcmem(buft);
}

static const struct ggml_backend_device_i ggml_backend_htp_device_i = {
    /* .get_name             = */ ggml_backend_htp_device_get_name,
    /* .get_description      = */ ggml_backend_htp_device_get_description,
    /* .get_memory           = */ ggml_backend_htp_device_get_memory,
    /* .get_type             = */ ggml_backend_htp_device_get_type,
    /* .get_props            = */ ggml_backend_htp_device_get_props,
    /* .init_backend         = */ ggml_backend_htp_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_htp_device_get_buffer_type,
    /* .get_host_buffer_type = */ nullptr,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op          = */ ggml_backend_htp_device_supports_op,
    /* .supports_buft        = */ ggml_backend_htp_device_supports_buft,
    /* .offload_op           = */ ggml_backend_htp_device_offload_op,
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
};

static const char * ggml_backend_htp_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return "HTP";
}

static size_t ggml_backend_htp_reg_get_device_count(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return 1;
}

static ggml_backend_dev_t ggml_backend_htp_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);

    static ggml_backend_device device = {
        /* .iface   = */ ggml_backend_htp_device_i,
        /* .reg     = */ reg,
        /* .context = */ nullptr,
    };

    return &device;
}

static const struct ggml_backend_reg_i ggml_backend_htp_reg_i = {
    /* .get_name         = */ ggml_backend_htp_reg_get_name,
    /* .get_device_count = */ ggml_backend_htp_reg_get_device_count,
    /* .get_device       = */ ggml_backend_htp_reg_get_device,
    /* .get_proc_address = */ nullptr,
};

ggml_backend_reg_t ggml_backend_htp_reg(void) {
    static ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_htp_reg_i,
        /* .context     = */ nullptr,
    };

    return &reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_htp_reg)
