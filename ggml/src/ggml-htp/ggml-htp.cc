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

static bool ggml_backend_htp_env_enabled(const char * name) {
    const char * value = std::getenv(name);
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
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

size_t ggml_backend_htp_opencl_retain_deferred_residual_inputs(
        struct ggml_cgraph * graph) {
    if (graph == nullptr) {
        return 0;
    }
    static constexpr char rho_prefix[] =
        "htp_block_sparse_residual_rho_b";
    size_t retained = 0;
    for (int i = 0; i < graph->n_nodes; ++i) {
        ggml_tensor * node = graph->nodes[i];
        if (node == nullptr || node->op != GGML_OP_CUSTOM ||
            std::strncmp(node->name, rho_prefix, sizeof(rho_prefix) - 1) != 0 ||
            node->src[0] == nullptr || node->src[1] == nullptr) {
            continue;
        }
        ggml_set_output(node->src[0]);
        ++retained;
    }
    return retained;
}

static bool ggml_backend_htp_use_hybrid_graph_compute() {
    return ggml_backend_htp_get_hybrid_enabled();
}

// 部分设备的 fastrpc 可用映射窗口明显小于 3 GiB，这里给一个保守默认值，并允许用环境变量按 MiB 覆盖。
static size_t ggml_backend_htp_get_max_active_map_size() {
    const char * env = std::getenv("GGML_HTP_MAX_ACTIVE_MAP_MB");
    if (env != nullptr && env[0] != '\0') {
        char * end = nullptr;
        unsigned long long mb = std::strtoull(env, &end, 10);
        if (end != env && end != nullptr && *end == '\0' && mb > 0) {
            return static_cast<size_t>(mb) * 1024ULL * 1024ULL;
        }
    }
    return 2560ULL * 1024ULL * 1024ULL;
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

ggml_backend_htp_context::ggml_backend_htp_context() : mapper(ggml_backend_htp_get_max_active_map_size(), true) {
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

    ops_backend_initialized = true;

#ifdef GGML_HTP_OPENCL_SIDECAR
    if (ggml_backend_htp_env_enabled("SD_HTP_SPARSE_OPENCL")) {
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

            auto close_session = reinterpret_cast<close_session_fn_type *>(dlsym(ops_dl_handle, "close_dsp_session"));
            if (close_session != nullptr) {
                close_session();
            }

            if (ops_msg_chan != nullptr && msg_chan_fd >= 0) {
                fastrpc_munmap(CDSP_DOMAIN_ID, msg_chan_fd, ops_msg_chan, MAX_MSG_SIZE);
                rpcmem_free(ops_msg_chan);
            }

            ops_backend_initialized = false;
        }

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

ggml_backend_htp_context * ggml_backend_htp_context::instance() {
    std::lock_guard<std::mutex> lock(g_ctx_mutex);
    if (g_ctx_ptr == nullptr) {
        g_ctx_ptr.reset(new ggml_backend_htp_context());
    }
    return g_ctx_ptr.get();
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
        ctx->owner->mapper.release_buffer(ctx->data);
        rpcmem_free(ctx->data);
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

static ggml_backend_buffer_t ggml_backend_htp_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    auto * ctx            = new ggml_backend_htp_buffer_context();
    auto * htp_ctx        = ggml_backend_htp_context::acquire_instance();
    ctx->owner            = htp_ctx;
    ctx->is_rpcmem        = htp_ctx->runtime_available;
    ctx->fd               = -1;
    ctx->logical_size     = size;
    ctx->allocation_size  = size;

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

bool ggml_backend_buft_is_rpcmem(ggml_backend_buffer_type_t buft) {
    return buft != nullptr && buft->iface.get_name == ggml_backend_htp_buffer_type_get_name;
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
    if (slot->owner->sparse_opencl_sidecar == nullptr) {
        ggml_backend_buffer_free(slot->buffer);
        ggml_free(slot->metadata_ctx);
        slot->buffer = nullptr;
        slot->metadata_ctx = nullptr;
        set_slot_reason("OpenCL sidecar is unavailable for async cache import");
        return nullptr;
    }
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
#else
    ggml_backend_buffer_free(slot->buffer);
    ggml_free(slot->metadata_ctx);
    slot->buffer = nullptr;
    slot->metadata_ctx = nullptr;
    set_slot_reason("HTP OpenCL sidecar support is not compiled in");
    return nullptr;
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
#ifdef GGML_HTP_OPENCL_SIDECAR
        char preflight_reason[1024] = {};
        const ggml_opencl_htp_result preflight = ggml_backend_htp_opencl_preflight(
            ctx, cgraph, preflight_reason, sizeof(preflight_reason));
        const char * trace = std::getenv("SD_HTP_SPARSE_OPENCL_TRACE");
        if (trace != nullptr && trace[0] != '\0' && std::strcmp(trace, "0") != 0) {
            std::fprintf(
                stderr,
                "ggml-htp: OpenCL sidecar preflight result=%d routed_nodes=%zu reason=%s\n",
                (int) preflight,
                ctx->sparse_opencl_routed_nodes.size(),
                preflight_reason[0] != '\0' ? preflight_reason : "unknown");
        }
        if (preflight == GGML_OPENCL_HTP_ERROR ||
            (preflight != GGML_OPENCL_HTP_OK &&
             preflight != GGML_OPENCL_HTP_NO_WORK &&
             ggml_backend_htp_env_enabled("SD_HTP_SPARSE_OPENCL_STRICT"))) {
            std::fprintf(
                stderr,
                "ggml-htp: OpenCL sidecar preflight rejected in %s mode: %s\n",
                preflight == GGML_OPENCL_HTP_ERROR ? "error" : "strict",
                preflight_reason[0] != '\0' ? preflight_reason : "unknown error");
            return GGML_STATUS_FAILED;
        }
        if (preflight == GGML_OPENCL_HTP_OK &&
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

    for (int i = 0; i < ggml_graph_n_nodes(cgraph); ++i) {
        ggml_tensor * node = ggml_graph_node(cgraph, i);
        if (!htp_ops_support_op(node)) {
            std::fprintf(stderr, "ggml-htp: unsupported node[%d] op=%s type=%s\n",
                         i,
                         node ? ggml_op_name(node->op) : "(null)",
                         node ? ggml_type_name(node->type) : "(null)");
            return GGML_STATUS_FAILED;
        }

        if (!ggml_backend_htp_node_has_rpcmem_buffers(node)) {
            std::fprintf(stderr, "ggml-htp: node[%d] missing rpcmem/contiguous buffers for op=%s\n",
                         i, node ? ggml_op_name(node->op) : "(null)");
            return GGML_STATUS_FAILED;
        }

        const int err = htp_ops_compute_op(&params, node);
        if (err != 0) {
            std::fprintf(stderr, "ggml-htp: node[%d] compute failed err=0x%x op=%s\n",
                         i, err, node ? ggml_op_name(node->op) : "(null)");
            return GGML_STATUS_FAILED;
        }
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
