// HTP 后端的内部上下文结构 ggml_backend_htp_context 定义。
// 作为单例，持有 RPC 内存映射器、工作缓冲区、libhtp_ops.so 的句柄、
// DSP 消息通道、线程池指针及运行时可用性标志，是整个 HTP 后端的核心状态管理单元。

#pragma once

#include <atomic>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "ggml.h"
#include "ggml-htp.h"
#include "rpcmem_mapper.h"

static const char * HTP_OPS_DL_PATH = "libhtp_ops.so";

#ifdef GGML_HTP_OPENCL_SIDECAR
struct ggml_opencl_htp_sidecar;
struct ggml_opencl_htp_graph;
#endif

struct ggml_backend_htp_context {
    static constexpr size_t MAX_MSG_SIZE = 4096;

    size_t work_size = 0;
    uint8_t * work_data = nullptr;

    int n_threads = 0;
    struct ggml_threadpool * threadpool = nullptr;

    RpcMemMapper mapper;

    void * ops_dl_handle = nullptr;
    bool ops_backend_initialized = false;
    void * ops_msg_chan = nullptr;
    int msg_chan_fd = -1;

    bool skip_htp_ops = false;
    bool runtime_available = false;
    std::atomic<int64_t> last_attention_elapsed_us{-1};
    ggml_htp_route_summary last_route_summary = {};
    bool has_last_route_summary = false;
    ggml_htp_memory_ledger_snapshot pending_memory_ledger = {};
    ggml_htp_memory_ledger_snapshot last_memory_ledger = {};
    bool has_pending_memory_ledger = false;
    bool has_last_memory_ledger = false;
    std::string runtime_error;
    std::unordered_map<const ggml_tensor *, ggml_tensor *> packed_f16_weights;
    // The HTP RPC channel, work buffer, and sidecar graph session are process singletons.
    std::mutex compute_mutex;

#ifdef GGML_HTP_OPENCL_SIDECAR
    struct cache_input_binding {
        std::string cache_key;
        uint64_t generation = 0;
    };
    struct deferred_opencl_graph {
        uint64_t arena_generation = 0;
        ggml_opencl_htp_graph * graph = nullptr;
    };
    ggml_opencl_htp_sidecar * sparse_opencl_sidecar = nullptr;
    ggml_opencl_htp_graph * sparse_opencl_graph = nullptr;
    std::vector<deferred_opencl_graph> deferred_opencl_graphs;
    uint64_t active_compute_arena_generation = 0;
    bool sparse_opencl_submission_thread_active = false;
    std::unordered_set<const ggml_tensor *> sparse_opencl_routed_nodes;
    std::unordered_map<const ggml_tensor *, cache_input_binding>
        sparse_opencl_cache_input_bindings;
    size_t sparse_opencl_page_size = 0;
    size_t sparse_opencl_ext_mem_padding = 0;
    std::string sparse_opencl_error;
#endif

    ggml_backend_htp_context();
    ~ggml_backend_htp_context();

    int init_message_channel();

    static ggml_backend_htp_context * instance();
    static ggml_backend_htp_context * acquire_instance();
    static void release_instance();
};
