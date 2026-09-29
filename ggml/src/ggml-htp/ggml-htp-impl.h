// HTP 后端的内部上下文结构 ggml_backend_htp_context 定义。
// 作为单例，持有 RPC 内存映射器、工作缓冲区、libhtp_ops.so 的句柄、
// DSP 消息通道、线程池指针及运行时可用性标志，是整个 HTP 后端的核心状态管理单元。

#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "ggml.h"
#include "ggml-htp.h"
#include "htp_session.h"
#include "htp_session_dispatch.h"
#include "rpcmem_mapper.h"

struct htp_ops_session;

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
    std::unique_ptr<RpcMemMapper> reserved_mapper;
    // Session 0 uses the legacy mapper member for ABI/source compatibility.
    // Every opt-in reserved session gets its own mapper and runtime record.
    std::vector<std::unique_ptr<RpcMemMapper>> session_mappers;
    std::vector<HtpSessionRuntime> sessions;

    // The default session remains the only execution path unless explicitly
    // enabled. The reserved session is kept separate until graph ownership
    // and transfer scheduling are enabled in a later migration phase.
    bool multi_session_enabled = false;
    htp_ops_session * reserved_session = nullptr;
    int reserved_domain_id = -1;
    void * reserved_session_open_fn = nullptr;
    void * reserved_session_init_fn = nullptr;
    void * reserved_session_create_channel_fn = nullptr;
    void * reserved_session_domain_fn = nullptr;
    void * reserved_session_close_fn = nullptr;
    void * reserved_ops_msg_chan = nullptr;
    int reserved_msg_chan_fd = -1;
    bool reserved_msg_chan_mapped = false;
    void * active_ops_msg_chan = nullptr;
    int active_session_index = 0;
    void * reserved_session_acquire_fn = nullptr;
    void * reserved_session_release_fn = nullptr;

    void * ops_dl_handle = nullptr;
    bool ops_backend_initialized = false;
    void * ops_msg_chan = nullptr;
    int msg_chan_fd = -1;

    bool skip_htp_ops = false;
    bool runtime_available = false;
    std::atomic<int64_t> last_attention_elapsed_us{-1};
    // Host instant of the DSP doorbell of the last dispatched node: everything
    // the host spends before the store is the submit, everything after it is the
    // wait for the DSP.  -1 until a node has been dispatched since the last
    // reset_attention_profile.
    std::atomic<int64_t> last_dispatch_doorbell_us{-1};
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
    int init_reserved_message_channel();
    int init_session_message_channel(HtpSessionRuntime & session);
    int set_active_session(int session_index);
    HtpSessionRuntime * session_runtime(int session_index) {
        if (session_index < 0 || static_cast<size_t>(session_index) >= sessions.size()) {
            return nullptr;
        }
        return &sessions[static_cast<size_t>(session_index)];
    }
    const HtpSessionRuntime * session_runtime(int session_index) const {
        if (session_index < 0 || static_cast<size_t>(session_index) >= sessions.size()) {
            return nullptr;
        }
        return &sessions[static_cast<size_t>(session_index)];
    }
    size_t session_count() const { return sessions.size(); }
    RpcMemMapper & active_mapper() {
        HtpSessionRuntime * runtime = session_runtime(active_session_index);
        return runtime != nullptr && runtime->mapper != nullptr ? *runtime->mapper : mapper;
    }
    // During a graph dispatch, resolve the mapper from the dispatch
    // context's selected runtime rather than from mutable global state.  The
    // legacy active mapper remains the fallback for single-session callers.
    RpcMemMapper & dispatch_mapper() {
        RpcMemMapper * selected = ggml_backend_htp_dispatch_mapper();
        return selected != nullptr ? *selected : active_mapper();
    }
    const RpcMemMapper & dispatch_mapper() const {
        RpcMemMapper * selected = ggml_backend_htp_dispatch_mapper();
        return selected != nullptr ? *selected : active_mapper();
    }
    const RpcMemMapper & active_mapper() const {
        const HtpSessionRuntime * runtime = session_runtime(active_session_index);
        return runtime != nullptr && runtime->mapper != nullptr ? *runtime->mapper : mapper;
    }
    void * message_channel() const {
        return active_ops_msg_chan != nullptr ? active_ops_msg_chan : ops_msg_chan;
    }

    static ggml_backend_htp_context * instance();
    static ggml_backend_htp_context * acquire_instance();
    static void release_instance();
};
