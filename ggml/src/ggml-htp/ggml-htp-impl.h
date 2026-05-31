// HTP 后端的内部上下文结构 ggml_backend_htp_context 定义。
// 作为单例，持有 RPC 内存映射器、工作缓冲区、libhtp_ops.so 的句柄、
// DSP 消息通道、线程池指针及运行时可用性标志，是整个 HTP 后端的核心状态管理单元。

#pragma once

#include <string>

#include "ggml.h"
#include "rpcmem_mapper.h"

static const char * HTP_OPS_DL_PATH = "libhtp_ops.so";

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
    std::string runtime_error;

    ggml_backend_htp_context();
    ~ggml_backend_htp_context();

    int init_message_channel();

    static ggml_backend_htp_context * instance();
    static void retain_instance();
    static void release_instance();
};
