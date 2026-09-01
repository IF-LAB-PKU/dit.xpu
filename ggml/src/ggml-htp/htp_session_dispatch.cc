#include "htp_session_dispatch.h"

#include "ggml-htp.h"
#include "ggml-htp-impl.h"

namespace {
thread_local HtpSessionDispatchContext * g_dispatch_context = nullptr;
}

extern "C" {

void ggml_backend_htp_dispatch_begin(int htp_node_count) {
    const int session_count = ggml_backend_htp_session_count_for_compute();
    static thread_local HtpSessionDispatchContext context(1);
    context = HtpSessionDispatchContext(session_count > 0 ? session_count : 1);
    context.begin_graph_view(htp_node_count, 0);
    g_dispatch_context = &context;
}

int ggml_backend_htp_dispatch_next_owner(void) {
    return g_dispatch_context != nullptr ? g_dispatch_context->next_htp_owner() : 0;
}

bool ggml_backend_htp_dispatch_acquire(int session_index) {
    if (g_dispatch_context == nullptr || !g_dispatch_context->valid_session(session_index)) {
        return false;
    }
    if (!ggml_backend_htp_select_session_for_compute(session_index)) {
        return false;
    }
    return g_dispatch_context->acquire(session_index);
}

void ggml_backend_htp_dispatch_force_owner(int session_index) {
    if (g_dispatch_context != nullptr) {
        g_dispatch_context->force_owner(session_index);
    }
}

void ggml_backend_htp_dispatch_end(void) {
    g_dispatch_context = nullptr;
}

RpcMemMapper * ggml_backend_htp_dispatch_mapper(void) {
    if (g_dispatch_context == nullptr || g_dispatch_context->active_session() < 0) {
        return nullptr;
    }
    auto * ctx = ggml_backend_htp_context::instance();
    if (ctx == nullptr) {
        return nullptr;
    }
    auto * runtime = ctx->session_runtime(g_dispatch_context->active_session());
    return runtime != nullptr && runtime->mapper != nullptr
        ? runtime->mapper : &ctx->mapper;
}

} // extern "C"

// The dispatch context is intentionally header-defined for now.  This
// translation unit gives build systems a stable source target while the
// runtime integration is moved behind the common API.
