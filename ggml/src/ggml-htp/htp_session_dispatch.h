#pragma once

#include "session_placement.h"

#include <cstddef>

struct RpcMemMapper;

// Common serial placement context used by full-graph and graph-view HTP
// execution.  It owns only deterministic placement/lease state; mapper and
// DSP channel ownership remain in HtpSessionRuntime.
class HtpSessionDispatchContext {
public:
    explicit HtpSessionDispatchContext(int session_count)
        : session_count_(session_count > 0 ? session_count : 1) {}

    void begin_graph_view(int htp_node_count, std::size_t graph_view_id) {
        htp_node_count_ = htp_node_count > 0 ? htp_node_count : 0;
        graph_view_id_ = graph_view_id;
        htp_ordinal_ = 0;
        active_session_ = -1;
        switch_count_ = 0;
    }

    int next_htp_owner() {
        if (forced_session_ >= 0) {
            ++htp_ordinal_;
            return forced_session_;
        }
        const int owner = htp_session_owner_for_ordinal(
            htp_ordinal_, htp_node_count_, session_count_);
        ++htp_ordinal_;
        return owner;
    }

    bool acquire(int session) {
        if (session < 0 || session >= session_count_) {
            return false;
        }
        if (session != active_session_) {
            active_session_ = session;
            ++switch_count_;
        }
        return true;
    }

    bool valid_session(int session) const {
        return session >= 0 && session < session_count_;
    }

    int active_session() const { return active_session_; }
    int switch_count() const { return switch_count_; }
    void force_owner(int session) { forced_session_ = valid_session(session) ? session : -1; }
    int htp_node_count() const { return htp_node_count_; }
    std::size_t graph_view_id() const { return graph_view_id_; }

private:
    int session_count_ = 1;
    int htp_node_count_ = 0;
    int htp_ordinal_ = 0;
    int active_session_ = -1;
    int switch_count_ = 0;
    int forced_session_ = -1;
    std::size_t graph_view_id_ = 0;
};

extern "C" {
RpcMemMapper * ggml_backend_htp_dispatch_mapper(void);
}
