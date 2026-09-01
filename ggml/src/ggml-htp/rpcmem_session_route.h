#pragma once

#include "htp_session.h"

#include <cstddef>
#include <unordered_set>
#include <utility>
#include <vector>

inline bool htp_rpcmem_mapping_fits(
        size_t max_size,
        size_t persistent_size,
        size_t active_size,
        size_t pending_size,
        size_t required_size) {
    size_t used = 0;
    for (const size_t size : {persistent_size, active_size, pending_size, required_size}) {
        if (size > max_size - used) {
            return false;
        }
        used += size;
    }
    return true;
}

template <typename DestroyFn, typename UnmapFn, typename FreeFn, typename CloseFn>
inline bool htp_teardown_rpcmem_channel(
        DestroyFn && destroy_dsp_channel,
        UnmapFn && unmap_host_channel,
        FreeFn && free_host_channel,
        CloseFn && close_session) {
    bool success = std::forward<DestroyFn>(destroy_dsp_channel)() == 0;
    if (success) {
        success = std::forward<UnmapFn>(unmap_host_channel)() == 0;
    }
    if (success) {
        std::forward<FreeFn>(free_host_channel)();
    }
    std::forward<CloseFn>(close_session)();
    return success;
}

inline int htp_rpcmem_session_index_for_domain(
        const std::vector<HtpSessionRuntime> & sessions,
        int domain) {
    for (size_t i = 0; i < sessions.size(); ++i) {
        if (static_cast<int>(sessions[i].domain) == domain) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

inline std::vector<RpcMemMapper *> htp_unique_rpcmem_session_mappers(
        const std::vector<HtpSessionRuntime> & sessions) {
    std::vector<RpcMemMapper *> result;
    std::unordered_set<RpcMemMapper *> seen;
    for (const HtpSessionRuntime & session : sessions) {
        if (session.mapper != nullptr && seen.insert(session.mapper).second) {
            result.push_back(session.mapper);
        }
    }
    return result;
}
