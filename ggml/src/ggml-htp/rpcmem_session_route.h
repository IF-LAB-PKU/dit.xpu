#pragma once

#include "htp_session.h"

#include <unordered_set>
#include <vector>

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
