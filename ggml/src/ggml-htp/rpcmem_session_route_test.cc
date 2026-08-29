#include "rpcmem_session_route.h"

#include <cassert>
#include <vector>

int main() {
    std::vector<HtpSessionRuntime> sessions = {
        HtpSessionRuntime{0, 3},
        HtpSessionRuntime{1, 7},
        HtpSessionRuntime{2, 11},
        HtpSessionRuntime{3, 15},
    };

    assert(htp_rpcmem_session_index_for_domain(sessions, 3) == 0);
    assert(htp_rpcmem_session_index_for_domain(sessions, 7) == 1);
    assert(htp_rpcmem_session_index_for_domain(sessions, 11) == 2);
    assert(htp_rpcmem_session_index_for_domain(sessions, 15) == 3);
    assert(htp_rpcmem_session_index_for_domain(sessions, 19) == -1);

    auto * mapper0 = reinterpret_cast<RpcMemMapper *>(0x10);
    auto * mapper1 = reinterpret_cast<RpcMemMapper *>(0x20);
    auto * mapper2 = reinterpret_cast<RpcMemMapper *>(0x30);
    sessions[0].mapper = mapper0;
    sessions[1].mapper = mapper1;
    sessions[2].mapper = mapper2;
    sessions[3].mapper = mapper1;

    const auto mappers = htp_unique_rpcmem_session_mappers(sessions);
    assert(mappers.size() == 3);
    assert(mappers[0] == mapper0);
    assert(mappers[1] == mapper1);
    assert(mappers[2] == mapper2);
    return 0;
}
