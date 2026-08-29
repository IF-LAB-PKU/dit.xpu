#include "message.h"
#include "op_reg.h"

#include <cassert>
#include <cstddef>

int main() {
    static_assert(offsetof(RpcmemMapRequest, fds) == sizeof(int32_t) * 2);
    static_assert(offsetof(RpcmemMapRequestV2, entries) == 24);
    assert(RPCMEM_MAP_REQUEST_VERSION_SESSION == 2);
    assert(RPCMEM_MAP_REQUEST_SESSION_MAGIC != 0);
    const size_t one_entry = sizeof(RpcmemMapRequestV2) + sizeof(RpcmemMapEntryV2);
    assert(one_entry == offsetof(RpcmemMapRequestV2, entries) + sizeof(RpcmemMapEntryV2));
    // Trace instrumentation must not change the host/DSP request ABI.  The
    // DSP skeleton can enable tracing independently from the HLOS backend.
    static_assert(sizeof(MatMulParams) == 92);
    return 0;
}
