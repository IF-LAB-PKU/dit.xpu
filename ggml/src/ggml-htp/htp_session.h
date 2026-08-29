#pragma once

#include <cstdint>

struct RpcMemMapper;
struct htp_ops_session;

struct HtpSessionRuntime {
    uint32_t id = 0;
    uint32_t domain = 3;
    bool initialized = true;
    bool lease_active = false;
    bool reserved = false;
    bool channel_initialized = false;
    htp_ops_session * native_session = nullptr;
    void * message_channel = nullptr;
    int message_channel_fd = -1;
    RpcMemMapper * mapper = nullptr;
    uint64_t cache_epoch = 0;

    HtpSessionRuntime() = default;
    HtpSessionRuntime(uint32_t session_id, uint32_t effective_domain)
        : id(session_id), domain(effective_domain) {}
};
