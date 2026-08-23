// 在运行时从 libcdsprpc.so 动态加载 DSPRPC 相关符号，封装为薄层 C 函数。
// 在 Android 平台会预先加载 libbinder.so 以避免供应商 binder 实现的符号冲突。
// 所有 rpcmem_* / fastrpc_* 函数通过此层间接调用，使得 HTP 后端可在缺少 Qualcomm DSP 的平台上优雅降级。

#include "dsprpc_interface.h"

#include <dlfcn.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>

using rpcmem_init_t   = decltype(rpcmem_init);
using rpcmem_deinit_t = decltype(rpcmem_deinit);
using rpcmem_alloc_t  = decltype(rpcmem_alloc);
using rpcmem_free_t   = decltype(rpcmem_free);
using rpcmem_to_fd_t  = decltype(rpcmem_to_fd);

using fastrpc_mmap_t   = decltype(fastrpc_mmap);
using fastrpc_munmap_t = decltype(fastrpc_munmap);

namespace {

bool rpcmem_trace_enabled() {
    static const bool enabled = [] {
        const char * value = std::getenv("SD_HTP_RPCMEM_TRACE");
        return value != nullptr && value[0] != '\0' && value[0] != '0';
    }();
    return enabled;
}

std::atomic<unsigned long long> rpcmem_trace_sequence{0};

unsigned long long next_rpcmem_trace_sequence() {
    return rpcmem_trace_sequence.fetch_add(1, std::memory_order_relaxed);
}

#if defined(__ANDROID__)
void preload_android_system_binder() {
    static std::once_flag once;

    std::call_once(once, []() {
        // Force the system binder implementation into the global namespace so
        // libbinder_ndk.so does not resolve against the vendor variant that
        // lacks openDeclaredPassthroughHal() on newer Android devices.
        (void) dlopen("/system/lib64/libbinder.so", RTLD_NOW | RTLD_GLOBAL);
    });
}
#endif

struct dsprpc_interface {
    void * lib = nullptr;
    std::string error;

    rpcmem_init_t * rpcmem_init_fn = nullptr;
    rpcmem_deinit_t * rpcmem_deinit_fn = nullptr;
    rpcmem_alloc_t * rpcmem_alloc_fn = nullptr;
    rpcmem_free_t * rpcmem_free_fn = nullptr;
    rpcmem_to_fd_t * rpcmem_to_fd_fn = nullptr;
    fastrpc_mmap_t * fastrpc_mmap_fn = nullptr;
    fastrpc_munmap_t * fastrpc_munmap_fn = nullptr;

    void load() {
        static std::once_flag once;
        std::call_once(once, [&]() {
#if defined(__ANDROID__)
            preload_android_system_binder();
#endif
            lib = dlopen("libcdsprpc.so", RTLD_LAZY | RTLD_LOCAL);
            if (!lib) {
                const char * err = dlerror();
                error            = err ? err : "unknown dlopen(libcdsprpc.so) failure";
                return;
            }

            rpcmem_init_fn   = reinterpret_cast<rpcmem_init_t *>(dlsym(lib, "rpcmem_init"));
            rpcmem_deinit_fn = reinterpret_cast<rpcmem_deinit_t *>(dlsym(lib, "rpcmem_deinit"));
            rpcmem_alloc_fn  = reinterpret_cast<rpcmem_alloc_t *>(dlsym(lib, "rpcmem_alloc"));
            rpcmem_free_fn   = reinterpret_cast<rpcmem_free_t *>(dlsym(lib, "rpcmem_free"));
            rpcmem_to_fd_fn  = reinterpret_cast<rpcmem_to_fd_t *>(dlsym(lib, "rpcmem_to_fd"));
            fastrpc_mmap_fn  = reinterpret_cast<fastrpc_mmap_t *>(dlsym(lib, "fastrpc_mmap"));
            fastrpc_munmap_fn = reinterpret_cast<fastrpc_munmap_t *>(dlsym(lib, "fastrpc_munmap"));
        });
    }

    static dsprpc_interface & instance() {
        static dsprpc_interface iface;
        iface.load();
        return iface;
    }
};

} // namespace

extern "C" {

bool dsprpc_is_available(void) {
    auto & iface = dsprpc_interface::instance();
    return iface.lib != nullptr;
}

const char * dsprpc_last_error(void) {
    auto & iface = dsprpc_interface::instance();
    return iface.error.empty() ? nullptr : iface.error.c_str();
}

void rpcmem_init(void) {
    auto & iface = dsprpc_interface::instance();
    if (iface.rpcmem_init_fn) {
        iface.rpcmem_init_fn();
    }
}

void rpcmem_deinit(void) {
    auto & iface = dsprpc_interface::instance();
    if (iface.rpcmem_deinit_fn) {
        iface.rpcmem_deinit_fn();
    }
}

void * rpcmem_alloc(int heap_id, uint32_t flags, int size) {
    auto & iface = dsprpc_interface::instance();
    void * result = iface.rpcmem_alloc_fn ? iface.rpcmem_alloc_fn(heap_id, flags, size) : nullptr;
    if (rpcmem_trace_enabled()) {
        std::fprintf(stderr, "[rpcmem-trace] #%llu alloc heap=%d flags=%u size=%d ptr=%p\n",
                     next_rpcmem_trace_sequence(), heap_id, flags, size, result);
    }
    return result;
}

void rpcmem_free(void * p) {
    auto & iface = dsprpc_interface::instance();
    if (rpcmem_trace_enabled()) {
        std::fprintf(stderr, "[rpcmem-trace] #%llu free ptr=%p\n",
                     next_rpcmem_trace_sequence(), p);
    }
    if (iface.rpcmem_free_fn) {
        iface.rpcmem_free_fn(p);
    }
}

int rpcmem_to_fd(void * p) {
    auto & iface = dsprpc_interface::instance();
    int result = iface.rpcmem_to_fd_fn ? iface.rpcmem_to_fd_fn(p) : -1;
    if (rpcmem_trace_enabled()) {
        std::fprintf(stderr, "[rpcmem-trace] #%llu to_fd ptr=%p fd=%d\n",
                     next_rpcmem_trace_sequence(), p, result);
    }
    return result;
}

int fastrpc_mmap(int domain, int fd, void * addr, int offset, size_t length, enum fastrpc_map_flags flags) {
    auto & iface = dsprpc_interface::instance();
    int result = iface.fastrpc_mmap_fn ? iface.fastrpc_mmap_fn(domain, fd, addr, offset, length, flags) : -1;
    if (rpcmem_trace_enabled()) {
        std::fprintf(stderr, "[rpcmem-trace] #%llu mmap domain=%d fd=%d addr=%p offset=%d length=%zu flags=%d ret=%d\n",
                     next_rpcmem_trace_sequence(), domain, fd, addr, offset, length, flags, result);
    }
    return result;
}

int fastrpc_munmap(int domain, int fd, void * addr, size_t length) {
    auto & iface = dsprpc_interface::instance();
    int result = iface.fastrpc_munmap_fn ? iface.fastrpc_munmap_fn(domain, fd, addr, length) : -1;
    if (rpcmem_trace_enabled()) {
        std::fprintf(stderr, "[rpcmem-trace] #%llu munmap domain=%d fd=%d addr=%p length=%zu ret=%d\n",
                     next_rpcmem_trace_sequence(), domain, fd, addr, length, result);
    }
    return result;
}

} // extern "C"
