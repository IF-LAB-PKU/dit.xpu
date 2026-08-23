#include <dlfcn.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <vector>

namespace {

using rpcmem_alloc_fn = void * (*)(int, uint32_t, int);
using rpcmem_free_fn = void (*)(void *);
using rpcmem_to_fd_fn = int (*)(void *);
using fastrpc_mmap_fn = int (*)(int, int, void *, int, size_t, int);
using fastrpc_munmap_fn = int (*)(int, int, void *, size_t);

struct Api {
    void * handle = nullptr;
    rpcmem_alloc_fn alloc = nullptr;
    rpcmem_free_fn free = nullptr;
    rpcmem_to_fd_fn to_fd = nullptr;
    fastrpc_mmap_fn mmap = nullptr;
    fastrpc_munmap_fn munmap = nullptr;
};

template <typename T>
T load_symbol(void * handle, const char * name) {
    return reinterpret_cast<T>(dlsym(handle, name));
}

bool load_api(Api * api) {
    api->handle = dlopen("libcdsprpc.so", RTLD_NOW | RTLD_LOCAL);
    if (api->handle == nullptr) {
        fprintf(stderr, "dlopen failed: %s\n", dlerror());
        return false;
    }
    api->alloc = load_symbol<rpcmem_alloc_fn>(api->handle, "rpcmem_alloc");
    api->free = load_symbol<rpcmem_free_fn>(api->handle, "rpcmem_free");
    api->to_fd = load_symbol<rpcmem_to_fd_fn>(api->handle, "rpcmem_to_fd");
    api->mmap = load_symbol<fastrpc_mmap_fn>(api->handle, "fastrpc_mmap");
    api->munmap = load_symbol<fastrpc_munmap_fn>(api->handle, "fastrpc_munmap");
    return api->alloc != nullptr && api->free != nullptr && api->to_fd != nullptr &&
           api->mmap != nullptr && api->munmap != nullptr;
}

} // namespace

int main(int argc, char ** argv) {
    const size_t mib = argc > 1 ? strtoull(argv[1], nullptr, 10) : 200;
    const int count = argc > 2 ? atoi(argv[2]) : 8;
    const int rounds = argc > 3 ? atoi(argv[3]) : 20;
    const size_t size = mib * 1024ULL * 1024ULL;
    constexpr int domain = 3;
    constexpr int heap = 25;
    constexpr uint32_t flags = 0;
    constexpr int map_flags = 2;

    Api api;
    if (!load_api(&api) || count <= 1 || rounds <= 0) {
        fprintf(stderr, "invalid setup\n");
        return 2;
    }

    struct Buffer { void * ptr; int fd; };
    std::vector<Buffer> buffers;
    buffers.reserve(static_cast<size_t>(count));
    for (int i = 0; i < count; ++i) {
        void * ptr = api.alloc(heap, flags, static_cast<int>(size));
        if (ptr == nullptr) {
            fprintf(stderr, "ALLOC_FAIL slot=%d size=%zu\n", i, size);
            for (const auto & b : buffers) api.free(b.ptr);
            return 3;
        }
        int fd = api.to_fd(ptr);
        if (fd < 0) {
            fprintf(stderr, "FD_FAIL slot=%d fd=%d\n", i, fd);
            api.free(ptr);
            for (const auto & b : buffers) api.free(b.ptr);
            return 4;
        }
        buffers.push_back({ptr, fd});
    }

    int rc = 0;
    printf("MULTI_STRESS size=%zu count=%d rounds=%d\n", size, count, rounds);
    for (int round = 0; round < rounds && rc == 0; ++round) {
        std::vector<int> mapped;
        mapped.reserve(static_cast<size_t>(count));
        for (int slot = 0; slot < count; ++slot) {
            int err = api.mmap(domain, buffers[slot].fd, buffers[slot].ptr, 0, size, map_flags);
            if (err != 0) {
                fprintf(stderr, "MAP_FAIL round=%d slot=%d fd=%d err=0x%x\n", round, slot,
                        buffers[slot].fd, err);
                rc = 5;
                break;
            }
            mapped.push_back(slot);
        }
        if (rc != 0) {
            for (int slot : mapped) api.munmap(domain, buffers[slot].fd, buffers[slot].ptr, size);
            break;
        }

        for (int index = count - 2; index >= 0; --index) {
            int err = api.munmap(domain, buffers[index].fd, buffers[index].ptr, size);
            if (err != 0) {
                fprintf(stderr, "UNMAP_FAIL round=%d slot=%d fd=%d err=0x%x\n", round, index,
                        buffers[index].fd, err);
                rc = 6;
                break;
            }
        }
        if (rc == 0) {
            for (int slot = 0; slot < count - 1; ++slot) {
                int err = api.mmap(domain, buffers[slot].fd, buffers[slot].ptr, 0, size, map_flags);
                if (err != 0) {
                    fprintf(stderr, "REMAP_FAIL round=%d slot=%d fd=%d err=0x%x\n", round, slot,
                            buffers[slot].fd, err);
                    rc = 7;
                    break;
                }
            }
        }
        for (int slot = count - 1; slot >= 0; --slot) {
            api.munmap(domain, buffers[slot].fd, buffers[slot].ptr, size);
        }
        if (rc == 0) {
            printf("ROUND_OK %d\n", round);
        }
    }

    for (const auto & b : buffers) api.free(b.ptr);
    printf("MULTI_STRESS_EXIT=%d\n", rc);
    return rc;
}
