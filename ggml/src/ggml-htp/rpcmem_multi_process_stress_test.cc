#include <dlfcn.h>
#include <errno.h>
#include <signal.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>

#include <vector>

namespace {

using alloc_fn = void * (*)(int, uint32_t, int);
using free_fn = void (*)(void *);
using to_fd_fn = int (*)(void *);
using mmap_fn = int (*)(int, int, void *, int, size_t, int);
using munmap_fn = int (*)(int, int, void *, size_t);

struct Api {
    void * handle = nullptr;
    alloc_fn alloc = nullptr;
    free_fn free = nullptr;
    to_fd_fn to_fd = nullptr;
    mmap_fn mmap = nullptr;
    munmap_fn munmap = nullptr;
};

template <typename T>
T symbol(void * handle, const char * name) {
    return reinterpret_cast<T>(dlsym(handle, name));
}

bool load(Api * api) {
    api->handle = dlopen("libcdsprpc.so", RTLD_NOW | RTLD_LOCAL);
    if (api->handle == nullptr) {
        fprintf(stderr, "DLOPEN_FAIL %s\n", dlerror());
        return false;
    }
    api->alloc = symbol<alloc_fn>(api->handle, "rpcmem_alloc");
    api->free = symbol<free_fn>(api->handle, "rpcmem_free");
    api->to_fd = symbol<to_fd_fn>(api->handle, "rpcmem_to_fd");
    api->mmap = symbol<mmap_fn>(api->handle, "fastrpc_mmap");
    api->munmap = symbol<munmap_fn>(api->handle, "fastrpc_munmap");
    return api->alloc && api->free && api->to_fd && api->mmap && api->munmap;
}

int worker(int worker_id, size_t size, int count, int window, bool free_on_failure) {
    constexpr int domain = 3;
    Api api;
    if (!load(&api)) {
        return 2;
    }

    struct Buffer {
        void * ptr = nullptr;
        int fd = -1;
        bool mapped = false;
        bool freed = false;
    };
    std::vector<Buffer> buffers(static_cast<size_t>(count));
    int next_free = 0;
    int rc = 0;

    printf("WORKER_BEGIN id=%d pid=%d size=%zu count=%d window=%d free_on_failure=%d\n",
           worker_id, static_cast<int>(getpid()), size, count, window, free_on_failure ? 1 : 0);
    fflush(stdout);

    for (int i = 0; i < count && rc == 0; ++i) {
        buffers[i].ptr = api.alloc(25, 0, static_cast<int>(size));
        if (buffers[i].ptr == nullptr) {
            fprintf(stderr, "WORKER_FAIL id=%d phase=alloc index=%d\n", worker_id, i);
            rc = 3;
            break;
        }
        buffers[i].fd = api.to_fd(buffers[i].ptr);
        if (buffers[i].fd < 0) {
            fprintf(stderr, "WORKER_FAIL id=%d phase=fd index=%d fd=%d\n", worker_id, i, buffers[i].fd);
            rc = 4;
            break;
        }

        if (i >= window) {
            Buffer & old = buffers[i - window];
            if (old.mapped) {
                int err = api.munmap(domain, old.fd, old.ptr, size);
                printf("WORKER_UNMAP id=%d index=%d fd=%d ret=%d\n", worker_id, i - window, old.fd, err);
                old.mapped = false;
            }
        }

        int err;
        do {
            err = api.mmap(domain, buffers[i].fd, buffers[i].ptr, 0, size, 2);
            printf("WORKER_MAP id=%d index=%d fd=%d ret=%d\n", worker_id, i, buffers[i].fd, err);
            if (err == 0) {
                buffers[i].mapped = true;
                break;
            }
            if (!free_on_failure) {
                rc = 5;
                break;
            }
            while (next_free < i && (buffers[next_free].freed || buffers[next_free].mapped)) {
                ++next_free;
            }
            if (next_free >= i) {
                rc = 6;
                break;
            }
            Buffer & victim = buffers[next_free++];
            api.free(victim.ptr);
            victim.ptr = nullptr;
            victim.freed = true;
            printf("WORKER_FREE id=%d index=%d fd=%d\n", worker_id, next_free - 1, victim.fd);
        } while (err != 0);
    }

    for (Buffer & buffer : buffers) {
        if (buffer.mapped) {
            api.munmap(domain, buffer.fd, buffer.ptr, size);
        }
        if (buffer.ptr != nullptr && !buffer.freed) {
            api.free(buffer.ptr);
        }
    }
    printf("WORKER_EXIT id=%d pid=%d rc=%d\n", worker_id, static_cast<int>(getpid()), rc);
    fflush(stdout);
    return rc;
}

} // namespace

int main(int argc, char ** argv) {
    const size_t mib = argc > 1 ? strtoull(argv[1], nullptr, 10) : 200;
    const int count = argc > 2 ? atoi(argv[2]) : 8;
    const int window = argc > 3 ? atoi(argv[3]) : 4;
    const int workers = argc > 4 ? atoi(argv[4]) : 2;
    const bool free_on_failure = argc > 5 && atoi(argv[5]) != 0;
    const size_t size = mib * 1024ULL * 1024ULL;
    if (count <= window || window <= 0 || workers <= 0 || workers > 8 || size > INT32_MAX) {
        fprintf(stderr, "invalid setup\n");
        return 2;
    }

    printf("MULTI_PROCESS_BEGIN size=%zu count=%d window=%d workers=%d free_on_failure=%d parent_pid=%d\n",
           size, count, window, workers, free_on_failure ? 1 : 0, static_cast<int>(getpid()));
    fflush(stdout);

    std::vector<pid_t> children;
    children.reserve(static_cast<size_t>(workers));
    for (int worker_id = 0; worker_id < workers; ++worker_id) {
        pid_t pid = fork();
        if (pid < 0) {
            perror("fork");
            return 3;
        }
        if (pid == 0) {
            _exit(worker(worker_id, size, count, window, free_on_failure));
        }
        children.push_back(pid);
    }

    int failed = 0;
    for (pid_t pid : children) {
        int status = 0;
        while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {
        }
        if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
            failed = 1;
            fprintf(stderr, "CHILD_FAIL pid=%d status=%d\n", static_cast<int>(pid), status);
        }
    }
    printf("MULTI_PROCESS_EXIT=%d\n", failed);
    return failed;
}
