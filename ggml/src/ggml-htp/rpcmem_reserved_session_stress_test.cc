#include <dlfcn.h>
#include <errno.h>
#include <AEEStdErr.h>
#include <remote.h>
#include <rpcmem.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <vector>

namespace {

using session_control_fn = int (*)(uint32_t, void *, uint32_t);

struct Api {
    void * handle = nullptr;
    session_control_fn session_control = nullptr;
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
    api->session_control = symbol<session_control_fn>(api->handle, "remote_session_control");
    return api->session_control != nullptr;
}

struct Buffer {
    void * ptr = nullptr;
    int fd = -1;
    bool mapped = false;
};

int run_window(int domain, size_t size, int count, int window) {
    std::vector<Buffer> buffers(static_cast<size_t>(count));
    int rc = 0;

    printf("WINDOW_BEGIN domain=%d size=%zu count=%d window=%d\n", domain, size, count, window);
    fflush(stdout);

    for (int i = 0; i < count && rc == 0; ++i) {
        Buffer & current = buffers[static_cast<size_t>(i)];
        current.ptr = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, static_cast<int>(size));
        if (current.ptr == nullptr) {
            fprintf(stderr, "WINDOW_FAIL phase=alloc index=%d\n", i);
            rc = 3;
            break;
        }
        current.fd = rpcmem_to_fd(current.ptr);
        if (current.fd < 0) {
            fprintf(stderr, "WINDOW_FAIL phase=fd index=%d fd=%d\n", i, current.fd);
            rc = 4;
            break;
        }

        if (i >= window) {
            Buffer & old = buffers[static_cast<size_t>(i - window)];
            if (old.mapped) {
                int err = fastrpc_munmap(domain, old.fd, old.ptr, size);
                printf("WINDOW_UNMAP index=%d fd=%d ret=%d\n", i - window, old.fd, err);
                old.mapped = false;
            }
        }

        int err = fastrpc_mmap(domain, current.fd, current.ptr, 0, size, FASTRPC_MAP_FD);
        printf("WINDOW_MAP index=%d fd=%d ret=%d\n", i, current.fd, err);
        if (err != 0) {
            rc = 5;
            break;
        }
        current.mapped = true;
    }

    for (Buffer & buffer : buffers) {
        if (buffer.mapped) {
            fastrpc_munmap(domain, buffer.fd, buffer.ptr, size);
        }
        if (buffer.ptr != nullptr) {
            rpcmem_free(buffer.ptr);
        }
    }
    printf("WINDOW_EXIT domain=%d rc=%d\n", domain, rc);
    fflush(stdout);
    return rc;
}

int run_two_domains(int first_domain, int second_domain, size_t size, int count, int window) {
    std::vector<Buffer> buffers(static_cast<size_t>(count) * 2);
    int rc = 0;
    printf("TWO_DOMAIN_BEGIN first=%d second=%d size=%zu count=%d window=%d\n", first_domain, second_domain, size,
           count, window);
    fflush(stdout);

    for (int group = 0; group < 2 && rc == 0; ++group) {
        const int domain = group == 0 ? first_domain : second_domain;
        for (int i = 0; i < count; ++i) {
            Buffer & buffer = buffers[static_cast<size_t>(group * count + i)];
            buffer.ptr = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, static_cast<int>(size));
            if (!buffer.ptr) {
                fprintf(stderr, "TWO_DOMAIN_FAIL phase=alloc group=%d index=%d\n", group, i);
                rc = 3;
                break;
            }
            buffer.fd = rpcmem_to_fd(buffer.ptr);
            if (buffer.fd < 0) {
                fprintf(stderr, "TWO_DOMAIN_FAIL phase=fd group=%d index=%d fd=%d\n", group, i, buffer.fd);
                rc = 4;
                break;
            }
            if (i >= window) {
                Buffer & old = buffers[static_cast<size_t>(group * count + i - window)];
                if (old.mapped) {
                    int unmap_err = fastrpc_munmap(domain, old.fd, old.ptr, size);
                    printf("TWO_DOMAIN_UNMAP group=%d domain=%d index=%d fd=%d ret=%d\n", group, domain,
                           i - window, old.fd, unmap_err);
                    old.mapped = false;
                }
            }
            int err = fastrpc_mmap(domain, buffer.fd, buffer.ptr, 0, size, FASTRPC_MAP_FD);
            printf("TWO_DOMAIN_MAP group=%d domain=%d index=%d fd=%d ret=%d\n", group, domain, i, buffer.fd, err);
            if (err != 0) {
                rc = 5;
                break;
            }
            buffer.mapped = true;
        }
    }

    for (size_t i = 0; i < buffers.size(); ++i) {
        Buffer & buffer = buffers[i];
        const int domain = i < static_cast<size_t>(count) ? first_domain : second_domain;
        if (buffer.mapped) {
            fastrpc_munmap(domain, buffer.fd, buffer.ptr, size);
        }
        if (buffer.ptr) {
            rpcmem_free(buffer.ptr);
        }
    }
    printf("TWO_DOMAIN_EXIT rc=%d\n", rc);
    fflush(stdout);
    return rc;
}

bool reserve_session(Api * api, uint32_t * session_id, uint32_t * effective_domain_id) {
    char domain_name[] = CDSP_DOMAIN_NAME;
    char session_name[] = "sd-rpcmem-reserved";
    remote_rpc_reserve_new_session request = {};
    request.domain_name = domain_name;
    request.domain_name_len = static_cast<uint32_t>(strlen(domain_name));
    request.session_name = session_name;
    request.session_name_len = static_cast<uint32_t>(strlen(session_name));

    int err = api->session_control(FASTRPC_RESERVE_NEW_SESSION, &request, sizeof(request));
    printf("RESERVE ret=%d session_id=%u effective_domain_id=%u\n", err, request.session_id,
           request.effective_domain_id);
    if (err != AEE_SUCCESS) {
        return false;
    }
    *session_id = request.session_id;
    *effective_domain_id = request.effective_domain_id;
    return true;
}

} // namespace

int main(int argc, char ** argv) {
    const size_t mib = argc > 1 ? strtoull(argv[1], nullptr, 10) : 256;
    const int count = argc > 2 ? atoi(argv[2]) : 16;
    const int window = argc > 3 ? atoi(argv[3]) : 4;
    const int mode = argc > 4 ? atoi(argv[4]) : 1; // 0=default, 1=reserved sliding window, 2=two domains
    const size_t size = mib * 1024ULL * 1024ULL;
    if (count <= window || window <= 0 || (mode < 0 || mode > 2) || size > INT32_MAX) {
        fprintf(stderr, "invalid setup\n");
        return 2;
    }

    Api api;
    if (!load(&api)) {
        return 2;
    }

    if (mode == 2) {
        uint32_t session_id = 0;
        uint32_t effective_domain_id = 0;
        if (!reserve_session(&api, &session_id, &effective_domain_id)) {
            return 6;
        }
        printf("TWO_DOMAIN_READY session_id=%u reserved_domain=%u default_domain=%d\n", session_id,
               effective_domain_id, CDSP_DOMAIN_ID);
        fflush(stdout);
        return run_two_domains(CDSP_DOMAIN_ID, static_cast<int>(effective_domain_id), size, count - 2, window);
    }

    int domain = CDSP_DOMAIN_ID;
    if (mode == 1) {
        uint32_t session_id = 0;
        uint32_t effective_domain_id = 0;
        if (!reserve_session(&api, &session_id, &effective_domain_id)) {
            return 6;
        }
        domain = static_cast<int>(effective_domain_id);
        printf("RESERVED_READY session_id=%u domain=%d\n", session_id, domain);
    } else {
        printf("DEFAULT_READY domain=%d\n", domain);
    }
    fflush(stdout);
    return run_window(domain, size, count, window);
}
