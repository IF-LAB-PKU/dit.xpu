#include "ggml-htp.h"
#include "ggml-opencl-htp.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <limits>
#include <string>

#include <sys/stat.h>

static bool contains_case_insensitive(const char * text, const char * needle) {
    std::string haystack = text != nullptr ? text : "";
    std::string query    = needle != nullptr ? needle : "";

    auto to_lower = [](unsigned char value) { return static_cast<char>(std::tolower(value)); };
    std::transform(haystack.begin(), haystack.end(), haystack.begin(), to_lower);
    std::transform(query.begin(), query.end(), query.begin(), to_lower);
    return haystack.find(query) != std::string::npos;
}

static void test_rpcmem_allocation_metadata(const ggml_opencl_qcom_requirements & requirements) {
    GGML_ASSERT(setenv("SD_HTP_SPARSE_OPENCL", "1", 1) == 0);

    ggml_backend_t htp = ggml_backend_htp_init();
    GGML_ASSERT(htp != nullptr);
    GGML_ASSERT(ggml_backend_htp_runtime_available());

    ggml_backend_buffer_t empty_buffer = ggml_backend_alloc_buffer(htp, 0);
    GGML_ASSERT(empty_buffer != nullptr);
    ggml_backend_htp_rpcmem_info empty_info = {};
    GGML_ASSERT(!ggml_backend_htp_get_rpcmem_info(empty_buffer, &empty_info));
    GGML_ASSERT(empty_info.base == nullptr && empty_info.fd == -1);
    ggml_backend_buffer_free(empty_buffer);

    ggml_backend_htp_rpcmem_info invalid_info = {
        reinterpret_cast<void *>(uintptr_t{1}), 7, 8, 9,
    };
    GGML_ASSERT(!ggml_backend_htp_get_rpcmem_info(nullptr, &invalid_info));
    GGML_ASSERT(invalid_info.base == nullptr && invalid_info.fd == -1);
    GGML_ASSERT(invalid_info.logical_size == 0 && invalid_info.allocation_size == 0);

    ggml_backend_t cpu = ggml_backend_cpu_init();
    GGML_ASSERT(cpu != nullptr);
    ggml_backend_buffer_t cpu_buffer = ggml_backend_alloc_buffer(cpu, 128);
    GGML_ASSERT(cpu_buffer != nullptr);
    GGML_ASSERT(!ggml_backend_htp_get_rpcmem_info(cpu_buffer, &invalid_info));
    GGML_ASSERT(invalid_info.base == nullptr && invalid_info.fd == -1);
    ggml_backend_buffer_free(cpu_buffer);
    ggml_backend_free(cpu);

    const size_t too_large = static_cast<size_t>(std::numeric_limits<int>::max()) + 1;
    GGML_ASSERT(ggml_backend_alloc_buffer(htp, too_large) == nullptr);

    const size_t logical_size = requirements.page_size + 123;
    ggml_backend_buffer_t buffer = ggml_backend_alloc_buffer(htp, logical_size);
    GGML_ASSERT(buffer != nullptr);
    GGML_ASSERT(ggml_backend_buffer_get_size(buffer) == logical_size);

    ggml_backend_htp_rpcmem_info info = {};
    GGML_ASSERT(ggml_backend_htp_get_rpcmem_info(buffer, &info));
    GGML_ASSERT(info.base != nullptr);
    GGML_ASSERT(info.fd >= 0);
    GGML_ASSERT(info.logical_size == logical_size);
    GGML_ASSERT(info.allocation_size >= info.logical_size);
    GGML_ASSERT(info.allocation_size - info.logical_size >= requirements.ext_mem_padding);
    GGML_ASSERT(reinterpret_cast<uintptr_t>(info.base) % requirements.page_size == 0);

    struct stat fd_stat = {};
    GGML_ASSERT(fstat(info.fd, &fd_stat) == 0);
    GGML_ASSERT(fd_stat.st_size >= 0);
    GGML_ASSERT(static_cast<uint64_t>(fd_stat.st_size) >= info.allocation_size);

    std::printf("rpcmem_base=%p\n", info.base);
    std::printf("rpcmem_fd=%d\n", info.fd);
    std::printf("rpcmem_logical_size=%zu\n", info.logical_size);
    std::printf("rpcmem_allocation_size=%zu\n", info.allocation_size);

    ggml_backend_free(htp);
    ggml_backend_buffer_free(buffer);
}

int main() {
    ggml_opencl_qcom_requirements requirements = {
        /* .ext_mem_padding = */ std::numeric_limits<size_t>::max(),
        /* .page_size       = */ 0,
    };
    char reason[512] = {};

    ggml_opencl_htp_sidecar * sidecar =
        ggml_backend_opencl_htp_sidecar_create(&requirements, reason, sizeof(reason));
    if (sidecar == nullptr) {
        std::fprintf(stderr, "OpenCL HTP sidecar capability probe failed: %s\n", reason);
        return 1;
    }

    const char * platform_vendor = ggml_backend_opencl_htp_sidecar_platform_vendor(sidecar);
    const char * device_name     = ggml_backend_opencl_htp_sidecar_device_name(sidecar);
    const char * extensions      = ggml_backend_opencl_htp_sidecar_device_extensions(sidecar);

    GGML_ASSERT(contains_case_insensitive(platform_vendor, "qualcomm"));
    GGML_ASSERT(contains_case_insensitive(device_name, "adreno"));
    GGML_ASSERT(ggml_backend_opencl_htp_sidecar_has_extension(sidecar, "cl_qcom_ext_host_ptr"));
    GGML_ASSERT(ggml_backend_opencl_htp_sidecar_has_extension(sidecar, "cl_qcom_dmabuf_host_ptr"));
    GGML_ASSERT(ggml_backend_opencl_htp_sidecar_has_extension(sidecar, "cl_khr_fp16"));
    GGML_ASSERT(!ggml_backend_opencl_htp_sidecar_has_extension(sidecar, "cl_qcom_ext_host"));
    GGML_ASSERT(!ggml_backend_opencl_htp_sidecar_has_extension(sidecar, "cl_qcom_ext_host_ptr_suffix"));
    GGML_ASSERT(ggml_backend_opencl_htp_sidecar_queue_is_in_order(sidecar));
    GGML_ASSERT(requirements.page_size > 0);
    GGML_ASSERT(requirements.ext_mem_padding != std::numeric_limits<size_t>::max());

    std::printf("platform_vendor=%s\n", platform_vendor);
    std::printf("device_name=%s\n", device_name);
    std::printf("extensions=%s\n", extensions);
    std::printf("page_size=%zu\n", requirements.page_size);
    std::printf("ext_padding=%zu\n", requirements.ext_mem_padding);

    ggml_backend_opencl_htp_sidecar_free(sidecar);
    test_rpcmem_allocation_metadata(requirements);
    return 0;
}
