#include "ggml-opencl-htp.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <limits>
#include <string>

static bool contains_case_insensitive(const char * text, const char * needle) {
    std::string haystack = text != nullptr ? text : "";
    std::string query    = needle != nullptr ? needle : "";

    auto to_lower = [](unsigned char value) { return static_cast<char>(std::tolower(value)); };
    std::transform(haystack.begin(), haystack.end(), haystack.begin(), to_lower);
    std::transform(query.begin(), query.end(), query.begin(), to_lower);
    return haystack.find(query) != std::string::npos;
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
    return 0;
}
