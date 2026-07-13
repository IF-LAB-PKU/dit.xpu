#define CL_TARGET_OPENCL_VERSION GGML_OPENCL_TARGET_VERSION

#include "ggml-opencl-htp.h"
#include "ggml-opencl-qcom.h"

#include <algorithm>
#include <cctype>
#include <cstdarg>
#include <cstdio>
#include <memory>
#include <new>
#include <string>
#include <vector>

struct ggml_opencl_htp_sidecar {
    cl_platform_id platform = nullptr;
    cl_device_id device = nullptr;
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    std::string platform_vendor;
    std::string device_name;
    std::string device_extensions;
    bool queue_is_in_order = false;

    ~ggml_opencl_htp_sidecar() {
        if (queue != nullptr) {
            clReleaseCommandQueue(queue);
        }
        if (context != nullptr) {
            clReleaseContext(context);
        }
    }
};

namespace {

void set_reason(char * reason, size_t capacity, const char * format, ...) {
    if (reason == nullptr || capacity == 0) {
        return;
    }

    va_list args;
    va_start(args, format);
    std::vsnprintf(reason, capacity, format, args);
    va_end(args);
    reason[capacity - 1] = '\0';
}

bool contains_case_insensitive(const std::string & text, const char * needle) {
    std::string normalized_text = text;
    std::string normalized_needle = needle != nullptr ? needle : "";
    auto to_lower = [](unsigned char value) { return static_cast<char>(std::tolower(value)); };
    std::transform(normalized_text.begin(), normalized_text.end(), normalized_text.begin(), to_lower);
    std::transform(normalized_needle.begin(), normalized_needle.end(), normalized_needle.begin(), to_lower);
    return normalized_text.find(normalized_needle) != std::string::npos;
}

bool has_extension_token(const std::string & extensions, const char * extension) {
    if (extension == nullptr || extension[0] == '\0') {
        return false;
    }

    const std::string token(extension);
    size_t begin = 0;
    while (begin < extensions.size()) {
        while (begin < extensions.size() && std::isspace(static_cast<unsigned char>(extensions[begin]))) {
            ++begin;
        }
        size_t end = begin;
        while (end < extensions.size() && !std::isspace(static_cast<unsigned char>(extensions[end]))) {
            ++end;
        }
        if (end - begin == token.size() && extensions.compare(begin, token.size(), token) == 0) {
            return true;
        }
        begin = end;
    }
    return false;
}

bool get_platform_string(cl_platform_id platform, cl_platform_info param, std::string * value) {
    size_t size = 0;
    if (clGetPlatformInfo(platform, param, 0, nullptr, &size) != CL_SUCCESS || size == 0) {
        return false;
    }
    std::vector<char> storage(size);
    if (clGetPlatformInfo(platform, param, size, storage.data(), nullptr) != CL_SUCCESS) {
        return false;
    }
    *value = storage.data();
    return true;
}

bool get_device_string(cl_device_id device, cl_device_info param, std::string * value) {
    size_t size = 0;
    if (clGetDeviceInfo(device, param, 0, nullptr, &size) != CL_SUCCESS || size == 0) {
        return false;
    }
    std::vector<char> storage(size);
    if (clGetDeviceInfo(device, param, size, storage.data(), nullptr) != CL_SUCCESS) {
        return false;
    }
    *value = storage.data();
    return true;
}

} // namespace

ggml_opencl_htp_sidecar * ggml_backend_opencl_htp_sidecar_create(
    ggml_opencl_qcom_requirements * requirements,
    char * reason,
    size_t reason_capacity) {
    if (requirements == nullptr) {
        set_reason(reason, reason_capacity, "requirements must not be null");
        return nullptr;
    }
    requirements->ext_mem_padding = 0;
    requirements->page_size = 0;
    set_reason(reason, reason_capacity, "uninitialized");

    cl_uint platform_count = 0;
    cl_int err = clGetPlatformIDs(0, nullptr, &platform_count);
    if (err != CL_SUCCESS || platform_count == 0) {
        set_reason(reason, reason_capacity, "no OpenCL platform available (error %d)", err);
        return nullptr;
    }

    std::vector<cl_platform_id> platforms(platform_count);
    err = clGetPlatformIDs(platform_count, platforms.data(), nullptr);
    if (err != CL_SUCCESS) {
        set_reason(reason, reason_capacity, "clGetPlatformIDs failed (error %d)", err);
        return nullptr;
    }

    std::unique_ptr<ggml_opencl_htp_sidecar> sidecar(new (std::nothrow) ggml_opencl_htp_sidecar());
    if (!sidecar) {
        set_reason(reason, reason_capacity, "failed to allocate sidecar state");
        return nullptr;
    }

    for (cl_platform_id platform : platforms) {
        std::string vendor;
        if (get_platform_string(platform, CL_PLATFORM_VENDOR, &vendor) &&
            contains_case_insensitive(vendor, "qualcomm")) {
            sidecar->platform = platform;
            sidecar->platform_vendor = std::move(vendor);
            break;
        }
    }
    if (sidecar->platform == nullptr) {
        set_reason(reason, reason_capacity, "no Qualcomm OpenCL platform found");
        return nullptr;
    }

    cl_uint device_count = 0;
    err = clGetDeviceIDs(sidecar->platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &device_count);
    if (err != CL_SUCCESS || device_count == 0) {
        set_reason(reason, reason_capacity, "no Qualcomm OpenCL GPU found (error %d)", err);
        return nullptr;
    }

    std::vector<cl_device_id> devices(device_count);
    err = clGetDeviceIDs(sidecar->platform, CL_DEVICE_TYPE_GPU, device_count, devices.data(), nullptr);
    if (err != CL_SUCCESS) {
        set_reason(reason, reason_capacity, "clGetDeviceIDs failed (error %d)", err);
        return nullptr;
    }

    for (cl_device_id device : devices) {
        std::string name;
        if (get_device_string(device, CL_DEVICE_NAME, &name) &&
            contains_case_insensitive(name, "adreno")) {
            sidecar->device = device;
            sidecar->device_name = std::move(name);
            break;
        }
    }
    if (sidecar->device == nullptr) {
        set_reason(reason, reason_capacity, "no Adreno OpenCL device found");
        return nullptr;
    }

    if (!get_device_string(sidecar->device, CL_DEVICE_EXTENSIONS, &sidecar->device_extensions)) {
        set_reason(reason, reason_capacity, "failed to query OpenCL device extensions");
        return nullptr;
    }

    const char * required_extensions[] = {
        "cl_qcom_ext_host_ptr",
        "cl_qcom_dmabuf_host_ptr",
        "cl_khr_fp16",
    };
    for (const char * extension : required_extensions) {
        if (!has_extension_token(sidecar->device_extensions, extension)) {
            set_reason(reason, reason_capacity, "required OpenCL extension is missing: %s", extension);
            return nullptr;
        }
    }

    cl_uint page_size = 0;
    err = clGetDeviceInfo(sidecar->device, CL_DEVICE_PAGE_SIZE_QCOM, sizeof(page_size), &page_size, nullptr);
    if (err != CL_SUCCESS || page_size == 0) {
        set_reason(reason, reason_capacity, "failed to query QCOM page size (error %d)", err);
        return nullptr;
    }

    cl_uint ext_mem_padding = 0;
    err = clGetDeviceInfo(sidecar->device, CL_DEVICE_EXT_MEM_PADDING_IN_BYTES_QCOM,
                          sizeof(ext_mem_padding), &ext_mem_padding, nullptr);
    if (err != CL_SUCCESS) {
        set_reason(reason, reason_capacity, "failed to query QCOM external memory padding (error %d)", err);
        return nullptr;
    }

    const cl_context_properties context_properties[] = {
        CL_CONTEXT_PLATFORM,
        reinterpret_cast<cl_context_properties>(sidecar->platform),
        0,
    };
    sidecar->context = clCreateContext(context_properties, 1, &sidecar->device, nullptr, nullptr, &err);
    if (err != CL_SUCCESS || sidecar->context == nullptr) {
        set_reason(reason, reason_capacity, "clCreateContext failed (error %d)", err);
        return nullptr;
    }

    const cl_queue_properties queue_properties[] = {
        CL_QUEUE_PROPERTIES,
        0,
        0,
    };
    sidecar->queue = clCreateCommandQueueWithProperties(
        sidecar->context, sidecar->device, queue_properties, &err);
    if (err != CL_SUCCESS || sidecar->queue == nullptr) {
        set_reason(reason, reason_capacity, "clCreateCommandQueueWithProperties failed (error %d)", err);
        return nullptr;
    }

    cl_command_queue_properties actual_queue_properties = 0;
    err = clGetCommandQueueInfo(sidecar->queue, CL_QUEUE_PROPERTIES,
                                sizeof(actual_queue_properties), &actual_queue_properties, nullptr);
    if (err != CL_SUCCESS) {
        set_reason(reason, reason_capacity, "failed to query OpenCL queue properties (error %d)", err);
        return nullptr;
    }
    sidecar->queue_is_in_order =
        (actual_queue_properties & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) == 0;
    if (!sidecar->queue_is_in_order) {
        set_reason(reason, reason_capacity, "OpenCL command queue is not in-order");
        return nullptr;
    }

    requirements->ext_mem_padding = ext_mem_padding;
    requirements->page_size = page_size;
    set_reason(reason, reason_capacity, "ok");
    return sidecar.release();
}

void ggml_backend_opencl_htp_sidecar_free(ggml_opencl_htp_sidecar * sidecar) {
    delete sidecar;
}

const char * ggml_backend_opencl_htp_sidecar_platform_vendor(const ggml_opencl_htp_sidecar * sidecar) {
    return sidecar != nullptr ? sidecar->platform_vendor.c_str() : "";
}

const char * ggml_backend_opencl_htp_sidecar_device_name(const ggml_opencl_htp_sidecar * sidecar) {
    return sidecar != nullptr ? sidecar->device_name.c_str() : "";
}

const char * ggml_backend_opencl_htp_sidecar_device_extensions(const ggml_opencl_htp_sidecar * sidecar) {
    return sidecar != nullptr ? sidecar->device_extensions.c_str() : "";
}

bool ggml_backend_opencl_htp_sidecar_has_extension(
    const ggml_opencl_htp_sidecar * sidecar,
    const char * extension) {
    return sidecar != nullptr && has_extension_token(sidecar->device_extensions, extension);
}

bool ggml_backend_opencl_htp_sidecar_queue_is_in_order(const ggml_opencl_htp_sidecar * sidecar) {
    return sidecar != nullptr && sidecar->queue_is_in_order;
}
