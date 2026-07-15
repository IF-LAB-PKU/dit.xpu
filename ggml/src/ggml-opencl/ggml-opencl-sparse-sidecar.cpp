#define CL_TARGET_OPENCL_VERSION GGML_OPENCL_TARGET_VERSION

#include "ggml-opencl-htp.h"
#include "ggml-opencl-qcom.h"
#include "ggml-impl.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <climits>
#include <cctype>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <vector>

#include <sys/stat.h>

struct ggml_opencl_htp_sidecar {
    cl_platform_id platform = nullptr;
    cl_device_id device = nullptr;
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_program block_sparse_program = nullptr;
    cl_kernel block_sparse_kernel = nullptr;
    std::string platform_vendor;
    std::string device_name;
    std::string device_extensions;
    size_t ext_mem_padding = 0;
    size_t page_size = 0;
    cl_ulong max_mem_alloc_size = 0;
    bool queue_is_in_order = false;
    std::mutex dispatch_mutex;

    ~ggml_opencl_htp_sidecar() {
        if (block_sparse_kernel != nullptr) {
            clReleaseKernel(block_sparse_kernel);
        }
        if (block_sparse_program != nullptr) {
            clReleaseProgram(block_sparse_program);
        }
        if (queue != nullptr) {
            clReleaseCommandQueue(queue);
        }
        if (context != nullptr) {
            clReleaseContext(context);
        }
    }
};

struct ggml_opencl_htp_alias {
    ggml_opencl_htp_import_descriptor descriptor = {};
    cl_mem memory = nullptr;
};

struct ggml_opencl_htp_tensor_binding {
    cl_mem memory = nullptr;
    ggml_backend_buffer_t backing = nullptr;
    size_t offset = 0;
    size_t span = 0;
};

enum class ggml_opencl_htp_graph_state {
    active,
    finished,
};

struct ggml_opencl_htp_graph {
    ggml_opencl_htp_sidecar * sidecar = nullptr;
    std::vector<ggml_opencl_htp_alias> aliases;
    std::vector<cl_event> events;
    std::vector<const ggml_tensor *> routed_nodes;
    ggml_opencl_htp_graph_state state = ggml_opencl_htp_graph_state::active;
    ggml_opencl_htp_result finish_result = GGML_OPENCL_HTP_OK;
    char finish_reason[256] = {};
    bool cleanup_failed = false;
    bool test_fail_next_event_cache = false;
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

std::string get_program_build_log(cl_program program, cl_device_id device) {
    size_t size = 0;
    cl_int err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &size);
    if (err != CL_SUCCESS || size == 0) {
        return "build log unavailable";
    }
    std::vector<char> storage(size + 1, '\0');
    err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, size, storage.data(), nullptr);
    if (err != CL_SUCCESS) {
        return "failed to query build log";
    }
    return storage.data();
}

ggml_opencl_htp_result ensure_block_sparse_kernel(
    ggml_opencl_htp_sidecar * sidecar,
    char * reason,
    size_t reason_capacity) {
    if (sidecar->block_sparse_kernel != nullptr) {
        return GGML_OPENCL_HTP_OK;
    }

#ifdef GGML_OPENCL_EMBED_KERNELS
    const std::string kernel_source {
#include "flash_attn_block_sparse_f32_f16.cl.h"
    };
#else
    set_reason(
        reason, reason_capacity,
        "block-sparse HTP sidecar requires GGML_OPENCL_EMBED_KERNELS");
    return GGML_OPENCL_HTP_UNSUPPORTED;
#endif

#ifdef GGML_OPENCL_EMBED_KERNELS
    const char * source = kernel_source.c_str();
    cl_int err = CL_SUCCESS;
    cl_program program = clCreateProgramWithSource(
        sidecar->context, 1, &source, nullptr, &err);
    if (err != CL_SUCCESS || program == nullptr) {
        set_reason(reason, reason_capacity, "clCreateProgramWithSource failed (error %d)", err);
        return GGML_OPENCL_HTP_ERROR;
    }

    static constexpr const char * build_options = "-cl-std=CL3.0 -cl-mad-enable";
    err = clBuildProgram(program, 1, &sidecar->device, build_options, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        const std::string build_log = get_program_build_log(program, sidecar->device);
        const cl_int release_err = clReleaseProgram(program);
        set_reason(
            reason, reason_capacity,
            "clBuildProgram failed (error %d, release error %d): %s",
            err, release_err, build_log.c_str());
        return GGML_OPENCL_HTP_ERROR;
    }

    cl_kernel kernel = clCreateKernel(program, "flash_attn_block_sparse_f32_f16", &err);
    if (err != CL_SUCCESS || kernel == nullptr) {
        const cl_int release_err = clReleaseProgram(program);
        set_reason(
            reason, reason_capacity,
            "clCreateKernel failed (error %d, program release error %d)",
            err, release_err);
        return GGML_OPENCL_HTP_ERROR;
    }

    sidecar->block_sparse_program = program;
    sidecar->block_sparse_kernel = kernel;
    return GGML_OPENCL_HTP_OK;
#endif
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

    size_t page_size = 0;
    static_assert(sizeof(page_size) == sizeof(size_t), "QCOM page size queries use size_t values");
    err = clGetDeviceInfo(sidecar->device, CL_DEVICE_PAGE_SIZE_QCOM, sizeof(page_size), &page_size, nullptr);
    if (err != CL_SUCCESS || page_size == 0) {
        set_reason(reason, reason_capacity, "failed to query QCOM page size (error %d)", err);
        return nullptr;
    }

    size_t ext_mem_padding = 0;
    static_assert(sizeof(ext_mem_padding) == sizeof(size_t), "QCOM padding queries use size_t values");
    err = clGetDeviceInfo(sidecar->device, CL_DEVICE_EXT_MEM_PADDING_IN_BYTES_QCOM,
                          sizeof(ext_mem_padding), &ext_mem_padding, nullptr);
    if (err != CL_SUCCESS) {
        set_reason(reason, reason_capacity, "failed to query QCOM external memory padding (error %d)", err);
        return nullptr;
    }

    cl_ulong max_mem_alloc_size = 0;
    err = clGetDeviceInfo(sidecar->device, CL_DEVICE_MAX_MEM_ALLOC_SIZE,
                          sizeof(max_mem_alloc_size), &max_mem_alloc_size, nullptr);
    if (err != CL_SUCCESS || max_mem_alloc_size == 0) {
        set_reason(reason, reason_capacity, "failed to query OpenCL max allocation size (error %d)", err);
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

    sidecar->ext_mem_padding = ext_mem_padding;
    sidecar->page_size = page_size;
    sidecar->max_mem_alloc_size = max_mem_alloc_size;
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

namespace {

bool descriptors_equal(
    const ggml_opencl_htp_import_descriptor & lhs,
    const ggml_opencl_htp_import_descriptor & rhs) {
    return lhs.backing == rhs.backing &&
           lhs.base == rhs.base &&
           lhs.fd == rhs.fd &&
           lhs.logical_size == rhs.logical_size &&
           lhs.allocation_size == rhs.allocation_size;
}

ggml_opencl_htp_alias * find_alias(
    ggml_opencl_htp_graph * graph,
    ggml_backend_buffer_t backing) {
    if (graph == nullptr) {
        return nullptr;
    }
    for (ggml_opencl_htp_alias & alias : graph->aliases) {
        if (alias.descriptor.backing == backing) {
            return &alias;
        }
    }
    return nullptr;
}

bool checked_mul_size(size_t lhs, size_t rhs, size_t * result) {
    if (lhs != 0 && rhs > SIZE_MAX / lhs) {
        return false;
    }
    *result = lhs * rhs;
    return true;
}

bool checked_add_size(size_t lhs, size_t rhs, size_t * result) {
    if (rhs > SIZE_MAX - lhs) {
        return false;
    }
    *result = lhs + rhs;
    return true;
}

ggml_opencl_htp_result resolve_tensor_binding(
    ggml_opencl_htp_graph * graph,
    const ggml_tensor * tensor,
    ggml_opencl_htp_tensor_binding * binding,
    char * reason,
    size_t reason_capacity) {
    if (graph == nullptr || tensor == nullptr || binding == nullptr) {
        set_reason(reason, reason_capacity, "graph, tensor, and binding must not be null");
        return GGML_OPENCL_HTP_ERROR;
    }
    if (tensor->data == nullptr) {
        set_reason(reason, reason_capacity, "tensor data must not be null");
        return GGML_OPENCL_HTP_ERROR;
    }

    ggml_backend_buffer_t backing = nullptr;
    const ggml_tensor * source = tensor;
    for (size_t depth = 0; source != nullptr && depth < 64; ++depth) {
        if (source->buffer != nullptr) {
            backing = source->buffer;
            break;
        }
        source = source->view_src;
    }
    if (backing == nullptr) {
        set_reason(reason, reason_capacity, "tensor has no allocated backing buffer");
        return GGML_OPENCL_HTP_ERROR;
    }

    ggml_opencl_htp_alias * alias = find_alias(graph, backing);
    if (alias == nullptr) {
        set_reason(
            reason, reason_capacity, "tensor backing %p has not been imported",
            static_cast<void *>(backing));
        return GGML_OPENCL_HTP_ERROR;
    }

    const int64_t block_size = ggml_blck_size(tensor->type);
    const size_t type_size = ggml_type_size(tensor->type);
    if (block_size <= 0 || type_size == 0) {
        set_reason(reason, reason_capacity, "tensor has an invalid element type");
        return GGML_OPENCL_HTP_ERROR;
    }
    for (size_t dimension = 0; dimension < GGML_MAX_DIMS; ++dimension) {
        if (tensor->ne[dimension] <= 0) {
            set_reason(
                reason, reason_capacity, "tensor dimension %zu must be positive",
                dimension);
            return GGML_OPENCL_HTP_ERROR;
        }
    }

    size_t span = type_size;
    const uint64_t block_count =
        static_cast<uint64_t>((tensor->ne[0] - 1) / block_size) + 1;
    if (block_count - 1 > SIZE_MAX) {
        set_reason(reason, reason_capacity, "tensor dimension 0 exceeds addressable size");
        return GGML_OPENCL_HTP_ERROR;
    }

    size_t contribution = 0;
    if (!checked_mul_size(static_cast<size_t>(block_count - 1), tensor->nb[0], &contribution) ||
        !checked_add_size(span, contribution, &span)) {
        set_reason(reason, reason_capacity, "tensor dimension 0 span overflows size_t");
        return GGML_OPENCL_HTP_ERROR;
    }
    for (size_t dimension = 1; dimension < GGML_MAX_DIMS; ++dimension) {
        const uint64_t count = static_cast<uint64_t>(tensor->ne[dimension] - 1);
        if (count > SIZE_MAX ||
            !checked_mul_size(static_cast<size_t>(count), tensor->nb[dimension], &contribution) ||
            !checked_add_size(span, contribution, &span)) {
            set_reason(
                reason, reason_capacity, "tensor dimension %zu span overflows size_t",
                dimension);
            return GGML_OPENCL_HTP_ERROR;
        }
    }

    const uintptr_t base_address = reinterpret_cast<uintptr_t>(alias->descriptor.base);
    const uintptr_t data_address = reinterpret_cast<uintptr_t>(tensor->data);
    if (data_address < base_address) {
        set_reason(reason, reason_capacity, "tensor data precedes imported backing base");
        return GGML_OPENCL_HTP_ERROR;
    }
    const uintptr_t raw_offset = data_address - base_address;
    if (raw_offset > SIZE_MAX) {
        set_reason(reason, reason_capacity, "tensor offset exceeds addressable size");
        return GGML_OPENCL_HTP_ERROR;
    }
    const size_t offset = static_cast<size_t>(raw_offset);
    if (offset > alias->descriptor.logical_size ||
        span > alias->descriptor.logical_size - offset) {
        set_reason(
            reason, reason_capacity,
            "tensor range offset=%zu span=%zu exceeds logical size %zu",
            offset, span, alias->descriptor.logical_size);
        return GGML_OPENCL_HTP_ERROR;
    }

    binding->memory = alias->memory;
    binding->backing = backing;
    binding->offset = offset;
    binding->span = span;
    set_reason(reason, reason_capacity, "ok");
    return GGML_OPENCL_HTP_OK;
}

ggml_opencl_htp_result finish_graph_state(
    ggml_opencl_htp_graph * graph,
    ggml_opencl_htp_result requested_result,
    const char * requested_reason,
    char * reason,
    size_t reason_capacity) {
    if (graph->state == ggml_opencl_htp_graph_state::finished) {
        set_reason(reason, reason_capacity, "%s", graph->finish_reason);
        return graph->finish_result;
    }

    const char * failed_operation = nullptr;
    cl_int failed_error = CL_SUCCESS;
    auto record_error = [&](const char * operation, cl_int error) {
        if (error != CL_SUCCESS && failed_operation == nullptr) {
            failed_operation = operation;
            failed_error = error;
        }
    };

    record_error("clFinish", clFinish(graph->sidecar->queue));
    for (cl_event event : graph->events) {
        if (event != nullptr) {
            record_error("clReleaseEvent", clReleaseEvent(event));
        }
    }
    graph->events.clear();
    for (ggml_opencl_htp_alias & alias : graph->aliases) {
        if (alias.memory != nullptr) {
            record_error("clReleaseMemObject", clReleaseMemObject(alias.memory));
            alias.memory = nullptr;
        }
    }
    graph->aliases.clear();
    graph->routed_nodes.clear();
    graph->state = ggml_opencl_htp_graph_state::finished;
    graph->cleanup_failed = failed_operation != nullptr;

    if (requested_result != GGML_OPENCL_HTP_OK) {
        graph->finish_result = GGML_OPENCL_HTP_ERROR;
        if (failed_operation != nullptr) {
            std::snprintf(
                graph->finish_reason, sizeof(graph->finish_reason),
                "%s; cleanup %s failed (error %d)",
                requested_reason, failed_operation, failed_error);
        } else {
            std::snprintf(
                graph->finish_reason, sizeof(graph->finish_reason), "%s", requested_reason);
        }
    } else if (failed_operation != nullptr) {
        graph->finish_result = GGML_OPENCL_HTP_ERROR;
        std::snprintf(
            graph->finish_reason, sizeof(graph->finish_reason),
            "%s failed (error %d)", failed_operation, failed_error);
    } else {
        graph->finish_result = GGML_OPENCL_HTP_OK;
        std::snprintf(graph->finish_reason, sizeof(graph->finish_reason), "ok");
    }

    set_reason(reason, reason_capacity, "%s", graph->finish_reason);
    return graph->finish_result;
}

ggml_opencl_htp_result fail_graph(
    ggml_opencl_htp_graph * graph,
    char * reason,
    size_t reason_capacity,
    const char * format,
    ...) {
    char failure_reason[256] = {};
    va_list args;
    va_start(args, format);
    std::vsnprintf(failure_reason, sizeof(failure_reason), format, args);
    va_end(args);
    failure_reason[sizeof(failure_reason) - 1] = '\0';
    return finish_graph_state(
        graph, GGML_OPENCL_HTP_ERROR, failure_reason, reason, reason_capacity);
}

ggml_opencl_htp_result fail_submitted_graph(
    ggml_opencl_htp_graph * graph,
    cl_event event,
    const char * operation,
    cl_int operation_error,
    char * reason,
    size_t reason_capacity) {
    const cl_int finish_err = clFinish(graph->sidecar->queue);
    const cl_int release_err = event != nullptr ? clReleaseEvent(event) : CL_SUCCESS;
    return fail_graph(
        graph, reason, reason_capacity,
        "%s failed (error %d, command drain error %d, event release error %d)",
        operation, operation_error, finish_err, release_err);
}

ggml_opencl_htp_result validate_import_descriptor(
    const ggml_opencl_htp_graph * graph,
    const ggml_opencl_htp_import_descriptor * descriptor,
    char * reason,
    size_t reason_capacity) {
    if (descriptor == nullptr) {
        set_reason(reason, reason_capacity, "import descriptor must not be null");
        return GGML_OPENCL_HTP_ERROR;
    }
    if (descriptor->backing == nullptr) {
        set_reason(reason, reason_capacity, "import backing must not be null");
        return GGML_OPENCL_HTP_ERROR;
    }
    if (descriptor->base == nullptr) {
        set_reason(reason, reason_capacity, "import base must not be null");
        return GGML_OPENCL_HTP_ERROR;
    }
    if (descriptor->fd < 0) {
        set_reason(reason, reason_capacity, "import fd must be non-negative");
        return GGML_OPENCL_HTP_ERROR;
    }
    if (descriptor->logical_size == 0) {
        set_reason(reason, reason_capacity, "import logical size must be nonzero");
        return GGML_OPENCL_HTP_ERROR;
    }
    if (descriptor->allocation_size < descriptor->logical_size) {
        set_reason(
            reason, reason_capacity,
            "import allocation size %zu is smaller than logical size %zu",
            descriptor->allocation_size, descriptor->logical_size);
        return GGML_OPENCL_HTP_ERROR;
    }

    const ggml_opencl_htp_sidecar * sidecar = graph->sidecar;
    if (sidecar->ext_mem_padding > descriptor->allocation_size - descriptor->logical_size) {
        set_reason(
            reason, reason_capacity,
            "import allocation size %zu does not include %zu bytes of external padding",
            descriptor->allocation_size, sidecar->ext_mem_padding);
        return GGML_OPENCL_HTP_ERROR;
    }
    if (sidecar->page_size == 0 ||
        reinterpret_cast<uintptr_t>(descriptor->base) % sidecar->page_size != 0) {
        set_reason(
            reason, reason_capacity,
            "import base %p is not aligned to page size %zu",
            descriptor->base, sidecar->page_size);
        return GGML_OPENCL_HTP_ERROR;
    }

    struct stat fd_stat = {};
    if (fstat(descriptor->fd, &fd_stat) != 0) {
        set_reason(
            reason, reason_capacity, "fstat failed for import fd %d: %s",
            descriptor->fd, std::strerror(errno));
        return GGML_OPENCL_HTP_ERROR;
    }
    if (fd_stat.st_size < 0 ||
        static_cast<uint64_t>(fd_stat.st_size) < descriptor->allocation_size) {
        set_reason(
            reason, reason_capacity,
            "import fd %d size is smaller than allocation size %zu",
            descriptor->fd, descriptor->allocation_size);
        return GGML_OPENCL_HTP_ERROR;
    }

    return GGML_OPENCL_HTP_OK;
}

struct ggml_opencl_htp_sparse_dispatch {
    const ggml_tensor * q = nullptr;
    const ggml_tensor * k = nullptr;
    const ggml_tensor * v = nullptr;
    const ggml_tensor * mask = nullptr;
    const ggml_tensor * output = nullptr;
    float scale = 0.0f;
    cl_int lq = 0;
    cl_int lk = 0;
    cl_int heads = 0;
    cl_int kv_heads = 0;
    cl_int batch = 0;
    cl_int mask_format = 0;
    cl_int max_kept_blocks = 0;
};

ggml_opencl_htp_result unsupported_sparse_node(
    char * reason,
    size_t reason_capacity,
    const char * format,
    ...) {
    if (reason != nullptr && reason_capacity > 0) {
        va_list args;
        va_start(args, format);
        std::vsnprintf(reason, reason_capacity, format, args);
        va_end(args);
        reason[reason_capacity - 1] = '\0';
    }
    return GGML_OPENCL_HTP_UNSUPPORTED;
}

ggml_opencl_htp_result validate_block_sparse_node(
    const ggml_tensor * node,
    ggml_opencl_htp_sparse_dispatch * dispatch,
    char * reason,
    size_t reason_capacity) {
    if (node == nullptr || dispatch == nullptr) {
        set_reason(reason, reason_capacity, "node and dispatch must not be null");
        return GGML_OPENCL_HTP_ERROR;
    }
    if (node->op != GGML_OP_FLASH_ATTN_EXT) {
        return unsupported_sparse_node(
            reason, reason_capacity, "node op %d is not FLASH_ATTN_EXT", node->op);
    }

    const ggml_tensor * q = node->src[0];
    const ggml_tensor * k = node->src[1];
    const ggml_tensor * v = node->src[2];
    const ggml_tensor * mask = node->src[3];
    if (q == nullptr || k == nullptr || v == nullptr || mask == nullptr) {
        return unsupported_sparse_node(
            reason, reason_capacity, "sparse attention requires Q, K, V, and mask tensors");
    }
    if (node->src[4] != nullptr) {
        return unsupported_sparse_node(
            reason, reason_capacity, "attention sinks are not supported by the sparse sidecar");
    }
    if (q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_F16 ||
        v->type != GGML_TYPE_F16 || mask->type != GGML_TYPE_I32 ||
        node->type != GGML_TYPE_F32) {
        return unsupported_sparse_node(
            reason, reason_capacity,
            "required types are Q=F32 K=F16 V=F16 mask=I32 output=F32");
    }

    const int32_t block_q = ggml_get_op_params_i32(node, 4);
    const int32_t block_k = ggml_get_op_params_i32(node, 5);
    const int32_t with_lse = ggml_get_op_params_i32(node, 6);
    const int32_t mask_format = ggml_get_op_params_i32(node, 7);
    if (with_lse != 1) {
        return unsupported_sparse_node(reason, reason_capacity, "with_lse must equal 1");
    }
    if (mask_format != GGML_FLASH_ATTN_EXT_MASK_BLOCK_MATRIX &&
        mask_format != GGML_FLASH_ATTN_EXT_MASK_BLOCK_INDEX) {
        return unsupported_sparse_node(
            reason, reason_capacity, "mask format must be BLOCK_MATRIX or BLOCK_INDEX");
    }
    if (block_q != 32 || block_k != 32) {
        return unsupported_sparse_node(
            reason, reason_capacity, "block sizes must be 32x32");
    }

    float scale = 0.0f;
    float max_bias = 0.0f;
    float logit_softcap = 0.0f;
    std::memcpy(&scale, node->op_params + 0, sizeof(scale));
    std::memcpy(&max_bias, node->op_params + 1, sizeof(max_bias));
    std::memcpy(&logit_softcap, node->op_params + 2, sizeof(logit_softcap));
    if (!std::isfinite(scale)) {
        return unsupported_sparse_node(reason, reason_capacity, "scale must be finite");
    }
    if (max_bias != 0.0f || logit_softcap != 0.0f) {
        return unsupported_sparse_node(
            reason, reason_capacity, "max_bias and logit_softcap must be zero");
    }

    auto valid_cl_int_dimension = [](int64_t value) {
        return value > 0 && value <= INT_MAX;
    };
    if (!valid_cl_int_dimension(q->ne[1]) || !valid_cl_int_dimension(k->ne[1]) ||
        !valid_cl_int_dimension(q->ne[2]) || !valid_cl_int_dimension(k->ne[2]) ||
        !valid_cl_int_dimension(q->ne[3])) {
        return unsupported_sparse_node(
            reason, reason_capacity, "attention dimensions must be positive cl_int values");
    }

    const int64_t lq = q->ne[1];
    const int64_t lk = k->ne[1];
    const int64_t heads = q->ne[2];
    const int64_t kv_heads = k->ne[2];
    const int64_t batch = q->ne[3];
    if (q->ne[0] != 64 || k->ne[0] != 64 || v->ne[0] != 64 || node->ne[0] != 65) {
        return unsupported_sparse_node(
            reason, reason_capacity, "head dimensions must be Q/K/V=64 and output=65");
    }
    if (kv_heads > heads || heads % kv_heads != 0) {
        return unsupported_sparse_node(
            reason, reason_capacity, "query heads must be divisible by KV heads");
    }
    if (heads > INT_MAX / batch) {
        return unsupported_sparse_node(
            reason, reason_capacity, "head and batch product exceeds cl_int range");
    }
    if (k->ne[3] != batch || v->ne[1] != lk || v->ne[2] != kv_heads || v->ne[3] != batch) {
        return unsupported_sparse_node(
            reason, reason_capacity, "K and V shapes or batch dimensions do not match");
    }
    if (node->ne[1] != heads || node->ne[2] != lq || node->ne[3] != batch) {
        return unsupported_sparse_node(
            reason, reason_capacity, "packed output shape must be [65,H,Lq,B]");
    }

    const int64_t q_blocks_per_head = 1 + (lq - 1) / 32;
    const int64_t k_blocks_per_head = 1 + (lk - 1) / 32;
    const int64_t expected_q_blocks = heads * q_blocks_per_head;
    const int64_t expected_k_blocks = kv_heads * k_blocks_per_head;
    if (expected_q_blocks > INT_MAX || expected_k_blocks > INT_MAX) {
        return unsupported_sparse_node(
            reason, reason_capacity, "flattened mask block dimensions exceed cl_int range");
    }
    if (mask_format == GGML_FLASH_ATTN_EXT_MASK_BLOCK_MATRIX) {
        if (mask->ne[0] != expected_k_blocks || mask->ne[1] != expected_q_blocks ||
            mask->ne[2] != 1 || mask->ne[3] != 1) {
            return unsupported_sparse_node(
                reason, reason_capacity,
                "matrix mask must have shape [Hkv*ceil(Lk/32),H*ceil(Lq/32),1,1]");
        }
    } else if (!valid_cl_int_dimension(mask->ne[0]) ||
               mask->ne[1] != expected_q_blocks || mask->ne[2] != 1 || mask->ne[3] != 1) {
        return unsupported_sparse_node(
            reason, reason_capacity,
            "index mask must have shape [max_kept,H*ceil(Lq/32),1,1]");
    }

    if (q->nb[0] != sizeof(float) || k->nb[0] != sizeof(ggml_fp16_t) ||
        v->nb[0] != sizeof(ggml_fp16_t) || node->nb[0] != sizeof(float) ||
        mask->nb[0] != sizeof(int32_t)) {
        return unsupported_sparse_node(
            reason, reason_capacity, "tensor dimension-zero strides do not match their types");
    }
    auto is_dense_scalar_permutation = [](const ggml_tensor * tensor, size_t scalar_size) {
        struct active_dimension {
            size_t stride;
            int64_t count;
        } dimensions[3];
        size_t active_count = 0;
        for (size_t dimension = 1; dimension < GGML_MAX_DIMS; ++dimension) {
            if (tensor->ne[dimension] > 1) {
                dimensions[active_count++] = { tensor->nb[dimension], tensor->ne[dimension] };
            }
        }
        std::sort(
            dimensions, dimensions + active_count,
            [](const active_dimension & lhs, const active_dimension & rhs) {
                return lhs.stride < rhs.stride;
            });
        size_t expected_stride = scalar_size * static_cast<size_t>(tensor->ne[0]);
        for (size_t i = 0; i < active_count; ++i) {
            if (dimensions[i].stride != expected_stride ||
                static_cast<uint64_t>(dimensions[i].count) > SIZE_MAX / expected_stride) {
                return false;
            }
            expected_stride *= static_cast<size_t>(dimensions[i].count);
        }
        return true;
    };
    if (!ggml_is_contiguous(q) ||
        !is_dense_scalar_permutation(k, sizeof(ggml_fp16_t)) ||
        !is_dense_scalar_permutation(v, sizeof(ggml_fp16_t)) ||
        !ggml_is_contiguous(mask) || !ggml_is_contiguous(node)) {
        return unsupported_sparse_node(
            reason, reason_capacity,
            "Q, mask, and output must be contiguous; K and V must be dense scalar permutations");
    }
    auto strides_are_aligned = [](const ggml_tensor * tensor, size_t alignment) {
        for (size_t dimension = 0; dimension < GGML_MAX_DIMS; ++dimension) {
            if (tensor->nb[dimension] % alignment != 0) {
                return false;
            }
        }
        return true;
    };
    if (!strides_are_aligned(q, sizeof(float)) ||
        !strides_are_aligned(k, sizeof(ggml_fp16_t)) ||
        !strides_are_aligned(v, sizeof(ggml_fp16_t)) ||
        !strides_are_aligned(mask, sizeof(int32_t)) ||
        !strides_are_aligned(node, sizeof(float))) {
        return unsupported_sparse_node(
            reason, reason_capacity, "tensor strides must preserve scalar alignment");
    }

    dispatch->q = q;
    dispatch->k = k;
    dispatch->v = v;
    dispatch->mask = mask;
    dispatch->output = node;
    dispatch->scale = scale;
    dispatch->lq = static_cast<cl_int>(lq);
    dispatch->lk = static_cast<cl_int>(lk);
    dispatch->heads = static_cast<cl_int>(heads);
    dispatch->kv_heads = static_cast<cl_int>(kv_heads);
    dispatch->batch = static_cast<cl_int>(batch);
    dispatch->mask_format = static_cast<cl_int>(mask_format);
    dispatch->max_kept_blocks = mask_format == GGML_FLASH_ATTN_EXT_MASK_BLOCK_INDEX ?
        static_cast<cl_int>(mask->ne[0]) : 0;
    set_reason(reason, reason_capacity, "ok");
    return GGML_OPENCL_HTP_OK;
}

bool bindings_overlap(
    const ggml_opencl_htp_tensor_binding & lhs,
    const ggml_opencl_htp_tensor_binding & rhs) {
    if (lhs.backing != rhs.backing) {
        return false;
    }
    return lhs.offset < rhs.offset + rhs.span && rhs.offset < lhs.offset + lhs.span;
}

struct ggml_opencl_htp_sparse_bindings {
    ggml_opencl_htp_tensor_binding q;
    ggml_opencl_htp_tensor_binding k;
    ggml_opencl_htp_tensor_binding v;
    ggml_opencl_htp_tensor_binding output;
    ggml_opencl_htp_tensor_binding mask;
};

ggml_opencl_htp_result resolve_sparse_bindings(
    ggml_opencl_htp_graph * graph,
    const ggml_opencl_htp_sparse_dispatch & dispatch,
    ggml_opencl_htp_sparse_bindings * bindings,
    char * reason,
    size_t reason_capacity) {
    struct binding_request {
        const ggml_tensor * tensor;
        const char * name;
        ggml_opencl_htp_tensor_binding * binding;
        size_t scalar_alignment;
    };
    const binding_request requests[] = {
        { dispatch.q, "Q", &bindings->q, sizeof(float) },
        { dispatch.k, "K", &bindings->k, sizeof(ggml_fp16_t) },
        { dispatch.v, "V", &bindings->v, sizeof(ggml_fp16_t) },
        { dispatch.output, "output", &bindings->output, sizeof(float) },
        { dispatch.mask, "mask", &bindings->mask, sizeof(int32_t) },
    };
    for (const binding_request & request : requests) {
        char binding_reason[256] = {};
        const ggml_opencl_htp_result binding_result = resolve_tensor_binding(
            graph, request.tensor, request.binding,
            binding_reason, sizeof(binding_reason));
        if (binding_result != GGML_OPENCL_HTP_OK) {
            return unsupported_sparse_node(
                reason, reason_capacity,
                "%s binding is unavailable: %s", request.name, binding_reason);
        }
        if (request.binding->offset % request.scalar_alignment != 0) {
            return unsupported_sparse_node(
                reason, reason_capacity,
                "%s binding offset %zu is not aligned to %zu bytes",
                request.name, request.binding->offset, request.scalar_alignment);
        }
    }

    if (bindings_overlap(bindings->output, bindings->q) ||
        bindings_overlap(bindings->output, bindings->k) ||
        bindings_overlap(bindings->output, bindings->v) ||
        bindings_overlap(bindings->output, bindings->mask)) {
        return unsupported_sparse_node(
            reason, reason_capacity,
            "sparse attention output storage overlaps an input span");
    }
    return GGML_OPENCL_HTP_OK;
}

ggml_backend_buffer_t tensor_backing(const ggml_tensor * tensor) {
    const ggml_tensor * source = tensor;
    for (size_t depth = 0; source != nullptr && depth < 64; ++depth, source = source->view_src) {
        if (source->buffer != nullptr) {
            return source->buffer;
        }
    }
    return nullptr;
}

bool is_sparse_route_candidate(const ggml_tensor * node) {
    if (node == nullptr || node->op != GGML_OP_FLASH_ATTN_EXT) {
        return false;
    }
    const ggml_tensor * q = node->src[0];
    const ggml_tensor * k = node->src[1];
    const ggml_tensor * v = node->src[2];
    const ggml_tensor * mask = node->src[3];
    if (q == nullptr || k == nullptr || v == nullptr || mask == nullptr || node->src[4] != nullptr ||
        q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_F16 ||
        v->type != GGML_TYPE_F16 || mask->type != GGML_TYPE_I32 ||
        node->type != GGML_TYPE_F32 || q->ne[0] != 64 || k->ne[0] != 64 ||
        v->ne[0] != 64 || node->ne[0] != 65) {
        return false;
    }
    const int32_t mask_format = ggml_get_op_params_i32(node, 7);
    if (ggml_get_op_params_i32(node, 4) != 32 ||
        ggml_get_op_params_i32(node, 5) != 32 ||
        ggml_get_op_params_i32(node, 6) != 1 ||
        (mask_format != GGML_FLASH_ATTN_EXT_MASK_BLOCK_MATRIX &&
         mask_format != GGML_FLASH_ATTN_EXT_MASK_BLOCK_INDEX)) {
        return false;
    }

    float scale = 0.0f;
    float max_bias = 0.0f;
    float logit_softcap = 0.0f;
    std::memcpy(&scale, node->op_params + 0, sizeof(scale));
    std::memcpy(&max_bias, node->op_params + 1, sizeof(max_bias));
    std::memcpy(&logit_softcap, node->op_params + 2, sizeof(logit_softcap));
    return std::isfinite(scale) && max_bias == 0.0f && logit_softcap == 0.0f;
}

} // namespace

ggml_opencl_htp_result ggml_backend_opencl_htp_graph_create(
    ggml_opencl_htp_sidecar * sidecar,
    ggml_opencl_htp_graph ** graph,
    char * reason,
    size_t reason_capacity) {
    if (graph == nullptr) {
        set_reason(reason, reason_capacity, "graph output must not be null");
        return GGML_OPENCL_HTP_ERROR;
    }
    *graph = nullptr;
    if (sidecar == nullptr) {
        set_reason(reason, reason_capacity, "sidecar must not be null");
        return GGML_OPENCL_HTP_ERROR;
    }
    if (sidecar->context == nullptr || sidecar->queue == nullptr ||
        sidecar->page_size == 0 || !sidecar->queue_is_in_order) {
        set_reason(reason, reason_capacity, "sidecar is not ready for graph imports");
        return GGML_OPENCL_HTP_UNSUPPORTED;
    }

    std::unique_ptr<ggml_opencl_htp_graph> created(new (std::nothrow) ggml_opencl_htp_graph());
    if (!created) {
        set_reason(reason, reason_capacity, "failed to allocate graph state");
        return GGML_OPENCL_HTP_ERROR;
    }
    created->sidecar = sidecar;
    std::snprintf(created->finish_reason, sizeof(created->finish_reason), "ok");
    *graph = created.release();
    set_reason(reason, reason_capacity, "ok");
    return GGML_OPENCL_HTP_OK;
}

ggml_opencl_htp_result ggml_backend_opencl_htp_graph_import(
    ggml_opencl_htp_graph * graph,
    const ggml_opencl_htp_import_descriptor * descriptor,
    char * reason,
    size_t reason_capacity) {
    if (graph == nullptr) {
        set_reason(reason, reason_capacity, "graph must not be null");
        return GGML_OPENCL_HTP_ERROR;
    }
    if (graph->state != ggml_opencl_htp_graph_state::active) {
        set_reason(reason, reason_capacity, "graph is already finished");
        return GGML_OPENCL_HTP_ERROR;
    }

    char validation_reason[256] = {};
    const ggml_opencl_htp_result validation = validate_import_descriptor(
        graph, descriptor, validation_reason, sizeof(validation_reason));
    if (validation != GGML_OPENCL_HTP_OK) {
        return finish_graph_state(
            graph, GGML_OPENCL_HTP_ERROR, validation_reason, reason, reason_capacity);
    }

    if (ggml_opencl_htp_alias * existing = find_alias(graph, descriptor->backing)) {
        if (!descriptors_equal(existing->descriptor, *descriptor)) {
            return fail_graph(
                graph,
                reason, reason_capacity,
                "backing %p was imported with different metadata",
                static_cast<void *>(descriptor->backing));
        }
        set_reason(reason, reason_capacity, "ok");
        return GGML_OPENCL_HTP_OK;
    }

    cl_mem_dmabuf_host_ptr host_ptr = {};
    host_ptr.ext_host_ptr.allocation_type = CL_MEM_DMABUF_HOST_PTR_QCOM;
    host_ptr.ext_host_ptr.host_cache_policy = CL_MEM_HOST_UNCACHED_QCOM;
    host_ptr.dmabuf_filedesc = descriptor->fd;
    host_ptr.dmabuf_hostptr = descriptor->base;

    const cl_mem_flags flags =
        CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM;
    cl_int err = CL_SUCCESS;
    cl_mem memory = clCreateBuffer(
        graph->sidecar->context, flags, descriptor->logical_size, &host_ptr, &err);
    if (err != CL_SUCCESS || memory == nullptr) {
        return fail_graph(
            graph,
            reason, reason_capacity,
            "clCreateBuffer failed for backing %p (error %d)",
            static_cast<void *>(descriptor->backing), err);
    }

    ggml_opencl_htp_alias alias;
    alias.descriptor = *descriptor;
    alias.memory = memory;
    try {
        graph->aliases.push_back(alias);
    } catch (...) {
        const cl_int release_err = clReleaseMemObject(memory);
        return fail_graph(
            graph, reason, reason_capacity,
            "failed to cache imported OpenCL alias (new alias release error %d)", release_err);
    }

    set_reason(reason, reason_capacity, "ok");
    return GGML_OPENCL_HTP_OK;
}

ggml_opencl_htp_result ggml_backend_opencl_htp_graph_preflight(
    ggml_opencl_htp_sidecar * sidecar,
    const ggml_cgraph * cgraph,
    const ggml_opencl_htp_import_descriptor * descriptors,
    size_t descriptor_count,
    ggml_opencl_htp_graph ** graph,
    char * reason,
    size_t reason_capacity) {
    if (graph == nullptr) {
        set_reason(reason, reason_capacity, "graph output must not be null");
        return GGML_OPENCL_HTP_ERROR;
    }
    *graph = nullptr;
    if (sidecar == nullptr || cgraph == nullptr || (descriptor_count != 0 && descriptors == nullptr)) {
        set_reason(reason, reason_capacity, "sidecar, cgraph, and descriptor array must be valid");
        return GGML_OPENCL_HTP_ERROR;
    }

    const char * enabled = std::getenv("SD_HTP_SPARSE_OPENCL");
    if (enabled == nullptr || std::strcmp(enabled, "1") != 0) {
        set_reason(reason, reason_capacity, "SD_HTP_SPARSE_OPENCL=1 is required");
        return GGML_OPENCL_HTP_UNSUPPORTED;
    }

    std::vector<ggml_opencl_htp_sparse_dispatch> dispatches;
    std::vector<const ggml_tensor *> routed_nodes;
    std::vector<ggml_backend_buffer_t> required_backings;
    try {
        const int node_count = ggml_graph_n_nodes(const_cast<ggml_cgraph *>(cgraph));
        for (int i = 0; i < node_count; ++i) {
            const ggml_tensor * node = ggml_graph_node(const_cast<ggml_cgraph *>(cgraph), i);
            if (!is_sparse_route_candidate(node)) {
                continue;
            }

            ggml_opencl_htp_sparse_dispatch dispatch;
            char validation_reason[256] = {};
            const ggml_opencl_htp_result validation = validate_block_sparse_node(
                node, &dispatch, validation_reason, sizeof(validation_reason));
            if (validation != GGML_OPENCL_HTP_OK) {
                set_reason(reason, reason_capacity, "eligible sparse node %d is unsupported: %s", i, validation_reason);
                return validation == GGML_OPENCL_HTP_ERROR ?
                    GGML_OPENCL_HTP_ERROR : GGML_OPENCL_HTP_UNSUPPORTED;
            }
            dispatches.push_back(dispatch);
            routed_nodes.push_back(node);

            const ggml_tensor * tensors[] = {
                dispatch.q, dispatch.k, dispatch.v, dispatch.mask, dispatch.output,
            };
            for (const ggml_tensor * tensor : tensors) {
                ggml_backend_buffer_t backing = tensor_backing(tensor);
                if (backing == nullptr) {
                    set_reason(reason, reason_capacity, "sparse node %d tensor has no backing buffer", i);
                    return GGML_OPENCL_HTP_UNSUPPORTED;
                }
                if (std::find(required_backings.begin(), required_backings.end(), backing) == required_backings.end()) {
                    required_backings.push_back(backing);
                }
            }
        }
    } catch (...) {
        set_reason(reason, reason_capacity, "failed to allocate preflight routing state");
        return GGML_OPENCL_HTP_ERROR;
    }

    if (dispatches.empty()) {
        set_reason(reason, reason_capacity, "graph has no eligible sparse with-LSE attention nodes");
        return GGML_OPENCL_HTP_UNSUPPORTED;
    }

    ggml_opencl_htp_graph * prepared = nullptr;
    const ggml_opencl_htp_result create_result = ggml_backend_opencl_htp_graph_create(
        sidecar, &prepared, reason, reason_capacity);
    if (create_result != GGML_OPENCL_HTP_OK) {
        return GGML_OPENCL_HTP_UNSUPPORTED;
    }
    auto reject_preflight = [&](ggml_opencl_htp_result result, const char * failure_reason) {
        char cleanup_reason[256] = {};
        const ggml_opencl_htp_result cleanup_result =
            ggml_backend_opencl_htp_graph_finish(prepared, cleanup_reason, sizeof(cleanup_reason));
        ggml_backend_opencl_htp_graph_free(&prepared);
        if (cleanup_result != GGML_OPENCL_HTP_OK) {
            set_reason(
                reason, reason_capacity, "%s; preflight cleanup failed: %s",
                failure_reason, cleanup_reason);
            return GGML_OPENCL_HTP_ERROR;
        }
        set_reason(reason, reason_capacity, "%s", failure_reason);
        return result;
    };

    std::vector<ggml_opencl_htp_import_descriptor> required_descriptors;
    try {
        required_descriptors.reserve(required_backings.size());
        for (ggml_backend_buffer_t backing : required_backings) {
            const ggml_opencl_htp_import_descriptor * match = nullptr;
            for (size_t i = 0; i < descriptor_count; ++i) {
                if (descriptors[i].backing != backing) {
                    continue;
                }
                if (match != nullptr && !descriptors_equal(*match, descriptors[i])) {
                    char failure_reason[256] = {};
                    std::snprintf(
                        failure_reason, sizeof(failure_reason),
                        "backing %p has inconsistent preflight descriptors",
                        static_cast<void *>(backing));
                    return reject_preflight(GGML_OPENCL_HTP_UNSUPPORTED, failure_reason);
                }
                match = &descriptors[i];
            }
            if (match == nullptr) {
                char failure_reason[256] = {};
                std::snprintf(
                    failure_reason, sizeof(failure_reason),
                    "missing RPCMEM descriptor for backing %p", static_cast<void *>(backing));
                return reject_preflight(GGML_OPENCL_HTP_UNSUPPORTED, failure_reason);
            }
            required_descriptors.push_back(*match);
        }
    } catch (...) {
        return reject_preflight(GGML_OPENCL_HTP_ERROR, "failed to allocate descriptor preflight state");
    }

    for (const ggml_opencl_htp_import_descriptor & descriptor : required_descriptors) {
        if (descriptor.allocation_size > sidecar->max_mem_alloc_size) {
            char failure_reason[256] = {};
            std::snprintf(
                failure_reason, sizeof(failure_reason),
                "backing %p allocation size %zu exceeds OpenCL maximum %llu",
                static_cast<void *>(descriptor.backing), descriptor.allocation_size,
                static_cast<unsigned long long>(sidecar->max_mem_alloc_size));
            return reject_preflight(GGML_OPENCL_HTP_UNSUPPORTED, failure_reason);
        }
        char validation_reason[256] = {};
        if (validate_import_descriptor(
                prepared, &descriptor, validation_reason, sizeof(validation_reason)) != GGML_OPENCL_HTP_OK) {
            char failure_reason[512] = {};
            std::snprintf(
                failure_reason, sizeof(failure_reason),
                "RPCMEM descriptor is invalid: %s", validation_reason);
            return reject_preflight(GGML_OPENCL_HTP_UNSUPPORTED, failure_reason);
        }
    }

    for (const ggml_opencl_htp_import_descriptor & descriptor : required_descriptors) {
        char import_reason[256] = {};
        if (ggml_backend_opencl_htp_graph_import(
                prepared, &descriptor, import_reason, sizeof(import_reason)) != GGML_OPENCL_HTP_OK) {
            const bool cleanup_failed = prepared->cleanup_failed;
            ggml_backend_opencl_htp_graph_free(&prepared);
            set_reason(reason, reason_capacity, "RPCMEM import failed: %s", import_reason);
            return cleanup_failed ? GGML_OPENCL_HTP_ERROR : GGML_OPENCL_HTP_UNSUPPORTED;
        }
    }

    std::vector<ggml_opencl_htp_tensor_binding> output_bindings;
    try {
        output_bindings.reserve(dispatches.size());
        for (const ggml_opencl_htp_sparse_dispatch & dispatch : dispatches) {
            ggml_opencl_htp_sparse_bindings bindings;
            char binding_reason[256] = {};
            if (resolve_sparse_bindings(
                    prepared, dispatch, &bindings,
                    binding_reason, sizeof(binding_reason)) != GGML_OPENCL_HTP_OK) {
                char failure_reason[512] = {};
                std::snprintf(
                    failure_reason, sizeof(failure_reason),
                    "sparse binding preflight failed: %s", binding_reason);
                return reject_preflight(GGML_OPENCL_HTP_UNSUPPORTED, failure_reason);
            }
            for (const ggml_opencl_htp_tensor_binding & prior : output_bindings) {
                if (bindings_overlap(bindings.output, prior)) {
                    return reject_preflight(
                        GGML_OPENCL_HTP_UNSUPPORTED,
                        "routed sparse attention output spans overlap");
                }
            }
            output_bindings.push_back(bindings.output);
        }
        prepared->routed_nodes = std::move(routed_nodes);
    } catch (...) {
        return reject_preflight(GGML_OPENCL_HTP_ERROR, "failed to allocate binding preflight state");
    }

    {
        std::lock_guard<std::mutex> lock(sidecar->dispatch_mutex);
        char kernel_reason[1024] = {};
        const ggml_opencl_htp_result kernel_result =
            ensure_block_sparse_kernel(sidecar, kernel_reason, sizeof(kernel_reason));
        if (kernel_result != GGML_OPENCL_HTP_OK) {
            return reject_preflight(GGML_OPENCL_HTP_UNSUPPORTED, kernel_reason);
        }
    }

    *graph = prepared;
    set_reason(reason, reason_capacity, "ok");
    return GGML_OPENCL_HTP_OK;
}

size_t ggml_backend_opencl_htp_graph_routed_node_count(const ggml_opencl_htp_graph * graph) {
    return graph != nullptr && graph->state == ggml_opencl_htp_graph_state::active ?
        graph->routed_nodes.size() : 0;
}

bool ggml_backend_opencl_htp_graph_routes_node(
    const ggml_opencl_htp_graph * graph,
    const ggml_tensor * node) {
    return graph != nullptr && graph->state == ggml_opencl_htp_graph_state::active &&
           node != nullptr &&
           std::find(graph->routed_nodes.begin(), graph->routed_nodes.end(), node) !=
               graph->routed_nodes.end();
}

ggml_opencl_htp_result ggml_backend_opencl_htp_graph_fill(
    ggml_opencl_htp_graph * graph,
    ggml_backend_buffer_t backing,
    size_t offset,
    size_t size,
    uint32_t pattern,
    char * reason,
    size_t reason_capacity) {
    if (graph == nullptr || backing == nullptr) {
        set_reason(reason, reason_capacity, "graph and fill backing must not be null");
        return GGML_OPENCL_HTP_ERROR;
    }
    if (graph->state != ggml_opencl_htp_graph_state::active) {
        set_reason(reason, reason_capacity, "graph is already finished");
        return GGML_OPENCL_HTP_ERROR;
    }

    ggml_opencl_htp_alias * alias = find_alias(graph, backing);
    if (alias == nullptr) {
        set_reason(
            reason, reason_capacity, "backing %p has not been imported",
            static_cast<void *>(backing));
        return GGML_OPENCL_HTP_ERROR;
    }
    if (offset > alias->descriptor.logical_size ||
        size > alias->descriptor.logical_size - offset) {
        set_reason(
            reason, reason_capacity,
            "fill range offset=%zu size=%zu exceeds logical size %zu",
            offset, size, alias->descriptor.logical_size);
        return GGML_OPENCL_HTP_ERROR;
    }
    if (size == 0 || offset % sizeof(pattern) != 0 || size % sizeof(pattern) != 0) {
        set_reason(
            reason, reason_capacity,
            "fill offset and nonzero size must be multiples of %zu",
            sizeof(pattern));
        return GGML_OPENCL_HTP_ERROR;
    }

    cl_event event = nullptr;
    const cl_int err = clEnqueueFillBuffer(
        graph->sidecar->queue, alias->memory, &pattern, sizeof(pattern),
        offset, size, 0, nullptr, &event);
    if (err != CL_SUCCESS || event == nullptr) {
        if (event != nullptr) {
            clReleaseEvent(event);
        }
        set_reason(reason, reason_capacity, "clEnqueueFillBuffer failed (error %d)", err);
        return GGML_OPENCL_HTP_ERROR;
    }

    bool event_cache_failed = graph->test_fail_next_event_cache;
    graph->test_fail_next_event_cache = false;
    if (!event_cache_failed) {
        try {
            graph->events.push_back(event);
        } catch (...) {
            event_cache_failed = true;
        }
    }
    if (event_cache_failed) {
        const cl_int finish_err = clFinish(graph->sidecar->queue);
        const cl_int release_err = clReleaseEvent(event);
        return fail_graph(
            graph, reason, reason_capacity,
            "failed to retain fill event (clFinish error %d, event release error %d)",
            finish_err, release_err);
    }

    set_reason(reason, reason_capacity, "ok");
    return GGML_OPENCL_HTP_OK;
}

ggml_opencl_htp_result ggml_backend_opencl_htp_graph_fill_tensor(
    ggml_opencl_htp_graph * graph,
    const ggml_tensor * tensor,
    uint32_t pattern,
    char * reason,
    size_t reason_capacity) {
    if (graph == nullptr) {
        set_reason(reason, reason_capacity, "graph must not be null");
        return GGML_OPENCL_HTP_ERROR;
    }
    if (graph->state != ggml_opencl_htp_graph_state::active) {
        set_reason(reason, reason_capacity, "graph is already finished");
        return GGML_OPENCL_HTP_ERROR;
    }

    ggml_opencl_htp_tensor_binding binding;
    const ggml_opencl_htp_result result = resolve_tensor_binding(
        graph, tensor, &binding, reason, reason_capacity);
    if (result != GGML_OPENCL_HTP_OK) {
        return result;
    }
    return ggml_backend_opencl_htp_graph_fill(
        graph, binding.backing, binding.offset, binding.span,
        pattern, reason, reason_capacity);
}

ggml_opencl_htp_result ggml_backend_opencl_htp_graph_compute_block_sparse_attn_sync(
    ggml_opencl_htp_graph * graph,
    const ggml_tensor * node,
    char * reason,
    size_t reason_capacity) {
    if (graph == nullptr) {
        set_reason(reason, reason_capacity, "graph must not be null");
        return GGML_OPENCL_HTP_ERROR;
    }
    if (graph->state != ggml_opencl_htp_graph_state::active) {
        set_reason(reason, reason_capacity, "graph is already finished");
        return GGML_OPENCL_HTP_ERROR;
    }

    ggml_opencl_htp_sparse_dispatch dispatch;
    const ggml_opencl_htp_result validation = validate_block_sparse_node(
        node, &dispatch, reason, reason_capacity);
    if (validation != GGML_OPENCL_HTP_OK) {
        return validation;
    }

    ggml_opencl_htp_sparse_bindings bindings;
    const ggml_opencl_htp_result binding_result = resolve_sparse_bindings(
        graph, dispatch, &bindings, reason, reason_capacity);
    if (binding_result != GGML_OPENCL_HTP_OK) {
        return binding_result;
    }

    static_assert(sizeof(size_t) <= sizeof(cl_ulong), "OpenCL offsets must hold host size_t values");
    const cl_mem q_memory = bindings.q.memory;
    const cl_mem k_memory = bindings.k.memory;
    const cl_mem v_memory = bindings.v.memory;
    const cl_mem output_memory = bindings.output.memory;
    const cl_mem mask_memory = bindings.mask.memory;
    const cl_ulong q_offset = static_cast<cl_ulong>(bindings.q.offset);
    const cl_ulong k_offset = static_cast<cl_ulong>(bindings.k.offset);
    const cl_ulong v_offset = static_cast<cl_ulong>(bindings.v.offset);
    const cl_ulong output_offset = static_cast<cl_ulong>(bindings.output.offset);
    const cl_ulong mask_offset = static_cast<cl_ulong>(bindings.mask.offset);
    const cl_float scale = dispatch.scale;
    const cl_int lq = dispatch.lq;
    const cl_int lk = dispatch.lk;
    const cl_int heads = dispatch.heads;
    const cl_int kv_heads = dispatch.kv_heads;
    const cl_int mask_format = dispatch.mask_format;
    const cl_int max_kept_blocks = dispatch.max_kept_blocks;
    const cl_ulong q_nb1 = static_cast<cl_ulong>(dispatch.q->nb[1]);
    const cl_ulong q_nb2 = static_cast<cl_ulong>(dispatch.q->nb[2]);
    const cl_ulong q_nb3 = static_cast<cl_ulong>(dispatch.q->nb[3]);
    const cl_ulong k_nb1 = static_cast<cl_ulong>(dispatch.k->nb[1]);
    const cl_ulong k_nb2 = static_cast<cl_ulong>(dispatch.k->nb[2]);
    const cl_ulong k_nb3 = static_cast<cl_ulong>(dispatch.k->nb[3]);
    const cl_ulong v_nb1 = static_cast<cl_ulong>(dispatch.v->nb[1]);
    const cl_ulong v_nb2 = static_cast<cl_ulong>(dispatch.v->nb[2]);
    const cl_ulong v_nb3 = static_cast<cl_ulong>(dispatch.v->nb[3]);
    const cl_ulong output_nb1 = static_cast<cl_ulong>(dispatch.output->nb[1]);
    const cl_ulong output_nb2 = static_cast<cl_ulong>(dispatch.output->nb[2]);
    const cl_ulong output_nb3 = static_cast<cl_ulong>(dispatch.output->nb[3]);
    const cl_ulong mask_nb0 = static_cast<cl_ulong>(dispatch.mask->nb[0]);
    const cl_ulong mask_nb1 = static_cast<cl_ulong>(dispatch.mask->nb[1]);

    std::lock_guard<std::mutex> lock(graph->sidecar->dispatch_mutex);
    char kernel_reason[1024] = {};
    const ggml_opencl_htp_result kernel_result = ensure_block_sparse_kernel(
        graph->sidecar, kernel_reason, sizeof(kernel_reason));
    if (kernel_result == GGML_OPENCL_HTP_UNSUPPORTED) {
        set_reason(reason, reason_capacity, "%s", kernel_reason);
        return kernel_result;
    }
    if (kernel_result != GGML_OPENCL_HTP_OK) {
        return fail_graph(
            graph, reason, reason_capacity,
            "block-sparse kernel creation failed: %s", kernel_reason);
    }

    struct kernel_arg {
        size_t size;
        const void * value;
    };
    const kernel_arg args[] = {
        { sizeof(q_memory), &q_memory },
        { sizeof(q_offset), &q_offset },
        { sizeof(k_memory), &k_memory },
        { sizeof(k_offset), &k_offset },
        { sizeof(v_memory), &v_memory },
        { sizeof(v_offset), &v_offset },
        { sizeof(output_memory), &output_memory },
        { sizeof(output_offset), &output_offset },
        { sizeof(mask_memory), &mask_memory },
        { sizeof(mask_offset), &mask_offset },
        { sizeof(scale), &scale },
        { sizeof(lq), &lq },
        { sizeof(lk), &lk },
        { sizeof(heads), &heads },
        { sizeof(kv_heads), &kv_heads },
        { sizeof(mask_format), &mask_format },
        { sizeof(max_kept_blocks), &max_kept_blocks },
        { sizeof(q_nb1), &q_nb1 },
        { sizeof(q_nb2), &q_nb2 },
        { sizeof(q_nb3), &q_nb3 },
        { sizeof(k_nb1), &k_nb1 },
        { sizeof(k_nb2), &k_nb2 },
        { sizeof(k_nb3), &k_nb3 },
        { sizeof(v_nb1), &v_nb1 },
        { sizeof(v_nb2), &v_nb2 },
        { sizeof(v_nb3), &v_nb3 },
        { sizeof(output_nb1), &output_nb1 },
        { sizeof(output_nb2), &output_nb2 },
        { sizeof(output_nb3), &output_nb3 },
        { sizeof(mask_nb0), &mask_nb0 },
        { sizeof(mask_nb1), &mask_nb1 },
    };
    static_assert(sizeof(args) / sizeof(args[0]) == 31, "block-sparse kernel ABI changed");
    for (cl_uint index = 0; index < sizeof(args) / sizeof(args[0]); ++index) {
        const cl_int err = clSetKernelArg(
            graph->sidecar->block_sparse_kernel, index, args[index].size, args[index].value);
        if (err != CL_SUCCESS) {
            return fail_graph(
                graph, reason, reason_capacity,
                "clSetKernelArg(%u) failed (error %d)", index, err);
        }
    }

    const size_t local_work_size[] = { 32, 1 };
    const size_t global_work_size[] = {
        static_cast<size_t>(1 + (dispatch.lq - 1) / 32) * 32,
        static_cast<size_t>(dispatch.heads) * static_cast<size_t>(dispatch.batch),
    };
    cl_event event = nullptr;
    std::atomic_thread_fence(std::memory_order_release);
    cl_int err = clEnqueueNDRangeKernel(
        graph->sidecar->queue, graph->sidecar->block_sparse_kernel,
        2, nullptr, global_work_size, local_work_size, 0, nullptr, &event);
    if (err != CL_SUCCESS || event == nullptr) {
        if (event != nullptr) {
            return fail_submitted_graph(
                graph, event, "clEnqueueNDRangeKernel", err, reason, reason_capacity);
        }
        return fail_graph(
            graph, reason, reason_capacity,
            "clEnqueueNDRangeKernel failed (error %d, no event)", err);
    }

    err = clFlush(graph->sidecar->queue);
    if (err != CL_SUCCESS) {
        return fail_submitted_graph(
            graph, event, "clFlush", err, reason, reason_capacity);
    }

    err = clWaitForEvents(1, &event);
    if (err != CL_SUCCESS) {
        return fail_submitted_graph(
            graph, event, "clWaitForEvents", err, reason, reason_capacity);
    }

    const cl_int release_err = clReleaseEvent(event);
    std::atomic_thread_fence(std::memory_order_acquire);
    if (release_err != CL_SUCCESS) {
        return fail_graph(
            graph, reason, reason_capacity,
            "clReleaseEvent failed (error %d)", release_err);
    }

    set_reason(reason, reason_capacity, "ok");
    return GGML_OPENCL_HTP_OK;
}

ggml_opencl_htp_result ggml_backend_opencl_htp_graph_finish(
    ggml_opencl_htp_graph * graph,
    char * reason,
    size_t reason_capacity) {
    if (graph == nullptr) {
        set_reason(reason, reason_capacity, "graph must not be null");
        return GGML_OPENCL_HTP_ERROR;
    }
    return finish_graph_state(
        graph, GGML_OPENCL_HTP_OK, "ok", reason, reason_capacity);
}

void ggml_backend_opencl_htp_graph_free(ggml_opencl_htp_graph ** graph) {
    if (graph == nullptr || *graph == nullptr) {
        return;
    }
    char ignored_reason[1] = {};
    ggml_backend_opencl_htp_graph_finish(*graph, ignored_reason, sizeof(ignored_reason));
    delete *graph;
    *graph = nullptr;
}

size_t ggml_backend_opencl_htp_graph_live_alias_count(const ggml_opencl_htp_graph * graph) {
    return graph != nullptr ? graph->aliases.size() : 0;
}

size_t ggml_backend_opencl_htp_graph_live_event_count(const ggml_opencl_htp_graph * graph) {
    return graph != nullptr ? graph->events.size() : 0;
}

void ggml_backend_opencl_htp_graph_test_fail_next_event_cache(ggml_opencl_htp_graph * graph) {
    if (graph != nullptr && graph->state == ggml_opencl_htp_graph_state::active) {
        graph->test_fail_next_event_cache = true;
    }
}
