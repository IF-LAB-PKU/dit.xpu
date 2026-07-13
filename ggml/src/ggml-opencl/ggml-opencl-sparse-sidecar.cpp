#define CL_TARGET_OPENCL_VERSION GGML_OPENCL_TARGET_VERSION

#include "ggml-opencl-htp.h"
#include "ggml-opencl-qcom.h"

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <new>
#include <string>
#include <vector>

#include <sys/stat.h>

struct ggml_opencl_htp_sidecar {
    cl_platform_id platform = nullptr;
    cl_device_id device = nullptr;
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    std::string platform_vendor;
    std::string device_name;
    std::string device_extensions;
    size_t ext_mem_padding = 0;
    size_t page_size = 0;
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
    ggml_opencl_htp_graph_state state = ggml_opencl_htp_graph_state::active;
    ggml_opencl_htp_result finish_result = GGML_OPENCL_HTP_OK;
    char finish_reason[256] = {};
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
    graph->state = ggml_opencl_htp_graph_state::finished;

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
