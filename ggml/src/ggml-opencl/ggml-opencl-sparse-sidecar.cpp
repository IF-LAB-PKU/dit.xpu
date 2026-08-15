#define CL_TARGET_OPENCL_VERSION GGML_OPENCL_TARGET_VERSION

#include "ggml-opencl-htp.h"
#include "ggml-opencl.h"
#include "ggml-opencl-qcom.h"
#include "ggml-impl.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <climits>
#include <cctype>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <thread>
#include <vector>

#include <sys/stat.h>

struct ggml_opencl_htp_cache_record {
    std::string key;
    uint64_t generation = 0;
    ggml_opencl_htp_cache_state state = GGML_OPENCL_HTP_CACHE_EMPTY;
    std::string failure_reason;
    std::string peer_key;
    cl_event producer_event = nullptr;
    cl_event profiling_dependency_event = nullptr;
    bool profiling_owner = false;
};

struct ggml_opencl_htp_persistent_import {
    std::string key;
    uint64_t generation = 0;
    ggml_opencl_htp_import_descriptor descriptor = {};
    cl_mem memory = nullptr;
};

enum class ggml_opencl_htp_block_sparse_variant {
    legacy32,
    pair64,
};

struct ggml_opencl_htp_block_sparse_kernel {
    int64_t head_dim = 0;
    ggml_opencl_htp_block_sparse_variant variant =
        ggml_opencl_htp_block_sparse_variant::legacy32;
    const char * variant_name = "legacy32";
    const char * kernel_name = "flash_attn_block_sparse_f32_f16";
    size_t work_group_size = 32;
    int lanes_per_query = 1;
    cl_program program = nullptr;
    cl_kernel kernel = nullptr;
};

struct ggml_opencl_htp_sidecar {
    cl_platform_id platform = nullptr;
    cl_device_id device = nullptr;
    cl_context context = nullptr;
    cl_command_queue sparse_queue = nullptr;
    cl_command_queue residual_queue = nullptr;
    ggml_opencl_htp_block_sparse_kernel block_sparse_kernels[3] = {
        { 64, ggml_opencl_htp_block_sparse_variant::legacy32,
          "legacy32", "flash_attn_block_sparse_f32_f16", 32, 1, nullptr, nullptr },
        { 128, ggml_opencl_htp_block_sparse_variant::legacy32,
          "legacy32", "flash_attn_block_sparse_f32_f16", 32, 1, nullptr, nullptr },
        { 128, ggml_opencl_htp_block_sparse_variant::pair64,
          "pair64", "flash_attn_block_sparse_d128_pair64_f32_f16", 64, 2, nullptr, nullptr },
    };
    cl_program residual_program = nullptr;
    cl_kernel rho_residual_kernel = nullptr;
    std::string platform_vendor;
    std::string device_name;
    std::string driver_version;
    std::string device_extensions;
    size_t ext_mem_padding = 0;
    size_t page_size = 0;
    cl_ulong max_mem_alloc_size = 0;
    bool sparse_queue_is_in_order = false;
    bool residual_queue_is_in_order = false;
    bool profiling_enabled = false;
    std::mutex queue_mutex;
    bool residual_queue_fallback_logged = false;
    std::mutex dispatch_mutex;
    mutable std::mutex cache_mutex;
    std::vector<ggml_opencl_htp_cache_record> cache_records;
    std::vector<ggml_opencl_htp_persistent_import> persistent_imports;
    bool test_fail_next_residual_wait = false;

    ~ggml_opencl_htp_sidecar() {
        for (ggml_opencl_htp_cache_record & record : cache_records) {
            if (record.producer_event != nullptr) {
                clReleaseEvent(record.producer_event);
                record.producer_event = nullptr;
            }
            if (record.profiling_dependency_event != nullptr) {
                clReleaseEvent(record.profiling_dependency_event);
                record.profiling_dependency_event = nullptr;
            }
        }
        for (ggml_opencl_htp_persistent_import & imported : persistent_imports) {
            if (imported.memory != nullptr) {
                clReleaseMemObject(imported.memory);
                imported.memory = nullptr;
            }
        }
        if (rho_residual_kernel != nullptr) {
            clReleaseKernel(rho_residual_kernel);
        }
        if (residual_program != nullptr) {
            clReleaseProgram(residual_program);
        }
        for (ggml_opencl_htp_block_sparse_kernel & variant : block_sparse_kernels) {
            if (variant.kernel != nullptr) {
                clReleaseKernel(variant.kernel);
            }
            if (variant.program != nullptr) {
                clReleaseProgram(variant.program);
            }
        }
        if (residual_queue != nullptr) {
            clReleaseCommandQueue(residual_queue);
        }
        if (sparse_queue != nullptr) {
            clReleaseCommandQueue(sparse_queue);
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
    size_t v_token_stride = 0;
    size_t v_head_stride = 0;
    size_t v_batch_stride = 0;
    int graph_index = -1;
};

struct ggml_opencl_htp_sparse_bindings {
    ggml_opencl_htp_tensor_binding q;
    ggml_opencl_htp_tensor_binding k;
    ggml_opencl_htp_tensor_binding v;
    ggml_opencl_htp_tensor_binding output;
    ggml_opencl_htp_tensor_binding mask;
};

struct ggml_opencl_htp_tensor_metadata {
    ggml_type type = GGML_TYPE_COUNT;
    ggml_op op = GGML_OP_COUNT;
    int64_t ne[GGML_MAX_DIMS] = {};
    size_t nb[GGML_MAX_DIMS] = {};
    const void * data = nullptr;
    ggml_backend_buffer_t backing = nullptr;
};

struct ggml_opencl_htp_prepared_sparse_dispatch {
    ggml_opencl_htp_sparse_dispatch dispatch;
    ggml_opencl_htp_sparse_bindings bindings;
    ggml_opencl_htp_tensor_metadata tensor_metadata[5];
    int32_t output_op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)] = {};
    cl_mem staged_mask = nullptr;
};

struct ggml_opencl_htp_prepared_residual_dispatch {
    ggml_opencl_htp_residual_dispatch dispatch = {};
    ggml_opencl_htp_tensor_binding dense;
    ggml_opencl_htp_tensor_binding sparse;
    ggml_opencl_htp_tensor_binding residual;
    ggml_opencl_htp_tensor_binding rho;
    ggml_opencl_htp_tensor_metadata tensor_metadata[4];
    std::string residual_cache_key;
    std::string rho_cache_key;
};

struct ggml_opencl_htp_pending_event {
    const ggml_tensor * producer = nullptr;
    cl_event event = nullptr;
    cl_mem staged_output = nullptr;
    ggml_opencl_htp_tensor_binding destination;
    int64_t attention_begin_us = 0;
    int64_t validation_us = 0;
    int64_t binding_us = 0;
    int64_t args_us = 0;
    int64_t enqueue_us = 0;
    int64_t flush_us = 0;
    bool record_attention_elapsed = false;
};

enum class ggml_opencl_htp_submission_state {
    queued,
    submitting,
    submitted,
    failed,
};

enum class ggml_opencl_htp_submission_job_kind {
    sparse_attention,
    exact_rho_residual,
};

struct ggml_opencl_htp_submission_job {
    ggml_opencl_htp_submission_job_kind kind =
        ggml_opencl_htp_submission_job_kind::sparse_attention;
    const ggml_tensor * producer = nullptr;
    int64_t published_us = 0;
};

struct ggml_opencl_htp_submission_record {
    ggml_opencl_htp_submission_job_kind kind =
        ggml_opencl_htp_submission_job_kind::sparse_attention;
    const ggml_tensor * producer = nullptr;
    ggml_opencl_htp_submission_state state = ggml_opencl_htp_submission_state::queued;
    ggml_opencl_htp_result result = GGML_OPENCL_HTP_OK;
    char reason[256] = {};
};

enum class ggml_opencl_htp_graph_state {
    active,
    finished,
};

struct ggml_opencl_htp_graph {
    ggml_opencl_htp_sidecar * sidecar = nullptr;
    std::vector<ggml_opencl_htp_alias> aliases;
    std::vector<cl_event> events;
    std::vector<ggml_opencl_htp_pending_event> pending_events;
    std::vector<const ggml_tensor *> routed_nodes;
    std::vector<const ggml_tensor *> residual_owned_sparse_producers;
    bool qkv_aliases_released = false;
    std::vector<ggml_opencl_htp_prepared_sparse_dispatch> prepared_sparse_dispatches;
    std::vector<ggml_opencl_htp_prepared_residual_dispatch> prepared_residual_dispatches;
    mutable std::mutex submission_mutex;
    std::condition_variable submission_cv;
    std::deque<ggml_opencl_htp_submission_job> sparse_submission_jobs;
    std::deque<ggml_opencl_htp_submission_job> residual_submission_jobs;
    std::vector<ggml_opencl_htp_submission_record> submission_records;
    std::thread submission_thread;
    bool submission_worker_active = false;
    bool submission_stop_requested = false;
    bool submission_cancel_requested = false;
    ggml_opencl_htp_result async_failure_result = GGML_OPENCL_HTP_OK;
    char async_failure_reason[256] = {};
    ggml_opencl_htp_graph_state state = ggml_opencl_htp_graph_state::active;
    ggml_opencl_htp_result finish_result = GGML_OPENCL_HTP_OK;
    char finish_reason[256] = {};
    bool cleanup_failed = false;
    bool test_fail_next_event_cache = false;
    bool test_fail_next_sparse_enqueue = false;
    bool test_fail_next_sparse_wait = false;
    bool test_fail_next_rho_enqueue = false;
    bool test_fail_next_residual_enqueue = false;
    bool use_residual_queue = false;
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

bool sidecar_trace_enabled() {
    const char * trace = std::getenv("SD_HTP_SPARSE_OPENCL_TRACE");
    return trace != nullptr && trace[0] != '\0' && std::strcmp(trace, "0") != 0;
}

void trace_sidecar_event(const char * event, const ggml_tensor * producer) {
    if (!sidecar_trace_enabled()) {
        return;
    }
    std::fprintf(
        stderr,
        "htp_opencl_event name=%s ts_us=%lld producer=%s\n",
        event,
        (long long) ggml_time_us(),
        producer != nullptr && producer->name[0] != '\0' ? producer->name : "<unnamed>");
}

void trace_sidecar_queue_event(
        const char * event,
        const char * queue,
        const ggml_tensor * producer) {
    if (!sidecar_trace_enabled()) {
        return;
    }
    std::fprintf(
        stderr,
        "htp_opencl_event name=%s queue=%s ts_us=%lld producer=%s\n",
        event,
        queue,
        (long long) ggml_time_us(),
            producer != nullptr && producer->name[0] != '\0' ?
            producer->name : "<unnamed>");
}

bool read_sparse_event_timestamps(
        cl_event event,
        ggml_opencl_attention_profile * profile) {
    if (event == nullptr || profile == nullptr) {
        return false;
    }
    cl_ulong queued_ns = 0;
    cl_ulong submit_ns = 0;
    cl_ulong start_ns = 0;
    cl_ulong end_ns = 0;
    if (clGetEventProfilingInfo(
            event, CL_PROFILING_COMMAND_QUEUED,
            sizeof(queued_ns), &queued_ns, nullptr) != CL_SUCCESS ||
        clGetEventProfilingInfo(
            event, CL_PROFILING_COMMAND_SUBMIT,
            sizeof(submit_ns), &submit_ns, nullptr) != CL_SUCCESS ||
        clGetEventProfilingInfo(
            event, CL_PROFILING_COMMAND_START,
            sizeof(start_ns), &start_ns, nullptr) != CL_SUCCESS ||
        clGetEventProfilingInfo(
            event, CL_PROFILING_COMMAND_END,
            sizeof(end_ns), &end_ns, nullptr) != CL_SUCCESS ||
        queued_ns > submit_ns || submit_ns > start_ns || start_ns > end_ns) {
        return false;
    }
    profile->event_queued_ns = queued_ns;
    profile->event_submit_ns = submit_ns;
    profile->event_start_ns = start_ns;
    profile->event_end_ns = end_ns;
    return true;
}

void trace_sparse_event_timing(
        const ggml_opencl_htp_sidecar * sidecar,
        cl_event event,
        const ggml_tensor * producer) {
    if (sidecar == nullptr || !sidecar->profiling_enabled || event == nullptr) {
        return;
    }
    ggml_opencl_attention_profile profile = {};
    if (!read_sparse_event_timestamps(event, &profile)) {
        return;
    }
    std::fprintf(
        stderr,
        "htp_opencl_sparse_timing producer=%s gpu_queued_ns=%llu "
        "gpu_submit_ns=%llu gpu_start_ns=%llu gpu_end_ns=%llu gpu_us=%.3f\n",
        producer != nullptr && producer->name[0] != '\0' ?
            producer->name : "<unnamed>",
        static_cast<unsigned long long>(profile.event_queued_ns),
        static_cast<unsigned long long>(profile.event_submit_ns),
        static_cast<unsigned long long>(profile.event_start_ns),
        static_cast<unsigned long long>(profile.event_end_ns),
        static_cast<double>(profile.event_end_ns - profile.event_start_ns) / 1000.0);
}

void trace_submission_scheduler_timing(
        const ggml_opencl_htp_submission_job & job,
        int64_t selected_us) {
    if (!sidecar_trace_enabled()) {
        return;
    }
    std::fprintf(
        stderr,
        "htp_opencl_submission_scheduler_timing kind=%s "
        "publication_to_selection_us=%lld producer=%s\n",
        job.kind == ggml_opencl_htp_submission_job_kind::sparse_attention ?
            "sparse" : "residual",
        (long long) std::max<int64_t>(0, selected_us - job.published_us),
        job.producer != nullptr && job.producer->name[0] != '\0' ?
            job.producer->name : "<unnamed>");
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

std::string profiling_token(std::string value) {
    if (value.empty()) {
        return "unknown";
    }
    for (char & character : value) {
        if (std::isspace(static_cast<unsigned char>(character)) || character == '=') {
            character = '_';
        }
    }
    return value;
}

struct subgroup_resource_info {
    size_t max_size = 0;
    size_t count = 0;
    bool max_size_known = false;
    bool count_known = false;
};

subgroup_resource_info query_subgroup_resources(
        cl_kernel kernel,
        cl_device_id device,
        size_t work_group_size) {
    subgroup_resource_info result;
#if defined(CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE) && \
    defined(CL_KERNEL_SUB_GROUP_COUNT_FOR_NDRANGE)
    const size_t local_work_size[] = { work_group_size, 1, 1 };
    result.max_size_known = clGetKernelSubGroupInfo(
        kernel, device, CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE,
        sizeof(local_work_size), local_work_size,
        sizeof(result.max_size), &result.max_size, nullptr) == CL_SUCCESS;
    result.count_known = clGetKernelSubGroupInfo(
        kernel, device, CL_KERNEL_SUB_GROUP_COUNT_FOR_NDRANGE,
        sizeof(local_work_size), local_work_size,
        sizeof(result.count), &result.count, nullptr) == CL_SUCCESS;
#else
    (void) kernel;
    (void) device;
    (void) work_group_size;
#endif
    return result;
}

void trace_block_sparse_kernel_resources(
        const ggml_opencl_htp_sidecar * sidecar,
        const ggml_opencl_htp_block_sparse_kernel & variant,
        size_t work_group_size) {
    if (sidecar == nullptr || !sidecar->profiling_enabled || variant.kernel == nullptr) {
        return;
    }

    size_t preferred_multiple = 0;
    size_t kernel_wg_max = 0;
    size_t compile_wg[3] = {};
    cl_ulong kernel_local_mem = 0;
    cl_ulong kernel_private_mem = 0;
    cl_ulong device_local_mem = 0;
    cl_uint compute_units = 0;
    clGetKernelWorkGroupInfo(
        variant.kernel, sidecar->device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
        sizeof(preferred_multiple), &preferred_multiple, nullptr);
    clGetKernelWorkGroupInfo(
        variant.kernel, sidecar->device, CL_KERNEL_WORK_GROUP_SIZE,
        sizeof(kernel_wg_max), &kernel_wg_max, nullptr);
    clGetKernelWorkGroupInfo(
        variant.kernel, sidecar->device, CL_KERNEL_COMPILE_WORK_GROUP_SIZE,
        sizeof(compile_wg), compile_wg, nullptr);
    clGetKernelWorkGroupInfo(
        variant.kernel, sidecar->device, CL_KERNEL_LOCAL_MEM_SIZE,
        sizeof(kernel_local_mem), &kernel_local_mem, nullptr);
    clGetKernelWorkGroupInfo(
        variant.kernel, sidecar->device, CL_KERNEL_PRIVATE_MEM_SIZE,
        sizeof(kernel_private_mem), &kernel_private_mem, nullptr);
    clGetDeviceInfo(
        sidecar->device, CL_DEVICE_LOCAL_MEM_SIZE,
        sizeof(device_local_mem), &device_local_mem, nullptr);
    clGetDeviceInfo(
        sidecar->device, CL_DEVICE_MAX_COMPUTE_UNITS,
        sizeof(compute_units), &compute_units, nullptr);

    std::string driver;
    std::string compiler;
    get_device_string(sidecar->device, CL_DRIVER_VERSION, &driver);
    get_device_string(sidecar->device, CL_DEVICE_OPENCL_C_VERSION, &compiler);
    const subgroup_resource_info subgroup32 =
        query_subgroup_resources(variant.kernel, sidecar->device, 32);
    const subgroup_resource_info subgroup64 =
        query_subgroup_resources(variant.kernel, sidecar->device, 64);
    const subgroup_resource_info subgroup128 =
        query_subgroup_resources(variant.kernel, sidecar->device, 128);
    const subgroup_resource_info & selected_subgroup = work_group_size == 128
        ? subgroup128 : work_group_size == 64 ? subgroup64 : subgroup32;

    std::fprintf(
        stderr,
        "[OCL_ATTN_RESOURCE] variant=%s device=%s driver=%s compiler=%s "
        "d=%lld wg=%zu lanes_per_query=%d preferred_wg_multiple=%zu "
        "kernel_wg_max=%zu compile_wg=%zu,%zu,%zu local_mem_bytes=%llu "
        "private_mem_bytes=%llu device_local_mem_bytes=%llu compute_units=%u "
        "max_subgroup_size=%zu subgroup_count=%zu subgroup_known=%d "
        "sg32_max=%zu sg32_count=%zu sg32_known=%d "
        "sg64_max=%zu sg64_count=%zu sg64_known=%d "
        "sg128_max=%zu sg128_count=%zu sg128_known=%d "
        "ext_khr_subgroups=%d ext_shuffle=%d ext_shuffle_relative=%d ext_qcom_reqd=%d\n",
        variant.variant_name,
        profiling_token(sidecar->device_name).c_str(),
        profiling_token(driver).c_str(), profiling_token(compiler).c_str(),
        static_cast<long long>(variant.head_dim), work_group_size,
        variant.lanes_per_query,
        preferred_multiple, kernel_wg_max,
        compile_wg[0], compile_wg[1], compile_wg[2],
        static_cast<unsigned long long>(kernel_local_mem),
        static_cast<unsigned long long>(kernel_private_mem),
        static_cast<unsigned long long>(device_local_mem), compute_units,
        selected_subgroup.max_size, selected_subgroup.count,
        selected_subgroup.max_size_known && selected_subgroup.count_known ? 1 : 0,
        subgroup32.max_size, subgroup32.count,
        subgroup32.max_size_known && subgroup32.count_known ? 1 : 0,
        subgroup64.max_size, subgroup64.count,
        subgroup64.max_size_known && subgroup64.count_known ? 1 : 0,
        subgroup128.max_size, subgroup128.count,
        subgroup128.max_size_known && subgroup128.count_known ? 1 : 0,
        has_extension_token(sidecar->device_extensions, "cl_khr_subgroups") ? 1 : 0,
        has_extension_token(sidecar->device_extensions, "cl_khr_subgroup_shuffle") ? 1 : 0,
        has_extension_token(sidecar->device_extensions, "cl_khr_subgroup_shuffle_relative") ? 1 : 0,
        has_extension_token(sidecar->device_extensions, "cl_qcom_reqd_sub_group_size") ? 1 : 0);
}

ggml_opencl_htp_result select_block_sparse_kernel(
        ggml_opencl_htp_sidecar * sidecar,
        int64_t head_dim,
        int mask_format,
        ggml_opencl_htp_block_sparse_kernel ** selected,
        char * reason,
        size_t reason_capacity) {
    if (selected == nullptr) {
        set_reason(reason, reason_capacity, "block-sparse kernel output must not be null");
        return GGML_OPENCL_HTP_ERROR;
    }
    *selected = nullptr;

    ggml_opencl_htp_block_sparse_variant requested =
        ggml_opencl_htp_block_sparse_variant::legacy32;
    if (head_dim == 128) {
        const bool pair64_eligible =
            mask_format == GGML_FLASH_ATTN_EXT_MASK_BLOCK_INDEX &&
            has_extension_token(
                sidecar->device_extensions,
                "cl_khr_subgroup_shuffle_relative");
        requested = pair64_eligible ?
            ggml_opencl_htp_block_sparse_variant::pair64 :
            ggml_opencl_htp_block_sparse_variant::legacy32;
    }

    if (requested == ggml_opencl_htp_block_sparse_variant::pair64) {
        if (mask_format != GGML_FLASH_ATTN_EXT_MASK_BLOCK_INDEX) {
            set_reason(
                reason, reason_capacity,
                "pair64 requires BLOCK_INDEX mask format");
            return GGML_OPENCL_HTP_UNSUPPORTED;
        }
        if (!has_extension_token(
                sidecar->device_extensions,
                "cl_khr_subgroup_shuffle_relative")) {
            set_reason(
                reason, reason_capacity,
                "pair64 requires cl_khr_subgroup_shuffle_relative");
            return GGML_OPENCL_HTP_UNSUPPORTED;
        }
    }

    for (ggml_opencl_htp_block_sparse_kernel & candidate :
         sidecar->block_sparse_kernels) {
        if (candidate.head_dim == head_dim && candidate.variant == requested) {
            *selected = &candidate;
            set_reason(reason, reason_capacity, "ok");
            return GGML_OPENCL_HTP_OK;
        }
    }

    if (requested == ggml_opencl_htp_block_sparse_variant::legacy32) {
        set_reason(
            reason, reason_capacity,
            "unsupported block-sparse head dimension %lld",
            (long long) head_dim);
    } else {
        set_reason(
            reason, reason_capacity,
            "pair64 is unavailable for block-sparse head dimension %lld",
            (long long) head_dim);
    }
    return GGML_OPENCL_HTP_UNSUPPORTED;
}

ggml_opencl_htp_result ensure_block_sparse_kernel(
        ggml_opencl_htp_sidecar * sidecar,
        ggml_opencl_htp_block_sparse_kernel * variant,
        char * reason,
        size_t reason_capacity) {
    if (variant == nullptr) {
        set_reason(reason, reason_capacity, "block-sparse kernel variant must not be null");
        return GGML_OPENCL_HTP_ERROR;
    }
    if (variant->kernel != nullptr) {
        return GGML_OPENCL_HTP_OK;
    }

#ifdef GGML_OPENCL_EMBED_KERNELS
    const std::string legacy_kernel_source {
#include "flash_attn_block_sparse_f32_f16.cl.h"
    };
    const std::string pair64_kernel_source {
#include "flash_attn_block_sparse_d128_f32_f16.cl.h"
    };
    const std::string & kernel_source =
        variant->variant == ggml_opencl_htp_block_sparse_variant::legacy32
            ? legacy_kernel_source : pair64_kernel_source;
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

    char build_options[320] = {};
    if (variant->variant == ggml_opencl_htp_block_sparse_variant::pair64) {
        std::snprintf(
            build_options, sizeof(build_options),
            "-cl-std=CL3.0 -cl-mad-enable "
            "-cl-unsafe-math-optimizations -cl-finite-math-only "
            "-cl-fast-relaxed-math");
    } else {
        std::snprintf(
            build_options, sizeof(build_options),
            "-cl-std=CL3.0 -cl-mad-enable "
            "-cl-unsafe-math-optimizations -cl-finite-math-only "
            "-cl-fast-relaxed-math -DBLOCK_M=32 -DBLOCK_N=32 "
            "-DHEAD_DIM=%lld -DVEC_COUNT=%lld",
            (long long) variant->head_dim,
            (long long) (variant->head_dim / 4));
    }
    err = clBuildProgram(program, 1, &sidecar->device, build_options, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        const std::string build_log = get_program_build_log(program, sidecar->device);
        const cl_int release_err = clReleaseProgram(program);
        set_reason(
            reason, reason_capacity,
            "clBuildProgram failed for %s (error %d, release error %d): %s",
            variant->variant_name, err, release_err, build_log.c_str());
        return GGML_OPENCL_HTP_ERROR;
    }

    cl_kernel kernel = clCreateKernel(program, variant->kernel_name, &err);
    if (err != CL_SUCCESS || kernel == nullptr) {
        const cl_int release_err = clReleaseProgram(program);
        set_reason(
            reason, reason_capacity,
            "clCreateKernel failed for %s (error %d, program release error %d)",
            variant->variant_name, err, release_err);
        return GGML_OPENCL_HTP_ERROR;
    }

    if (variant->variant == ggml_opencl_htp_block_sparse_variant::pair64) {
        const subgroup_resource_info subgroup =
            query_subgroup_resources(kernel, sidecar->device, variant->work_group_size);
        const size_t required_count = 1;
        const bool wrong_size = subgroup.max_size_known && subgroup.max_size != 64;
        const bool wrong_count = subgroup.count_known && subgroup.count != required_count;
        if (wrong_size || wrong_count) {
            const cl_int kernel_release_err = clReleaseKernel(kernel);
            const cl_int program_release_err = clReleaseProgram(program);
            set_reason(
                reason, reason_capacity,
                "%s requires %zu 64-lane subgroup(s) (max=%zu count=%zu known=%d/%d, "
                "kernel release error %d, program release error %d)",
                variant->variant_name, required_count,
                subgroup.max_size, subgroup.count,
                subgroup.max_size_known ? 1 : 0, subgroup.count_known ? 1 : 0,
                kernel_release_err, program_release_err);
            return GGML_OPENCL_HTP_UNSUPPORTED;
        }
    }

    variant->program = program;
    variant->kernel = kernel;
    trace_block_sparse_kernel_resources(sidecar, *variant, variant->work_group_size);
    set_reason(reason, reason_capacity, "ok");
    return GGML_OPENCL_HTP_OK;
#endif
}

ggml_opencl_htp_result ensure_residual_kernel(
        ggml_opencl_htp_sidecar * sidecar,
        char * reason,
        size_t reason_capacity) {
    if (sidecar->rho_residual_kernel != nullptr) {
        return GGML_OPENCL_HTP_OK;
    }

    static constexpr const char * kernel_source = R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void exact_rho_residual_f16(
        __global const float * dense,
        ulong dense_offset_bytes,
        __global const float * sparse,
        ulong sparse_offset_bytes,
        __global float * rho,
        ulong rho_offset_bytes,
        __global half * residual,
        ulong residual_offset_bytes,
        ulong head_dim,
        ulong row_count) {
    const ulong row = (ulong) get_group_id(0);
    const ulong lane = (ulong) get_local_id(0);
    if (row >= row_count) {
        return;
    }
    const ulong packed_dim = head_dim + 1;
    const ulong dense_base = dense_offset_bytes / sizeof(float);
    const ulong sparse_base = sparse_offset_bytes / sizeof(float);
    const ulong rho_base = rho_offset_bytes / sizeof(float);
    const ulong residual_base = residual_offset_bytes / sizeof(half);
    __local float row_rho[1];
    if (lane == 0) {
        const float value = exp(
            sparse[sparse_base + row * packed_dim + head_dim] -
            dense[dense_base + row * packed_dim + head_dim]);
        row_rho[0] = value;
        rho[rho_base + row] = value;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (ulong d = lane; d < head_dim; d += (ulong) get_local_size(0)) {
        const float value = dense[dense_base + row * packed_dim + d] -
            sparse[sparse_base + row * packed_dim + d] * row_rho[0];
        residual[residual_base + row * head_dim + d] = convert_half_rte(value);
    }
}
)CLC";

    const char * source = kernel_source;
    cl_int err = CL_SUCCESS;
    cl_program program = clCreateProgramWithSource(
        sidecar->context, 1, &source, nullptr, &err);
    if (err != CL_SUCCESS || program == nullptr) {
        set_reason(reason, reason_capacity, "clCreateProgramWithSource for residual failed (error %d)", err);
        return GGML_OPENCL_HTP_ERROR;
    }
    static constexpr const char * build_options = "-cl-std=CL3.0 -cl-mad-enable";
    err = clBuildProgram(program, 1, &sidecar->device, build_options, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        const std::string build_log = get_program_build_log(program, sidecar->device);
        const cl_int release_err = clReleaseProgram(program);
        set_reason(
            reason, reason_capacity,
            "residual clBuildProgram failed (error %d, release error %d): %s",
            err, release_err, build_log.c_str());
        return GGML_OPENCL_HTP_ERROR;
    }
    cl_kernel kernel = clCreateKernel(program, "exact_rho_residual_f16", &err);
    if (err != CL_SUCCESS || kernel == nullptr) {
        const cl_int release_err = clReleaseProgram(program);
        set_reason(
            reason, reason_capacity,
            "clCreateKernel exact_rho_residual_f16 failed (error %d, release error %d)",
            err, release_err);
        return GGML_OPENCL_HTP_ERROR;
    }
    sidecar->residual_program = program;
    sidecar->rho_residual_kernel = kernel;
    set_reason(reason, reason_capacity, "ok");
    return GGML_OPENCL_HTP_OK;
}

ggml_opencl_htp_result ensure_residual_queue(
        ggml_opencl_htp_sidecar * sidecar,
        char * reason,
        size_t reason_capacity) {
    std::lock_guard<std::mutex> lock(sidecar->queue_mutex);
    if (sidecar->residual_queue != nullptr &&
        sidecar->residual_queue_is_in_order) {
        set_reason(reason, reason_capacity, "ok");
        return GGML_OPENCL_HTP_OK;
    }
    const char * fail_create = std::getenv(
        "SD_HTP_SPARSE_OPENCL_TEST_FAIL_RESIDUAL_QUEUE_CREATE");
    if (fail_create != nullptr && fail_create[0] != '\0' &&
        std::strcmp(fail_create, "0") != 0) {
        set_reason(
            reason, reason_capacity,
            "injected residual queue creation failure");
        return GGML_OPENCL_HTP_ERROR;
    }

    const cl_queue_properties queue_properties[] = {
        CL_QUEUE_PROPERTIES,
        static_cast<cl_queue_properties>(
            sidecar->profiling_enabled ? CL_QUEUE_PROFILING_ENABLE : 0),
        0,
    };
    cl_int err = CL_SUCCESS;
    cl_command_queue queue = clCreateCommandQueueWithProperties(
        sidecar->context, sidecar->device, queue_properties, &err);
    if (err != CL_SUCCESS || queue == nullptr) {
        set_reason(
            reason, reason_capacity,
            "residual clCreateCommandQueueWithProperties failed (error %d)", err);
        return GGML_OPENCL_HTP_ERROR;
    }

    cl_command_queue_properties actual_properties = 0;
    err = clGetCommandQueueInfo(
        queue, CL_QUEUE_PROPERTIES,
        sizeof(actual_properties), &actual_properties, nullptr);
    if (err != CL_SUCCESS) {
        const cl_int release_err = clReleaseCommandQueue(queue);
        set_reason(
            reason, reason_capacity,
            "failed to query residual OpenCL queue properties "
            "(error %d, release error %d)",
            err, release_err);
        return GGML_OPENCL_HTP_ERROR;
    }
    if ((actual_properties & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) != 0) {
        const cl_int release_err = clReleaseCommandQueue(queue);
        set_reason(
            reason, reason_capacity,
            "OpenCL residual command queue is not in-order "
            "(release error %d)", release_err);
        return GGML_OPENCL_HTP_ERROR;
    }

    sidecar->residual_queue = queue;
    sidecar->residual_queue_is_in_order = true;
    set_reason(reason, reason_capacity, "ok");
    return GGML_OPENCL_HTP_OK;
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
    if (!get_device_string(sidecar->device, CL_DRIVER_VERSION, &sidecar->driver_version)) {
        set_reason(reason, reason_capacity, "failed to query OpenCL driver version");
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

    const char * profiling = std::getenv("SD_HTP_SPARSE_OPENCL_PROFILE");
    sidecar->profiling_enabled =
        profiling != nullptr && profiling[0] != '\0' && std::strcmp(profiling, "0") != 0;
    const cl_queue_properties queue_properties[] = {
        CL_QUEUE_PROPERTIES,
        static_cast<cl_queue_properties>(
            sidecar->profiling_enabled ? CL_QUEUE_PROFILING_ENABLE : 0),
        0,
    };
    sidecar->sparse_queue = clCreateCommandQueueWithProperties(
        sidecar->context, sidecar->device, queue_properties, &err);
    if (err != CL_SUCCESS || sidecar->sparse_queue == nullptr) {
        set_reason(reason, reason_capacity,
                   "sparse clCreateCommandQueueWithProperties failed (error %d)", err);
        return nullptr;
    }

    cl_command_queue_properties actual_queue_properties = 0;
    err = clGetCommandQueueInfo(sidecar->sparse_queue, CL_QUEUE_PROPERTIES,
                                sizeof(actual_queue_properties), &actual_queue_properties, nullptr);
    if (err != CL_SUCCESS) {
        set_reason(reason, reason_capacity, "failed to query OpenCL queue properties (error %d)", err);
        return nullptr;
    }
    sidecar->sparse_queue_is_in_order =
        (actual_queue_properties & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) == 0;
    if (!sidecar->sparse_queue_is_in_order) {
        set_reason(reason, reason_capacity, "OpenCL sparse command queue is not in-order");
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
    return sidecar != nullptr && sidecar->sparse_queue_is_in_order &&
           (sidecar->residual_queue == nullptr ||
            sidecar->residual_queue_is_in_order);
}

size_t ggml_backend_opencl_htp_sidecar_live_persistent_import_count(
        const ggml_opencl_htp_sidecar * sidecar) {
    if (sidecar == nullptr) {
        return 0;
    }
    std::lock_guard<std::mutex> lock(sidecar->cache_mutex);
    return sidecar->persistent_imports.size();
}

size_t ggml_backend_opencl_htp_sidecar_live_cache_event_count(
        const ggml_opencl_htp_sidecar * sidecar) {
    if (sidecar == nullptr) {
        return 0;
    }
    std::lock_guard<std::mutex> lock(sidecar->cache_mutex);
    size_t count = 0;
    for (const ggml_opencl_htp_cache_record & record : sidecar->cache_records) {
        count += record.producer_event != nullptr;
        count += record.profiling_dependency_event != nullptr;
    }
    return count;
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

ggml_opencl_htp_persistent_import * find_persistent_import(
        ggml_opencl_htp_sidecar * sidecar,
        ggml_backend_buffer_t backing) {
    if (sidecar == nullptr || backing == nullptr) {
        return nullptr;
    }
    for (ggml_opencl_htp_persistent_import & imported : sidecar->persistent_imports) {
        if (imported.descriptor.backing == backing) {
            return &imported;
        }
    }
    return nullptr;
}

ggml_opencl_htp_persistent_import * find_persistent_import(
        ggml_opencl_htp_sidecar * sidecar,
        const char * key,
        uint64_t generation) {
    if (sidecar == nullptr || key == nullptr) {
        return nullptr;
    }
    for (ggml_opencl_htp_persistent_import & imported : sidecar->persistent_imports) {
        if (imported.generation == generation && imported.key == key) {
            return &imported;
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
    size_t reason_capacity,
    bool finish_queue = true) {
    std::lock_guard<std::mutex> submission_lock(graph->submission_mutex);
    GGML_ASSERT(!graph->submission_worker_active);
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

    if (finish_queue) {
        record_error("clFinish(sparse)", clFinish(graph->sidecar->sparse_queue));
        if (graph->use_residual_queue) {
            record_error(
                "clFinish(residual)", clFinish(graph->sidecar->residual_queue));
        }
    }
    for (cl_event event : graph->events) {
        if (event != nullptr) {
            record_error("clReleaseEvent", clReleaseEvent(event));
        }
    }
    graph->events.clear();
    for (ggml_opencl_htp_pending_event & pending : graph->pending_events) {
        if (pending.event != nullptr) {
            record_error("clReleaseEvent", clReleaseEvent(pending.event));
            pending.event = nullptr;
        }
        if (pending.staged_output != nullptr) {
            record_error("clReleaseMemObject", clReleaseMemObject(pending.staged_output));
            pending.staged_output = nullptr;
        }
    }
    graph->pending_events.clear();
    for (ggml_opencl_htp_alias & alias : graph->aliases) {
        if (alias.memory != nullptr) {
            record_error("clReleaseMemObject", clReleaseMemObject(alias.memory));
            alias.memory = nullptr;
        }
    }
    graph->aliases.clear();
    graph->routed_nodes.clear();
    graph->residual_owned_sparse_producers.clear();
    for (ggml_opencl_htp_prepared_sparse_dispatch & prepared :
            graph->prepared_sparse_dispatches) {
        if (prepared.staged_mask != nullptr) {
            record_error("clReleaseMemObject", clReleaseMemObject(prepared.staged_mask));
            prepared.staged_mask = nullptr;
        }
    }
    graph->prepared_sparse_dispatches.clear();
    graph->prepared_residual_dispatches.clear();
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

ggml_opencl_htp_result fail_runtime_graph(
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

    {
        std::lock_guard<std::mutex> lock(graph->submission_mutex);
        if (graph->submission_worker_active) {
            if (graph->async_failure_result == GGML_OPENCL_HTP_OK) {
                graph->async_failure_result = GGML_OPENCL_HTP_ERROR;
                std::snprintf(
                    graph->async_failure_reason, sizeof(graph->async_failure_reason),
                    "%s", failure_reason);
            }
            graph->submission_stop_requested = true;
            graph->submission_cancel_requested = true;
            for (ggml_opencl_htp_submission_record & queued : graph->submission_records) {
                if (queued.state != ggml_opencl_htp_submission_state::queued) {
                    continue;
                }
                queued.result = GGML_OPENCL_HTP_ERROR;
                queued.state = ggml_opencl_htp_submission_state::failed;
                std::snprintf(
                    queued.reason, sizeof(queued.reason),
                    "submission cancelled after graph failure: %s", failure_reason);
            }
            graph->sparse_submission_jobs.clear();
            graph->residual_submission_jobs.clear();
            set_reason(reason, reason_capacity, "%s", failure_reason);
            graph->submission_cv.notify_all();
            return GGML_OPENCL_HTP_ERROR;
        }
    }

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
    const cl_int finish_err = clFinish(graph->sidecar->sparse_queue);
    const cl_int release_err = event != nullptr ? clReleaseEvent(event) : CL_SUCCESS;
    return fail_runtime_graph(
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
    const int64_t head_dim = q->ne[0];
    if ((head_dim != 64 && head_dim != 128) ||
        k->ne[0] != head_dim || v->ne[0] != head_dim ||
        node->ne[0] != head_dim + 1) {
        return unsupported_sparse_node(
            reason, reason_capacity,
            "head dimensions must be matching Q/K/V=64 or 128 and output=head_dim+1");
    }
    if (kv_heads > heads || heads % kv_heads != 0) {
        return unsupported_sparse_node(
            reason, reason_capacity, "query heads must be divisible by KV heads");
    }
    if (heads > INT_MAX / batch) {
        return unsupported_sparse_node(
            reason, reason_capacity, "head and batch product exceeds cl_int range");
    }
    const bool v_token_head = v->ne[1] == lk && v->ne[2] == kv_heads;
    const bool v_head_token = v->ne[1] == kv_heads && v->ne[2] == lk;
    if (k->ne[3] != batch || v->ne[3] != batch || v_token_head == v_head_token) {
        return unsupported_sparse_node(
            reason, reason_capacity,
            "V must have exactly one legal layout [D,Lk,Hkv,B] or [D,Hkv,Lk,B] "
            "and K/V batch dimensions must match Q");
    }
    if (node->ne[1] != heads || node->ne[2] != lq || node->ne[3] != batch) {
        return unsupported_sparse_node(
            reason, reason_capacity, "packed output shape must be [head_dim+1,H,Lq,B]");
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
    if (!is_dense_scalar_permutation(q, sizeof(float)) ||
        !is_dense_scalar_permutation(k, sizeof(ggml_fp16_t)) ||
        !is_dense_scalar_permutation(v, sizeof(ggml_fp16_t)) ||
        !ggml_is_contiguous(mask) || !ggml_is_contiguous(node)) {
        return unsupported_sparse_node(
            reason, reason_capacity,
            "Q, K, and V must be dense scalar permutations; mask and output must be contiguous");
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
    dispatch->v_token_stride = v_token_head ? v->nb[1] : v->nb[2];
    dispatch->v_head_stride = v_token_head ? v->nb[2] : v->nb[1];
    dispatch->v_batch_stride = v->nb[3];
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

ggml_opencl_htp_tensor_metadata capture_tensor_metadata(const ggml_tensor * tensor) {
    ggml_opencl_htp_tensor_metadata metadata;
    metadata.type = tensor->type;
    metadata.op = tensor->op;
    std::memcpy(metadata.ne, tensor->ne, sizeof(metadata.ne));
    std::memcpy(metadata.nb, tensor->nb, sizeof(metadata.nb));
    metadata.data = tensor->data;
    metadata.backing = tensor_backing(tensor);
    return metadata;
}

bool tensor_metadata_matches(
        const ggml_tensor * tensor,
        const ggml_opencl_htp_tensor_metadata & metadata) {
    return tensor != nullptr &&
           tensor->type == metadata.type &&
           tensor->op == metadata.op &&
           std::memcmp(tensor->ne, metadata.ne, sizeof(metadata.ne)) == 0 &&
           std::memcmp(tensor->nb, metadata.nb, sizeof(metadata.nb)) == 0 &&
           tensor->data == metadata.data &&
           tensor_backing(tensor) == metadata.backing;
}

ggml_opencl_htp_cache_record * find_cache_record(
        ggml_opencl_htp_sidecar * sidecar,
        const char * key,
        uint64_t generation) {
    if (sidecar == nullptr || key == nullptr) {
        return nullptr;
    }
    for (ggml_opencl_htp_cache_record & record : sidecar->cache_records) {
        if (record.generation == generation && record.key == key) {
            return &record;
        }
    }
    return nullptr;
}

const ggml_opencl_htp_cache_record * find_cache_record(
        const ggml_opencl_htp_sidecar * sidecar,
        const char * key,
        uint64_t generation) {
    return find_cache_record(
        const_cast<ggml_opencl_htp_sidecar *>(sidecar), key, generation);
}

bool same_tensor_shape(const ggml_tensor * lhs, const ggml_tensor * rhs) {
    return lhs != nullptr && rhs != nullptr &&
           std::memcmp(lhs->ne, rhs->ne, sizeof(lhs->ne)) == 0;
}

const ggml_tensor * producer_through_views(const ggml_tensor * tensor) {
    for (size_t depth = 0; tensor != nullptr && depth < 64; ++depth) {
        const bool view_only = tensor->op == GGML_OP_VIEW ||
                               tensor->op == GGML_OP_RESHAPE ||
                               tensor->op == GGML_OP_PERMUTE ||
                               tensor->op == GGML_OP_TRANSPOSE;
        if (!view_only) {
            return tensor;
        }
        tensor = tensor->src[0] != nullptr ? tensor->src[0] : tensor->view_src;
    }
    return nullptr;
}

ggml_opencl_htp_result validate_residual_dispatch(
        ggml_opencl_htp_graph * graph,
        const ggml_opencl_htp_residual_dispatch * dispatch,
        ggml_opencl_htp_prepared_residual_dispatch * prepared,
        char * reason,
        size_t reason_capacity) {
    if (graph == nullptr || dispatch == nullptr || prepared == nullptr) {
        set_reason(reason, reason_capacity, "graph, residual dispatch, and prepared output must not be null");
        return GGML_OPENCL_HTP_ERROR;
    }
    struct required_tensor {
        const char * name;
        const ggml_tensor * tensor;
    };
    const required_tensor required[] = {
        { "dense_packed", dispatch->dense_packed },
        { "sparse_packed_producer", dispatch->sparse_packed_producer },
        { "residual_destination", dispatch->residual_destination },
        { "rho_destination", dispatch->rho_destination },
    };
    for (const required_tensor & item : required) {
        if (item.tensor == nullptr) {
            return unsupported_sparse_node(
                reason, reason_capacity, "%s must not be null", item.name);
        }
        const bool expected_f32 = std::strcmp(item.name, "residual_destination") != 0;
        const ggml_type expected_type = expected_f32 ? GGML_TYPE_F32 : GGML_TYPE_F16;
        if (item.tensor->type != expected_type) {
            return unsupported_sparse_node(
                reason, reason_capacity, "%s must use %s storage", item.name,
                expected_f32 ? "F32" : "F16");
        }
        if (!ggml_is_contiguous(item.tensor)) {
            return unsupported_sparse_node(
                reason, reason_capacity, "%s must be contiguous", item.name);
        }
    }
    if (dispatch->residual_cache_key == nullptr || dispatch->residual_cache_key[0] == '\0' ||
        dispatch->rho_cache_key == nullptr || dispatch->rho_cache_key[0] == '\0') {
        return unsupported_sparse_node(
            reason, reason_capacity, "residual and rho cache keys must not be empty");
    }
    if (std::strcmp(dispatch->residual_cache_key, dispatch->rho_cache_key) == 0) {
        return unsupported_sparse_node(
            reason, reason_capacity, "residual and rho cache keys must be distinct");
    }
    if (!same_tensor_shape(dispatch->dense_packed, dispatch->sparse_packed_producer) ||
        dispatch->dense_packed->ne[0] < 2) {
        return unsupported_sparse_node(
            reason, reason_capacity, "dense and sparse packed shape must match and include O plus LSE");
    }

    const int64_t head_dim = dispatch->dense_packed->ne[0] - 1;
    const ggml_tensor * residual = dispatch->residual_destination;
    const ggml_tensor * rho = dispatch->rho_destination;
    if (residual->ne[0] != head_dim ||
        residual->ne[1] != dispatch->dense_packed->ne[1] ||
        residual->ne[2] != dispatch->dense_packed->ne[2] ||
        residual->ne[3] != dispatch->dense_packed->ne[3]) {
        return unsupported_sparse_node(
            reason, reason_capacity, "residual destination shape does not match packed O shape");
    }
    if (rho->ne[0] != 1 ||
        rho->ne[1] != dispatch->dense_packed->ne[1] ||
        rho->ne[2] != dispatch->dense_packed->ne[2] ||
        rho->ne[3] != dispatch->dense_packed->ne[3]) {
        return unsupported_sparse_node(
            reason, reason_capacity, "rho destination shape does not match packed LSE shape");
    }

    struct binding_request {
        const char * name;
        const ggml_tensor * tensor;
        ggml_opencl_htp_tensor_binding * binding;
    };
    const binding_request bindings[] = {
        { "dense packed input", dispatch->dense_packed, &prepared->dense },
        { "sparse packed input", dispatch->sparse_packed_producer, &prepared->sparse },
        { "residual destination", dispatch->residual_destination, &prepared->residual },
        { "rho destination", dispatch->rho_destination, &prepared->rho },
    };
    for (const binding_request & request : bindings) {
        char binding_reason[256] = {};
        if (resolve_tensor_binding(
                graph, request.tensor, request.binding,
                binding_reason, sizeof(binding_reason)) != GGML_OPENCL_HTP_OK) {
            return unsupported_sparse_node(
                reason, reason_capacity, "%s binding is unavailable: %s",
                request.name, binding_reason);
        }
        const size_t alignment = std::strcmp(request.name, "residual destination") == 0
            ? sizeof(ggml_fp16_t) : sizeof(float);
        if (request.binding->offset % alignment != 0) {
            return unsupported_sparse_node(
                reason, reason_capacity, "%s offset is not correctly aligned", request.name);
        }
    }
    if (prepared->dense.backing == prepared->sparse.backing) {
        const ggml_tensor * dense_producer =
            producer_through_views(dispatch->dense_packed);
        const ggml_tensor * sparse_producer =
            producer_through_views(dispatch->sparse_packed_producer);
        if (bindings_overlap(prepared->dense, prepared->sparse)) {
            return unsupported_sparse_node(
                reason, reason_capacity,
                "dense and sparse packed input spans overlap");
        }
        if (dense_producer == sparse_producer) {
            if (dense_producer == nullptr ||
                dense_producer->op != GGML_OP_CUSTOM ||
                std::strcmp(dense_producer->name,
                            "htp_fused_anchor_attention") != 0 ||
                prepared->dense.offset + prepared->dense.span !=
                    prepared->sparse.offset) {
                return unsupported_sparse_node(
                    reason, reason_capacity,
                    "shared fused dense/sparse input spans must be adjacent and "
                    "come from htp_fused_anchor_attention");
            }
        } else if (dense_producer == nullptr || sparse_producer == nullptr ||
                   ggml_op_is_empty(dense_producer->op) ||
                   ggml_op_is_empty(sparse_producer->op)) {
            return unsupported_sparse_node(
                reason, reason_capacity,
                "shared disjoint dense/sparse inputs require distinct live producers");
        }
    }
    if (bindings_overlap(prepared->residual, prepared->rho)) {
        return unsupported_sparse_node(
            reason, reason_capacity, "residual and rho destination spans overlap");
    }
    if (bindings_overlap(prepared->residual, prepared->dense) ||
        bindings_overlap(prepared->residual, prepared->sparse) ||
        bindings_overlap(prepared->rho, prepared->dense) ||
        bindings_overlap(prepared->rho, prepared->sparse)) {
        return unsupported_sparse_node(
            reason, reason_capacity, "residual destination storage overlaps a packed input span");
    }

    prepared->dispatch = *dispatch;
    prepared->residual_cache_key = dispatch->residual_cache_key;
    prepared->rho_cache_key = dispatch->rho_cache_key;
    const ggml_tensor * tensors[] = {
        dispatch->dense_packed,
        dispatch->sparse_packed_producer,
        dispatch->residual_destination,
        dispatch->rho_destination,
    };
    for (size_t i = 0; i < sizeof(tensors) / sizeof(tensors[0]); ++i) {
        prepared->tensor_metadata[i] = capture_tensor_metadata(tensors[i]);
    }
    set_reason(reason, reason_capacity, "ok");
    return GGML_OPENCL_HTP_OK;
}

ggml_opencl_htp_prepared_sparse_dispatch make_prepared_sparse_dispatch(
        const ggml_opencl_htp_sparse_dispatch & dispatch,
        const ggml_opencl_htp_sparse_bindings & bindings) {
    ggml_opencl_htp_prepared_sparse_dispatch prepared;
    prepared.dispatch = dispatch;
    prepared.bindings = bindings;
    const ggml_tensor * tensors[] = {
        dispatch.q, dispatch.k, dispatch.v, dispatch.mask, dispatch.output,
    };
    for (size_t i = 0; i < sizeof(tensors) / sizeof(tensors[0]); ++i) {
        prepared.tensor_metadata[i] = capture_tensor_metadata(tensors[i]);
    }
    std::memcpy(
        prepared.output_op_params, dispatch.output->op_params,
        sizeof(prepared.output_op_params));
    return prepared;
}

bool prepared_sparse_metadata_matches(
        const ggml_opencl_htp_prepared_sparse_dispatch & prepared,
        const ggml_tensor * output) {
    const ggml_tensor * tensors[] = {
        output->src[0], output->src[1], output->src[2], output->src[3], output,
    };
    for (size_t i = 0; i < sizeof(tensors) / sizeof(tensors[0]); ++i) {
        if (!tensor_metadata_matches(tensors[i], prepared.tensor_metadata[i])) {
            return false;
        }
    }
    return std::memcmp(
        output->op_params, prepared.output_op_params,
        sizeof(prepared.output_op_params)) == 0;
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
        node->type != GGML_TYPE_F32 ||
        (q->ne[0] != 64 && q->ne[0] != 128) ||
        k->ne[0] != q->ne[0] || v->ne[0] != q->ne[0] ||
        node->ne[0] != q->ne[0] + 1) {
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

bool is_view_only_node(const ggml_tensor * tensor) {
    return tensor != nullptr &&
        (tensor->op == GGML_OP_VIEW || tensor->op == GGML_OP_RESHAPE ||
         tensor->op == GGML_OP_PERMUTE || tensor->op == GGML_OP_TRANSPOSE);
}

const ggml_tensor * allocating_root(const ggml_tensor * tensor) {
    for (size_t depth = 0; tensor != nullptr && depth < 64; ++depth) {
        if (tensor->view_src != nullptr) {
            tensor = tensor->view_src;
            continue;
        }
        if (is_view_only_node(tensor) && tensor->src[0] != nullptr) {
            tensor = tensor->src[0];
            continue;
        }
        return tensor;
    }
    return nullptr;
}

bool sparse_inputs_have_output_lifetime(const ggml_tensor * sparse) {
    if (sparse == nullptr || sparse->op != GGML_OP_FLASH_ATTN_EXT) {
        return false;
    }
    for (int source = 0; source < 4; ++source) {
        const ggml_tensor * root = allocating_root(sparse->src[source]);
        if (root == nullptr ||
            (root->flags & GGML_TENSOR_FLAG_OUTPUT) == 0) {
            return false;
        }
    }
    return true;
}

bool reaches_producer_through_views(
    const ggml_tensor * tensor,
    const ggml_tensor * producer) {
    for (size_t depth = 0; tensor != nullptr && depth < 64; ++depth) {
        if (tensor == producer) {
            return true;
        }
        if (!is_view_only_node(tensor)) {
            return false;
        }
        tensor = tensor->src[0] != nullptr ? tensor->src[0] : tensor->view_src;
    }
    return false;
}

bool output_lifetime_ends_before(
    const ggml_cgraph * cgraph,
    const ggml_tensor * producer,
    int producer_index,
    int reuse_index) {
    bool has_real_consumer = false;
    const int node_count = ggml_graph_n_nodes(const_cast<ggml_cgraph *>(cgraph));
    for (int i = producer_index + 1; i < node_count; ++i) {
        const ggml_tensor * node = ggml_graph_node(const_cast<ggml_cgraph *>(cgraph), i);
        if (node == nullptr || ggml_op_is_empty(node->op) ||
            (node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }
        bool consumes_producer = false;
        for (int source = 0; source < GGML_MAX_SRC; ++source) {
            consumes_producer = consumes_producer ||
                reaches_producer_through_views(node->src[source], producer);
        }
        if (!consumes_producer) {
            continue;
        }
        has_real_consumer = true;
        if (i >= reuse_index) {
            return false;
        }
    }
    return has_real_consumer;
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
    if (sidecar->context == nullptr || sidecar->sparse_queue == nullptr ||
        sidecar->page_size == 0 || !sidecar->sparse_queue_is_in_order) {
        set_reason(reason, reason_capacity, "sidecar is not ready for graph imports");
        return GGML_OPENCL_HTP_UNSUPPORTED;
    }

    std::unique_ptr<ggml_opencl_htp_graph> created(new (std::nothrow) ggml_opencl_htp_graph());
    if (!created) {
        set_reason(reason, reason_capacity, "failed to allocate graph state");
        return GGML_OPENCL_HTP_ERROR;
    }
    created->sidecar = sidecar;
    const char * residual_queue =
        std::getenv("SD_HTP_SPARSE_OPENCL_RESIDUAL_QUEUE");
    created->use_residual_queue =
        residual_queue == nullptr ||
        (residual_queue[0] != '\0' &&
         std::strcmp(residual_queue, "0") != 0);
    if (created->use_residual_queue) {
        char queue_reason[256] = {};
        if (ensure_residual_queue(
                sidecar, queue_reason, sizeof(queue_reason)) !=
            GGML_OPENCL_HTP_OK) {
            const char * strict = std::getenv("SD_HTP_SPARSE_OPENCL_STRICT");
            const bool strict_enabled =
                strict != nullptr && strict[0] != '\0' &&
                std::strcmp(strict, "0") != 0;
            if (strict_enabled) {
                set_reason(reason, reason_capacity, "%s", queue_reason);
                return GGML_OPENCL_HTP_ERROR;
            }
            created->use_residual_queue = false;
            std::lock_guard<std::mutex> lock(sidecar->queue_mutex);
            if (!sidecar->residual_queue_fallback_logged) {
                std::fprintf(
                    stderr,
                    "ggml-opencl: residual queue unavailable; using sparse queue: %s\n",
                    queue_reason[0] != '\0' ? queue_reason : "unknown error");
                sidecar->residual_queue_fallback_logged = true;
            }
        }
    }
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

    {
        std::lock_guard<std::mutex> lock(graph->sidecar->cache_mutex);
        if (ggml_opencl_htp_persistent_import * imported =
                find_persistent_import(graph->sidecar, descriptor->backing)) {
            if (!descriptors_equal(imported->descriptor, *descriptor)) {
                return fail_graph(
                    graph,
                    reason, reason_capacity,
                    "persistent backing %p was imported with different metadata",
                    static_cast<void *>(descriptor->backing));
            }
            const cl_int retain_err = clRetainMemObject(imported->memory);
            if (retain_err != CL_SUCCESS) {
                return fail_graph(
                    graph, reason, reason_capacity,
                    "failed to retain persistent OpenCL import (error %d)", retain_err);
            }
            try {
                graph->aliases.push_back({ imported->descriptor, imported->memory });
            } catch (...) {
                const cl_int release_err = clReleaseMemObject(imported->memory);
                return fail_graph(
                    graph, reason, reason_capacity,
                    "failed to cache retained persistent import (release error %d)",
                    release_err);
            }
            set_reason(reason, reason_capacity, "ok");
            return GGML_OPENCL_HTP_OK;
        }
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

ggml_opencl_htp_result ggml_backend_opencl_htp_persistent_import_acquire(
        ggml_opencl_htp_sidecar * sidecar,
        const ggml_opencl_htp_import_descriptor * descriptor,
        const char * cache_key,
        uint64_t generation,
        uint64_t * import_identity,
        char * reason,
        size_t reason_capacity) {
    if (sidecar == nullptr || cache_key == nullptr || cache_key[0] == '\0' ||
        import_identity == nullptr) {
        set_reason(reason, reason_capacity,
                   "sidecar, cache key, and import identity output must be valid");
        return GGML_OPENCL_HTP_ERROR;
    }
    ggml_opencl_htp_graph validation_graph;
    validation_graph.sidecar = sidecar;
    const ggml_opencl_htp_result validation = validate_import_descriptor(
        &validation_graph, descriptor, reason, reason_capacity);
    if (validation != GGML_OPENCL_HTP_OK) {
        return validation;
    }

    std::lock_guard<std::mutex> lock(sidecar->cache_mutex);
    if (ggml_opencl_htp_persistent_import * existing =
            find_persistent_import(sidecar, cache_key, generation)) {
        if (!descriptors_equal(existing->descriptor, *descriptor)) {
            set_reason(reason, reason_capacity,
                       "cache generation already has a different RPCMEM backing");
            return GGML_OPENCL_HTP_UNSUPPORTED;
        }
        *import_identity = reinterpret_cast<uint64_t>(existing->memory);
        set_reason(reason, reason_capacity, "ok");
        return GGML_OPENCL_HTP_OK;
    }
    if (find_persistent_import(sidecar, descriptor->backing) != nullptr) {
        set_reason(reason, reason_capacity,
                   "RPCMEM backing is already owned by another cache generation");
        return GGML_OPENCL_HTP_UNSUPPORTED;
    }
    const size_t generation_count = static_cast<size_t>(std::count_if(
        sidecar->persistent_imports.begin(), sidecar->persistent_imports.end(),
        [cache_key](const ggml_opencl_htp_persistent_import & imported) {
            return imported.key == cache_key;
        }));
    if (generation_count >= 1) {
        set_reason(reason, reason_capacity,
                   "cache key already owns the maximum of one generation");
        return GGML_OPENCL_HTP_UNSUPPORTED;
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
        sidecar->context, flags, descriptor->logical_size, &host_ptr, &err);
    if (err != CL_SUCCESS || memory == nullptr) {
        if (memory != nullptr) {
            clReleaseMemObject(memory);
        }
        set_reason(reason, reason_capacity,
                   "clCreateBuffer failed for persistent cache backing (error %d)", err);
        return GGML_OPENCL_HTP_ERROR;
    }

    try {
        sidecar->persistent_imports.push_back({
            cache_key, generation, *descriptor, memory,
        });
    } catch (...) {
        const cl_int release_err = clReleaseMemObject(memory);
        set_reason(reason, reason_capacity,
                   "failed to retain persistent import state (release error %d)",
                   release_err);
        return GGML_OPENCL_HTP_ERROR;
    }
    *import_identity = reinterpret_cast<uint64_t>(memory);
    set_reason(reason, reason_capacity, "ok");
    return GGML_OPENCL_HTP_OK;
}

ggml_opencl_htp_result ggml_backend_opencl_htp_persistent_import_release(
        ggml_opencl_htp_sidecar * sidecar,
        const char * cache_key,
        uint64_t generation,
        char * reason,
        size_t reason_capacity) {
    if (sidecar == nullptr || cache_key == nullptr || cache_key[0] == '\0') {
        set_reason(reason, reason_capacity, "sidecar and cache key must be valid");
        return GGML_OPENCL_HTP_ERROR;
    }
    std::lock_guard<std::mutex> lock(sidecar->cache_mutex);
    const auto found = std::find_if(
        sidecar->persistent_imports.begin(), sidecar->persistent_imports.end(),
        [cache_key, generation](const ggml_opencl_htp_persistent_import & imported) {
            return imported.generation == generation && imported.key == cache_key;
        });
    if (found == sidecar->persistent_imports.end()) {
        set_reason(reason, reason_capacity, "persistent cache generation was not found");
        return GGML_OPENCL_HTP_UNSUPPORTED;
    }
    const cl_int release_err = found->memory != nullptr ?
        clReleaseMemObject(found->memory) : CL_SUCCESS;
    sidecar->persistent_imports.erase(found);
    cl_int event_release_err = CL_SUCCESS;
    for (auto record = sidecar->cache_records.begin();
            record != sidecar->cache_records.end();) {
        if (record->generation != generation || record->key != cache_key) {
            ++record;
            continue;
        }
        if (record->producer_event != nullptr) {
            const cl_int err = clReleaseEvent(record->producer_event);
            if (event_release_err == CL_SUCCESS) {
                event_release_err = err;
            }
            record->producer_event = nullptr;
        }
        if (record->profiling_dependency_event != nullptr) {
            const cl_int err = clReleaseEvent(
                record->profiling_dependency_event);
            if (event_release_err == CL_SUCCESS) {
                event_release_err = err;
            }
            record->profiling_dependency_event = nullptr;
        }
        record = sidecar->cache_records.erase(record);
    }
    if (release_err != CL_SUCCESS || event_release_err != CL_SUCCESS) {
        set_reason(reason, reason_capacity,
                   "failed to release persistent OpenCL import/cache event "
                   "(memory error %d, event error %d)",
                   release_err, event_release_err);
        return GGML_OPENCL_HTP_ERROR;
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

    auto env_enabled = [](const char * name) {
        const char * value = std::getenv(name);
        return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
    };
    const bool sparse_enabled = env_enabled("SD_HTP_SPARSE_OPENCL");
    const bool residual_enabled = env_enabled("SD_HTP_SPARSE_OPENCL_RESIDUAL");
    const bool sparse_anchor_only =
        env_enabled("SD_HTP_SPARSE_OPENCL_ANCHOR_ONLY");
    if (!sparse_enabled && !residual_enabled) {
        set_reason(reason, reason_capacity,
                   "neither sparse attention nor residual OpenCL routing is enabled");
        return GGML_OPENCL_HTP_UNSUPPORTED;
    }

    std::vector<ggml_opencl_htp_sparse_dispatch> dispatches;
    struct residual_pair_candidate {
        const ggml_tensor * rho = nullptr;
        const ggml_tensor * residual = nullptr;
        const ggml_tensor * dense = nullptr;
        const ggml_tensor * sparse = nullptr;
        std::string rho_key;
        std::string residual_key;
        uint64_t generation = 0;
    };
    std::vector<residual_pair_candidate> residual_pairs;
    std::vector<const ggml_tensor *> routed_nodes;
    std::vector<ggml_backend_buffer_t> required_backings;
    try {
        const int node_count = ggml_graph_n_nodes(const_cast<ggml_cgraph *>(cgraph));
        bool graph_has_anchor_residual = false;
        if (sparse_anchor_only) {
            for (int i = 0; i < node_count; ++i) {
                const ggml_tensor * node =
                    ggml_graph_node(const_cast<ggml_cgraph *>(cgraph), i);
                if (node != nullptr && node->op == GGML_OP_CUSTOM &&
                    std::strncmp(node->name,
                                 "htp_block_sparse_residual_b", 27) == 0) {
                    graph_has_anchor_residual = true;
                    break;
                }
            }
        }
        if (sparse_enabled && (!sparse_anchor_only || graph_has_anchor_residual)) {
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
                dispatch.graph_index = i;
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
        }
        if (residual_enabled) {
            for (int i = 0; i < node_count; ++i) {
                const ggml_tensor * residual =
                    ggml_graph_node(const_cast<ggml_cgraph *>(cgraph), i);
                if (residual == nullptr || residual->op != GGML_OP_CUSTOM ||
                    std::strncmp(residual->name,
                                 "htp_block_sparse_residual_b", 27) != 0) {
                    continue;
                }
                const ggml_tensor * rho = residual->src[2];
                if (rho == nullptr || rho->op != GGML_OP_CUSTOM ||
                    std::strncmp(rho->name,
                                 "htp_block_sparse_residual_rho_b", 31) != 0 ||
                    residual->src[0] == nullptr || residual->src[1] == nullptr ||
                    rho->src[0] != residual->src[0] || rho->src[1] != residual->src[1]) {
                    set_reason(reason, reason_capacity,
                               "tagged residual node %d has an invalid rho pair", i);
                    return GGML_OPENCL_HTP_UNSUPPORTED;
                }

                ggml_opencl_htp_persistent_import * residual_import = nullptr;
                ggml_opencl_htp_persistent_import * rho_import = nullptr;
                {
                    std::lock_guard<std::mutex> lock(sidecar->cache_mutex);
                    residual_import = find_persistent_import(
                        sidecar, tensor_backing(residual));
                    rho_import = find_persistent_import(sidecar, tensor_backing(rho));
                    if (residual_import == nullptr || rho_import == nullptr ||
                        residual_import->generation != rho_import->generation ||
                        residual_import->key != residual->name ||
                        rho_import->key != rho->name) {
                        set_reason(reason, reason_capacity,
                                   "tagged residual pair lacks matching persistent cache imports");
                        return GGML_OPENCL_HTP_UNSUPPORTED;
                    }
                    residual_pairs.push_back({
                        rho,
                        residual,
                        residual->src[0],
                        residual->src[1],
                        rho_import->key,
                        residual_import->key,
                        residual_import->generation,
                    });
                }
                routed_nodes.push_back(rho);
                routed_nodes.push_back(residual);
                const ggml_tensor * tensors[] = {
                    residual->src[0], residual->src[1], rho, residual,
                };
                for (const ggml_tensor * tensor : tensors) {
                    ggml_backend_buffer_t backing = tensor_backing(tensor);
                    if (backing == nullptr) {
                        set_reason(reason, reason_capacity,
                                   "tagged residual tensor has no backing buffer");
                        return GGML_OPENCL_HTP_UNSUPPORTED;
                    }
                    if (std::find(required_backings.begin(), required_backings.end(), backing) ==
                        required_backings.end()) {
                        required_backings.push_back(backing);
                    }
                }
            }
        }
    } catch (...) {
        set_reason(reason, reason_capacity, "failed to allocate preflight routing state");
        return GGML_OPENCL_HTP_ERROR;
    }

    if (dispatches.empty() && residual_pairs.empty()) {
        set_reason(reason, reason_capacity,
                   "graph has no eligible sparse attention or residual nodes");
        return GGML_OPENCL_HTP_NO_WORK;
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

    struct routed_output_binding {
        ggml_opencl_htp_tensor_binding binding;
        const ggml_tensor * producer = nullptr;
        int graph_index = -1;
    };
    std::vector<routed_output_binding> output_bindings;
    try {
        output_bindings.reserve(dispatches.size());
        prepared->prepared_sparse_dispatches.reserve(dispatches.size());
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
            for (const routed_output_binding & prior : output_bindings) {
                if (bindings_overlap(bindings.output, prior.binding) &&
                    !output_lifetime_ends_before(
                        cgraph, prior.producer, prior.graph_index, dispatch.graph_index)) {
                    return reject_preflight(
                        GGML_OPENCL_HTP_UNSUPPORTED,
                        "routed sparse attention output spans overlap");
                }
            }
            output_bindings.push_back({ bindings.output, dispatch.output, dispatch.graph_index });
            prepared->prepared_sparse_dispatches.push_back(
                make_prepared_sparse_dispatch(dispatch, bindings));
        }
        for (const residual_pair_candidate & pair : residual_pairs) {
            const ggml_opencl_htp_residual_dispatch residual_dispatch = {
                pair.dense,
                pair.sparse,
                pair.residual,
                pair.rho,
                pair.residual_key.c_str(),
                pair.rho_key.c_str(),
                pair.generation,
            };
            char residual_reason[512] = {};
            const ggml_opencl_htp_result residual_result =
                ggml_backend_opencl_htp_graph_prepare_exact_rho_residual(
                    prepared, &residual_dispatch,
                    residual_reason, sizeof(residual_reason));
            if (residual_result != GGML_OPENCL_HTP_OK) {
                return reject_preflight(
                    residual_result,
                    residual_reason[0] != '\0' ? residual_reason :
                        "failed to prepare tagged residual pair");
            }

            bool has_other_consumer = false;
            const int node_count =
                ggml_graph_n_nodes(const_cast<ggml_cgraph *>(cgraph));
            for (int node_index = 0;
                 node_index < node_count && !has_other_consumer;
                 ++node_index) {
                const ggml_tensor * consumer =
                    ggml_graph_node(const_cast<ggml_cgraph *>(cgraph), node_index);
                if (consumer == pair.rho || consumer == pair.residual) {
                    continue;
                }
                for (int source = 0; source < GGML_MAX_SRC; ++source) {
                    const ggml_tensor * input =
                        consumer != nullptr ? consumer->src[source] : nullptr;
                    for (size_t depth = 0; input != nullptr && depth < 64; ++depth) {
                        if (input == pair.sparse) {
                            has_other_consumer = true;
                            break;
                        }
                        if (input->op != GGML_OP_VIEW &&
                            input->op != GGML_OP_RESHAPE &&
                            input->op != GGML_OP_PERMUTE &&
                            input->op != GGML_OP_TRANSPOSE) {
                            break;
                        }
                        input = input->src[0] != nullptr ?
                            input->src[0] : input->view_src;
                    }
                    if (has_other_consumer) {
                        break;
                    }
                }
            }
            if (!has_other_consumer &&
                sparse_inputs_have_output_lifetime(pair.sparse) &&
                std::find(
                    prepared->residual_owned_sparse_producers.begin(),
                    prepared->residual_owned_sparse_producers.end(),
                    pair.sparse) ==
                    prepared->residual_owned_sparse_producers.end()) {
                prepared->residual_owned_sparse_producers.push_back(pair.sparse);
            }
        }
        prepared->routed_nodes = std::move(routed_nodes);
    } catch (...) {
        return reject_preflight(GGML_OPENCL_HTP_ERROR, "failed to allocate binding preflight state");
    }

    {
        std::lock_guard<std::mutex> lock(sidecar->dispatch_mutex);
        for (const ggml_opencl_htp_prepared_sparse_dispatch & sparse :
             prepared->prepared_sparse_dispatches) {
            char kernel_reason[1024] = {};
            ggml_opencl_htp_block_sparse_kernel * selected_kernel = nullptr;
            const ggml_opencl_htp_result selection_result = select_block_sparse_kernel(
                sidecar, sparse.dispatch.q->ne[0], sparse.dispatch.mask_format,
                &selected_kernel, kernel_reason, sizeof(kernel_reason));
            if (selection_result != GGML_OPENCL_HTP_OK) {
                return reject_preflight(GGML_OPENCL_HTP_UNSUPPORTED, kernel_reason);
            }
            const ggml_opencl_htp_result kernel_result = ensure_block_sparse_kernel(
                sidecar, selected_kernel, kernel_reason, sizeof(kernel_reason));
            if (kernel_result != GGML_OPENCL_HTP_OK) {
                return reject_preflight(GGML_OPENCL_HTP_UNSUPPORTED, kernel_reason);
            }
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

ggml_opencl_htp_result ggml_backend_opencl_htp_graph_prepare_exact_rho_residual(
        ggml_opencl_htp_graph * graph,
        const ggml_opencl_htp_residual_dispatch * dispatch,
        char * reason,
        size_t reason_capacity) {
    if (graph == nullptr || dispatch == nullptr) {
        set_reason(reason, reason_capacity, "graph and residual dispatch must not be null");
        return GGML_OPENCL_HTP_ERROR;
    }
    if (graph->state != ggml_opencl_htp_graph_state::active) {
        set_reason(reason, reason_capacity, "graph is already finished");
        return GGML_OPENCL_HTP_ERROR;
    }

    ggml_opencl_htp_prepared_residual_dispatch prepared;
    const ggml_opencl_htp_result validation = validate_residual_dispatch(
        graph, dispatch, &prepared, reason, reason_capacity);
    if (validation != GGML_OPENCL_HTP_OK) {
        return validation;
    }

    ggml_opencl_htp_sidecar * sidecar = graph->sidecar;
    std::lock_guard<std::mutex> lock(sidecar->cache_mutex);
    if (find_cache_record(sidecar, dispatch->residual_cache_key, dispatch->generation) != nullptr ||
        find_cache_record(sidecar, dispatch->rho_cache_key, dispatch->generation) != nullptr) {
        set_reason(
            reason, reason_capacity,
            "duplicate generation publication for cache key at generation %llu",
            static_cast<unsigned long long>(dispatch->generation));
        return GGML_OPENCL_HTP_UNSUPPORTED;
    }

    try {
        sidecar->cache_records.push_back({
            prepared.residual_cache_key,
            dispatch->generation,
            GGML_OPENCL_HTP_CACHE_ALLOCATED,
            {},
            prepared.rho_cache_key,
            nullptr,
        });
        sidecar->cache_records.push_back({
            prepared.rho_cache_key,
            dispatch->generation,
            GGML_OPENCL_HTP_CACHE_ALLOCATED,
            {},
            prepared.residual_cache_key,
            nullptr,
        });
        graph->prepared_residual_dispatches.push_back(std::move(prepared));
    } catch (...) {
        sidecar->cache_records.erase(
            std::remove_if(
                sidecar->cache_records.begin(), sidecar->cache_records.end(),
                [&](const ggml_opencl_htp_cache_record & record) {
                    return record.generation == dispatch->generation &&
                           (record.key == dispatch->residual_cache_key ||
                            record.key == dispatch->rho_cache_key);
                }),
            sidecar->cache_records.end());
        set_reason(reason, reason_capacity, "failed to allocate residual preparation state");
        return GGML_OPENCL_HTP_ERROR;
    }

    set_reason(reason, reason_capacity, "ok");
    return GGML_OPENCL_HTP_OK;
}

ggml_opencl_htp_result ggml_backend_opencl_htp_graph_queue_exact_rho_residual(
        ggml_opencl_htp_graph * graph,
        const char * residual_cache_key,
        uint64_t generation,
        char * reason,
        size_t reason_capacity) {
    if (graph == nullptr || residual_cache_key == nullptr || residual_cache_key[0] == '\0') {
        set_reason(reason, reason_capacity, "graph and residual cache key must be valid");
        return GGML_OPENCL_HTP_ERROR;
    }
    if (graph->state != ggml_opencl_htp_graph_state::active) {
        set_reason(reason, reason_capacity, "graph is already finished");
        return GGML_OPENCL_HTP_ERROR;
    }
    const char * residual_profile = std::getenv("SD_HTP_RESIDUAL_PROFILE");
    const bool residual_profile_enabled =
        residual_profile != nullptr && residual_profile[0] != '\0' &&
        std::strcmp(residual_profile, "0") != 0;
    const int64_t host_begin_us = residual_profile_enabled ? ggml_time_us() : 0;

    ggml_opencl_htp_prepared_residual_dispatch * prepared = nullptr;
    for (ggml_opencl_htp_prepared_residual_dispatch & candidate :
            graph->prepared_residual_dispatches) {
        if (candidate.dispatch.generation == generation &&
            candidate.residual_cache_key == residual_cache_key) {
            prepared = &candidate;
            break;
        }
    }
    if (prepared == nullptr) {
        set_reason(reason, reason_capacity, "residual job was not prepared");
        return GGML_OPENCL_HTP_UNSUPPORTED;
    }

    const ggml_tensor * tensors[] = {
        prepared->dispatch.dense_packed,
        prepared->dispatch.sparse_packed_producer,
        prepared->dispatch.residual_destination,
        prepared->dispatch.rho_destination,
    };
    for (size_t i = 0; i < sizeof(tensors) / sizeof(tensors[0]); ++i) {
        if (!tensor_metadata_matches(tensors[i], prepared->tensor_metadata[i])) {
            set_reason(reason, reason_capacity, "prepared residual tensor metadata changed");
            return GGML_OPENCL_HTP_ERROR;
        }
    }

    {
        std::lock_guard<std::mutex> lock(graph->sidecar->cache_mutex);
        ggml_opencl_htp_cache_record * residual_record = find_cache_record(
            graph->sidecar, prepared->residual_cache_key.c_str(), generation);
        ggml_opencl_htp_cache_record * rho_record = find_cache_record(
            graph->sidecar, prepared->rho_cache_key.c_str(), generation);
        if (residual_record == nullptr || rho_record == nullptr ||
            residual_record->state != GGML_OPENCL_HTP_CACHE_ALLOCATED ||
            rho_record->state != GGML_OPENCL_HTP_CACHE_ALLOCATED) {
            set_reason(reason, reason_capacity, "residual job is not in allocated state");
            return GGML_OPENCL_HTP_UNSUPPORTED;
        }
        residual_record->state = GGML_OPENCL_HTP_CACHE_QUEUED;
        rho_record->state = GGML_OPENCL_HTP_CACHE_QUEUED;
    }

    auto fail_cache_job = [&](const char * failure) {
        std::lock_guard<std::mutex> lock(graph->sidecar->cache_mutex);
        ggml_opencl_htp_cache_record * residual_record = find_cache_record(
            graph->sidecar, prepared->residual_cache_key.c_str(), generation);
        ggml_opencl_htp_cache_record * rho_record = find_cache_record(
            graph->sidecar, prepared->rho_cache_key.c_str(), generation);
        if (residual_record != nullptr) {
            residual_record->state = GGML_OPENCL_HTP_CACHE_FAILED;
            residual_record->failure_reason = failure;
        }
        if (rho_record != nullptr) {
            rho_record->state = GGML_OPENCL_HTP_CACHE_FAILED;
            rho_record->failure_reason = failure;
        }
        set_reason(reason, reason_capacity, "%s", failure);
        return GGML_OPENCL_HTP_ERROR;
    };

    cl_event fused_event = nullptr;
    cl_event sparse_dependency_event = nullptr;
    cl_command_queue command_queue = graph->use_residual_queue ?
        graph->sidecar->residual_queue : graph->sidecar->sparse_queue;
    {
        std::lock_guard<std::mutex> dispatch_lock(graph->sidecar->dispatch_mutex);
        char kernel_reason[1024] = {};
        if (ensure_residual_kernel(
                graph->sidecar, kernel_reason, sizeof(kernel_reason)) != GGML_OPENCL_HTP_OK) {
            return fail_cache_job(kernel_reason);
        }

        const auto sparse_pending = std::find_if(
            graph->pending_events.begin(), graph->pending_events.end(),
            [&](const ggml_opencl_htp_pending_event & pending) {
                return pending.producer == prepared->dispatch.sparse_packed_producer;
            });
        const bool uses_staged_sparse =
            sparse_pending != graph->pending_events.end() &&
            sparse_pending->staged_output != nullptr;
        const cl_mem dense_memory = prepared->dense.memory;
        const cl_mem sparse_memory = uses_staged_sparse ?
            sparse_pending->staged_output : prepared->sparse.memory;
        const cl_mem rho_memory = prepared->rho.memory;
        const cl_mem residual_memory = prepared->residual.memory;
        const cl_ulong dense_offset = static_cast<cl_ulong>(prepared->dense.offset);
        const cl_ulong sparse_offset = uses_staged_sparse ? 0 :
            static_cast<cl_ulong>(prepared->sparse.offset);
        const cl_ulong rho_offset = static_cast<cl_ulong>(prepared->rho.offset);
        const cl_ulong residual_offset = static_cast<cl_ulong>(prepared->residual.offset);
        const cl_ulong head_dim = static_cast<cl_ulong>(
            prepared->dispatch.dense_packed->ne[0] - 1);
        const cl_ulong row_count = static_cast<cl_ulong>(
            prepared->dispatch.dense_packed->ne[1] *
            prepared->dispatch.dense_packed->ne[2] *
            prepared->dispatch.dense_packed->ne[3]);
        struct kernel_arg {
            size_t size;
            const void * value;
        };
        const kernel_arg args[] = {
            { sizeof(dense_memory), &dense_memory },
            { sizeof(dense_offset), &dense_offset },
            { sizeof(sparse_memory), &sparse_memory },
            { sizeof(sparse_offset), &sparse_offset },
            { sizeof(rho_memory), &rho_memory },
            { sizeof(rho_offset), &rho_offset },
            { sizeof(residual_memory), &residual_memory },
            { sizeof(residual_offset), &residual_offset },
            { sizeof(head_dim), &head_dim },
            { sizeof(row_count), &row_count },
        };
        for (cl_uint index = 0; index < sizeof(args) / sizeof(args[0]); ++index) {
            const cl_int err = clSetKernelArg(
                graph->sidecar->rho_residual_kernel, index,
                args[index].size, args[index].value);
            if (err != CL_SUCCESS) {
                char failure[256] = {};
                std::snprintf(failure, sizeof(failure),
                              "rho-residual clSetKernelArg(%u) failed (error %d)", index, err);
                return fail_cache_job(failure);
            }
        }

        const size_t local = 64;
        const size_t global = static_cast<size_t>(row_count) * local;
        cl_event sparse_event = sparse_pending != graph->pending_events.end() ?
            sparse_pending->event : nullptr;
        sparse_dependency_event = sparse_event;
        std::atomic_thread_fence(std::memory_order_release);
        if (graph->test_fail_next_rho_enqueue) {
            graph->test_fail_next_rho_enqueue = false;
            return fail_cache_job("injected rho enqueue failure");
        }
        if (graph->test_fail_next_residual_enqueue) {
            graph->test_fail_next_residual_enqueue = false;
            return fail_cache_job("injected residual enqueue failure");
        }
        cl_int err = clEnqueueNDRangeKernel(
            command_queue, graph->sidecar->rho_residual_kernel,
            1, nullptr, &global, &local,
            sparse_event != nullptr ? 1 : 0,
            sparse_event != nullptr ? &sparse_event : nullptr,
            &fused_event);
        if (err != CL_SUCCESS || fused_event == nullptr) {
            if (fused_event != nullptr) {
                clReleaseEvent(fused_event);
                fused_event = nullptr;
            }
            char failure[256] = {};
            std::snprintf(failure, sizeof(failure),
                          "rho-residual clEnqueueNDRangeKernel failed (error %d)", err);
            return fail_cache_job(failure);
        }
        trace_sidecar_queue_event(
            "gpu_rho_residual_enqueue",
            graph->use_residual_queue ? "residual" : "sparse",
            prepared->dispatch.residual_destination);
        const char * fail_residual_flush = std::getenv(
            "SD_HTP_SPARSE_OPENCL_TEST_FAIL_RESIDUAL_QUEUE_FLUSH");
        const bool inject_residual_flush_failure =
            graph->use_residual_queue && fail_residual_flush != nullptr &&
            fail_residual_flush[0] != '\0' &&
            std::strcmp(fail_residual_flush, "0") != 0;
        err = inject_residual_flush_failure ?
            CL_OUT_OF_RESOURCES : clFlush(command_queue);
        if (err != CL_SUCCESS) {
            clFinish(command_queue);
            clReleaseEvent(fused_event);
            fused_event = nullptr;
            char failure[256] = {};
            std::snprintf(
                failure, sizeof(failure),
                inject_residual_flush_failure
                    ? "injected residual queue flush failure (error %d)"
                    : "clFlush failed after residual enqueue (error %d)",
                err);
            return fail_cache_job(failure);
        }
    }

    const cl_int retain_err = clRetainEvent(fused_event);
    if (retain_err != CL_SUCCESS) {
        clFinish(command_queue);
        clReleaseEvent(fused_event);
        return fail_cache_job("failed to retain residual cache event");
    }
    if (graph->sidecar->profiling_enabled &&
        sparse_dependency_event != nullptr) {
        const cl_int dependency_retain_err =
            clRetainEvent(sparse_dependency_event);
        if (dependency_retain_err != CL_SUCCESS) {
            clFinish(command_queue);
            clReleaseEvent(fused_event);
            clReleaseEvent(fused_event);
            return fail_cache_job(
                "failed to retain sparse profiling dependency event");
        }
    } else {
        sparse_dependency_event = nullptr;
    }
    {
        std::lock_guard<std::mutex> lock(graph->sidecar->cache_mutex);
        ggml_opencl_htp_cache_record * residual_record = find_cache_record(
            graph->sidecar, prepared->residual_cache_key.c_str(), generation);
        ggml_opencl_htp_cache_record * rho_record = find_cache_record(
            graph->sidecar, prepared->rho_cache_key.c_str(), generation);
        GGML_ASSERT(residual_record != nullptr && rho_record != nullptr);
        residual_record->state = GGML_OPENCL_HTP_CACHE_SUBMITTED;
        rho_record->state = GGML_OPENCL_HTP_CACHE_SUBMITTED;
        residual_record->producer_event = fused_event;
        rho_record->producer_event = fused_event;
        residual_record->profiling_dependency_event =
            sparse_dependency_event;
        residual_record->profiling_owner = true;
        rho_record->profiling_owner = false;
    }
    if (residual_profile_enabled) {
        const int64_t host_end_us = ggml_time_us();
        std::fprintf(
            stderr,
            "htp_opencl_residual_host_timing generation=%llu total_us=%lld\n",
            static_cast<unsigned long long>(generation),
            (long long) (host_end_us - host_begin_us));
    }
    set_reason(reason, reason_capacity, "submitted");
    return GGML_OPENCL_HTP_OK;
}

ggml_opencl_htp_result ggml_backend_opencl_htp_graph_compute_exact_rho_residual_sync(
        ggml_opencl_htp_graph * graph,
        const ggml_tensor * residual_node,
        char * reason,
        size_t reason_capacity) {
    if (graph == nullptr || residual_node == nullptr) {
        set_reason(reason, reason_capacity, "graph and residual node must not be null");
        return GGML_OPENCL_HTP_ERROR;
    }
    ggml_opencl_htp_prepared_residual_dispatch * prepared = nullptr;
    for (ggml_opencl_htp_prepared_residual_dispatch & candidate :
            graph->prepared_residual_dispatches) {
        if (candidate.dispatch.residual_destination == residual_node ||
            candidate.dispatch.rho_destination == residual_node) {
            prepared = &candidate;
            break;
        }
    }
    if (prepared == nullptr) {
        set_reason(reason, reason_capacity, "node is not a prepared residual destination");
        return GGML_OPENCL_HTP_UNSUPPORTED;
    }
    trace_sidecar_event("gpu_residual_route", residual_node);
    const ggml_opencl_htp_result queue_result =
        ggml_backend_opencl_htp_graph_queue_exact_rho_residual(
            graph,
            prepared->residual_cache_key.c_str(),
            prepared->dispatch.generation,
            reason,
            reason_capacity);
    if (queue_result != GGML_OPENCL_HTP_OK) {
        return queue_result;
    }
    bool waited = false;
    return ggml_backend_opencl_htp_cache_wait(
        graph->sidecar,
        prepared->residual_cache_key.c_str(),
        prepared->dispatch.generation,
        &waited,
        reason,
        reason_capacity);
}

ggml_opencl_htp_result ggml_backend_opencl_htp_cache_query(
        const ggml_opencl_htp_sidecar * sidecar,
        const char * cache_key,
        uint64_t generation,
        ggml_opencl_htp_cache_state * state,
        char * reason,
        size_t reason_capacity) {
    if (sidecar == nullptr || cache_key == nullptr || cache_key[0] == '\0' || state == nullptr) {
        set_reason(reason, reason_capacity, "sidecar, cache key, and state output must be valid");
        return GGML_OPENCL_HTP_ERROR;
    }
    std::lock_guard<std::mutex> lock(sidecar->cache_mutex);
    ggml_opencl_htp_sidecar * mutable_sidecar =
        const_cast<ggml_opencl_htp_sidecar *>(sidecar);
    ggml_opencl_htp_cache_record * record =
        find_cache_record(mutable_sidecar, cache_key, generation);
    if (record == nullptr) {
        *state = GGML_OPENCL_HTP_CACHE_EMPTY;
        set_reason(reason, reason_capacity, "cache generation was not found");
        return GGML_OPENCL_HTP_UNSUPPORTED;
    }
    if (mutable_sidecar->test_fail_next_residual_wait) {
        mutable_sidecar->test_fail_next_residual_wait = false;
        record->state = GGML_OPENCL_HTP_CACHE_FAILED;
        record->failure_reason = "injected residual event wait failure";
        if (ggml_opencl_htp_cache_record * peer = find_cache_record(
                mutable_sidecar, record->peer_key.c_str(), generation)) {
            peer->state = GGML_OPENCL_HTP_CACHE_FAILED;
            peer->failure_reason = record->failure_reason;
        }
        *state = GGML_OPENCL_HTP_CACHE_FAILED;
        set_reason(reason, reason_capacity, "%s", record->failure_reason.c_str());
        return GGML_OPENCL_HTP_ERROR;
    }
    if (record->state == GGML_OPENCL_HTP_CACHE_SUBMITTED &&
        record->producer_event != nullptr) {
        cl_int execution_status = CL_QUEUED;
        const cl_int status_err = clGetEventInfo(
            record->producer_event,
            CL_EVENT_COMMAND_EXECUTION_STATUS,
            sizeof(execution_status), &execution_status, nullptr);
        if (status_err != CL_SUCCESS) {
            record->state = GGML_OPENCL_HTP_CACHE_FAILED;
            record->failure_reason = "failed to query residual event status";
        } else if (execution_status == CL_COMPLETE) {
            ggml_opencl_htp_cache_record * peer = find_cache_record(
                mutable_sidecar, record->peer_key.c_str(), generation);
            ggml_opencl_htp_cache_record * primary =
                record->profiling_owner ? record : peer;
            if (primary != nullptr && primary->producer_event != nullptr &&
                sidecar->profiling_enabled) {
                cl_ulong fused_start = 0;
                cl_ulong fused_end = 0;
                cl_ulong sparse_end = 0;
                const bool dependency_profiling_ok =
                    primary->profiling_dependency_event == nullptr ||
                    clGetEventProfilingInfo(
                        primary->profiling_dependency_event,
                        CL_PROFILING_COMMAND_END,
                        sizeof(sparse_end), &sparse_end, nullptr) == CL_SUCCESS;
                const bool profiling_ok = dependency_profiling_ok &&
                    clGetEventProfilingInfo(
                        primary->producer_event, CL_PROFILING_COMMAND_START,
                        sizeof(fused_start), &fused_start, nullptr) == CL_SUCCESS &&
                    clGetEventProfilingInfo(
                        primary->producer_event, CL_PROFILING_COMMAND_END,
                        sizeof(fused_end), &fused_end, nullptr) == CL_SUCCESS;
                if (profiling_ok) {
                    std::fprintf(
                        stderr,
                        "htp_opencl_residual_timing generation=%llu "
                        "fused_gpu_us=%.3f "
                        "sparse_end_ns=%llu fused_start_ns=%llu\n",
                        static_cast<unsigned long long>(generation),
                        (fused_end - fused_start) / 1000.0,
                        static_cast<unsigned long long>(sparse_end),
                        static_cast<unsigned long long>(fused_start));
                }
            }
            auto mark_ready_and_release = [](ggml_opencl_htp_cache_record * item) {
                if (item == nullptr) {
                    return;
                }
                item->state = GGML_OPENCL_HTP_CACHE_READY;
                if (item->producer_event != nullptr) {
                    clReleaseEvent(item->producer_event);
                    item->producer_event = nullptr;
                }
                if (item->profiling_dependency_event != nullptr) {
                    clReleaseEvent(item->profiling_dependency_event);
                    item->profiling_dependency_event = nullptr;
                }
            };
            mark_ready_and_release(record);
            mark_ready_and_release(peer);
            trace_sidecar_event("gpu_residual_complete", nullptr);
        } else if (execution_status < 0) {
            record->state = GGML_OPENCL_HTP_CACHE_FAILED;
            record->failure_reason = "residual OpenCL command failed";
            if (ggml_opencl_htp_cache_record * peer = find_cache_record(
                    mutable_sidecar, record->peer_key.c_str(), generation)) {
                peer->state = GGML_OPENCL_HTP_CACHE_FAILED;
                peer->failure_reason = record->failure_reason;
            }
        }
    }
    *state = record->state;
    set_reason(reason, reason_capacity, "%s",
               record->failure_reason.empty() ? "ok" : record->failure_reason.c_str());
    return record->state == GGML_OPENCL_HTP_CACHE_FAILED ?
        GGML_OPENCL_HTP_ERROR : GGML_OPENCL_HTP_OK;
}

ggml_opencl_htp_result ggml_backend_opencl_htp_cache_wait(
        ggml_opencl_htp_sidecar * sidecar,
        const char * cache_key,
        uint64_t generation,
        bool * waited,
        char * reason,
        size_t reason_capacity) {
    if (waited != nullptr) {
        *waited = false;
    }
    ggml_opencl_htp_cache_state state = GGML_OPENCL_HTP_CACHE_EMPTY;
    const ggml_opencl_htp_result query = ggml_backend_opencl_htp_cache_query(
        sidecar, cache_key, generation, &state, reason, reason_capacity);
    if (query != GGML_OPENCL_HTP_OK) {
        return query;
    }
    if (state == GGML_OPENCL_HTP_CACHE_READY) {
        set_reason(reason, reason_capacity, "ok");
        return GGML_OPENCL_HTP_OK;
    }
    if (state != GGML_OPENCL_HTP_CACHE_SUBMITTED) {
        set_reason(reason, reason_capacity, "cache generation has not been submitted");
        return GGML_OPENCL_HTP_UNSUPPORTED;
    }

    cl_event event = nullptr;
    {
        std::lock_guard<std::mutex> lock(sidecar->cache_mutex);
        ggml_opencl_htp_cache_record * record =
            find_cache_record(sidecar, cache_key, generation);
        if (record == nullptr || record->producer_event == nullptr) {
            set_reason(reason, reason_capacity, "cache producer event is unavailable");
            return GGML_OPENCL_HTP_ERROR;
        }
        event = record->producer_event;
        const cl_int retain_err = clRetainEvent(event);
        if (retain_err != CL_SUCCESS) {
            set_reason(reason, reason_capacity, "failed to retain cache producer event (error %d)", retain_err);
            return GGML_OPENCL_HTP_ERROR;
        }
    }
    const cl_int wait_err = clWaitForEvents(1, &event);
    const cl_int release_err = clReleaseEvent(event);
    if (wait_err != CL_SUCCESS || release_err != CL_SUCCESS) {
        set_reason(
            reason, reason_capacity,
            "cache producer wait failed (wait error %d, release error %d)",
            wait_err, release_err);
        return GGML_OPENCL_HTP_ERROR;
    }
    if (waited != nullptr) {
        *waited = true;
    }
    return ggml_backend_opencl_htp_cache_query(
        sidecar, cache_key, generation, &state, reason, reason_capacity);
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
        graph->sidecar->sparse_queue, alias->memory, &pattern, sizeof(pattern),
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
        const cl_int finish_err = clFinish(graph->sidecar->sparse_queue);
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

ggml_opencl_htp_result ggml_backend_opencl_htp_graph_stage_block_sparse_attn_mask(
    ggml_opencl_htp_graph * graph,
    const ggml_tensor * node,
    char * reason,
    size_t reason_capacity) {
    if (graph == nullptr || node == nullptr) {
        set_reason(reason, reason_capacity, "graph and node must not be null");
        return GGML_OPENCL_HTP_ERROR;
    }
    if (graph->state != ggml_opencl_htp_graph_state::active) {
        set_reason(reason, reason_capacity, "graph is already finished");
        return GGML_OPENCL_HTP_ERROR;
    }

    std::lock_guard<std::mutex> lock(graph->sidecar->dispatch_mutex);
    const auto prepared = std::find_if(
        graph->prepared_sparse_dispatches.begin(), graph->prepared_sparse_dispatches.end(),
        [node](const ggml_opencl_htp_prepared_sparse_dispatch & item) {
            return item.dispatch.output == node;
        });
    if (prepared == graph->prepared_sparse_dispatches.end()) {
        set_reason(reason, reason_capacity, "sparse attention node was not prepared");
        return GGML_OPENCL_HTP_UNSUPPORTED;
    }
    if (prepared->staged_mask != nullptr) {
        set_reason(reason, reason_capacity, "ok");
        return GGML_OPENCL_HTP_OK;
    }

    cl_int err = CL_SUCCESS;
    cl_mem staged_mask = clCreateBuffer(
        graph->sidecar->context, CL_MEM_READ_ONLY,
        prepared->bindings.mask.span, nullptr, &err);
    if (err != CL_SUCCESS || staged_mask == nullptr) {
        if (staged_mask != nullptr) {
            clReleaseMemObject(staged_mask);
        }
        return fail_runtime_graph(
            graph, reason, reason_capacity,
            "clCreateBuffer failed for early sparse mask staging (span %zu, error %d)",
            prepared->bindings.mask.span, err);
    }

    trace_sidecar_event("gpu_sparse_mask_stage_begin", node);
    cl_event copy_event = nullptr;
    err = clEnqueueCopyBuffer(
        graph->sidecar->sparse_queue,
        prepared->bindings.mask.memory,
        staged_mask,
        prepared->bindings.mask.offset,
        0,
        prepared->bindings.mask.span,
        0,
        nullptr,
        &copy_event);
    if (err == CL_SUCCESS && copy_event == nullptr) {
        err = CL_INVALID_EVENT;
    }
    if (err == CL_SUCCESS) {
        err = clFlush(graph->sidecar->sparse_queue);
    }
    if (err == CL_SUCCESS) {
        err = clWaitForEvents(1, &copy_event);
    }
    const cl_int event_release_err = copy_event != nullptr
        ? clReleaseEvent(copy_event) : CL_SUCCESS;
    if (err != CL_SUCCESS || event_release_err != CL_SUCCESS) {
        const cl_int memory_release_err = clReleaseMemObject(staged_mask);
        return fail_runtime_graph(
            graph, reason, reason_capacity,
            "failed to stage early sparse mask for producer %p "
            "(error %d, event release error %d, memory release error %d)",
            static_cast<const void *>(node), err,
            event_release_err, memory_release_err);
    }

    std::atomic_thread_fence(std::memory_order_acquire);
    prepared->staged_mask = staged_mask;
    trace_sidecar_event("gpu_sparse_mask_stage_end", node);
    set_reason(reason, reason_capacity, "ok");
    return GGML_OPENCL_HTP_OK;
}

static ggml_opencl_htp_result compute_block_sparse_attn_async_impl(
    ggml_opencl_htp_graph * graph,
    const ggml_tensor * node,
    bool stage_output,
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

    const bool trace_timing = sidecar_trace_enabled();
    const bool diagnostic_timing = trace_timing || graph->sidecar->profiling_enabled;
    const int64_t timing_begin_us = diagnostic_timing ? ggml_time_us() : 0;
    bool cached_dispatch = false;
    cl_mem staged_mask = nullptr;
    ggml_opencl_htp_sparse_dispatch dispatch;
    ggml_opencl_htp_sparse_bindings bindings;
    const auto prepared = std::find_if(
        graph->prepared_sparse_dispatches.begin(), graph->prepared_sparse_dispatches.end(),
        [node](const ggml_opencl_htp_prepared_sparse_dispatch & item) {
            return item.dispatch.output == node;
        });
    if (prepared != graph->prepared_sparse_dispatches.end()) {
        dispatch = prepared->dispatch;
        bindings = prepared->bindings;
        staged_mask = prepared->staged_mask;
        cached_dispatch = true;
        if (node->src[0] != dispatch.q || node->src[1] != dispatch.k ||
            node->src[2] != dispatch.v || node->src[3] != dispatch.mask ||
            !prepared_sparse_metadata_matches(*prepared, node)) {
            return fail_runtime_graph(
                graph, reason, reason_capacity,
                "preflight sparse dispatch tensor or metadata changed");
        }
    } else {
        const ggml_opencl_htp_result validation = validate_block_sparse_node(
            node, &dispatch, reason, reason_capacity);
        if (validation != GGML_OPENCL_HTP_OK) {
            return validation;
        }
    }
    const int64_t validation_end_us = diagnostic_timing ? ggml_time_us() : 0;

    if (!cached_dispatch) {
        const ggml_opencl_htp_result binding_result = resolve_sparse_bindings(
            graph, dispatch, &bindings, reason, reason_capacity);
        if (binding_result != GGML_OPENCL_HTP_OK) {
            return binding_result;
        }
    }
    const int64_t binding_end_us = diagnostic_timing ? ggml_time_us() : 0;

    static_assert(sizeof(size_t) <= sizeof(cl_ulong), "OpenCL offsets must hold host size_t values");
    const cl_mem q_memory = bindings.q.memory;
    const cl_mem k_memory = bindings.k.memory;
    const cl_mem v_memory = bindings.v.memory;
    cl_int staging_error = CL_SUCCESS;
    cl_mem staged_output = nullptr;
    if (stage_output) {
        staged_output = clCreateBuffer(
            graph->sidecar->context, CL_MEM_READ_WRITE,
            bindings.output.span, nullptr, &staging_error);
        if (staging_error != CL_SUCCESS || staged_output == nullptr) {
            if (staged_output != nullptr) {
                clReleaseMemObject(staged_output);
            }
            return fail_runtime_graph(
                graph, reason, reason_capacity,
                "clCreateBuffer failed for early sparse output staging "
                "(span %zu, error %d)",
                bindings.output.span, staging_error);
        }
    }
    struct staged_output_guard {
        cl_mem memory = nullptr;
        ~staged_output_guard() {
            if (memory != nullptr) {
                clReleaseMemObject(memory);
            }
        }
        cl_mem release() {
            cl_mem result = memory;
            memory = nullptr;
            return result;
        }
    } staging_guard = { staged_output };

    const cl_mem output_memory = stage_output ? staged_output : bindings.output.memory;
    const cl_mem mask_memory = stage_output && staged_mask != nullptr
        ? staged_mask : bindings.mask.memory;
    const cl_ulong q_offset = static_cast<cl_ulong>(bindings.q.offset);
    const cl_ulong k_offset = static_cast<cl_ulong>(bindings.k.offset);
    const cl_ulong v_offset = static_cast<cl_ulong>(bindings.v.offset);
    const cl_ulong output_offset = stage_output
        ? 0 : static_cast<cl_ulong>(bindings.output.offset);
    const cl_ulong mask_offset = stage_output && staged_mask != nullptr
        ? 0 : static_cast<cl_ulong>(bindings.mask.offset);
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
    const cl_ulong v_token_stride = static_cast<cl_ulong>(dispatch.v_token_stride);
    const cl_ulong v_head_stride = static_cast<cl_ulong>(dispatch.v_head_stride);
    const cl_ulong v_batch_stride = static_cast<cl_ulong>(dispatch.v_batch_stride);
    const cl_ulong output_nb1 = static_cast<cl_ulong>(dispatch.output->nb[1]);
    const cl_ulong output_nb2 = static_cast<cl_ulong>(dispatch.output->nb[2]);
    const cl_ulong output_nb3 = static_cast<cl_ulong>(dispatch.output->nb[3]);
    const cl_ulong mask_nb0 = static_cast<cl_ulong>(dispatch.mask->nb[0]);
    const cl_ulong mask_nb1 = static_cast<cl_ulong>(dispatch.mask->nb[1]);

    std::lock_guard<std::mutex> lock(graph->sidecar->dispatch_mutex);
    const auto pending = std::find_if(
        graph->pending_events.begin(), graph->pending_events.end(),
        [node](const ggml_opencl_htp_pending_event & item) {
            return item.producer == node;
        });
    if (pending != graph->pending_events.end()) {
        return fail_runtime_graph(
            graph, reason, reason_capacity,
            "sparse attention producer %p already has a pending event",
            static_cast<const void *>(node));
    }

    char kernel_reason[1024] = {};
    const int64_t head_dim = dispatch.q->ne[0];
    ggml_opencl_htp_block_sparse_kernel * selected_kernel = nullptr;
    const ggml_opencl_htp_result selection_result = select_block_sparse_kernel(
        graph->sidecar, head_dim, dispatch.mask_format, &selected_kernel,
        kernel_reason, sizeof(kernel_reason));
    if (selection_result == GGML_OPENCL_HTP_UNSUPPORTED) {
        set_reason(reason, reason_capacity, "%s", kernel_reason);
        return selection_result;
    }
    if (selection_result != GGML_OPENCL_HTP_OK) {
        return fail_runtime_graph(
            graph, reason, reason_capacity,
            "block-sparse kernel selection failed: %s", kernel_reason);
    }
    const ggml_opencl_htp_result kernel_result = ensure_block_sparse_kernel(
        graph->sidecar, selected_kernel, kernel_reason, sizeof(kernel_reason));
    if (kernel_result == GGML_OPENCL_HTP_UNSUPPORTED) {
        set_reason(reason, reason_capacity, "%s", kernel_reason);
        return kernel_result;
    }
    if (kernel_result != GGML_OPENCL_HTP_OK) {
        return fail_runtime_graph(
            graph, reason, reason_capacity,
            "block-sparse kernel creation failed: %s", kernel_reason);
    }

    cl_kernel block_sparse_kernel = selected_kernel->kernel;
    if (block_sparse_kernel == nullptr) {
        return fail_runtime_graph(
            graph, reason, reason_capacity,
            "block-sparse kernel for head dimension %lld is unavailable",
            (long long) head_dim);
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
        { sizeof(v_token_stride), &v_token_stride },
        { sizeof(v_head_stride), &v_head_stride },
        { sizeof(v_batch_stride), &v_batch_stride },
        { sizeof(output_nb1), &output_nb1 },
        { sizeof(output_nb2), &output_nb2 },
        { sizeof(output_nb3), &output_nb3 },
        { sizeof(mask_nb0), &mask_nb0 },
        { sizeof(mask_nb1), &mask_nb1 },
    };
    static_assert(sizeof(args) / sizeof(args[0]) == 31, "block-sparse kernel ABI changed");
    for (cl_uint index = 0; index < sizeof(args) / sizeof(args[0]); ++index) {
        const cl_int arg_err = clSetKernelArg(
            block_sparse_kernel, index, args[index].size, args[index].value);
        if (arg_err != CL_SUCCESS) {
            return fail_runtime_graph(
                graph, reason, reason_capacity,
                "clSetKernelArg(%u) failed (error %d)", index, arg_err);
        }
    }
    const int64_t kernel_args_end_us = diagnostic_timing ? ggml_time_us() : 0;

    const size_t local_work_size[] = { selected_kernel->work_group_size, 1 };
    const size_t global_work_size[] = {
        static_cast<size_t>(1 + (dispatch.lq - 1) / 32) *
            selected_kernel->work_group_size,
        static_cast<size_t>(dispatch.heads) * static_cast<size_t>(dispatch.batch),
    };
    cl_event event = nullptr;
    const bool profile_attention =
        ggml_backend_opencl_attention_profiling_enabled();
    if (profile_attention) {
        const cl_int finish_err = clFinish(graph->sidecar->sparse_queue);
        if (finish_err != CL_SUCCESS) {
            return fail_runtime_graph(
                graph, reason, reason_capacity,
                "clFinish failed before sparse attention profiling (error %d)",
                finish_err);
        }
    }
    const int64_t attention_begin_us =
        (profile_attention || graph->sidecar->profiling_enabled) ? ggml_time_us() : 0;
    std::atomic_thread_fence(std::memory_order_release);
    cl_int err = clEnqueueNDRangeKernel(
        graph->sidecar->sparse_queue, block_sparse_kernel,
        2, nullptr, global_work_size, local_work_size, 0, nullptr, &event);
    if (err != CL_SUCCESS || event == nullptr) {
        if (event != nullptr) {
            return fail_submitted_graph(
                graph, event, "clEnqueueNDRangeKernel", err, reason, reason_capacity);
        }
        return fail_runtime_graph(
            graph, reason, reason_capacity,
            "clEnqueueNDRangeKernel failed (error %d, no event)", err);
    }
    const int64_t enqueue_end_us = diagnostic_timing ? ggml_time_us() : 0;

    if (graph->test_fail_next_sparse_enqueue) {
        graph->test_fail_next_sparse_enqueue = false;
        return fail_submitted_graph(
            graph, event, "injected sparse enqueue failure",
            CL_INVALID_OPERATION, reason, reason_capacity);
    }

    try {
        ggml_opencl_htp_pending_event pending_event;
        pending_event.producer = node;
        pending_event.event = event;
        pending_event.staged_output = staging_guard.memory;
        pending_event.destination = bindings.output;
        pending_event.attention_begin_us = attention_begin_us;
        pending_event.validation_us = validation_end_us - timing_begin_us;
        pending_event.binding_us = binding_end_us - validation_end_us;
        pending_event.args_us = kernel_args_end_us - binding_end_us;
        pending_event.enqueue_us = enqueue_end_us - kernel_args_end_us;
        pending_event.record_attention_elapsed = profile_attention;
        graph->pending_events.push_back(pending_event);
        staging_guard.release();
    } catch (...) {
        return fail_submitted_graph(
            graph, event, "cache sparse producer event", CL_OUT_OF_HOST_MEMORY,
            reason, reason_capacity);
    }
    trace_sidecar_queue_event("gpu_sparse_enqueue", "sparse", node);

    err = clFlush(graph->sidecar->sparse_queue);
    if (err != CL_SUCCESS) {
        return fail_runtime_graph(
            graph, reason, reason_capacity,
            "clFlush failed after sparse enqueue (error %d)", err);
    }
    trace_sidecar_event("gpu_sparse_flush", node);
    if (diagnostic_timing) {
        const int64_t flush_end_us = ggml_time_us();
        graph->pending_events.back().flush_us = flush_end_us - enqueue_end_us;
        if (!trace_timing) {
            set_reason(reason, reason_capacity, "ok");
            return GGML_OPENCL_HTP_OK;
        }
        std::fprintf(
            stderr,
            "htp_opencl_submit_timing producer=%s head_dim=%lld cached=%d validation_us=%lld "
            "binding_us=%lld kernel_args_us=%lld enqueue_us=%lld flush_us=%lld total_us=%lld\n",
            node->name[0] != '\0' ? node->name : "<unnamed>",
            (long long) head_dim,
            cached_dispatch ? 1 : 0,
            (long long) (validation_end_us - timing_begin_us),
            (long long) (binding_end_us - validation_end_us),
            (long long) (kernel_args_end_us - binding_end_us),
            (long long) (enqueue_end_us - kernel_args_end_us),
            (long long) (flush_end_us - enqueue_end_us),
            (long long) (flush_end_us - timing_begin_us));
    }

    set_reason(reason, reason_capacity, "ok");
    return GGML_OPENCL_HTP_OK;
}

ggml_opencl_htp_result ggml_backend_opencl_htp_graph_compute_block_sparse_attn_async(
    ggml_opencl_htp_graph * graph,
    const ggml_tensor * node,
    char * reason,
    size_t reason_capacity) {
    return compute_block_sparse_attn_async_impl(
        graph, node, false, reason, reason_capacity);
}

ggml_opencl_htp_result ggml_backend_opencl_htp_graph_compute_block_sparse_attn_early_async(
    ggml_opencl_htp_graph * graph,
    const ggml_tensor * node,
    char * reason,
    size_t reason_capacity) {
    return compute_block_sparse_attn_async_impl(
        graph, node, true, reason, reason_capacity);
}

ggml_opencl_htp_result ggml_backend_opencl_htp_graph_start_submission_thread(
    ggml_opencl_htp_graph * graph,
    char * reason,
    size_t reason_capacity) {
    if (graph == nullptr) {
        set_reason(reason, reason_capacity, "graph must not be null");
        return GGML_OPENCL_HTP_ERROR;
    }

    std::lock_guard<std::mutex> lock(graph->submission_mutex);
    if (graph->state != ggml_opencl_htp_graph_state::active) {
        set_reason(reason, reason_capacity, "graph is already finished");
        return GGML_OPENCL_HTP_ERROR;
    }
    if (graph->submission_thread.joinable()) {
        set_reason(reason, reason_capacity, "submission thread is already running");
        return GGML_OPENCL_HTP_ERROR;
    }
    if (graph->routed_nodes.empty()) {
        set_reason(reason, reason_capacity, "graph has no routed sparse nodes");
        return GGML_OPENCL_HTP_UNSUPPORTED;
    }
    if (const char * fail_start =
            std::getenv("SD_HTP_SPARSE_OPENCL_TEST_FAIL_SUBMIT_THREAD_START");
        fail_start != nullptr && fail_start[0] != '\0' && std::strcmp(fail_start, "0") != 0) {
        set_reason(reason, reason_capacity, "injected OpenCL submission thread start failure");
        return GGML_OPENCL_HTP_ERROR;
    }

    graph->submission_stop_requested = false;
    graph->submission_cancel_requested = false;
    graph->async_failure_result = GGML_OPENCL_HTP_OK;
    graph->async_failure_reason[0] = '\0';
    graph->sparse_submission_jobs.clear();
    graph->residual_submission_jobs.clear();
    graph->submission_records.clear();
    try {
        graph->submission_records.reserve(graph->routed_nodes.size());
        graph->submission_thread = std::thread([graph]() {
            for (;;) {
                ggml_opencl_htp_submission_job job;
                {
                    std::unique_lock<std::mutex> lock(graph->submission_mutex);
                    graph->submission_cv.wait(lock, [graph]() {
                        return graph->submission_stop_requested ||
                            !graph->sparse_submission_jobs.empty() ||
                            !graph->residual_submission_jobs.empty();
                    });
                    const char * hold_residual = std::getenv(
                        "SD_HTP_SPARSE_OPENCL_TEST_HOLD_RESIDUAL_FOR_SPARSE");
                    while (!graph->submission_stop_requested &&
                           hold_residual != nullptr && hold_residual[0] != '\0' &&
                           std::strcmp(hold_residual, "0") != 0 &&
                           graph->sparse_submission_jobs.empty() &&
                           !graph->residual_submission_jobs.empty()) {
                        graph->submission_cv.wait(lock);
                    }
                    if (graph->sparse_submission_jobs.empty() &&
                        graph->residual_submission_jobs.empty()) {
                        if (graph->submission_stop_requested) {
                            return;
                        }
                        continue;
                    }
                    if (!graph->sparse_submission_jobs.empty()) {
                        job = graph->sparse_submission_jobs.front();
                        graph->sparse_submission_jobs.pop_front();
                    } else {
                        job = graph->residual_submission_jobs.front();
                        graph->residual_submission_jobs.pop_front();
                    }
                    const auto record = std::find_if(
                        graph->submission_records.begin(), graph->submission_records.end(),
                        [&job](const ggml_opencl_htp_submission_record & item) {
                            return item.kind == job.kind &&
                                   item.producer == job.producer;
                        });
                    if (record != graph->submission_records.end()) {
                        record->state = ggml_opencl_htp_submission_state::submitting;
                    }
                }

                trace_submission_scheduler_timing(job, ggml_time_us());

                char submit_reason[256] = {};
                ggml_opencl_htp_result result = GGML_OPENCL_HTP_ERROR;
                try {
                    const char * delay_value =
                        std::getenv("SD_HTP_SPARSE_OPENCL_TEST_SUBMIT_DELAY_US");
                    if (delay_value != nullptr && delay_value[0] != '\0') {
                        char * end = nullptr;
                        const long long delay_us = std::strtoll(delay_value, &end, 10);
                        if (end != delay_value && *end == '\0' && delay_us > 0 && delay_us <= 1000000) {
                            std::this_thread::sleep_for(std::chrono::microseconds(delay_us));
                        }
                    }
                    bool cancelled = false;
                    {
                        std::lock_guard<std::mutex> lock(graph->submission_mutex);
                        cancelled = graph->submission_cancel_requested;
                    }
                    if (cancelled) {
                        std::snprintf(
                            submit_reason, sizeof(submit_reason),
                            "OpenCL submission cancelled before enqueue");
                    } else {
                        if (job.kind ==
                            ggml_opencl_htp_submission_job_kind::sparse_attention) {
                            result =
                                ggml_backend_opencl_htp_graph_compute_block_sparse_attn_early_async(
                                    graph, job.producer,
                                    submit_reason, sizeof(submit_reason));
                        } else {
                            ggml_opencl_htp_prepared_residual_dispatch * prepared = nullptr;
                            for (ggml_opencl_htp_prepared_residual_dispatch & candidate :
                                    graph->prepared_residual_dispatches) {
                                if (candidate.dispatch.rho_destination == job.producer) {
                                    prepared = &candidate;
                                    break;
                                }
                            }
                            if (prepared == nullptr) {
                                std::snprintf(
                                    submit_reason, sizeof(submit_reason),
                                    "residual submission job is no longer prepared");
                            } else {
                                trace_sidecar_event(
                                    "gpu_residual_route",
                                    prepared->dispatch.residual_destination);
                                result =
                                    ggml_backend_opencl_htp_graph_queue_exact_rho_residual(
                                        graph,
                                        prepared->residual_cache_key.c_str(),
                                        prepared->dispatch.generation,
                                        submit_reason,
                                        sizeof(submit_reason));
                            }
                        }
                    }
                } catch (...) {
                    std::snprintf(
                        submit_reason, sizeof(submit_reason),
                        "unexpected exception in OpenCL submission thread");
                }

                {
                    std::lock_guard<std::mutex> lock(graph->submission_mutex);
                    const auto record = std::find_if(
                        graph->submission_records.begin(), graph->submission_records.end(),
                        [&job](const ggml_opencl_htp_submission_record & item) {
                            return item.kind == job.kind &&
                                   item.producer == job.producer;
                        });
                    if (record != graph->submission_records.end()) {
                        record->result = result;
                        record->state = result == GGML_OPENCL_HTP_OK
                            ? ggml_opencl_htp_submission_state::submitted
                            : ggml_opencl_htp_submission_state::failed;
                        std::snprintf(
                            record->reason, sizeof(record->reason), "%s",
                            submit_reason[0] != '\0' ? submit_reason : "OpenCL submission failed");
                    }
                    if (result != GGML_OPENCL_HTP_OK) {
                        graph->submission_stop_requested = true;
                        graph->submission_cancel_requested = true;
                        for (ggml_opencl_htp_submission_record & queued : graph->submission_records) {
                            if (queued.state != ggml_opencl_htp_submission_state::queued) {
                                continue;
                            }
                            queued.result = result;
                            queued.state = ggml_opencl_htp_submission_state::failed;
                            std::snprintf(
                                queued.reason, sizeof(queued.reason),
                                "submission thread stopped after producer failure: %s",
                                submit_reason[0] != '\0' ? submit_reason : "unknown error");
                        }
                        graph->sparse_submission_jobs.clear();
                        graph->residual_submission_jobs.clear();
                    }
                }
                graph->submission_cv.notify_all();
            }
        });
        graph->submission_worker_active = true;
    } catch (...) {
        graph->submission_worker_active = false;
        graph->submission_stop_requested = true;
        set_reason(reason, reason_capacity, "failed to start OpenCL submission thread");
        return GGML_OPENCL_HTP_ERROR;
    }

    set_reason(reason, reason_capacity, "ok");
    return GGML_OPENCL_HTP_OK;
}

ggml_opencl_htp_result ggml_backend_opencl_htp_graph_queue_block_sparse_attn(
    ggml_opencl_htp_graph * graph,
    const ggml_tensor * node,
    char * reason,
    size_t reason_capacity) {
    if (graph == nullptr || node == nullptr) {
        set_reason(reason, reason_capacity, "graph and node must not be null");
        return GGML_OPENCL_HTP_ERROR;
    }

    std::lock_guard<std::mutex> lock(graph->submission_mutex);
    if (graph->state != ggml_opencl_htp_graph_state::active) {
        set_reason(reason, reason_capacity, "graph is already finished");
        return GGML_OPENCL_HTP_ERROR;
    }
    if (!graph->submission_thread.joinable() || graph->submission_stop_requested) {
        set_reason(reason, reason_capacity, "submission thread is not accepting work");
        return GGML_OPENCL_HTP_ERROR;
    }
    if (std::find(graph->routed_nodes.begin(), graph->routed_nodes.end(), node) ==
        graph->routed_nodes.end()) {
        set_reason(reason, reason_capacity, "node is not routed to the OpenCL sidecar");
        return GGML_OPENCL_HTP_UNSUPPORTED;
    }
    const auto existing = std::find_if(
        graph->submission_records.begin(), graph->submission_records.end(),
        [node](const ggml_opencl_htp_submission_record & item) {
            return item.producer == node;
        });
    if (existing != graph->submission_records.end()) {
        set_reason(reason, reason_capacity, "sparse attention producer was already queued");
        return GGML_OPENCL_HTP_ERROR;
    }
    if (graph->submission_records.size() >= graph->routed_nodes.size()) {
        set_reason(reason, reason_capacity, "OpenCL submission queue capacity exceeded");
        return GGML_OPENCL_HTP_ERROR;
    }

    try {
        ggml_opencl_htp_submission_record record;
        record.kind = ggml_opencl_htp_submission_job_kind::sparse_attention;
        record.producer = node;
        std::snprintf(record.reason, sizeof(record.reason), "queued");
        graph->submission_records.push_back(record);
        graph->sparse_submission_jobs.push_back({
            ggml_opencl_htp_submission_job_kind::sparse_attention, node,
            sidecar_trace_enabled() ? ggml_time_us() : 0,
        });
    } catch (...) {
        const auto record = std::find_if(
            graph->submission_records.begin(), graph->submission_records.end(),
            [node](const ggml_opencl_htp_submission_record & item) {
                return item.producer == node;
            });
        if (record != graph->submission_records.end()) {
            graph->submission_records.erase(record);
        }
        set_reason(reason, reason_capacity, "failed to publish OpenCL submission job");
        return GGML_OPENCL_HTP_ERROR;
    }

    trace_sidecar_event("gpu_sparse_submit_job", node);
    graph->submission_cv.notify_one();
    set_reason(reason, reason_capacity, "ok");
    return GGML_OPENCL_HTP_OK;
}

ggml_opencl_htp_result
ggml_backend_opencl_htp_graph_queue_exact_rho_residual_after_dense(
        ggml_opencl_htp_graph * graph,
        const ggml_tensor * dense_producer,
        char * reason,
        size_t reason_capacity) {
    if (graph == nullptr || dense_producer == nullptr) {
        set_reason(reason, reason_capacity,
                   "graph and dense producer must not be null");
        return GGML_OPENCL_HTP_ERROR;
    }

    std::lock_guard<std::mutex> lock(graph->submission_mutex);
    if (graph->state != ggml_opencl_htp_graph_state::active) {
        set_reason(reason, reason_capacity, "graph is already finished");
        return GGML_OPENCL_HTP_ERROR;
    }
    if (!graph->submission_thread.joinable() || graph->submission_stop_requested) {
        set_reason(reason, reason_capacity,
                   "submission thread is not accepting residual work");
        return GGML_OPENCL_HTP_ERROR;
    }

    size_t queued_count = 0;
    for (ggml_opencl_htp_prepared_residual_dispatch & prepared :
            graph->prepared_residual_dispatches) {
        if (prepared.dispatch.dense_packed != dense_producer) {
            continue;
        }
        const ggml_tensor * producer = prepared.dispatch.rho_destination;
        const auto existing = std::find_if(
            graph->submission_records.begin(), graph->submission_records.end(),
            [producer](const ggml_opencl_htp_submission_record & item) {
                return item.kind ==
                           ggml_opencl_htp_submission_job_kind::exact_rho_residual &&
                       item.producer == producer;
            });
        if (existing != graph->submission_records.end()) {
            continue;
        }
        if (graph->submission_records.size() >= graph->routed_nodes.size()) {
            set_reason(reason, reason_capacity,
                       "OpenCL submission queue capacity exceeded");
            return GGML_OPENCL_HTP_ERROR;
        }
        try {
            ggml_opencl_htp_submission_record record;
            record.kind =
                ggml_opencl_htp_submission_job_kind::exact_rho_residual;
            record.producer = producer;
            std::snprintf(record.reason, sizeof(record.reason), "queued");
            graph->submission_records.push_back(record);
            graph->residual_submission_jobs.push_back({
                ggml_opencl_htp_submission_job_kind::exact_rho_residual,
                producer,
                sidecar_trace_enabled() ? ggml_time_us() : 0,
            });
        } catch (...) {
            const auto record = std::find_if(
                graph->submission_records.begin(), graph->submission_records.end(),
                [producer](const ggml_opencl_htp_submission_record & item) {
                    return item.kind ==
                               ggml_opencl_htp_submission_job_kind::exact_rho_residual &&
                           item.producer == producer;
                });
            if (record != graph->submission_records.end()) {
                graph->submission_records.erase(record);
            }
            set_reason(reason, reason_capacity,
                       "failed to publish residual submission job");
            return GGML_OPENCL_HTP_ERROR;
        }
        ++queued_count;
        trace_sidecar_event("gpu_residual_submit_job", producer);
    }
    if (queued_count == 0) {
        set_reason(reason, reason_capacity,
                   "dense producer has no unqueued residual jobs");
        return GGML_OPENCL_HTP_UNSUPPORTED;
    }
    graph->submission_cv.notify_one();
    set_reason(reason, reason_capacity, "ok");
    return GGML_OPENCL_HTP_OK;
}

ggml_opencl_htp_result ggml_backend_opencl_htp_graph_stop_submission_thread(
    ggml_opencl_htp_graph * graph,
    char * reason,
    size_t reason_capacity) {
    if (graph == nullptr) {
        set_reason(reason, reason_capacity, "graph must not be null");
        return GGML_OPENCL_HTP_ERROR;
    }

    std::thread worker;
    {
        std::lock_guard<std::mutex> lock(graph->submission_mutex);
        if (!graph->submission_worker_active) {
            GGML_ASSERT(!graph->submission_thread.joinable());
            set_reason(reason, reason_capacity, "ok");
            return GGML_OPENCL_HTP_OK;
        }
        GGML_ASSERT(graph->submission_thread.joinable());
        if (graph->submission_thread.get_id() == std::this_thread::get_id()) {
            set_reason(reason, reason_capacity, "submission thread cannot join itself");
            return GGML_OPENCL_HTP_ERROR;
        }
        graph->submission_stop_requested = true;
        worker = std::move(graph->submission_thread);
    }
    graph->submission_cv.notify_all();
    worker.join();

    std::lock_guard<std::mutex> lock(graph->submission_mutex);
    graph->submission_worker_active = false;
    if (graph->async_failure_result != GGML_OPENCL_HTP_OK) {
        set_reason(reason, reason_capacity, "%s", graph->async_failure_reason);
        return graph->async_failure_result;
    }
    const auto failed = std::find_if(
        graph->submission_records.begin(), graph->submission_records.end(),
        [](const ggml_opencl_htp_submission_record & item) {
            return item.state == ggml_opencl_htp_submission_state::failed;
        });
    if (failed != graph->submission_records.end()) {
        set_reason(reason, reason_capacity, "%s", failed->reason);
        return failed->result == GGML_OPENCL_HTP_OK
            ? GGML_OPENCL_HTP_ERROR : failed->result;
    }
    set_reason(reason, reason_capacity, "ok");
    return GGML_OPENCL_HTP_OK;
}

ggml_opencl_htp_result ggml_backend_opencl_htp_graph_cancel_submission_thread(
    ggml_opencl_htp_graph * graph,
    char * reason,
    size_t reason_capacity) {
    if (graph == nullptr) {
        set_reason(reason, reason_capacity, "graph must not be null");
        return GGML_OPENCL_HTP_ERROR;
    }

    {
        std::lock_guard<std::mutex> lock(graph->submission_mutex);
        if (!graph->submission_worker_active) {
            set_reason(reason, reason_capacity, "ok");
            return GGML_OPENCL_HTP_OK;
        }
        graph->submission_cancel_requested = true;
        graph->submission_stop_requested = true;
        for (ggml_opencl_htp_submission_record & queued : graph->submission_records) {
            if (queued.state != ggml_opencl_htp_submission_state::queued) {
                continue;
            }
            queued.result = GGML_OPENCL_HTP_ERROR;
            queued.state = ggml_opencl_htp_submission_state::failed;
            std::snprintf(
                queued.reason, sizeof(queued.reason),
                "OpenCL submission cancelled before enqueue");
        }
        graph->sparse_submission_jobs.clear();
        graph->residual_submission_jobs.clear();
    }
    graph->submission_cv.notify_all();
    set_reason(reason, reason_capacity, "ok");
    return GGML_OPENCL_HTP_OK;
}

bool ggml_backend_opencl_htp_graph_sparse_output_is_residual_owned(
        const ggml_opencl_htp_graph * graph,
        const ggml_tensor * sparse_producer) {
    return graph != nullptr && sparse_producer != nullptr &&
        std::find(
            graph->residual_owned_sparse_producers.begin(),
            graph->residual_owned_sparse_producers.end(),
            sparse_producer) != graph->residual_owned_sparse_producers.end();
}

ggml_opencl_htp_result
ggml_backend_opencl_htp_graph_wait_for_sparse_input_lifetime(
        ggml_opencl_htp_graph * graph,
        const ggml_tensor * sparse_producer,
        bool * waited,
        char * reason,
        size_t reason_capacity) {
    if (waited != nullptr) {
        *waited = false;
    }
    if (!ggml_backend_opencl_htp_graph_sparse_output_is_residual_owned(
            graph, sparse_producer)) {
        set_reason(reason, reason_capacity,
                   "sparse producer is not residual-owned");
        return GGML_OPENCL_HTP_UNSUPPORTED;
    }
    {
        std::unique_lock<std::mutex> lock(graph->submission_mutex);
        auto find_submission = [graph, sparse_producer]() {
            return std::find_if(
                graph->submission_records.begin(),
                graph->submission_records.end(),
                [sparse_producer](const ggml_opencl_htp_submission_record & item) {
                    return item.kind ==
                               ggml_opencl_htp_submission_job_kind::sparse_attention &&
                           item.producer == sparse_producer;
                });
        };
        auto submission = find_submission();
        if (submission != graph->submission_records.end()) {
            graph->submission_cv.wait(lock, [&find_submission, graph]() {
                const auto current = find_submission();
                GGML_ASSERT(current != graph->submission_records.end());
                return current->state == ggml_opencl_htp_submission_state::submitted ||
                       current->state == ggml_opencl_htp_submission_state::failed;
            });
            submission = find_submission();
            if (submission->state == ggml_opencl_htp_submission_state::failed) {
                set_reason(reason, reason_capacity, "%s", submission->reason);
                return submission->result == GGML_OPENCL_HTP_OK ?
                    GGML_OPENCL_HTP_ERROR : submission->result;
            }
        }
    }

    cl_event event = nullptr;
    {
        std::lock_guard<std::mutex> lock(graph->sidecar->dispatch_mutex);
        const auto pending = std::find_if(
            graph->pending_events.begin(), graph->pending_events.end(),
            [sparse_producer](const ggml_opencl_htp_pending_event & item) {
                return item.producer == sparse_producer;
            });
        if (pending == graph->pending_events.end() || pending->event == nullptr) {
            set_reason(reason, reason_capacity,
                       "residual-owned sparse event is unavailable");
            return GGML_OPENCL_HTP_ERROR;
        }
        event = pending->event;
        const cl_int retain_err = clRetainEvent(event);
        if (retain_err != CL_SUCCESS) {
            set_reason(reason, reason_capacity,
                       "failed to retain sparse input lifetime event (error %d)",
                       retain_err);
            return GGML_OPENCL_HTP_ERROR;
        }
    }

    cl_int execution_status = CL_QUEUED;
    cl_int err = clGetEventInfo(
        event, CL_EVENT_COMMAND_EXECUTION_STATUS,
        sizeof(execution_status), &execution_status, nullptr);
    if (err == CL_SUCCESS && execution_status != CL_COMPLETE) {
        trace_sidecar_event("gpu_sparse_input_lifetime_wait_begin", sparse_producer);
        err = clWaitForEvents(1, &event);
        trace_sidecar_event("gpu_sparse_input_lifetime_wait_end", sparse_producer);
        if (waited != nullptr) {
            *waited = true;
        }
    }
    const cl_int release_err = clReleaseEvent(event);
    if (err != CL_SUCCESS || release_err != CL_SUCCESS) {
        set_reason(reason, reason_capacity,
                   "sparse input lifetime wait failed (error %d, release error %d)",
                   err, release_err);
        return GGML_OPENCL_HTP_ERROR;
    }
    set_reason(reason, reason_capacity, "ok");
    return GGML_OPENCL_HTP_OK;
}

ggml_opencl_htp_result ggml_backend_opencl_htp_graph_wait_for_tensor(
    ggml_opencl_htp_graph * graph,
    const ggml_tensor * producer,
    bool * waited,
    char * reason,
    size_t reason_capacity) {
    if (waited != nullptr) {
        *waited = false;
    }
    if (graph == nullptr || producer == nullptr) {
        set_reason(reason, reason_capacity, "graph and producer must not be null");
        return GGML_OPENCL_HTP_ERROR;
    }
    {
        std::unique_lock<std::mutex> lock(graph->submission_mutex);
        auto find_submission = [graph, producer]() {
            return std::find_if(
                graph->submission_records.begin(), graph->submission_records.end(),
                [producer](const ggml_opencl_htp_submission_record & item) {
                    return item.producer == producer;
                });
        };
        auto submission = find_submission();
        if (submission != graph->submission_records.end()) {
            const bool waits_for_publication =
                submission->state == ggml_opencl_htp_submission_state::queued ||
                submission->state == ggml_opencl_htp_submission_state::submitting;
            if (waits_for_publication) {
                trace_sidecar_event("gpu_sparse_publication_wait_begin", producer);
            }
            graph->submission_cv.wait(lock, [&find_submission, graph]() {
                const auto current = find_submission();
                GGML_ASSERT(current != graph->submission_records.end());
                return current->state == ggml_opencl_htp_submission_state::submitted ||
                       current->state == ggml_opencl_htp_submission_state::failed;
            });
            submission = find_submission();
            if (waits_for_publication) {
                trace_sidecar_event("gpu_sparse_publication_wait_end", producer);
            }
            if (submission->state == ggml_opencl_htp_submission_state::failed) {
                set_reason(reason, reason_capacity, "%s", submission->reason);
                return submission->result == GGML_OPENCL_HTP_OK
                    ? GGML_OPENCL_HTP_ERROR : submission->result;
            }
        }
        if (graph->state != ggml_opencl_htp_graph_state::active) {
            set_reason(reason, reason_capacity, "graph is already finished");
            return GGML_OPENCL_HTP_ERROR;
        }
    }

    for (ggml_opencl_htp_prepared_residual_dispatch & prepared :
            graph->prepared_residual_dispatches) {
        const bool is_residual =
            prepared.dispatch.residual_destination == producer;
        const bool is_rho = prepared.dispatch.rho_destination == producer;
        if (!is_residual && !is_rho) {
            continue;
        }
        const std::string & key = is_residual ?
            prepared.residual_cache_key : prepared.rho_cache_key;
        return ggml_backend_opencl_htp_cache_wait(
            graph->sidecar,
            key.c_str(),
            prepared.dispatch.generation,
            waited,
            reason,
            reason_capacity);
    }

    cl_event pending_event = nullptr;
    cl_mem staged_output = nullptr;
    ggml_opencl_htp_tensor_binding destination;
    int64_t attention_begin_us = 0;
    int64_t validation_us = 0;
    int64_t binding_us = 0;
    int64_t args_us = 0;
    int64_t enqueue_us = 0;
    int64_t flush_us = 0;
    bool record_attention_elapsed = false;
    cl_int retain_err = CL_SUCCESS;
    bool inject_wait_failure = false;
    {
        std::lock_guard<std::mutex> lock(graph->sidecar->dispatch_mutex);
        const auto pending = std::find_if(
            graph->pending_events.begin(), graph->pending_events.end(),
            [producer](const ggml_opencl_htp_pending_event & item) {
                return item.producer == producer;
            });
        if (pending == graph->pending_events.end()) {
            set_reason(reason, reason_capacity, "ok");
            return GGML_OPENCL_HTP_OK;
        }
        pending_event = pending->event;
        staged_output = pending->staged_output;
        destination = pending->destination;
        attention_begin_us = pending->attention_begin_us;
        validation_us = pending->validation_us;
        binding_us = pending->binding_us;
        args_us = pending->args_us;
        enqueue_us = pending->enqueue_us;
        flush_us = pending->flush_us;
        record_attention_elapsed = pending->record_attention_elapsed;
        inject_wait_failure = graph->test_fail_next_sparse_wait;
        graph->test_fail_next_sparse_wait = false;
        if (!inject_wait_failure) {
            retain_err = clRetainEvent(pending_event);
        }
    }

    if (inject_wait_failure) {
        return fail_runtime_graph(
            graph, reason, reason_capacity,
            "injected sparse wait failure after submission");
    }
    if (retain_err != CL_SUCCESS) {
        return fail_runtime_graph(
            graph, reason, reason_capacity,
            "clRetainEvent failed for sparse producer %p (error %d)",
            static_cast<const void *>(producer), retain_err);
    }

    if (const char * wait_delay =
            std::getenv("SD_HTP_SPARSE_OPENCL_TEST_WAIT_DELAY_US");
        wait_delay != nullptr && wait_delay[0] != '\0') {
        char * end = nullptr;
        const long long delay_us = std::strtoll(wait_delay, &end, 10);
        if (end != wait_delay && *end == '\0' && delay_us > 0 && delay_us <= 1000000) {
            std::this_thread::sleep_for(std::chrono::microseconds(delay_us));
        }
    }

    trace_sidecar_event("gpu_sparse_wait_begin", producer);
    const int64_t wait_begin_us = graph->sidecar->profiling_enabled ? ggml_time_us() : 0;
    const cl_int wait_err = clWaitForEvents(1, &pending_event);
    const int64_t wait_end_us = graph->sidecar->profiling_enabled ? ggml_time_us() : 0;
    if (wait_err != CL_SUCCESS) {
        const cl_int release_err = clReleaseEvent(pending_event);
        return fail_runtime_graph(
            graph, reason, reason_capacity,
            "clWaitForEvents failed for sparse producer %p "
            "(error %d, retained event release error %d)",
            static_cast<const void *>(producer), wait_err, release_err);
    }
    if (record_attention_elapsed && attention_begin_us > 0) {
        ggml_backend_opencl_record_attention_elapsed_us(
            wait_end_us > 0 ? wait_end_us - attention_begin_us : ggml_time_us() - attention_begin_us);
    }
    if (graph->sidecar->profiling_enabled) {
        ggml_opencl_attention_profile profile = {};
        profile.host_total_us = attention_begin_us > 0 ? wait_end_us - attention_begin_us : 0;
        profile.validation_us = validation_us;
        profile.binding_us = binding_us;
        profile.args_us = args_us;
        profile.enqueue_us = enqueue_us;
        profile.flush_us = flush_us;
        profile.wait_us = wait_end_us - wait_begin_us;
        if (!read_sparse_event_timestamps(pending_event, &profile)) {
            const cl_int release_err = clReleaseEvent(pending_event);
            return fail_runtime_graph(
                graph, reason, reason_capacity,
                "failed to read sparse event profiling timestamps for producer %p "
                "(retained event release error %d)",
                static_cast<const void *>(producer), release_err);
        }
        ggml_backend_opencl_record_attention_profile(&profile);
    }

    cl_event publish_event = nullptr;
    if (staged_output != nullptr) {
        trace_sidecar_event("gpu_sparse_publish_begin", producer);
        cl_int publish_err = clEnqueueCopyBuffer(
            graph->sidecar->sparse_queue,
            staged_output,
            destination.memory,
            0,
            destination.offset,
            destination.span,
            0,
            nullptr,
            &publish_event);
        if (publish_err == CL_SUCCESS && publish_event == nullptr) {
            publish_err = CL_INVALID_EVENT;
        }
        if (publish_err == CL_SUCCESS) {
            publish_err = clFlush(graph->sidecar->sparse_queue);
        }
        if (publish_err == CL_SUCCESS) {
            publish_err = clWaitForEvents(1, &publish_event);
        }
        if (publish_err != CL_SUCCESS) {
            const cl_int publish_release_err = publish_event != nullptr
                ? clReleaseEvent(publish_event) : CL_SUCCESS;
            const cl_int retained_release_err = clReleaseEvent(pending_event);
            return fail_runtime_graph(
                graph, reason, reason_capacity,
                "failed to publish staged sparse output for producer %p "
                "(error %d, publish event release error %d, "
                "retained event release error %d)",
                static_cast<const void *>(producer), publish_err,
                publish_release_err, retained_release_err);
        }
        trace_sidecar_event("gpu_sparse_publish_end", producer);
    }

    bool removed_pending_event = false;
    {
        std::lock_guard<std::mutex> lock(graph->sidecar->dispatch_mutex);
        const auto pending = std::find_if(
            graph->pending_events.begin(), graph->pending_events.end(),
            [producer, pending_event](const ggml_opencl_htp_pending_event & item) {
                return item.producer == producer && item.event == pending_event;
            });
        if (pending != graph->pending_events.end()) {
            graph->pending_events.erase(pending);
            removed_pending_event = true;
        }
    }
    const cl_int owner_release_err =
        removed_pending_event ? clReleaseEvent(pending_event) : CL_SUCCESS;
    const cl_int retained_release_err = clReleaseEvent(pending_event);
    const cl_int publish_release_err = publish_event != nullptr
        ? clReleaseEvent(publish_event) : CL_SUCCESS;
    const cl_int staging_release_err = removed_pending_event && staged_output != nullptr
        ? clReleaseMemObject(staged_output) : CL_SUCCESS;
    std::atomic_thread_fence(std::memory_order_acquire);
    if (!removed_pending_event || owner_release_err != CL_SUCCESS ||
        retained_release_err != CL_SUCCESS || publish_release_err != CL_SUCCESS ||
        staging_release_err != CL_SUCCESS) {
        return fail_runtime_graph(
            graph, reason, reason_capacity,
            "sparse producer %p event completion cleanup failed "
            "(pending=%d, owner release error %d, retained release error %d, "
            "publish release error %d, staging release error %d)",
            static_cast<const void *>(producer), removed_pending_event ? 1 : 0,
            owner_release_err, retained_release_err,
            publish_release_err, staging_release_err);
    }
    if (waited != nullptr) {
        *waited = true;
    }
    trace_sidecar_event("gpu_sparse_wait_end", producer);

    set_reason(reason, reason_capacity, "ok");
    return GGML_OPENCL_HTP_OK;
}

ggml_opencl_htp_result ggml_backend_opencl_htp_graph_compute_block_sparse_attn_sync(
    ggml_opencl_htp_graph * graph,
    const ggml_tensor * node,
    char * reason,
    size_t reason_capacity) {
    const ggml_opencl_htp_result enqueue_result =
        ggml_backend_opencl_htp_graph_compute_block_sparse_attn_async(
            graph, node, reason, reason_capacity);
    if (enqueue_result != GGML_OPENCL_HTP_OK) {
        return enqueue_result;
    }
    bool waited = false;
    return ggml_backend_opencl_htp_graph_wait_for_tensor(
        graph, node, &waited, reason, reason_capacity);
}

namespace {

ggml_opencl_htp_result query_deferred_graph_ready(
        ggml_opencl_htp_graph * graph,
        bool include_graph_events,
        bool * ready,
        char * reason,
        size_t reason_capacity) {
    if (graph == nullptr || ready == nullptr) {
        set_reason(reason, reason_capacity, "graph and ready output must be valid");
        return GGML_OPENCL_HTP_ERROR;
    }
    *ready = true;
    for (ggml_opencl_htp_prepared_residual_dispatch & prepared :
            graph->prepared_residual_dispatches) {
        ggml_opencl_htp_cache_state state = GGML_OPENCL_HTP_CACHE_EMPTY;
        const ggml_opencl_htp_result query = ggml_backend_opencl_htp_cache_query(
            graph->sidecar,
            prepared.residual_cache_key.c_str(),
            prepared.dispatch.generation,
            &state,
            reason,
            reason_capacity);
        if (query != GGML_OPENCL_HTP_OK) {
            return query;
        }
        if (state != GGML_OPENCL_HTP_CACHE_READY) {
            *ready = false;
        }
    }
    if (!include_graph_events) {
        set_reason(reason, reason_capacity, "ok");
        return GGML_OPENCL_HTP_OK;
    }

    auto query_event = [&](cl_event event) {
        if (event == nullptr) {
            return GGML_OPENCL_HTP_OK;
        }
        cl_int status = CL_QUEUED;
        const cl_int err = clGetEventInfo(
            event, CL_EVENT_COMMAND_EXECUTION_STATUS,
            sizeof(status), &status, nullptr);
        if (err != CL_SUCCESS || status < 0) {
            set_reason(
                reason, reason_capacity,
                "deferred graph event query failed (error %d, status %d)",
                err, status);
            return GGML_OPENCL_HTP_ERROR;
        }
        if (status != CL_COMPLETE) {
            *ready = false;
        }
        return GGML_OPENCL_HTP_OK;
    };
    for (cl_event event : graph->events) {
        if (query_event(event) != GGML_OPENCL_HTP_OK) {
            return GGML_OPENCL_HTP_ERROR;
        }
    }
    for (const ggml_opencl_htp_pending_event & pending : graph->pending_events) {
        if (query_event(pending.event) != GGML_OPENCL_HTP_OK) {
            return GGML_OPENCL_HTP_ERROR;
        }
    }
    set_reason(reason, reason_capacity, "ok");
    return GGML_OPENCL_HTP_OK;
}

} // namespace

ggml_opencl_htp_result ggml_backend_opencl_htp_graph_defer(
        ggml_opencl_htp_graph * graph,
        bool * pending,
        char * reason,
        size_t reason_capacity) {
    if (graph == nullptr || pending == nullptr) {
        set_reason(reason, reason_capacity, "graph and pending output must be valid");
        return GGML_OPENCL_HTP_ERROR;
    }
    *pending = false;
    char stop_reason[256] = {};
    const ggml_opencl_htp_result stop =
        ggml_backend_opencl_htp_graph_stop_submission_thread(
            graph, stop_reason, sizeof(stop_reason));
    if (stop != GGML_OPENCL_HTP_OK) {
        set_reason(reason, reason_capacity, "%s", stop_reason);
        return stop;
    }
    {
        std::lock_guard<std::mutex> lock(graph->submission_mutex);
        if (graph->submission_cancel_requested) {
            set_reason(reason, reason_capacity, "submission was cancelled");
            return GGML_OPENCL_HTP_UNSUPPORTED;
        }
        if (graph->async_failure_result != GGML_OPENCL_HTP_OK) {
            set_reason(reason, reason_capacity, "%s", graph->async_failure_reason);
            return graph->async_failure_result;
        }
    }
    bool ready = true;
    const ggml_opencl_htp_result query = query_deferred_graph_ready(
        graph, false, &ready, reason, reason_capacity);
    if (query != GGML_OPENCL_HTP_OK) {
        return query;
    }
    // Keep the graph alive whenever OpenCL sparse work still owns aliases,
    // even if the residual cache is already READY. The HTP runner must reap
    // QKV aliases from the sparse completion path rather than letting the
    // dense graph finish release them opportunistically.
    const bool sparse_pending =
        ggml_backend_opencl_htp_graph_pending_sparse_submission_count(graph) != 0 ||
        ggml_backend_opencl_htp_graph_pending_sparse_event_count(graph) != 0 ||
        ggml_backend_opencl_htp_graph_submitted_sparse_count(graph) != 0;
    *pending = sparse_pending ||
        (!graph->prepared_residual_dispatches.empty() && !ready);
    set_reason(reason, reason_capacity, "ok");
    return GGML_OPENCL_HTP_OK;
}

ggml_opencl_htp_result ggml_backend_opencl_htp_graph_deferred_ready(
        ggml_opencl_htp_graph * graph,
        bool * ready,
        char * reason,
        size_t reason_capacity) {
    return query_deferred_graph_ready(
        graph, true, ready, reason, reason_capacity);
}

ggml_opencl_htp_result ggml_backend_opencl_htp_graph_reap_qkv_aliases(
        ggml_opencl_htp_graph * graph,
        bool wait_for_sparse,
        bool * released,
        char * reason,
        size_t reason_capacity) {
    if (graph == nullptr || released == nullptr) {
        set_reason(reason, reason_capacity, "graph and released output must be valid");
        return GGML_OPENCL_HTP_ERROR;
    }
    *released = false;
    if (graph->qkv_aliases_released) {
        *released = true;
        set_reason(reason, reason_capacity, "already released");
        return GGML_OPENCL_HTP_OK;
    }
    {
        std::unique_lock<std::mutex> lock(graph->submission_mutex);
        auto sparse_submission_done = [graph]() {
            for (const ggml_opencl_htp_submission_record & record :
                    graph->submission_records) {
                if (record.kind != ggml_opencl_htp_submission_job_kind::sparse_attention) {
                    continue;
                }
                if (record.state == ggml_opencl_htp_submission_state::failed) {
                    return true;
                }
                if (record.state != ggml_opencl_htp_submission_state::submitted) {
                    return false;
                }
            }
            return true;
        };
        if (wait_for_sparse) {
            graph->submission_cv.wait(lock, sparse_submission_done);
        } else if (!sparse_submission_done()) {
            set_reason(reason, reason_capacity, "sparse submission is not complete");
            return GGML_OPENCL_HTP_OK;
        }
        for (const ggml_opencl_htp_submission_record & record :
                graph->submission_records) {
            if (record.kind == ggml_opencl_htp_submission_job_kind::sparse_attention &&
                record.state == ggml_opencl_htp_submission_state::failed) {
                set_reason(reason, reason_capacity, "%s", record.reason);
                return record.result == GGML_OPENCL_HTP_OK
                    ? GGML_OPENCL_HTP_ERROR : record.result;
            }
        }
    }

    std::lock_guard<std::mutex> lock(graph->sidecar->dispatch_mutex);
    for (const ggml_opencl_htp_pending_event & pending : graph->pending_events) {
        if (pending.event == nullptr) {
            continue;
        }
        if (wait_for_sparse) {
            const cl_int wait_err = clWaitForEvents(1, &pending.event);
            if (wait_err != CL_SUCCESS) {
                set_reason(reason, reason_capacity,
                           "sparse event wait failed (error %d)", wait_err);
                return GGML_OPENCL_HTP_ERROR;
            }
            trace_sparse_event_timing(
                graph->sidecar, pending.event, pending.producer);
        } else {
            cl_int status = CL_QUEUED;
            const cl_int err = clGetEventInfo(
                pending.event, CL_EVENT_COMMAND_EXECUTION_STATUS,
                sizeof(status), &status, nullptr);
            if (err != CL_SUCCESS || status < 0) {
                set_reason(reason, reason_capacity,
                           "sparse event query failed (error %d, status %d)", err, status);
                return GGML_OPENCL_HTP_ERROR;
            }
            if (status != CL_COMPLETE) {
                set_reason(reason, reason_capacity, "sparse event is pending");
                return GGML_OPENCL_HTP_OK;
            }
            trace_sparse_event_timing(
                graph->sidecar, pending.event, pending.producer);
        }
    }
    bool found = false;
    for (auto it = graph->aliases.begin(); it != graph->aliases.end();) {
        ggml_backend_buffer_type_t buft =
            ggml_backend_buffer_get_type(it->descriptor.backing);
        if (std::strcmp(ggml_backend_buft_name(buft), "RPCMEM-QKV") != 0) {
            ++it;
            continue;
        }
        found = true;
        const cl_int err = clReleaseMemObject(it->memory);
        if (err != CL_SUCCESS) {
            set_reason(reason, reason_capacity,
                       "QKV alias release failed (error %d)", err);
            return GGML_OPENCL_HTP_ERROR;
        }
        it = graph->aliases.erase(it);
    }
    graph->qkv_aliases_released = true;
    *released = true;
    if (found) {
        trace_sidecar_event("gpu_qkv_alias_release", nullptr);
    }
    set_reason(reason, reason_capacity, found ? "ok" : "no QKV aliases");
    return GGML_OPENCL_HTP_OK;
}

ggml_opencl_htp_result ggml_backend_opencl_htp_graph_wait_deferred(
        ggml_opencl_htp_graph * graph,
        char * reason,
        size_t reason_capacity) {
    if (graph == nullptr) {
        set_reason(reason, reason_capacity, "graph must not be null");
        return GGML_OPENCL_HTP_ERROR;
    }
    for (ggml_opencl_htp_prepared_residual_dispatch & prepared :
            graph->prepared_residual_dispatches) {
        ggml_opencl_htp_cache_state state = GGML_OPENCL_HTP_CACHE_EMPTY;
        ggml_opencl_htp_result result = ggml_backend_opencl_htp_cache_query(
            graph->sidecar,
            prepared.residual_cache_key.c_str(),
            prepared.dispatch.generation,
            &state,
            reason,
            reason_capacity);
        if (result != GGML_OPENCL_HTP_OK) {
            return result;
        }
        if (state == GGML_OPENCL_HTP_CACHE_READY) {
            continue;
        }
        if (state != GGML_OPENCL_HTP_CACHE_SUBMITTED) {
            set_reason(reason, reason_capacity, "deferred residual was not submitted");
            return GGML_OPENCL_HTP_ERROR;
        }
        bool waited = false;
        result = ggml_backend_opencl_htp_cache_wait(
            graph->sidecar,
            prepared.residual_cache_key.c_str(),
            prepared.dispatch.generation,
            &waited,
            reason,
            reason_capacity);
        if (result != GGML_OPENCL_HTP_OK) {
            return result;
        }
    }

    std::vector<cl_event> events;
    events.reserve(graph->events.size() + graph->pending_events.size());
    for (cl_event event : graph->events) {
        if (event != nullptr) {
            events.push_back(event);
        }
    }
    for (const ggml_opencl_htp_pending_event & pending : graph->pending_events) {
        if (pending.event != nullptr) {
            events.push_back(pending.event);
        }
    }
    if (!events.empty()) {
        const cl_int err = clWaitForEvents(
            static_cast<cl_uint>(events.size()), events.data());
        if (err != CL_SUCCESS) {
            set_reason(reason, reason_capacity,
                       "deferred graph event wait failed (error %d)", err);
            return GGML_OPENCL_HTP_ERROR;
        }
    }
    for (const ggml_opencl_htp_pending_event & pending : graph->pending_events) {
        trace_sparse_event_timing(graph->sidecar, pending.event, pending.producer);
    }
    set_reason(reason, reason_capacity, "ok");
    return GGML_OPENCL_HTP_OK;
}

ggml_opencl_htp_result ggml_backend_opencl_htp_graph_reap_deferred(
        ggml_opencl_htp_graph * graph,
        char * reason,
        size_t reason_capacity) {
    bool ready = false;
    const ggml_opencl_htp_result query =
        ggml_backend_opencl_htp_graph_deferred_ready(
            graph, &ready, reason, reason_capacity);
    if (query != GGML_OPENCL_HTP_OK) {
        return query;
    }
    if (!ready) {
        set_reason(reason, reason_capacity, "deferred graph is still pending");
        return GGML_OPENCL_HTP_UNSUPPORTED;
    }
    return finish_graph_state(
        graph, GGML_OPENCL_HTP_OK, "ok", reason, reason_capacity, false);
}

ggml_opencl_htp_result ggml_backend_opencl_htp_graph_finish(
    ggml_opencl_htp_graph * graph,
    char * reason,
    size_t reason_capacity) {
    if (graph == nullptr) {
        set_reason(reason, reason_capacity, "graph must not be null");
        return GGML_OPENCL_HTP_ERROR;
    }
    char stop_reason[256] = {};
    const ggml_opencl_htp_result stop_result =
        ggml_backend_opencl_htp_graph_stop_submission_thread(
            graph, stop_reason, sizeof(stop_reason));
    ggml_opencl_htp_result residual_result = GGML_OPENCL_HTP_OK;
    char residual_reason[256] = {};
    bool traced_residual_wait = false;
    if (stop_result == GGML_OPENCL_HTP_OK) {
        for (ggml_opencl_htp_prepared_residual_dispatch & prepared :
                graph->prepared_residual_dispatches) {
            ggml_opencl_htp_cache_state state = GGML_OPENCL_HTP_CACHE_EMPTY;
            const ggml_opencl_htp_result query =
                ggml_backend_opencl_htp_cache_query(
                    graph->sidecar,
                    prepared.residual_cache_key.c_str(),
                    prepared.dispatch.generation,
                    &state,
                    residual_reason,
                    sizeof(residual_reason));
            if (query != GGML_OPENCL_HTP_OK) {
                residual_result = query;
                break;
            }
            if (state == GGML_OPENCL_HTP_CACHE_READY) {
                continue;
            }
            if (state != GGML_OPENCL_HTP_CACHE_SUBMITTED) {
                // Aborted graphs can legitimately leave prepared cache slots
                // unqueued. Their normal error path is reported separately.
                continue;
            }
            if (!traced_residual_wait) {
                trace_sidecar_event(
                    "gpu_residual_graph_finish_wait_begin",
                    prepared.dispatch.dense_packed);
                traced_residual_wait = true;
            }
            bool waited = false;
            residual_result = ggml_backend_opencl_htp_cache_wait(
                graph->sidecar,
                prepared.residual_cache_key.c_str(),
                prepared.dispatch.generation,
                &waited,
                residual_reason,
                sizeof(residual_reason));
            if (residual_result != GGML_OPENCL_HTP_OK) {
                break;
            }
        }
    }
    if (traced_residual_wait) {
        trace_sidecar_event("gpu_residual_graph_finish_wait_end", nullptr);
    }
    const bool finish_ok =
        stop_result == GGML_OPENCL_HTP_OK &&
        residual_result == GGML_OPENCL_HTP_OK;
    const char * finish_reason = stop_result != GGML_OPENCL_HTP_OK
        ? stop_reason
        : residual_result != GGML_OPENCL_HTP_OK
            ? residual_reason
            : "ok";
    return finish_graph_state(
        graph,
        finish_ok ? GGML_OPENCL_HTP_OK : GGML_OPENCL_HTP_ERROR,
        finish_reason,
        reason, reason_capacity);
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
    return graph != nullptr ? graph->events.size() + graph->pending_events.size() : 0;
}

size_t ggml_backend_opencl_htp_graph_pending_sparse_event_count(
        const ggml_opencl_htp_graph * graph) {
    if (graph == nullptr) {
        return 0;
    }
    std::lock_guard<std::mutex> lock(graph->sidecar->dispatch_mutex);
    return graph->pending_events.size();
}

size_t ggml_backend_opencl_htp_graph_pending_sparse_submission_count(
        const ggml_opencl_htp_graph * graph) {
    if (graph == nullptr) {
        return 0;
    }
    std::lock_guard<std::mutex> lock(graph->submission_mutex);
    return static_cast<size_t>(std::count_if(
        graph->submission_records.begin(), graph->submission_records.end(),
        [](const ggml_opencl_htp_submission_record & item) {
            return item.state == ggml_opencl_htp_submission_state::queued ||
                   item.state == ggml_opencl_htp_submission_state::submitting;
        }));
}

size_t ggml_backend_opencl_htp_graph_submitted_sparse_count(
        const ggml_opencl_htp_graph * graph) {
    if (graph == nullptr) {
        return 0;
    }
    std::lock_guard<std::mutex> lock(graph->submission_mutex);
    return static_cast<size_t>(std::count_if(
        graph->submission_records.begin(), graph->submission_records.end(),
        [](const ggml_opencl_htp_submission_record & item) {
            return item.kind ==
                       ggml_opencl_htp_submission_job_kind::sparse_attention &&
                   item.state == ggml_opencl_htp_submission_state::submitted;
        }));
}

size_t ggml_backend_opencl_htp_graph_pending_residual_submission_count(
        const ggml_opencl_htp_graph * graph) {
    if (graph == nullptr) {
        return 0;
    }
    std::lock_guard<std::mutex> lock(graph->submission_mutex);
    return static_cast<size_t>(std::count_if(
        graph->submission_records.begin(), graph->submission_records.end(),
        [](const ggml_opencl_htp_submission_record & item) {
            return item.kind ==
                       ggml_opencl_htp_submission_job_kind::exact_rho_residual &&
                   (item.state == ggml_opencl_htp_submission_state::queued ||
                    item.state == ggml_opencl_htp_submission_state::submitting);
        }));
}

size_t ggml_backend_opencl_htp_graph_pending_residual_event_count(
        const ggml_opencl_htp_graph * graph) {
    if (graph == nullptr || graph->sidecar == nullptr) {
        return 0;
    }
    std::lock_guard<std::mutex> lock(graph->sidecar->cache_mutex);
    size_t count = 0;
    for (const ggml_opencl_htp_prepared_residual_dispatch & prepared :
            graph->prepared_residual_dispatches) {
        const ggml_opencl_htp_cache_record * record = find_cache_record(
            graph->sidecar, prepared.residual_cache_key.c_str(),
            prepared.dispatch.generation);
        count += record != nullptr &&
                 record->state == GGML_OPENCL_HTP_CACHE_SUBMITTED;
    }
    return count;
}

void ggml_backend_opencl_htp_graph_test_fail_next_event_cache(ggml_opencl_htp_graph * graph) {
    if (graph != nullptr && graph->state == ggml_opencl_htp_graph_state::active) {
        graph->test_fail_next_event_cache = true;
    }
}

void ggml_backend_opencl_htp_graph_test_fail_next_sparse_enqueue(
        ggml_opencl_htp_graph * graph) {
    if (graph != nullptr && graph->state == ggml_opencl_htp_graph_state::active) {
        graph->test_fail_next_sparse_enqueue = true;
    }
}

void ggml_backend_opencl_htp_graph_test_fail_next_sparse_wait(
        ggml_opencl_htp_graph * graph) {
    if (graph != nullptr && graph->state == ggml_opencl_htp_graph_state::active) {
        graph->test_fail_next_sparse_wait = true;
    }
}

void ggml_backend_opencl_htp_graph_test_fail_next_rho_enqueue(
        ggml_opencl_htp_graph * graph) {
    if (graph != nullptr && graph->state == ggml_opencl_htp_graph_state::active) {
        graph->test_fail_next_rho_enqueue = true;
    }
}

void ggml_backend_opencl_htp_graph_test_fail_next_residual_enqueue(
        ggml_opencl_htp_graph * graph) {
    if (graph != nullptr && graph->state == ggml_opencl_htp_graph_state::active) {
        graph->test_fail_next_residual_enqueue = true;
    }
}

void ggml_backend_opencl_htp_graph_test_fail_next_residual_wait(
        ggml_opencl_htp_graph * graph) {
    if (graph == nullptr || graph->sidecar == nullptr ||
        graph->state != ggml_opencl_htp_graph_state::active) {
        return;
    }
    std::lock_guard<std::mutex> lock(graph->sidecar->cache_mutex);
    graph->sidecar->test_fail_next_residual_wait = true;
}

void ggml_backend_opencl_htp_sidecar_test_fail_next_residual_wait(
        ggml_opencl_htp_sidecar * sidecar) {
    if (sidecar == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(sidecar->cache_mutex);
    sidecar->test_fail_next_residual_wait = true;
}
