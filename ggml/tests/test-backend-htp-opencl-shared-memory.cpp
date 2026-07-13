#include "ggml-htp.h"
#include "ggml-opencl-htp.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

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

static ggml_opencl_htp_import_descriptor make_import_descriptor(
    ggml_backend_buffer_t backing,
    const ggml_backend_htp_rpcmem_info & info) {
    return {
        /* .backing         = */ backing,
        /* .base            = */ info.base,
        /* .fd              = */ info.fd,
        /* .logical_size    = */ info.logical_size,
        /* .allocation_size = */ info.allocation_size,
    };
}

static void assert_filled_range(
    const uint8_t * bytes,
    size_t logical_size,
    size_t offset,
    size_t size,
    uint32_t pattern,
    uint8_t guard) {
    GGML_ASSERT(offset <= logical_size);
    GGML_ASSERT(size <= logical_size - offset);

    for (size_t i = 0; i < offset; ++i) {
        GGML_ASSERT(bytes[i] == guard);
    }
    for (size_t i = offset; i < offset + size; i += sizeof(pattern)) {
        uint32_t actual = 0;
        std::memcpy(&actual, bytes + i, sizeof(actual));
        GGML_ASSERT(actual == pattern);
    }
    for (size_t i = offset + size; i < logical_size; ++i) {
        GGML_ASSERT(bytes[i] == guard);
    }
}

static void test_tensor_view_bindings(
    ggml_opencl_htp_sidecar * sidecar,
    ggml_backend_t htp,
    const ggml_opencl_qcom_requirements & requirements) {
    constexpr uint8_t guard = 0x6d;
    constexpr uint32_t strided_pattern = 0x12345678;
    constexpr uint32_t second_pattern = 0x90abcdef;

    const size_t logical_size = requirements.page_size + 1024;
    ggml_backend_buffer_t backing = ggml_backend_alloc_buffer(htp, logical_size);
    GGML_ASSERT(backing != nullptr);
    ggml_backend_htp_rpcmem_info info = {};
    GGML_ASSERT(ggml_backend_htp_get_rpcmem_info(backing, &info));
    std::memset(info.base, guard, info.logical_size);

    ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead() * 8,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    GGML_ASSERT(ctx != nullptr);

    ggml_tensor * root = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 16, 8);
    const size_t root_offset = ggml_backend_buffer_get_alignment(backing) * 2;
    GGML_ASSERT(root_offset + ggml_nbytes(root) <= info.logical_size);
    GGML_ASSERT(ggml_backend_tensor_alloc(
        backing, root, static_cast<uint8_t *>(info.base) + root_offset) == GGML_STATUS_SUCCESS);

    constexpr size_t view_offset = 20;
    constexpr size_t row_stride = 96;
    ggml_tensor * strided_view = ggml_view_2d(ctx, root, 4, 3, row_stride, view_offset);
    GGML_ASSERT(ggml_backend_view_init(strided_view) == GGML_STATUS_SUCCESS);
    GGML_ASSERT(strided_view->buffer == backing);
    GGML_ASSERT(strided_view->view_src == root);
    GGML_ASSERT(strided_view->data == static_cast<uint8_t *>(root->data) + view_offset);
    strided_view->buffer = nullptr; // Exercise resolution through view_src without changing data.

    constexpr size_t second_view_offset = 384;
    ggml_tensor * second_view = ggml_view_1d(ctx, root, 8, second_view_offset);
    GGML_ASSERT(ggml_backend_view_init(second_view) == GGML_STATUS_SUCCESS);
    GGML_ASSERT(second_view->buffer == backing);

    auto descriptor = make_import_descriptor(backing, info);
    char reason[512] = {};
    ggml_opencl_htp_graph * graph = nullptr;
    GGML_ASSERT(ggml_backend_opencl_htp_graph_create(
        sidecar, &graph, reason, sizeof(reason)) == GGML_OPENCL_HTP_OK);
    GGML_ASSERT(ggml_backend_opencl_htp_graph_import(
        graph, &descriptor, reason, sizeof(reason)) == GGML_OPENCL_HTP_OK);

    ggml_tensor overflow = *strided_view;
    overflow.nb[1] = std::numeric_limits<size_t>::max();
    GGML_ASSERT(ggml_backend_opencl_htp_graph_fill_tensor(
        graph, &overflow, strided_pattern, reason, sizeof(reason)) == GGML_OPENCL_HTP_ERROR);
    GGML_ASSERT(reason[0] != '\0');

    ggml_tensor out_of_bounds = *strided_view;
    out_of_bounds.data = static_cast<uint8_t *>(info.base) + info.logical_size - sizeof(uint32_t);
    GGML_ASSERT(ggml_backend_opencl_htp_graph_fill_tensor(
        graph, &out_of_bounds, strided_pattern, reason, sizeof(reason)) == GGML_OPENCL_HTP_ERROR);
    GGML_ASSERT(reason[0] != '\0');
    GGML_ASSERT(ggml_backend_opencl_htp_graph_live_event_count(graph) == 0);

    GGML_ASSERT(ggml_backend_opencl_htp_graph_fill_tensor(
        graph, strided_view, strided_pattern, reason, sizeof(reason)) == GGML_OPENCL_HTP_OK);
    GGML_ASSERT(ggml_backend_opencl_htp_graph_fill_tensor(
        graph, second_view, second_pattern, reason, sizeof(reason)) == GGML_OPENCL_HTP_OK);
    GGML_ASSERT(ggml_backend_opencl_htp_graph_live_alias_count(graph) == 1);
    GGML_ASSERT(ggml_backend_opencl_htp_graph_live_event_count(graph) == 2);
    GGML_ASSERT(ggml_backend_opencl_htp_graph_finish(
        graph, reason, sizeof(reason)) == GGML_OPENCL_HTP_OK);

    const size_t strided_span =
        (strided_view->ne[0] - 1) * strided_view->nb[0] +
        (strided_view->ne[1] - 1) * strided_view->nb[1] + ggml_type_size(strided_view->type);
    const size_t strided_begin = root_offset + view_offset;
    const size_t second_begin = root_offset + second_view_offset;
    const uint8_t * bytes = static_cast<const uint8_t *>(info.base);
    GGML_ASSERT(bytes[strided_begin - 1] == guard);
    GGML_ASSERT(bytes[strided_begin + strided_span] == guard);
    GGML_ASSERT(bytes[second_begin - 1] == guard);
    GGML_ASSERT(bytes[second_begin + ggml_nbytes(second_view)] == guard);
    for (size_t offset = 0; offset < strided_span; offset += sizeof(uint32_t)) {
        uint32_t actual = 0;
        std::memcpy(&actual, bytes + strided_begin + offset, sizeof(actual));
        GGML_ASSERT(actual == strided_pattern);
    }
    for (size_t offset = 0; offset < ggml_nbytes(second_view); offset += sizeof(uint32_t)) {
        uint32_t actual = 0;
        std::memcpy(&actual, bytes + second_begin + offset, sizeof(actual));
        GGML_ASSERT(actual == second_pattern);
    }

    GGML_ASSERT(fcntl(info.fd, F_GETFD) != -1);
    ggml_backend_opencl_htp_graph_free(&graph);
    ggml_free(ctx);
    ggml_backend_buffer_free(backing);
}

static void test_event_cache_failure_cleanup(
    ggml_opencl_htp_sidecar * sidecar,
    ggml_backend_buffer_t backing,
    const ggml_opencl_htp_import_descriptor & descriptor) {
    constexpr uint32_t pattern = 0xa1b2c3d4;
    char reason[512] = {};
    ggml_opencl_htp_graph * graph = nullptr;
    GGML_ASSERT(ggml_backend_opencl_htp_graph_create(
        sidecar, &graph, reason, sizeof(reason)) == GGML_OPENCL_HTP_OK);
    GGML_ASSERT(ggml_backend_opencl_htp_graph_import(
        graph, &descriptor, reason, sizeof(reason)) == GGML_OPENCL_HTP_OK);

    ggml_backend_opencl_htp_graph_test_fail_next_event_cache(graph);
    GGML_ASSERT(ggml_backend_opencl_htp_graph_fill(
        graph, backing, 64, sizeof(pattern), pattern,
        reason, sizeof(reason)) == GGML_OPENCL_HTP_ERROR);
    const std::string failure_reason = reason;
    GGML_ASSERT(!failure_reason.empty());
    GGML_ASSERT(ggml_backend_opencl_htp_graph_live_alias_count(graph) == 0);
    GGML_ASSERT(ggml_backend_opencl_htp_graph_live_event_count(graph) == 0);
    GGML_ASSERT(fcntl(descriptor.fd, F_GETFD) != -1);

    GGML_ASSERT(ggml_backend_opencl_htp_graph_finish(
        graph, reason, sizeof(reason)) == GGML_OPENCL_HTP_ERROR);
    GGML_ASSERT(reason == failure_reason);
    GGML_ASSERT(ggml_backend_opencl_htp_graph_finish(
        graph, reason, sizeof(reason)) == GGML_OPENCL_HTP_ERROR);
    GGML_ASSERT(reason == failure_reason);
    ggml_backend_opencl_htp_graph_free(&graph);
    GGML_ASSERT(graph == nullptr);
    GGML_ASSERT(fcntl(descriptor.fd, F_GETFD) != -1);
}

static void test_graph_scoped_rpcmem_aliases(
    ggml_opencl_htp_sidecar * sidecar,
    const ggml_opencl_qcom_requirements & requirements) {
    constexpr uint8_t guard = 0xa5;
    constexpr uint32_t first_pattern = 0x11223344;
    constexpr uint32_t second_pattern = 0x55667788;
    constexpr size_t fill_size = 64;

    GGML_ASSERT(sidecar != nullptr);
    GGML_ASSERT(setenv("SD_HTP_SPARSE_OPENCL", "1", 1) == 0);

    ggml_backend_t htp = ggml_backend_htp_init();
    GGML_ASSERT(htp != nullptr);
    GGML_ASSERT(ggml_backend_htp_runtime_available());

    const size_t logical_size = requirements.page_size * 2 + 257;
    const size_t fill_offset = requirements.page_size + 20;
    GGML_ASSERT(fill_offset % sizeof(uint32_t) == 0);
    GGML_ASSERT(fill_offset + fill_size < logical_size);

    ggml_backend_buffer_t first = ggml_backend_alloc_buffer(htp, logical_size);
    ggml_backend_buffer_t second = ggml_backend_alloc_buffer(htp, logical_size);
    GGML_ASSERT(first != nullptr && second != nullptr && first != second);

    ggml_backend_htp_rpcmem_info first_info = {};
    ggml_backend_htp_rpcmem_info second_info = {};
    GGML_ASSERT(ggml_backend_htp_get_rpcmem_info(first, &first_info));
    GGML_ASSERT(ggml_backend_htp_get_rpcmem_info(second, &second_info));
    GGML_ASSERT(first_info.fd != second_info.fd);

    test_tensor_view_bindings(sidecar, htp, requirements);

    auto first_desc = make_import_descriptor(first, first_info);
    auto second_desc = make_import_descriptor(second, second_info);
    std::memset(first_info.base, guard, first_info.logical_size);
    std::memset(second_info.base, guard, second_info.logical_size);

    test_event_cache_failure_cleanup(sidecar, first, first_desc);
    std::memset(first_info.base, guard, first_info.logical_size);

    void * second_mapping = mmap(
        nullptr, first_info.allocation_size, PROT_READ | PROT_WRITE, MAP_SHARED, first_info.fd, 0);
    GGML_ASSERT(second_mapping != MAP_FAILED);

    char reason[512] = {};
    ggml_opencl_htp_graph * graph = nullptr;
    GGML_ASSERT(ggml_backend_opencl_htp_graph_create(
        sidecar, &graph, reason, sizeof(reason)) == GGML_OPENCL_HTP_OK);
    GGML_ASSERT(graph != nullptr);
    GGML_ASSERT(ggml_backend_opencl_htp_graph_live_alias_count(graph) == 0);
    GGML_ASSERT(ggml_backend_opencl_htp_graph_live_event_count(graph) == 0);
    GGML_ASSERT(ggml_backend_opencl_htp_graph_fill_tensor(
        graph, nullptr, first_pattern, reason, sizeof(reason)) == GGML_OPENCL_HTP_ERROR);

    GGML_ASSERT(ggml_backend_opencl_htp_graph_import(
        graph, &first_desc, reason, sizeof(reason)) == GGML_OPENCL_HTP_OK);
    GGML_ASSERT(ggml_backend_opencl_htp_graph_live_alias_count(graph) == 1);

    // A second view of the exact same backing must reuse the graph-scoped alias.
    GGML_ASSERT(ggml_backend_opencl_htp_graph_import(
        graph, &first_desc, reason, sizeof(reason)) == GGML_OPENCL_HTP_OK);
    GGML_ASSERT(ggml_backend_opencl_htp_graph_live_alias_count(graph) == 1);
    GGML_ASSERT(ggml_backend_opencl_htp_graph_fill(
        graph, first, 64, sizeof(first_pattern), first_pattern,
        reason, sizeof(reason)) == GGML_OPENCL_HTP_OK);
    GGML_ASSERT(ggml_backend_opencl_htp_graph_live_event_count(graph) == 1);

    auto conflicting_desc = first_desc;
    --conflicting_desc.logical_size;
    GGML_ASSERT(ggml_backend_opencl_htp_graph_import(
        graph, &conflicting_desc, reason, sizeof(reason)) == GGML_OPENCL_HTP_ERROR);
    GGML_ASSERT(reason[0] != '\0');
    GGML_ASSERT(ggml_backend_opencl_htp_graph_live_alias_count(graph) == 0);
    GGML_ASSERT(ggml_backend_opencl_htp_graph_live_event_count(graph) == 0);
    GGML_ASSERT(ggml_backend_opencl_htp_graph_finish(
        graph, reason, sizeof(reason)) == GGML_OPENCL_HTP_ERROR);
    GGML_ASSERT(ggml_backend_opencl_htp_graph_finish(
        graph, reason, sizeof(reason)) == GGML_OPENCL_HTP_ERROR);
    ggml_backend_opencl_htp_graph_free(&graph);
    GGML_ASSERT(graph == nullptr);
    std::memset(first_info.base, guard, first_info.logical_size);

    GGML_ASSERT(ggml_backend_opencl_htp_graph_create(
        sidecar, &graph, reason, sizeof(reason)) == GGML_OPENCL_HTP_OK);
    GGML_ASSERT(ggml_backend_opencl_htp_graph_import(
        graph, &first_desc, reason, sizeof(reason)) == GGML_OPENCL_HTP_OK);

    auto invalid_desc = second_desc;
    invalid_desc.allocation_size = invalid_desc.logical_size - 1;
    GGML_ASSERT(ggml_backend_opencl_htp_graph_import(
        graph, &invalid_desc, reason, sizeof(reason)) == GGML_OPENCL_HTP_ERROR);
    GGML_ASSERT(reason[0] != '\0');
    GGML_ASSERT(ggml_backend_opencl_htp_graph_live_alias_count(graph) == 0);
    GGML_ASSERT(ggml_backend_opencl_htp_graph_finish(
        graph, reason, sizeof(reason)) == GGML_OPENCL_HTP_ERROR);
    ggml_backend_opencl_htp_graph_free(&graph);
    GGML_ASSERT(graph == nullptr);

    GGML_ASSERT(ggml_backend_opencl_htp_graph_create(
        sidecar, &graph, reason, sizeof(reason)) == GGML_OPENCL_HTP_OK);
    GGML_ASSERT(ggml_backend_opencl_htp_graph_import(
        graph, &first_desc, reason, sizeof(reason)) == GGML_OPENCL_HTP_OK);
    GGML_ASSERT(ggml_backend_opencl_htp_graph_import(
        graph, &second_desc, reason, sizeof(reason)) == GGML_OPENCL_HTP_OK);
    GGML_ASSERT(ggml_backend_opencl_htp_graph_live_alias_count(graph) == 2);

    GGML_ASSERT(ggml_backend_opencl_htp_graph_fill(
        graph, first, std::numeric_limits<size_t>::max() - 3, 8, first_pattern,
        reason, sizeof(reason)) == GGML_OPENCL_HTP_ERROR);
    GGML_ASSERT(reason[0] != '\0');
    GGML_ASSERT(ggml_backend_opencl_htp_graph_fill(
        graph, first, first_info.logical_size - 2, 4, first_pattern,
        reason, sizeof(reason)) == GGML_OPENCL_HTP_ERROR);
    GGML_ASSERT(reason[0] != '\0');
    GGML_ASSERT(ggml_backend_opencl_htp_graph_live_event_count(graph) == 0);

    GGML_ASSERT(ggml_backend_opencl_htp_graph_fill(
        graph, first, fill_offset, fill_size, first_pattern,
        reason, sizeof(reason)) == GGML_OPENCL_HTP_OK);
    GGML_ASSERT(ggml_backend_opencl_htp_graph_fill(
        graph, second, fill_offset, fill_size, second_pattern,
        reason, sizeof(reason)) == GGML_OPENCL_HTP_OK);
    GGML_ASSERT(ggml_backend_opencl_htp_graph_live_event_count(graph) == 2);

    GGML_ASSERT(ggml_backend_opencl_htp_graph_finish(
        graph, reason, sizeof(reason)) == GGML_OPENCL_HTP_OK);
    GGML_ASSERT(ggml_backend_opencl_htp_graph_live_alias_count(graph) == 0);
    GGML_ASSERT(ggml_backend_opencl_htp_graph_live_event_count(graph) == 0);
    GGML_ASSERT(fcntl(first_info.fd, F_GETFD) != -1);
    GGML_ASSERT(fcntl(second_info.fd, F_GETFD) != -1);

    assert_filled_range(
        static_cast<const uint8_t *>(first_info.base), first_info.logical_size,
        fill_offset, fill_size, first_pattern, guard);
    assert_filled_range(
        static_cast<const uint8_t *>(second_mapping), first_info.logical_size,
        fill_offset, fill_size, first_pattern, guard);
    assert_filled_range(
        static_cast<const uint8_t *>(second_info.base), second_info.logical_size,
        fill_offset, fill_size, second_pattern, guard);

    // Finishing and freeing a graph are separately idempotent operations.
    GGML_ASSERT(ggml_backend_opencl_htp_graph_finish(
        graph, reason, sizeof(reason)) == GGML_OPENCL_HTP_OK);
    ggml_backend_opencl_htp_graph_free(&graph);
    GGML_ASSERT(graph == nullptr);
    ggml_backend_opencl_htp_graph_free(&graph);
    GGML_ASSERT(fcntl(first_info.fd, F_GETFD) != -1);

    // Repeated graph lifetimes must not retain stale aliases or events.
    const size_t repeated_offset = 128;
    std::vector<int> retired_fds;
    bool saw_fd_reuse = false;
    for (uint32_t iteration = 0; iteration < 100; ++iteration) {
        ggml_backend_buffer_t repeated = ggml_backend_alloc_buffer(htp, logical_size);
        GGML_ASSERT(repeated != nullptr);
        ggml_backend_htp_rpcmem_info repeated_info = {};
        GGML_ASSERT(ggml_backend_htp_get_rpcmem_info(repeated, &repeated_info));
        saw_fd_reuse = saw_fd_reuse ||
            std::find(retired_fds.begin(), retired_fds.end(), repeated_info.fd) != retired_fds.end();
        auto repeated_desc = make_import_descriptor(repeated, repeated_info);
        std::memset(repeated_info.base, guard, repeated_info.logical_size);

        GGML_ASSERT(ggml_backend_opencl_htp_graph_create(
            sidecar, &graph, reason, sizeof(reason)) == GGML_OPENCL_HTP_OK);
        GGML_ASSERT(ggml_backend_opencl_htp_graph_import(
            graph, &repeated_desc, reason, sizeof(reason)) == GGML_OPENCL_HTP_OK);
        GGML_ASSERT(ggml_backend_opencl_htp_graph_import(
            graph, &repeated_desc, reason, sizeof(reason)) == GGML_OPENCL_HTP_OK);
        GGML_ASSERT(ggml_backend_opencl_htp_graph_live_alias_count(graph) == 1);
        GGML_ASSERT(ggml_backend_opencl_htp_graph_fill(
            graph, repeated, repeated_offset, sizeof(iteration), iteration,
            reason, sizeof(reason)) == GGML_OPENCL_HTP_OK);
        GGML_ASSERT(ggml_backend_opencl_htp_graph_finish(
            graph, reason, sizeof(reason)) == GGML_OPENCL_HTP_OK);
        GGML_ASSERT(ggml_backend_opencl_htp_graph_live_alias_count(graph) == 0);
        GGML_ASSERT(ggml_backend_opencl_htp_graph_live_event_count(graph) == 0);

        uint32_t repeated_value = std::numeric_limits<uint32_t>::max();
        const uint8_t * repeated_bytes = static_cast<const uint8_t *>(repeated_info.base);
        std::memcpy(&repeated_value, repeated_bytes + repeated_offset, sizeof(repeated_value));
        GGML_ASSERT(repeated_value == iteration);
        GGML_ASSERT(repeated_bytes[repeated_offset - 1] == guard);
        GGML_ASSERT(repeated_bytes[repeated_offset + sizeof(iteration)] == guard);
        GGML_ASSERT(fcntl(repeated_info.fd, F_GETFD) != -1);

        ggml_backend_opencl_htp_graph_free(&graph);
        GGML_ASSERT(graph == nullptr);
        GGML_ASSERT(fcntl(repeated_info.fd, F_GETFD) != -1);
        const int retired_fd = repeated_info.fd;
        ggml_backend_buffer_free(repeated);
        errno = 0;
        GGML_ASSERT(fcntl(retired_fd, F_GETFD) == -1);
        GGML_ASSERT(errno == EBADF);
        retired_fds.push_back(retired_fd);
    }
    std::printf("fd_reuse_observed=%s\n", saw_fd_reuse ? "yes" : "no");
    GGML_ASSERT(fcntl(first_info.fd, F_GETFD) != -1);

    GGML_ASSERT(munmap(second_mapping, first_info.allocation_size) == 0);
    ggml_backend_buffer_free(second);
    ggml_backend_buffer_free(first);
    ggml_backend_free(htp);
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

    test_rpcmem_allocation_metadata(requirements);
    test_graph_scoped_rpcmem_aliases(sidecar, requirements);
    ggml_backend_opencl_htp_sidecar_free(sidecar);
    return 0;
}
