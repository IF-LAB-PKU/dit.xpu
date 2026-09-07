#include "htp_packed_staging.h"

#include "htp_tensor_flags.h"
#include "flux.hpp"
#include "model.h"
#include "model_io/safetensors_io.h"
#include "stable-diffusion.h"

#include "ggml-cpu.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <vector>

#define CHECK(condition)                                                                 \
    do {                                                                                 \
        if (!(condition)) {                                                              \
            fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #condition); \
            std::abort();                                                                \
        }                                                                                \
    } while (0)

#if !defined(GGML_USE_HTP)
extern "C" ggml_backend_buffer_type_t ggml_backend_htp_qkv_buffer_type(void) { return nullptr; }
extern "C" bool ggml_backend_htp_opencl_release_qkv_generation(ggml_backend_t, uint64_t, bool, bool* released, char*, size_t) {
    if (released != nullptr) *released = true;
    return true;
}
#endif

static TensorWriteInfo info_for(ggml_tensor* tensor) {
    TensorWriteInfo info;
    info.n_dims = ggml_n_dims(tensor);
    info.tensor = tensor;
    for (int i = 0; i < info.n_dims; ++i) info.ne[i] = tensor->ne[i];
    return info;
}

int main() {
    // Packed Flux2 Q/K/V must remain graph views so gallocr can release the
    // shared storage after the attention node consumes the last view.
    ggml_context* packed_ctx = ggml_init({64 * 1024 * 1024, nullptr, false});
    CHECK(packed_ctx != nullptr);
    GGMLRunnerContext packed_runner;
    packed_runner.ggml_ctx = packed_ctx;
    ggml_tensor* packed_storage = ggml_new_tensor_1d(
        packed_ctx, GGML_TYPE_F16, 3 * 24 * 48 * 4 * 1024);
    CHECK(packed_storage != nullptr);
    const Flux::PackedRopeQkv packed_views = Flux::make_packed_flux2_qkv_views(
        &packed_runner, packed_storage, 1536);
    CHECK(packed_views.storage == packed_storage);
    CHECK(packed_views.qkv.size() == 3);
    CHECK(packed_views.qkv[0]->op == GGML_OP_VIEW);
    CHECK(packed_views.qkv[1]->op == GGML_OP_VIEW);
    CHECK(packed_views.qkv[2]->op == GGML_OP_VIEW);
    ggml_free(packed_ctx);

    CHECK(htp_packed_staging_owner_count(true, true, 4) == 4);
    CHECK(htp_packed_staging_owner_count(true, true, 1) == 1);
    CHECK(htp_packed_staging_owner_count(true, false, 4) == 1);
    CHECK(htp_packed_staging_owner_count(false, true, 4) == 1);
    CHECK(htp_packed_staging_owner_count(false, false, 4) == 1);

    ggml_context* ctx = ggml_init({1024 * 1024, nullptr, false});
    CHECK(ctx != nullptr);

    ggml_tensor* packed_policy_tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 32, 32);
    packed_policy_tensor->flags |= SD_GGML_TENSOR_FLAG_HTP_MUL_MAT_WEIGHT |
                                   SD_GGML_TENSOR_FLAG_HTP_PACKED_W16A32;
    CHECK(htp_should_defer_packed_param(true, true, 1, packed_policy_tensor));
    CHECK(!htp_should_defer_packed_param(false, true, 1, packed_policy_tensor));
    CHECK(!htp_should_defer_packed_param(true, false, 1, packed_policy_tensor));
    CHECK(!htp_should_defer_packed_param(true, true, 0, packed_policy_tensor));

    ggml_tensor* ordinary_policy_tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 32, 32);
    ordinary_policy_tensor->flags |= SD_GGML_TENSOR_FLAG_HTP_MUL_MAT_WEIGHT;
    CHECK(!htp_should_defer_packed_param(true, true, 1, ordinary_policy_tensor));

    ggml_tensor* packed_policy_view = ggml_view_tensor(ctx, packed_policy_tensor);
    CHECK(!htp_should_defer_packed_param(true, true, 1, packed_policy_view));
    CHECK((packed_policy_tensor->flags & SD_GGML_TENSOR_FLAG_HTP_DEFERRED_PACKED_PARAM) == 0);

    ggml_context* params_ctx = ggml_init({
        4 * ggml_tensor_overhead(), nullptr, true});
    CHECK(params_ctx != nullptr);
    ggml_tensor* ordinary_param = ggml_new_tensor_1d(params_ctx, GGML_TYPE_F32, 4);
    ggml_tensor* deferred_param = ggml_new_tensor_2d(params_ctx, GGML_TYPE_F16, 32, 32);
    deferred_param->flags |= SD_GGML_TENSOR_FLAG_HTP_MUL_MAT_WEIGHT |
                             SD_GGML_TENSOR_FLAG_HTP_PACKED_W16A32 |
                             SD_GGML_TENSOR_FLAG_HTP_DEFERRED_PACKED_PARAM;
    ggml_backend_t cpu_backend = ggml_backend_cpu_init();
    CHECK(cpu_backend != nullptr);
    HtpNonDeferredParamsAllocation non_deferred_allocation;
    CHECK(non_deferred_allocation.allocate(params_ctx, cpu_backend));
    CHECK(ordinary_param->buffer != nullptr && ordinary_param->data != nullptr);
    CHECK(deferred_param->buffer == nullptr && deferred_param->data == nullptr &&
          deferred_param->extra == nullptr);
    CHECK(non_deferred_allocation.logical_bytes() ==
          ggml_nbytes(ordinary_param) + ggml_nbytes(deferred_param));
    CHECK(non_deferred_allocation.host_backed_bytes() == ggml_nbytes(ordinary_param));
    CHECK(non_deferred_allocation.deferred_bytes() == ggml_nbytes(deferred_param));
    non_deferred_allocation.clear();
    CHECK(ordinary_param->buffer == nullptr && ordinary_param->data == nullptr &&
          ordinary_param->extra == nullptr);
    CHECK(deferred_param->buffer == nullptr && deferred_param->data == nullptr &&
          deferred_param->extra == nullptr);
    ggml_backend_free(cpu_backend);
    ggml_free(params_ctx);

    ggml_tensor* source = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    ggml_set_name(source, "packed.weight");
    const float values[] = {10.f, 20.f, 30.f, 40.f};
    memcpy(source->data, values, sizeof(values));

    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto path = std::filesystem::temp_directory_path() / ("sd-htp-staging-" + std::to_string(nonce) + ".safetensors");
    std::string error;
    CHECK(write_safetensors_file(path.string(), {info_for(source)}, &error));

    ModelLoader loader;
    CHECK(loader.init_from_file(path.string()));
    ggml_tensor* destination = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    memset(destination->data, 0, ggml_nbytes(destination));

    size_t allocations = 0;
    HtpPackedStagingPool pool(
        128, 64,
        [&allocations](size_t bytes) {
            ++allocations;
            return HtpPackedStagingPool::Allocation{std::malloc(bytes), nullptr};
        },
        [](const HtpPackedStagingPool::Allocation& allocation) { std::free(allocation.data); });

    HtpPackedStagingPool::Handle handle;
    CHECK(pool.acquire(64, &handle));
    CHECK(pool.place_tensor(
        [&loader](const std::string& name, void* dst, size_t bytes) {
            return loader.reload_tensor_bytes(name, dst, bytes);
        },
        handle, "packed.weight", destination));
    CHECK(htp_packed_staging_contains(
        std::vector<ggml_tensor*>{destination}, destination));
    CHECK(!htp_packed_staging_contains(std::vector<ggml_tensor*>{}, destination));
    CHECK(memcmp(destination->data, values, sizeof(values)) == 0);
    void* staged_data = destination->data;
    destination->buffer = nullptr;
    destination->data = nullptr;
    CHECK(pool.rebind_tensor(handle, destination));
    CHECK(destination->data == staged_data);
    CHECK(!pool.release(&handle));
    CHECK(pool.unbind_tensor(handle, destination));
    CHECK(destination->buffer == nullptr && destination->data != nullptr);
    CHECK(pool.release(&handle));
    CHECK(pool.active_slots() == 0);

    HtpPackedStagingPool::Handle second;
    CHECK(pool.acquire(64, &second));
    CHECK(allocations == 1);
    CHECK(pool.release(&second));

    size_t reclaimed = 0;
    HtpPackedStagingPool reclaim_pool(
        128, 32, 2,
        [](size_t bytes, int session) {
            return HtpPackedStagingPool::Allocation{
                std::malloc(bytes), nullptr, session, 3 + 4 * session};
        },
        [&reclaimed](const HtpPackedStagingPool::Allocation& allocation) {
            ++reclaimed;
            std::free(allocation.data);
        });
    HtpPackedStagingPool::Handle old_handles[4];
    CHECK(reclaim_pool.acquire_for_session(32, 0, &old_handles[0]));
    CHECK(reclaim_pool.acquire_for_session(32, 0, &old_handles[1]));
    CHECK(reclaim_pool.acquire_for_session(32, 1, &old_handles[2]));
    CHECK(reclaim_pool.acquire_for_session(32, 1, &old_handles[3]));
    for (auto& old_handle : old_handles) {
        CHECK(reclaim_pool.release(&old_handle));
    }

    HtpPackedStagingPool::Handle reused_session_0;
    HtpPackedStagingPool::Handle reused_session_1;
    CHECK(reclaim_pool.acquire_for_session(32, 0, &reused_session_0));
    CHECK(reclaim_pool.acquire_for_session(32, 1, &reused_session_1));
    CHECK(reclaim_pool.reclaim_unleased() == 2);
    CHECK(reclaimed == 2);
    CHECK(reclaim_pool.active_slots() == 2);
    CHECK(reclaim_pool.allocated_bytes() == 64);
    CHECK(reclaim_pool.session_allocated_bytes(0) == 32);
    CHECK(reclaim_pool.session_allocated_bytes(1) == 32);
    CHECK(reclaim_pool.session_for_handle(reused_session_0) == 0);
    CHECK(reclaim_pool.session_for_handle(reused_session_1) == 1);

    CHECK(reclaim_pool.release(&reused_session_0));
    CHECK(reclaim_pool.release(&reused_session_1));
    CHECK(reclaim_pool.reclaim_unleased() == 2);
    CHECK(reclaimed == 4);
    CHECK(reclaim_pool.allocated_bytes() == 0);

    HtpPackedStagingPool::Handle first;
    CHECK(pool.acquire(65, &first) == false);

    ggml_context* deferred_staging_ctx = ggml_init({
        4 * ggml_tensor_overhead(), nullptr, true});
    CHECK(deferred_staging_ctx != nullptr);
    ggml_tensor* deferred_destination =
        ggml_new_tensor_1d(deferred_staging_ctx, GGML_TYPE_F32, 4);
    deferred_destination->flags |= SD_GGML_TENSOR_FLAG_HTP_DEFERRED_PACKED_PARAM;
    HtpPackedStagingPool::Handle deferred_handle;
    CHECK(pool.acquire(64, &deferred_handle));
    CHECK(pool.place_tensor(
        [&loader](const std::string& name, void* dst, size_t bytes) {
            return loader.reload_tensor_bytes(name, dst, bytes);
        },
        deferred_handle, "packed.weight", deferred_destination));
    CHECK(memcmp(deferred_destination->data, values, sizeof(values)) == 0);
    CHECK((deferred_destination->flags &
           SD_GGML_TENSOR_FLAG_HTP_DEFERRED_PACKED_PARAM) != 0);
    CHECK(pool.unbind_tensor(deferred_handle, deferred_destination));
    CHECK(deferred_destination->buffer == nullptr &&
          deferred_destination->data == nullptr &&
          deferred_destination->extra == nullptr);
    CHECK(pool.release(&deferred_handle));
    CHECK(pool.active_slots() == 0);

    ggml_tensor* rollback_first =
        ggml_new_tensor_1d(deferred_staging_ctx, GGML_TYPE_F32, 4);
    ggml_tensor* rollback_second =
        ggml_new_tensor_1d(deferred_staging_ctx, GGML_TYPE_F32, 4);
    rollback_first->flags |= SD_GGML_TENSOR_FLAG_HTP_DEFERRED_PACKED_PARAM;
    rollback_second->flags |= SD_GGML_TENSOR_FLAG_HTP_DEFERRED_PACKED_PARAM;
    HtpPackedStagingPool rollback_pool(
        256, 256,
        [](size_t bytes) {
            return HtpPackedStagingPool::Allocation{std::malloc(bytes), nullptr};
        },
        [](const HtpPackedStagingPool::Allocation& allocation) {
            std::free(allocation.data);
        });
    HtpPackedStagingPool::Handle rollback_handle;
    CHECK(rollback_pool.acquire(256, &rollback_handle));
    CHECK(rollback_pool.place_tensor(
        [&loader](const std::string& name, void* dst, size_t bytes) {
            return loader.reload_tensor_bytes(name, dst, bytes);
        },
        rollback_handle, "packed.weight", rollback_first));
    bool failing_loader_called = false;
    CHECK(!rollback_pool.place_tensor(
        [&failing_loader_called](const std::string&, void*, size_t) {
            failing_loader_called = true;
            return false;
        },
        rollback_handle, "reload.failure", rollback_second));
    CHECK(failing_loader_called);
    CHECK(rollback_pool.unbind_tensor(rollback_handle, rollback_first));
    CHECK(rollback_pool.release(&rollback_handle));
    CHECK(rollback_first->buffer == nullptr && rollback_first->data == nullptr &&
          rollback_first->extra == nullptr);
    CHECK(rollback_second->buffer == nullptr && rollback_second->data == nullptr &&
          rollback_second->extra == nullptr);
    CHECK(rollback_pool.active_slots() == 0);

    HtpPackedStagingPool exhaustion_pool(
        64, 64,
        [](size_t bytes) {
            return HtpPackedStagingPool::Allocation{std::malloc(bytes), nullptr};
        },
        [](const HtpPackedStagingPool::Allocation& allocation) {
            std::free(allocation.data);
        });
    HtpPackedStagingPool::Handle exhaustion_handle;
    CHECK(exhaustion_pool.acquire(64, &exhaustion_handle));
    HtpPackedStagingPool::Handle unavailable_handle;
    CHECK(!exhaustion_pool.acquire(64, &unavailable_handle));
    CHECK(exhaustion_pool.release(&exhaustion_handle));
    CHECK(exhaustion_pool.active_slots() == 0);
    ggml_free(deferred_staging_ctx);

    std::vector<HtpPackedStagingPool::DiagnosticEvent> diagnostics;
    HtpPackedStagingPool diagnostic_pool(
        128, 64,
        [](size_t bytes) {
            return HtpPackedStagingPool::Allocation{
                std::malloc(bytes), nullptr, 0, 3};
        },
        [](const HtpPackedStagingPool::Allocation& allocation) { std::free(allocation.data); },
        [&diagnostics](const HtpPackedStagingPool::DiagnosticEvent& event) {
            diagnostics.push_back(event);
        });
    CHECK(diagnostics.size() == 1);
    CHECK(diagnostics.back().kind == HtpPackedStagingPool::DiagnosticKind::PoolCreated);
    CHECK(diagnostics.back().total_allocated_bytes == 0);

    HtpPackedStagingPool::Handle diagnostic_handle;
    CHECK(diagnostic_pool.acquire(16, &diagnostic_handle));
    CHECK(diagnostics.back().kind == HtpPackedStagingPool::DiagnosticKind::SlotAcquired);
    CHECK(diagnostics.back().requested_bytes == 16);
    CHECK(diagnostics.back().slot_index == 0);
    CHECK(diagnostics.back().logical_session_index == 0);
    CHECK(diagnostics.back().domain == 3);
    CHECK(diagnostics.back().session_allocated_bytes == 64);
    CHECK(diagnostics.back().total_allocated_bytes == 64);
    CHECK(!diagnostics.back().reused);

    CHECK(diagnostic_pool.place_tensor(
        [&loader](const std::string& name, void* dst, size_t bytes) {
            return loader.reload_tensor_bytes(name, dst, bytes);
        },
        diagnostic_handle, "packed.weight", destination));
    CHECK(diagnostics.back().kind == HtpPackedStagingPool::DiagnosticKind::TensorPlaced);
    CHECK(diagnostics.back().requested_bytes == sizeof(values));
    CHECK(diagnostics.back().slot_index == 0);
    CHECK(diagnostic_pool.unbind_tensor(diagnostic_handle, destination));
    CHECK(diagnostic_pool.release(&diagnostic_handle));
    CHECK(diagnostics.back().kind == HtpPackedStagingPool::DiagnosticKind::SlotReleased);
    CHECK(diagnostics.back().session_allocated_bytes == 64);
    CHECK(diagnostics.back().total_allocated_bytes == 64);

    CHECK(diagnostic_pool.acquire(32, &diagnostic_handle));
    CHECK(diagnostics.back().kind == HtpPackedStagingPool::DiagnosticKind::SlotAcquired);
    CHECK(diagnostics.back().requested_bytes == 32);
    CHECK(diagnostics.back().slot_index == 0);
    CHECK(diagnostics.back().session_allocated_bytes == 64);
    CHECK(diagnostics.back().total_allocated_bytes == 64);
    CHECK(diagnostics.back().reused);
    CHECK(diagnostic_pool.release(&diagnostic_handle));

    std::vector<int> requested_owners;
    std::vector<HtpPackedStagingPool::DiagnosticEvent> distributed_diagnostics;
    HtpPackedStagingPool distributed_pool(
        512, 64, 4,
        [&requested_owners](size_t bytes, int logical_session_index) {
            requested_owners.push_back(logical_session_index);
            return HtpPackedStagingPool::Allocation{
                std::malloc(bytes), nullptr, logical_session_index,
                3 + logical_session_index};
        },
        [](const HtpPackedStagingPool::Allocation& allocation) { std::free(allocation.data); },
        [&distributed_diagnostics](const HtpPackedStagingPool::DiagnosticEvent& event) {
            distributed_diagnostics.push_back(event);
        });
    std::vector<HtpPackedStagingPool::Handle> distributed_handles(8);
    for (auto& distributed_handle : distributed_handles) {
        CHECK(distributed_pool.acquire(64, &distributed_handle));
    }
    CHECK((requested_owners == std::vector<int>{0, 1, 2, 3, 0, 1, 2, 3}));
    CHECK(distributed_pool.allocated_bytes() == 512);
    for (int session = 0; session < 4; ++session) {
        CHECK(distributed_pool.session_allocated_bytes(session) == 128);
    }
    HtpPackedStagingPool::Handle over_budget;
    CHECK(!distributed_pool.acquire(64, &over_budget));

    for (auto& distributed_handle : distributed_handles) {
        CHECK(distributed_pool.release(&distributed_handle));
    }
    distributed_diagnostics.clear();
    std::vector<HtpPackedStagingPool::Handle> reused_handles(4);
    for (auto& reused_handle : reused_handles) {
        CHECK(distributed_pool.acquire(64, &reused_handle));
    }
    std::vector<int> reused_owners;
    for (const auto& event : distributed_diagnostics) {
        if (event.kind == HtpPackedStagingPool::DiagnosticKind::SlotAcquired) {
            CHECK(event.reused);
            reused_owners.push_back(event.logical_session_index);
        }
    }
    CHECK((reused_owners == std::vector<int>{0, 1, 2, 3}));
    for (auto& reused_handle : reused_handles) {
        CHECK(distributed_pool.release(&reused_handle));
    }

    std::vector<std::vector<int>> segment_owners;
    for (int segment = 0; segment < 2; ++segment) {
        distributed_diagnostics.clear();
        std::vector<HtpPackedStagingPool::Handle> segment_handles(8);
        for (auto& segment_handle : segment_handles) {
            CHECK(distributed_pool.acquire(64, &segment_handle));
        }
        std::vector<int> owners;
        for (const auto& event : distributed_diagnostics) {
            if (event.kind == HtpPackedStagingPool::DiagnosticKind::SlotAcquired) {
                owners.push_back(event.logical_session_index);
            }
        }
        segment_owners.push_back(std::move(owners));
        for (auto& segment_handle : segment_handles) {
            CHECK(distributed_pool.release(&segment_handle));
        }
    }
    const std::vector<int> expected_segment_owners{0, 1, 2, 3, 0, 1, 2, 3};
    CHECK(segment_owners.size() == 2);
    CHECK(segment_owners[0] == expected_segment_owners);
    CHECK(segment_owners[1] == expected_segment_owners);
    CHECK(distributed_pool.allocated_bytes() == 512);

    std::remove(path.string().c_str());
    ggml_free(ctx);
    return 0;
}
