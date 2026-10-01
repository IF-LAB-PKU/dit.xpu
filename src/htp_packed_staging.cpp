#include "htp_packed_staging.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <utility>

#include "ggml-backend.h"
#include "ggml.h"
#include "htp_tensor_flags.h"
#include "model.h"

namespace {
constexpr size_t kAlignment = 128;

size_t align_up(size_t value) {
    return (value + kAlignment - 1) & ~(kAlignment - 1);
}
}  // namespace

bool htp_packed_qkv_node_name_is_supported(const char* name) {
    return name != nullptr &&
           (std::strcmp(name, "qkv_mul_rms_packed") == 0 ||
            std::strcmp(name, "qkv_mul_rms_packed_rope") == 0 ||
            std::strcmp(name, "qkv_mul_rms_packed_dual") == 0 ||
            std::strcmp(name, "qkv_mul_rms_packed_dual_rope") == 0);
}

std::vector<int> htp_packed_qkv_weight_src_indices(const char* name) {
    if (name == nullptr) {
        return {};
    }
    if (std::strcmp(name, "qkv_mul_rms_packed_dual") == 0 ||
        std::strcmp(name, "qkv_mul_rms_packed_dual_rope") == 0) {
        return {1, 6};
    }
    if (std::strcmp(name, "qkv_mul_rms_packed") == 0 ||
        std::strcmp(name, "qkv_mul_rms_packed_rope") == 0) {
        return {1};
    }
    return {};
}

std::vector<int> htp_packed_qkv_staging_src_indices(const char* name) {
    if (name == nullptr) {
        return {};
    }
    if (std::strcmp(name, "qkv_mul_rms_packed_dual_rope") == 0) {
        return {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    }
    if (std::strcmp(name, "qkv_mul_rms_packed_dual") == 0) {
        return {1, 2, 3, 4, 6, 7, 8, 9};
    }
    if (std::strcmp(name, "qkv_mul_rms_packed_rope") == 0) {
        return {1, 2, 3, 4, 5};
    }
    if (std::strcmp(name, "qkv_mul_rms_packed") == 0) {
        return {1, 2, 3, 4};
    }
    return {};
}

bool htp_should_defer_packed_param(bool htp_routing_active,
                                   bool packed_staging_enabled,
                                   size_t max_graph_vram_bytes,
                                   const ggml_tensor* tensor) {
    return htp_routing_active && packed_staging_enabled &&
           max_graph_vram_bytes > 0 && tensor != nullptr &&
           tensor->view_src == nullptr &&
           (tensor->flags & SD_GGML_TENSOR_FLAG_HTP_MUL_MAT_WEIGHT) != 0 &&
           (tensor->flags & SD_GGML_TENSOR_FLAG_HTP_PACKED_W16A32) != 0 &&
           tensor->type == GGML_TYPE_F16 && ggml_n_dims(tensor) == 2 &&
           ggml_is_contiguous(tensor) &&
           tensor->ne[0] % 32 == 0 && tensor->ne[1] % 32 == 0;
}

HtpNonDeferredParamsAllocation::~HtpNonDeferredParamsAllocation() {
    clear();
}

bool HtpNonDeferredParamsAllocation::allocate(ggml_context* source_ctx,
                                              ggml_backend_t backend) {
    clear();
    if (source_ctx == nullptr || backend == nullptr) {
        return false;
    }

    std::vector<ggml_tensor*> tensors_to_allocate;
    for (ggml_tensor* tensor = ggml_get_first_tensor(source_ctx);
         tensor != nullptr;
         tensor = ggml_get_next_tensor(source_ctx, tensor)) {
        const size_t bytes = ggml_nbytes(tensor);
        logical_bytes_ += bytes;
        if ((tensor->flags & SD_GGML_TENSOR_FLAG_HTP_DEFERRED_PACKED_PARAM) != 0) {
            deferred_bytes_ += bytes;
            if (tensor->buffer != nullptr || tensor->data != nullptr || tensor->extra != nullptr) {
                clear();
                return false;
            }
            continue;
        }
        if (tensor->data == nullptr) {
            tensors_to_allocate.push_back(tensor);
            host_backed_bytes_ += bytes;
        }
    }

    if (tensors_to_allocate.empty()) {
        return true;
    }

    ggml_init_params params;
    params.mem_size = tensors_to_allocate.size() * ggml_tensor_overhead();
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    allocation_ctx_ = ggml_init(params);
    if (allocation_ctx_ == nullptr) {
        clear();
        return false;
    }

    bindings_.reserve(tensors_to_allocate.size());
    for (ggml_tensor* tensor : tensors_to_allocate) {
        ggml_tensor* allocated = ggml_dup_tensor(allocation_ctx_, tensor);
        ggml_set_name(allocated, tensor->name);
        bindings_.push_back({tensor, allocated});
    }

    buffer_ = ggml_backend_alloc_ctx_tensors(allocation_ctx_, backend);
    if (buffer_ == nullptr) {
        clear();
        return false;
    }
    ggml_backend_buffer_set_usage(buffer_, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    for (const auto& binding : bindings_) {
        binding.first->buffer = binding.second->buffer;
        binding.first->data = binding.second->data;
        binding.first->extra = binding.second->extra;
    }
    return true;
}

void HtpNonDeferredParamsAllocation::clear() {
    for (const auto& binding : bindings_) {
        if (binding.first != nullptr) {
            binding.first->buffer = nullptr;
            binding.first->data = nullptr;
            binding.first->extra = nullptr;
        }
    }
    bindings_.clear();
    if (buffer_ != nullptr) {
        ggml_backend_buffer_free(buffer_);
        buffer_ = nullptr;
    }
    if (allocation_ctx_ != nullptr) {
        ggml_free(allocation_ctx_);
        allocation_ctx_ = nullptr;
    }
    logical_bytes_ = 0;
    host_backed_bytes_ = 0;
    deferred_bytes_ = 0;
}

size_t HtpNonDeferredParamsAllocation::buffer_size() const {
    return buffer_ != nullptr ? ggml_backend_buffer_get_size(buffer_) : 0;
}

HtpPackedStagingPool::HtpPackedStagingPool(size_t budget_bytes,
                                           size_t slot_capacity,
                                           AllocateFn allocate,
                                           FreeFn free,
                                           DiagnosticFn diagnostic)
    : HtpPackedStagingPool(
          budget_bytes,
          slot_capacity,
          1,
          [allocate = std::move(allocate)](size_t bytes, int) {
              return allocate != nullptr ? allocate(bytes) : Allocation{};
          },
          std::move(free),
          std::move(diagnostic)) {}

HtpPackedStagingPool::HtpPackedStagingPool(size_t budget_bytes,
                                           size_t slot_capacity,
                                           int session_count,
                                           SessionAllocateFn allocate,
                                           FreeFn free,
                                           DiagnosticFn diagnostic)
    : budget_bytes_(budget_bytes),
      slot_capacity_(slot_capacity),
      session_count_(std::max(1, session_count)),
      allocate_(std::move(allocate)),
      free_(std::move(free)),
      diagnostic_(std::move(diagnostic)) {
    emit(DiagnosticKind::PoolCreated, 0, static_cast<size_t>(-1), nullptr, false);
}

HtpPackedStagingPool::~HtpPackedStagingPool() {
    for (Slot& slot : slots_) {
        if (!slot.leased && slot.allocation.data != nullptr) {
            free_slot_allocation(slot);
        }
    }
}

bool HtpPackedStagingPool::valid_handle(Handle handle) const {
    return handle.valid() && handle.slot < slots_.size() &&
           slots_[handle.slot].leased && slots_[handle.slot].generation == handle.generation;
}

HtpPackedStagingPool::Slot* HtpPackedStagingPool::get_slot(Handle handle) {
    return valid_handle(handle) ? &slots_[handle.slot] : nullptr;
}

const HtpPackedStagingPool::Slot* HtpPackedStagingPool::get_slot(Handle handle) const {
    return valid_handle(handle) ? &slots_[handle.slot] : nullptr;
}

size_t HtpPackedStagingPool::reserve(Slot& slot, size_t bytes) {
    const size_t offset = align_up(slot.used_bytes);
    if (offset > slot_capacity_ || bytes > slot_capacity_ - offset) {
        return static_cast<size_t>(-1);
    }
    slot.used_bytes = offset + bytes;
    return offset;
}

void HtpPackedStagingPool::emit(DiagnosticKind kind,
                                size_t requested_bytes,
                                size_t slot_index,
                                const Slot* slot,
                                bool reused,
                                const std::string& tensor_name) const {
    if (!diagnostic_) {
        return;
    }
    DiagnosticEvent event;
    event.kind = kind;
    event.requested_bytes = requested_bytes;
    event.slot_index = slot_index;
    event.reused = reused;
    event.total_allocated_bytes = allocated_bytes_;
    event.tensor_name = tensor_name;
    if (slot != nullptr) {
        event.logical_session_index = slot->allocation.logical_session_index;
        event.domain = slot->allocation.domain;
        event.session_allocated_bytes = session_allocated_bytes(
            slot->allocation.logical_session_index);
    }
    diagnostic_(event);
}

bool HtpPackedStagingPool::acquire(size_t required_bytes, Handle* handle) {
    return acquire_for_session(required_bytes, -1, handle);
}

bool HtpPackedStagingPool::acquire_for_session(size_t required_bytes,
                                               int requested_session,
                                               Handle* handle) {
    if (handle == nullptr || required_bytes > slot_capacity_ || allocate_ == nullptr) {
        return false;
    }
    if (requested_session >= session_count_) {
        return false;
    }
    *handle = {};

    std::vector<size_t> active_slots(static_cast<size_t>(session_count_), 0);
    for (const Slot& slot : slots_) {
        if (slot.leased && slot.allocation.logical_session_index >= 0 &&
            slot.allocation.logical_session_index < session_count_) {
            ++active_slots[static_cast<size_t>(slot.allocation.logical_session_index)];
        }
    }
    size_t reusable_index = static_cast<size_t>(-1);
    size_t reusable_active_slots = std::numeric_limits<size_t>::max();
    int reusable_session = std::numeric_limits<int>::max();
    for (size_t i = 0; i < slots_.size(); ++i) {
        const Slot& slot = slots_[i];
        const int session = slot.allocation.logical_session_index;
        if (slot.leased || slot.allocation.data == nullptr ||
            session < 0 || session >= session_count_ ||
            (requested_session >= 0 && session != requested_session)) {
            continue;
        }
        const size_t session_active_slots = active_slots[static_cast<size_t>(session)];
        if (session_active_slots < reusable_active_slots ||
            (session_active_slots == reusable_active_slots && session < reusable_session)) {
            reusable_index = i;
            reusable_active_slots = session_active_slots;
            reusable_session = session;
        }
    }
    if (reusable_index != static_cast<size_t>(-1)) {
        Slot& slot = slots_[reusable_index];
        slot.leased = true;
        slot.used_bytes = 0;
        slot.bindings.clear();
        slot.generation = next_generation_++;
        *handle = {reusable_index, slot.generation};
        emit(DiagnosticKind::SlotAcquired, required_bytes, reusable_index, &slot, true);
        return true;
    }

    if (slot_capacity_ == 0 || slot_capacity_ > budget_bytes_ ||
        allocated_bytes_ > budget_bytes_ - slot_capacity_) {
        return false;
    }
    int owner_session = requested_session >= 0 ? requested_session : 0;
    size_t owner_bytes = session_allocated_bytes(owner_session);
    if (requested_session < 0) {
        for (int session = 1; session < session_count_; ++session) {
            const size_t allocated = session_allocated_bytes(session);
            if (allocated < owner_bytes) {
                owner_session = session;
                owner_bytes = allocated;
            }
        }
    }
    Allocation allocation = allocate_(slot_capacity_, owner_session);
    if (allocation.data == nullptr) {
        return false;
    }
    allocation.logical_session_index = owner_session;
    Slot slot;
    slot.allocation = allocation;
    slot.leased = true;
    slot.generation = next_generation_++;
    slots_.push_back(std::move(slot));
    allocated_bytes_ += slot_capacity_;
    session_allocated_bytes_[allocation.logical_session_index] += slot_capacity_;
    *handle = {slots_.size() - 1, slots_.back().generation};
    emit(DiagnosticKind::SlotAcquired, required_bytes, handle->slot,
         &slots_.back(), false);
    return true;
}

bool HtpPackedStagingPool::place_tensor(const LoadFn& load,
                                        Handle handle,
                                        const std::string& name,
                                        ggml_tensor* tensor) {
    Slot* slot = get_slot(handle);
    if (slot == nullptr || tensor == nullptr || !load) {
        return false;
    }
    const size_t bytes = ggml_nbytes(tensor);
    if (bytes != ggml_nbytes(tensor)) {
        return false;
    }
    const size_t offset = reserve(*slot, bytes);
    if (offset == static_cast<size_t>(-1) ||
        !load(name, static_cast<uint8_t*>(slot->allocation.data) + offset, bytes)) {
        return false;
    }
    slot->bindings.push_back({tensor, tensor->buffer, tensor->data, tensor->extra,
                              slot->allocation.buffer,
                              static_cast<uint8_t*>(slot->allocation.data) + offset});
    tensor->buffer = slot->allocation.buffer;
    tensor->data = static_cast<uint8_t*>(slot->allocation.data) + offset;
    tensor->extra = nullptr;
    emit(DiagnosticKind::TensorPlaced, bytes, handle.slot, slot, false, name);
    return true;
}

bool HtpPackedStagingPool::place_tensor(const TypedLoadFn& load,
                                        Handle handle,
                                        const std::string& name,
                                        ggml_tensor* tensor) {
    if (!load || tensor == nullptr) {
        return false;
    }
    return place_tensor_impl(
        [&load, tensor](const std::string& load_name, void* dst, size_t bytes) {
            return load(load_name, dst, bytes, tensor->type);
        },
        handle, name, tensor);
}

bool HtpPackedStagingPool::place_tensor_impl(const LoadFn& load,
                                             Handle handle,
                                             const std::string& name,
                                             ggml_tensor* tensor) {
    return place_tensor(load, handle, name, tensor);
}

bool HtpPackedStagingPool::rebind_tensor(Handle handle, ggml_tensor* tensor) {
    Slot* slot = get_slot(handle);
    if (slot == nullptr || tensor == nullptr) {
        return false;
    }
    auto it = std::find_if(slot->bindings.begin(), slot->bindings.end(),
                           [tensor](const Binding& binding) { return binding.tensor == tensor; });
    if (it == slot->bindings.end()) {
        return false;
    }
    tensor->buffer = it->staged_buffer;
    tensor->data = it->staged_data;
    tensor->extra = nullptr;
    return true;
}

bool HtpPackedStagingPool::unbind_tensor(Handle handle, ggml_tensor* tensor) {
    Slot* slot = get_slot(handle);
    if (slot == nullptr || tensor == nullptr) {
        return false;
    }
    auto it = std::find_if(slot->bindings.begin(), slot->bindings.end(),
                           [tensor](const Binding& binding) { return binding.tensor == tensor; });
    if (it == slot->bindings.end()) {
        return false;
    }
    tensor->buffer = it->old_buffer;
    tensor->data = it->old_data;
    tensor->extra = it->old_extra;
    slot->bindings.erase(it);
    return true;
}

bool HtpPackedStagingPool::release(Handle* handle) {
    if (handle == nullptr) {
        return false;
    }
    Slot* slot = get_slot(*handle);
    if (slot == nullptr || !slot->bindings.empty()) {
        return false;
    }
    slot->leased = false;
    slot->used_bytes = 0;
    emit(DiagnosticKind::SlotReleased, 0, handle->slot, slot, false);
    *handle = {};
    return true;
}

void HtpPackedStagingPool::free_slot_allocation(Slot& slot) {
    GGML_ASSERT(!slot.leased && slot.bindings.empty());
    if (slot.allocation.data == nullptr) {
        return;
    }

    const int session = slot.allocation.logical_session_index;
    if (free_) {
        free_(slot.allocation);
    }
    GGML_ASSERT(allocated_bytes_ >= slot_capacity_);
    allocated_bytes_ -= slot_capacity_;
    auto session_it = session_allocated_bytes_.find(session);
    GGML_ASSERT(session_it != session_allocated_bytes_.end());
    GGML_ASSERT(session_it->second >= slot_capacity_);
    session_it->second -= slot_capacity_;
    if (session_it->second == 0) {
        session_allocated_bytes_.erase(session_it);
    }
    slot.allocation = {};
}

size_t HtpPackedStagingPool::reclaim_unleased() {
    size_t reclaimed = 0;
    for (Slot& slot : slots_) {
        if (!slot.leased && slot.allocation.data != nullptr) {
            free_slot_allocation(slot);
            ++reclaimed;
        }
    }
    return reclaimed;
}

size_t HtpPackedStagingPool::active_slots() const {
    size_t count = 0;
    for (const Slot& slot : slots_) {
        count += slot.leased ? 1 : 0;
    }
    return count;
}

size_t HtpPackedStagingPool::session_allocated_bytes(int logical_session_index) const {
    const auto it = session_allocated_bytes_.find(logical_session_index);
    return it != session_allocated_bytes_.end() ? it->second : 0;
}

ggml_backend_buffer_t HtpPackedStagingPool::buffer(Handle handle) const {
    const Slot* slot = get_slot(handle);
    return slot != nullptr ? slot->allocation.buffer : nullptr;
}

int HtpPackedStagingPool::session_for_handle(Handle handle) const {
    const Slot* slot = get_slot(handle);
    return slot != nullptr ? slot->allocation.logical_session_index : -1;
}
