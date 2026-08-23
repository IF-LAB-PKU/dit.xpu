#include "htp_packed_staging.h"

#include <algorithm>
#include <cstring>
#include <utility>

#include "ggml-backend.h"
#include "ggml.h"
#include "model.h"

namespace {
constexpr size_t kAlignment = 128;

size_t align_up(size_t value) {
    return (value + kAlignment - 1) & ~(kAlignment - 1);
}
}  // namespace

HtpPackedStagingPool::HtpPackedStagingPool(size_t budget_bytes,
                                           size_t slot_capacity,
                                           AllocateFn allocate,
                                           FreeFn free)
    : budget_bytes_(budget_bytes),
      slot_capacity_(slot_capacity),
      allocate_(std::move(allocate)),
      free_(std::move(free)) {}

HtpPackedStagingPool::~HtpPackedStagingPool() {
    for (Slot& slot : slots_) {
        if (!slot.leased && slot.allocation.data != nullptr && free_) {
            free_(slot.allocation);
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

bool HtpPackedStagingPool::acquire(size_t required_bytes, Handle* handle) {
    if (handle == nullptr || required_bytes > slot_capacity_ || allocate_ == nullptr) {
        return false;
    }
    *handle = {};

    for (size_t i = 0; i < slots_.size(); ++i) {
        Slot& slot = slots_[i];
        if (!slot.leased && slot.allocation.data != nullptr) {
            slot.leased = true;
            slot.used_bytes = 0;
            slot.bindings.clear();
            slot.generation = next_generation_++;
            *handle = {i, slot.generation};
            return true;
        }
    }

    if (slot_capacity_ == 0 || slot_capacity_ > budget_bytes_ ||
        allocated_bytes_ > budget_bytes_ - slot_capacity_) {
        return false;
    }
    Allocation allocation = allocate_(slot_capacity_);
    if (allocation.data == nullptr) {
        return false;
    }
    Slot slot;
    slot.allocation = allocation;
    slot.leased = true;
    slot.generation = next_generation_++;
    slots_.push_back(std::move(slot));
    allocated_bytes_ += slot_capacity_;
    *handle = {slots_.size() - 1, slots_.back().generation};
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
    slot->bindings.push_back({tensor, tensor->buffer, tensor->data, tensor->extra});
    tensor->buffer = slot->allocation.buffer;
    tensor->data = static_cast<uint8_t*>(slot->allocation.data) + offset;
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
    *handle = {};
    return true;
}

size_t HtpPackedStagingPool::active_slots() const {
    size_t count = 0;
    for (const Slot& slot : slots_) {
        count += slot.leased ? 1 : 0;
    }
    return count;
}

ggml_backend_buffer_t HtpPackedStagingPool::buffer(Handle handle) const {
    const Slot* slot = get_slot(handle);
    return slot != nullptr ? slot->allocation.buffer : nullptr;
}
