#ifndef __HTP_PACKED_STAGING_H__
#define __HTP_PACKED_STAGING_H__

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "ggml-backend.h"

class ModelLoader;
struct ggml_tensor;

// A bounded owner for file-backed packed weights.  The allocator is injected
// so host tests can use malloc while the HTP integration can return an RPCMEM
// backend buffer and its base address.
class HtpPackedStagingPool {
public:
    struct Allocation {
        void * data = nullptr;
        ggml_backend_buffer_t buffer = nullptr;
    };
    using AllocateFn = std::function<Allocation(size_t)>;
    using FreeFn = std::function<void(const Allocation&)>;
    using LoadFn = std::function<bool(const std::string&, void*, size_t)>;

    struct Handle {
        size_t slot = static_cast<size_t>(-1);
        uint64_t generation = 0;

        bool valid() const { return slot != static_cast<size_t>(-1); }
    };

    HtpPackedStagingPool(size_t budget_bytes,
                         size_t slot_capacity,
                         AllocateFn allocate,
                         FreeFn free);
    ~HtpPackedStagingPool();

    HtpPackedStagingPool(const HtpPackedStagingPool&) = delete;
    HtpPackedStagingPool& operator=(const HtpPackedStagingPool&) = delete;

    bool acquire(size_t required_bytes, Handle* handle);
    bool place_tensor(const LoadFn& load,
                      Handle handle,
                      const std::string& name,
                      ggml_tensor* tensor);
    bool unbind_tensor(Handle handle, ggml_tensor* tensor);
    bool release(Handle* handle);

    size_t budget_bytes() const { return budget_bytes_; }
    size_t slot_capacity() const { return slot_capacity_; }
    size_t allocated_bytes() const { return allocated_bytes_; }
    size_t active_slots() const;
    size_t slot_count() const { return slots_.size(); }
    ggml_backend_buffer_t buffer(Handle handle) const;

private:
    struct Binding {
        ggml_tensor* tensor = nullptr;
        ggml_backend_buffer_t old_buffer = nullptr;
        void* old_data = nullptr;
        void* old_extra = nullptr;
    };
    struct Slot {
        Allocation allocation;
        size_t used_bytes = 0;
        uint64_t generation = 0;
        bool leased = false;
        std::vector<Binding> bindings;
    };

    bool valid_handle(Handle handle) const;
    Slot* get_slot(Handle handle);
    const Slot* get_slot(Handle handle) const;
    size_t reserve(Slot& slot, size_t bytes);

    size_t budget_bytes_;
    size_t slot_capacity_;
    size_t allocated_bytes_ = 0;
    uint64_t next_generation_ = 1;
    AllocateFn allocate_;
    FreeFn free_;
    std::vector<Slot> slots_;
};

#endif  // __HTP_PACKED_STAGING_H__
