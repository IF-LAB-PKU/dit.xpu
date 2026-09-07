#ifndef __HTP_PACKED_STAGING_H__
#define __HTP_PACKED_STAGING_H__

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include "ggml-backend.h"

class ModelLoader;
struct ggml_context;
struct ggml_tensor;

bool htp_should_defer_packed_param(bool htp_routing_active,
                                   bool packed_staging_enabled,
                                   size_t max_graph_vram_bytes,
                                   const ggml_tensor* tensor);

// Packed QKV custom nodes carry their offline-packed projection weights at
// different source positions for single- and dual-stream variants.
bool htp_packed_qkv_node_name_is_supported(const char* name);
std::vector<int> htp_packed_qkv_weight_src_indices(const char* name);

class HtpNonDeferredParamsAllocation {
public:
    ~HtpNonDeferredParamsAllocation();

    HtpNonDeferredParamsAllocation() = default;
    HtpNonDeferredParamsAllocation(const HtpNonDeferredParamsAllocation&) = delete;
    HtpNonDeferredParamsAllocation& operator=(const HtpNonDeferredParamsAllocation&) = delete;

    bool allocate(ggml_context* source_ctx, ggml_backend_t backend);
    void clear();

    size_t logical_bytes() const { return logical_bytes_; }
    size_t host_backed_bytes() const { return host_backed_bytes_; }
    size_t deferred_bytes() const { return deferred_bytes_; }
    size_t buffer_size() const;

private:
    ggml_context* allocation_ctx_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;
    std::vector<std::pair<ggml_tensor*, ggml_tensor*>> bindings_;
    size_t logical_bytes_ = 0;
    size_t host_backed_bytes_ = 0;
    size_t deferred_bytes_ = 0;
};

// A bounded owner for file-backed packed weights.  The allocator is injected
// so host tests can use malloc while the HTP integration can return an RPCMEM
// backend buffer and its base address.
class HtpPackedStagingPool {
public:
    struct Allocation {
        void * data = nullptr;
        ggml_backend_buffer_t buffer = nullptr;
        int logical_session_index = 0;
        int domain = -1;
    };
    using AllocateFn = std::function<Allocation(size_t)>;
    using SessionAllocateFn = std::function<Allocation(size_t, int)>;
    using FreeFn = std::function<void(const Allocation&)>;
    using LoadFn = std::function<bool(const std::string&, void*, size_t)>;

    enum class DiagnosticKind {
        PoolCreated,
        SlotAcquired,
        TensorPlaced,
        SlotReleased,
    };
    struct DiagnosticEvent {
        DiagnosticKind kind = DiagnosticKind::PoolCreated;
        size_t requested_bytes = 0;
        size_t slot_index = static_cast<size_t>(-1);
        int logical_session_index = -1;
        int domain = -1;
        size_t session_allocated_bytes = 0;
        size_t total_allocated_bytes = 0;
        bool reused = false;
        std::string tensor_name;
    };
    using DiagnosticFn = std::function<void(const DiagnosticEvent&)>;

    struct Handle {
        size_t slot = static_cast<size_t>(-1);
        uint64_t generation = 0;

        bool valid() const { return slot != static_cast<size_t>(-1); }
    };

    HtpPackedStagingPool(size_t budget_bytes,
                         size_t slot_capacity,
                         AllocateFn allocate,
                         FreeFn free,
                         DiagnosticFn diagnostic = {});
    HtpPackedStagingPool(size_t budget_bytes,
                         size_t slot_capacity,
                         int session_count,
                         SessionAllocateFn allocate,
                         FreeFn free,
                         DiagnosticFn diagnostic = {});
    ~HtpPackedStagingPool();

    HtpPackedStagingPool(const HtpPackedStagingPool&) = delete;
    HtpPackedStagingPool& operator=(const HtpPackedStagingPool&) = delete;

    bool acquire(size_t required_bytes, Handle* handle);
    // Acquire a slice from a chunk physically owned by the requested session.
    // The session is a mapping/domain affinity, not a graph-cut owner.
    bool acquire_for_session(size_t required_bytes,
                             int logical_session_index,
                             Handle* handle);
    bool place_tensor(const LoadFn& load,
                      Handle handle,
                      const std::string& name,
                      ggml_tensor* tensor);
    // Rebind an already loaded tensor to its slot without invoking the loader
    // again. This is used when a graph view temporarily swaps tensor buffers.
    bool rebind_tensor(Handle handle, ggml_tensor* tensor);
    bool unbind_tensor(Handle handle, ggml_tensor* tensor);
    bool release(Handle* handle);
    // Destroy allocations left unused after the next segment has acquired all
    // slots it can reuse. Active leases are never reclaimed.
    size_t reclaim_unleased();

    size_t budget_bytes() const { return budget_bytes_; }
    size_t slot_capacity() const { return slot_capacity_; }
    size_t allocated_bytes() const { return allocated_bytes_; }
    size_t active_slots() const;
    size_t slot_count() const { return slots_.size(); }
    size_t session_allocated_bytes(int logical_session_index) const;
    int session_for_handle(Handle handle) const;
    ggml_backend_buffer_t buffer(Handle handle) const;

private:
    struct Binding {
        ggml_tensor* tensor = nullptr;
        ggml_backend_buffer_t old_buffer = nullptr;
        void* old_data = nullptr;
        void* old_extra = nullptr;
        ggml_backend_buffer_t staged_buffer = nullptr;
        void* staged_data = nullptr;
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
    void free_slot_allocation(Slot& slot);
    void emit(DiagnosticKind kind,
              size_t requested_bytes,
              size_t slot_index,
              const Slot* slot,
              bool reused,
              const std::string& tensor_name = {}) const;

    size_t budget_bytes_;
    size_t slot_capacity_;
    int session_count_ = 1;
    size_t allocated_bytes_ = 0;
    uint64_t next_generation_ = 1;
    SessionAllocateFn allocate_;
    FreeFn free_;
    DiagnosticFn diagnostic_;
    std::unordered_map<int, size_t> session_allocated_bytes_;
    std::vector<Slot> slots_;
};

// Returns whether a tensor is currently backed by the active staging view.
// Keeping this predicate in the staging module makes parameter offload code
// use the same ownership definition as the staging executor.
inline bool htp_packed_staging_contains(const std::vector<ggml_tensor*>& staged_tensors,
                                        const ggml_tensor* tensor) {
    return tensor != nullptr &&
           std::find(staged_tensors.begin(), staged_tensors.end(), tensor) != staged_tensors.end();
}

inline int htp_packed_staging_owner_count(bool graph_cut_active,
                                          bool multi_session_active,
                                          int initialized_session_count) {
    return graph_cut_active && multi_session_active && initialized_session_count > 1
        ? initialized_session_count : 1;
}

#endif  // __HTP_PACKED_STAGING_H__
