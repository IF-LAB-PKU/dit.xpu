#pragma once

#include "htp_packed_staging.h"
#include "ggml.h"
#include <atomic>
#include <future>
#include <memory>

// Exactly one future segment, with a private bank. All pool operations and
// tensor access happen on the caller; the worker only reads copied metadata
// and fills raw reservations. The source loader must be initialized first.
class HtpWeightPrefetch {
public:
    struct Weight {
        ggml_tensor* tensor;
        std::string name;
        size_t bytes;
        ggml_type type;
        int owner;
    };
    struct Prepared {
        std::unique_ptr<HtpPackedStagingPool> pool;
        std::vector<Weight> weights;
        std::vector<HtpPackedStagingPool::Handle> handles;
        std::vector<HtpPackedStagingPool::Reservation> reservations;
        int64_t load_us = 0;
        ~Prepared() {
            if (!pool) return;
            for (auto handle : handles) {
                for (const auto& weight : weights) pool->unbind_tensor(handle, weight.tensor);
                pool->release(&handle);
            }
        }
        bool bind() {
            for (size_t i = 0; i < weights.size(); ++i) {
                if (!pool->bind_reserved(reservations[i], weights[i].tensor)) return false;
            }
            return true;
        }
    };
    ~HtpWeightPrefetch() { cancel(); }
    bool pending() const { return future_.valid(); }
    size_t allocated_bytes() const { return batch_ ? batch_->pool->allocated_bytes() : 0; }
    bool start(std::unique_ptr<HtpPackedStagingPool> bank,
               const std::vector<Weight>& weights, HtpPackedStagingPool::TypedLoadFn loader) {
        if (pending() || !bank || !loader || weights.empty()) return false;
        auto batch = std::make_unique<Prepared>();
        batch->pool = std::move(bank);
        batch->weights = weights;
        for (const auto& weight : weights) {
            HtpPackedStagingPool::Reservation reservation;
            bool reserved = false;
            for (auto handle : batch->handles) {
                if (batch->pool->session_for_handle(handle) == weight.owner &&
                    batch->pool->reserve_unbound(handle, weight.bytes, &reservation)) {
                    reserved = true;
                    break;
                }
            }
            if (!reserved) {
                HtpPackedStagingPool::Handle handle;
                if (!batch->pool->acquire_for_session(weight.bytes, weight.owner, &handle)) return false;
                batch->handles.push_back(handle);
                if (!batch->pool->reserve_unbound(handle, weight.bytes, &reservation)) return false;
            }
            batch->reservations.push_back(reservation);
        }
        // Discard old, unleased allocations only after every required slot has
        // been reserved. Active reservations remain leased through the join.
        batch->pool->reclaim_unleased();
        batch_ = std::move(batch);
        cancelled_.store(false);
        try {
            future_ = std::async(std::launch::async, [this, loader = std::move(loader)] {
                const int64_t start = ggml_time_us();
                bool ok = true;
                for (size_t i = 0; i < batch_->weights.size(); ++i) {
                    const auto& w = batch_->weights[i];
                    if (cancelled_.load() || !loader(w.name, batch_->reservations[i].data, w.bytes, w.type)) {
                        ok = false;
                        break;
                    }
                }
                batch_->load_us = ggml_time_us() - start;
                return ok;
            });
        } catch (...) {
            batch_.reset();
            return false;
        }
        ++started;
        return true;
    }
    std::unique_ptr<Prepared> take() {
        if (!pending()) return nullptr;
        const int64_t start = ggml_time_us();
        bool ok = false;
        try { ok = future_.get(); } catch (...) { ok = false; }
        wait_us += ggml_time_us() - start;
        load_us += batch_->load_us;
        if (!ok) { batch_.reset(); return nullptr; }
        return std::move(batch_);
    }
    void cancel() {
        cancelled_.store(true);
        if (future_.valid()) {
            try { future_.get(); } catch (...) {}
        }
        batch_.reset();
    }
    size_t started = 0;
    int64_t wait_us = 0;
    int64_t load_us = 0;
private:
    std::unique_ptr<Prepared> batch_;
    std::future<bool> future_;
    std::atomic<bool> cancelled_{false};
};
