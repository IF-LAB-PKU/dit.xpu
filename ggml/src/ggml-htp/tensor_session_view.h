#pragma once

#include "htp_session.h"

#include <cstdint>
#include <map>
#include <string>
#include <tuple>

enum class HtpTensorRole : uint8_t {
    Input,
    Output,
    Activation,
    Weight,
};

struct HtpTensorIdentity {
    std::string name;
    uint64_t backing_id = 0;

    bool operator<(const HtpTensorIdentity & other) const {
        return std::tie(backing_id, name) < std::tie(other.backing_id, other.name);
    }
};

struct HtpTensorSessionView {
    uint32_t session_id = 0;
    uint32_t domain = 3;
    int fd = -1;
    uint64_t offset = 0;
    uint64_t size = 0;
    HtpTensorRole role = HtpTensorRole::Activation;
    uintptr_t dsp_address = 0;
    uint32_t map_refcount = 0;
    uint64_t producer_epoch = 0;
};

class HtpTensorSessionViews {
public:
    bool ensure_view(const HtpTensorIdentity & tensor,
                     const HtpSessionRuntime & session,
                     int fd,
                     uint64_t offset,
                     uint64_t size,
                     HtpTensorRole role) {
        if (!session.initialized || size == 0 || fd < 0) {
            return false;
        }

        const Key key{tensor, session.id};
        auto it = views_.find(key);
        if (it != views_.end()) {
            auto & view = it->second;
            if (view.fd != fd || view.offset != offset || view.size != size || view.role != role ||
                view.domain != session.domain) {
                return false;
            }
            ++view.map_refcount;
            return true;
        }

        HtpTensorSessionView view;
        view.session_id = session.id;
        view.domain = session.domain;
        view.fd = fd;
        view.offset = offset;
        view.size = size;
        view.role = role;
        view.dsp_address = next_dsp_address_++;
        view.map_refcount = 1;
        views_.emplace(key, view);
        return true;
    }

    bool has_view(const HtpTensorIdentity & tensor, uint32_t session_id) const {
        return views_.find(Key{tensor, session_id}) != views_.end();
    }

    bool release_view(const HtpTensorIdentity & tensor, uint32_t session_id) {
        auto it = views_.find(Key{tensor, session_id});
        if (it == views_.end() || it->second.map_refcount == 0) {
            return false;
        }
        if (--it->second.map_refcount == 0) {
            views_.erase(it);
        }
        return true;
    }

    const HtpTensorSessionView & view(const HtpTensorIdentity & tensor, uint32_t session_id) const {
        return views_.at(Key{tensor, session_id});
    }

    bool publish_output(const HtpTensorIdentity & tensor, uint32_t producer_session_id) {
        auto it = views_.find(Key{tensor, producer_session_id});
        if (it == views_.end() || it->second.map_refcount == 0) {
            return false;
        }
        auto & state = tensors_[tensor];
        ++state.latest_epoch;
        state.producer_session_id = producer_session_id;
        state.has_value = true;
        it->second.producer_epoch = state.latest_epoch;
        return true;
    }

    bool prepare_input(const HtpTensorIdentity & tensor, uint32_t consumer_session_id) {
        auto view_it = views_.find(Key{tensor, consumer_session_id});
        if (view_it == views_.end() || view_it->second.map_refcount == 0) {
            return false;
        }
        auto state_it = tensors_.find(tensor);
        if (state_it == tensors_.end() || !state_it->second.has_value) {
            return false;
        }
        view_it->second.producer_epoch = state_it->second.latest_epoch;
        return true;
    }

    bool prepare_input(const HtpTensorIdentity & tensor,
                       uint32_t consumer_session_id,
                       uint64_t expected_producer_epoch) {
        auto state_it = tensors_.find(tensor);
        if (state_it == tensors_.end() || !state_it->second.has_value ||
            state_it->second.latest_epoch != expected_producer_epoch) {
            return false;
        }
        return prepare_input(tensor, consumer_session_id);
    }

private:
    struct Key {
        HtpTensorIdentity tensor;
        uint32_t session_id;

        bool operator<(const Key & other) const {
            return std::tie(tensor, session_id) < std::tie(other.tensor, other.session_id);
        }
    };

    struct TensorState {
        uint64_t latest_epoch = 0;
        uint32_t producer_session_id = 0;
        bool has_value = false;
    };

    std::map<Key, HtpTensorSessionView> views_;
    std::map<HtpTensorIdentity, TensorState> tensors_;
    uintptr_t next_dsp_address_ = 0x1000;
};
