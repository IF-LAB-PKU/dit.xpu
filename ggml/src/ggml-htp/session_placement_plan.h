#pragma once

#include "session_placement.h"

#include <cstddef>
#include <cstdint>
#include <vector>

// Immutable owner lookup for one ordered non-staging graph execution.  The
// plan is deliberately independent of backend/session handles so it can be
// shared by direct and hybrid dispatch and tested on the host.
class HtpSessionPlacementPlan {
public:
    void begin(int session_count, int htp_node_count, uint64_t generation) {
        session_count_ = session_count > 0 ? session_count : 1;
        htp_node_count_ = htp_node_count > 0 ? htp_node_count : 0;
        generation_ = generation;
        owners_.clear();
        owners_.reserve(static_cast<size_t>(htp_node_count_));
        for (int ordinal = 0; ordinal < htp_node_count_; ++ordinal) {
            owners_.push_back(htp_session_owner_for_ordinal(
                ordinal, htp_node_count_, session_count_));
        }
        valid_ = true;
    }

    void end() {
        valid_ = false;
        owners_.clear();
        htp_node_count_ = 0;
        generation_ = 0;
        session_count_ = 1;
    }

    bool valid() const { return valid_; }
    uint64_t generation() const { return generation_; }
    int session_count() const { return session_count_; }
    int htp_node_count() const { return htp_node_count_; }

    int owner_for_ordinal(int ordinal) const {
        if (!valid_ || ordinal < 0 || ordinal >= htp_node_count_) {
            return -1;
        }
        return owners_[static_cast<size_t>(ordinal)];
    }

private:
    int session_count_ = 1;
    int htp_node_count_ = 0;
    uint64_t generation_ = 0;
    bool valid_ = false;
    std::vector<int> owners_;
};
