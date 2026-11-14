#pragma once

#include "ggml.h"
#include <array>
#include <cstdio>
#include <string>

// Exclusive host spans for one runner request, including all graph-cut views.
// No per-node I/O and no clock reads when disabled. Async compute measures the
// backend call; completion waits are charged to the span that performs them.
class RunnerProfile {
public:
    enum Phase { Build, Plan, PackedPrepare, Staging, Offload, Reserve, Allocate,
                 CopyIn, Compute, Cache, Readback, Release, Count };
    class Timer {
    public:
        Timer(RunnerProfile& owner, Phase phase) : owner_(owner), phase_(phase),
            start_(owner.enabled_ ? ggml_time_us() : 0), active_(owner.enabled_) {}
        Timer(const Timer&) = delete;
        ~Timer() { finish(); }
        void finish() {
            if (active_) {
                owner_.us_[phase_] += ggml_time_us() - start_;
                ++owner_.calls_[phase_];
                active_ = false;
            }
        }
    private:
        RunnerProfile& owner_;
        Phase phase_;
        int64_t start_;
        bool active_;
    };
    class Request {
    public:
        Request(RunnerProfile& owner, bool enabled, const std::string& name)
            : owner_(owner), name_(enabled ? name : "") {
            owner_.enabled_ = enabled;
            owner_.us_.fill(0);
            owner_.calls_.fill(0);
            start_ = enabled ? ggml_time_us() : 0;
        }
        Request(const Request&) = delete;
        ~Request() {
            if (!owner_.enabled_) return;
            const int64_t total = ggml_time_us() - start_;
            int64_t accounted = 0;
            static const char* names[] = {"build", "plan", "packed_prepare", "staging", "offload",
                "reserve", "allocate", "copy_in", "compute", "cache", "readback", "release"};
            std::string line = "runner_profile name=\"" + name_ + "\" status=" +
                (success ? "ok" : "failed") + " total_us=" + std::to_string(total);
            for (int i = 0; i < Count; ++i) {
                accounted += owner_.us_[i];
                line += " " + std::string(names[i]) + "_us=" + std::to_string(owner_.us_[i]);
            }
            line += " other_us=" + std::to_string(total - accounted) +
                " graphs=" + std::to_string(owner_.calls_[Compute]) +
                " reserves=" + std::to_string(owner_.calls_[Reserve]);
            std::fprintf(stderr, "%s\n", line.c_str());
            owner_.enabled_ = false;
        }
        bool success = false;
    private:
        RunnerProfile& owner_;
        std::string name_;
        int64_t start_ = 0;
    };
    Timer timer(Phase phase) { return Timer(*this, phase); }
    template <typename F> decltype(auto) measure(Phase phase, F&& fn) {
        auto span = timer(phase);
        return fn();
    }
private:
    bool enabled_ = false;
    std::array<int64_t, Count> us_{};
    std::array<size_t, Count> calls_{};
};
