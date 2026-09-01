#include "ggml_graph_cut.h"

#include <limits>

int main() {
    // A graph-cut segment starts from the first session; its HTP nodes then
    // advance through all available sessions via ordinal placement.  The
    // outer segment index must not alternate between session 0 and 1.
    if (sd::ggml_graph_cut::segment_anchor_session(0, 4) != 0 ||
        sd::ggml_graph_cut::segment_anchor_session(1, 4) != 0 ||
        sd::ggml_graph_cut::segment_anchor_session(7, 4) != 0 ||
        sd::ggml_graph_cut::segment_anchor_session(2, 1) != 0 ||
        sd::ggml_graph_cut::segment_anchor_session(2, 0) != 0) {
        return 1;
    }

    sd::ggml_graph_cut::Segment segment;
    segment.compute_buffer_size      = 100;
    segment.output_bytes             = 20;
    segment.input_external_bytes     = 30;
    segment.input_previous_cut_bytes = 40;
    segment.input_param_bytes        = 101;

    const size_t shared_bytes = 100 + 20 + 30 + 40;
    const int sessions[] = { -2, 0, 1, 3, 4 };
    const size_t expected_with_negative[] = {
        shared_bytes + 101, shared_bytes + 101, shared_bytes + 101, shared_bytes + 34, shared_bytes + 26,
    };
    for (size_t i = 0; i < sizeof(sessions) / sizeof(sessions[0]); ++i) {
        if (sd::ggml_graph_cut::segment_vram_bytes_for_sessions(segment, sessions[i]) != expected_with_negative[i]) {
            return 1;
        }
    }
    if (sd::ggml_graph_cut::segment_vram_bytes_for_sessions(segment, 4, true) != shared_bytes + 101) {
        return 1;
    }

    sd::ggml_graph_cut::Segment overflowing;
    overflowing.compute_buffer_size = std::numeric_limits<size_t>::max();
    overflowing.output_bytes = 1;
    if (sd::ggml_graph_cut::segment_vram_bytes_for_sessions(overflowing, 4) !=
        std::numeric_limits<size_t>::max()) {
        return 1;
    }

    return 0;
}
