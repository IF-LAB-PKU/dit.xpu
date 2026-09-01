#include "htp_session_dispatch.h"

#include <cassert>
#include <vector>

int main() {
    HtpSessionDispatchContext context(/*session_count=*/4);
    context.begin_graph_view(/*htp_node_count=*/8, /*graph_view_id=*/17);

    const int expected[] = {0, 0, 1, 1, 2, 2, 3, 3};
    for (int ordinal = 0; ordinal < 8; ++ordinal) {
        assert(context.next_htp_owner() == expected[ordinal]);
    }

    // A new graph view follows the same full-graph policy and starts its
    // ordinal sequence from the first HTP node, independent of segment index.
    context.begin_graph_view(/*htp_node_count=*/4, /*graph_view_id=*/18);
    assert(context.next_htp_owner() == 0);
    assert(context.next_htp_owner() == 1);
    assert(context.next_htp_owner() == 2);
    assert(context.next_htp_owner() == 3);

    assert(context.acquire(2));
    assert(context.active_session() == 2);
    assert(context.acquire(2));
    assert(context.switch_count() == 1);
    assert(context.acquire(0));
    assert(context.active_session() == 0);
    assert(context.switch_count() == 2);
    return 0;
}
