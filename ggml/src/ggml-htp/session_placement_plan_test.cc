#include "session_placement_plan.h"

#include <cassert>

int main() {
    HtpSessionPlacementPlan plan;
    plan.begin(/*session_count=*/2, /*htp_node_count=*/4, /*generation=*/11);

    assert(plan.valid());
    assert(plan.generation() == 11);
    assert(plan.owner_for_ordinal(0) == 0);
    assert(plan.owner_for_ordinal(1) == 0);
    assert(plan.owner_for_ordinal(2) == 1);
    assert(plan.owner_for_ordinal(3) == 1);
    assert(plan.owner_for_ordinal(4) == -1);
    assert(plan.owner_for_ordinal(-1) == -1);

    // Rebuilding a graph view replaces, rather than mutates, the prior plan.
    plan.begin(/*session_count=*/4, /*htp_node_count=*/2, /*generation=*/12);
    assert(plan.owner_for_ordinal(0) == 0);
    assert(plan.owner_for_ordinal(1) == 2);
    assert(plan.owner_for_ordinal(2) == -1);

    plan.end();
    assert(!plan.valid());
    assert(plan.owner_for_ordinal(0) == -1);
    return 0;
}
