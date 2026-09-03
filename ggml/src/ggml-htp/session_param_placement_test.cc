#include "session_placement.h"

#include <cassert>

int main() {
    const int expected[] = {0, 0, 0, 1, 1, 2, 2, 2, 3, 3};
    for (int i = 0; i < 10; ++i) {
        assert(htp_session_owner_for_ordinal(i, 10, 4) == expected[i]);
    }

    // A parameter chunk planner uses the same contiguous interval policy as
    // operation placement; this is intentionally independent of staging.
    assert(htp_session_owner_for_ordinal(0, 2, 2) == 0);
    assert(htp_session_owner_for_ordinal(1, 2, 2) == 1);
    return 0;
}
