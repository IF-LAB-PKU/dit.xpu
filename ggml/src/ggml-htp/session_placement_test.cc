#include "session_placement.h"

#include <cassert>

int main() {
    // Ten HTP nodes split across four sessions produce contiguous ranges:
    // 0,0,0,1,1,2,2,2,3,3.
    const int expected[] = {0, 0, 0, 1, 1, 2, 2, 2, 3, 3};
    for (int ordinal = 0; ordinal < 10; ++ordinal) {
        assert(htp_session_owner_for_ordinal(ordinal, 10, 4) == expected[ordinal]);
    }

    // A graph with fewer HTP nodes than sessions must not produce invalid ids.
    assert(htp_session_owner_for_ordinal(0, 2, 4) == 0);
    assert(htp_session_owner_for_ordinal(1, 2, 4) == 2);

    // Invalid ordinals/counts and a single session remain deterministic.
    assert(htp_session_owner_for_ordinal(-1, 10, 4) == 0);
    assert(htp_session_owner_for_ordinal(10, 10, 4) == 3);
    assert(htp_session_owner_for_ordinal(3, 10, 0) == 0);
    assert(htp_session_owner_for_ordinal(3, 10, 1) == 0);
    return 0;
}
