#include "session_ownership.h"

#include <cassert>

int main() {
    HtpSessionOwnership ownership;
    const HtpSessionOwner default_session{0, 3};
    const HtpSessionOwner reserved_session{1, 7};

    assert(!ownership.has("cut"));
    assert(ownership.record("cut", default_session));
    assert(ownership.has("cut"));
    assert(!ownership.needs_transfer("cut", default_session));
    assert(ownership.needs_transfer("cut", reserved_session));

    // A cache name cannot silently change its backing session.
    assert(!ownership.record("cut", reserved_session));
    assert(ownership.owner("cut").session == 0);
    assert(ownership.owner("cut").domain == 3);

    // A view is tracked by its backing cache name and follows the same owner.
    assert(ownership.record("cut:view", default_session));
    assert(ownership.needs_transfer("cut:view", reserved_session));
    return 0;
}
