#include "rpcmem_mapper.h"

#include <cassert>

int main() {
    RpcMemRangeLedger ledger(/*budget=*/256);

    assert(ledger.reserve(/*fd=*/7, /*offset=*/0, /*size=*/128));
    assert(ledger.active_bytes() == 128);
    assert(ledger.cumulative_bytes() == 128);

    // Re-validating the same view is idempotent, while a disjoint range on the
    // same fd is a separate session-local view and consumes its own budget.
    assert(ledger.reserve(7, 0, 128));
    assert(ledger.active_bytes() == 128);
    assert(ledger.reserve(7, 128, 64));
    assert(ledger.active_bytes() == 192);
    assert(ledger.contains(7, 128, 64));

    // A new range that exceeds the active budget is rejected before mmap.
    assert(!ledger.reserve(7, 192, 128));
    assert(ledger.active_bytes() == 192);

    // Releasing a view frees active capacity but never rewinds cumulative VA
    // accounting for the still-live backing fd.
    ledger.release(7, 0, 128);
    assert(ledger.active_bytes() == 64);
    assert(ledger.cumulative_bytes() == 192);
    assert(!ledger.contains(7, 0, 128));
}
