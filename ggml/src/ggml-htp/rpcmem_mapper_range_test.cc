#include "rpcmem_mapper.h"

#include <cassert>

int main() {
    RpcMemRangeLedger ledger(/*budget=*/256);

    assert(ledger.reserve(/*fd=*/7, /*offset=*/0, /*size=*/1, /*allocation_size=*/128));
    assert(ledger.active_bytes() == 128);
    assert(ledger.cumulative_bytes() == 128);

    // Re-validating the same view is idempotent, while a disjoint range on the
    // same fd is a separate session-local view and consumes its own budget.
    assert(ledger.reserve(7, 0, 128, 128));
    assert(ledger.active_bytes() == 128);
    assert(ledger.reserve(7, 128, 64, 128));
    assert(ledger.active_bytes() == 128);
    assert(ledger.contains(7, 128, 64));

    // A new allocation that exceeds the active budget is rejected before mmap.
    assert(!ledger.reserve(8, 0, 1, 129));
    assert(ledger.active_bytes() == 128);

    // Releasing a view frees active capacity but never rewinds cumulative VA
    // accounting for the still-live backing fd.
    ledger.release(7, 0, 1);
    assert(ledger.active_bytes() == 128);
    assert(ledger.cumulative_bytes() == 128);
    assert(!ledger.contains(7, 0, 1));
    ledger.release(7, 128, 64);
    ledger.release(7, 0, 128);
    assert(ledger.active_bytes() == 0);
}
