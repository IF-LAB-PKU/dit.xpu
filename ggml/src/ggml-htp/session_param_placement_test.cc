#include "session_placement.h"
#include "htp_param_placement.h"

#include <cassert>
#include <vector>

int main() {
    const int expected[] = {0, 0, 0, 1, 1, 2, 2, 2, 3, 3};
    for (int i = 0; i < 10; ++i) {
        assert(htp_session_owner_for_ordinal(i, 10, 4) == expected[i]);
    }

    const std::vector<size_t> tensor_sizes = {128, 256, 13 * 1024 * 1024};
    const auto chunks = htp_per_tensor_chunk_sizes(tensor_sizes, 1024 * 1024 * 256);
    assert(chunks.size() == tensor_sizes.size());
    assert(chunks == tensor_sizes);
    return 0;
}
