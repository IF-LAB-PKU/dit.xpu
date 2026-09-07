#include "../../src/htp_packed_staging.h"

#include <cstdio>
#include <cstring>

namespace {

void check(bool condition, const char * expression) {
    if (!condition) {
        std::fprintf(stderr, "CHECK failed: %s\n", expression);
        std::abort();
    }
}

#define CHECK(condition) check((condition), #condition)

} // namespace

int main() {
    CHECK(htp_packed_qkv_node_name_is_supported("qkv_mul_rms_packed"));
    CHECK(htp_packed_qkv_node_name_is_supported("qkv_mul_rms_packed_rope"));
    CHECK(htp_packed_qkv_node_name_is_supported("qkv_mul_rms_packed_dual"));
    CHECK(htp_packed_qkv_node_name_is_supported("qkv_mul_rms_packed_dual_rope"));
    CHECK(!htp_packed_qkv_node_name_is_supported("qkv_mul_rms"));
    CHECK(!htp_packed_qkv_node_name_is_supported(nullptr));

    CHECK(htp_packed_qkv_weight_src_indices("qkv_mul_rms_packed") ==
          std::vector<int>({1}));
    CHECK(htp_packed_qkv_weight_src_indices("qkv_mul_rms_packed_rope") ==
          std::vector<int>({1}));
    CHECK(htp_packed_qkv_weight_src_indices("qkv_mul_rms_packed_dual") ==
          std::vector<int>({1, 6}));
    CHECK(htp_packed_qkv_weight_src_indices("qkv_mul_rms_packed_dual_rope") ==
          std::vector<int>({1, 6}));
    CHECK(htp_packed_qkv_weight_src_indices("unknown").empty());
    return 0;
}
