#include "ggml-htp.h"

#include <cstring>

extern "C" {

bool ggml_backend_is_htp(ggml_backend_t) {
    return false;
}

bool ggml_backend_htp_runtime_available(void) {
    return false;
}

const char* ggml_backend_htp_runtime_error(void) {
    return "HTP is disabled in the host contract build";
}

void ggml_backend_htp_set_n_threads(ggml_backend_t, int) {}

void ggml_backend_htp_pack_f16_weight(ggml_fp16_t* dst,
                                      const ggml_fp16_t* src,
                                      int64_t k,
                                      int64_t n) {
    std::memcpy(dst, src, static_cast<size_t>(k * n) * sizeof(ggml_fp16_t));
}

void ggml_backend_htp_register_packed_f16_weight(const ggml_tensor*, ggml_tensor*) {}

void ggml_backend_htp_unregister_packed_f16_weight(const ggml_tensor*) {}

void ggml_backend_htp_set_hybrid_enabled(bool) {}

bool ggml_backend_htp_opencl_release_qkv_generation(ggml_backend_t,
                                                    uint64_t,
                                                    bool,
                                                    bool* released,
                                                    char*,
                                                    size_t) {
    if (released != nullptr) {
        *released = true;
    }
    return true;
}

}  // extern "C"
