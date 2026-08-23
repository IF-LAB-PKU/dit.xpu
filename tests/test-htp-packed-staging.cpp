#include "htp_packed_staging.h"

#include "model.h"
#include "model_io/safetensors_io.h"
#include "stable-diffusion.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <vector>

#define CHECK(condition)                                                                 \
    do {                                                                                 \
        if (!(condition)) {                                                              \
            fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #condition); \
            std::abort();                                                                \
        }                                                                                \
    } while (0)

#if !defined(GGML_USE_HTP)
extern "C" ggml_backend_buffer_type_t ggml_backend_htp_qkv_buffer_type(void) { return nullptr; }
extern "C" bool ggml_backend_htp_opencl_release_qkv_generation(ggml_backend_t, uint64_t, bool, bool* released, char*, size_t) {
    if (released != nullptr) *released = true;
    return true;
}
#endif

static TensorWriteInfo info_for(ggml_tensor* tensor) {
    TensorWriteInfo info;
    info.n_dims = ggml_n_dims(tensor);
    info.tensor = tensor;
    for (int i = 0; i < info.n_dims; ++i) info.ne[i] = tensor->ne[i];
    return info;
}

int main() {
    ggml_context* ctx = ggml_init({1024 * 1024, nullptr, false});
    CHECK(ctx != nullptr);
    ggml_tensor* source = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    ggml_set_name(source, "packed.weight");
    const float values[] = {10.f, 20.f, 30.f, 40.f};
    memcpy(source->data, values, sizeof(values));

    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto path = std::filesystem::temp_directory_path() / ("sd-htp-staging-" + std::to_string(nonce) + ".safetensors");
    std::string error;
    CHECK(write_safetensors_file(path.string(), {info_for(source)}, &error));

    ModelLoader loader;
    CHECK(loader.init_from_file(path.string()));
    ggml_tensor* destination = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    memset(destination->data, 0, ggml_nbytes(destination));

    size_t allocations = 0;
    HtpPackedStagingPool pool(
        128, 64,
        [&allocations](size_t bytes) {
            ++allocations;
            return HtpPackedStagingPool::Allocation{std::malloc(bytes), nullptr};
        },
        [](const HtpPackedStagingPool::Allocation& allocation) { std::free(allocation.data); });

    HtpPackedStagingPool::Handle handle;
    CHECK(pool.acquire(64, &handle));
    CHECK(pool.place_tensor(
        [&loader](const std::string& name, void* dst, size_t bytes) {
            return loader.reload_tensor_bytes(name, dst, bytes);
        },
        handle, "packed.weight", destination));
    CHECK(memcmp(destination->data, values, sizeof(values)) == 0);
    CHECK(!pool.release(&handle));
    CHECK(pool.unbind_tensor(handle, destination));
    CHECK(destination->buffer == nullptr && destination->data != nullptr);
    CHECK(pool.release(&handle));
    CHECK(pool.active_slots() == 0);

    HtpPackedStagingPool::Handle second;
    CHECK(pool.acquire(64, &second));
    CHECK(allocations == 1);
    CHECK(pool.release(&second));

    HtpPackedStagingPool::Handle first;
    CHECK(pool.acquire(65, &first) == false);
    std::remove(path.string().c_str());
    ggml_free(ctx);
    return 0;
}
