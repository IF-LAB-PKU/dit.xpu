// 验证 HTP packed DIT GGUF 的元数据闭环：写出 packed tensor 名单后，
// 读取端必须把对应 TensorStorage 标记成 HTP_PERMUTED_W16A32。

#include "../../src/model_io/gguf_io.h"

#include <ggml.h>

#include <cstdio>
#include <string>
#include <vector>

int main() {
    ggml_context* ctx = ggml_init({1024 * 1024, nullptr, false});
    GGML_ASSERT(ctx != nullptr);

    ggml_tensor* w0 = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 64, 96);
    GGML_ASSERT(w0 != nullptr);
    ggml_set_name(w0, "model.diffusion_model.test.weight");

    TensorWriteInfo info;
    info.tensor = w0;
    info.n_dims = 2;
    info.ne[0]  = 64;
    info.ne[1]  = 96;

    GGUFWriteOptions options;
    options.htp.htp_packed_dit = true;
    options.htp.htp_layout = "permuted_w16a32";
    options.htp.htp_packed_tensors.push_back("model.diffusion_model.test.weight");

    std::string error;
    GGML_ASSERT(write_gguf_file("/data/local/tmp/sdcpp-htp/packed-metadata-test.gguf", {info}, &error, &options));

    std::vector<TensorStorage> storages;
    GGUFHTPMetadata metadata;
    GGML_ASSERT(read_gguf_file("/data/local/tmp/sdcpp-htp/packed-metadata-test.gguf", storages, &error, &metadata));
    GGML_ASSERT(metadata.htp_packed_dit);
    GGML_ASSERT(metadata.htp_layout == "permuted_w16a32");
    GGML_ASSERT(metadata.htp_packed_tensors.size() == 1);
    GGML_ASSERT(metadata.htp_packed_tensors[0] == "model.diffusion_model.test.weight");
    GGML_ASSERT(storages.size() == 1);
    GGML_ASSERT(storages[0].layout == TensorDataLayout::HTP_PERMUTED_W16A32);

    ggml_free(ctx);
    std::printf("test-backend-htp-packed-gguf-metadata passed\n");
    return 0;
}
