#include "../examples/cli/htp_defaults.hpp"
#include <cassert>
#include <cstdio>

using namespace sd_cli_htp;

static void clear() {
    for (const char* key : {"SD_PRELOAD_BACKENDS", "SD_HTP_OPS", "SD_HTP_COMPUTE", "SD_HTP_ATTENTION",
        "SD_HTP_TEXT", "SD_HTP_TEXT_NATIVE", "SD_HTP_TEXT_NATIVE_AUX", "SD_HTP_TEXT_FUSION",
        "SD_HTP_VAE", "SD_HTP_VAE_NATIVE", "SD_HTP_VAE_NATIVE_CONV", "SD_HTP_VAE_FUSION",
        "SD_HTP_MULTI_SESSION", "SD_HTP_MULTI_SESSION_COUNT", "SD_HTP_DEFER_UNMAP",
        "SD_HTP_PACKED_STAGING", "SD_HTP_PACKED_STAGING_BUDGET_MB", "SD_HTP_PACKED_STAGING_SLOT_MB",
        "SD_HTP_PACKED_PREFETCH", "SD_HTP_SD3_DIT_PREFETCH", "SD_HTP_SD3_DIT_NATIVE",
        "SD_HTP_DIT_PREFETCH", "SD_HTP_DIT_NATIVE", "SD_HTP_DIT_CONCAT",
        "SD_HTP_BLOCK_SPARSE_ATTN", "SD_HTP_SPARSE_OPENCL", "SD_HTP_PACKED_ATTENTION",
        "SD_HTP_HALO_KV_LOAD"}) unsetenv(key);
}

static void value(const char* key, const char* expected) {
    assert(std::getenv(key) && !std::strcmp(std::getenv(key), expected));
}

int main() {
    String2TensorStorage tensors;
    const std::string qkv = "model.diffusion_model.joint_blocks.0.x_block.attn.qkv.weight";
    tensors[qkv].layout = TensorDataLayout::HTP_PERMUTED_W16A32;
    tensors[qkv].ne[0] = 1536;
    tensors["first_stage_model.decoder.conv_in.weight"].layout = TensorDataLayout::HTP_CONV2D_W16A32;
    for (auto* encoder : {"clip_l", "clip_g", "t5xxl"})
        tensors[std::string("text_encoders.")+encoder+".projection.weight"].layout = TensorDataLayout::HTP_PERMUTED_W16A32;
    assert(identify(VERSION_SD3, tensors) == Model::sd35_medium);
    tensors[qkv].ne[0] = 2432;
    assert(identify(VERSION_SD3, tensors) == Model::none); // Large is not Medium.
    tensors[qkv].ne[0] = 1536;
    tensors["text_encoders.t5xxl.projection.weight"].layout = TensorDataLayout::REFERENCE;
    assert(identify(VERSION_SD3, tensors) == Model::none); // Mixed raw/packed is opt-in.
    tensors["model.diffusion_model.img_in.weight"].ne[1] = 3072;
    tensors["text_encoders.llm.projection.weight"].layout = TensorDataLayout::HTP_PERMUTED_W16A32;
    assert(identify(VERSION_FLUX2_KLEIN, tensors) == Model::klein4b);
    tensors["model.diffusion_model.img_in.weight"].ne[1] = 4096;
    assert(identify(VERSION_FLUX2_KLEIN, tensors) == Model::none);

    clear();
    sd_ctx_params_t params{};
    configure(Model::sd35_medium, params, false);
    value("SD_HTP_MULTI_SESSION", "0"); value("SD_HTP_PACKED_STAGING_BUDGET_MB", "512");
    value("SD_HTP_SD3_DIT_NATIVE", "0"); value("SD_HTP_SD3_DIT_PREFETCH", "1");
    assert(params.max_vram == .5f);
    assert(!std::getenv("SD_HTP_PACKED_ATTENTION") && !std::getenv("SD_HTP_HALO_KV_LOAD"));
    clear();
    setenv("SD_HTP_MULTI_SESSION", "1", 1);
    setenv("SD_HTP_BLOCK_SPARSE_ATTN", "1", 1);
    setenv("SD_HTP_SPARSE_OPENCL", "1", 1);
    setenv("SD_HTP_PACKED_ATTENTION", "0", 1);
    setenv("SD_HTP_HALO_KV_LOAD", "baseline", 1);
    configure(Model::sd35_medium, params, false);
    value("SD_HTP_MULTI_SESSION", "1"); value("SD_HTP_PACKED_ATTENTION", "0");
    value("SD_HTP_HALO_KV_LOAD", "baseline");
    value("SD_HTP_PACKED_STAGING", "1"); value("SD_HTP_PACKED_PREFETCH", "1");
    clear();
    params.max_vram = 0.f;
    setenv("SD_HTP_BLOCK_SPARSE_ATTN", "1", 1);
    setenv("SD_HTP_SPARSE_OPENCL", "1", 1);
    configure(Model::sd35_medium, params, false);
    value("SD_HTP_MULTI_SESSION", "0"); value("SD_HTP_DEFER_UNMAP", "0");
    value("SD_HTP_PACKED_STAGING_BUDGET_MB", "512"); value("SD_HTP_PACKED_PREFETCH", "1");
    value("SD_HTP_SD3_DIT_PREFETCH", "1"); value("SD_HTP_SD3_DIT_NATIVE", "0");
    value("SD_HTP_TEXT_NATIVE", "1"); value("SD_HTP_VAE_NATIVE", "1");
    value("SD_HTP_PACKED_ATTENTION", "1"); value("SD_HTP_HALO_KV_LOAD", "l2fetch");
    assert(params.max_vram == .5f);
    clear();
    params.max_vram = 0.f;
    setenv("SD_HTP_DIT_CONCAT", "0", 1);
    setenv("SD_HTP_PACKED_STAGING_BUDGET_MB", "1536", 1);
    configure(Model::klein4b, params, true);
    assert(params.max_vram == 0.f); // Explicit --max-vram 0 is respected.
    value("SD_HTP_DIT_CONCAT", "0"); value("SD_HTP_PACKED_STAGING_BUDGET_MB", "1536");
    value("SD_HTP_DIT_NATIVE", "1"); value("SD_HTP_DIT_PREFETCH", "1");
    clear();
    puts("HTP CLI model selection, dense/sparse defaults and explicit overrides PASSED");
}
