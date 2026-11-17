#pragma once

#include <cstdlib>
#include <cstring>
#include <string>

#include "model.h"
#include "stable-diffusion.h"

// CLI defaults are resolved before any backend/session is initialized. Inspect
// model metadata, never filenames: ordinary/partially packed models stay opt-in.
namespace sd_cli_htp {
enum class Model { none, sd35_medium, klein4b };

inline bool enabled(const char* name) {
    const char* value = std::getenv(name);
    return value && *value && std::strcmp(value, "0") != 0;
}

inline void default_env(const char* name, const char* value) {
    // An explicit zero (or other value) always wins, including in sparse mode.
#ifdef _WIN32
    if (!std::getenv(name)) _putenv_s(name, value);
#else
    setenv(name, value, 0);
#endif
}

inline Model identify(SDVersion version, const String2TensorStorage& tensors) {
    bool dit = false, clip_l = false, clip_g = false, t5 = false, llm = false, vae = false;
    bool medium = false, klein = false;
    for (const auto& item : tensors) {
        const auto& name = item.first;
        const auto& tensor = item.second;
        if (tensor.layout == TensorDataLayout::HTP_PERMUTED_W16A32) {
            if (name.find("model.diffusion_model.") == 0) dit = true;
            if (name.find("text_encoders.clip_l.") == 0) clip_l = true;
            if (name.find("text_encoders.clip_g.") == 0) clip_g = true;
            if (name.find("text_encoders.t5xxl.") == 0) t5 = true;
            if (name.find("text_encoders.llm.") == 0) llm = true;
        }
        if (tensor.layout == TensorDataLayout::HTP_CONV2D_W16A32 &&
            name.find("first_stage_model.") == 0) vae = true;
        if (name == "model.diffusion_model.joint_blocks.0.x_block.attn.qkv.weight")
            medium = tensor.ne[0] == 1536;
        if (name == "model.diffusion_model.img_in.weight")
            klein = tensor.ne[1] == 3072;
    }
    if (!dit || !vae) return Model::none;
    if (version == VERSION_SD3 && medium && clip_l && clip_g && t5) return Model::sd35_medium;
    if (version == VERSION_FLUX2_KLEIN && klein && llm) return Model::klein4b;
    return Model::none;
}

inline void configure(Model model, sd_ctx_params_t& params, bool max_vram_explicit) {
    if (enabled("SD_HTP_BLOCK_SPARSE_ATTN") || enabled("SD_HTP_SPARSE_OPENCL")) {
        default_env("SD_HTP_HALO_KV_LOAD", "l2fetch");
        default_env("SD_HTP_PACKED_ATTENTION", "1");
    }
    const char* common[] = {"SD_HTP_OPS", "SD_HTP_COMPUTE", "SD_HTP_ATTENTION",
        "SD_HTP_TEXT", "SD_HTP_TEXT_NATIVE", "SD_HTP_TEXT_NATIVE_AUX", "SD_HTP_TEXT_FUSION",
        "SD_HTP_VAE", "SD_HTP_VAE_NATIVE", "SD_HTP_VAE_NATIVE_CONV", "SD_HTP_VAE_FUSION"};
    default_env("SD_PRELOAD_BACKENDS", "htp");
    for (auto* name : common) default_env(name, "1");
    default_env("SD_HTP_MULTI_SESSION", "0");
    default_env("SD_HTP_MULTI_SESSION_COUNT", "2");
    // Attention flags never select a different weight/session/text/VAE policy.
    default_env("SD_HTP_DEFER_UNMAP", "0");
    default_env("SD_HTP_PACKED_STAGING", "1");
    default_env("SD_HTP_PACKED_STAGING_BUDGET_MB", model == Model::klein4b ? "1024" : "512");
    default_env("SD_HTP_PACKED_STAGING_SLOT_MB", "512");
    default_env("SD_HTP_PACKED_PREFETCH", "1");
    if (model == Model::sd35_medium) {
        default_env("SD_HTP_SD3_DIT_PREFETCH", "1");
        default_env("SD_HTP_SD3_DIT_NATIVE", "0");
    } else {
        default_env("SD_HTP_DIT_PREFETCH", "1");
        default_env("SD_HTP_DIT_NATIVE", "1");
        default_env("SD_HTP_DIT_CONCAT", "1");
    }
    if (!max_vram_explicit && params.max_vram == 0.f) params.max_vram = .5f;
}

inline const char* apply(sd_ctx_params_t& params, bool max_vram_explicit) {
#ifdef GGML_USE_HTP
    const char* opt = std::getenv("SD_HTP_MODEL_DEFAULTS");
    if ((opt && !std::strcmp(opt, "0")) ||
        !params.staged_params_loading || !params.backend || !params.params_backend ||
        std::strcmp(params.backend, "cpu") || std::strcmp(params.params_backend, "cpu")) return nullptr;

    ModelLoader loader;
    const auto load = [&](const char* path, const char* prefix) {
        return !path || !*path || loader.init_from_file(path, prefix);
    };
    if (!params.diffusion_model_path || !*params.diffusion_model_path ||
        !loader.init_from_file(params.diffusion_model_path, "model.diffusion_model.") ||
        !load(params.clip_l_path, "text_encoders.clip_l.transformer.") ||
        !load(params.clip_g_path, "text_encoders.clip_g.transformer.") ||
        !load(params.t5xxl_path, "text_encoders.t5xxl.transformer.") ||
        !load(params.llm_path, "text_encoders.llm.") || !load(params.vae_path, "vae.")) return nullptr;
    loader.convert_tensors_name();
    const Model model = identify(loader.get_sd_version(), loader.get_tensor_storage_map());
    if (model == Model::none) return nullptr;
    const bool sparse = enabled("SD_HTP_BLOCK_SPARSE_ATTN") || enabled("SD_HTP_SPARSE_OPENCL");
    configure(model, params, max_vram_explicit);
    if (model == Model::sd35_medium) return sparse ? "sd35-medium-sparse-attention" : "sd35-medium-dense";
    return sparse ? "klein4b-sparse-attention" : "klein4b-dense";
#else
    (void)params; (void)max_vram_explicit;
    return nullptr;
#endif
}
} // namespace sd_cli_htp
