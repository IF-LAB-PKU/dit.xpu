#include "model.h"

#include <cstdio>
#include <cstdlib>
#include <initializer_list>

static void check(bool condition) {
    if (!condition) {
        std::fprintf(stderr, "model family check failed\n");
        std::abort();
    }
}

static void add(ModelLoader& loader, const char* name,
                std::initializer_list<int64_t> shape = {1}) {
    TensorStorage tensor;
    tensor.name = name;
    tensor.n_dims = static_cast<int>(shape.size());
    int i = 0;
    for (int64_t extent : shape) {
        tensor.ne[i++] = extent;
    }
    loader.get_tensor_storage_map()[name] = std::move(tensor);
}

static void expect(SDVersion expected, const char* marker) {
    ModelLoader loader;
    add(loader, marker);
    check(loader.get_sd_version() == expected);
    check(sd_version_is_dit(expected));
}

int main() {
    expect(VERSION_SD3, "model.diffusion_model.joint_blocks.0.x_block.attn.qkv.weight");
    expect(VERSION_CHROMA_RADIANCE, "model.diffusion_model.nerf_final_layer_conv.weight");
    expect(VERSION_QWEN_IMAGE, "model.diffusion_model.transformer_blocks.0.img_mod.1.weight");
    expect(VERSION_ANIMA, "llm_adapter.blocks.0.cross_attn.q_proj.weight");
    expect(VERSION_OVIS_IMAGE, "model.diffusion_model.double_blocks.0.img_mlp.gate_proj.weight");
    expect(VERSION_Z_IMAGE, "model.diffusion_model.cap_embedder.0.weight");
    expect(VERSION_ERNIE_IMAGE, "model.diffusion_model.layers.0.adaLN_sa_ln.weight");
    expect(VERSION_LTXAV, "model.diffusion_model.adaln_single.emb.timestep_embedder.linear_1.bias");

    ModelLoader hidream;
    add(hidream, "model.x_embedder.proj1.weight");
    add(hidream, "model.language_model.layers.0.self_attn.q_proj.weight");
    check(hidream.get_sd_version() == VERSION_HIDREAM_O1);

    for (auto test : {std::pair<int64_t, SDVersion>{64, VERSION_FLUX},
                      {384, VERSION_FLUX_FILL}, {128, VERSION_FLUX_CONTROLS},
                      {196, VERSION_FLEX_2}}) {
        ModelLoader flux;
        add(flux, "model.diffusion_model.double_blocks.0.img_attn.qkv.weight");
        add(flux, "model.diffusion_model.img_in.weight", {test.first, 3072});
        check(flux.get_sd_version() == test.second);
    }
    ModelLoader longcat;
    add(longcat, "model.diffusion_model.double_blocks.0.img_attn.qkv.weight");
    add(longcat, "model.diffusion_model.txt_in.weight", {3584, 3072});
    check(longcat.get_sd_version() == VERSION_LONGCAT);

    ModelLoader flux2;
    add(flux2, "model.diffusion_model.double_stream_modulation_img.lin.weight");
    check(flux2.get_sd_version() == VERSION_FLUX2_KLEIN);
    add(flux2, "model.diffusion_model.single_blocks.47.linear1.weight");
    check(flux2.get_sd_version() == VERSION_FLUX2);

    for (auto test : {std::pair<int64_t, SDVersion>{16, VERSION_WAN2},
                      {184320, VERSION_WAN2_2_I2V}, {147456, VERSION_WAN2_2_TI2V}}) {
        ModelLoader wan;
        add(wan, "model.diffusion_model.blocks.0.cross_attn.norm_k.weight");
        add(wan, "model.diffusion_model.patch_embedding.weight", {1, 1, 1, test.first});
        check(wan.get_sd_version() == test.second);
        add(wan, "model.diffusion_model.img_emb.proj.0.weight");
        check(wan.get_sd_version() == VERSION_WAN2);
    }

    // Legacy diffusion weights and text encoders must no longer select a model.
    for (int64_t width : {768, 1024, 1280}) {
        ModelLoader legacy;
        add(legacy, "cond_stage_model.transformer.text_model.embeddings.token_embedding.weight", {width, 100});
        add(legacy, "model.diffusion_model.input_blocks.0.0.weight", {3, 3, 4, 320});
        add(legacy, "model.diffusion_model.middle_block.1.transformer_blocks.0.attn1.to_k.weight");
        add(legacy, "model.diffusion_model.input_blocks.8.0.time_mixer.mix_factor");
        add(legacy, "conditioner.embedders.1.model.token_embedding.weight");
        check(legacy.get_sd_version() == VERSION_COUNT);
    }
    for (int version = 0; version < VERSION_COUNT; ++version) {
        check(sd_version_is_dit(static_cast<SDVersion>(version)));
    }
    check(!sd_version_is_dit(VERSION_COUNT));
    std::puts("DiT families retained; UNet families rejected");
}
