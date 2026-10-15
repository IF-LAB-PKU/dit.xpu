#include <unordered_map>
#include <unordered_set>

#include "name_conversion.h"
#include "util.h"

void replace_with_name_map(std::string& name, const std::vector<std::pair<std::string, std::string>>& name_map) {
    for (auto kv : name_map) {
        size_t pos = name.find(kv.first);
        if (pos != std::string::npos) {
            name.replace(pos, kv.first.size(), kv.second);
        }
    }
}

void replace_with_prefix_map(std::string& name, const std::vector<std::pair<std::string, std::string>>& prefix_map) {
    for (const auto& [old_prefix, new_prefix] : prefix_map) {
        if (starts_with(name, old_prefix)) {
            name = new_prefix + name.substr(old_prefix.size());
            break;
        }
    }
}

void replace_with_prefix_map(std::string& name, const std::unordered_map<std::string, std::string>& prefix_map) {
    for (const auto& [old_prefix, new_prefix] : prefix_map) {
        if (starts_with(name, old_prefix)) {
            name = new_prefix + name.substr(old_prefix.size());
            break;
        }
    }
}

std::string convert_open_clip_to_hf_clip_name(std::string name) {
    static std::unordered_map<std::string, std::string> open_clip_to_hf_clip_model = {
        {"model.ln_final.bias", "transformer.text_model.final_layer_norm.bias"},
        {"model.ln_final.weight", "transformer.text_model.final_layer_norm.weight"},
        {"model.positional_embedding", "transformer.text_model.embeddings.position_embedding.weight"},
        {"model.token_embedding.weight", "transformer.text_model.embeddings.token_embedding.weight"},
        {"model.text_projection", "transformer.text_model.text_projection"},
        {"model.visual.class_embedding", "transformer.vision_model.embeddings.class_embedding"},
        {"model.visual.conv1.weight", "transformer.vision_model.embeddings.patch_embedding.weight"},
        {"model.visual.ln_post.bias", "transformer.vision_model.post_layernorm.bias"},
        {"model.visual.ln_post.weight", "transformer.vision_model.post_layernorm.weight"},
        {"model.visual.ln_pre.bias", "transformer.vision_model.pre_layernorm.bias"},
        {"model.visual.ln_pre.weight", "transformer.vision_model.pre_layernorm.weight"},
        {"model.visual.positional_embedding", "transformer.vision_model.embeddings.position_embedding.weight"},
        {"model.visual.proj", "transformer.visual_projection.weight"},
    };

    static std::unordered_map<std::string, std::string> open_clip_to_hf_clip_resblock = {
        {"attn.in_proj_bias", "self_attn.in_proj.bias"},
        {"attn.in_proj_weight", "self_attn.in_proj.weight"},
        {"attn.out_proj.bias", "self_attn.out_proj.bias"},
        {"attn.out_proj.weight", "self_attn.out_proj.weight"},
        {"ln_1.bias", "layer_norm1.bias"},
        {"ln_1.weight", "layer_norm1.weight"},
        {"ln_2.bias", "layer_norm2.bias"},
        {"ln_2.weight", "layer_norm2.weight"},
        {"mlp.c_fc.bias", "mlp.fc1.bias"},
        {"mlp.c_fc.weight", "mlp.fc1.weight"},
        {"mlp.c_proj.bias", "mlp.fc2.bias"},
        {"mlp.c_proj.weight", "mlp.fc2.weight"},
    };

    static std::unordered_map<std::string, std::string> cond_model_name_map = {
        {"transformer.vision_model.pre_layrnorm.weight", "transformer.vision_model.pre_layernorm.weight"},
        {"transformer.vision_model.pre_layrnorm.bias", "transformer.vision_model.pre_layernorm.bias"},
    };

    if (open_clip_to_hf_clip_model.find(name) != open_clip_to_hf_clip_model.end()) {
        name = open_clip_to_hf_clip_model[name];
    }

    if (cond_model_name_map.find(name) != cond_model_name_map.end()) {
        name = cond_model_name_map[name];
    }

    std::string open_clip_resblock_prefix = "model.transformer.resblocks.";
    std::string hf_clip_resblock_prefix   = "transformer.text_model.encoder.layers.";

    auto replace_suffix = [&]() {
        if (name.find(open_clip_resblock_prefix) == 0) {
            std::string remain = name.substr(open_clip_resblock_prefix.length());
            std::string idx    = remain.substr(0, remain.find("."));
            std::string suffix = remain.substr(idx.length() + 1);

            if (open_clip_to_hf_clip_resblock.find(suffix) != open_clip_to_hf_clip_resblock.end()) {
                std::string new_suffix = open_clip_to_hf_clip_resblock[suffix];
                name                   = hf_clip_resblock_prefix + idx + "." + new_suffix;
            }
        }
    };

    replace_suffix();

    open_clip_resblock_prefix = "model.visual.transformer.resblocks.";
    hf_clip_resblock_prefix   = "transformer.vision_model.encoder.layers.";

    replace_suffix();

    return name;
}

std::string convert_cond_stage_model_name(std::string name, std::string prefix) {
    static const std::vector<std::pair<std::string, std::string>> clip_name_map{
        {"transformer.text_projection.weight", "transformer.text_model.text_projection"},
        {"model.text_projection.weight", "transformer.text_model.text_projection"},
        {"vision_model.visual_projection.weight", "visual_projection.weight"},
    };

    // llama.cpp to original
    static const std::vector<std::pair<std::string, std::string>> t5_name_map{
        {"enc.", "encoder."},
        {"blk.", "block."},
        {"output_norm.", "final_layer_norm."},
        {"attn_q.", "layer.0.SelfAttention.q."},
        {"attn_k.", "layer.0.SelfAttention.k."},
        {"attn_v.", "layer.0.SelfAttention.v."},
        {"attn_o.", "layer.0.SelfAttention.o."},
        {"attn_norm.", "layer.0.layer_norm."},
        {"ffn_norm.", "layer.1.layer_norm."},
        {"ffn_up.", "layer.1.DenseReluDense.wi_1."},
        {"ffn_down.", "layer.1.DenseReluDense.wo."},
        {"ffn_gate.", "layer.1.DenseReluDense.wi_0."},
        {"attn_rel_b.", "layer.0.SelfAttention.relative_attention_bias."},
        {"token_embd.", "shared."},
    };

    static const std::vector<std::pair<std::string, std::string>> llm_name_map{
        {"token_embd.", "model.embed_tokens."},
        {"blk.", "model.layers."},
        {"attn_q.", "self_attn.q_proj."},
        {"attn_k.", "self_attn.k_proj."},
        {"attn_v.", "self_attn.v_proj."},
        {"attn_q_norm.", "self_attn.q_norm."},
        {"attn_k_norm.", "self_attn.k_norm."},
        {"attn_output.", "self_attn.o_proj."},
        {"attn_norm.", "input_layernorm."},
        {"ffn_down.", "mlp.down_proj."},
        {"ffn_gate.", "mlp.gate_proj."},
        {"ffn_up.", "mlp.up_proj."},
        {"ffn_norm.", "post_attention_layernorm."},
        {"output_norm.", "model.norm."},
    };

    static const std::vector<std::pair<std::string, std::string>> llm_vision_name_map{
        {"mm.", "merger.mlp."},
        {"v.post_ln.", "merger.ln_q."},
        {"v.patch_embd.weight", "patch_embed.proj.0.weight"},
        {"patch_embed.proj.0.weight.1", "patch_embed.proj.1.weight"},
        {"v.patch_embd.weight.1", "patch_embed.proj.1.weight"},
        {"v.blk.", "blocks."},
        {"attn_q.", "attn.q_proj."},
        {"attn_k.", "attn.k_proj."},
        {"attn_v.", "attn.v_proj."},
        {"attn_out.", "attn.proj."},
        {"ffn_down.", "mlp.down_proj."},
        {"ffn_gate.", "mlp.gate_proj."},
        {"ffn_up.", "mlp.up_proj."},
        {"ln1.", "norm1."},
        {"ln2.", "norm2."},
    };
    if (contains(name, "t5xxl")) {
        replace_with_name_map(name, t5_name_map);
    } else if (contains(name, "llm")) {
        if (contains(name, "llm.visual")) {
            replace_with_name_map(name, llm_vision_name_map);
        } else {
            replace_with_name_map(name, llm_name_map);
        }
    } else {
        name = convert_open_clip_to_hf_clip_name(name);
        replace_with_name_map(name, clip_name_map);
    }
    return name;
}

// ref: https://github.com/huggingface/diffusers/blob/main/scripts/convert_diffusers_to_original_stable_diffusion.py

std::string convert_diffusers_dit_to_original_sd3(std::string name) {
    int num_layers = 38;
    static std::unordered_map<std::string, std::string> sd3_name_map;

    if (sd3_name_map.empty()) {
        // --- time_text_embed ---
        sd3_name_map["time_text_embed.timestep_embedder.linear_1.weight"] = "t_embedder.mlp.0.weight";
        sd3_name_map["time_text_embed.timestep_embedder.linear_1.bias"]   = "t_embedder.mlp.0.bias";
        sd3_name_map["time_text_embed.timestep_embedder.linear_2.weight"] = "t_embedder.mlp.2.weight";
        sd3_name_map["time_text_embed.timestep_embedder.linear_2.bias"]   = "t_embedder.mlp.2.bias";

        sd3_name_map["time_text_embed.text_embedder.linear_1.weight"] = "y_embedder.mlp.0.weight";
        sd3_name_map["time_text_embed.text_embedder.linear_1.bias"]   = "y_embedder.mlp.0.bias";
        sd3_name_map["time_text_embed.text_embedder.linear_2.weight"] = "y_embedder.mlp.2.weight";
        sd3_name_map["time_text_embed.text_embedder.linear_2.bias"]   = "y_embedder.mlp.2.bias";

        sd3_name_map["pos_embed.pos_embed"]   = "pos_embed";
        sd3_name_map["pos_embed.proj.weight"] = "x_embedder.proj.weight";
        sd3_name_map["pos_embed.proj.bias"]   = "x_embedder.proj.bias";

        // --- transformer blocks ---
        for (int i = 0; i < num_layers; ++i) {
            std::string block_prefix = "transformer_blocks." + std::to_string(i) + ".";
            std::string dst_prefix   = "joint_blocks." + std::to_string(i) + ".";

            sd3_name_map[block_prefix + "norm1.linear.weight"]         = dst_prefix + "x_block.adaLN_modulation.1.weight";
            sd3_name_map[block_prefix + "norm1.linear.bias"]           = dst_prefix + "x_block.adaLN_modulation.1.bias";
            sd3_name_map[block_prefix + "norm1_context.linear.weight"] = dst_prefix + "context_block.adaLN_modulation.1.weight";
            sd3_name_map[block_prefix + "norm1_context.linear.bias"]   = dst_prefix + "context_block.adaLN_modulation.1.bias";

            // attn
            sd3_name_map[block_prefix + "attn.to_q.weight"] = dst_prefix + "x_block.attn.qkv.weight";
            sd3_name_map[block_prefix + "attn.to_q.bias"]   = dst_prefix + "x_block.attn.qkv.bias";
            sd3_name_map[block_prefix + "attn.to_k.weight"] = dst_prefix + "x_block.attn.qkv.weight.1";
            sd3_name_map[block_prefix + "attn.to_k.bias"]   = dst_prefix + "x_block.attn.qkv.bias.1";
            sd3_name_map[block_prefix + "attn.to_v.weight"] = dst_prefix + "x_block.attn.qkv.weight.2";
            sd3_name_map[block_prefix + "attn.to_v.bias"]   = dst_prefix + "x_block.attn.qkv.bias.2";

            sd3_name_map[block_prefix + "attn.add_q_proj.weight"] = dst_prefix + "context_block.attn.qkv.weight";
            sd3_name_map[block_prefix + "attn.add_q_proj.bias"]   = dst_prefix + "context_block.attn.qkv.bias";
            sd3_name_map[block_prefix + "attn.add_k_proj.weight"] = dst_prefix + "context_block.attn.qkv.weight.1";
            sd3_name_map[block_prefix + "attn.add_k_proj.bias"]   = dst_prefix + "context_block.attn.qkv.bias.1";
            sd3_name_map[block_prefix + "attn.add_v_proj.weight"] = dst_prefix + "context_block.attn.qkv.weight.2";
            sd3_name_map[block_prefix + "attn.add_v_proj.bias"]   = dst_prefix + "context_block.attn.qkv.bias.2";

            // attn2
            sd3_name_map[block_prefix + "attn2.to_q.weight"] = dst_prefix + "x_block.attn2.qkv.weight";
            sd3_name_map[block_prefix + "attn2.to_q.bias"]   = dst_prefix + "x_block.attn2.qkv.bias";
            sd3_name_map[block_prefix + "attn2.to_k.weight"] = dst_prefix + "x_block.attn2.qkv.weight.1";
            sd3_name_map[block_prefix + "attn2.to_k.bias"]   = dst_prefix + "x_block.attn2.qkv.bias.1";
            sd3_name_map[block_prefix + "attn2.to_v.weight"] = dst_prefix + "x_block.attn2.qkv.weight.2";
            sd3_name_map[block_prefix + "attn2.to_v.bias"]   = dst_prefix + "x_block.attn2.qkv.bias.2";

            sd3_name_map[block_prefix + "attn2.add_q_proj.weight"] = dst_prefix + "context_block.attn2.qkv.weight";
            sd3_name_map[block_prefix + "attn2.add_q_proj.bias"]   = dst_prefix + "context_block.attn2.qkv.bias";
            sd3_name_map[block_prefix + "attn2.add_k_proj.weight"] = dst_prefix + "context_block.attn2.qkv.weight.1";
            sd3_name_map[block_prefix + "attn2.add_k_proj.bias"]   = dst_prefix + "context_block.attn2.qkv.bias.1";
            sd3_name_map[block_prefix + "attn2.add_v_proj.weight"] = dst_prefix + "context_block.attn2.qkv.weight.2";
            sd3_name_map[block_prefix + "attn2.add_v_proj.bias"]   = dst_prefix + "context_block.attn2.qkv.bias.2";

            // norm
            sd3_name_map[block_prefix + "attn.norm_q.weight"]       = dst_prefix + "x_block.attn.ln_q.weight";
            sd3_name_map[block_prefix + "attn.norm_k.weight"]       = dst_prefix + "x_block.attn.ln_k.weight";
            sd3_name_map[block_prefix + "attn.norm_added_q.weight"] = dst_prefix + "context_block.attn.ln_q.weight";
            sd3_name_map[block_prefix + "attn.norm_added_k.weight"] = dst_prefix + "context_block.attn.ln_k.weight";

            // norm2
            sd3_name_map[block_prefix + "attn2.norm_q.weight"] = dst_prefix + "x_block.attn2.ln_q.weight";
            sd3_name_map[block_prefix + "attn2.norm_k.weight"] = dst_prefix + "x_block.attn2.ln_k.weight";

            // ff
            sd3_name_map[block_prefix + "ff.net.0.proj.weight"] = dst_prefix + "x_block.mlp.fc1.weight";
            sd3_name_map[block_prefix + "ff.net.0.proj.bias"]   = dst_prefix + "x_block.mlp.fc1.bias";
            sd3_name_map[block_prefix + "ff.net.2.weight"]      = dst_prefix + "x_block.mlp.fc2.weight";
            sd3_name_map[block_prefix + "ff.net.2.bias"]        = dst_prefix + "x_block.mlp.fc2.bias";

            sd3_name_map[block_prefix + "ff_context.net.0.proj.weight"] = dst_prefix + "context_block.mlp.fc1.weight";
            sd3_name_map[block_prefix + "ff_context.net.0.proj.bias"]   = dst_prefix + "context_block.mlp.fc1.bias";
            sd3_name_map[block_prefix + "ff_context.net.2.weight"]      = dst_prefix + "context_block.mlp.fc2.weight";
            sd3_name_map[block_prefix + "ff_context.net.2.bias"]        = dst_prefix + "context_block.mlp.fc2.bias";

            // output projections
            sd3_name_map[block_prefix + "attn.to_out.0.weight"]   = dst_prefix + "x_block.attn.proj.weight";
            sd3_name_map[block_prefix + "attn.to_out.0.bias"]     = dst_prefix + "x_block.attn.proj.bias";
            sd3_name_map[block_prefix + "attn.to_add_out.weight"] = dst_prefix + "context_block.attn.proj.weight";
            sd3_name_map[block_prefix + "attn.to_add_out.bias"]   = dst_prefix + "context_block.attn.proj.bias";

            // output projections 2
            sd3_name_map[block_prefix + "attn2.to_out.0.weight"]   = dst_prefix + "x_block.attn2.proj.weight";
            sd3_name_map[block_prefix + "attn2.to_out.0.bias"]     = dst_prefix + "x_block.attn2.proj.bias";
            sd3_name_map[block_prefix + "attn2.to_add_out.weight"] = dst_prefix + "context_block.attn2.proj.weight";
            sd3_name_map[block_prefix + "attn2.to_add_out.bias"]   = dst_prefix + "context_block.attn2.proj.bias";
        }

        // --- final layers ---
        sd3_name_map["proj_out.weight"]        = "final_layer.linear.weight";
        sd3_name_map["proj_out.bias"]          = "final_layer.linear.bias";
        sd3_name_map["norm_out.linear.weight"] = "final_layer.adaLN_modulation.1.weight";
        sd3_name_map["norm_out.linear.bias"]   = "final_layer.adaLN_modulation.1.bias";
    }

    replace_with_prefix_map(name, sd3_name_map);

    return name;
}

std::string convert_diffusers_dit_to_original_flux(std::string name) {
    int num_layers        = 19;
    int num_single_layers = 38;
    static std::unordered_map<std::string, std::string> flux_name_map;

    if (flux_name_map.empty()) {
        // --- time_embed (longcat) ---
        flux_name_map["time_embed.timestep_embedder.linear_1.weight"] = "time_in.in_layer.weight";
        flux_name_map["time_embed.timestep_embedder.linear_1.bias"]   = "time_in.in_layer.bias";
        flux_name_map["time_embed.timestep_embedder.linear_2.weight"] = "time_in.out_layer.weight";
        flux_name_map["time_embed.timestep_embedder.linear_2.bias"]   = "time_in.out_layer.bias";

        // --- time_text_embed ---
        flux_name_map["time_text_embed.timestep_embedder.linear_1.weight"] = "time_in.in_layer.weight";
        flux_name_map["time_text_embed.timestep_embedder.linear_1.bias"]   = "time_in.in_layer.bias";
        flux_name_map["time_text_embed.timestep_embedder.linear_2.weight"] = "time_in.out_layer.weight";
        flux_name_map["time_text_embed.timestep_embedder.linear_2.bias"]   = "time_in.out_layer.bias";

        flux_name_map["time_text_embed.text_embedder.linear_1.weight"] = "vector_in.in_layer.weight";
        flux_name_map["time_text_embed.text_embedder.linear_1.bias"]   = "vector_in.in_layer.bias";
        flux_name_map["time_text_embed.text_embedder.linear_2.weight"] = "vector_in.out_layer.weight";
        flux_name_map["time_text_embed.text_embedder.linear_2.bias"]   = "vector_in.out_layer.bias";

        // guidance
        flux_name_map["time_text_embed.guidance_embedder.linear_1.weight"] = "guidance_in.in_layer.weight";
        flux_name_map["time_text_embed.guidance_embedder.linear_1.bias"]   = "guidance_in.in_layer.bias";
        flux_name_map["time_text_embed.guidance_embedder.linear_2.weight"] = "guidance_in.out_layer.weight";
        flux_name_map["time_text_embed.guidance_embedder.linear_2.bias"]   = "guidance_in.out_layer.bias";

        // --- context_embedder / x_embedder ---
        flux_name_map["context_embedder.weight"] = "txt_in.weight";
        flux_name_map["context_embedder.bias"]   = "txt_in.bias";
        flux_name_map["x_embedder.weight"]       = "img_in.weight";
        flux_name_map["x_embedder.bias"]         = "img_in.bias";

        // --- double transformer blocks ---
        for (int i = 0; i < num_layers; ++i) {
            std::string block_prefix = "transformer_blocks." + std::to_string(i) + ".";
            std::string dst_prefix   = "double_blocks." + std::to_string(i) + ".";

            flux_name_map[block_prefix + "norm1.linear.weight"]         = dst_prefix + "img_mod.lin.weight";
            flux_name_map[block_prefix + "norm1.linear.bias"]           = dst_prefix + "img_mod.lin.bias";
            flux_name_map[block_prefix + "norm1_context.linear.weight"] = dst_prefix + "txt_mod.lin.weight";
            flux_name_map[block_prefix + "norm1_context.linear.bias"]   = dst_prefix + "txt_mod.lin.bias";

            // attn
            flux_name_map[block_prefix + "attn.to_q.weight"] = dst_prefix + "img_attn.qkv.weight";
            flux_name_map[block_prefix + "attn.to_q.bias"]   = dst_prefix + "img_attn.qkv.bias";
            flux_name_map[block_prefix + "attn.to_k.weight"] = dst_prefix + "img_attn.qkv.weight.1";
            flux_name_map[block_prefix + "attn.to_k.bias"]   = dst_prefix + "img_attn.qkv.bias.1";
            flux_name_map[block_prefix + "attn.to_v.weight"] = dst_prefix + "img_attn.qkv.weight.2";
            flux_name_map[block_prefix + "attn.to_v.bias"]   = dst_prefix + "img_attn.qkv.bias.2";

            flux_name_map[block_prefix + "attn.add_q_proj.weight"] = dst_prefix + "txt_attn.qkv.weight";
            flux_name_map[block_prefix + "attn.add_q_proj.bias"]   = dst_prefix + "txt_attn.qkv.bias";
            flux_name_map[block_prefix + "attn.add_k_proj.weight"] = dst_prefix + "txt_attn.qkv.weight.1";
            flux_name_map[block_prefix + "attn.add_k_proj.bias"]   = dst_prefix + "txt_attn.qkv.bias.1";
            flux_name_map[block_prefix + "attn.add_v_proj.weight"] = dst_prefix + "txt_attn.qkv.weight.2";
            flux_name_map[block_prefix + "attn.add_v_proj.bias"]   = dst_prefix + "txt_attn.qkv.bias.2";

            // norm
            flux_name_map[block_prefix + "attn.norm_q.weight"]       = dst_prefix + "img_attn.norm.query_norm.scale";
            flux_name_map[block_prefix + "attn.norm_k.weight"]       = dst_prefix + "img_attn.norm.key_norm.scale";
            flux_name_map[block_prefix + "attn.norm_added_q.weight"] = dst_prefix + "txt_attn.norm.query_norm.scale";
            flux_name_map[block_prefix + "attn.norm_added_k.weight"] = dst_prefix + "txt_attn.norm.key_norm.scale";
            // Comfy-Org/LongCat-Image stores already-converted RMSNorm tensors as *.weight.
            flux_name_map[dst_prefix + "img_attn.norm.query_norm.weight"] = dst_prefix + "img_attn.norm.query_norm.scale";
            flux_name_map[dst_prefix + "img_attn.norm.key_norm.weight"]   = dst_prefix + "img_attn.norm.key_norm.scale";
            flux_name_map[dst_prefix + "txt_attn.norm.query_norm.weight"] = dst_prefix + "txt_attn.norm.query_norm.scale";
            flux_name_map[dst_prefix + "txt_attn.norm.key_norm.weight"]   = dst_prefix + "txt_attn.norm.key_norm.scale";

            // ff
            flux_name_map[block_prefix + "ff.net.0.proj.weight"] = dst_prefix + "img_mlp.0.weight";
            flux_name_map[block_prefix + "ff.net.0.proj.bias"]   = dst_prefix + "img_mlp.0.bias";
            flux_name_map[block_prefix + "ff.net.2.weight"]      = dst_prefix + "img_mlp.2.weight";
            flux_name_map[block_prefix + "ff.net.2.bias"]        = dst_prefix + "img_mlp.2.bias";

            flux_name_map[block_prefix + "ff_context.net.0.proj.weight"] = dst_prefix + "txt_mlp.0.weight";
            flux_name_map[block_prefix + "ff_context.net.0.proj.bias"]   = dst_prefix + "txt_mlp.0.bias";
            flux_name_map[block_prefix + "ff_context.net.2.weight"]      = dst_prefix + "txt_mlp.2.weight";
            flux_name_map[block_prefix + "ff_context.net.2.bias"]        = dst_prefix + "txt_mlp.2.bias";

            // output projections
            flux_name_map[block_prefix + "attn.to_out.0.weight"]   = dst_prefix + "img_attn.proj.weight";
            flux_name_map[block_prefix + "attn.to_out.0.bias"]     = dst_prefix + "img_attn.proj.bias";
            flux_name_map[block_prefix + "attn.to_add_out.weight"] = dst_prefix + "txt_attn.proj.weight";
            flux_name_map[block_prefix + "attn.to_add_out.bias"]   = dst_prefix + "txt_attn.proj.bias";
        }

        // --- single transformer blocks ---
        for (int i = 0; i < num_single_layers; ++i) {
            std::string block_prefix = "single_transformer_blocks." + std::to_string(i) + ".";
            std::string dst_prefix   = "single_blocks." + std::to_string(i) + ".";

            flux_name_map[block_prefix + "norm.linear.weight"] = dst_prefix + "modulation.lin.weight";
            flux_name_map[block_prefix + "norm.linear.bias"]   = dst_prefix + "modulation.lin.bias";

            flux_name_map[block_prefix + "attn.to_q.weight"] = dst_prefix + "linear1.weight";
            flux_name_map[block_prefix + "attn.to_q.bias"]   = dst_prefix + "linear1.bias";
            flux_name_map[block_prefix + "attn.to_k.weight"] = dst_prefix + "linear1.weight.1";
            flux_name_map[block_prefix + "attn.to_k.bias"]   = dst_prefix + "linear1.bias.1";
            flux_name_map[block_prefix + "attn.to_v.weight"] = dst_prefix + "linear1.weight.2";
            flux_name_map[block_prefix + "attn.to_v.bias"]   = dst_prefix + "linear1.bias.2";
            flux_name_map[block_prefix + "proj_mlp.weight"]  = dst_prefix + "linear1.weight.3";
            flux_name_map[block_prefix + "proj_mlp.bias"]    = dst_prefix + "linear1.bias.3";

            flux_name_map[block_prefix + "attn.norm_q.weight"] = dst_prefix + "norm.query_norm.scale";
            flux_name_map[block_prefix + "attn.norm_k.weight"] = dst_prefix + "norm.key_norm.scale";
            // Comfy-Org/LongCat-Image stores already-converted RMSNorm tensors as *.weight.
            flux_name_map[dst_prefix + "norm.query_norm.weight"] = dst_prefix + "norm.query_norm.scale";
            flux_name_map[dst_prefix + "norm.key_norm.weight"]   = dst_prefix + "norm.key_norm.scale";
            flux_name_map[block_prefix + "proj_out.weight"]      = dst_prefix + "linear2.weight";
            flux_name_map[block_prefix + "proj_out.bias"]        = dst_prefix + "linear2.bias";
        }

        // --- final layers ---
        flux_name_map["proj_out.weight"]        = "final_layer.linear.weight";
        flux_name_map["proj_out.bias"]          = "final_layer.linear.bias";
        flux_name_map["norm_out.linear.weight"] = "final_layer.adaLN_modulation.1.weight";
        flux_name_map["norm_out.linear.bias"]   = "final_layer.adaLN_modulation.1.bias";
    }

    replace_with_prefix_map(name, flux_name_map);

    return name;
}

std::string convert_diffusers_dit_to_original_lumina2(std::string name) {
    int num_layers         = 30;
    int num_refiner_layers = 2;
    static std::unordered_map<std::string, std::string> z_image_name_map;

    if (z_image_name_map.empty()) {
        z_image_name_map["all_x_embedder.2-1."]  = "x_embedder.";
        z_image_name_map["all_final_layer.2-1."] = "final_layer.";

        // --- transformer blocks ---
        auto add_attention_map = [&](const std::string& prefix, int num) {
            for (int i = 0; i < num; ++i) {
                std::string block_prefix = prefix + std::to_string(i) + ".";
                std::string dst_prefix   = prefix + std::to_string(i) + ".";

                z_image_name_map[block_prefix + "attention.norm_q."]   = dst_prefix + "attention.q_norm.";
                z_image_name_map[block_prefix + "attention.norm_k."]   = dst_prefix + "attention.k_norm.";
                z_image_name_map[block_prefix + "attention.to_out.0."] = dst_prefix + "attention.out.";

                z_image_name_map[block_prefix + "attention.to_q.weight"] = dst_prefix + "attention.qkv.weight";
                z_image_name_map[block_prefix + "attention.to_q.bias"]   = dst_prefix + "attention.qkv.bias";
                z_image_name_map[block_prefix + "attention.to_k.weight"] = dst_prefix + "attention.qkv.weight.1";
                z_image_name_map[block_prefix + "attention.to_k.bias"]   = dst_prefix + "attention.qkv.bias.1";
                z_image_name_map[block_prefix + "attention.to_v.weight"] = dst_prefix + "attention.qkv.weight.2";
                z_image_name_map[block_prefix + "attention.to_v.bias"]   = dst_prefix + "attention.qkv.bias.2";
            }
        };

        add_attention_map("noise_refiner.", num_refiner_layers);
        add_attention_map("context_refiner.", num_refiner_layers);
        add_attention_map("layers.", num_layers);
    }

    replace_with_prefix_map(name, z_image_name_map);

    return name;
}

std::string convert_other_dit_to_original_anima(std::string name) {
    static const std::string anima_net_prefix = "net.";
    if (!starts_with(name, anima_net_prefix)) {
        name = anima_net_prefix + name;
    }
    return name;
}

std::string convert_diffusion_model_name(std::string name, std::string prefix, SDVersion version) {
    if (sd_version_is_sd3(version)) {
        name = convert_diffusers_dit_to_original_sd3(name);
    } else if (sd_version_is_flux(version) || sd_version_is_flux2(version) || sd_version_is_longcat(version)) {
        name = convert_diffusers_dit_to_original_flux(name);
    } else if (sd_version_is_z_image(version)) {
        name = convert_diffusers_dit_to_original_lumina2(name);
    } else if (sd_version_is_anima(version)) {
        name = convert_other_dit_to_original_anima(name);
    }
    return name;
}

std::string convert_diffusers_vae_to_original_kl(std::string name) {
    static const std::vector<std::pair<std::string, std::string>> vae_conversion_map_base = {
        {"nin_shortcut", "conv_shortcut"},
        {"norm_out", "conv_norm_out"},
        {"mid.attn_1.", "mid_block.attentions.0."},
    };

    static std::vector<std::pair<std::string, std::string>> vae_conversion_map_layer;
    if (vae_conversion_map_layer.empty()) {
        for (int i = 0; i < 4; ++i) {
            // --- encoder down blocks ---
            for (int j = 0; j < 2; ++j) {
                std::string hf_down_prefix = "encoder.down_blocks." + std::to_string(i) + ".resnets." + std::to_string(j) + ".";
                std::string sd_down_prefix = "encoder.down." + std::to_string(i) + ".block." + std::to_string(j) + ".";
                vae_conversion_map_layer.emplace_back(sd_down_prefix, hf_down_prefix);
            }

            if (i < 3) {
                std::string hf_downsample_prefix = "down_blocks." + std::to_string(i) + ".downsamplers.0.";
                std::string sd_downsample_prefix = "down." + std::to_string(i) + ".downsample.";
                vae_conversion_map_layer.emplace_back(sd_downsample_prefix, hf_downsample_prefix);

                std::string hf_upsample_prefix = "up_blocks." + std::to_string(i) + ".upsamplers.0.";
                std::string sd_upsample_prefix = "up." + std::to_string(3 - i) + ".upsample.";
                vae_conversion_map_layer.emplace_back(sd_upsample_prefix, hf_upsample_prefix);
            }

            // --- decoder up blocks (reverse) ---
            for (int j = 0; j < 3; ++j) {
                std::string hf_up_prefix = "decoder.up_blocks." + std::to_string(i) + ".resnets." + std::to_string(j) + ".";
                std::string sd_up_prefix = "decoder.up." + std::to_string(3 - i) + ".block." + std::to_string(j) + ".";
                vae_conversion_map_layer.emplace_back(sd_up_prefix, hf_up_prefix);
            }
        }

        // --- mid block (encoder + decoder) ---
        for (int i = 0; i < 2; ++i) {
            std::string hf_mid_res_prefix = "mid_block.resnets." + std::to_string(i) + ".";
            std::string sd_mid_res_prefix = "mid.block_" + std::to_string(i + 1) + ".";
            vae_conversion_map_layer.emplace_back(sd_mid_res_prefix, hf_mid_res_prefix);
        }
    }

    static const std::vector<std::pair<std::string, std::string>> vae_conversion_map_attn = {
        {"norm.", "group_norm."},
        {"q.", "query."},
        {"k.", "key."},
        {"v.", "value."},
        {"proj_out.", "proj_attn."},
    };

    static const std::vector<std::pair<std::string, std::string>> vae_extra_conversion_map = {
        {"to_q", "q"},
        {"to_k", "k"},
        {"to_v", "v"},
        {"to_out.0", "proj_out"},
    };

    std::string result = name;

    for (const auto& p : vae_conversion_map_base) {
        size_t pos = result.find(p.second);
        if (pos != std::string::npos) {
            result.replace(pos, p.second.size(), p.first);
        }
    }

    for (const auto& p : vae_conversion_map_layer) {
        size_t pos = result.find(p.second);
        if (pos != std::string::npos) {
            result.replace(pos, p.second.size(), p.first);
        }
    }

    if (name.find("attentions") != std::string::npos) {
        for (const auto& p : vae_conversion_map_attn) {
            size_t pos = result.find(p.second);
            if (pos != std::string::npos) {
                result.replace(pos, p.second.size(), p.first);
            }
        }
    }

    if (result.find("mid.attn_1.") != std::string::npos) {
        for (const auto& p : vae_extra_conversion_map) {
            size_t pos = result.find(p.first);
            if (pos != std::string::npos) {
                result.replace(pos, p.first.size(), p.second);
            }
        }
    }

    return result;
}

std::string convert_first_stage_model_name(std::string name, std::string prefix) {
    static std::unordered_map<std::string, std::string> vae_name_map = {
        {"decoder.post_quant_conv.", "post_quant_conv."},
        {"encoder.quant_conv.", "quant_conv."},
    };
    replace_with_prefix_map(name, vae_name_map);
    name = convert_diffusers_vae_to_original_kl(name);
    return name;
}

std::string convert_sep_to_dot(std::string name) {
    const std::vector<std::string> protected_tokens = {
        "self_attn",
        "out_proj",
        "q_proj",
        "k_proj",
        "v_proj",
        "to_k",
        "to_q",
        "to_v",
        "to_out",
        "text_model",
        "down_blocks",
        "mid_block",
        "up_block",
        "proj_in",
        "proj_out",
        "transformer_blocks",
        "single_transformer_blocks",
        "single_blocks",
        "diffusion_model",
        "cond_stage_model",
        "first_stage_model",
        "conv_in",
        "conv_out",
        "time_emb_proj",
        "conv_shortcut",
        "time_embedding",
        "conv_norm_out",
        "double_blocks",
        "txt_attn",
        "img_attn",
        "input_blocks",
        "output_blocks",
        "middle_block",
        "skip_connection",
        "emb_layers",
        "in_layers",
        "out_layers",
        "add_q_proj",
        "add_k_proj",
        "add_v_proj",
        "add_out_proj",
        "ff_context",
        "norm_added_q",
        "norm_added_v",
        "to_add_out",
        "txt_mod",
        "img_mod",
        "txt_mlp",
        "img_mlp",
        "proj_mlp",
        "wi_0",
        "wi_1",
        "norm1_context",
        "ff_context",
        "x_embedder",
    };

    // record the positions of underscores that should NOT be replaced
    std::unordered_set<size_t> protected_positions;

    for (const auto& token : protected_tokens) {
        size_t start = 0;
        while ((start = name.find(token, start)) != std::string::npos) {
            size_t local_pos = token.find('_');
            while (local_pos != std::string::npos) {
                protected_positions.insert(start + local_pos);
                local_pos = token.find('_', local_pos + 1);
            }
            start += token.size();
        }
    }

    for (size_t i = 0; i < name.size(); ++i) {
        if (name[i] == '_' && !protected_positions.count(i)) {
            name[i] = '.';
        }
    }

    return name;
}

std::vector<std::string> cond_stage_model_prefix_vec = {
    "cond_stage_model.1.",
    "cond_stage_model.",
    "conditioner.embedders.",
    "text_encoders.",
};

std::vector<std::string> diffuison_model_prefix_vec = {
    "model.diffusion_model.",
};

std::vector<std::string> first_stage_model_prefix_vec = {
    "first_stage_model.",
    "vae.",
};

bool is_cond_stage_model_name(const std::string& name) {
    for (const auto& prefix : cond_stage_model_prefix_vec) {
        if (starts_with(name, prefix)) {
            return true;
        }
    }
    return false;
}

bool is_diffusion_model_name(const std::string& name) {
    for (const auto& prefix : diffuison_model_prefix_vec) {
        if (starts_with(name, prefix)) {
            return true;
        }
    }
    return false;
}

bool is_first_stage_model_name(const std::string& name) {
    for (const auto& prefix : first_stage_model_prefix_vec) {
        if (starts_with(name, prefix)) {
            return true;
        }
    }
    return false;
}

std::string convert_tensor_name(std::string name, SDVersion version) {
    std::unordered_map<std::string, std::string> prefix_map = {
        {"diffusion_model.", "model.diffusion_model."},
        {"unet.", "model.diffusion_model."},
        {"transformer.", "model.diffusion_model."},  // dit
        {"vae.", "first_stage_model."},
        {"text_encoder.", "cond_stage_model.transformer."},
        {"te.", "cond_stage_model.transformer."},
        {"text_encoder.2.", "cond_stage_model.1.transformer."},
        {"conditioner.embedders.0.open_clip.", "cond_stage_model."},
        // https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
        {"conditioner.embedders.0.", "cond_stage_model."},
        {"conditioner.embedders.1.", "cond_stage_model.1."},
        // {"te2.text_model.encoder.layers.", "cond_stage_model.1.model.transformer.resblocks."},
        {"te2.", "cond_stage_model.1.transformer."},
        {"te1.", "cond_stage_model.transformer."},
        {"te3.", "text_encoders.t5xxl.transformer."},
    };

    if (sd_version_is_flux(version)) {
        prefix_map["te1."] = "text_encoders.clip_l.transformer.";
    }

    replace_with_prefix_map(name, prefix_map);

    // diffusion model
    {
        for (const auto& prefix : diffuison_model_prefix_vec) {
            if (starts_with(name, prefix)) {
                name = convert_diffusion_model_name(name.substr(prefix.size()), prefix, version);
                name = prefix + name;
                break;
            }
        }
    }

    // cond_stage_model
    {
        for (const auto& prefix : cond_stage_model_prefix_vec) {
            if (starts_with(name, prefix)) {
                name = convert_cond_stage_model_name(name.substr(prefix.size()), prefix);
                name = prefix + name;
                break;
            }
        }
    }

    // first_stage_model
    {
        for (const auto& prefix : first_stage_model_prefix_vec) {
            if (starts_with(name, prefix)) {
                name = convert_first_stage_model_name(name.substr(prefix.size()), prefix);
                name = prefix + name;
                break;
            }
        }
    }

    return name;
}
