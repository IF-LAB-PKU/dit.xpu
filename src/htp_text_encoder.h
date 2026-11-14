#ifndef SD_HTP_TEXT_ENCODER_H
#define SD_HTP_TEXT_ENCODER_H

#include <string>
#include "util.h"

inline std::string sd_htp_text_prefix(const std::string& kind) {
    if (kind != "clip_l" && kind != "clip_g" && kind != "t5xxl") return {};
    return "text_encoders." + kind + ".transformer.";
}

// Only Linear consumers may read tiled weights. Embeddings (including T5's
// relative position bias table) must remain row-major for GET_ROWS.
inline bool sd_htp_text_linear_name(const std::string& name, const std::string& kind) {
    const auto prefix = sd_htp_text_prefix(kind);
    if (prefix.empty() || !starts_with(name, prefix)) return false;
    const auto local = name.substr(prefix.size());
    if (kind == "t5xxl") {
        if (!starts_with(local, "encoder.block.")) return false;
        for (const auto* suffix : {".SelfAttention.q.weight", ".SelfAttention.k.weight",
                                   ".SelfAttention.v.weight", ".SelfAttention.o.weight",
                                   ".DenseReluDense.wi.weight", ".DenseReluDense.wi_0.weight",
                                   ".DenseReluDense.wi_1.weight", ".DenseReluDense.wo.weight"}) {
            if (ends_with(local, suffix)) return true;
        }
    } else {
        if (local == "text_model.text_projection") return true;
        if (!starts_with(local, "text_model.encoder.layers.")) return false;
        for (const auto* suffix : {".self_attn.q_proj.weight", ".self_attn.k_proj.weight",
                                   ".self_attn.v_proj.weight", ".self_attn.out_proj.weight",
                                   ".self_attn.in_proj.weight", ".mlp.fc1.weight", ".mlp.fc2.weight"}) {
            if (ends_with(local, suffix)) return true;
        }
    }
    return false;
}

#endif
