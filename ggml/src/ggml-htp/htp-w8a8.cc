#include "htp-w8a8.h"
#include "ggml-backend.h"
#include "ggml-htp.h"
#include "op_reg.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>

namespace {
struct Constants {
    ggml_tensor *tensor;
    W8A8Header header;
    std::string path;
    bool loaded = false;
};
// Non-owning bindings. The model runner owns the shared allocation context/buffers.
std::unordered_map<const ggml_tensor *, Constants> bindings;

std::string family(const char *name) {
    if (std::strncmp(name,"model.diffusion_model.",22)) return {};
    if (std::strstr(name,"adaLN_modulation.1.weight")) return "adaln";
    if (std::strstr(name,".mlp.fc1.weight")) return "fc1";
    if (std::strstr(name,".mlp.fc2.weight")) return "fc2";
    if (std::strstr(name,".attn.qkv.weight") || std::strstr(name,".attn2.qkv.weight")) return "qkv";
    if (std::strstr(name,".attn.proj.weight") || std::strstr(name,".attn2.proj.weight")) return "proj";
    return {};
}
void no_cpu_fallback(ggml_tensor *, int, int, void *) {
    GGML_ABORT("native W8A8 linear requires HTP dispatch; no FP16 weight is loaded");
}
}

bool ggml_backend_htp_w8a8_selected(const ggml_tensor *weight) {
    if (!weight) return false;
    const char *layers = std::getenv("SD_HTP_W8A8_LAYERS");
    if (!layers || !*layers || !std::strcmp(layers,"0")) return false;
    const std::string kind = family(weight->name);
    if (kind.empty()) return false;
    const std::string options = ","+std::string(layers)+",";
    return options.find(",all,") != std::string::npos || options.find(","+kind+",") != std::string::npos;
}

ggml_tensor *ggml_backend_htp_w8a8_create(ggml_context *ctx, const ggml_tensor *weight) {
    if (!ggml_backend_htp_w8a8_selected(weight)) return nullptr;
    if (bindings.count(weight)) GGML_ABORT("duplicate W8A8 binding %s",weight->name);
    const char *legacy = std::getenv("SD_HTP_W8A8_STAGING");
    if (legacy && *legacy && std::strcmp(legacy,"0"))
        GGML_ABORT("SD_HTP_W8A8_STAGING is obsolete; W8A8 replaces FP16 in shared RPCMEM chunks");
    const size_t k = weight->ne[0], n = weight->ne[1];
    if (weight->type!=GGML_TYPE_F16 || k<32 || k>6144 || k%32 || n<64 || n>16384 || n%64 ||
        weight->ne[2]!=1 || weight->ne[3]!=1)
        GGML_ABORT("unsupported selected W8A8 weight %s",weight->name);
    const char *dir = std::getenv("SD_HTP_W8A8_DIR");
    if (!dir || !*dir) GGML_ABORT("SD_HTP_W8A8_DIR is required when W8A8 layers are enabled");
    const std::string path = std::string(dir)+"/"+weight->name+".hmw8";
    FILE *file = std::fopen(path.c_str(),"rb");
    if (!file) GGML_ABORT("cannot open W8A8 constants %s",path.c_str());
    W8A8Header h{};
    const size_t size = 128+k*n+n*8+k*4;
    const bool valid = std::fread(&h,1,sizeof(h),file)==sizeof(h) && h.magic==0x38574d48 && h.version==1 &&
        h.k==k && h.n==n && h.weight_offset==128 && h.scale_offset==128+k*n &&
        h.sum_offset==h.scale_offset+n*4 && h.inverse_offset==h.sum_offset+n*4 && h.bytes==size;
    std::fclose(file);
    if (!valid) GGML_ABORT("incompatible W8A8 constants for %s",weight->name);
    ggml_tensor *blob = ggml_new_tensor_1d(ctx,GGML_TYPE_I8,size);
    ggml_set_name(blob,weight->name);
    bindings.emplace(weight,Constants{blob,h,path,false});
    return blob;
}

bool ggml_backend_htp_w8a8_load(const ggml_tensor *weight, uint64_t source_signature) {
    auto it = bindings.find(weight);
    if (it==bindings.end()) return false;
    Constants &entry = it->second;
    if (entry.loaded || !entry.tensor->data || !entry.tensor->buffer ||
        entry.header.original_f16_signature!=source_signature) return false;
    FILE *file = std::fopen(entry.path.c_str(),"rb");
    if (!file) return false;
    uint8_t *data = static_cast<uint8_t *>(entry.tensor->data);
    const size_t size = ggml_nbytes(entry.tensor);
    const bool read_ok = std::fread(data,1,size,file)==size && std::fgetc(file)==EOF;
    std::fclose(file);
    if (!read_ok || std::memcmp(data,&entry.header,sizeof(entry.header))) return false;
    const auto *scales = reinterpret_cast<const float *>(data+entry.header.scale_offset);
    const auto *inverse = reinterpret_cast<const float *>(data+entry.header.inverse_offset);
    for (size_t i=0;i<entry.header.n;++i) if (!std::isfinite(scales[i]) || scales[i]<=0) return false;
    for (size_t i=0;i<entry.header.k;++i) if (!std::isfinite(inverse[i]) || inverse[i]<=0) return false;
    entry.loaded = true;
    std::fprintf(stderr,"[htp-w8a8] loaded weight=%s k=%u n=%u bytes=%zu storage=shared-rpcmem replacement=1\n",
                 weight->name,entry.header.k,entry.header.n,size);
    return true;
}

ggml_tensor *ggml_backend_htp_w8a8_linear(ggml_context *ctx, const ggml_tensor *weight, ggml_tensor *activation) {
    const auto it = bindings.find(weight);
    if (it==bindings.end() || !it->second.loaded) GGML_ABORT("W8A8 constants not loaded for %s",weight->name);
    if (!activation || activation->type!=GGML_TYPE_F32 || activation->ne[0]!=weight->ne[0] ||
        ggml_nrows(activation)<1 || ggml_nrows(activation)>4096 || !ggml_is_contiguous(activation))
        GGML_ABORT("unsupported W8A8 activation for %s",weight->name);
    ggml_tensor *args[] = {activation,it->second.tensor};
    ggml_tensor *result = ggml_custom_4d(ctx,GGML_TYPE_F32,weight->ne[1],activation->ne[1],
        activation->ne[2],activation->ne[3],args,2,no_cpu_fallback,1,nullptr);
    ggml_set_name(result,"htp_w8a8_linear");
    return result;
}

bool htp_w8a8_is_node(const ggml_tensor *dst) {
    if (!dst || dst->op!=GGML_OP_CUSTOM || std::strcmp(dst->name,"htp_w8a8_linear") ||
        dst->type!=GGML_TYPE_F32 || !dst->src[0] || !dst->src[1] || dst->src[2]) return false;
    const auto *a = dst->src[0], *blob = dst->src[1];
    const size_t m = ggml_nrows(a), k = a->ne[0], n = dst->ne[0];
    return a->type==GGML_TYPE_F32 && blob->type==GGML_TYPE_I8 && ggml_is_contiguous(a) &&
        ggml_is_contiguous(dst) && ggml_is_contiguous(blob) && m>=1 && m<=4096 &&
        k>=32 && k<=6144 && k%32==0 && n>=64 && n<=16384 && n%64==0 &&
        ggml_nrows(dst)==(int64_t)m && ggml_nbytes(blob)==128+k*n+n*8+k*4;
}

void htp_w8a8_release(const ggml_tensor *weight) { bindings.erase(weight); }
void ggml_backend_htp_w8a8_unregister(const ggml_tensor *weight) { htp_w8a8_release(weight); }
