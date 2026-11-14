// Isolate epilogue error from the existing HMX FP16 accumulation error.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-htp.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

int main() {
    auto *htp=ggml_backend_init_by_name("HTP",nullptr);
    GGML_ASSERT(htp && ggml_backend_htp_runtime_available());
    ggml_backend_htp_set_hybrid_enabled(true);
    for (int m : {1,31,33,77,257}) {
        auto *ctx=ggml_init({1024*1024,nullptr,true});
        auto *w=ggml_new_tensor_2d(ctx,GGML_TYPE_F16,64,128);
        auto *a=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,64,m);
        auto *b=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,128);
        auto *b3=ggml_new_tensor_1d(ctx,GGML_TYPE_F16,3);
        auto *raw=ggml_mul_mat(ctx,w,a);
        std::vector<ggml_tensor *> outputs;
        for (int mode=0;mode<4;++mode) {
            outputs.push_back(ggml_htp_f16_epilogue(ctx,w,a,b,mode,128));
            outputs.push_back(ggml_htp_f16_epilogue(ctx,w,a,nullptr,mode,128));
        }
        outputs.push_back(ggml_htp_f16_epilogue(ctx,w,a,b3,3,3));
        auto *g=ggml_new_graph_custom(ctx,128,false);
        ggml_build_forward_expand(g,raw);
        for (auto *o:outputs) ggml_build_forward_expand(g,o);
        auto *buf=ggml_backend_alloc_ctx_tensors(ctx,htp);
        GGML_ASSERT(buf);
        std::vector<ggml_fp16_t> weights(64*128), packed(weights.size());
        std::vector<float> activation(64*m),bias(128);
        for (size_t i=0;i<weights.size();++i) weights[i]=ggml_fp32_to_fp16(.11f*std::sin(float(i)*.17f));
        for (size_t i=0;i<activation.size();++i) activation[i]=8.f*std::cos(float(i)*.13f);
        // Exercise negative, positive and saturated GELU/QuickGELU inputs.
        for (int i=0;i<128;++i) bias[i]=(i-32)*.6f;
        ggml_backend_htp_pack_f16_weight(packed.data(),weights.data(),64,128);
        ggml_backend_tensor_set(w,packed.data(),0,ggml_nbytes(w));
        ggml_backend_tensor_set(a,activation.data(),0,ggml_nbytes(a));
        ggml_backend_tensor_set(b,bias.data(),0,ggml_nbytes(b));
        ggml_fp16_t bias3[]={ggml_fp32_to_fp16(.1f),ggml_fp32_to_fp16(-.2f),ggml_fp32_to_fp16(.3f)};
        ggml_backend_tensor_set(b3,bias3,0,sizeof(bias3));
        ggml_backend_htp_register_packed_f16_weight(w,w);
        GGML_ASSERT(ggml_backend_graph_compute(htp,g)==GGML_STATUS_SUCCESS);
        std::vector<float> reference(128*m);
        ggml_backend_tensor_get(raw,reference.data(),0,ggml_nbytes(raw));
        for (size_t index=0;index<outputs.size();++index) {
            int mode=index==8 ? 3 : index/2, channels=index==8 ? 3 : 128;
            bool has_bias=index==8 || index%2==0;
            std::vector<float> actual(channels*m);
            ggml_backend_tensor_get(outputs[index],actual.data(),0,ggml_nbytes(outputs[index]));
            float maximum=0;
            for (int row=0;row<m;++row) for (int ch=0;ch<channels;++ch) {
                float x=reference[row*128+ch]+(has_bias ? (index==8 ? ggml_fp16_to_fp32(bias3[ch]) : bias[ch]) : 0.f);
                float expected=x;
                if (mode==1) expected=.5f*x*(1.f+std::tanh(.7978845608028654f*x*(1.f+.044715f*x*x)));
                if (mode==2) expected=x/(1.f+std::exp(-1.702f*x));
                float y=actual[mode==3 ? ch*m+row : row*channels+ch];
                GGML_ASSERT(std::isfinite(y));
                maximum=std::max(maximum,std::abs(y-expected));
            }
            std::printf("m=%d mode=%d channels=%d bias=%d max_abs=%.9g\n",m,mode,channels,has_bias,maximum);
            GGML_ASSERT(maximum<.0001f);
        }
        ggml_backend_htp_unregister_packed_f16_weight(w);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }
    ggml_backend_free(htp);
    std::puts("HTP F16 fused epilogues PASSED");
}
