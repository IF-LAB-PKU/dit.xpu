// Benchmarks the two production RPC dispatches, not a copied diagnostic kernel.
#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-htp.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static std::vector<unsigned char> read(const char *path, size_t bytes) {
    FILE *f = std::fopen(path, "rb");
    GGML_ASSERT(f);
    std::vector<unsigned char> data(bytes);
    GGML_ASSERT(std::fread(data.data(), 1, bytes, f) == bytes && std::fgetc(f) == EOF);
    std::fclose(f);
    return data;
}
static ggml_context *context() {
    ggml_init_params p{ggml_tensor_overhead()*16 + ggml_graph_overhead() + 16384, nullptr, true};
    return ggml_init(p);
}
static void save(const char *prefix, const char *suffix, ggml_tensor *t) {
    std::vector<float> data(ggml_nelements(t));
    ggml_backend_tensor_get(t, data.data(), 0, ggml_nbytes(t));
    for (float x : data) GGML_ASSERT(std::isfinite(x));
    char path[1024]; std::snprintf(path, sizeof(path), "%s.%s.f32", prefix, suffix);
    FILE *f = std::fopen(path, "wb"); GGML_ASSERT(f);
    GGML_ASSERT(std::fwrite(data.data(), 1, ggml_nbytes(t), f) == ggml_nbytes(t));
    std::fclose(f);
}
int main(int argc, char **argv) {
    if (argc != 11) {
        std::fprintf(stderr,"usage: %s WEIGHT M K N PACKED_F16 ACTIVATION CONSTANT_DIR OUT_PREFIX TRIALS REPEATS\n",argv[0]);
        return 2;
    }
    const int m=std::atoi(argv[2]), k=std::atoi(argv[3]), n=std::atoi(argv[4]);
    const int trials=std::atoi(argv[9]), repeats=std::atoi(argv[10]);
    GGML_ASSERT(m>0 && k>0 && n>0 && trials>=3 && repeats>0);
    setenv("SD_HTP_W8A8_LAYERS", "all", 1);
    setenv("SD_HTP_W8A8_DIR", argv[7], 1);
    setenv("SD_HTP_LINEAR_PROFILE", "1", 1);
    // Initialize the real HTP backend directly. Registry enumeration also
    // initializes OpenCL and compiles GPU kernels, unrelated to this test.
    ggml_backend_t backend=ggml_backend_htp_init();
    GGML_ASSERT(backend && ggml_backend_htp_runtime_available());
    auto buft=ggml_backend_get_default_buffer_type(backend);
    auto packed=read(argv[5],size_t(k)*n*2);
    auto activation=read(argv[6],size_t(k)*m*4);
    uint64_t signature=14695981039346656037ULL;
    for (size_t i=0;i<4096;++i) signature=(signature ^ packed[i*(packed.size()-1)/4095])*1099511628211ULL;

    auto fp=context(); auto qp=context(); auto metadata=context();
    auto w=ggml_new_tensor_2d(fp,GGML_TYPE_F16,k,n);
    auto a=ggml_new_tensor_2d(fp,GGML_TYPE_F32,k,m);
    auto y=ggml_mul_mat(fp,w,a);
    auto fg=ggml_new_graph(fp); ggml_build_forward_expand(fg,y);
    auto fb=ggml_backend_alloc_ctx_tensors_from_buft(fp,buft); GGML_ASSERT(fb);
    ggml_backend_tensor_set(w,packed.data(),0,packed.size());
    ggml_backend_tensor_set(a,activation.data(),0,activation.size());
    ggml_backend_htp_register_packed_f16_weight(w,w);

    // The original quantized weight is metadata only, just as in inference.
    auto original=ggml_new_tensor_2d(metadata,GGML_TYPE_F16,k,n);
    ggml_set_name(original,argv[1]);
    auto blob=ggml_backend_htp_w8a8_create(qp,original); GGML_ASSERT(blob);
    auto qa=ggml_new_tensor_2d(qp,GGML_TYPE_F32,k,m);
    auto qb=ggml_backend_alloc_ctx_tensors_from_buft(qp,buft); GGML_ASSERT(qb);
    GGML_ASSERT(ggml_backend_htp_w8a8_load(original,signature));
    ggml_backend_tensor_set(qa,activation.data(),0,activation.size());
    auto outctx=context();
    auto qy=ggml_backend_htp_w8a8_linear(outctx,original,qa);
    auto qg=ggml_new_graph(outctx); ggml_build_forward_expand(qg,qy);
    auto ob=ggml_backend_alloc_ctx_tensors_from_buft(outctx,buft); GGML_ASSERT(ob);
    ggml_cgraph *graphs[]={fg,qg};
    auto compute=[&](int kind) {
        GGML_ASSERT(ggml_backend_graph_compute(backend,graphs[kind])==GGML_STATUS_SUCCESS);
        ggml_backend_synchronize(backend);
    };
    const char * cache_check=std::getenv("SD_HTP_LINEAR_CACHE_CHECK");
    if (cache_check && std::strcmp(cache_check,"1")==0) {
        // Exercise external scale visibility across changing activations and
        // output allocation lifetimes within one live HTP/OpenCL context.
        std::vector<float> base(size_t(m)*k),changed(base.size());
        std::memcpy(base.data(),activation.data(),activation.size());
        for (int phase=0;phase<5;++phase) {
            if (phase==2 || phase==4) {
                ggml_backend_buffer_free(ob);ggml_free(outctx);
                outctx=context();
                qy=ggml_backend_htp_w8a8_linear(outctx,original,qa);
                qg=ggml_new_graph(outctx);ggml_build_forward_expand(qg,qy);
                ob=ggml_backend_alloc_ctx_tensors_from_buft(outctx,buft);GGML_ASSERT(ob);
                graphs[1]=qg;
            }
            for (size_t i=0;i<base.size();++i) {
                const float factor=phase==1 ? 0.5f : phase==2 ? (i/size_t(k)%3==0 ? 0.25f : -2.0f) :
                                   phase==3 ? 0.0f : 1.0f;
                changed[i]=base[i]*factor;
            }
            ggml_backend_tensor_set(qa,changed.data(),0,activation.size());
            char phase_prefix[1024];
            std::snprintf(phase_prefix,sizeof(phase_prefix),"%s.cache%d",argv[8],phase);
            compute(1);save(phase_prefix,"w8a8",qy);
            compute(1);save(phase_prefix,"repeat",qy);
            std::printf("LINEAR_CACHE phase=%d output_churn=%d M=%d K=%d N=%d\n",
                        phase,phase==2 || phase==4,m,k,n);
        }
    } else {
      for (int i=0;i<5;++i) {compute(0);compute(1);}
      for (int t=0;t<trials;++t) for (int j=0;j<2;++j) {
        const int kind=(j+t)%2;
        auto start=std::chrono::steady_clock::now();
        double dsp_us=0;
        for(int r=0;r<repeats;++r) {
            compute(kind);
            dsp_us += ggml_backend_htp_get_last_linear_elapsed_us(backend);
        }
        double us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/repeats;
        std::printf("LINEAR type=%s M=%d K=%d N=%d trial=%d repeats=%d us=%.3f dsp_us=%.3f\n",kind?"W8A8":"FP16-production",m,k,n,t,repeats,us,dsp_us/repeats);
        std::fflush(stdout);
    }
      save(argv[8],"fp16",y); save(argv[8],"w8a8",qy);
    }
    ggml_backend_htp_w8a8_unregister(original);
    ggml_backend_htp_unregister_packed_f16_weight(w);
    ggml_backend_buffer_free(ob); ggml_backend_buffer_free(qb); ggml_backend_buffer_free(fb);
    ggml_free(outctx); ggml_free(metadata); ggml_free(qp); ggml_free(fp);
    ggml_backend_free(backend);
    return 0;
}
