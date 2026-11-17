#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-htp.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>

static void check(const std::vector<float>& actual, const std::vector<double>& expected, double tolerance, double relative = 0) {
    double maximum = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        GGML_ASSERT(std::isfinite(actual[i]));
        const double error = std::abs(actual[i] - expected[i]);
        maximum = std::max(maximum, error);
        if(error >= tolerance + relative*std::abs(expected[i]))
            printf("mismatch index=%zu actual=%.12g expected=%.12g error=%.12g bound=%.12g\n",i,actual[i],expected[i],error,tolerance+relative*std::abs(expected[i]));
        fflush(stdout);
        GGML_ASSERT(error < tolerance + relative*std::abs(expected[i]));
    }
    printf("max_abs=%.9g\n", maximum);
}

int main() {
    auto* backend = ggml_backend_init_by_name("HTP", nullptr);
    GGML_ASSERT(backend);
    ggml_backend_htp_set_hybrid_enabled(true);
    if (std::getenv("SD_HTP_STAGING_LIFETIME_CHECK")) {
        // Run single-session with a 64 MiB mapping budget and staging enabled.
        // Each 16 MiB backing fits; allocation/free over 1 GiB catches leaked
        // active/persistent byte accounting even when virtual addresses reuse.
        const int width = 32, tokens = 65536;
        std::vector<float> input(width*tokens, .25f);
        for (int iteration = 0; iteration < 80; ++iteration) {
            auto* ctx = ggml_init({1024*1024, nullptr, true});
            auto* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, tokens);
            auto* out = ggml_htp_native_aux(ctx, x, nullptr, nullptr, 1, 1, 1e-6f);
            auto* graph = ggml_new_graph_custom(ctx, 32, false);
            ggml_build_forward_expand(graph, out);
            auto* buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
            GGML_ASSERT(buffer);
            ggml_backend_htp_buffer_set_persistent_staging(buffer, iteration%2 == 0);
            ggml_backend_tensor_set(x, input.data(), 0, ggml_nbytes(x));
            GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
            float sample[2];
            ggml_backend_tensor_get(out, sample, 0, sizeof(float));
            ggml_backend_tensor_get(out, sample+1, ggml_nbytes(out)-sizeof(float), sizeof(float));
            const double expected = .25/sqrt(.25*.25 + 1e-6f);
            for (float value : sample) GGML_ASSERT(std::abs(value-expected) < 1e-5);
            ggml_backend_buffer_free(buffer);
            ggml_free(ctx);
        }
        ggml_backend_free(backend);
        puts("Staging mapping allocation/free lifetime PASSED (80 backings, >1 GiB cumulative)");
        return 0;
    }
    if (std::getenv("SD_HTP_DIT_CONCAT_BENCH")) {
        // Isolate real Klein single-block layout with resident inputs. This
        // measures graph dispatch + SwiGLU + concat, not a whole DiT forward.
        const int width=9216, aw=3072, tokens=768, stride=3*aw+2*width;
        auto* ctx=ggml_init({4*1024*1024,nullptr,true});
        auto* storage=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,stride,tokens);
        auto* x=ggml_view_2d(ctx,storage,2*width,tokens,stride*4,3*aw*4);
        auto* a=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,aw,tokens);
        auto* old=ggml_concat(ctx,a,ggml_htp_dit_swiglu(ctx,x,1),0);
        auto* fused=ggml_htp_dit_swiglu_concat(ctx,x,a,1);
        auto* g0=ggml_new_graph_custom(ctx,64,false);ggml_build_forward_expand(g0,old);
        auto* g1=ggml_new_graph_custom(ctx,64,false);ggml_build_forward_expand(g1,fused);
        auto* buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);GGML_ASSERT(buffer);
        std::vector<float> input(ggml_nelements(storage)),attention(ggml_nelements(a));
        for(size_t i=0;i<input.size();++i) input[i]=sinf(i*.07f)*30.f;
        for(size_t i=0;i<attention.size();++i) attention[i]=cosf(i*.013f)*17.f;
        ggml_backend_tensor_set(storage,input.data(),0,ggml_nbytes(storage));
        ggml_backend_tensor_set(a,attention.data(),0,ggml_nbytes(a));
        for(int threads : {1,4}) {
            ggml_backend_htp_set_n_threads(backend,threads);
            GGML_ASSERT(ggml_backend_graph_compute(backend,g0)==GGML_STATUS_SUCCESS);
            GGML_ASSERT(ggml_backend_graph_compute(backend,g1)==GGML_STATUS_SUCCESS);
            std::vector<float> before(ggml_nelements(old)),after(before.size());
            ggml_backend_tensor_get(old,before.data(),0,ggml_nbytes(old));
            ggml_backend_tensor_get(fused,after.data(),0,ggml_nbytes(fused));
            GGML_ASSERT(!memcmp(before.data(),after.data(),ggml_nbytes(old)));
            for(int round=0;round<10;++round) for(int mode : {0,1,1,0}) {
                int64_t start=ggml_time_us();
                GGML_ASSERT(ggml_backend_graph_compute(backend,mode?g1:g0)==GGML_STATUS_SUCCESS);
                printf("concat_bench threads=%d round=%d mode=%d elapsed_us=%lld exact=1\n",threads,round,mode,(long long)(ggml_time_us()-start));fflush(stdout);
            }
        }
        ggml_backend_buffer_free(buffer);ggml_free(ctx);ggml_backend_free(backend);return 0;
    }
    // Channel tails, multiple tokens, shared/per-batch parameters, nonzero
    // view offsets, constant rows, negative/zero gates and real Klein width.
    for (int width : {13, 128, 1536, 3072}) for (int tokens : {1, 19, 257}) for (int parameter_batches : {1, 2}) {
        const int batch = 2, count = width * tokens * batch;
        auto* ctx = ggml_init({4*1024*1024, nullptr, true});
        auto* x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, width, tokens, batch);
        auto* residual = ggml_dup_tensor(ctx, x);
        auto* parameters = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, 3*parameter_batches);
        const size_t bytes = width*parameter_batches*sizeof(float);
        auto* scale = ggml_view_2d(ctx, parameters, width, parameter_batches, width*sizeof(float), 0);
        auto* shift = ggml_view_2d(ctx, parameters, width, parameter_batches, width*sizeof(float), bytes);
        auto* gate = ggml_view_2d(ctx, parameters, width, parameter_batches, width*sizeof(float), 2*bytes);
        auto* norm = ggml_htp_dit_norm_modulate(ctx, x, scale, shift, 1e-6f);
        auto* added = ggml_htp_dit_gate_residual(ctx, x, gate, residual);
        GGML_ASSERT(ggml_backend_supports_op(backend, norm));
        GGML_ASSERT(ggml_backend_supports_op(backend, added));
        auto* graph = ggml_new_graph_custom(ctx, 64, false);
        ggml_build_forward_expand(graph, norm);
        ggml_build_forward_expand(graph, added);
        auto* buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        GGML_ASSERT(buffer);
        std::vector<float> xs(count), rs(count), ps(ggml_nelements(parameters)), output(count);
        for (int i = 0; i < count; ++i) {
            xs[i] = (i/width)%3 == 0 ? .125f : .1f + .7f*sinf(i*.017f);
            rs[i] = cosf(i*.03f)*3.f;
        }
        for (size_t i = 0; i < ps.size(); ++i) ps[i] = i%7 == 0 ? 0.f : .4f*sinf(i*.07f);
        ggml_backend_tensor_set(x, xs.data(), 0, ggml_nbytes(x));
        ggml_backend_tensor_set(residual, rs.data(), 0, ggml_nbytes(residual));
        ggml_backend_tensor_set(parameters, ps.data(), 0, ggml_nbytes(parameters));
        GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
        std::vector<double> expected(count);
        for (int row = 0; row < tokens*batch; ++row) {
            double mean = 0, variance = 0;
            for (int c = 0; c < width; ++c) mean += xs[row*width+c];
            mean /= width;
            for (int c = 0; c < width; ++c) variance += pow(xs[row*width+c]-mean, 2);
            double inv = 1./sqrt(variance/width + 1e-6f);
            int offset = (parameter_batches == 1 ? 0 : row/tokens)*width;
            for (int c = 0; c < width; ++c) {
                double n = (xs[row*width+c]-mean)*inv;
                expected[row*width+c] = n + n*ps[offset+c] + ps[width*parameter_batches+offset+c];
            }
        }
        printf("norm_mod width=%d tokens=%d parameter_batches=%d ", width, tokens, parameter_batches);
        ggml_backend_tensor_get(norm, output.data(), 0, ggml_nbytes(norm));
        check(output, expected, .00015);
        for (int i = 0; i < count; ++i) {
            int offset = (parameter_batches == 1 ? 0 : i/(width*tokens))*width + i%width;
            expected[i] = double(xs[i])*ps[2*width*parameter_batches+offset] + rs[i];
        }
        printf("gate_residual width=%d tokens=%d parameter_batches=%d ", width, tokens, parameter_batches);
        ggml_backend_tensor_get(added, output.data(), 0, ggml_nbytes(added));
        check(output, expected, .000002);
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
    }
    // Arbitrary matrices catch sign/transposition errors that symmetric cos/sin
    // inputs could hide; compare every output in the actual attention layout.
    for (int dim : {2, 30, 64, 128}) for (int tokens : {1, 19, 257}) for (int pe_batches : {1, 2}) {
        const int heads = 9, batch = 2;
        auto* ctx = ggml_init({4*1024*1024, nullptr, true});
        auto* x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, dim, heads, tokens, batch);
        auto* pe = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 2, 2, dim/2, tokens*pe_batches);
        auto* out = ggml_htp_dit_rope(ctx, x, pe);
        GGML_ASSERT(ggml_backend_supports_op(backend, out));
        auto* graph = ggml_new_graph_custom(ctx, 32, false);
        ggml_build_forward_expand(graph, out);
        auto* buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        GGML_ASSERT(buffer);
        std::vector<float> xs(ggml_nelements(x)), matrices(ggml_nelements(pe)), actual(xs.size());
        std::vector<double> expected(xs.size());
        for (size_t i = 0; i < xs.size(); ++i) xs[i] = sinf(i*.03f);
        for (size_t i = 0; i < matrices.size(); ++i) matrices[i] = cosf(i*.07f)*.7f;
        ggml_backend_tensor_set(x, xs.data(), 0, ggml_nbytes(x));
        ggml_backend_tensor_set(pe, matrices.data(), 0, ggml_nbytes(pe));
        GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
        ggml_backend_tensor_get(out, actual.data(), 0, ggml_nbytes(out));
        for (int b = 0; b < batch; ++b) for (int t = 0; t < tokens; ++t) for (int h = 0; h < heads; ++h) {
            int src = ((b*tokens+t)*heads+h)*dim, dst = ((b*heads+h)*tokens+t)*dim;
            int pos = ((pe_batches == 1 ? 0 : b)*tokens+t)*dim*2;
            for (int d = 0; d < dim; d += 2) {
                expected[dst+d] = double(xs[src+d])*matrices[pos+2*d] + double(xs[src+d+1])*matrices[pos+2*d+1];
                expected[dst+d+1] = double(xs[src+d])*matrices[pos+2*d+2] + double(xs[src+d+1])*matrices[pos+2*d+3];
            }
        }
        printf("rope dim=%d tokens=%d pe_batches=%d ", dim, tokens, pe_batches);
        check(actual, expected, .000002);
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
    }
    for (int width : {13, 96, 9216}) for (int tokens : {1, 31}) for (int gate_first : {0, 1}) for (int strided : {0, 1}) {
        auto* ctx = ggml_init({4*1024*1024, nullptr, true});
        const int row_stride = 2*width + (strided ? 17 : 0), offset = strided ? 7 : 0;
        auto* storage = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, row_stride, tokens, 2);
        auto* x = strided ? ggml_view_3d(ctx, storage, 2*width, tokens, 2,
            row_stride*sizeof(float), row_stride*tokens*sizeof(float), offset*sizeof(float)) : storage;
        auto* out = ggml_htp_dit_swiglu(ctx, x, gate_first);
        if (strided) GGML_ASSERT(out->src[0] == x);
        GGML_ASSERT(ggml_backend_supports_op(backend, out));
        auto* graph = ggml_new_graph_custom(ctx, 32, false);
        ggml_build_forward_expand(graph, out);
        auto* buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        GGML_ASSERT(buffer);
        std::vector<float> input(ggml_nelements(storage)), actual(ggml_nelements(out));
        std::vector<double> expected(actual.size());
        for (size_t i = 0; i < input.size(); ++i) input[i] = sinf(i*.07f)*30.f;
        for (int row = 0; row < tokens*2; ++row) for (int c = 0; c < width; ++c) {
            double gate = input[row*row_stride+offset+c+(gate_first?0:width)];
            double up = input[row*row_stride+offset+c+(gate_first?width:0)];
            expected[row*width+c] = gate/(1.+exp(-gate))*up;
        }
        ggml_backend_tensor_set(storage, input.data(), 0, ggml_nbytes(storage));
        GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
        ggml_backend_tensor_get(out, actual.data(), 0, ggml_nbytes(out));
        printf("swiglu width=%d tokens=%d gate_first=%d strided=%d ", width, tokens, gate_first, strided);
        // F32 sigmoid/multiply: relative bound plus a near-zero floor.
        check(actual, expected, .000002, .000002);
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
    }
    // Concatenated output uses unequal/tail widths and distinct batch rows.
    // Attention must copy bit-for-bit; SwiGLU must also match mode 12 exactly.
    for (int width : {13, 96, 9216}) for (int tokens : {1, 31})
    for (int gate_first : {0, 1}) for (int strided : {0, 1}) for (int aw : {3, 3072}) {
        const int batch=2, stride=2*width+(strided?17:0), offset=strided?7:0;
        auto* ctx=ggml_init({4*1024*1024,nullptr,true});
        auto* storage=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,stride*tokens*batch+offset);
        auto* x=ggml_view_3d(ctx,storage,2*width,tokens,batch,stride*4,stride*tokens*4,offset*4);
        auto* attention=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,aw,tokens,batch);
        auto* raw=ggml_htp_dit_swiglu(ctx,x,gate_first);
        auto* out=ggml_htp_dit_swiglu_concat(ctx,x,attention,gate_first);
        GGML_ASSERT(ggml_backend_supports_op(backend,out));
        auto* graph=ggml_new_graph_custom(ctx,64,false);
        ggml_build_forward_expand(graph,raw);ggml_build_forward_expand(graph,out);
        auto* buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);GGML_ASSERT(buffer);
        std::vector<float> input(ggml_nelements(storage)),a(ggml_nelements(attention)),actual(ggml_nelements(out)),plain(ggml_nelements(raw));
        for(size_t i=0;i<input.size();++i) input[i]=sinf(i*.07f)*30.f;
        for(size_t i=0;i<a.size();++i) a[i]=cosf(i*.013f)*17.f;
        a[0]=-0.f;
        ggml_backend_tensor_set(storage,input.data(),0,ggml_nbytes(storage));
        ggml_backend_tensor_set(attention,a.data(),0,ggml_nbytes(attention));
        GGML_ASSERT(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS);
        ggml_backend_tensor_get(out,actual.data(),0,ggml_nbytes(out));
        ggml_backend_tensor_get(raw,plain.data(),0,ggml_nbytes(raw));
        std::vector<double> expected(actual.size());
        for(int row=0;row<tokens*batch;++row) {
            GGML_ASSERT(!memcmp(actual.data()+row*(aw+width),a.data()+row*aw,aw*sizeof(float)));
            GGML_ASSERT(!memcmp(actual.data()+row*(aw+width)+aw,plain.data()+row*width,width*sizeof(float)));
            for(int c=0;c<aw;++c) expected[row*(aw+width)+c]=a[row*aw+c];
            for(int c=0;c<width;++c) {
                double gate=input[offset+row*stride+c+(gate_first?0:width)];
                double up=input[offset+row*stride+c+(gate_first?width:0)];
                expected[row*(aw+width)+aw+c]=gate/(1.+exp(-gate))*up;
            }
        }
        printf("swiglu_concat width=%d tokens=%d gate_first=%d strided=%d attention=%d ",width,tokens,gate_first,strided,aw);
        check(actual,expected,.000002,.000002);
        ggml_backend_buffer_free(buffer);ggml_free(ctx);
    }
    ggml_backend_free(backend);
    puts("DiT fused auxiliaries PASSED");
}
