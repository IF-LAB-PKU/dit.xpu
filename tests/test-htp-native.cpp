#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-htp.h"
#include <cmath>
#include <cstdio>
#include <vector>
#include <algorithm>
int main() {
 auto *htp=ggml_backend_init_by_name("HTP",nullptr);GGML_ASSERT(htp);ggml_backend_htp_set_hybrid_enabled(true);
 // Qwen3 uses compact KV heads. Check its 4:1 GQA mapping, causal mask,
 // padded query rows and token tails against an independent double reference.
 for (int len : {1, 19, 32, 77}) {
  const int dim=128,heads=32,kvheads=8;
  auto *ctx=ggml_init({4*1024*1024,nullptr,true});
  auto *qb=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,dim,heads,len);
  auto *kb=ggml_new_tensor_3d(ctx,GGML_TYPE_F16,dim,kvheads,len);
  auto *vb=ggml_dup_tensor(ctx,kb);
  const int mask_stride=GGML_PAD(len,64);
  auto *mask=ggml_new_tensor_2d(ctx,GGML_TYPE_F16,mask_stride,mask_stride);
  auto *out=ggml_flash_attn_ext(ctx,ggml_permute(ctx,qb,0,2,1,3),ggml_permute(ctx,kb,0,2,1,3),
                               ggml_permute(ctx,vb,0,2,1,3),mask,1.f/sqrtf(dim),0,0);
  auto *g=ggml_new_graph_custom(ctx,32,false);ggml_build_forward_expand(g,out);
  GGML_ASSERT(ggml_backend_supports_op(htp,out));auto *buf=ggml_backend_alloc_ctx_tensors(ctx,htp);GGML_ASSERT(buf);
  std::vector<float> qs(ggml_nelements(qb)),ys(ggml_nelements(out));
  std::vector<ggml_fp16_t> ks(ggml_nelements(kb)),vs(ks.size()),ms(ggml_nelements(mask));
  for(size_t i=0;i<qs.size();++i)qs[i]=.4f*sinf(i*.139f);
  for(size_t i=0;i<ks.size();++i){ks[i]=ggml_fp32_to_fp16(.3f*cosf(i*.117f));vs[i]=ggml_fp32_to_fp16(sinf(i*.019f));}
  for(int row=0;row<mask_stride;++row)for(int col=0;col<mask_stride;++col)ms[row*mask_stride+col]=ggml_fp32_to_fp16(col>=len||col>row?-INFINITY:0);
  ggml_backend_tensor_set(qb,qs.data(),0,ggml_nbytes(qb));ggml_backend_tensor_set(kb,ks.data(),0,ggml_nbytes(kb));
  ggml_backend_tensor_set(vb,vs.data(),0,ggml_nbytes(vb));ggml_backend_tensor_set(mask,ms.data(),0,ggml_nbytes(mask));
  GGML_ASSERT(ggml_backend_graph_compute(htp,g)==GGML_STATUS_SUCCESS);
  GGML_ASSERT(ggml_backend_htp_get_last_attention_elapsed_us(htp)>0);
  ggml_backend_tensor_get(out,ys.data(),0,ggml_nbytes(out));double maxerr=0;
  for(int row=0;row<len;++row)for(int h=0;h<heads;++h){
   int kh=h/(heads/kvheads);std::vector<double> scores(row+1);double maximum=-INFINITY,sum=0;
   for(int col=0;col<=row;++col){double dot=0;for(int d=0;d<dim;++d)dot+=double(qs[(row*heads+h)*dim+d])*ggml_fp16_to_fp32(ks[(col*kvheads+kh)*dim+d]);scores[col]=dot/sqrt(double(dim));maximum=std::max(maximum,scores[col]);}
   for(auto &s:scores){s=exp(s-maximum);sum+=s;}
   for(int d=0;d<dim;++d){double ref=0;for(int col=0;col<=row;++col)ref+=scores[col]/sum*ggml_fp16_to_fp32(vs[(col*kvheads+kh)*dim+d]);
    float actual=ys[(row*heads+h)*dim+d];GGML_ASSERT(std::isfinite(actual));maxerr=std::max(maxerr,std::abs(actual-ref));}
  }
  printf("qwen3_gqa len=%d heads=%d kv_heads=%d max_abs=%g\n",len,heads,kvheads,maxerr);fflush(stdout);GGML_ASSERT(maxerr<.012);
  ggml_backend_buffer_free(buf);ggml_free(ctx);
 }
 for(int dim:{64,512}) for(int mode=0;mode<4;++mode) for(int len:{1,31,77,128}) {
  int heads=dim==512?1:3,batch=2,keys=len+3,channels=heads*dim;
  auto *ctx=ggml_init({4*1024*1024,nullptr,true});
  auto *q=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,channels,len,batch);
  auto *k=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,channels,keys,batch);
  auto *v=ggml_dup_tensor(ctx,k);
  auto *mask=ggml_new_tensor_4d(ctx,GGML_TYPE_F32,keys,len,mode>=2?heads:1,1);
  float scale=mode>=2?1.f:1.f/sqrtf(dim);
  auto *out=ggml_htp_native_attention(ctx,q,k,v,mode?mask:nullptr,heads,scale);
  auto *g=ggml_new_graph_custom(ctx,128,false);ggml_build_forward_expand(g,out);
  GGML_ASSERT(ggml_backend_supports_op(htp,out));auto *buf=ggml_backend_alloc_ctx_tensors(ctx,htp);GGML_ASSERT(buf);
  std::vector<float> qs(ggml_nelements(q)),ks(ggml_nelements(k)),vs(ks.size()),ms(ggml_nelements(mask)),ys(ggml_nelements(out));
  for(size_t i=0;i<qs.size();++i) qs[i]=sinf(i*.139f)*.7f;
  for(size_t i=0;i<ks.size();++i) {ks[i]=cosf(i*.117f)*.8f;vs[i]=sinf(i*.019f);}
  for(int h=0;h<(mode>=2?heads:1);++h) for(int row=0;row<len;++row) for(int col=0;col<keys;++col)
    ms[(h*len+row)*keys+col]=mode==3 && row%7==0 ? -INFINITY : col>row&&mode==1 ? -INFINITY : mode>=2 ? 2.f*sinf(h*7+col*.2f+row*.1f)-.5f : 0.f;
  ggml_backend_tensor_set(q,qs.data(),0,ggml_nbytes(q));ggml_backend_tensor_set(k,ks.data(),0,ggml_nbytes(k));ggml_backend_tensor_set(v,vs.data(),0,ggml_nbytes(v));ggml_backend_tensor_set(mask,ms.data(),0,ggml_nbytes(mask));
  GGML_ASSERT(ggml_backend_graph_compute(htp,g)==GGML_STATUS_SUCCESS);ggml_backend_tensor_get(out,ys.data(),0,ggml_nbytes(out));
  double maxerr=0,squared=0;
  for(int b=0;b<batch;++b) for(int h=0;h<heads;++h) for(int row=0;row<len;++row) {
    std::vector<double> probs(keys);double maximum=-INFINITY,sum=0;
    for(int col=0;col<keys;++col) {double dot=0;for(int d=0;d<dim;++d) dot+=qs[(b*len+row)*channels+h*dim+d]*ks[(b*keys+col)*channels+h*dim+d];
      probs[col]=dot*scale+(mode?ms[((mode>=2?h:0)*len+row)*keys+col]:0);maximum=std::max(maximum,probs[col]);}
    for(auto &p:probs) {p=std::isfinite(maximum)?exp(p-maximum):0;sum+=p;} if(sum==0) sum=1;
    for(int d=0;d<dim;++d) {double expected=0;for(int col=0;col<keys;++col) expected+=probs[col]/sum*vs[(b*keys+col)*channels+h*dim+d];
      float actual=ys[(b*len+row)*channels+h*dim+d];GGML_ASSERT(std::isfinite(actual));double e=actual-expected;maxerr=std::max(maxerr,std::abs(e));squared+=e*e;}
  }
  printf("attention dim=%d mode=%d len=%d max_abs=%g rmse=%g\n",dim,mode,len,maxerr,sqrt(squared/ys.size()));fflush(stdout);GGML_ASSERT(maxerr<.012);
  ggml_backend_buffer_free(buf);ggml_free(ctx);
 }
 for(int width:{13,64,768}) for(int mode=0;mode<8;++mode) for(int fused=0;fused<(mode==2?2:1);++fused) {
  int channels=12,height=7,batch=2,count=width*height*channels*batch,normwidth=mode==2?channels:width;
  auto *ctx=ggml_init({4*1024*1024,nullptr,true});auto *x=ggml_new_tensor_4d(ctx,GGML_TYPE_F32,width,height,channels,batch);
  auto *w=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,mode<=2?normwidth:count);auto *b=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,normwidth);
  auto *out=ggml_htp_native_aux(ctx,x,mode<=4?w:nullptr,mode<=2?b:nullptr,mode,3,1e-6f);
  ((ggml_htp_native_metadata *)ggml_htp_native_meta(out))->v[2]=fused;
  auto *g=ggml_new_graph_custom(ctx,128,false);ggml_build_forward_expand(g,out);auto *buf=ggml_backend_alloc_ctx_tensors(ctx,htp);GGML_ASSERT(buf);
  std::vector<float> xs(count),ws(ggml_nelements(w)),bs(normwidth),actual(count),expected(count);
  for(int i=0;i<count;++i) xs[i]=.01f+sinf(i*.017f)*.02f;
  for(size_t i=0;i<ws.size();++i) ws[i]=.3f+cosf(i*.03f);for(int i=0;i<normwidth;++i) bs[i]=sinf(i*.2f)*.1f;
  ggml_backend_tensor_set(x,xs.data(),0,ggml_nbytes(x));ggml_backend_tensor_set(w,ws.data(),0,ggml_nbytes(w));ggml_backend_tensor_set(b,bs.data(),0,ggml_nbytes(b));
  GGML_ASSERT(ggml_backend_graph_compute(htp,g)==GGML_STATUS_SUCCESS);ggml_backend_tensor_get(out,actual.data(),0,ggml_nbytes(out));
  int group=mode==2?width*height*(channels/3):width;
  if(mode<=2) for(int first=0;first<count;first+=group) {
   double mean=0,var=0;if(mode!=1) {for(int j=0;j<group;++j) mean+=xs[first+j];mean/=group;}
   for(int j=0;j<group;++j) {double v=xs[first+j]-mean;var+=v*v;}double inv=1./sqrt(var/group+1e-6);
   for(int j=0;j<group;++j) {int ch=mode==2?((first+j)/(width*height))%channels:j;double y=(xs[first+j]-mean)*inv*ws[ch]+bs[ch];expected[first+j]=fused?y/(1.+exp(-y)):y;}
  } else for(int i=0;i<count;++i) {double x=xs[i];expected[i]=mode==3?x+ws[i]:mode==4?x*ws[i]:mode==5?x/(1+exp(-x)):mode==6?.5*x*(1+tanh(.7978845608028654*x*(1+.044715*x*x))):x/(1+exp(-1.702*x));}
  double maximum=0;for(int i=0;i<count;++i) {GGML_ASSERT(std::isfinite(actual[i]));maximum=std::max(maximum,std::abs(double(actual[i])-expected[i]));}
  printf("aux mode=%d width=%d fused=%d max_abs=%g\n",mode,width,fused,maximum);fflush(stdout);GGML_ASSERT(maximum<.00015);
  ggml_backend_buffer_free(buf);ggml_free(ctx);
 }
 for(int rows:{1,13,64,77}) for(int cols:{3,32,99}) {
  auto *ctx=ggml_init({1024*1024,nullptr,true});auto *x=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,cols,rows,2);
  auto *y=ggml_htp_native_transpose(ctx,x,rows,cols,2,rows,cols,2,1);
  auto *z=ggml_htp_native_transpose(ctx,y,cols,rows,2,cols,rows,2,1);
  auto *g=ggml_new_graph_custom(ctx,32,false);ggml_build_forward_expand(g,z);auto *buf=ggml_backend_alloc_ctx_tensors(ctx,htp);GGML_ASSERT(buf);
  std::vector<float> input(ggml_nelements(x)),actual(input.size());for(size_t i=0;i<input.size();++i) input[i]=sinf(i*.17f);
  ggml_backend_tensor_set(x,input.data(),0,ggml_nbytes(x));GGML_ASSERT(ggml_backend_graph_compute(htp,g)==GGML_STATUS_SUCCESS);
  ggml_backend_tensor_get(z,actual.data(),0,ggml_nbytes(z));GGML_ASSERT(input==actual);
  std::vector<float> transposed(input.size());ggml_backend_tensor_get(y,transposed.data(),0,ggml_nbytes(y));
  for(int b=0;b<2;++b) for(int row=0;row<rows;++row) for(int col=0;col<cols;++col) GGML_ASSERT(transposed[(b*cols+col)*rows+row]==input[(b*rows+row)*cols+col]);
  printf("transpose rows=%d cols=%d batch=2 exact=1\n",rows,cols);ggml_backend_buffer_free(buf);ggml_free(ctx);
 }
 ggml_backend_free(htp);puts("Native attention, norm, elementwise and transpose PASSED");
}
