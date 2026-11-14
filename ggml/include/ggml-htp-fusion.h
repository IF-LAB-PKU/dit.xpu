#pragma once
#include "ggml.h"
#include <string.h>

// Store the contract in unused custom op_params bytes: runner output naming
// and graph-cut labels may legitimately replace tensor names.
static inline void ggml_htp_set_custom_tag(struct ggml_tensor *t,int kind) {
    uint32_t tag[2]={0x4854504eU,(uint32_t)kind};memcpy((char *)t->op_params+GGML_MAX_OP_PARAMS-sizeof(tag),tag,sizeof(tag));
}
static inline int ggml_htp_custom_tag(const struct ggml_tensor *t) {
    if(!t || t->op!=GGML_OP_CUSTOM) return 0;
    uint32_t tag[2];memcpy(tag,(const char *)t->op_params+GGML_MAX_OP_PARAMS-sizeof(tag),sizeof(tag));
    return tag[0]==0x4854504eU ? (int)tag[1] : 0;
}
// The persistent tag survives runner renaming; names recognize legacy nodes.
// Sources use the regular MUL_MAT order.
static inline int ggml_htp_f16_epilogue_mode(const struct ggml_tensor *t) {
    if (!t || t->op != GGML_OP_CUSTOM) return -1;
    int tag=ggml_htp_custom_tag(t);if(tag>=100 && tag<=103) return tag-100;
    if (!strcmp(t->name,"htp_f16_bias")) return 0;
    if (!strcmp(t->name,"htp_f16_gelu")) return 1;
    if (!strcmp(t->name,"htp_f16_quick_gelu")) return 2;
    if (!strcmp(t->name,"htp_f16_vae_store")) return 3;
    return -1;
}
static inline void ggml_htp_f16_epilogue_no_cpu(struct ggml_tensor *dst, int ith, int nth, void *data) {
    (void)dst; (void)ith; (void)nth; (void)data;
    GGML_ABORT("HTP custom op cannot fall back to CPU: node=%s tag=%d src0=%s src1=%s",dst->name,ggml_htp_custom_tag(dst),dst->src[0]?dst->src[0]->name:"none",dst->src[1]?dst->src[1]->name:"none");
}
static inline struct ggml_tensor *ggml_htp_f16_epilogue(
    struct ggml_context *ctx, struct ggml_tensor *w, struct ggml_tensor *a,
    struct ggml_tensor *bias, int mode, int64_t channels) {
    GGML_ASSERT(mode>=0 && mode<=3 && channels>0 && channels<=w->ne[1]);
    GGML_ASSERT(a->type==GGML_TYPE_F32 && w->type==GGML_TYPE_F16);
    // Ordinary bias parameters can live in CPU memory. Materialize a small
    // graph-owned F32 copy so the runner places it in RPCMEM with this op.
    if (bias) bias=bias->type!=GGML_TYPE_F32 ? ggml_cast(ctx,bias,GGML_TYPE_F32) : ggml_cont(ctx,bias);
    struct ggml_tensor *args[]={w,a,bias};
    struct ggml_tensor *out=ggml_custom_4d(ctx,GGML_TYPE_F32,
        mode==3 ? a->ne[1] : channels, mode==3 ? channels : a->ne[1],1,1,
        args,bias ? 3 : 2,ggml_htp_f16_epilogue_no_cpu,1,NULL);
    const char *names[]={"htp_f16_bias","htp_f16_gelu","htp_f16_quick_gelu","htp_f16_vae_store"};
    ggml_htp_set_custom_tag(out,100+mode);ggml_set_name(out,names[mode]);
    return out;
}

struct ggml_htp_native_metadata { int v[10]; float scale; };
struct ggml_htp_custom_storage { ggml_custom_op_t fun; int tasks; void *userdata; };
static inline int ggml_htp_native_kind(const struct ggml_tensor *t) {
    if (!t || t->op!=GGML_OP_CUSTOM) return 0;
    int tag=ggml_htp_custom_tag(t);if(tag>=1 && tag<=3) return tag;
    if (!strcmp(t->name,"htp_native_conv")) return 1;
    if (!strcmp(t->name,"htp_native_attention")) return 2;
    if (!strcmp(t->name,"htp_native_aux")) return 3;
    return 0;
}
static inline const struct ggml_htp_native_metadata *ggml_htp_native_meta(const struct ggml_tensor *t) {
    struct ggml_htp_custom_storage p; memcpy(&p,t->op_params,sizeof(p));
    return (const struct ggml_htp_native_metadata *)p.userdata;
}
static inline struct ggml_htp_native_metadata *ggml_htp_native_new_meta(struct ggml_context *ctx) {
    struct ggml_htp_native_metadata *p=(struct ggml_htp_native_metadata *)ggml_new_buffer(ctx,sizeof(*p));
    memset(p,0,sizeof(*p)); return p;
}
static inline struct ggml_tensor *ggml_htp_native_activation(struct ggml_context *ctx,struct ggml_tensor *x) {
    // Computed contiguous tensors are allocated directly in RPCMEM by the
    // runner. Copy only strided views or already allocated external leaves.
    return !ggml_is_contiguous(x) || (x->op==GGML_OP_NONE && x->buffer) ? ggml_cont(ctx,x) : x;
}
static inline struct ggml_tensor *ggml_htp_native_conv(struct ggml_context *ctx,
    struct ggml_tensor *w, struct ggml_tensor *x, struct ggml_tensor *bias,
    int kw,int kh,int oc,int sx,int sy,int px,int py,int dx,int dy) {
    struct ggml_htp_native_metadata *p=ggml_htp_native_new_meta(ctx);
    int v[]={kw,kh,oc,sx,sy,px,py,dx,dy}; memcpy(p->v,v,sizeof(v));
    if (bias) bias=bias->type!=GGML_TYPE_F32 ? ggml_cast(ctx,bias,GGML_TYPE_F32) : ggml_cont(ctx,bias);
    if (!ggml_is_contiguous(x)) x=ggml_htp_native_activation(ctx,x);
    int64_t ow=(x->ne[0]+2*px-dx*(kw-1)-1)/sx+1;
    int64_t oh=(x->ne[1]+2*py-dy*(kh-1)-1)/sy+1;
    p->v[9]=1;
    int32_t upscale_mode=0;memcpy(&upscale_mode,x->op_params,sizeof(upscale_mode));
    if(x->op==GGML_OP_UPSCALE && upscale_mode==GGML_SCALE_MODE_NEAREST &&
       x->ne[0]==2*x->src[0]->ne[0] && x->ne[1]==2*x->src[0]->ne[1] &&
       x->ne[2]==x->src[0]->ne[2] && x->ne[3]==x->src[0]->ne[3]) {
        x=x->src[0];if(!ggml_is_contiguous(x)) x=ggml_htp_native_activation(ctx,x);p->v[9]=2;
    }
    struct ggml_tensor *args[]={w,x,bias};
    struct ggml_tensor *out=ggml_custom_4d(ctx,GGML_TYPE_F32,ow,oh,oc,x->ne[3],
        args,bias?3:2,ggml_htp_f16_epilogue_no_cpu,1,p);
    ggml_htp_set_custom_tag(out,1);ggml_set_name(out,"htp_native_conv"); return out;
}
// Inputs/output [channels, tokens, batch]; mask [keys,queries,heads,batch].
static inline struct ggml_tensor *ggml_htp_native_attention(struct ggml_context *ctx,
    struct ggml_tensor *q,struct ggml_tensor *k,struct ggml_tensor *v,
    struct ggml_tensor *mask,int heads,float scale) {
    struct ggml_htp_native_metadata *p=ggml_htp_native_new_meta(ctx); p->v[0]=heads;p->scale=scale;
    q=ggml_htp_native_activation(ctx,q);k=ggml_htp_native_activation(ctx,k);v=ggml_htp_native_activation(ctx,v);
    if(mask) mask=mask->type!=GGML_TYPE_F32 ? ggml_cast(ctx,mask,GGML_TYPE_F32) : ggml_htp_native_activation(ctx,mask);
    struct ggml_tensor *args[]={q,k,v,mask};
    struct ggml_tensor *out=ggml_custom_4d(ctx,GGML_TYPE_F32,q->ne[0],q->ne[1],q->ne[2],1,
        args,mask?4:3,ggml_htp_f16_epilogue_no_cpu,1,p);
    ggml_htp_set_custom_tag(out,2);ggml_set_name(out,"htp_native_attention");return out;
}

// mode 0=LayerNorm,1=RMSNorm,2=GroupNorm,3=ADD,4=MUL,5=SiLU,6=GELU,7=QuickGELU.
static inline struct ggml_tensor *ggml_htp_native_aux(struct ggml_context *ctx,
 struct ggml_tensor *x,struct ggml_tensor *w,struct ggml_tensor *bias,int mode,int groups,float eps) {
 struct ggml_htp_native_metadata *p=ggml_htp_native_new_meta(ctx);p->v[0]=mode;p->v[1]=groups;p->scale=eps;
 x=ggml_htp_native_activation(ctx,x);
 if(w) w=w->type!=GGML_TYPE_F32?ggml_cast(ctx,w,GGML_TYPE_F32):mode<=2?ggml_cont(ctx,w):ggml_htp_native_activation(ctx,w);
 if(bias) bias=bias->type!=GGML_TYPE_F32?ggml_cast(ctx,bias,GGML_TYPE_F32):ggml_cont(ctx,bias);
 struct ggml_tensor *args[]={x,w,bias};
 struct ggml_tensor *out=ggml_custom_4d(ctx,GGML_TYPE_F32,x->ne[0],x->ne[1],x->ne[2],x->ne[3],args,bias?3:w?2:1,ggml_htp_f16_epilogue_no_cpu,1,p);
 ggml_htp_set_custom_tag(out,3);ggml_set_name(out,"htp_native_aux");return out;
}

// Transpose each row-major [rows,cols] matrix, preserving exact F32 values.
static inline struct ggml_tensor *ggml_htp_native_transpose(struct ggml_context *ctx,
 struct ggml_tensor *x,int rows,int cols,int batch,int64_t n0,int64_t n1,int64_t n2,int64_t n3) {
 GGML_ASSERT(ggml_is_contiguous(x) && ggml_nelements(x)==(int64_t)rows*cols*batch);
 struct ggml_htp_native_metadata *p=ggml_htp_native_new_meta(ctx);p->v[0]=8;p->v[3]=rows;p->v[4]=cols;p->v[5]=batch;
 struct ggml_tensor *args[]={x};
 struct ggml_tensor *out=ggml_custom_4d(ctx,GGML_TYPE_F32,n0,n1,n2,n3,args,1,ggml_htp_f16_epilogue_no_cpu,1,p);
 ggml_htp_set_custom_tag(out,3);ggml_set_name(out,"htp_native_aux");return out;
}
