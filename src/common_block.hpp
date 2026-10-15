#ifndef __COMMON_BLOCK_HPP__
#define __COMMON_BLOCK_HPP__

#include "ggml-backend.h"
#include "ggml_extend.hpp"
#include "util.h"

class DownSampleBlock : public GGMLBlock {
protected:
    int channels;
    int out_channels;
    bool vae_downsample;

public:
    DownSampleBlock(int channels,
                    int out_channels,
                    bool vae_downsample = false)
        : channels(channels),
          out_channels(out_channels),
          vae_downsample(vae_downsample) {
        if (vae_downsample) {
            blocks["conv"] = std::shared_ptr<GGMLBlock>(new Conv2d(channels, out_channels, {3, 3}, {2, 2}, {0, 0}));
        } else {
            blocks["op"] = std::shared_ptr<GGMLBlock>(new Conv2d(channels, out_channels, {3, 3}, {2, 2}, {1, 1}));
        }
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
        // x: [N, channels, h, w]
        if (vae_downsample) {
            auto conv = std::dynamic_pointer_cast<Conv2d>(blocks["conv"]);

            x = ggml_ext_pad(ctx->ggml_ctx, x, 1, 1, 0, 0, ctx->circular_x_enabled, ctx->circular_y_enabled);
            x = conv->forward(ctx, x);
        } else {
            auto conv = std::dynamic_pointer_cast<Conv2d>(blocks["op"]);

            x = conv->forward(ctx, x);
        }
        return x;  // [N, out_channels, h/2, w/2]
    }
};

class UpSampleBlock : public GGMLBlock {
protected:
    int channels;
    int out_channels;

public:
    UpSampleBlock(int channels,
                  int out_channels)
        : channels(channels),
          out_channels(out_channels) {
        blocks["conv"] = std::shared_ptr<GGMLBlock>(new Conv2d(channels, out_channels, {3, 3}, {1, 1}, {1, 1}));
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
        // x: [N, channels, h, w]
        auto conv = std::dynamic_pointer_cast<Conv2d>(blocks["conv"]);

        x = ggml_upscale(ctx->ggml_ctx, x, 2, GGML_SCALE_MODE_NEAREST);  // [N, channels, h*2, w*2]
        x = conv->forward(ctx, x);                                       // [N, out_channels, h*2, w*2]
        return x;
    }
};

class GEGLU : public UnaryBlock {
protected:
    int64_t dim_in;
    int64_t dim_out;

public:
    GEGLU(int64_t dim_in, int64_t dim_out)
        : dim_in(dim_in), dim_out(dim_out) {
        blocks["proj"] = std::shared_ptr<GGMLBlock>(new Linear(dim_in, dim_out * 2));
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
        // x: [ne3, ne2, ne1, dim_in]
        // return: [ne3, ne2, ne1, dim_out]
        auto proj = std::dynamic_pointer_cast<Linear>(blocks["proj"]);

        x          = proj->forward(ctx, x);  // [ne3, ne2, ne1, dim_out*2]
        auto x_vec = ggml_ext_chunk(ctx->ggml_ctx, x, 2, 0, false);
        x          = x_vec[0];  // [ne3, ne2, ne1, dim_out]
        auto gate  = x_vec[1];  // [ne3, ne2, ne1, dim_out]

        gate = ggml_cont(ctx->ggml_ctx, gate);

        gate = ggml_ext_gelu(ctx->ggml_ctx, gate, true);

        x = ggml_mul(ctx->ggml_ctx, x, gate);  // [ne3, ne2, ne1, dim_out]

        return x;
    }
};

class GELU : public UnaryBlock {
public:
    GELU(int64_t dim_in, int64_t dim_out, bool bias = true) {
        blocks["proj"] = std::shared_ptr<GGMLBlock>(new Linear(dim_in, dim_out, bias));
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
        // x: [ne3, ne2, ne1, dim_in]
        // return: [ne3, ne2, ne1, dim_out]
        auto proj = std::dynamic_pointer_cast<Linear>(blocks["proj"]);

        x = proj->forward(ctx, x);
        x = ggml_ext_gelu(ctx->ggml_ctx, x, true);
        return x;
    }
};

class FeedForward : public GGMLBlock {
public:
    enum class Activation {
        GEGLU,
        GELU
    };
    FeedForward(int64_t dim,
                int64_t dim_out,
                int64_t mult          = 4,
                Activation activation = Activation::GEGLU,
                bool precision_fix    = false) {
        int64_t inner_dim = dim * mult;
        if (activation == Activation::GELU) {
            blocks["net.0"] = std::shared_ptr<GGMLBlock>(new GELU(dim, inner_dim));
        } else {
            blocks["net.0"] = std::shared_ptr<GGMLBlock>(new GEGLU(dim, inner_dim));
        }

        // net_1 is nn.Dropout(), skip for inference
        bool force_prec_f32 = false;
        float scale         = 1.f;
        if (precision_fix) {
            scale = 1.f / 128.f;
        }
        // The purpose of the scale here is to prevent NaN issues in certain situations.
        // For example, when using Vulkan without enabling force_prec_f32,
        // or when using CUDA but the weights are k-quants.
        blocks["net.2"] = std::shared_ptr<GGMLBlock>(new Linear(inner_dim, dim_out, true, false, force_prec_f32, scale));
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
        // x: [ne3, ne2, ne1, dim]
        // return: [ne3, ne2, ne1, dim_out]

        auto net_0 = std::dynamic_pointer_cast<UnaryBlock>(blocks["net.0"]);
        auto net_2 = std::dynamic_pointer_cast<Linear>(blocks["net.2"]);
        if (sd_backend_is(ctx->backend, "Vulkan")) {
            net_2->set_force_prec_f32(true);
        }

        x = net_0->forward(ctx, x);  // [ne3, ne2, ne1, inner_dim]
        x = net_2->forward(ctx, x);  // [ne3, ne2, ne1, dim_out]
        return x;
    }
};

#endif  // __COMMON_BLOCK_HPP__
