# How to Use

You can run Z-Image with stable-diffusion.cpp on GPUs with 4GB of VRAM — or even less.

## Download weights

- Download Z-Image-Turbo
    - safetensors: https://huggingface.co/Comfy-Org/z_image_turbo/tree/main/split_files/diffusion_models
    - gguf: https://huggingface.co/leejet/Z-Image-Turbo-GGUF/tree/main
- Download Z-Image
    - safetensors: https://huggingface.co/Comfy-Org/z_image/tree/main/split_files/diffusion_models
    - gguf: https://huggingface.co/unsloth/Z-Image-GGUF/tree/main
- Download vae
    - safetensors: https://huggingface.co/black-forest-labs/FLUX.1-schnell/tree/main
- Download Qwen3 4b
    - safetensors: https://huggingface.co/Comfy-Org/z_image_turbo/tree/main/split_files/text_encoders
    - gguf: https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/tree/main

## Examples

### Z-Image-Turbo

```
.\bin\Release\sd-cli.exe --diffusion-model  z_image_turbo-Q3_K.gguf --vae ..\..\ComfyUI\models\vae\ae.sft  --llm ..\..\ComfyUI\models\text_encoders\Qwen3-4B-Instruct-2507-Q4_K_M.gguf -p "A cinematic, melancholic photograph of a solitary hooded figure walking through a sprawling, rain-slicked metropolis at night. The city lights are a chaotic blur of neon orange and cool blue, reflecting on the wet asphalt. The scene evokes a sense of being a single component in a vast machine. Superimposed over the image in a sleek, modern, slightly glitched font is the philosophical quote: 'THE CITY IS A CIRCUIT BOARD, AND I AM A BROKEN TRANSISTOR.' -- moody, atmospheric, profound, dark academic" --cfg-scale 1.0 -v --offload-to-cpu --diffusion-fa -H 1024 -W 512 --steps 8
```

<img width="256" alt="z-image example" src="../assets/z_image/q3_K.png" />

### Z-Image-Base

```
.\bin\Release\sd-cli.exe --diffusion-model  ..\..\ComfyUI\models\diffusion_models\z_image_bf16.safetensors --vae ..\..\ComfyUI\models\vae\ae.sft  --llm ..\..\ComfyUI\models\text_encoders\qwen_3_4b.safetensors -p "A cinematic, melancholic photograph of a solitary hooded figure walking through a sprawling, rain-slicked metropolis at night. The city lights are a chaotic blur of neon orange and cool blue, reflecting on the wet asphalt. The scene evokes a sense of being a single component in a vast machine. Superimposed over the image in a sleek, modern, slightly glitched font is the philosophical quote: 'THE CITY IS A CIRCUIT BOARD, AND I AM A BROKEN TRANSISTOR.' -- moody, atmospheric, profound, dark academic" --cfg-scale 5.0 -v --offload-to-cpu --diffusion-fa -H 1024 -W 512
```

<img width="256" alt="z-image example" src="../assets/z_image/base_bf16.png" />

## Comparison of Different Quantization Types

| bf16 | q8_0 | q6_K | q5_0 | q4_K | q4_0 | q3_K | q2_K|
|---|---|---|---|---|---|---|---|
| <img width="256" alt="bf16" src="../assets/z_image/bf16.png" /> | <img width="256" alt="q8_0" src="../assets/z_image/q8_0.png" /> | <img width="256" alt="q6_K" src="../assets/z_image/q6_K.png" /> | <img width="256" alt="q5_0" src="../assets/z_image/q5_0.png" />  | <img width="256" alt="q4_K" src="../assets/z_image/q4_K.png" /> | <img width="256" alt="q4_0" src="../assets/z_image/q4_0.png" /> | <img width="256" alt="q3_K" src="../assets/z_image/q3_K.png" /> | <img width="256" alt="q2_K" src="../assets/z_image/q2_K.png" /> |

## HTP attention routes

Z-Image keeps the dense text-first graph as the compatibility baseline. HTP
attention and block-sparse attention are opt-in runtime features; packed
attention is an additional opt-in and is disabled unless explicitly requested.

The route is selected per attention construction:

| Route | Conditions |
|---|---|
| `dense` | HTP is disabled, sparse attention is disabled, or the graph is not eligible |
| `reordered` | HTP and sparse attention are enabled, using generic raw HTP attention |
| `packed` | `reordered` is eligible and packed Z-Image QKV weights are available; image-only and text-only refiners use the single-stream packed op |

Set these environment variables when running on the Qualcomm 8 Gen 3 target:

```text
SD_HTP_COMPUTE=1
SD_HTP_ATTENTION=1
SD_HTP_PACKED_ATTENTION=1
SD_HTP_PACKED_Z_IMAGE=1
SD_HTP_Z_IMAGE_TRACE=1
```

`SD_HTP_PACKED_Z_IMAGE=1` does not force an unsupported graph onto the packed
kernel. Reference latents, residual-anchor steps, unsupported token geometry,
missing packed weights, and invalid packed shapes fall back to dense or raw
reordered attention. The stable trace format is:

```text
z_image_attention route=<dense|reordered|packed> fallback=<reason> heads=<n> kv_heads=<n> head_dim=<d> image_tokens=<n> text_tokens=<n> token_h=<h> token_w=<w>
```

Sparse static-halo geometry requires an image token count divisible by 32,
`token_h` divisible by 4, and `token_w` divisible by 8. The packed route
currently targets the Z-Image contract of 30 heads, 30 KV heads, and head
dimension 128. L2-prefetch and UserDMA halo modes remain HTP backend settings;
the route trace is identical for both modes. The noise refiner is image-only
and uses the actual image grid. The context refiner is text-only; its packed
route uses a synthetic 4-by-(tokens/4) grid and a full-keep block mask, so it
does not apply an image halo to text tokens. Both single-stream routes require
the padded token count to be divisible by 32.
