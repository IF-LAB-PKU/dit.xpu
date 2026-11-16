# FLUX.2 Klein 4B on HTP

This opt-in path reuses the SD3 offline W16A32 Linear/convolution kernels,
native model auxiliaries, Flash Attention and segment-scoped weight streaming.
It runs a dynamic ggml graph; no QNN graph export is involved. The tested target
is Klein 4B at 256x256 on Snapdragon 8 Gen 3 / HTP v75.

## Models

Convert the three standalone components with the matching runtime:

```sh
sd-cli -M convert -m flux-2-klein-4b-f16.safetensors \
  --htp-pack-dit -o flux2_klein_4b_htp_f16.gguf
sd-cli -M convert -m qwen3-4b.gguf \
  --htp-pack-text qwen3 -o qwen3_4b_htp_w16a32.gguf
sd-cli -M convert -m flux2_ae.safetensors \
  --htp-pack-vae -o flux2_vae_htp_w16a32.gguf
```

The tested Qwen3 package has 252 packed Linear tensors, DiT has 88 and VAE
has 72 packed convolution/projection tensors. Embeddings and norm weights
remain row-major. Packed files require HTP; retain original models for CPU
reference. Dequantizing a Q8 source increases its size and does not recover
the original unquantized weights.

## Runtime

Build with `GGML_HTP=ON`; for this CPU/HTP configuration use `SD_OPENCL=OFF`
and `GGML_OPENCL=OFF` to avoid initializing unused OpenCL kernels. Deploy
matching host and DSP libraries and configure their library search paths.

```sh
export SD_PRELOAD_BACKENDS=htp SD_HTP_COMPUTE=1 SD_HTP_OPS=1 SD_HTP_ATTENTION=1
export SD_HTP_MULTI_SESSION=1 SD_HTP_MULTI_SESSION_COUNT=2
export SD_HTP_TEXT=1 SD_HTP_TEXT_NATIVE=1 SD_HTP_TEXT_NATIVE_AUX=1
export SD_HTP_VAE=1 SD_HTP_VAE_NATIVE=1 SD_HTP_VAE_NATIVE_CONV=1
export SD_HTP_TEXT_FUSION=1 SD_HTP_VAE_FUSION=1
export SD_HTP_PACKED_STAGING=1 SD_HTP_PACKED_STAGING_BUDGET_MB=1024
export SD_HTP_PACKED_STAGING_SLOT_MB=512 SD_HTP_PACKED_PREFETCH=1
export SD_HTP_DIT_PREFETCH=1
export SD_HTP_BLOCK_SPARSE_ATTN=0 SD_HTP_SPARSE_OPENCL=0
export SD_HTP_PACKED_FLUX2=0 SD_HTP_PACKED_ATTENTION=0

sd-cli --backend cpu --params-backend cpu -t 4 --max-vram 0.5 \
  --diffusion-model flux2_klein_4b_htp_f16.gguf \
  --llm qwen3_4b_htp_w16a32.gguf --vae flux2_vae_htp_w16a32.gguf \
  --diffusion-fa --sampling-method euler --steps 4 --cfg-scale 1 \
  -W 256 -H 256 -s 42 -p 'a cat sitting on a wooden table' -o klein.png
```

This repository defaults to a discrete scheduler with the Flux2 flow shift.
MobiInfer defaults to a different Flux2 sigma grid. For a 256px / four-step
comparison with that runtime, explicitly add
`--sigmas 1,0.955390751,0.877134204,0.704111695,0` to use its logged sequence.
These values are specific to that resolution and step count. The three-image
prefetch A/B used this repository's default grid. A subsequent comparison with
the explicit grid above completed on the same device: sd.cpp averaged 34.853 s,
MobiInfer 22.288 s, including PNG saving and excluding initialization. Both
used three distinct prompts, so conditioning cache hits were excluded.
MobiInfer used its automatic CPU thread setting and one HTP session; sd.cpp
used the four-thread/two-session configuration above. These are sequential
device observations without locked clocks or cleared page caches.

The tested DiT double-stream block needs about 684 MiB of packed weights,
so a 512 MiB staging bank cannot execute it. The 1 GiB bank budget above
supports it even when the graph planner cannot split an individual block to
meet `--max-vram 0.5`. Prefetch uses a second bank, with up to 2 GiB of bank
allocations total; compute arenas, graph-cut caches and ordinary parameters
are additional. These budgets are distinct from the graph planner budget.

`SD_HTP_DIT_PREFETCH=1` extends the existing T5/Qwen3 prefetch path to dense
Flux segments. It also requires `SD_HTP_PACKED_PREFETCH=1`; sparse/packed-Flux
experimental graphs are excluded. All segment weights must already be deferred
offline F16 tiles, otherwise execution retains synchronous staging. Set the
DiT switch to zero for an A/B comparison. Qwen3 prefetch is independent of it.

Four CPU threads measured faster than eight for this workload. This is a
device/workload setting, not a global change to the CPU thread count. Single
session execution exhausted this device's mapping budget in validation.

Qwen3 Linear, compact GQA, RMSNorm and residual/gating/SiLU run on HTP; CPU
still performs tokenization, embeddings, RoPE and layout/mask preparation.
VAE uses native HTP convolution, GroupNorm/SiLU and attention. Flux DiT uses
HTP GEMM and Flash Attention but retains CPU modulation, normalization, RoPE
and other unsupported operations. Do not describe this as all-operator HTP.

## Correctness tools

`test-text-encoder qwen3` exercises the real conditioner; see
[text encoder documentation](htp_text_encoders.md). `test-htp-native` checks
compact GQA against a double reference. For an isolated Flux2 VAE comparison:

```sh
test-vae-decode flux2_ae.safetensors cpu 256 1 cpu.f32 - 4 flux2
test-vae-decode flux2_vae_htp_w16a32.gguf htp 256 1 htp.f32 - 4 flux2
```

The final `flux2` argument selects the Flux2 latent layout and batch-normalized
decoder. Outputs are raw F32 before display clipping, using the same synthetic
latent by default. The native HMX arithmetic has FP16 intermediates and is
not bitwise equivalent to CPU reference arithmetic.
