# FLUX.2 Klein 4B on HTP

This path reuses the SD3 offline W16A32 Linear/convolution kernels,
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
With CPU runtime/parameter backends and the current three offline packages,
`sd-cli` automatically selects one session, immediate unmap,1 GiB per bank,
native text/VAE, dense DiT prefetch/native/concat, and a0.5 GiB graph-cut budget.
No tuning environment exports or launcher are needed:

```sh
sd-cli --backend cpu --params-backend cpu -t 4 \
  --diffusion-model flux2_klein_4b_htp_f16.gguf \
  --llm qwen3_4b_htp_w16a32.gguf --vae flux2_vae_htp_w16a32.gguf \
  --diffusion-fa --sampling-method euler --steps 4 --cfg-scale 1 \
  -W 256 -H 256 -s 42 -p 'a cat sitting on a wooden table' -o klein.png
```

Library/DSP search paths are still required. Selection uses tensor metadata,
not filenames. Explicit environment exports and `--max-vram` override defaults;
`SD_HTP_MODEL_DEFAULTS=0` disables auto-configuration. C API callers keep their
existing opt-in settings. Optional `bin/htp/run-flux2-klein.sh` remains available
for convenience; it writes relative outputs under runtime `bin/` and accepts
appended CLI overrides. The equivalent manual environment (also for the C API)
is:

```sh
export SD_PRELOAD_BACKENDS=htp SD_HTP_COMPUTE=1 SD_HTP_OPS=1 SD_HTP_ATTENTION=1
export SD_HTP_MULTI_SESSION=0 SD_HTP_DEFER_UNMAP=0
export SD_HTP_TEXT=1 SD_HTP_TEXT_NATIVE=1 SD_HTP_TEXT_NATIVE_AUX=1
export SD_HTP_VAE=1 SD_HTP_VAE_NATIVE=1 SD_HTP_VAE_NATIVE_CONV=1
export SD_HTP_TEXT_FUSION=1 SD_HTP_VAE_FUSION=1
export SD_HTP_PACKED_STAGING=1 SD_HTP_PACKED_STAGING_BUDGET_MB=1024
export SD_HTP_PACKED_STAGING_SLOT_MB=512 SD_HTP_PACKED_PREFETCH=1
export SD_HTP_DIT_PREFETCH=1 SD_HTP_DIT_NATIVE=1 SD_HTP_DIT_CONCAT=1
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
used the earlier four-thread/two-session configuration, before the current launcher defaults. These are sequential
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

The default launcher enables dense Klein DiT auxiliary kernels with
`SD_HTP_DIT_NATIVE=1` and the matching host/DSP libraries. This fuses
LayerNorm/modulation, gated residuals, matrix RoPE and SwiGLU, and enables
native Q/K RMSNorm. It reuses the existing offline DiT model. See
[native DiT auxiliaries](htp_dit.md) for coverage and validation.

Four CPU threads measured faster than eight for this workload. This is a
device/workload setting, not a global change to the CPU thread count. Earlier
single-session tests failed before the staging mapping-accounting fix; current
single-session validation below succeeds with the new libraries.

Qwen3 Linear, compact GQA, RMSNorm and residual/gating/SiLU run on HTP; CPU
still performs tokenization, embeddings, RoPE and layout/mask preparation.
VAE uses native HTP convolution, GroupNorm/SiLU and attention. Flux DiT uses
HTP GEMM and Flash Attention; without `SD_HTP_DIT_NATIVE`, modulation,
normalization and RoPE retain their CPU paths. Layout and other operations
still have CPU work with the switch enabled. Do not describe this as
all-operator HTP.

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


The default launcher also enables native dense Klein single-stream SwiGLU +
concatenation fusion with `SD_HTP_DIT_CONCAT=1`. This writes the combined attention/MLP projection input
on DSP and avoids CPU CONCAT. It requires matching mode-13 host/DSP libraries;
see [native DiT validation and limitations](htp_dit.md).

## Single-session validation (2026-11-17)

With the latest staging mapping-accounting fix, dense Klein4B also completes
single-session generation. Override `SD_HTP_MULTI_SESSION=0` and
`SD_HTP_DEFER_UNMAP=0` in the existing native+concat/prefetch environment.
Keep `SD_HTP_PACKED_STAGING_BUDGET_MB=1024` and slot512 MiB: the current Klein
block sizing does not fit the512 MiB bank used in the SD3.5 experiment.
Two prefetch banks peak at2 GiB of staging allocation, excluding other memory.

A complete cat/dog/fox process at256×256, Euler4, CFG1, seed42, matched sigmas
and4 CPU threads succeeded. Mean denoising17.740s, generation+PNG25.131s;
all three PNGs are byte-identical to prior dual-session native+concat outputs.
This is a functional check, not a matched timing comparison against two
sessions or proof for other FLUX.2 variants/resolutions. No new weight conversion.
See the [retained validation report](../../framework-results-20261117/klein-single-session/README.md).
