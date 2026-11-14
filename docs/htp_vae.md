# Experimental HTP VAE convolution

This page describes the base hybrid convolution path. [Native HTP models](htp_native_models.md) adds DSP convolution sampling, attention, GroupNorm/SiLU and residual operations, eliminating full CPU im2col.

`SD_HTP_VAE=1` enables a hybrid VAE path, initially validated with SD3.5 Medium.
Standard `Conv2d` layers run as CPU F32 im2col plus the existing HTP FP16-weight
GEMM. GroupNorm, activations, attention and output layout conversion retain their
CPU paths. This does not require exporting a QNN context or changing DSP kernels.

Keep the VAE runtime and parameter backends on CPU; the runner explicitly routes
eligible convolution GEMMs to HTP. Add the following to your existing HTP command:

```sh
export SD_PRELOAD_BACKENDS=htp
export SD_HTP_COMPUTE=1
export SD_HTP_VAE=1
# Run sd-cli with --backend cpu --params-backend cpu (or VAE-specific CPU assignments).
```

`SD_HTP_OPS=1` separately enables the existing diffusion routing.

## Offline VAE conversion and inference

Convert the standalone original VAE once, using the new CLI option:

```sh
./bin/sd-cli -M convert -m sd3.5vae.safetensors \
  --htp-pack-vae -o sd3.5vae_htp_w16a32.gguf
```

The converter pads convolution K/N dimensions to multiples of 32, converts their
weights to F16, and writes the existing W16A32 tile layout. Other tensors retain
their source precision; this mode does not apply `--type` or tensor type rules.
The GGUF stores each packed convolution's original `[KW, KH, IC, OC]` shape.
Do not combine this standalone conversion mode with `--vae`, `--htp-pack-dit`
or `--force-f16`. Conversion requires no connected HTP device.

Use the output directly in the existing generation command:

```sh
SD_PRELOAD_BACKENDS=htp SD_HTP_COMPUTE=1 SD_HTP_VAE=1 \
  ./bin/sd-cli --backend cpu --params-backend cpu \
  --vae sd3.5vae_htp_w16a32.gguf \
  <your existing diffusion model, text encoders and generation arguments>
```

Packed parameters load directly into HTP RPCMEM and are registered for GEMM.
Inference allocates no original row-major convolution weights and performs no
online weight packing. IM2COL receives a shape-only tensor with no weight storage.
The packed GGUF is HTP-only: CPU fallback and convolution weight adapters are
unsupported. To run the original CPU path, use the original VAE file and unset
`SD_HTP_VAE` or set it to `0`.

The original safetensors file still supports the online packing path described
below. Offline packing changes weight preparation and residency; CPU im2col and
the remaining CPU operations are unchanged.

With the original VAE file, weights are packed once per loaded VAE into separate RPCMEM allocations. Original
row-major weights remain available for fallback; packed buffers are registered
only with HTP and cannot be consumed by CPU matmul. Reloading/freeing VAE parameters,
changing the HTP backend or disabling this path clears the cache. Weight-adapter
convolutions use their existing implementation.

Large convolutions keep im2col in CPU RAM and copy spatial chunks of at most
128 MiB into RPCMEM for GEMM. The outputs are concatenated in original spatial and
batch order. This avoids requesting a >1 GiB RPCMEM activation at 512px. It still
materializes the full im2col on CPU, so it is an intermediate solution before a
native HTP convolution implementation; resolutions above 512px are not validated.

## Reproduce the checks

Build with `GGML_HTP=ON` and `SD_BUILD_TESTS=ON`, and deploy `test-vae-decode` with
the same host/DSP libraries as `sd-cli`:

```sh
cmake --build <android-build> --target test-vae-decode test-vae-offline-pack sd-cli -j4
# In the device environment with LD_LIBRARY_PATH / DSP_LIBRARY_PATH configured:
GGML_HTP_TRACE_OPS=1 ./bin/test-vae-decode --self-test
./bin/test-vae-decode /path/to/sd3.5vae.safetensors cpu 256 2 cpu.f32
./bin/test-vae-decode /path/to/sd3.5vae.safetensors htp 256 2 htp.f32
./bin/test-vae-offline-pack
./bin/test-vae-decode /path/to/sd3.5vae_htp_w16a32.gguf htp 256 2 offline.f32
```

The self-test requires a working HTP runtime (it does not silently skip). It checks
padding, stride, dilation, unequal spatial dimensions, batch=2, cache reuse and
cache release. Small chunk limits exercise concatenation and partial final chunks.

The VAE tool uses identical deterministic synthetic latents in VAE space and
writes the final **unclipped** F32 output in GGML `[W,H,C,B]` order. Optional arguments
are a binary F32 latent file (`-` uses synthetic data) and thread count (default 4):

```sh
./bin/test-vae-decode /path/to/sd3.5vae.safetensors htp 512 2 output.f32 latent.f32 4
```

`decode seconds` includes graph construction, first-use weight packing, compute
allocation, execution, result readback and compute-buffer release. It excludes
backend initialization, model loading and file writing. Later repeats reuse only
the packed weight cache when using the original file; offline GGUF runs have no
such cache. Both paths rebuild and allocate the compute graph. These are
VAE decode measurements, not end-to-end text-to-image times.

## Device measurements (2026-11-14)

OnePlus PJD110 / SM8650 (Snapdragon 8 Gen 3, HTP v75), four CPU threads,
SD3.5 Medium `sd3.5vae.safetensors` (BF16 source loaded into existing F16
convolution weights), no tiling. Synthetic latents and identical weights were
used for both paths. The table measures the final 128 MiB chunk implementation.

| Output | CPU decode | HTP first decode | HTP second decode | Decode speedup |
|---|---:|---:|---:|---:|
| 256×256 | 39.15 / 37.52 s | 3.60 s | 3.27 s | 11.48× (second runs) |
| 512×512 | 165.67 s (one run) | 18.39 s | 18.36 s | 9.02× (CPU vs second HTP run) |

All 35 decoder convolutions use the packed-weight cache, which adds 97,001,472
bytes (92.51 MiB) to the ordinary parameter allocation. The recorded compute
allocation sizes are:

| Output | CPU baseline RAM | Hybrid RAM | Hybrid RPCMEM |
|---|---:|---:|---:|
| 256×256 | 416.06 MiB | 744.95 MiB | 156.44 MiB |
| 512×512 | 1664.25 MiB | 3008.24 MiB | 256.00 MiB |

These are allocator buffer sizes, not process peak RSS, and exclude parameter
buffers and backend workspace. The first unchunked prototype was faster at
256px (2.60 s for its second decode), but failed to allocate the 512px RPCMEM
graph. The final chunked path trades some speed for bounded RPCMEM activations.

Unclipped outputs stay finite. At 512px the raw maximum absolute error is
0.02827, RMSE is 0.002877, cosine similarity is 0.999994, and PSNR after conversion
to display range is 58.95 dB. This comparison uses synthetic latents; it is not a
model quality evaluation over prompts.

### Offline packing measurements

The same original BF16 VAE exported to a 167,750,304-byte GGUF (159.98 MiB),
containing 62 packed convolutions across encoder and decoder. The decoder uses
35 of them. Its directly loaded parameters total 99,249,696 bytes (94.65 MiB),
including non-convolution parameters, with `packed_count=0` and `packed_bytes=0`.
This removes the 92.51 MiB online packing cache, with about 0.08 MiB of additional
padding in the parameter allocation. Compute buffers remain as listed above.

| Output | Offline first decode | Offline second decode | Raw difference vs online HTP |
|---|---:|---:|---:|
| 256×256 | 3.51 s | 3.35 s | Bit-identical |
| 512×512 | 18.16 s | 18.12 s | Bit-identical |

These are individual observations, not a repeated-run throughput comparison.
The main offline packing benefit is eliminating runtime repacking and the extra
resident weight copy. Metadata validation, zero-padding/layout round trips,
precision preservation, canonical VAE naming and existing DiT packing
compatibility are covered by `test-vae-offline-pack`. The existing F16 conversion
and subset-loader regression tests also pass on the device.

## End-to-end generation check

The same device completed both real SD3.5 Medium generations: 256×256, Euler,
4 steps, CFG 4.5, seed 42, prompt `a cat sitting on a wooden table`, CLIP-L/G and
T5-XXL FP8. Both runs used HTP dense diffusion attention and two FastRPC sessions
(`SD_HTP_MULTI_SESSION=1`, `SD_HTP_MULTI_SESSION_COUNT=2`); only `SD_HTP_VAE` differed.

| Phase | CPU VAE run | HTP VAE run |
|---|---:|---:|
| Text conditioning | 103.90 s | 113.70 s |
| Sampling | 13.07 s | 12.13 s |
| VAE decoding | 40.56 s | 3.71 s |
| `generate_image` total | 162.90 s | 135.60 s |

This one A/B pair reduced generation time by 27.30 s (16.8%). The generation
function includes text encoding, sampling, VAE decoding and weight-stage
transitions; it excludes initial process/backend/context setup and PNG writing.
Text encoding varied by 9.8 s between runs and remains the main bottleneck.
These numbers are an initial device observation, not a repeated-run average.

Both PNGs were written successfully. Their pixel PSNR is 58.98 dB and the maximum
8-bit channel difference is 2/255. Four sampling steps serve as an integration
and timing check, not a quality assessment for the non-distilled SD3.5 Medium.

With the offline packed VAE, the same generation command also completed and
registered all 35 decoder convolutions directly. Text conditioning took 107.87 s,
sampling 11.75 s, VAE decoding 3.58 s, and `generate_image` 128.05 s. Its PNG is
pixel-identical to the online HTP VAE run above. The total includes stage
transitions but excludes initial setup and PNG writing; timing differences
between these single runs do not establish an offline packing throughput gain.

On this 8 Gen 3 device, a single-session full-generation baseline failed in the
existing DiT FastRPC mapping path. The documented two-session option completed
the A/B pair. Enabling staging with `--max-vram 1` separately hit an existing
T5 graph-cut cache error, so staging was disabled for the successful runs. Neither
workaround changes the VAE implementation; no DSP-library changes were required.

Optional native DSP epilogue fusion is described in [htp_fusion.md](htp_fusion.md).
