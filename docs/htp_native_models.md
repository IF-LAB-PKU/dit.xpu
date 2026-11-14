# Native HTP text encoders and VAE

This opt-in path runs CLIP-L/CLIP-G/T5 attention and SD3.5 Medium VAE convolution on the DSP/HMX backend. It extends the hybrid offline-packed path; it does not export a QNN graph. CPU remains the graph scheduler and weight-streaming host. Deploy matching sd.cpp and htp-ops-lib-dit libraries.

## Enable

Add to the existing packed-model command (CPU runtime and parameter backend):

```sh
export SD_PRELOAD_BACKENDS=htp SD_HTP_COMPUTE=1
export SD_HTP_TEXT=1 SD_HTP_VAE=1
export SD_HTP_TEXT_NATIVE=1 SD_HTP_VAE_NATIVE=1
export SD_HTP_TEXT_FUSION=1 SD_HTP_VAE_FUSION=1
# Current SD3.5 Medium device configuration:
export SD_HTP_MULTI_SESSION=1 SD_HTP_MULTI_SESSION_COUNT=2
```

`SD_HTP_TEXT_NATIVE` enables attention, LayerNorm/RMSNorm, residual additions and T5 gating multiplication, plus packed Linear bias/MLP epilogues. To measure attention and Linear epilogues with CPU normalization/residual/gating, set `SD_HTP_TEXT_NATIVE_AUX=0`. The auxiliary switch defaults to enabled when text native mode is enabled. Embeddings, tokenization, CLIP position addition, T5 bias lookup/scaling and parameter-copy nodes remain on CPU. Reshape/view nodes only describe layouts.

`SD_HTP_VAE_NATIVE` enables native convolution, attention, GroupNorm+SiLU, residual addition and attention-layout transpose. `SD_HTP_VAE_NATIVE_CONV` can override convolution independently. Setting it alone retains CPU normalization/attention. Two-times nearest upsampling followed by a native convolution is absorbed into its input sampler. Bias is fused into the convolution output store.

All native switches default to off. Unsupported packed operations fail explicitly rather than executing tiled weights as ordinary CPU weights. Weight adapters and unsupported convolution variants retain their existing implementation. This implementation targets standard Conv2d, not grouped/depthwise/transposed convolutions. It has been tested on SD3.5 Medium decode; broader models and VAE encoding are not yet validated.

## Offline models

Existing text GGUF packages are reusable. Re-export the original VAE to include the four decoder attention projections (and four encoder projections) as equivalent packed 1x1 convolutions:

```sh
./bin/sd-cli -M convert -m sd3.5vae.safetensors \
  --htp-pack-vae -o sd3.5vae_htp_w16a32.gguf
```

The tested full package contains 70 packed tensors, including 39 used by the decoder, and is 167,751,008 bytes. It uses the existing original-convolution-shape metadata. Old 62-tensor packages remain loadable but leave four decoder projections on the old Linear path. No runtime weight rearrangement is required. Decoder parameter payload is 99,249,696 bytes; HTP compute buffers and temporary DSP allocations are separate.

## Kernels and execution

Shared-message ABI opcodes 17/18/19 implement convolution, attention and model auxiliaries; opcode 16 implements FP16 GEMM epilogues. Both wire headers assert matching payload sizes (124/80/76/56 bytes). Host capability markers reject older libraries. DSP errors propagate to the backend.

Convolution samples F32 NCHW input into bounded VTCM tiles using HVX, converts each tile to HMX FP16 format and multiplies it by offline-packed weights. Bias and NCHW output storage run on DSP. CPU/full-DDR im2col is eliminated. VTCM rows are skewed to avoid scatter bank conflicts and HMX tile addresses retain 2 KiB alignment. This is implicit tiled convolution implemented with HMX GEMM, not a separate QNN convolution operator. Gathering currently dominates many large layers.

Attention converts K/V on DSP, then uses streaming HMX QK and PV with vector softmax. It supports CLIP causal masks and T5 per-head additive relative-position bias, arbitrary token tails, batch input and fully masked rows (zero output). CLIP/VAE use 1/sqrt(head_dim), T5 uses scale 1. Existing DiT attention keeps its original semantics. The HMX intermediate/accumulation path uses FP16, so results are not bitwise equivalent to CPU FP32 attention. The wrapper allocates temporary DSP K/V/mask storage per call; these allocations are not included in runner arena totals.

Auxiliary kernels implement epsilon-aware LayerNorm/RMSNorm/GroupNorm, affine transforms, residual/gating operations, SiLU/GELU/QuickGELU and exact F32 transpose. GroupNorm followed by SiLU is one node. Small auxiliary tensors avoid worker-pool wakeups. Packed Linear and convolution consumer validation and graph-cut cache placement support the new nodes; native tags survive graph renaming.

## Device verification and measurements

Device: OnePlus PJD110, SM8650 / Snapdragon 8 Gen 3, HTP v75, 24 GB; SDK 6.7 and Hexagon Tools 21. This is not an 8 Elite result.

`test-htp-native` compares attention against an independent double-precision softmax reference, including lengths 1/31/77/128, head dimensions 64/512, batch 2, causal/per-head additive masks and empty rows. It also checks norm/elementwise results and exact transpose. `test-vae-decode --self-test` covers borders, channel tails, large spatial/channel sizes, stride/dilation/batch and absorbed upsampling against CPU convolution. `test-vae-offline-pack` checks projection conversion and metadata. Graph-cut, packed-staging, session-budget, text-packing and message-layout regressions are included in the device report.

The comparison baseline already enables the previous GEMM epilogue fusion, while retaining CPU attention/norm/im2col. Text results below are medians of calls 1–3 after call 0; VAE uses all three calls. Each mode runs sequentially in a separate process, so thermal/frequency and weight-I/O variation remain.

| Component | Previous hybrid (s) | Full native HTP (s) |
|---|---:|---:|
| CLIP-L | 0.522 | 0.686 |
| CLIP-G | 2.456 | 2.908 |
| T5-XXL | 6.907 | 7.182 |
| VAE 256x256 | 3.294 | 3.600 |
| VAE 512x512 | 17.138 | 14.039 |

The additional `SD_HTP_TEXT_NATIVE_AUX=0` diagnostic measured 0.649 / 2.587 / 7.192 s for CLIP-L / CLIP-G / T5 (same warm-median rule, sequential runs). It reduced the CLIP-G regression but did not establish a general text speedup; the switch is for independent routing/profiling rather than a guaranteed faster preset.

512px VAE decode improved about 18%; 256px and these text measurements regressed. Do not enable every path solely on the assumption that DSP execution is faster. Short operators, staging and CPU/HTP synchronization can outweigh kernel savings.

The VAE runner compute arenas changed from 728.95 MiB CPU + 156.44 MiB RPCMEM to 0 + 192 MiB at 256px, and from 2944.24 + 256 MiB to 0 + 768 MiB at 512px. These are compute arenas, not process peak RSS or total model memory. Small parameter copies still execute on CPU.

A single paired 256px Euler/4-step generation (CFG 4.5, seed 42, prompt `a cat sitting on a wooden table`, empty negative prompt) measured:

| Stage | Previous hybrid (s) | Full native HTP (s) |
|---|---:|---:|
| Conditioning | 19.01 | 23.08 |
| Sampling (unchanged graph) | 13.16 | 11.92 |
| VAE | 3.25 | 3.65 |
| generate_image | 40.31 | 44.07 |

`generate_image` includes stage weight loading and generation, excludes initial context/backend setup and PNG encoding. It is not just denoising. The changed sampling time reflects run variation; no speedup is attributed to it. Full native mode does not establish an end-to-end improvement at 256px.

All model outputs were finite. Native versus previous-hybrid output cosine similarity was 0.99999994 (CLIP-L), 0.99999301 (CLIP-G), 0.99997852 (T5), 0.99999203 (VAE256) and 0.99999587 (VAE512). The fixed generated PNG pair had PSNR 31.82 dB and max channel difference 84/255. This is a correctness smoke test, not a broad image-quality evaluation.

Device logs, scripts, output comparisons and binary hashes were recorded outside this repository in the workspace `../native-results-20261114/`; they are not included in Git. The measurements above retain the observed results. `SD_HTP_MODEL_COVERAGE=1` prints CPU/HTP graph categories; `SD_HTP_MODEL_PROFILE=1` prints native kernel timings, including convolution gather/weight/HMX/store breakdowns.


## Organized device deployment

The current phone stores all five packed models in `/data/local/tmp/hjy/model/sd3.5-medium`: `sd3.5_medium_htp_w16a32.gguf`, `clip_l_htp_w16a32.gguf`, `clip_g_htp_w16a32.gguf`, `t5xxl_htp_w16a32.gguf` and the updated 70-tensor `sd3.5vae_htp_w16a32.gguf`. Original safetensors remain available. The VAE was previously named `sd3.5vae_htp_native.gguf`; only its filename changed.

Matched binaries and libraries are in `/data/local/tmp/hjy/bin/sd3.5-medium-htp`. Its `generate.sh` selects these model paths and explicitly enables full native mode. The six temporary `sd-*-20261114` validation directories have been removed; historical commands referring to them require paths to be adjusted.

```sh
adb shell sh /data/local/tmp/hjy/bin/sd3.5-medium-htp/generate.sh
```
