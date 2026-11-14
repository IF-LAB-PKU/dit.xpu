# SD3 text encoders with offline HTP weights

`SD_HTP_TEXT=1` routes SD3 CLIP-L, CLIP-G and T5-XXL Linear layers to the existing
HTP FP16-weight/F32-activation GEMM. Embedding lookups, normalization, activation,
attention and tokenization retain their CPU implementations. No new DSP kernels
or QNN contexts are required. The first validated target is SD3.5 Medium.

## Convert once

Convert each standalone original text encoder with its explicit kind:

```sh
./bin/sd-cli -M convert -m clip_l.safetensors \
  --htp-pack-text clip_l -o clip_l_htp_w16a32.gguf
./bin/sd-cli -M convert -m clip_g.safetensors \
  --htp-pack-text clip_g -o clip_g_htp_w16a32.gguf
./bin/sd-cli -M convert -m t5xxl_fp8_e4m3fn.safetensors \
  --htp-pack-text t5xxl -o t5xxl_htp_w16a32.gguf
```

The converter selects attention projections, feed-forward Linear weights and
CLIP-G's pooled text projection. It writes F16 weights in the existing W16A32
tile layout and records the encoder kind and packed tensor names in GGUF.
Embedding tables, including T5 relative attention bias, remain row-major.
BF16/F32 Linear sources convert to F16; other tensors retain their loaded type.
FP8 safetensors are decoded to F16 by the existing loader, so the FP8 T5 export
is larger on disk. No additional quantization is applied. This dedicated mode
does not apply `--type` or tensor type rules, and cannot be combined with another
packing mode, `--vae` or `--force-f16`.

| Encoder | Packed Linear weights | Packed Linear payload | Full GGUF |
|---|---:|---:|---:|
| CLIP-L | 72 | 162.00 MiB | 234.75 MiB |
| CLIP-G | 193 | 1203.12 MiB | 1325.03 MiB |
| T5-XXL | 168 | 8832.00 MiB | 9083.43 MiB |

The table describes the original files used on the test device. Packed GGUFs
are HTP-only and cannot be repacked or used as CPU row-major weights. To return
to CPU inference, supply the original encoder files and unset `SD_HTP_TEXT`.
Convolution/Linear weight adapters cannot read packed weights.

## Use during inference

Keep the text encoder runtime and parameter backends on CPU, preload HTP, and
replace the text encoder paths in the existing generation command:

```sh
export SD_PRELOAD_BACKENDS=htp
export SD_HTP_COMPUTE=1
export SD_HTP_TEXT=1
./bin/sd-cli --backend cpu --params-backend cpu \
  --clip_l clip_l_htp_w16a32.gguf \
  --clip_g clip_g_htp_w16a32.gguf \
  --t5xxl t5xxl_htp_w16a32.gguf \
  <your diffusion model, VAE and generation arguments>
```

Existing DiT/VAE HTP switches are independent. Full SD3.5 Medium generation on
the current Snapdragon 8 Gen 3 device also uses `SD_HTP_MULTI_SESSION=1` and
`SD_HTP_MULTI_SESSION_COUNT=2`, as documented in [HTP VAE](htp_vae.md).

## Weight memory and graph segments

T5's packed Linear payload exceeds 8 GiB. Text HTP mode therefore enables
segment-scoped staging locally for text runners; it does not enable DiT staging
or change the global environment. Packed Linear parameters have metadata only
until their graph segment executes. The runner reads the required tiles directly
from GGUF into a reusable RPCMEM pool, registers them and invokes HTP GEMM.
There is no runtime weight rearrangement or complete row-major Linear copy.

The default staging pool budget and graph budget are 512 MiB, using the existing
`SD_HTP_PACKED_STAGING_BUDGET_MB` and `SD_HTP_PACKED_STAGING_SLOT_MB` controls.
A configured graph budget is capped at the staging budget. Ordinary embedding,
norm and bias parameters still consume CPU RAM; compute buffers and backend
workspace are additional. Parameter-stage release also frees text staging pools.

T5's relative position bias is explicitly retained across graph segments.
Graph-cut caching copies existing views and their backing storage directly,
including offsets, instead of adding a CONT node to an already closed graph.

## Reproduce checks

```sh
cmake --build <android-build> --target sd-cli test-text-offline-pack \
  test-graph-cut-views test-text-encoder -j4
# Run in the deployed host/DSP library environment:
./bin/test-text-offline-pack
./bin/test-graph-cut-views
./bin/test-text-encoder clip_l clip_l.safetensors cpu 2 clip_l-cpu
./bin/test-text-encoder clip_l clip_l_htp_w16a32.gguf htp 2 clip_l-htp
```

The real-model tool supports `clip_l`, `clip_g` and `t5xxl`. It uses the SD3
tokenizers and production encoder runners, with default prompt
`a cat sitting on a wooden table`; an optional final argument changes the prompt.
It saves finite, unclipped hidden-state and pooled-vector F32 outputs for each
repeat. For CLIP, each repeat includes the separate hidden and pooled forwards
used by normal SD3 conditioning. T5 has one hidden-state forward and a dummy
zero vector output. Encoding time includes graph planning, segment weight reads,
allocation, execution and readback; backend/model initialization and output file
writing are excluded.

## Device observations (2026-11-14)

OnePlus PJD110 / SM8650 (Snapdragon 8 Gen 3, HTP v75), four CPU threads,
two FastRPC sessions, 77 tokens, the default prompt above. Each cell lists two
consecutive encoding calls in one loaded process; these are individual device
observations rather than a statistical throughput benchmark.

| Encoder | CPU encoding | HTP encoding |
|---|---:|---:|
| CLIP-L | 1.52 / 1.50 s | 0.36 / 0.51 s |
| CLIP-G | 12.01 / 11.96 s | 2.73 / 2.28 s |
| T5-XXL | 43.72 / 40.57 s | 8.50 / 7.15 s |

The second outputs were compared against the original-file CPU path. All were
finite. Hidden-state cosine similarities were 0.99999986 (CLIP-L), 0.99997426
(CLIP-G), and 0.99993630 (T5). Their raw RMSE values were 0.003531, 0.004502 and
0.001598, respectively. Pooled-vector cosine similarities were 0.99999875 and
0.99999739 for CLIP-L/G. These results establish numerical proximity for this
prompt; they do not establish bit identity or broad prompt/image quality.

Packing/embedding-preservation, graph-cut view offsets and shared-input lifetime,
existing VAE/DiT packing, F16 conversion, subset loading, packed staging and
graph-cut session-budget regression tests pass on the device. Omitting
`SD_HTP_TEXT=1` for a packed encoder produces an explicit initialization error.

### Full generation

The device completed a full generation with all three packed text encoders,
the existing packed DiT, and the offline packed VAE: 256×256, Euler, four steps,
CFG 4.5, seed 42, prompt `a cat sitting on a wooden table` and empty negative
prompt. Text HTP mode remained local to conditioner runners; DiT staging stayed
disabled. The conditioner stage initially loaded 469,630,976 ordinary parameter
bytes and deferred 10,692,460,544 packed Linear bytes until segment execution.

| Phase | Earlier CPU-text / HTP-VAE run | Packed HTP-text / HTP-VAE run |
|---|---:|---:|
| Text conditioning (positive and negative) | 107.87 s | 20.55 s |
| Sampling | 11.75 s | 12.55 s |
| VAE decoding | 3.58 s | 3.58 s |
| `generate_image` total | 128.05 s | 42.14 s |

The generation function includes weight-stage transitions but excludes initial
process/backend/context setup and PNG writing. This pair of single-run
observations reduced its elapsed time by 67.1% (3.04×). It is not a repeated-run
average. Four steps are an integration check for this non-distilled model.

The generated PNG is finite and saved successfully. Compared with the earlier
CPU-text image, RGB RMSE is 8.78/255 and PSNR is 29.26 dB, with maximum channel
difference 104/255. Text outputs and resulting images are not bit-identical;
these checks do not establish quality over a broad prompt set.

The tested packed models are stored on the device under
`/data/local/tmp/sd-text-htp-20261114/`. Host logs, F32 comparisons, the generation
script and PNG are in the sibling workspace directory
`text-htp-results-20261114/`.
