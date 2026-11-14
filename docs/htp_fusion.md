# Text encoder and VAE HTP fusion

For native attention, convolution and model auxiliaries beyond these epilogues, see [Native HTP models](htp_native_models.md). Its native switches are independent from the legacy fusion-only comparison below.

This opt-in path extends the existing offline W16A32 HMX GEMM with native DSP/HVX epilogues. It reuses the same packed GGUF files; no model conversion is needed.

```sh
# Add to the existing CPU-runtime / packed-HTP command:
export SD_HTP_TEXT=1 SD_HTP_VAE=1
export SD_HTP_TEXT_FUSION=1 SD_HTP_VAE_FUSION=1
```

Both fusion switches default to off and can be controlled independently. Continue using the existing preload/compute settings, CPU runtime, and two-session setup required by SD3.5 Medium. Deploy the updated `libggml-htp.so`, `libstable-diffusion.so`, `libhtp_ops.so`, and v75 `libhtp_ops_skel.so` together. The new shared-message opcode is 16 and its fixed-size payload is 56 bytes. The host capability marker rejects older host libraries; older DSP skeletons reject the unfamiliar opcode. Fused nodes cannot fall back to a CPU kernel that would misinterpret tiled weights.

## Fused operations

- CLIP-L: MLP `fc1` GEMM + bias + QuickGELU.
- CLIP-G: MLP `fc1` GEMM + bias + GELU (tanh approximation).
- T5: MLP `wi_0` GEMM + GELU. The second branch, gating multiplication and `wo` retain their existing execution paths.
- VAE: convolution GEMM + bias + removal of padded channels + channel-major output store. Im2col, normalization, SiLU, residual additions and attention retain their existing execution paths.

Bias-only attention projections were excluded after initial measurements showed a regression: they added bias mapping work without removing a CPU synchronization point. Text fusion is restricted to named text MLP weights so enabling it does not change DiT operations.

The epilogue consumes the HMX tile directly in VTCM before any intermediate GEMM output is written to DDR. GELU/QuickGELU use float vector arithmetic, a stable sigmoid and three reciprocal Newton steps on a denominator bounded to [1,2]. VAE output uses an HVX scatter transpose within VTCM and contiguous spatial writes to DDR. Larger VAE epilogues run through the existing HVX worker pool with private transpose scratch for each worker. Short text epilogues stay on the current DSP thread to avoid worker wakeup overhead. The HMX accumulation and offline weight layout are unchanged.

VAE retains bounded activation chunks and concatenates their spatial outputs. A batch of one can return the channel-major result directly; larger batches need the final batch/channel permutation. Ordinary CPU bias tensors are materialized as small F32 RPCMEM graph buffers. Existing graph-cut weight staging, packed-only consumer validation and session placement also admit the fused nodes.

## Verification

`test-htp-f16-fusion` is an explicit device test, built with `GGML_HTP=ON`. It compares every epilogue with the existing HTP GEMM output, separating fusion error from HMX accumulation error. Cases include M=1/31/33/77/257, no bias, F32/F16 biases, saturated positive/negative activations, RGB tails and parallel output tiles. `test-vae-decode --self-test` compares fused and unfused convolutions with the CPU implementation, including forced chunking, asymmetric spatial sizes, stride, dilation and batch=2.

For paired model benchmarks, `SD_HTP_FUSION_AB=1` makes `test-text-encoder` and `test-vae-decode` alternate fusion off/on between repeats in one process. This reduces variation from model loading and device initialization. Timing includes normal tokenization/graph construction/staging and compute in the text test, and the complete decoder call in the VAE test. Neither includes initial model loading or PNG encoding.

These tests do not establish bitwise equivalence. Model outputs must remain finite and are compared separately before display clipping. Device timings and comparison artifacts are recorded in `../fusion-results-20261114/` outside the source repository.

## SD3.5 Medium / Snapdragon 8 Gen 3 results

| Component | Unfused seconds | Fused seconds | Reduction |
|---|---:|---:|---:|
| CLIP-L | 0.562 | 0.465 | 17.3% |
| CLIP-G | 2.371 | 2.273 | 4.1% |
| T5 | 7.000 | 6.868 | 1.9% |
| VAE 256 | 3.481 | 3.242 | 6.9% |
| VAE 512 | 18.928 | 16.809 | 11.2% |

These are device observations for the fixed 77-token test prompt, not guaranteed speedups. Text medians use two retained off/on pairs after discarding the first pair; VAE medians use three calls per mode. The single 256px/4-step generation A/B measured conditioning 19.56 -> 19.06 s, VAE 3.59 -> 3.26 s, and generate_image 42.47 -> 40.00 s. Sampling also varied from 13.53 to 12.58 s without a graph change, so the entire end-to-end difference cannot be attributed to fusion.

The 512px CPU compute arena shrank by 64 MiB; RPCMEM compute remains 256 MiB. Model outputs were finite, with cosine similarity above 0.999994 relative to the unfused HTP path. The generated PNG pair had PSNR 36.33 dB and max channel difference 36/255. The epilogue test's maximum absolute error against the existing HTP GEMM was below 1e-4. These results do not establish broader image quality equivalence.
