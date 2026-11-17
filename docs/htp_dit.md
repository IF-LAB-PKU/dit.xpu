# Native DiT auxiliaries on HTP

`SD_HTP_DIT_NATIVE=1` enables an optional path for **dense FLUX.2 Klein**.
It defaults to off and requires the matched host and DSP libraries from this
change, plus the existing HTP compute/preload and offline DiT model setup.
No model reconversion is needed. Sparse/packed-Flux graphs retain their
existing paths. SD3 has separate switches described below; do not assume
that moving every auxiliary operation to HTP is faster.

```sh
# Add to the FLUX.2 Klein HTP environment:
export SD_HTP_DIT_NATIVE=1
```

The path adds four DSP/HVX operations, reusing the native auxiliary message
(opcode 19, 76-byte payload) with modes 9–12:

| Operation | Behavior |
| --- | --- |
| LayerNorm + modulation | Computes `n + n * scale + shift` in one pass after normalization; parameters stay compact per channel/batch. |
| Gated residual | Computes `residual + branch * gate` without materializing the product. |
| Matrix RoPE | Reads the existing interleaved 2×2 position matrices and writes attention layout directly, eliminating repeated tensors and intermediate permutations. Matrix preparation is shared across heads. |
| SwiGLU | Computes `SiLU(gate) * up`; accepts the row-strided MLP slice of a single-stream QKV/MLP projection directly, avoiding a CPU copy. |

Q/K RMSNorm and channel scaling also use the existing native F32 auxiliary
kernel. Graph-cut cached tensors live in RPCMEM so subsequent native nodes
can consume them. Weight staging/session ownership and the HMX GEMM/attention
arithmetic are unchanged. This does not make every DiT operation run on HTP:
layout copies, concatenation, timestep/modulation preparation and graph
scheduling still have CPU work.

New modes require the `htp_ops_dit_aux_version` host capability marker;
older DSP skeletons reject them. Deploy the matched `libggml-htp.so`,
`libstable-diffusion.so`, `libhtp_ops.so` and `libhtp_ops_skel.so` together.
Do not combine a new host library with an old DSP skeleton.

`test-htp-dit` checks complete outputs against independent double references,
including batches, compact broadcast parameters, offset views, channel tails,
shared/per-batch RoPE matrices, saturated gates and row-strided SwiGLU slices.
The operations use F32/QFloat HVX arithmetic and are not bitwise equivalent to
the prior CPU path. The SwiGLU reciprocal iterations round to F32 to preserve
accuracy for large gates; legacy text/VAE epilogues retain their arithmetic.


## Device validation (2026-11-16)

On PJD110 / Snapdragon 8 Gen 3 / HTP v75, dense Klein 4B at 256×256,
Euler 4 steps, CFG1 and 4 CPU threads, an ABBA run (six images per mode)
observed denoising **28.040 → 25.150 s**, generation including PNG saving
**36.247 → 33.441 s**. Both modes used the same offline models, prefetch,
two sessions and 1 GiB staging banks. Initialization is excluded.
Baseline process averages drifted 35.264 → 37.231 s; clocks, temperature and
page cache were not controlled. The observed 10.3% / 7.7% reductions are not
a guarantee of stable speedup or a new strict comparison against MobiInfer.

84 new double-reference cases passed, together with existing native and F16
fusion regression suites. Baseline/native PNGs differ slightly (three-prompt
RGB RMSE 0.842–1.209 on 0–255 scale); each mode's repeated PNGs are identical.
This does not establish quality across models/prompts/resolutions.

One native process failed acquiring session 1 during its third image; the
whole failed process was excluded and a fresh three-image process passed.
The underlying resource was not identified. Keep the path opt-in pending
further resource/stability investigation. Full successful samples, failed
log, precision checks and matched runtimes are retained in the
[local validation report](../../framework-results-20261116/dit-native/README.md).


## Fused single-stream MLP concatenation

With the native dense Klein path enabled, add `SD_HTP_DIT_CONCAT=1` to fuse
single-stream SwiGLU and attention/MLP concatenation. It defaults to off.
The DSP writes directly into the next projection's complete input, removing
the separate activated MLP tensor and CPU CONCAT. Attention values and mode-12
SwiGLU values are preserved bit-for-bit in numerical tests; no weight format
changes. New auxiliary mode 13 requires `htp_ops_dit_concat_version` and matched
host/DSP libraries. The prior native path remains the concat=0 baseline.

An additional 48 full-output cases cover tails, batches, offset/strided inputs,
both gate orders and unequal attention widths. Real Klein resident shape
(768 tokens, attention3072, MLP9216) at 4 CPU threads measured 47.118 → 23.278 ms
for the isolated SwiGLU/concat graph (20 samples/mode), including dispatch and
CPU synchronization, excluding weights/I/O and other model work. This does
not establish whole-image speedup. Full details and resource failures are in
the [local concat fusion report](../../framework-results-20261116/dit-concat/README.md).


Nine images from complete successful processes observed four-step denoising
21.030 → 19.783 s, generation+PNG 29.282 → 27.287 s. Baseline has 3 images and
fused 6; the reverse baseline failed acquiring VTCM, as did its fresh-process
retry during text encoding. Text time also drifted 4.223 → 3.387 s with unchanged
text code. This is **not a completed ABBA or confirmed stable end-to-end gain**.
All three prompt PNGs match the baseline byte-for-byte. The resource problem
recurred after reboot; allocation-policy experiments were reverted. Final
code adds checked default-session switching/error codes, without retries.


## SD3.5 Medium: overlap DiT weight loading

`SD_HTP_SD3_DIT_PREFETCH=1` enables the existing two-bank prefetch pipeline for
**MMDiT** graph-cut execution, including SD3.5 sparse/OpenCL attention. It is
independent of native auxiliaries and opt-in in the library (recognized packed
CLI models enable it). It requires offline packed weights, an active staging
loader, valid graph cuts and `SD_HTP_PACKED_PREFETCH=1`. The original dense
validation below measured loading overlap without changing attention or GEMM
arithmetic; sparse compatibility is described in the direct-CLI section.

The validated configuration on PJD110 / SM8650 / HTP v75 uses one session and
512 MiB per staging bank (512 MiB slots); DiT prefetch uses two banks, peaking
at 1 GiB. This is staging allocation, not total process or DSP memory.
Apply these overrides to the existing SD3 HTP environment and use the same
five offline models already used for text, DiT and VAE:

```sh
export SD_HTP_BLOCK_SPARSE_ATTN=0
export SD_HTP_SPARSE_OPENCL=0
export SD_HTP_SPARSE_OPENCL_RESIDUAL=0
export SD_HTP_BLOCK_SPARSE_RESIDUAL_CACHE=0
export SD_HTP_PACKED_ATTENTION=0
export SD_HTP_MULTI_SESSION=0
export SD_HTP_DEFER_UNMAP=0
export SD_HTP_PACKED_STAGING=1
export SD_HTP_PACKED_STAGING_BUDGET_MB=512
export SD_HTP_PACKED_STAGING_SLOT_MB=512
export SD_HTP_PACKED_PREFETCH=1
export SD_HTP_SD3_DIT_PREFETCH=1
export SD_HTP_SD3_DIT_NATIVE=0
```

At 256×256, Euler 4 steps, CFG4.5, seed42, 4 CPU threads, same cat prompt and
empty negative prompt, a complete ABBA (two images per mode) measured:

| Mean | Synchronous DiT loading | DiT prefetch |
| --- | ---: | ---: |
| Denoising | 22.720 s | 15.405 s |
| Generation + PNG | 36.957 s | 30.326 s |
| Foreground staging over eight MMDiT forwards | 13.792 s | 6.416 s |
| Compute over those forwards | 7.957 s | 7.893 s |

Observed reductions are 32.2% / 17.9%; all four PNGs are byte-identical.
Timing excludes context initialization and ADB transfer, includes lazy model
loading, graph preparation, text and VAE in the complete-image metric. The
baseline drifted 34.949 → 38.966 s; prefetch runs were 29.795 /30.857 s.
No frequency/temperature/page-cache controls, and this is a small sample at
one resolution. Four steps are a performance/integration check for this
non-distilled model, not a validation of image quality or a MobiInfer matchup.

### Staging mapping lifetime fix

The whole-buffer staging branch now registers allocations in `accounted_bufs`,
matching the range/plain mapping paths. Previously it incremented persistent
or active bytes but `release_buffer()` did not decrement them, eventually
rejecting a valid allocation in single-session execution. A 64 MiB-budget
device test alternates persistent/transient buffers: the old host library
reproduces budget exhaustion, while the fixed library completes 80 backings
with over 1 GiB cumulative allocation. This fixes accounting, not the separate
VTCM `HAP_compute_res_acquire` failure seen with two sessions.

### Experimental native SD3 auxiliaries

`SD_HTP_SD3_DIT_NATIVE=1` independently enables dense MMDiT LayerNorm+modulation,
gated residuals, Q/K affine normalization and packed GEMM+bias/GELU using the
existing matched HTP kernels. No new DSP ABI or weight conversion is needed.
Bare LayerNorm is kept visible for modulation fusion; the two MMDiT-X
attention branches each normalize within their fused operation. Ordinary
layout/concatenation/timestep work can still run on CPU.

**This option is not recommended for speed on the tested configuration.**
A complete separate ABBA, four images per mode across cat/dog prompts, measured
denoising 24.498 →35.793 s and generation+PNG 38.282 →49.420 s with prefetch off.
Compute and staging both increased; these are aggregate changes, not proof
that any individual kernel is slower. Outputs repeat exactly within each
mode; baseline/native RGB RMSE is 1.412 /0.851 on 0–255 scale. All 144 DiT
reference checks, native regressions (including 64-wide normalization) and
F16 fusion regressions passed. Keep this flag off when using the measured
prefetch optimization. Full logs, failures, configurations, PNGs, matching
runtimes and source patches are in the
[SD3.5 validation report](../../framework-results-20261116/sd35-dit-native/README.md).


A separate prefetch-enabled 20-step cat generation (CFG4.5, 40 MMDiT forwards)
also completed: denoising72.18s, generation+PNG85.443s. This is a single longer
stability/quality smoke without a baseline, not another speedup comparison.


## Direct CLI defaults and optional launchers

For packed SD3.5 Medium/Klein4B, CPU runtime/params and staged image generation,
`sd-cli` now resolves these defaults from tensor layout/shape metadata before
backend initialization. Dense SD3 uses one session, immediate unmap,
512 MiB/bank, DiT prefetch and native DiT off. Dense Klein uses one session,
1024 MiB/bank and DiT prefetch/native/concat on. Default graph-cut budget is
0.5 GiB unless `--max-vram` was explicit (including zero). User environment
values always win. `SD_HTP_MODEL_DEFAULTS=0` disables CLI auto-configuration;
other models/backends and the C API keep their previous opt-in behavior.

The SD3.5 sparse flags in the [README](../README.md#7--run-inference-and-retrieve-the-image)
change only DiT attention, its packed QKV and residual reconstruction. Sparse
defaults to `SD_HTP_HALO_KV_LOAD=l2fetch` and `SD_HTP_PACKED_ATTENTION=1`,
so those two exports are optional; explicit values still win. With
`SD_HTP_BLOCK_SPARSE_PATCH_HALO=1`, an unset keep ratio uses the existing
`SD_HTP_BLOCK_SPARSE_KEEP_RATIO=0` default; an explicit keep ratio still wins.
OpenCL sparse residual caching now also defaults to lifetime mode `reuse`;
`SD_HTP_SPARSE_OPENCL_LIFETIME_MODE=deferred` or `strict` remains an explicit override.
Both attention modes share the same single-session, immediate-unmap, staging,
text/DiT prefetch and native text/VAE defaults. Sparse/OpenCL no longer disables
text prefetch. Packed QKV F32 auxiliaries are included in typed lookahead
loading rather than disqualifying the complete DiT plan. Deferred GPU work
retains its own Q/K/V and residual arenas; projection weights have completed
HTP consumption before their bank is released. Build with OpenCL support for
this attention route. Explicit environment overrides remain effective. The
default build includes both HTP and OpenCL; dense inference does not request
OpenCL attention/residual execution. Klein sparse remains separately unvalidated.

`bin/htp/run-sd35-medium.sh` and `run-flux2-klein.sh` remain optional convenience
launchers with generation defaults (SD3 Euler20/CFG4.5, Klein Euler4/CFG1,
both256×256/seed42/four CPU threads). They are not required by the direct CLI.
