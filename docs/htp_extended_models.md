# SD3.5 Large and Z-Image on HTP

These models reuse offline W16A32 projections, native text attention and VAE
convolution, segment-scoped staging and immediate unmap with one HTP session.
SD3.5 Large shares the packed CLIP-L/CLIP-G/T5 and SD3 VAE used by Medium.
Z-Image Base/Turbo share their own Qwen3-4B encoder and Flux-style VAE. Do not
substitute the Klein text package just because both encoders are called Qwen3.

## Prepare models

Keep the existing packed Large DiT. Convert Z-Image's original DiTs, standalone
text encoder and matching VAE with the current `sd-cli` once:

```sh
sd-cli -M convert -m z_image_f16.safetensors --htp-pack-dit \
    -o z_image_htp_w16a32.gguf
sd-cli -M convert -m z_image_turbo_f16.safetensors --htp-pack-dit \
    -o z_image_turbo_htp_w16a32.gguf
sd-cli -M convert -m qwen3_4b_zimage.gguf --htp-pack-text qwen3 \
    -o qwen3_4b_zimage_htp_w16a32.gguf
sd-cli -M convert -m ae.safetensors --htp-pack-vae \
    -o z_image_vae_htp_w16a32.gguf
```

The tested encoder source is quantized. The converter dequantizes selected
Linear weights to F16 before packing; this preserves the quantized reference,
does not restore original full precision and increases file size. Its 252
packed projections occupy 6930 MiB; embeddings/norms retain source precision.
Retain the original encoder for CPU reference.

Use the `split_files/vae/ae.safetensors` from the Comfy-Org Z-Image or
Z-Image-Turbo package (both publish the same file). The SD3 VAE has compatible
tensor shapes but different weights and is not a substitute; using it produces
invalid images even though inference completes. Keep its packed package
separate from the SD3 and Flux2 VAEs.

Z-Image requires the updated runtime even with an existing offline DiT: its
cached timestep must remain in RPCMEM because packed AdaLN projections read
it directly through views. A CPU cache leaf previously caused those projections
to fall back and interpret packed bytes as row-major weights. The hybrid runtime
now rejects that fallback with an error. This guard does not prevent normal
CPU fallback for weights that retain a row-major copy.

## Run

With CPU runtime/parameter backends, staged loading and all components packed,
the CLI selects resource defaults from model tensor shapes and layout metadata.
Large uses 2432-wide MMDiT projections; Z-Image uses its 64-to-3840 input
projection. Base and Turbo have the same shapes, so automatic selection never
changes their CFG, sampler or step count. Explicit environment/CLI values win;
`SD_HTP_MODEL_DEFAULTS=0` disables selection. The C API remains opt-in.

Set runtime library/DSP search paths as in the main README. Under the standard
phone layout the optional launchers provide complete commands:

```sh
sh /data/local/tmp/hjy/bin/htp/run-sd35-large.sh
sh /data/local/tmp/hjy/bin/htp/run-z-image.sh
sh /data/local/tmp/hjy/bin/htp/run-z-image-turbo.sh
```

The launchers use 256x256, Euler, seed42 and four CPU threads, with respectively
20/30/8 steps and CFG4.5/5/1. Appended CLI flags override these examples. Large
reads shared encoders/VAE from `model/sd3.5-medium`; Z-Image reads its encoder
and DiTs/VAE from `model/z-image`. `SD_HTP_SD3_SHARED_DIR`
(Large), `SD_HTP_VAE_PATH` (Z-Image), `SD_HTP_MODEL_DIR`, `SD_HTP_BIN_DIR` and
`SD_HTP_ROOT` override these locations.

The common policy uses one session, immediate unmap, 1024 MiB capacity per bank,
512 MiB staging slots, 0.5 GiB graph-cut planning, packed/text prefetch and
native text/VAE kernels. Two-bank capacity is a limit, not the measured resident
or mapped memory; allocations depend on segment sizes.

Large reuses `SD_HTP_SD3_DIT_PREFETCH=1`; native MMDiT auxiliaries stay off.
Z-Image adds `SD_HTP_Z_IMAGE_DIT_PREFETCH=1` for dense graph-cut execution,
overlapping the next segment's private-bank weight read with current compute.
Binding/validation happens on the main thread after join, with cancellation
joined before freeing either bank. The existing typed eligibility checks,
fallback and error propagation apply. This changes execution scheduling, not
GEMM, attention math or sampling. Z-Image sparse/OpenCL/packed-Z requests
retain synchronous DiT staging; this extension does not validate those routes.
The refiner-to-joint concatenation also has its own cut group, preventing a
reused prelude group from pulling both refiner stacks into one oversized segment.

For C API callers set the common switches documented in
[Klein HTP usage](htp_flux2_klein.md#runtime), choose the prefetch switch above,
and keep `SD_HTP_DIT_NATIVE=0`/`SD_HTP_DIT_CONCAT=0` for these architectures.
Do not copy Klein's native-DiT switches as a claimed optimization for Z-Image.

Phone verification and timing boundaries are recorded in the architecture
review and the accompanying three-model measurement report.
