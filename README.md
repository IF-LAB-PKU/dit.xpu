# sd.cpp-npu ⚡

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Platform: Android](https://img.shields.io/badge/Platform-Android-3DDC84?logo=android&logoColor=white)](#quick-start)
[![Backends: CPU / OpenCL / HTP](https://img.shields.io/badge/Backends-CPU%20%7C%20OpenCL%20%7C%20HTP-7C3AED)](#quick-start)

A C/C++ inference runtime for DiT image generation on Qualcomm Snapdragon devices. Built on stable-diffusion.cpp, htp-ops-lib, and ggml, it supports CPU, OpenCL, and HTP backends, including an HTP sparse attention inference path.

[🧩 Supported Models](#supported-models) · [🚀 Quick Start](#quick-start) · [🐛 Issues](#issues) · [🙏 Foundational projects](#foundational-projects-this-build-depends-on)

<a id="supported-models"></a>

## 🧩 Supported Models

| Model | Guide |
| --- | --- |
| SD3.5 Medium | [SD3 / SD3.5](docs/sd3.md) |
| SD3.5 Large | [SD3 / SD3.5](docs/sd3.md) |
| FLUX.2 Klein 4B | [FLUX.2](docs/flux2.md) · [HTP offline models and runtime](docs/htp_flux2_klein.md) |
| Z-Image Turbo | [Z-Image](docs/z_image.md) |
| Z-Image Base | [Z-Image](docs/z_image.md) |

<a id="quick-start"></a>

## 🚀 Quick Start

This example uses **SD3.5 Medium** to build in Docker and run **256×256 HTP/OpenCL sparse inference with 20 sampling steps** on an Android phone. The DiT, VAE and all three text encoders use offline-packed weights; native HTP kernels handle text attention and VAE convolution as well.

### 1. 🛠️ Prerequisites

- Install Docker, Git, and ADB on the host, and enable USB debugging on the phone.
- Prepare the Linux version of [Hexagon SDK 6.7.0.0](https://www.qualcomm.com/developer/software/hexagon-sdk), including Tools 21.0.04, NDK r27d, and CMake 3.28.3.
- Use a phone with HTP and OpenCL support (OpenCL is used by the sparse route below). Allow about 40 GB of free storage to keep both the original and converted model files. The build below targets DSP v75.

Run the following commands in the same terminal on the host.

### 2. 📦 Clone the Repositories and Retrieve the OpenCL Library

Clone the current `main` branches of both repositories and build the matching HTP host/DSP libraries together with sd.cpp. Native DiT/concat kernels require HTP commit `e2663f6` or later; the earlier `3315a61` revision lacks these kernels.

```bash
mkdir sdcpp-work
cd sdcpp-work
git clone https://github.com/destined555/sd.cpp-npu.git sdcpp
git clone https://github.com/destined555/htp-ops-lib-sd.cpp.git htp-ops-lib
git clone --depth 1 --branch v2024.10.24 \
  https://github.com/KhronosGroup/OpenCL-Headers.git OpenCL-Headers

adb devices -l
adb pull /vendor/lib64/libOpenCL.so ./libOpenCL.so
```

### 3. 🐳 Set Up the Container, Dependencies, and SDK

Set `SDK_DIR` to the absolute path of your Linux SDK installation.

```bash
SDK_DIR=/absolute/path/to/Hexagon_SDK/6.7.0.0

docker run -d --name sdcpp-build --platform linux/amd64 \
  ubuntu:22.04 sleep infinity

docker exec sdcpp-build bash -c '
  set -e
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ca-certificates git build-essential ninja-build cmake python3 \
    python-is-python3 curl unzip pkg-config libglib2.0-0 libncurses5 libtinfo5 libxml2
  mkdir -p /work /opt/Hexagon_SDK
'

docker cp "$SDK_DIR" sdcpp-build:/opt/Hexagon_SDK/6.7.0.0
docker cp sdcpp sdcpp-build:/work/sdcpp
docker cp htp-ops-lib sdcpp-build:/work/htp-ops-lib
docker cp OpenCL-Headers sdcpp-build:/work/OpenCL-Headers
docker cp libOpenCL.so sdcpp-build:/work/libOpenCL.so
docker exec sdcpp-build bash -c '
  git config --global --add safe.directory /work/sdcpp
  git config --global --add safe.directory /work/htp-ops-lib
'
```

### 4. 🔨 Build and Collect Runtime Files

Use [script/build.sh](script/build.sh) to build the Android host library, DSP v75 library, and `sd-cli` with CPU / HTP / OpenCL support. Dense is the default inference route; OpenCL is used when sparse flags are explicitly enabled. Runtime files and optional convenience launchers are collected in `output/bin/`. `SD_HTP_BUILD_OPENCL=OFF` builds a smaller CPU/HTP-only package, which cannot run the OpenCL sparse route.

```bash
docker exec sdcpp-build bash /work/sdcpp/script/build.sh
docker cp sdcpp-build:/work/output ./output
```

### 5. 📥 Download Model Files

Download the following five files from Hugging Face and save them in `models/sd3.5-medium/` on the host.

| Component | Download | Local Filename |
| --- | --- | --- |
| DiT | [sd3.5_medium-F16.gguf](https://huggingface.co/city96/stable-diffusion-3.5-medium-gguf/blob/main/sd3.5_medium-F16.gguf) | `sd3.5_medium-F16.gguf` |
| VAE | [vae/diffusion_pytorch_model.safetensors](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium/blob/main/vae/diffusion_pytorch_model.safetensors) | `sd3.5vae.safetensors` |
| CLIP-L | [text_encoders/clip_l.safetensors](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium/blob/main/text_encoders/clip_l.safetensors) | `clip_l.safetensors` |
| CLIP-G | [text_encoders/clip_g.safetensors](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium/blob/main/text_encoders/clip_g.safetensors) | `clip_g.safetensors` |
| T5-XXL | [text_encoders/t5xxl_fp8_e4m3fn.safetensors](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium/blob/main/text_encoders/t5xxl_fp8_e4m3fn.safetensors) | `t5xxl_fp8_e4m3fn.safetensors` |

### 6. 📱 Deploy to the Phone and Convert to HTP GGUF

Convert the original files once into the following HTP packages. The VAE conversion includes attention projections as packed 1×1 convolutions; text conversion packs Linear weights while retaining embedding tables in row-major layout.

| Component | Converted Filename |
| --- | --- |
| DiT | `sd3.5_medium_htp_w16a32.gguf` |
| VAE | `sd3.5vae_htp_w16a32.gguf` |
| CLIP-L | `clip_l_htp_w16a32.gguf` |
| CLIP-G | `clip_g_htp_w16a32.gguf` |
| T5-XXL | `t5xxl_htp_w16a32.gguf` |

```bash
adb shell 'mkdir -p /data/local/tmp/sdcpp/model'
adb push output/bin /data/local/tmp/sdcpp/
adb push models/sd3.5-medium /data/local/tmp/sdcpp/model/
adb shell 'chmod +x /data/local/tmp/sdcpp/bin/sd-cli'

adb shell '
  set -e
  cd /data/local/tmp/sdcpp || exit 1
  export LD_LIBRARY_PATH="$PWD/bin"
  ./bin/sd-cli -M convert \
    -m model/sd3.5-medium/sd3.5_medium-F16.gguf \
    --type f16 --convert-name --htp-pack-dit \
    -o model/sd3.5-medium/sd3.5_medium_htp_w16a32.gguf
  ./bin/sd-cli -M convert \
    -m model/sd3.5-medium/sd3.5vae.safetensors \
    --htp-pack-vae -o model/sd3.5-medium/sd3.5vae_htp_w16a32.gguf
  ./bin/sd-cli -M convert \
    -m model/sd3.5-medium/clip_l.safetensors \
    --htp-pack-text clip_l -o model/sd3.5-medium/clip_l_htp_w16a32.gguf
  ./bin/sd-cli -M convert \
    -m model/sd3.5-medium/clip_g.safetensors \
    --htp-pack-text clip_g -o model/sd3.5-medium/clip_g_htp_w16a32.gguf
  ./bin/sd-cli -M convert \
    -m model/sd3.5-medium/t5xxl_fp8_e4m3fn.safetensors \
    --htp-pack-text t5xxl -o model/sd3.5-medium/t5xxl_htp_w16a32.gguf
'
```

T5's FP8 source is decoded to F16 for HTP GEMM, so its converted file is about 8.87 GiB. These packages are HTP-only; keep the original files for CPU inference. Re-export an older VAE package to include the attention projections. Deploy the matching host/DSP libraries built above; the old `9e12214` DSP revision lacks the native text/VAE kernels.

### 7. 🎨 Run Inference and Retrieve the Image
Generate a 256×256 image with 20 sampling steps, the prompt cat, and seed 42. Text Linear/attention/norm and VAE convolution/attention/norm use HTP; the DiT retains the HTP/OpenCL sparse path. CPU remains the graph scheduler and handles unsupported operations. The CPU backend assignments below are required by the hybrid HTP runner.


```bash
adb shell '
  cd /data/local/tmp/sdcpp || exit 1
  export LD_LIBRARY_PATH="$PWD/bin:/vendor/lib64:/system/lib64"
  export DSP_LIBRARY_PATH="$PWD/bin;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/vendor/dsp;/dsp"
  export ADSP_LIBRARY_PATH="$DSP_LIBRARY_PATH"

  export SD_HTP_BLOCK_SPARSE_ATTN=1
  export SD_HTP_BLOCK_SPARSE_PATCH_HALO=1
  export SD_HTP_BLOCK_SPARSE_WARMUP_RATIO=0.2
  export SD_HTP_BLOCK_SPARSE_DENSE_TAIL_RATIO=0
  export SD_HTP_BLOCK_SPARSE_DENSE_EVERY_N_SPARSE_STEPS=5
  export SD_HTP_BLOCK_SPARSE_RESIDUAL_CACHE=1
  export SD_HTP_BLOCK_SPARSE_RESIDUAL_EXACT_RHO=1
  export SD_HTP_SPARSE_OPENCL=1
  export SD_HTP_SPARSE_OPENCL_RESIDUAL=1

  ./bin/sd-cli -v -t 4 --backend cpu --params-backend cpu \
    --clip_l model/sd3.5-medium/clip_l_htp_w16a32.gguf \
    --clip_g model/sd3.5-medium/clip_g_htp_w16a32.gguf \
    --t5xxl model/sd3.5-medium/t5xxl_htp_w16a32.gguf \
    --diffusion-model model/sd3.5-medium/sd3.5_medium_htp_w16a32.gguf \
    --vae model/sd3.5-medium/sd3.5vae_htp_w16a32.gguf \
    --clip-on-cpu --diffusion-fa -p cat --steps 20 --cfg-scale 4.5 \
    --guidance 3.5 --sampling-method euler -W 256 -H 256 -s 42 \
    -o sdcpp-sparse-256.png > sdcpp-sparse-256.log 2>&1
  status=$?
  echo "EXIT:$status" >> sdcpp-sparse-256.log
  tail -n 8 sdcpp-sparse-256.log
  exit "$status"
'

mkdir -p results
adb pull /data/local/tmp/sdcpp/sdcpp-sparse-256.png results/
adb pull /data/local/tmp/sdcpp/sdcpp-sparse-256.log results/
```

<a id="issues"></a>

## 🐛 Issues

### FastRPC Mapping Limits: Staging and Optional Multi Session

Snapdragon 8 Gen 3 and earlier devices have limited DSP virtual address space; see [qualcomm/fastrpc#137](https://github.com/qualcomm/fastrpc/issues/137). Current packed dense CLI defaults use bounded staging, immediate unmap and the staging byte-accounting fix, so single-session execution is the default for the tested SD3.5 Medium/Klein4B configurations.

Keep each graph-cut segment within its staging bank budget. Klein needs a 1 GiB bank because its existing double-stream block contains about 684 MiB of packed weights. `--max-vram 0.5` controls graph-cut planning; it cannot split every individual block and is distinct from the pool allocation budget.

Multi session remains an optional experiment, explicitly enabled with `SD_HTP_MULTI_SESSION=1` and `SD_HTP_MULTI_SESSION_COUNT=2`. It is not required by the current dense defaults and does not imply parallel execution. Two-session switching has shown VTCM acquire errors (`0x80000402`) on the current phone; a later single-session initialization also returned this code after repeated runs and required a reboot. That resource issue is separate from the fixed host mapping accounting, and single session is not a guarantee against it. Larger resolutions and other models require separate validation. Sparse/OpenCL execution remains an explicit alternative rather than the default.

For other problems, open a [GitHub issue](https://github.com/destined555/sd.cpp-npu/issues) with your device and SDK versions, repository commit, model filenames, full command, and logs.

<a id="foundational-projects-this-build-depends-on"></a>

## 🙏 Foundational projects this build depends on

This project builds on the following projects and tools:

- [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp): Upstream C/C++ diffusion model inference implementation.
- [ggml](https://github.com/ggml-org/ggml): Tensor operations, computation graphs, and backend infrastructure.
- [htp-ops-lib](https://github.com/haozixu/htp-ops-lib): Upstream Hexagon HTP operator library. This build uses our modified fork, [htp-ops-lib-sd.cpp](https://github.com/destined555/htp-ops-lib-sd.cpp).
- [Qualcomm Hexagon SDK](https://www.qualcomm.com/developer/software/hexagon-sdk): Build and runtime tools for Hexagon DSP.
- [OpenCL-Headers](https://github.com/KhronosGroup/OpenCL-Headers): Headers required to build the OpenCL backend.
