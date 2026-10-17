# sd.cpp-npu

支持以下 5 个模型：

- SD3.5 Medium
- SD3.5 Large
- FLUX.2 Klein 4B
- Z-Image Turbo
- Z-Image Base

以下以 SD3.5 Medium 为例，介绍容器编译、模型准备及 HTP sparse 推理流程。

## 1. 准备环境

安装 Docker、Git、curl、ADB，手机开启 USB 调试。准备 Linux 版 [Hexagon SDK 6.7.0.0](https://www.qualcomm.com/developer/software/hexagon-sdk)（Tools 21.0.04、NDK r27d、CMake 3.28.3）。以下命令在宿主机同一个终端执行。

## 2. 克隆仓库，获取手机 OpenCL 库

```bash
mkdir sdcpp-work
cd sdcpp-work
git clone https://github.com/destined555/sd.cpp-npu.git sdcpp
git clone https://github.com/destined555/htp-ops-lib-sd.cpp.git htp-ops-lib
git -C htp-ops-lib checkout --detach 9e122149185295b7ef4517b40465d0445bd52b9a
git clone --depth 1 --branch v2024.10.24 \
  https://github.com/KhronosGroup/OpenCL-Headers.git OpenCL-Headers

adb devices -l
adb pull /vendor/lib64/libOpenCL.so ./libOpenCL.so
```

## 3. 创建容器，安装依赖和 SDK

将 `SDK_DIR` 改为完整的 Linux SDK 安装目录。

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

## 4. 编译并收集运行文件

编译 Android host 库、DSP v75 库和启用 CPU / HTP / OpenCL 的 sd-cli。

```bash
docker exec sdcpp-build bash /work/sdcpp/script/build.sh
docker cp sdcpp-build:/work/output ./output
```

## 5. 下载模型

从 Hugging Face 下载独立的 DiT、VAE 和文本编码器，保存到以下位置：

- [DiT：sd3.5_medium-F16.gguf](https://huggingface.co/city96/stable-diffusion-3.5-medium-gguf/tree/main) → `models/sd3.5-medium/sd3.5_medium-F16.gguf`
- [VAE：vae/diffusion_pytorch_model.safetensors](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium/tree/main/vae) → `models/sd3.5-medium/sd3.5vae.safetensors`
- [文本编码器](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium/tree/main/text_encoders)：`clip_l.safetensors`、`clip_g.safetensors`、`t5xxl_fp8_e4m3fn.safetensors` → `models/sd3.5-medium/` 下，保留原文件名。

## 6. 部署到手机，转换 HTP GGUF

手机预留至少 20 GB 存储空间。

```bash
adb shell 'mkdir -p /data/local/tmp/sdcpp/model'
adb push output/bin /data/local/tmp/sdcpp/
adb push models/sd3.5-medium /data/local/tmp/sdcpp/model/
adb shell 'chmod +x /data/local/tmp/sdcpp/bin/sd-cli'

adb shell '
  cd /data/local/tmp/sdcpp || exit 1
  export LD_LIBRARY_PATH="$PWD/bin"
  ./bin/sd-cli -M convert \
    -m model/sd3.5-medium/sd3.5_medium-F16.gguf \
    --type f16 --convert-name --htp-pack-dit \
    -o model/sd3.5-medium/sd3.5_medium_htp_w16a32.gguf
'
```

## 7. 运行


```bash
adb shell '
  cd /data/local/tmp/sdcpp || exit 1
  export LD_LIBRARY_PATH="$PWD/bin"
  export DSP_LIBRARY_PATH="$PWD/bin;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp"
  export ADSP_LIBRARY_PATH="$DSP_LIBRARY_PATH"
  export SD_PRELOAD_BACKENDS=htp
  export SD_HTP_OPS=1
  export SD_HTP_COMPUTE=1
  export SD_HTP_ATTENTION=1
  export SD_HTP_BLOCK_SPARSE_ATTN=1
  export SD_HTP_BLOCK_SPARSE_KEEP_RATIO=0
  export SD_HTP_BLOCK_SPARSE_PATCH_HALO=1
  export SD_HTP_BLOCK_SPARSE_WARMUP_RATIO=0.2
  export SD_HTP_BLOCK_SPARSE_DENSE_TAIL_RATIO=0
  export SD_HTP_BLOCK_SPARSE_DENSE_EVERY_N_SPARSE_STEPS=5
  export SD_HTP_BLOCK_SPARSE_RESIDUAL_CACHE=1
  export SD_HTP_BLOCK_SPARSE_RESIDUAL_EXACT_RHO=1
  export SD_HTP_SPARSE_OPENCL=1
  export SD_HTP_SPARSE_OPENCL_RESIDUAL=1
  export SD_HTP_SPARSE_OPENCL_LIFETIME_MODE=reuse
  export SD_HTP_HALO_KV_LOAD=l2fetch
  export SD_HTP_PACKED_ATTENTION=1
  export GGML_HTP_CPU_FALLBACK_THREADS=4

  ./bin/sd-cli -v -t 4 --backend cpu --params-backend cpu \
    --clip_l model/sd3.5-medium/clip_l.safetensors \
    --clip_g model/sd3.5-medium/clip_g.safetensors \
    --t5xxl model/sd3.5-medium/t5xxl_fp8_e4m3fn.safetensors \
    --diffusion-model model/sd3.5-medium/sd3.5_medium_htp_w16a32.gguf \
    --vae model/sd3.5-medium/sd3.5vae.safetensors \
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
