# Build from scratch

This fork retains the CPU, CUDA, OpenCL, and HTP backends.

## Get the Code

```
git clone --recursive https://github.com/leejet/stable-diffusion.cpp
cd stable-diffusion.cpp
```

- If you have already cloned the repository, you can use the following command to update the repository to the latest code.

```
cd stable-diffusion.cpp
git pull origin master
git submodule init
git submodule update
```

## WebP and WebM Support in Examples

The example applications (`examples/cli` and `examples/server`) use `libwebp` to support WebP image I/O, and `examples/cli` can also use `libwebm` for `.webm` video output. Both are enabled by default. WebM output currently reuses `libwebp` to encode each frame as VP8 before muxing with `libwebm`.

If you do not want WebP/WebM support, you can disable them at configure time:

```shell
mkdir build && cd build
cmake .. -DSD_WEBP=OFF -DSD_WEBM=OFF
cmake --build . --config Release
```

If the submodules are not available, you can also link against system packages instead:

```shell
mkdir build && cd build
cmake .. -DSD_USE_SYSTEM_WEBP=ON -DSD_USE_SYSTEM_WEBM=ON
cmake --build . --config Release
```

## Build (CPU only)

If you don't have a GPU or CUDA installed, you can build a CPU-only version.

```shell
mkdir build && cd build
cmake ..
cmake --build . --config Release
```

## Build with CUDA

This provides GPU acceleration using NVIDIA GPU. Make sure to have the CUDA toolkit installed. You can download it from your Linux distro's package manager (e.g. `apt install nvidia-cuda-toolkit`) or from here: [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads). Recommended to have at least 4 GB of VRAM.

```shell
mkdir build && cd build
cmake .. -DSD_CUDA=ON
cmake --build . --config Release
```

## Build with OpenCL (for Adreno GPU)

Currently, it supports only Adreno GPUs and is primarily optimized for Q4_0 type

To build for Windows ARM please refers to [Windows 11 Arm64](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/OPENCL.md#windows-11-arm64)

Building for Android:

  Android NDK:
       Download and install the Android NDK from the [official Android developer site](https://developer.android.com/ndk/downloads).

Setup OpenCL Dependencies for NDK:

You need to provide OpenCL headers and the ICD loader library to your NDK sysroot.

*   OpenCL Headers:
    ```bash
    # In a temporary working directory
    git clone https://github.com/KhronosGroup/OpenCL-Headers
    cd OpenCL-Headers
    # Replace <YOUR_NDK_PATH> with your actual NDK installation path
    # e.g., cp -r CL /path/to/android-ndk-r26c/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/include
    sudo cp -r CL <YOUR_NDK_PATH>/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/include
    cd ..
    ```

*   OpenCL ICD Loader:
    ```shell
    # In the same temporary working directory
    git clone https://github.com/KhronosGroup/OpenCL-ICD-Loader
    cd OpenCL-ICD-Loader
    mkdir build_ndk && cd build_ndk

    # Replace <YOUR_NDK_PATH> in the CMAKE_TOOLCHAIN_FILE and OPENCL_ICD_LOADER_HEADERS_DIR
    cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_TOOLCHAIN_FILE=<YOUR_NDK_PATH>/build/cmake/android.toolchain.cmake \
      -DOPENCL_ICD_LOADER_HEADERS_DIR=<YOUR_NDK_PATH>/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/include \
      -DANDROID_ABI=arm64-v8a \
      -DANDROID_PLATFORM=24 \
      -DANDROID_STL=c++_shared

    ninja
    # Replace <YOUR_NDK_PATH>
    # e.g., cp libOpenCL.so /path/to/android-ndk-r26c/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android
    sudo cp libOpenCL.so <YOUR_NDK_PATH>/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android
    cd ../..
    ```

Build `stable-diffusion.cpp` for Android with OpenCL:

```shell
mkdir build-android && cd build-android

# Replace <YOUR_NDK_PATH> with your actual NDK installation path
# e.g., -DCMAKE_TOOLCHAIN_FILE=/path/to/android-ndk-r26c/build/cmake/android.toolchain.cmake
cmake .. -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE=<YOUR_NDK_PATH>/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-28 \
  -DGGML_OPENMP=OFF \
  -DSD_OPENCL=ON

ninja
```
*(Note: Don't forget to include `LD_LIBRARY_PATH=/vendor/lib64` in your command line before running the binary)*
