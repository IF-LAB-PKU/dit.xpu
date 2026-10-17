#!/usr/bin/env bash
set -eo pipefail
sdk=/opt/Hexagon_SDK/6.7.0.0
source "$sdk/setup_sdk_env.source"
set -u
cmake_bin="$CMAKE_ROOT_PATH/bin/cmake"
ops=/work/htp-ops-lib
sd_cpp=/work/sdcpp
host_build="$ops/android_ReleaseG_aarch64"
dsp_build="$ops/hexagon_ReleaseG_toolv21_v75"

"$cmake_bin" -S "$ops" -B "$host_build" -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE="$sdk/build/cmake/android_toolchain.cmake" \
  -DHEXAGON_SDK_ROOT="$sdk" -DHEXAGON_CMAKE_ROOT="$sdk/build/cmake" \
  -DANDROID_NDK="$ANDROID_ROOT_DIR" -DANDROID_ABI=arm64-v8a \
  -DANDROID_NATIVE_API_LEVEL=26 -DANDROID_STL=none \
  -DCMAKE_BUILD_TYPE=Release -DOS_TYPE=HLOS -DDSP_TYPE=3 \
  -DPREBUILT_LIB_DIR=android_aarch64 -DV=android_ReleaseG_aarch64 \
  -DCMAKE_LIBRARY_OUTPUT_DIRECTORY="$host_build/ship" \
  -DCMAKE_RUNTIME_OUTPUT_DIRECTORY="$host_build/ship" \
  -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY="$host_build/ship"
"$cmake_bin" --build "$host_build" -j4

"$cmake_bin" -S "$ops" -B "$dsp_build" -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE="$sdk/build/cmake/hexagon_toolchain.cmake" \
  -DHEXAGON_SDK_ROOT="$sdk" -DHEXAGON_CMAKE_ROOT="$sdk/build/cmake" \
  -DHEXAGON_TOOLS_ROOT="$DEFAULT_HEXAGON_TOOLS_ROOT" \
  -DDSP_VERSION=v75 -DQURT_OS=1 \
  -DPREBUILT_LIB_DIR=hexagon_toolv21_v75 -DV=hexagon_ReleaseG_toolv21_v75 \
  -DDO_SYSTEM_INCLUDE=0 -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_FLAGS=-Wno-error=pass-failed -DHTP_OPS_QUIET_BENCH=ON \
  -DCMAKE_LIBRARY_OUTPUT_DIRECTORY="$dsp_build/ship"
"$cmake_bin" --build "$dsp_build" -j4

"$cmake_bin" -S "$sd_cpp" -B "$sd_cpp/build-htp-opencl" -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_ROOT_DIR/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 -DANDROID_STL=c++_shared \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
  -DSD_BUILD_SHARED_LIBS=ON -DSD_BUILD_SHARED_GGML_LIB=ON \
  -DSD_BUILD_TESTS=OFF -DSD_BUILD_EXAMPLES=ON \
  -DGGML_HTP=ON -DSD_OPENCL=ON -DGGML_NATIVE=OFF \
  -DSD_WEBP=OFF -DSD_WEBM=OFF -DSD_SERVER_BUILD_FRONTEND=OFF \
  -DOpenCL_INCLUDE_DIR=/work/OpenCL-Headers \
  -DOpenCL_LIBRARY=/work/libOpenCL.so
"$cmake_bin" --build "$sd_cpp/build-htp-opencl" --target sd-cli -j4

mkdir -p /work/output/bin
cp "$sd_cpp/build-htp-opencl/bin/sd-cli" /work/output/bin/
for lib in stable-diffusion ggml ggml-base ggml-cpu ggml-htp ggml-opencl; do
  cp "$sd_cpp/build-htp-opencl/bin/lib$lib.so" /work/output/bin/
done
cp "$host_build/ship/libhtp_ops.so" /work/output/bin/
cp "$dsp_build/ship/libhtp_ops_skel.so" /work/output/bin/
for lib in libc++.so.1 libc++abi.so.1; do
  cp -L "$DEFAULT_HEXAGON_TOOLS_ROOT/Tools/target/hexagon/lib/v75/G0/pic/$lib" /work/output/bin/
done
ndk_lib="$ANDROID_ROOT_DIR/toolchains/llvm/prebuilt/linux-x86_64"
cp "$ndk_lib/lib/clang/18/lib/linux/aarch64/libomp.so" /work/output/bin/
cp "$ndk_lib/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so" /work/output/bin/
