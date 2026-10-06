# 8 Gen 5 (SM8850 / HTP v81) 支持

## 问题

`sd.cpp-npu` 的 HTP 后端依赖配套的 DSP 算子库（`libhtp_ops.so` + `libhtp_ops_skel.so`）。
设备上部署的 skel 是为 v79（8 Gen 4）编译的。在 8 Gen 5（SM8850，HTP v81）上会出现：

1. **HMX shared lock 被拒绝**：skel 在 `hmx_manager_enable_execution()` 中调用
   `HAP_compute_res_hmx_lock2(..., HAP_COMPUTE_RES_HMX_SHARED)`，SM8850 拒绝 SHARED 模式。
   HMX 在未加锁状态下执行导致 DSP SSR（`dmesg` 里 `PD_ERR ... CAUSE=0x1801`），
   主机侧表现为首个 `HTP_OPS_MAT_MUL_PERMUTED_W16A32` 的 `dispatch-timeout`（120s），
   随后 `hybrid op failed err=0xffffff92`（`-ETIMEDOUT`）。
2. **FA 内核 VTCM 假设不成立**：`simple_flash_attn*` 按每个 HVX worker 1 MiB、
   总量不超过 6 MiB 分配 VTCM；SM8850 暴露 8 个 HVX128 上下文
   （`num_hvx128_contexts = 8`，8 × 1 MiB > 6 MiB），FA 越界
   （带 assert 的构建直接 trap，不带 assert 的构建数据访问异常）。

这两个问题都在 DSP 侧算子库，不在宿主 `ggml-htp` 代码里。

## 修复

对配套仓库 `destined555/htp-ops-lib-sd.cpp` 应用 `dsp-8gen5.patch`：

- `src/dsp/hmx_mgr.c`：改为在每个 kernel 临界区内用
  `HAP_compute_res_hmx_lock()/unlock()`（non-shared）加锁，
  receiver 与 HMX worker 线程都能安全使用（`HTP_OPS_HMX_LOCK_PER_SECTION`）。
- `src/dsp/ops/mat_mul.c`：`transfer_permuted_weight_chunk_fp16` 的 DMA 载入路径
  在 v81 上会挂起（v79 正常），改用 worker-pool 向量拷贝路径（`use_dma = false`）。
- `src/dsp/ops/flash_attn.c`、`src/dsp/ops/flash_attn_sp_hdim.c`：
  每线程 VTCM 改为 `min(1 MiB, 6 MiB / n_workers)`，兼容 8 个 HVX 上下文。

重新构建 skel（Hexagon SDK 6.6.0.0）：

```sh
source <Hexagon_SDK>/setup_sdk_env.source
cd htp-ops-lib-sd.cpp
git apply /path/to/dsp-8gen5.patch
build_cmake hexagon DSP_ARCH=v81
# 产物: hexagon_ReleaseG_toolv19_v81/ship/libhtp_ops_skel.so
```

## 预编译产物

- `libhtp_ops_skel_v81.so`：按上述补丁构建的 v81 skel
  （基线 `htp-ops-lib-sd.cpp@20f5248`，Hexagon Tools 19.0.07，`DSP_ARCH=v81`）。
- `libhtp_ops.so`：与 skel 配套的宿主 session 库（Android aarch64）。
  宿主 `ggml-htp` 通过 `dlopen("libhtp_ops.so")` 打开会话/消息通道，
  它与 skel 的 FastRPC IDL 必须配套；仓库自带的宿主库只保证与
  本目录的 skel 配套，**必须一起替换**（只换 skel 会报
  `htp runtime unavailable`）。

部署（同时替换宿主 session 库与 skel）：

```sh
./deploy-8gen5.sh [device_ip] [ssh_port] [device_bin_dir]
# 默认: 192.168.31.189 2222 /data/local/tmp/sd-new-htp/bin
```

运行（与 8 Gen 4 相同的宿主命令）：

```sh
export LD_LIBRARY_PATH=$PWD/bin DSP_LIBRARY_PATH=$PWD/bin ADSP_LIBRARY_PATH=$PWD/bin
export SD_PRELOAD_BACKENDS=htp SD_HTP_OPS=1 SD_HTP_COMPUTE=1 SD_HTP_ATTENTION=1
./bin/sd-cli -v -t 4 --backend cpu --params-backend cpu --vae-tiling \
  --diffusion-model model/sd3.5-medium/sd3.5_medium_htp_w16a32.gguf \
  --vae model/sd3.5-medium/sd3.5vae.safetensors \
  --clip_l model/sd3.5-medium/clip_l.safetensors \
  --clip_g model/sd3.5-medium/clip_g.safetensors \
  --t5xxl model/sd3.5-medium/t5xxl_fp8_e4m3fn.safetensors \
  --clip-on-cpu --diffusion-fa -p cat --steps 4 -W 512 -H 512 -s 42 -o out.png
```

## 验证

2026-10-06 在 SM8850（8 Gen 5）上使用本仓库 HEAD 构建的宿主 +
`libhtp_ops_skel_v81.so`：

- SD3.5 Medium（`sd3.5_medium_htp_w16a32.gguf`，256×256，2 步）：
  `sampling completed`，`generate_image completed`，`EXIT:0`，无 dispatch 超时。
- FLUX.2 klein 4B（`flux-2-klein-4b_htp_w16a32.gguf`，256×256，1 步）：
  `sampling completed`，`generate_image completed`，`EXIT:0`。

## 时延测试

`bench-8gen5-npu.sh` 自动测试 sd3.5m / sd3.5l / flux2 / zimage / zimageturbo
× 256/512 的 NPU 时延（HTP dense，`--diffusion-fa`，euler，seed 42）。
脚本按需从宿主机传输大模型（sd3.5l / z-image / z-image-turbo），
测试完默认删除传输的模型以节省设备空间（`AUTO_CLEAN=0` 保留）。
每例结果含 `sampling_s`、每分支 `compute` 平均值、`generate_s`。

```sh
script/htp-8gen5/bench-8gen5-npu.sh
# 结果: /data/anqian/8gen5test/8gen5result/npu-all/summary-npu-all.tsv
MODELS="sd3.5m flux2" SIZES="256" script/htp-8gen5/bench-8gen5-npu.sh
```

## 限制

本目录预编译的 skel 来自 `htp-ops-lib-sd.cpp@20f5248`，包含 8 个算子
（RMS_NORM、4 种 MUL_MAT、FLASH_ATTN、BLOCK_SPARSE_ATTN/SCORE）。
宿主 `op_reg.h` 中更新的 packed 算子（`HTP_OPS_QKV_MUL_RMS_PACKED_*`、
`HTP_OPS_FUSED_ANCHOR_ATTN_*` 等，索引 8+）需要更新的 DSP 源；
默认 dense 路径（SD3.5 / FLUX.2）不涉及这些算子。
