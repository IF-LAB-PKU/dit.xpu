#!/usr/bin/env bash
# ============================================================
# 8 Gen 5 (SM8850 / HTP v81) NPU 时延自动测试
#   sd3.5m / sd3.5l / flux2 / zimage / zimageturbo  x  256 / 512
#
# 前置:
#   - 设备已部署本仓库 Android 构建与修复后的 HTP 库
#     (script/htp-8gen5/deploy-8gen5.sh 或手动替换 bin/)
#   - 宿主机模型目录 HOST_MODEL_DIR / HOST_TURBO_DIR
#
# 用法:
#   bench-8gen5-npu.sh                 # 全量
#   MODELS="sd3.5m flux2" SIZES="256" bench-8gen5-npu.sh
#   AUTO_CLEAN=0 ...                   # 保留传输到设备的大模型
# 结果:
#   $RESULT_DIR/summary-npu-all.tsv + 每例 log/png
# ============================================================
set -uo pipefail

H=${H:-192.168.31.189}
P=${P:-2222}
DEV=${DEV:-/data/local/tmp/sd-new-htp}
HOST_MODEL_DIR=${HOST_MODEL_DIR:-/data/anqian/test/model}
HOST_TURBO_DIR=${HOST_TURBO_DIR:-/data/anqian/8gen5/models}
RESULT_DIR=${RESULT_DIR:-/data/anqian/8gen5test/8gen5result/npu-all}
SUMMARY=${SUMMARY:-$RESULT_DIR/summary-npu-all.tsv}
LOG=${LOG:-/data/anqian/8gen5/bench-8gen5-npu-all.log}
MODELS=${MODELS:-"sd3.5m sd3.5l flux2 zimage zimageturbo"}
SIZES=${SIZES:-"256 512"}
AUTO_CLEAN=${AUTO_CLEAN:-1}
WAIT_IDLE=${WAIT_IDLE:-/data/anqian/wait-adb-idle.sh}

SSHc=(ssh -o BatchMode=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=4 -o ConnectTimeout=15 -p "$P" root@"$H")

stamp() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

mkdir -p "$RESULT_DIR"
if [ ! -s "$SUMMARY" ]; then
  printf 'model\tsize\tsteps\tsampling_s\tbranches\tsum_compute_ms\tavg_branch_ms\tgenerate_s\texit\n' > "$SUMMARY"
fi

# 设备侧 HTP packed 权重的 RPCMEM 占用（MB）。16GB 级 8 Gen 5 上
# 12GB 左右可跑，sd3.5l 的 15.3GB 会触发内核 OOM/重启，默认跳过。
MAX_PACKED_MB=${MAX_PACKED_MB:-13000}
packed_mb_for() { case "$1" in sd3.5m) echo 4249;; sd3.5l) echo 15311;; flux2) echo 7290;; zimage) echo 11734;; zimageturbo) echo 11734;; *) echo 0;; esac; }

# ---- HTP 运行环境 ----
PRE="cd $DEV; export LD_LIBRARY_PATH=$DEV/bin; export DSP_LIBRARY_PATH=$DEV/bin; export ADSP_LIBRARY_PATH=$DEV/bin; export SD_PRELOAD_BACKENDS=htp; export SD_HTP_OPS=1; export SD_HTP_COMPUTE=1; export SD_HTP_ATTENTION=1; export SD_HTP_DENOISE_STEP_TRACE=1; export GGML_HTP_CPU_FALLBACK_THREADS=4;"

# ---- 各模型参数 ----
steps_for() { case "$1" in sd3.5m) echo 20;; sd3.5l) echo 8;; flux2) echo 4;; zimage) echo 10;; zimageturbo) echo 8;; esac; }
args_for() {
  case "$1" in
    sd3.5m)
      echo "--clip_l model/sd3.5-medium/clip_l.safetensors --clip_g model/sd3.5-medium/clip_g.safetensors --t5xxl model/sd3.5-medium/t5xxl_fp8_e4m3fn.safetensors --diffusion-model model/sd3.5-medium/sd3.5_medium_htp_w16a32.gguf --vae model/sd3.5-medium/sd3.5vae.safetensors --clip-on-cpu --diffusion-fa -p cat --steps 20 --cfg-scale 4.5 --guidance 3.5 --sampling-method euler -s 42" ;;
    sd3.5l)
      echo "--clip_l model/sd3.5-medium/clip_l.safetensors --clip_g model/sd3.5-medium/clip_g.safetensors --t5xxl model/sd3.5-medium/t5xxl_fp8_e4m3fn.safetensors --diffusion-model model/sd3.5-large/sd3.5_large_htp_w16a32.gguf --vae model/sd3.5-medium/sd3.5vae.safetensors --clip-on-cpu --diffusion-fa -p cat --steps 8 --cfg-scale 4.5 --guidance 3.5 --sampling-method euler -s 42" ;;
    flux2)
      echo "--diffusion-model model/flux2-klein/flux-2-klein-4b_htp_w16a32.gguf --vae model/flux2-klein/flux2_ae.safetensors --llm model/flux2-klein/model/qwen3-4b/ggml/weights.gguf --clip-on-cpu --diffusion-fa -p 'a lovely cat' --steps 4 --cfg-scale 1.0 --sampling-method euler -s 42" ;;
    zimage)
      echo "--diffusion-model model/z-image/z_image_bf16_htp_w16a32.gguf --vae model/sd3.5-medium/sd3.5vae.safetensors --llm model/flux2-klein/model/qwen3-4b/ggml/weights.gguf --clip-on-cpu --diffusion-fa -p 'a lovely cat' --steps 10 --cfg-scale 1.0 --guidance 1.0 --sampling-method euler -s 42" ;;
    zimageturbo)
      echo "--diffusion-model model/z-image/z_image_turbo_bf16_htp_w16a32.gguf --vae model/sd3.5-medium/sd3.5vae.safetensors --llm model/flux2-klein/model/qwen3-4b/ggml/weights.gguf --clip-on-cpu --diffusion-fa -p 'a lovely cat' --steps 8 --cfg-scale 1.0 --guidance 1.0 --sampling-method euler -s 42" ;;
  esac
}

# ---- 大模型文件: 需要时从宿主机传输 ----
model_file() { case "$1" in sd3.5l) echo model/sd3.5-large/sd3.5_large_htp_w16a32.gguf;; zimage) echo model/z-image/z_image_bf16_htp_w16a32.gguf;; zimageturbo) echo model/z-image/z_image_turbo_bf16_htp_w16a32.gguf;; esac; }
host_file() {
  case "$1" in
    sd3.5l) echo "$HOST_MODEL_DIR/sd3.5_large_htp_w16a32.gguf" ;;
    zimage) echo "$HOST_MODEL_DIR/z_image_bf16_htp_w16a32.gguf" ;;
    zimageturbo) echo "$HOST_TURBO_DIR/z_image_turbo_bf16_htp_w16a32.gguf" ;;
  esac
}

dev_exists() { "${SSHc[@]}" "test -f '$DEV/$1' && echo yes || echo no" 2>/dev/null | tr -d '\r\n' | grep -q yes; }
ensure_model() { # $1=model tag; sets TRANSFERRED=1 when pushed
  TRANSFERRED=0
  local mf hf sz free
  mf=$(model_file "$1"); [ -n "$mf" ] || return 0
  if dev_exists "$mf"; then stamp "[model] $1 present"; return 0; fi
  hf=$(host_file "$1")
  [ -f "$hf" ] || { stamp "[ERROR] missing host model: $hf"; return 1; }
  sz=$(stat -c %s "$hf")
  free=$("${SSHc[@]}" "df -k '$DEV' | tail -1 | awk '{print \$4*1024}'" 2>/dev/null | tr -d '\r')
  if [ -n "$free" ] && [ "$free" -lt $((sz + 1073741824)) ]; then
    stamp "[ERROR] not enough device space for $1: need $((sz/1048576))MB free=$((free/1048576))MB"
    return 1
  fi
  stamp "[model] pushing $1 ($((sz/1048576))MB) -> $DEV/$mf"
  "${SSHc[@]}" "mkdir -p '$DEV/$(dirname "$mf")'"
  cat "$hf" | "${SSHc[@]}" "cat > '$DEV/$mf'" || return 1
  TRANSFERRED=1
  return 0
}

cleanup_model() { # $1=model tag
  local mf
  mf=$(model_file "$1"); [ -n "$mf" ] || return 0
  [ "$AUTO_CLEAN" = "1" ] || return 0
  [ "${TRANSFERRED:-0}" = "1" ] || return 0
  stamp "[model] removing transferred $1 from device"
  "${SSHc[@]}" "rm -f '$DEV/$mf'"
}

# ---- 解析 ----
parse_case() { # $1=model $2=W
  local tag="$1-$2" base="$RESULT_DIR/$tag"
  local sampling gen branches sum avg
  sampling=$(grep -a -oE 'sampling completed, taking [0-9.]+s' "$base.log" 2>/dev/null | tail -1 | grep -a -oE '[0-9.]+' | tr -d '\r' || true)
  gen=$(grep -a -oE 'generate_image completed in [0-9.]+s' "$base.log" 2>/dev/null | tail -1 | grep -a -oE '[0-9.]+' | tr -d '\r' || true)
  branches=$(grep -a -cE 'htp_denoise_step_timing step=' "$base.log" 2>/dev/null || true)
  sum=$(grep -a -oE 'compute=[0-9.]+ms' "$base.log" 2>/dev/null | grep -a -oE '[0-9.]+' | awk '{s+=$1} END{printf "%.0f", s+0}' 2>/dev/null || echo 0)
  if [ "${branches:-0}" -gt 0 ] 2>/dev/null; then avg=$(awk -v s="${sum:-0}" -v n="$branches" 'BEGIN{printf "%.0f", s/n}'); else avg=0; fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$1" "$2" "$(steps_for "$1")" "${sampling:-NA}" "${branches:-0}" "${sum:-0}" "$avg" "${gen:-NA}" "${CASE_EXIT:-NA}" >> "$SUMMARY"
}

run_case() { # $1=model $2=W
  local model="$1" W="$2" tag="$1-$2" attempt cmd
  for attempt in 1 2 3; do
    stamp "=== $tag (attempt $attempt/3, steps $(steps_for "$model")) ==="
    cmd="$PRE ./bin/sd-cli -v -t 4 --backend cpu --params-backend cpu --vae-tiling $(args_for "$model") -W $W -H $W -o $tag.png > $tag.log 2>&1; rc=\$?; /system/bin/echo EXIT:\$rc"
    if [ -x "$WAIT_IDLE" ]; then
      CASE_EXIT=$("$WAIT_IDLE" -H "$H" -p "$P" -i 5 -t 45 -- "${SSHc[@]}" "$cmd" 2>&1 | tee -a "$LOG" | grep -a -oE 'EXIT:[0-9]+' | tail -1 | cut -d: -f2)
    else
      CASE_EXIT=$("${SSHc[@]}" "$cmd" 2>&1 | tee -a "$LOG" | grep -a -oE 'EXIT:[0-9]+' | tail -1 | cut -d: -f2)
    fi
    "${SSHc[@]}" "cat $DEV/$tag.log" > "$RESULT_DIR/$tag.log" 2>/dev/null || true
    "${SSHc[@]}" "test -f $DEV/$tag.png && cat $DEV/$tag.png" > "$RESULT_DIR/$tag.png" 2>/dev/null || true
    if grep -q 'generate_image completed' "$RESULT_DIR/$tag.log" 2>/dev/null; then
      stamp "[done] $tag (attempt $attempt, exit=${CASE_EXIT:-?})"
      parse_case "$model" "$W"
      return 0
    fi
    stamp "[FAIL] $tag attempt $attempt exit=${CASE_EXIT:-?}; retry in 120s"
    sleep 120
  done
  parse_case "$model" "$W"
  return 1
}

# ---- 主流程 ----
for model in $MODELS; do
  packed=$(packed_mb_for "$model")
  if [ "$packed" -gt "$MAX_PACKED_MB" ] && [ "${ALLOW_BIG_MODELS:-0}" != "1" ]; then
    stamp "[skip] $model: HTP packed ${packed}MB > MAX_PACKED_MB=${MAX_PACKED_MB} (set ALLOW_BIG_MODELS=1 to force; this device reboots)"
    continue
  fi
  ensure_model "$model" || continue
  for W in $SIZES; do
    run_case "$model" "$W"
  done
  cleanup_model "$model"
done

stamp "===== NPU summary ($SUMMARY) ====="
column -t -s $'\t' "$SUMMARY" 2>/dev/null || cat "$SUMMARY"
