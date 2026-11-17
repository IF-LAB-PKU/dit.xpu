#!/system/bin/sh
set -eu
htp_script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
SD_HTP_ROOT=${SD_HTP_ROOT:-$(CDPATH= cd -- "$htp_script_dir/../.." && pwd)}
SD_HTP_BIN_DIR=${SD_HTP_BIN_DIR:-$SD_HTP_ROOT/bin}
. "$htp_script_dir/env.sh"
export SD_HTP_PACKED_STAGING_BUDGET_MB="${SD_HTP_PACKED_STAGING_BUDGET_MB:-1024}"
export SD_HTP_SD3_DIT_PREFETCH="${SD_HTP_SD3_DIT_PREFETCH:-1}"
export SD_HTP_SD3_DIT_NATIVE="${SD_HTP_SD3_DIT_NATIVE:-0}"
htp_model_dir=${SD_HTP_MODEL_DIR:-$SD_HTP_ROOT/model/sd3.5-large}
htp_shared_dir=${SD_HTP_SD3_SHARED_DIR:-$SD_HTP_ROOT/model/sd3.5-medium}
cd "$SD_HTP_BIN_DIR"
exec "${SD_HTP_CLI:-$SD_HTP_BIN_DIR/sd-cli}" \
    --backend cpu --params-backend cpu -t 4 --max-vram 0.5 \
    --diffusion-model "$htp_model_dir/sd3.5_large_htp_w16a32.gguf" \
    --clip_l "$htp_shared_dir/clip_l_htp_w16a32.gguf" \
    --clip_g "$htp_shared_dir/clip_g_htp_w16a32.gguf" \
    --t5xxl "$htp_shared_dir/t5xxl_htp_w16a32.gguf" \
    --vae "$htp_shared_dir/sd3.5vae_htp_w16a32.gguf" \
    --clip-on-cpu --diffusion-fa --sampling-method euler --steps 20 --cfg-scale 4.5 \
    -W 256 -H 256 -s 42 -p 'a cat sitting on a wooden table' -o sd35-large.png "$@"
