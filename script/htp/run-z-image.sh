#!/system/bin/sh
set -eu
htp_script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
SD_HTP_ROOT=${SD_HTP_ROOT:-$(CDPATH= cd -- "$htp_script_dir/../.." && pwd)}
SD_HTP_BIN_DIR=${SD_HTP_BIN_DIR:-$SD_HTP_ROOT/bin}
. "$htp_script_dir/env.sh"
export SD_HTP_PACKED_STAGING_BUDGET_MB="${SD_HTP_PACKED_STAGING_BUDGET_MB:-1024}"
export SD_HTP_Z_IMAGE_DIT_PREFETCH="${SD_HTP_Z_IMAGE_DIT_PREFETCH:-1}"
htp_model_dir=${SD_HTP_MODEL_DIR:-$SD_HTP_ROOT/model/z-image}
htp_vae=${SD_HTP_VAE_PATH:-$htp_model_dir/z_image_vae_htp_w16a32.gguf}
cd "$SD_HTP_BIN_DIR"
exec "${SD_HTP_CLI:-$SD_HTP_BIN_DIR/sd-cli}" \
    --backend cpu --params-backend cpu -t 4 --max-vram 0.5 \
    --diffusion-model "$htp_model_dir/z_image_htp_w16a32.gguf" \
    --llm "$htp_model_dir/qwen3_4b_zimage_htp_w16a32.gguf" --vae "$htp_vae" \
    --diffusion-fa --sampling-method euler --steps 30 --cfg-scale 5 \
    -W 256 -H 256 -s 42 -p 'a cat sitting on a wooden table' -o z-image.png "$@"
