#!/system/bin/sh
set -eu
htp_script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
SD_HTP_ROOT=${SD_HTP_ROOT:-$(CDPATH= cd -- "$htp_script_dir/../.." && pwd)}
SD_HTP_BIN_DIR=${SD_HTP_BIN_DIR:-$SD_HTP_ROOT/bin}
. "$htp_script_dir/env.sh"
export SD_HTP_PACKED_STAGING_BUDGET_MB="${SD_HTP_PACKED_STAGING_BUDGET_MB:-1024}"
export SD_HTP_DIT_PREFETCH="${SD_HTP_DIT_PREFETCH:-1}"
export SD_HTP_DIT_NATIVE="${SD_HTP_DIT_NATIVE:-1}" SD_HTP_DIT_CONCAT="${SD_HTP_DIT_CONCAT:-1}"
htp_model_dir=${SD_HTP_MODEL_DIR:-$SD_HTP_ROOT/model/flux2-klein}
cd "$SD_HTP_BIN_DIR"
exec "${SD_HTP_CLI:-$SD_HTP_BIN_DIR/sd-cli}" \
    --backend cpu --params-backend cpu -t 4 --max-vram 0.5 \
    --diffusion-model "$htp_model_dir/flux2_klein_4b_htp_f16.gguf" \
    --llm "$htp_model_dir/qwen3_4b_htp_w16a32.gguf" \
    --vae "$htp_model_dir/flux2_vae_htp_w16a32.gguf" \
    --diffusion-fa --sampling-method euler --steps 4 --cfg-scale 1 \
    -W 256 -H 256 -s 42 -p 'a cat sitting on a wooden table' -o flux2-klein.png "$@"
