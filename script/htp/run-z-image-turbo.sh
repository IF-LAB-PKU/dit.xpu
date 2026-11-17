#!/system/bin/sh
set -eu
htp_script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
htp_root=${SD_HTP_ROOT:-$(CDPATH= cd -- "$htp_script_dir/../.." && pwd)}
htp_model_dir=${SD_HTP_MODEL_DIR:-$htp_root/model/z-image}
# The architectures share resource defaults; only model/sampling arguments differ.
exec sh "$htp_script_dir/run-z-image.sh" \
    --diffusion-model "$htp_model_dir/z_image_turbo_htp_w16a32.gguf" \
    --steps 8 --cfg-scale 1 -o z-image-turbo.png "$@"
