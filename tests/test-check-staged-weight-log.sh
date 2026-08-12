#!/usr/bin/env bash

set -eu

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
checker="$repo_root/script/check-staged-weight-log.sh"
tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT

cat > "$tmp_dir/valid.log" <<'EOF'
[INFO] staged parameter loading enabled
[INFO] loading weight stage CONDITIONER (10 tensors, 100 bytes, htp_packed=0 bytes)
[INFO] weight stage CONDITIONER loaded in 0.10s
[INFO] weight stage CONDITIONER released
[INFO] loading weight stage DIFFUSION (20 tensors, 200 bytes, htp_packed=150 bytes)
[INFO] weight stage DIFFUSION loaded in 0.20s
[INFO] weight stage DIFFUSION released
[INFO] loading weight stage VAE (5 tensors, 50 bytes, htp_packed=0 bytes)
[INFO] weight stage VAE loaded in 0.05s
[INFO] weight stage VAE released
EXIT:0
EOF

"$checker" "$tmp_dir/valid.log"

cat > "$tmp_dir/out-of-order.log" <<'EOF'
[INFO] staged parameter loading enabled
[INFO] loading weight stage CONDITIONER (10 tensors, 100 bytes, htp_packed=0 bytes)
[INFO] loading weight stage DIFFUSION (20 tensors, 200 bytes, htp_packed=150 bytes)
EXIT:0
EOF

if "$checker" "$tmp_dir/out-of-order.log"; then
    echo "checker accepted an overlapping stage transition" >&2
    exit 1
fi

cat > "$tmp_dir/eager-peak.log" <<'EOF'
[INFO] staged parameter loading enabled
[INFO] loaded stage bytes: conditioner=100 diffusion=200 vae=50
EXIT:0
EOF

if "$checker" "$tmp_dir/eager-peak.log"; then
    echo "checker accepted simultaneous staged weights" >&2
    exit 1
fi
