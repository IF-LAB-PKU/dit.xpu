#!/usr/bin/env bash
# Push the fixed HTP v81 skeleton + matching host session library into a device workdir.
# Usage: deploy-8gen5.sh [device_ip] [ssh_port] [device_bin_dir]
set -euo pipefail

HOST=${1:-192.168.31.189}
PORT=${2:-2222}
BIN_DIR=${3:-/data/local/tmp/sd-new-htp/bin}
DIR="$(cd "$(dirname "$0")" && pwd)"

ssh -o BatchMode=yes -p "$PORT" "root@$HOST" "test -d '$BIN_DIR'"
ssh -o BatchMode=yes -p "$PORT" "root@$HOST" "cat > '$BIN_DIR/libhtp_ops.so'" < "$DIR/libhtp_ops.so"
ssh -o BatchMode=yes -p "$PORT" "root@$HOST" "cat > '$BIN_DIR/libhtp_ops_skel.so'" < "$DIR/libhtp_ops_skel_v81.so"
echo "deployed libhtp_ops.so + libhtp_ops_skel_v81.so -> $HOST:$BIN_DIR/"
