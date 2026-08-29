#!/bin/sh
set -eu

cache=${1:?usage: $0 /path/to/CMakeCache.txt /path/to/compile_commands.json}
commands=${2:?usage: $0 /path/to/CMakeCache.txt /path/to/compile_commands.json}

if ! grep -qx 'GGML_HTP_OP_TRACE:BOOL=OFF' "$cache"; then
    echo "GGML_HTP_OP_TRACE must be OFF in a performance build" >&2
    exit 1
fi

if grep -F -- '-DHTP_OPS_OP_TRACE' "$commands" | grep -Fq '/ggml-htp/'; then
    echo "ggml-htp compile commands must not define HTP_OPS_OP_TRACE in a performance build" >&2
    exit 1
fi

echo "HTP host release build contract: PASS"
