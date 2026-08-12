#!/usr/bin/env bash

set -eu

if [[ $# -ne 1 ]]; then
    echo "usage: $0 LOG_FILE" >&2
    exit 2
fi

log_file="$1"
if [[ ! -f "$log_file" ]]; then
    echo "log file not found: $log_file" >&2
    exit 2
fi

grep -aFq "staged parameter loading enabled" "$log_file" || {
    echo "missing staged loading marker" >&2
    exit 1
}
grep -aFq "EXIT:0" "$log_file" || {
    echo "staged run did not exit successfully" >&2
    exit 1
}

if grep -aEq 'loaded stage bytes: conditioner=[1-9][0-9]* diffusion=[1-9][0-9]* vae=[1-9][0-9]*' "$log_file"; then
    echo "all primary weight stages were resident simultaneously" >&2
    exit 1
fi

awk '
    function stage_after(prefix, line, stage) {
        stage = line
        sub("^.*" prefix " ", "", stage)
        sub(" .*$", "", stage)
        return stage
    }

    /loading weight stage (CONDITIONER|DIFFUSION|VAE)/ {
        if ($0 !~ /\([0-9]+ tensors, [0-9]+ bytes, htp_packed=[0-9]+ bytes\)/) {
            print "stage load record is missing byte diagnostics: " $0 > "/dev/stderr"
            failed = 1
        }
        stage = stage_after("loading weight stage", $0)
        if (active != "") {
            print "stage " stage " loaded before " active " was released" > "/dev/stderr"
            failed = 1
        }
        active = stage
    }

    /weight stage (CONDITIONER|DIFFUSION|VAE) loaded in/ {
        stage = stage_after("weight stage", $0)
        if (active != stage) {
            print "stage loaded without matching begin record: " stage > "/dev/stderr"
            failed = 1
        }
        if (stage == "DIFFUSION" && diffusion_loaded == 0) {
            diffusion_loaded = NR
        }
        if (stage == "VAE" && vae_loaded == 0) {
            vae_loaded = NR
        }
    }

    /weight stage (CONDITIONER|DIFFUSION|VAE) released/ {
        stage = stage_after("weight stage", $0)
        if (active != stage) {
            print "stage released without being active: " stage > "/dev/stderr"
            failed = 1
        }
        if (stage == "CONDITIONER" && conditioner_released == 0) {
            conditioner_released = NR
        }
        if (stage == "DIFFUSION" && diffusion_released == 0) {
            diffusion_released = NR
        }
        active = ""
    }

    END {
        if (active != "") {
            print "stage was not released at end of run: " active > "/dev/stderr"
            failed = 1
        }
        if (!(conditioner_released < diffusion_loaded &&
              diffusion_loaded < diffusion_released &&
              diffusion_released < vae_loaded)) {
            print "missing CONDITIONER -> DIFFUSION -> VAE transition order" > "/dev/stderr"
            failed = 1
        }
        exit failed
    }
' "$log_file"
