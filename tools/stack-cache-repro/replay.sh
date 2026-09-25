#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CRATE_MANIFEST="$SCRIPT_DIR/Cargo.toml"
WORKLOAD="$SCRIPT_DIR/target/release/stack-cache-repro"
OUT_DIR=${OUT_DIR:-"$SCRIPT_DIR/target/stack-cache-repro"}
PERF_DATA="$OUT_DIR/repro.perf.data"

mkdir -p "$OUT_DIR"
cargo build --release --manifest-path "$CRATE_MANIFEST"

if [[ ! -e "$PERF_DATA" || ${RERECORD:-0} == 1 ]]; then
    perf record \
        -e cycles:u \
        --call-graph dwarf,32000 \
        -F 499 \
        -m 16 \
        -o "$PERF_DATA" \
        -- "$WORKLOAD"
fi

if (($# == 0)); then
    set -- samply
fi

outputs=()
for samply in "$@"; do
    name=$(basename -- "$samply")
    output="$OUT_DIR/$name.json.gz"
    "$samply" import \
        --presymbolicate \
        --save-only \
        -o "$output" \
        "$PERF_DATA"
    outputs+=("$output")
done

"$SCRIPT_DIR/stats.py" "${outputs[@]}"
