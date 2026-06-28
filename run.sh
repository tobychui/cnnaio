#!/usr/bin/env bash
# =============================================================================
#  run.sh - build (if needed) and run the ncnn-in-wazero demo.
#  Works in Git Bash on Windows, and on Linux / macOS.
#
#  Usage:
#     ./run.sh                 # classify tests/1.jpg (default)
#     ./run.sh tests/2.jpg     # classify a specific image
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

# --- Build the wasm module if it doesn't exist yet ---------------------------
if [ ! -f mod/ncnn/ncnn_classify.wasm ]; then
    echo "[run] mod/ncnn/ncnn_classify.wasm not found - building it..."
    bash build/build.sh
fi

# --- Run the pure-Go host (wazero) ------------------------------------------
echo "[run] running classifier via wazero..."
exec go run . "$@"
