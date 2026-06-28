#!/usr/bin/env bash
# Link ncnn (prebuilt wasm32 static archive) + stb_image + classify.c into a
# single self-contained WASI module. The result is written to mod/ncnn/ where it
# is embedded into the Go binary via go:embed.
#
# Requires only the wasi-sdk extracted under build/ (no system clang needed).
set -euo pipefail
cd "$(dirname "$0")/.."   # project root

# Locate the WASI SDK extracted under build/ (any platform/arch). configure.sh
# downloads it as build/wasi-sdk-33.0-<arch>-<os>/.
WASI_SDK="$(ls -d build/wasi-sdk-33.0-* 2>/dev/null | head -1)"
if [ -z "$WASI_SDK" ] || [ ! -x "$WASI_SDK/bin/clang" ]; then
  echo "WASI SDK not found under build/. Run ./configure.sh first (or see README > Build from source)." >&2
  exit 1
fi
CLANG="$WASI_SDK/bin/clang"
CLANGXX="$WASI_SDK/bin/clang++"
NCNN="ncnn-20260526-webassembly/basic"
OUT="mod/ncnn/ncnn_classify.wasm"

echo ">> compiling classify.c (C, with stb_image) -> classify.o"
"$CLANG" \
  --target=wasm32-wasip1 \
  -O2 -DNDEBUG \
  -I "$NCNN/include" \
  -I "build" \
  -c build/classify.c \
  -o build/classify.o

echo ">> compiling shim.c (libc/exception forwarders) -> shim.o"
"$CLANG" \
  --target=wasm32-wasip1 \
  -O2 -fno-builtin \
  -c build/shim.c \
  -o build/shim.o

echo ">> linking with libncnn.a (C++ -> pulls libc++/libc++abi) -> $OUT"
"$CLANGXX" \
  --target=wasm32-wasip1 \
  -O2 \
  -Wl,-z,stack-size=8388608 \
  -Wl,--initial-memory=33554432 \
  build/classify.o \
  build/shim.o \
  "$NCNN/lib/libncnn.a" \
  -lm \
  -o "$OUT"

echo ">> done:"
ls -la "$OUT"
