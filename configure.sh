#!/usr/bin/env bash
# =============================================================================
#  configure.sh — fetch the build-time dependencies that are NOT committed to
#  git, so you can rebuild the embedded ncnn wasm (build/build.sh) and `make`.
#
#  Downloads, for your host platform:
#    1. ncnn <ver> webassembly prebuilt  ->  ./ncnn-<ver>-webassembly/
#    2. WASI SDK <ver> (clang toolchain) ->  ./build/wasi-sdk-<ver>-<arch>-<os>/
#
#  Usage:
#    ./configure.sh          # download anything missing
#    ./configure.sh --force  # re-download even if the folders already exist
#
#  Requirements: bash, curl (or wget), tar, and an unzip capability
#  (unzip / bsdtar / PowerShell). Works in Git Bash on Windows, Linux, macOS.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

# --- versions (bump these together with build/build.sh + README) -------------
NCNN_VER="20260526"
WASI_TAG="wasi-sdk-33"
WASI_VER="33.0"

FORCE=0
[ "${1:-}" = "--force" ] && FORCE=1

have() { command -v "$1" >/dev/null 2>&1; }

fetch() { # url out
  echo ">> downloading $(basename "$2")"
  echo "   $1"
  if   have curl; then curl -fL --progress-bar -o "$2" "$1"
  elif have wget; then wget -q --show-progress -O "$2" "$1"
  else echo "ERROR: need curl or wget on PATH" >&2; exit 1; fi
}

extract_zip() { # zipfile destdir
  mkdir -p "$2"
  if   have unzip; then unzip -q "$1" -d "$2"
  elif tar --version 2>/dev/null | grep -qi bsdtar; then tar -xf "$1" -C "$2"
  elif have powershell; then
    powershell -NoProfile -Command "Expand-Archive -Force -LiteralPath '$1' -DestinationPath '$2'"
  else
    echo "ERROR: need 'unzip', bsdtar, or PowerShell to unpack a .zip" >&2; exit 1
  fi
}

extract_targz() { mkdir -p "$2"; tar -xzf "$1" -C "$2"; } # tarball destdir

# --- detect platform for the wasi-sdk asset ----------------------------------
case "$(uname -s)" in
  Linux)               WOS=linux ;;
  Darwin)              WOS=macos ;;
  MINGW*|MSYS*|CYGWIN*) WOS=windows ;;
  *) echo "unsupported OS: $(uname -s)" >&2; exit 1 ;;
esac
case "$(uname -m)" in
  x86_64|amd64)  WARCH=x86_64 ;;
  arm64|aarch64) WARCH=arm64 ;;
  *) echo "unsupported arch: $(uname -m)" >&2; exit 1 ;;
esac
[ "$WOS" = windows ] && WARCH=x86_64   # only an x86_64 windows build is published

NCNN_DIR="ncnn-${NCNN_VER}-webassembly"
NCNN_URL="https://github.com/Tencent/ncnn/releases/download/${NCNN_VER}/${NCNN_DIR}.zip"

WASI_DIR="build/wasi-sdk-${WASI_VER}-${WARCH}-${WOS}"
WASI_URL="https://github.com/WebAssembly/wasi-sdk/releases/download/${WASI_TAG}/wasi-sdk-${WASI_VER}-${WARCH}-${WOS}.tar.gz"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# --- 1. ncnn webassembly -----------------------------------------------------
if [ -d "$NCNN_DIR" ] && [ "$FORCE" -eq 0 ]; then
  echo "== ncnn: '$NCNN_DIR' already present (use --force to re-download)"
else
  rm -rf "$NCNN_DIR"
  fetch "$NCNN_URL" "$TMP/ncnn.zip"
  extract_zip "$TMP/ncnn.zip" "$TMP/ncnn"
  # The zip may carry a top-level '<name>/' folder or its bare contents.
  if [ -d "$TMP/ncnn/$NCNN_DIR" ]; then
    mv "$TMP/ncnn/$NCNN_DIR" "$NCNN_DIR"
  else
    mkdir -p "$NCNN_DIR"; mv "$TMP/ncnn/"* "$NCNN_DIR/"
  fi
  echo "== ncnn -> $NCNN_DIR/"
fi

# --- 2. wasi-sdk -------------------------------------------------------------
if [ -d "$WASI_DIR" ] && [ "$FORCE" -eq 0 ]; then
  echo "== wasi-sdk: '$WASI_DIR' already present (use --force to re-download)"
else
  rm -rf "$WASI_DIR"
  fetch "$WASI_URL" "$TMP/wasi.tar.gz"
  extract_targz "$TMP/wasi.tar.gz" "$TMP/wasi"
  src="$(find "$TMP/wasi" -mindepth 1 -maxdepth 1 -type d | head -1)"
  mkdir -p build
  mv "$src" "$WASI_DIR"
  echo "== wasi-sdk -> $WASI_DIR/"
fi

echo
echo "Done. Next steps:"
echo "  bash build/build.sh     # compile ncnn -> mod/ncnn/ncnn_classify.wasm"
echo "  make                    # (optional) cross-platform bundles -> ./dist"
