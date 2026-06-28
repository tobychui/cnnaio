#!/usr/bin/env bash
# Shared config + helpers for the cnnaio REST API example scripts.
# Every numbered script sources this file.
#
# Override anything from the environment, e.g.:
#   BASE_URL=http://192.168.1.10:8080 CNNAIO_KEY=cxn-... ./03_detect_yolo.sh
set -euo pipefail

# --- where the server lives --------------------------------------------------
BASE_URL="${BASE_URL:-http://localhost:8080}"

# API token. The shipped conf/config.json has "no_auth": true, so this is
# optional; set CNNAIO_KEY (or run `go run . -nt`) once you enable auth.
API_KEY="${CNNAIO_KEY:-}"
auth_args=()
[ -n "$API_KEY" ] && auth_args=(-H "Authorization: Bearer $API_KEY")

# --- the image these scripts send -------------------------------------------
# Defaults to the sample shipped with the examples (has a face + objects).
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${IMAGE:-$HERE/../testdata/test.png}"
IMAGE_A="${IMAGE_A:-$IMAGE}"   # face comparison: first image
IMAGE_B="${IMAGE_B:-$IMAGE}"   # face comparison: second image

# --- helpers ----------------------------------------------------------------
# pp: pretty-print JSON if jq is installed, otherwise pass through.
pp() { if command -v jq >/dev/null 2>&1; then jq .; else cat; fi; }

# data_uri <path>: print a base64 data URI for the given image (for JSON bodies).
data_uri() {
  local f="$1" mime="image/png"
  case "$f" in
    *.jpg|*.jpeg) mime="image/jpeg" ;;
    *.webp)       mime="image/webp" ;;
    *.bmp)        mime="image/bmp"  ;;
  esac
  printf 'data:%s;base64,%s' "$mime" "$(base64 "$f" | tr -d '\n')"
}

# json_post <url> <json-file>: POST a JSON body file and pretty-print the reply.
json_post() {
  curl -sS -X POST "$1" "${auth_args[@]}" \
    -H "Content-Type: application/json" \
    --data-binary "@$2" | pp
}

echo "# POST -> $BASE_URL  (image: $IMAGE)" >&2
