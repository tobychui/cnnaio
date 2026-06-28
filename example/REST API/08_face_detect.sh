#!/usr/bin/env bash
# Face detection — Ultra-Light RFB-320 (default; use ultraface-slim-320 for speed).
# POST /v1/faces/detections
source "$(dirname "$0")/_common.sh"

curl -sS -X POST "$BASE_URL/v1/faces/detections" "${auth_args[@]}" \
  -F image=@"$IMAGE" \
  -F model=ultraface-rfb-320 \
  -F score_threshold=0.7 \
  -F nms_threshold=0.3 \
  -F max_results=100 \
  -F render=false \
  -F async=false \
  | pp
