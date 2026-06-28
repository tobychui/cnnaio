#!/usr/bin/env bash
# Image classification — YOLO11n-cls (ImageNet-1000).
# POST /v1/images/classifications
source "$(dirname "$0")/_common.sh"

curl -sS -X POST "$BASE_URL/v1/images/classifications" "${auth_args[@]}" \
  -F image=@"$IMAGE" \
  -F model=yolo11n-cls \
  -F top_k=5 \
  -F render=false \
  -F async=false \
  | pp
