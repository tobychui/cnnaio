#!/usr/bin/env bash
# Image classification — MobileNetV2 (ImageNet-1000).
# POST /v1/images/classifications
source "$(dirname "$0")/_common.sh"

curl -sS -X POST "$BASE_URL/v1/images/classifications" "${auth_args[@]}" \
  -F image=@"$IMAGE" \
  -F model=mobilenet-v2 \
  -F top_k=5 \
  -F render=false \
  -F async=false \
  | pp
