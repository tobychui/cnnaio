#!/usr/bin/env bash
# Face attribute — gender classification (MobileNetV2-0.35).
# POST /v1/faces/gender
# cropped=false -> classify the largest face in the photo; true -> input is a face.
source "$(dirname "$0")/_common.sh"

curl -sS -X POST "$BASE_URL/v1/faces/gender" "${auth_args[@]}" \
  -F image=@"$IMAGE" \
  -F model=gender-mbv2-0.35 \
  -F cropped=false \
  -F render=false \
  -F async=false \
  | pp
