#!/usr/bin/env bash
# Facial landmarks — PFLD (98 points per face).
# POST /v1/faces/landmarks
# cropped=false  -> detect faces first, then landmark each one.
# cropped=true   -> treat the whole image as a single, pre-cropped face.
source "$(dirname "$0")/_common.sh"

curl -sS -X POST "$BASE_URL/v1/faces/landmarks" "${auth_args[@]}" \
  -F image=@"$IMAGE" \
  -F model=pfld \
  -F score_threshold=0.7 \
  -F nms_threshold=0.3 \
  -F cropped=false \
  -F render=false \
  -F async=false \
  | pp
