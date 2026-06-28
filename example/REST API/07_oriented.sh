#!/usr/bin/env bash
# Oriented object detection (OBB) — YOLO11n-obb (DOTA-15).
# POST /v1/images/oriented   (aerial/top-down imagery only; expect 0 on photos)
source "$(dirname "$0")/_common.sh"

curl -sS -X POST "$BASE_URL/v1/images/oriented" "${auth_args[@]}" \
  -F image=@"$IMAGE" \
  -F model=yolo11n-obb \
  -F score_threshold=0.25 \
  -F nms_threshold=0.45 \
  -F max_results=100 \
  -F render=false \
  -F async=false \
  | pp
