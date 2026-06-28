#!/usr/bin/env bash
# Object detection — YOLO11n (COCO-80).
# POST /v1/images/detections
source "$(dirname "$0")/_common.sh"

curl -sS -X POST "$BASE_URL/v1/images/detections" "${auth_args[@]}" \
  -F image=@"$IMAGE" \
  -F model=yolo11n \
  -F score_threshold=0.25 \
  -F nms_threshold=0.45 \
  -F max_results=100 \
  -F render=false \
  -F async=false \
  | pp
