#!/usr/bin/env bash
# Instance segmentation — YOLO11n-seg (COCO-80).
# POST /v1/images/segmentations   (each item carries a per-instance mask PNG)
source "$(dirname "$0")/_common.sh"

curl -sS -X POST "$BASE_URL/v1/images/segmentations" "${auth_args[@]}" \
  -F image=@"$IMAGE" \
  -F model=yolo11n-seg \
  -F score_threshold=0.25 \
  -F nms_threshold=0.45 \
  -F max_results=100 \
  -F render=false \
  -F async=false \
  | pp
