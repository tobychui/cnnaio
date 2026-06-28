#!/usr/bin/env bash
# Object detection — NanoDet-Plus-m (COCO-80, smaller/faster).
# POST /v1/images/detections   (note nanodet's task defaults: score 0.4, nms 0.5)
source "$(dirname "$0")/_common.sh"

curl -sS -X POST "$BASE_URL/v1/images/detections" "${auth_args[@]}" \
  -F image=@"$IMAGE" \
  -F model=nanodet-plus-m \
  -F score_threshold=0.4 \
  -F nms_threshold=0.5 \
  -F max_results=100 \
  -F render=false \
  -F async=false \
  | pp
