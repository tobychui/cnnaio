#!/usr/bin/env bash
# Pose estimation — YOLO11n-pose (17 COCO keypoints per person).
# POST /v1/images/poses
source "$(dirname "$0")/_common.sh"

curl -sS -X POST "$BASE_URL/v1/images/poses" "${auth_args[@]}" \
  -F image=@"$IMAGE" \
  -F model=yolo11n-pose \
  -F score_threshold=0.25 \
  -F nms_threshold=0.45 \
  -F max_results=100 \
  -F render=false \
  -F async=false \
  | pp
