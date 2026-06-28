#!/usr/bin/env bash
# Combined analysis — run several tasks over one image in a single round trip.
# POST /v1/vision/analyze   (JSON only — it carries a tasks array + per-task options)
# Valid tasks: classify, detect, segment, pose, oriented, faces, landmarks, gender.
source "$(dirname "$0")/_common.sh"

body="$(mktemp)"; trap 'rm -f "$body"' EXIT
cat > "$body" <<JSON
{
  "image": "$(data_uri "$IMAGE")",
  "tasks": ["classify", "detect", "faces", "landmarks", "gender"],
  "options": {
    "classify":  { "model": "mobilenet-v2", "top_k": 5 },
    "detect":    { "model": "yolo11n", "score_threshold": 0.25, "nms_threshold": 0.45, "max_results": 100 },
    "faces":     { "model": "ultraface-rfb-320", "score_threshold": 0.7, "nms_threshold": 0.3 },
    "landmarks": { "model": "pfld", "cropped": false },
    "gender":    { "model": "gender-mbv2-0.35", "cropped": false }
  },
  "render": false,
  "async": false
}
JSON

json_post "$BASE_URL/v1/vision/analyze" "$body"
