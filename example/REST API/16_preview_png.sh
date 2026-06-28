#!/usr/bin/env bash
# Preview mode — get the annotated PNG back instead of JSON.
# Add ?preview (or Accept: image/png) to any single-image detection-like request.
# Here: detection boxes rendered onto the image, saved to ./preview.png.
source "$(dirname "$0")/_common.sh"

out="preview.png"
curl -sS -X POST "$BASE_URL/v1/images/detections?preview" "${auth_args[@]}" \
  -F image=@"$IMAGE" \
  -F model=yolo11n \
  -F score_threshold=0.25 \
  -F nms_threshold=0.45 \
  -o "$out"

echo "wrote $out"
