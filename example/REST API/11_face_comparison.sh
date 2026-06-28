#!/usr/bin/env bash
# Face comparison — MobileFaceNet cosine similarity between two images.
# POST /v1/faces/comparisons   (JSON only — it carries two images)
# By default both images are example/testdata/test.png, so same=true.
# Override:  IMAGE_A=a.jpg IMAGE_B=b.jpg ./11_face_comparison.sh
source "$(dirname "$0")/_common.sh"

body="$(mktemp)"; trap 'rm -f "$body"' EXIT
cat > "$body" <<JSON
{
  "model": "mbv2facenet",
  "image_a": "$(data_uri "$IMAGE_A")",
  "image_b": "$(data_uri "$IMAGE_B")",
  "a_cropped": false,
  "b_cropped": false,
  "threshold": 0.5
}
JSON

json_post "$BASE_URL/v1/faces/comparisons" "$body"
