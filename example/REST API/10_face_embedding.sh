#!/usr/bin/env bash
# Face embedding — MobileFaceNet (L2-normalized 128-d vector per face).
# POST /v1/faces/embeddings
# cropped=false -> embed the largest face in the photo; true -> input is a face.
source "$(dirname "$0")/_common.sh"

curl -sS -X POST "$BASE_URL/v1/faces/embeddings" "${auth_args[@]}" \
  -F image=@"$IMAGE" \
  -F model=mbv2facenet \
  -F cropped=false \
  -F render=false \
  -F async=false \
  | pp
