#!/usr/bin/env bash
# Async mode — submit with async=true to get a 202 + job object, then poll
# GET /v1/jobs/{id} until status is "succeeded".
source "$(dirname "$0")/_common.sh"

# 1. Submit the job (async=true).
job="$(curl -sS -X POST "$BASE_URL/v1/images/detections" "${auth_args[@]}" \
  -F image=@"$IMAGE" \
  -F model=yolo11n \
  -F score_threshold=0.25 \
  -F nms_threshold=0.45 \
  -F max_results=100 \
  -F render=false \
  -F async=true)"
echo "submitted: $job"

# 2. Extract the job id (jq if available, else a crude grep).
if command -v jq >/dev/null 2>&1; then
  id="$(printf '%s' "$job" | jq -r '.id')"
else
  id="$(printf '%s' "$job" | grep -o '"id"[^,]*' | head -1 | cut -d'"' -f4)"
fi
[ -z "${id:-}" ] && { echo "could not read job id"; exit 1; }

# 3. Poll until done.
for _ in $(seq 1 30); do
  res="$(curl -sS "$BASE_URL/v1/jobs/$id" "${auth_args[@]}")"
  status="$(printf '%s' "$res" | { command -v jq >/dev/null 2>&1 && jq -r '.status' || grep -o '"status"[^,]*' | cut -d'"' -f4; })"
  echo "job $id: $status"
  case "$status" in
    succeeded|failed) printf '%s' "$res" | pp; exit 0 ;;
  esac
  sleep 1
done
echo "timed out waiting for job $id"
