#!/usr/bin/env bash
# List available models (and their task/classes/input size).
# GET /v1/models          — all models
# GET /v1/models/{id}     — one model (uncomment below)
source "$(dirname "$0")/_common.sh"

curl -sS "$BASE_URL/v1/models" "${auth_args[@]}" | pp

# curl -sS "$BASE_URL/v1/models/yolo11n" "${auth_args[@]}" | pp
