#!/usr/bin/env bash
# Health check (always public — no auth required).
# GET /v1/health
source "$(dirname "$0")/_common.sh"

curl -sS "$BASE_URL/v1/health" | pp
