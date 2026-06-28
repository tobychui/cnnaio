#!/usr/bin/env bash
# Run every numbered example script in order against a running server.
# Usage:  ./run_all.sh        (start the server first: `go run . -dev`)
set -euo pipefail
cd "$(dirname "$0")"

for s in [0-9][0-9]_*.sh; do
  echo
  echo "============================================================"
  echo "  $s"
  echo "============================================================"
  bash "$s" || echo "($s exited non-zero)"
done
