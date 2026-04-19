#!/usr/bin/env bash
# Launches the OceanPulse FastAPI backend.
#
# Usage:
#   ./run.sh                  # bind 0.0.0.0:8000
#   PORT=8080 ./run.sh        # override port
#   HOST=127.0.0.1 ./run.sh   # override host
set -euo pipefail

cd "$(dirname "$0")"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

exec uvicorn app.main:app --host "$HOST" --port "$PORT" --reload
