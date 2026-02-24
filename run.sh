#!/usr/bin/env bash
# run.sh — start the retinal disease classifier API
# Usage: bash run.sh [--port 8000] [--reload]
#
# Always run from the Team-B-Backend/ directory regardless of where
# this script is called from.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
RELOAD=""

for arg in "$@"; do
  case $arg in
    --reload) RELOAD="--reload" ;;
    --port=*) PORT="${arg#*=}" ;;
    --port)   shift; PORT="$1" ;;
  esac
done

echo "Starting Retinal Disease Classifier API"
echo "  Directory : $SCRIPT_DIR"
echo "  Host      : $HOST"
echo "  Port      : $PORT"
echo ""

exec uvicorn app.main:app \
  --app-dir "$SCRIPT_DIR" \
  --host "$HOST" \
  --port "$PORT" \
  $RELOAD

