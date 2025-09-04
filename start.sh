#!/usr/bin/env bash
set -e

MODE="${MODE:-web}"   # train | test | web
echo "[start.sh] MODE=$MODE"

if [ "$MODE" = "train" ]; then
  python -m src.train
elif [ "$MODE" = "test" ]; then
  pytest -q
elif [ "$MODE" = "web" ]; then
  uvicorn app.main:app --host 0.0.0.0 --port 8000
else
  echo "Unknown MODE: $MODE (expect train|test|web)"
  exit 1
fi
