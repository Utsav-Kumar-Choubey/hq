#!/usr/bin/env bash
# Start the FairLens server, then open http://127.0.0.1:8000
#   bash run_server.sh
cd "$(dirname "$0")"
PY=python3
[ -x .venv/bin/python ] && PY=.venv/bin/python
exec "$PY" -m uvicorn server:app --host 127.0.0.1 --port "${PORT:-8000}"
