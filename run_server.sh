#!/usr/bin/env bash
# Launch the FairLens FastAPI server.
#   bash run_server.sh
cd "$(dirname "$0")"
exec uv run uvicorn server:app --host 127.0.0.1 --port 8000
