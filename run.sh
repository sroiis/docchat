#!/usr/bin/env bash
# One command to get running. Creates a virtualenv, installs deps, builds the
# frontend, and starts the server.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  echo "[docchat] creating virtualenv..."
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

echo "[docchat] installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

if [ ! -d frontend/node_modules ]; then
  echo "[docchat] installing frontend dependencies..."
  (cd frontend && npm install)
fi
if [ ! -d frontend/dist ]; then
  echo "[docchat] building frontend..."
  (cd frontend && npm run build)
fi

echo "[docchat] starting server on http://127.0.0.1:8000"
echo "[docchat] demo login: demo@docchat.local / demo   (API explorer: /swagger)"
exec uvicorn app.main:app --reload --port 8000
