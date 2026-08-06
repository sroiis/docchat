#!/usr/bin/env bash
# One command to get running. Creates a virtualenv, installs deps, starts server.
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

echo "[docchat] starting server on http://127.0.0.1:8000"
echo "[docchat] open that URL in your browser, or the API explorer at /swagger"
exec uvicorn app.main:app --reload --port 8000
