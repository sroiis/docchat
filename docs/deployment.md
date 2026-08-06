# Deployment

docchat ships ready to run anywhere a Docker container can run, and the
frontend is served by the same process as the API (one container, one port).

## Quick local start (Docker)

```bash
docker compose up --build
```

- App:  http://localhost:8000
- Swagger: http://localhost:8000/swagger

On first start a **demo user** is created (`demo@docchat.local` / `demo`) and
the sample docs are indexed, so you can ask questions immediately.

## Local without Docker

```bash
python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cd frontend && npm install && npm run build && cd ..  # optional but recommended
uvicorn app.main:app --reload
```

`DOCCHAT_AUTH_ENABLED=false` removes the login requirement entirely for local
tinkering (requests act as the demo user).

## Configuration

All settings are environment variables with the `DOCCHAT_` prefix. See
`.env.example` for the full list with comments. The important ones:

| Variable | Default | Purpose |
|----------|---------|---------|
| `DOCCHAT_AUTH_ENABLED` | `true` | Turn off for a key-less demo |
| `DOCCHAT_SECRET_KEY` | dev value | **Change in production** — signs JWTs |
| `DOCCHAT_EMBEDDING_PROVIDER` | `tfidf` | `tfidf` \| `openai` \| `local` |
| `DOCCHAT_LLM_PROVIDER` | `none` | `none` \| `ollama` \| `openai` |
| `DOCCHAT_LLM_MODEL` | `llama3.1` | Model used for generation |
| `DOCCHAT_OLLAMA_BASE_URL` | `http://localhost:11434` | Local LLM endpoint |
| `DOCCHAT_OPENAI_API_KEY` | — | Key for OpenAI-compatible APIs |
| `DOCCHAT_DB_PATH` | `data/docchat.db` | SQLite file location |
| `DOCCHAT_SEED_DEMO_USER` | `true` | Create demo user + sample docs |

### Running the demo user's credentials
`demo@docchat.local` / `demo`. If you disabled seeding, register via the UI.

## LLM backends

### Offline (default, no keys)
Set `DOCCHAT_LLM_PROVIDER=none`. Retrieval-only answers with sources — nothing
leaves your machine.

### Local LLM (Ollama)
```bash
ollama pull llama3.1 && ollama serve
```
then set:
```
DOCCHAT_LLM_PROVIDER=ollama
DOCCHAT_OLLAMA_BASE_URL=http://localhost:11434   # or host.docker.internal inside Docker
```
Keep embeddings on `tfidf` or set `local` (needs `pip install -r
requirements-optional.txt`).

### OpenAI-compatible API
```
DOCCHAT_LLM_PROVIDER=openai
DOCCHAT_OPENAI_API_KEY=sk-...
# optional: DOCCHAT_OPENAI_BASE_URL for proxies (OpenAI-compatible)
DOCCHAT_EMBEDDING_PROVIDER=openai
DOCCHAT_EMBEDDING_MODEL=text-embedding-3-small
```
Any provider speaking the OpenAI HTTP surface (OpenAI, OpenRouter, Together,
LM Studio, local vLLM…) works.

## Production deployment

### Deploying to Render
1. Push the repo to GitHub.
2. On Render, create a **Web Service**, connect the repo.
3. Build command: `docker build -t docchat .` (Render supports the Dockerfile
   automatically when present).
4. Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
5. Add env vars: a real `DOCCHAT_SECRET_KEY`, and a mounted disk at
   `/app/data` with `DOCCHAT_DB_PATH=/app/data/docchat.db` so the index
   survives restarts.
6. Set `DOCCHAT_ALLOWED_ORIGINS` to your app's public URL.

### Deploying to Fly.io
`flyctl launch` will detect the Dockerfile. Configure secrets:
```bash
fly secrets set DOCCHAT_SECRET_KEY="$(openssl rand -hex 32)"
fly secrets set DOCCHAT_LLM_PROVIDER=ollama   # if you add an Ollama machine
fly volume create docchat_data --size 1
# set DOCCHAT_DB_PATH=/data/docchat.db and mount the volume at /data
```

### Any platform with a persistent volume
The same pattern everywhere: persist `DOCCHAT_DB_PATH`, set a real
`DOCCHAT_SECRET_KEY`, keep `DOCCHAT_AUTH_ENABLED=true`. If you need multiple
instances, move storage to Postgres + a real vector DB (see architecture notes).

## CI

`.github/workflows/ci.yml` runs on every push/PR to `main`:

- **Backend**: `pytest -q` (isolated temp DB)
- **Lint**: `ruff check app tests`
- **Frontend**: `npm ci && npm run build` (strict TypeScript)

Add the branch protection rule "Require status checks to pass" on GitHub to
make green CI a merge gate.

## Health check

`GET /health` returns status, version, and enabled providers — wire it to your
platform's healthcheck:
```
curl http://localhost:8000/health
# {"status":"ok","version":"2.0.0",...,"users":1}
```
