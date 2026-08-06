# Deployment

docchat ships ready to run anywhere a Docker container can run, and the
frontend is served by the same process as the API (one container, one port).

The database is portable: SQLite locally, Postgres in production, chosen by
`DOCCHAT_DATABASE_URL`. See [Database](#database) below.

## Deploy free on Render (recommended for "anyone can use")

[render.yaml](../render.yaml) provisions a free web service + free Postgres.
No credit card. Three steps:

1. Push this repo to GitHub (public).
2. On Render: **New → Blueprint** → connect the repo.
3. Render creates the Postgres DB, wires `DOCCHAT_DATABASE_URL`, and deploys.

Free-tier caveats: the web service **sleeps after 15 min idle** (first visit
cold-starts in ~30–60 s), and the free Postgres **expires after 90 days**
(delete + recreate the DB to refresh, or upgrade).

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
| `DOCCHAT_DATABASE_URL` | (none) | Set to a Postgres URL to use Postgres; empty = SQLite at `DOCCHAT_DB_PATH` |
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

### Deploying to Render (Blueprint)
The recommended path — see the top of this page. Render's `render.yaml`
creates a free Postgres and wires `DOCCHAT_DATABASE_URL` automatically, so
data survives restarts (no mounted disk needed).

### Deploying to Render (manual Web Service)
1. Push the repo to GitHub.
2. On Render, create a **Web Service**, connect the repo.
3. Build command: `docker build -t docchat .` (Render supports the Dockerfile
   automatically when present).
4. Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
5. Create a **Postgres** database in Render and set
   `DOCCHAT_DATABASE_URL` to its internal connection string.
6. Add env vars: a real `DOCCHAT_SECRET_KEY`, `DOCCHAT_EMBEDDING_PROVIDER=tfidf`,
   `DOCCHAT_LLM_PROVIDER=none` (or `ollama`/`openai`).
7. Set `DOCCHAT_ALLOWED_ORIGINS` to your app's public URL.

### Deploying to Fly.io
`flyctl launch` will detect the Dockerfile. Configure secrets:
```bash
fly secrets set DOCCHAT_SECRET_KEY="$(openssl rand -hex 32)"
fly secrets set DOCCHAT_LLM_PROVIDER=ollama   # if you add an Ollama machine
fly postgres create --name docchat-db
fly secrets set DOCCHAT_DATABASE_URL="<connection string from fly postgres>"
```

### Any platform with Postgres
Set `DOCCHAT_DATABASE_URL` to any Postgres, `DOCCHAT_SECRET_KEY` to a long
random string, keep `DOCCHAT_AUTH_ENABLED=true`. Multiple instances share one
database safely.

## Database

The app is portable across SQLite and Postgres through one table-definition
layer (`app/db.py`):

- **Local dev / small self-hosts:** nothing to configure — SQLite is used at
  `DOCCHAT_DB_PATH` (default `data/docchat.db`).
- **Production:** set `DOCCHAT_DATABASE_URL` to a Postgres URL, e.g.
  `postgresql://user:pass@host:5432/docchat`. Tables are created on startup.

`postgres://` URLs (as Render/Fly emit) are normalised automatically. Install
the Postgres driver in production: `pip install -r requirements-prod.txt`
(the Docker image already includes it).

Chunks and their embedding vectors are stored as rows, so a restart restores
every user's index without recomputing embeddings.

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
