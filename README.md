# docchat 🗂️💬

**Chat with your documents.** A production-grade RAG (Retrieval-Augmented
Generation) service — FastAPI backend + React frontend — that indexes your
`.md`/`.txt` files and answers questions with **cited sources**, in real time.

- **Offline by default**: runs entirely on your machine with no API keys
- **Multi-user**: JWT auth, every user's documents are private to them
- **Persistent**: your index survives restarts (SQLite)
- **Pluggable**: swap embeddings (TF-IDF / OpenAI / local) and the LLM
  (none / Ollama / OpenAI-compatible) with one env var
- **Streaming**: answers token-by-token over SSE
- **One command**: `docker compose up` gives you the whole product

```bash
docker compose up --build
# → http://localhost:8000   (demo login: demo@docchat.local / demo)
```

---

## Quick start (local, no Docker)

```bash
python3 -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cd frontend && npm install && npm run build && cd ..  # build the web UI
uvicorn app.main:app --reload
```

Open **http://127.0.0.1:8000**, log in, upload a document, ask a question.
Interactive API explorer: http://127.0.0.1:8000/swagger

> No account yet? Register in the UI, or start a key-less demo with
> `DOCCHAT_AUTH_ENABLED=false`.

## Features

### Full RAG, not just retrieval
docchat implements *both* halves of RAG:

1. **Retrieve** — documents are chunked, embedded, and searched with cosine
   similarity (see `app/chunking.py`, `app/embeddings.py`, `app/store.py`).
2. **Generate** — top chunks are fed to an LLM that writes a grounded answer
   with citations (`app/generator.py`).

### Three embedding backends

| Provider | Env value | Notes |
|----------|-----------|-------|
| TF-IDF (default) | `tfidf` | Offline, zero dependencies, explainable |
| OpenAI | `openai` | Any `/embeddings`-compatible API |
| Local neural | `local` | sentence-transformers (`pip install -r requirements-optional.txt`) |

### Three LLM backends

| Provider | Env value | Notes |
|----------|-----------|-------|
| None (default) | `none` | Retrieval-only synthesis, no keys needed |
| Ollama | `ollama` | Local LLM, fully offline |
| OpenAI-compatible | `openai` | OpenAI, OpenRouter, vLLM, LM Studio… |

### Streaming chat
`POST /api/chat` returns a Server-Sent-Events stream:
`sources` → `delta` tokens → `done`. The React UI renders it live.

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/auth/register` | Create an account |
| POST | `/api/auth/login` | Get a bearer token |
| GET | `/api/auth/me` | Current user |
| GET | `/api/documents` | List the caller's documents |
| POST | `/api/documents/upload` | Upload & index a `.md`/`.txt` file |
| DELETE | `/api/documents/{doc_id}` | Remove a document |
| POST | `/api/ingest` | Index inline `{filename: content}` |
| POST | `/api/ask` | One-shot answer + sources |
| POST | `/api/chat` | Streaming answer (SSE) |
| GET | `/health` | Health + provider info |

Try it:

```bash
# register
TOKEN=$(curl -s -X POST localhost:8000/api/auth/register \
  -H 'content-type: application/json' \
  -d '{"email":"you@example.com","password":"password123"}' | jq -r .access_token)

# ask a question
curl -s localhost:8000/api/ask -H "authorization: Bearer $TOKEN" \
  -H 'content-type: application/json' \
  -d '{"question":"How does TF-IDF decide relevance?","k":3}'

# stream a chat answer
curl -sN localhost:8000/api/chat -H "authorization: Bearer $TOKEN" \
  -H 'content-type: application/json' \
  -d '{"question":"What is RAG?","k":3}'
```

## Configuration

Everything is an env var with the `DOCCHAT_` prefix. See `.env.example` and
`docs/deployment.md` for the full list. Key knobs:

```bash
DOCCHAT_AUTH_ENABLED=true
DOCCHAT_SECRET_KEY=change-me-in-production
DOCCHAT_EMBEDDING_PROVIDER=tfidf      # tfidf | openai | local
DOCCHAT_LLM_PROVIDER=none             # none | ollama | openai
DOCCHAT_LLM_MODEL=llama3.1
DOCCHAT_OLLAMA_BASE_URL=http://localhost:11434
```

## How it works (read the code in this order)

| Step | File | What it does |
|------|------|--------------|
| 1 | `app/chunking.py` | Split docs into overlapping word windows |
| 2 | `app/embeddings.py` | Text → vectors (pluggable: TF-IDF / OpenAI / local) |
| 3 | `app/store.py` | Vector store + nearest-neighbour search |
| 4 | `app/generator.py` | LLM generation (none / Ollama / OpenAI) |
| 5 | `app/rag.py` | Per-user ingest + retrieve + generate pipeline |
| 6 | `app/db.py` | SQLite persistence |
| 7 | `app/auth.py` | JWT auth + password hashing |
| 8 | `app/api.py` | All `/api` endpoints |
| 9 | `app/main.py` | App wiring + static frontend |
| 10 | `frontend/` | React + Vite SPA |

## Tests

```bash
pytest -q          # 22 tests: unit + API + auth + generator
```

The suite doubles as documentation for each component. CI runs lint
(`ruff`), backend tests, and a strict frontend build on every push.

## Deployment

See **[docs/deployment.md](docs/deployment.md)** — Docker, Render, Fly.io, and
production checklist (persistent volume, real secret key, health checks).

## Architecture

See **[docs/architecture.md](docs/architecture.md)** — the design decisions
behind the pluggable interfaces, multi-tenancy, persistence, and streaming.

## Repository layout

```
app/                FastAPI backend (RAG + auth + persistence)
frontend/           React SPA (built into frontend/dist, served by FastAPI)
sample_docs/        demo documents indexed for the demo user
tests/              pytest suite
docs/               architecture & deployment guides
Dockerfile          multi-stage (build SPA → run Python app)
docker-compose.yml  one-command local deployment
.github/workflows/  CI (lint + tests + frontend build)
.env.example        all configuration knobs
```
