# Architecture

docchat is a multi-tenant, persistent RAG service. This document explains the
system's shape and the reasoning behind each decision — the kind of thing that
reads well in a portfolio.

## System overview

```
                       ┌─────────────────────────────────────────────┐
   Browser (React SPA) │  FastAPI                                     │
  /api/chat (SSE) ────▶│                                              │
  /api/ask      ──────▶│  api.py ──▶ rag.py ──▶ embeddings.py         │
  /api/documents──────▶│      │         │         │   │               │
                       │      │         │         │   └─ tfidf/openai │
                       │      │         │         │                   │
                       │      │         └─▶ generator.py (LLM)        │
                       │      │              │  └─ none/ollama/openai │
                       │      │              │                        │
                       │      └─▶ db.py (SQLite: users, docs, chunks) │
                       └─────────────────────────────────────────────┘
```

One binary, one process. Everything runs in-process so a single `docker compose
up` gives you the whole product.

## Layering

| Module        | Responsibility                                             |
|---------------|------------------------------------------------------------|
| `app/config.py`  | Typed settings from `DOCCHAT_*` env vars (pydantic-settings) |
| `app/db.py`      | SQLite persistence: users, documents, chunks + vectors      |
| `app/auth.py`    | PBKDF2 password hashing, JWT issue/verify, FastAPI dep      |
| `app/embeddings.py` | `Embedder` interface + TF-IDF / OpenAI / local backends     |
| `app/generator.py` | `Generator` interface + none / Ollama / OpenAI backends     |
| `app/rag.py`     | Per-user RAG engine: ingest, retrieve, generate             |
| `app/api.py`     | All HTTP endpoints under `/api`                             |
| `app/main.py`    | App wiring, lifespan, CORS, static frontend                 |
| `frontend/`      | React + Vite SPA (chat UI, doc management, auth)            |

The two interfaces — `Embedder` and `Generator` — are the seams that make
docchat pluggable. The rest of the code never knows which backend is active.

## The RAG pipeline

1. **Ingest** (`rag.py`): a user's `.md`/`.txt` files are chunked into
   overlapping word windows (`chunking.py`), embedded, and stored in SQLite
   with their vectors.
2. **Retrieve** (`store.py`): a query is embedded with the *same* embedder and
   matched against all chunks via a matrix multiply — cosine similarity as a
   dot product over L2-normalised vectors. Top-k chunks are returned.
3. **Generate** (`generator.py`): the top chunks are assembled into a prompt
   and streamed through the configured LLM, or synthesised locally when no LLM
   is configured (`llm_provider=none`).

## Key design decisions

### Multi-tenancy by user
Every table is scoped by `user_id`, and the API resolves the caller from their
JWT. Users can never see each other's documents. `RagEngine` instances are
cached per user and rebuilt on ingest/delete.

### Persistent index
The original docchat rebuilt everything in memory at startup. Now chunks and
their embedding vectors are persisted in SQLite. A restart restores every
user's index without recomputing embeddings — important when embeddings are
paid API calls.

### TF-IDF by default, neural optional
`tfidf` runs anywhere with zero downloads and every result is explainable. It's
the default. Swap in `openai` or `local` (sentence-transformers) via one env
var; the interface makes the change invisible to the rest of the system.

### Hybrid retrieval (BM25 + dense, RRF)
TF-IDF is *lexical* — it matches words, not meaning. To fix the "car" vs
"automobile" gap, retrieval is now **hybrid** (`app/lexical.py`,
`app/search.py`): a BM25 sparse index runs alongside the dense vector store,
and their rankings are fused with reciprocal rank fusion (RRF, k=60). With a
neural embedder active this gives meaning-matching plus exact-term recall;
with TF-IDF it degrades to vector-only, so the default is unchanged.
`DOCCHAT_SEARCH_MODE=hybrid|sparse|dense` controls the path.

### Generation is a bolt-on
`llm_provider=none` returns a retrieval-only synthesis, so the whole product
works offline with no keys. `ollama` runs a local LLM; `openai` targets any
compatible API. Streaming is SSE over `POST /api/chat`, consumed natively by
the React frontend.

### Streaming, not polling
`/api/chat` returns `text/event-stream`. Each frame is a JSON payload:
`{"type":"sources",...}`, repeated `{"type":"delta","text":...}`, then
`{"type":"done"}`. The client renders tokens as they arrive.

### Dependency-light security
Passwords are hashed with PBKDF2-SHA256 (stdlib, per-user salt) and JWTs are
signed with HS256. No bcrypt C-extensions, no framework-embedded identity —
both are easy to audit.

## Data model

```sql
users      (id, email UNIQUE, password_hash, created_at)
documents  (id, user_id→users, doc_id, file_name, content, created_at)
chunks     (id, user_id→users, doc_id, chunk_id, pos, text, vector BLOB)
```

`documents.content` holds the source text (needed to re-chunk and re-fit
TF-IDF on ingest); `chunks.vector` stores the embedding so restarts are cheap.

## Failure & scale notes

- Single-process: the engine cache and SQLite assume one app instance. Scale
  horizontally by moving to Postgres + a shared vector store (documented as a
  path, not built here).
- Ingest re-embeds the user's whole corpus. Fine for personal/team docs; batch
  or incremental indexing would be the next step for large corpora.
- The auth secret (`DOCCHAT_SECRET_KEY`) must be a long random value in
  production; the demo default is for local use only.

## Repository layout

```
app/                backend (FastAPI + RAG)
frontend/           React SPA (built into frontend/dist, served by FastAPI)
sample_docs/        demo documents indexed for the demo user
tests/              pytest suite (unit + API + auth + generator)
docs/               architecture & deployment guides
Dockerfile          multi-stage: build SPA, then run Python app
docker-compose.yml  one-command local deploy
.github/workflows/  CI: lint + backend tests + frontend build
```
