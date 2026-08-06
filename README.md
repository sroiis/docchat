# docchat 🗂️💬

A **fully-offline "chat with your documents" service** — built for backend/infra
engineers who want to understand how modern AI search (RAG) actually works,
without any ML background, model downloads, GPUs, or API keys.

You drop in text/markdown files, and it answers questions about them by
returning the most relevant passages with a relevance score and source. It's
shaped like a normal backend service: a FastAPI app over an ingest pipeline and
a vector store.

## Why this project (if you're new to AI)

Most "AI features" companies ask backend engineers to build are **RAG**:
Retrieval-Augmented Generation. The hard, interesting part — chunking,
embeddings, a vector store, low-latency search — is *backend work*, not math.
This project implements that part from scratch in ~400 readable lines so you can
see there's no magic. The optional last step (calling an LLM to phrase the
answer) is described but not required.

## Quick start

```bash
# 1. unzip, then:
cd docchat
./run.sh
```

That creates a virtualenv, installs deps, and starts the server on
`http://127.0.0.1:8000`. Open it in a browser and ask a question — the sample
docs (which explain RAG, embeddings, and this architecture) are indexed
automatically, so you can immediately ask things like:

- *"How does TF-IDF decide relevance?"*
- *"Where would an LLM plug in?"*
- *"What is cosine similarity?"*

Interactive API explorer: `http://127.0.0.1:8000/swagger`

### Prefer manual steps?

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Try the API with curl

```bash
# Ask a question
curl -s localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question":"how does chunking work?","k":3}' | python3 -m json.tool

# See what's indexed
curl -s localhost:8000/docs | python3 -m json.tool

# Index your own docs (a directory of .md/.txt files)
curl -s localhost:8000/ingest -H 'content-type: application/json' \
  -d '{"directory":"/path/to/your/notes"}'

# ...or pass documents inline
curl -s localhost:8000/ingest -H 'content-type: application/json' \
  -d '{"documents":{"note.md":"my important text here"}}'
```

## Use it on YOUR documents

Point it at any folder of `.md`/`.txt` files:

```bash
DOCCHAT_DOCS_DIR=/path/to/your/notes ./run.sh
```

Great for: your team's runbooks, onboarding docs, meeting notes, or an
engineering wiki export.

## How it works (read the code in this order)

| Step | File | What it does |
|------|------|--------------|
| 1 | `app/chunking.py`   | Split docs into overlapping word windows |
| 2 | `app/embeddings.py` | Turn text into TF-IDF vectors (pluggable interface) |
| 3 | `app/store.py`      | Hold vectors, do nearest-neighbour search |
| 4 | `app/rag.py`        | Ingest + query pipeline |
| 5 | `app/main.py`       | FastAPI endpoints + a tiny built-in web UI |

## Run the tests

```bash
source .venv/bin/activate   # if not already active
pytest -q
```

The tests double as documentation for each component.

## Level-up ideas (when you're ready)

- **Add real semantic search:** implement the `Embedder` interface with
  `sentence-transformers`. Nothing else changes — that's the payoff of the
  interface seam in `embeddings.py`.
- **Add the generation step:** in `rag.py`, take the top chunks, build a prompt,
  and call a local LLM (Ollama) or the Claude API to write a natural-language
  answer. That turns retrieval-only into full RAG.
- **Swap the store for a real vector DB:** replace `VectorStore` with FAISS or
  pgvector to handle millions of chunks.
- **Persist the index:** currently rebuilt on startup; save vectors to disk.

## Layout

```
docchat/
├── app/
│   ├── chunking.py     # document -> chunks
│   ├── embeddings.py   # text -> vectors (TF-IDF)
│   ├── store.py        # vector store + search
│   ├── rag.py          # ingest + ask pipeline
│   ├── config.py       # env-var settings
│   └── main.py         # FastAPI app + web UI
├── sample_docs/        # auto-indexed on startup
├── tests/              # pytest
├── requirements.txt
├── run.sh              # one-command start
└── README.md
```

No external services. No API keys. No data leaves your machine.
