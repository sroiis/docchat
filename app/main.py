"""FastAPI HTTP layer.

This is the part that looks like your day job: a service with clean endpoints
over a domain engine. All the AI logic lives in rag.py; this file just does
request/response plumbing and lifecycle.

Endpoints
---------
GET  /health          -> liveness + how many chunks are indexed
GET  /docs            -> which documents are indexed and their chunk counts
POST /ingest          -> (re)index a directory, or raw docs passed in the body
POST /ask             -> ask a question, get an answer + sources

Run it:   uvicorn app.main:app --reload
Then open http://127.0.0.1:8000/  (interactive API docs at /swagger)
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .config import settings
from .rag import RagEngine

engine = RagEngine()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Auto-index the sample docs on startup so the service is useful immediately.
    if os.path.isdir(settings.docs_dir):
        count = engine.ingest_directory(settings.docs_dir)
        print(f"[docchat] indexed {count} chunks from '{settings.docs_dir}'")
    else:
        print(f"[docchat] docs dir '{settings.docs_dir}' not found; starting empty")
    yield


app = FastAPI(
    title="docchat",
    version="1.0.0",
    description="A fully-offline 'chat with your docs' RAG service.",
    docs_url="/swagger",   # interactive API explorer
    lifespan=lifespan,
)


# ---- request/response models ---------------------------------------------


class AskRequest(BaseModel):
    question: str = Field(..., examples=["How does chunking work?"])
    k: int = Field(default=settings.default_k, ge=1, le=20)


class IngestRequest(BaseModel):
    # Either point at a directory...
    directory: Optional[str] = Field(default=None, examples=["sample_docs"])
    # ...or pass documents inline as {filename: content}.
    documents: Optional[dict[str, str]] = None


# ---- endpoints ------------------------------------------------------------


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "indexed_chunks": engine.store.size}


@app.get("/docs")
def list_docs() -> dict:
    return {"documents": engine.store.documents(), "total_chunks": engine.store.size}


@app.post("/ingest")
def ingest(req: IngestRequest) -> dict:
    if req.documents:
        count = engine.ingest_texts(req.documents)
        source = "inline documents"
    else:
        directory = req.directory or settings.docs_dir
        count = engine.ingest_directory(directory)
        source = directory
    return {"indexed_chunks": count, "source": source}


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    answer = engine.ask(req.question, k=req.k)
    return answer.__dict__


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    # A minimal built-in UI so you can try it without curl or Swagger.
    return _INDEX_HTML


_INDEX_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>docchat</title>
  <style>
    :root { color-scheme: light dark; }
    body { font-family: system-ui, sans-serif; max-width: 760px; margin: 40px auto;
           padding: 0 16px; line-height: 1.5; }
    h1 { margin-bottom: 4px; }
    .sub { opacity: .7; margin-top: 0; }
    input { width: 100%; padding: 12px; font-size: 16px; box-sizing: border-box; }
    button { margin-top: 10px; padding: 10px 18px; font-size: 16px; cursor: pointer; }
    .hit { border: 1px solid #8884; border-radius: 8px; padding: 12px; margin: 12px 0; }
    .score { font-size: 13px; opacity: .7; }
    pre { white-space: pre-wrap; }
  </style>
</head>
<body>
  <h1>docchat</h1>
  <p class="sub">Ask a question about the indexed documents. Fully offline.</p>
  <input id="q" placeholder="e.g. How does TF-IDF decide relevance?"
         onkeydown="if(event.key==='Enter')run()" autofocus />
  <button onclick="run()">Ask</button>
  <div id="out"></div>
  <script>
    async function run() {
      const q = document.getElementById('q').value.trim();
      const out = document.getElementById('out');
      if (!q) return;
      out.innerHTML = 'Searching…';
      const res = await fetch('/ask', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({question: q, k: 4})
      });
      const data = await res.json();
      let html = '<h3>Answer</h3><pre>' + escapeHtml(data.answer) + '</pre>';
      if (data.sources && data.sources.length) {
        html += '<h3>Sources</h3>';
        for (const s of data.sources) {
          html += '<div class="hit"><div class="score">' + s.doc_id +
                  ' — relevance ' + (s.score*100).toFixed(1) + '%</div><pre>' +
                  escapeHtml(s.text) + '</pre></div>';
        }
      }
      out.innerHTML = html;
    }
    function escapeHtml(t){return t.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
  </script>
</body>
</html>
"""
