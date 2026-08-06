"""FastAPI application entrypoint.

Splits concerns:
  - app/api.py   -> every /api endpoint
  - app/rag.py   -> the RAG domain logic
  - app/db.py    -> SQLite persistence
  - app/config.py-> env-driven settings
This module wires it together: the app object, lifecycle, CORS, the built
frontend, and a couple of non-API routes (/health, /swagger).

Run it:   uvicorn app.main:app --reload
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from . import db
from .api import router as api_router
from .config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    if settings.seed_demo_user:
        _seed_demo()
    yield


def _seed_demo() -> None:
    """Create a demo user and index the sample docs so the app is useful now."""
    from .auth import _demo_user_id
    from .rag import RagEngine

    user_id = _demo_user_id()
    engine = RagEngine()
    if os.path.isdir(settings.docs_dir):
        count = engine.ingest_directory(user_id, settings.docs_dir)
        print(f"[docchat] indexed {count} sample chunks for demo user")
    else:
        print(f"[docchat] docs dir '{settings.docs_dir}' not found; demo user empty")


app = FastAPI(
    title="docchat",
    version=settings.version,
    description="A production-grade 'chat with your documents' RAG service.",
    docs_url="/swagger",
    lifespan=lifespan,
)

_origins = [o.strip() for o in settings.allowed_origins.split(",") if o.strip()]
if _origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.include_router(api_router)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "version": settings.version,
        "auth_enabled": settings.auth_enabled,
        "embedding_provider": settings.embedding_provider,
        "llm_provider": settings.llm_provider,
        "users": _user_count(),
    }


def _user_count() -> int:
    try:
        with db._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS n FROM users").fetchone()
        return int(row["n"]) if row else 0
    except Exception:
        return 0


# ---- built frontend (optional) ---------------------------------------------

_INDEX_FALLBACK = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>docchat</title></head>
<body style="font-family:system-ui;max-width:760px;margin:60px auto;line-height:1.6">
<h1>docchat</h1>
<p>The frontend isn't built yet. Either build it
(<code>cd frontend &amp;&amp; npm install &amp;&amp; npm run build</code>) or use the
API directly: <a href="/swagger">/swagger</a></p>
</body></html>"""

_dist = settings.frontend_dist
if os.path.isdir(_dist):
    app.mount("/", StaticFiles(directory=_dist, html=True), name="frontend")
else:

    @app.get("/", response_class=HTMLResponse)
    def home() -> str:
        return _INDEX_FALLBACK
