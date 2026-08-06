"""API router for docchat (all endpoints live under /api)."""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from . import db
from .auth import (
    create_access_token,
    get_current_user,
    hash_password,
    verify_password,
)
from .config import settings
from .rag import RagEngine

router = APIRouter(prefix="/api")

# Per-user engine cache. RagEngine.load() reads from the DB, so this is just
# an optimisation — recreated on ingest/delete.
_engines: dict[int, RagEngine] = {}


def _engine(user_id: int) -> RagEngine:
    engine = _engines.get(user_id)
    if engine is None:
        engine = RagEngine()
        engine.load(user_id)
        engine.generator = None
        _engines[user_id] = engine
    return engine


def _invalidate(user_id: int) -> None:
    _engines.pop(user_id, None)


# ---- schemas ----------------------------------------------------------------


class RegisterRequest(BaseModel):
    email: str = Field(..., examples=["you@example.com"])
    password: str = Field(..., min_length=8)


class LoginRequest(BaseModel):
    email: str
    password: str


class AskRequest(BaseModel):
    question: str = Field(..., examples=["How does TF-IDF decide relevance?"])
    k: int = Field(default=settings.default_k, ge=1, le=20)


class IngestRequest(BaseModel):
    documents: dict[str, str] | None = None


# ---- auth -------------------------------------------------------------------


@router.post("/auth/register")
def register(req: RegisterRequest) -> dict:
    email = req.email.strip().lower()
    if db.get_user_by_email(email):
        raise HTTPException(status_code=409, detail="Email already registered")
    user_id = db.create_user(email, hash_password(req.password))
    return {"access_token": create_access_token(user_id), "token_type": "bearer"}


@router.post("/auth/login")
def login(req: LoginRequest) -> dict:
    email = req.email.strip().lower()
    user = db.get_user_by_email(email)
    if user is None or not verify_password(req.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"access_token": create_access_token(user["id"]), "token_type": "bearer"}


@router.get("/auth/me")
def me(user: dict = Depends(get_current_user)) -> dict:
    return user


# ---- documents ---------------------------------------------------------------


@router.get("/documents")
def list_docs(user: dict = Depends(get_current_user)) -> dict:
    return {"documents": _engine(user["id"]).documents(user["id"])}


@router.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    user: dict = Depends(get_current_user),
) -> dict:
    if not file.filename.lower().endswith((".md", ".txt")):
        raise HTTPException(status_code=400, detail="Only .md and .txt files")
    content = (await file.read()).decode("utf-8", errors="replace")
    if not content.strip():
        raise HTTPException(status_code=400, detail="Empty file")

    count = _engine(user["id"]).ingest_texts(user["id"], {file.filename: content})
    _invalidate(user["id"])
    return {"doc_id": file.filename, "indexed_chunks": count}


@router.delete("/documents/{doc_id}")
def delete_doc(doc_id: str, user: dict = Depends(get_current_user)) -> dict:
    _invalidate(user["id"])
    removed = _engine(user["id"]).delete_document(user["id"], doc_id)
    if not removed:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"deleted": doc_id}


@router.post("/ingest")
def ingest(req: IngestRequest, user: dict = Depends(get_current_user)) -> dict:
    if not req.documents:
        raise HTTPException(status_code=400, detail="Provide {doc_id: content}")
    count = _engine(user["id"]).ingest_texts(user["id"], req.documents)
    _invalidate(user["id"])
    return {"indexed_chunks": count, "source": "inline documents"}


# ---- Q&A --------------------------------------------------------------------


@router.post("/ask")
def ask(req: AskRequest, user: dict = Depends(get_current_user)) -> dict:
    engine = _engine(user["id"])
    return engine.ask(user["id"], req.question, k=req.k).__dict__


@router.post("/chat")
async def chat(req: AskRequest, user: dict = Depends(get_current_user)):
    engine = _engine(user["id"])
    prep = await engine.stream_answer(user["id"], req.question, k=req.k)
    sources = prep["sources"]

    async def event_stream():
        yield _sse({"type": "sources", "sources": sources})
        if not sources:
            yield _sse({"type": "delta",
                        "text": "No documents indexed yet. Upload files first."})
            yield _sse({"type": "done"})
            return

        generator = prep["generator"]
        system, prompt = RagEngine.build_prompt(req.question, sources)
        async for token in generator.stream(system, prompt):
            yield _sse({"type": "delta", "text": token})
        yield _sse({"type": "done"})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"
