"""SQLite persistence layer.

The original docchat was fully in-memory: every restart wiped your index.
A real product persists data. This module is the single place that talks to
SQLite — users, documents, and the vector index live here.

Design notes
------------
- One database file (config.db_path), created lazily.
- Every operation opens a short-lived connection (fine for this scale,
  keeps it thread-safe for FastAPI's threadpool).
- Documents are stored per-user; nothing is ever shared across users.
- Vectors are stored as BLOBs next to the chunk text, so a fresh start can
  rebuild every user's index without recomputing embeddings.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from .config import settings

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    email         TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    doc_id     TEXT NOT NULL,
    file_name  TEXT NOT NULL,
    content    TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (user_id, doc_id)
);

CREATE TABLE IF NOT EXISTS chunks (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id  INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    doc_id   TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    "order"  INTEGER NOT NULL,
    text     TEXT NOT NULL,
    vector   BLOB NOT NULL,
    UNIQUE (user_id, chunk_id)
);

CREATE INDEX IF NOT EXISTS idx_documents_user ON documents(user_id);
CREATE INDEX IF NOT EXISTS idx_chunks_user ON chunks(user_id);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(user_id, doc_id);
"""


# --- connection helpers ------------------------------------------------------


def _connect() -> sqlite3.Connection:
    _ensure_schema()
    conn = sqlite3.connect(settings.db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _now() -> str:
    return datetime.now(UTC).isoformat()


_initialized = False


def _ensure_schema() -> None:
    """Create the data directory and schema once per process, lazily.

    Called on every connect so the library is safe to use directly (e.g. in
    tests and scripts) without going through the FastAPI lifespan.
    """
    global _initialized
    if _initialized:
        return
    Path(settings.db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.db_path)
    try:
        conn.executescript(_SCHEMA)
    finally:
        conn.close()
    _initialized = True


def init_db() -> None:
    """Public alias: force schema creation now (used by the app lifespan)."""
    _ensure_schema()


# --- vector helpers -----------------------------------------------------------


def _pack(vector: np.ndarray) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes()


def _unpack(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


# --- users -------------------------------------------------------------------


def create_user(email: str, password_hash: str) -> int:
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO users (email, password_hash, created_at) VALUES (?, ?, ?)",
            (email, password_hash, _now()),
        )
        return int(cur.lastrowid)


def get_user_by_email(email: str) -> sqlite3.Row | None:
    with _connect() as conn:
        return conn.execute(
            "SELECT * FROM users WHERE email = ?", (email,)
        ).fetchone()


def get_user_by_id(user_id: int) -> sqlite3.Row | None:
    with _connect() as conn:
        return conn.execute(
            "SELECT * FROM users WHERE id = ?", (user_id,)
        ).fetchone()


def get_or_create_demo_user(email: str, password_hash: str) -> sqlite3.Row:
    existing = get_user_by_email(email)
    if existing:
        return existing
    uid = create_user(email, password_hash)
    return get_user_by_id(uid)


# --- documents ---------------------------------------------------------------


def add_document(user_id: int, doc_id: str, file_name: str, content: str) -> None:
    """Register a document row. Idempotent: same (user, doc_id) = update."""
    with _connect() as conn:
        conn.execute(
            "INSERT INTO documents (user_id, doc_id, file_name, content, created_at) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(user_id, doc_id) "
            "DO UPDATE SET file_name = excluded.file_name, content = excluded.content",
            (user_id, doc_id, file_name, content, _now()),
        )


def delete_document(user_id: int, doc_id: str) -> bool:
    with _connect() as conn:
        cur = conn.execute(
            "DELETE FROM documents WHERE user_id = ? AND doc_id = ?",
            (user_id, doc_id),
        )
        conn.execute(
            "DELETE FROM chunks WHERE user_id = ? AND doc_id = ?",
            (user_id, doc_id),
        )
        return cur.rowcount > 0


def list_documents(user_id: int) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT d.doc_id, d.file_name, d.created_at, "
            "COUNT(c.id) AS chunk_count "
            "FROM documents d LEFT JOIN chunks c "
            "ON c.user_id = d.user_id AND c.doc_id = d.doc_id "
            "WHERE d.user_id = ? "
            "GROUP BY d.doc_id ORDER BY d.created_at ASC",
            (user_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def list_all_doc_ids(user_id: int) -> list[str]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT doc_id FROM documents WHERE user_id = ? ORDER BY doc_id",
            (user_id,),
        ).fetchall()
    return [r["doc_id"] for r in rows]


def list_all_docs(user_id: int) -> dict[str, str]:
    """Map of doc_id -> content for all of a user's documents."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT doc_id, content FROM documents WHERE user_id = ? ORDER BY doc_id",
            (user_id,),
        ).fetchall()
    return {r["doc_id"]: r["content"] for r in rows}


# --- chunks ------------------------------------------------------------------


def replace_chunks(user_id: int, chunks: list[dict]) -> None:
    """Replace a user's entire index with the given chunk list.

    Each chunk dict: {doc_id, chunk_id, order, text, vector: np.ndarray}.
    Called after a full re-embed of a user's corpus.
    """
    with _connect() as conn:
        conn.execute("DELETE FROM chunks WHERE user_id = ?", (user_id,))
        conn.executemany(
            "INSERT INTO chunks (user_id, doc_id, chunk_id, \"order\", text, vector) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [
                (
                    user_id,
                    c["doc_id"],
                    c["chunk_id"],
                    c["order"],
                    c["text"],
                    _pack(c["vector"]),
                )
                for c in chunks
            ],
        )


def load_chunks(user_id: int) -> list[dict]:
    """Load every chunk+vector for a user, ordered by doc/position."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT doc_id, chunk_id, \"order\" AS ord, text, vector "
            "FROM chunks WHERE user_id = ? "
            "ORDER BY doc_id, \"order\" ASC",
            (user_id,),
        ).fetchall()
    return [
        {
            "doc_id": r["doc_id"],
            "chunk_id": r["chunk_id"],
            "order": r["ord"],
            "text": r["text"],
            "vector": _unpack(r["vector"]),
        }
        for r in rows
    ]


def chunk_count(user_id: int) -> int:
    with _connect() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM chunks WHERE user_id = ?", (user_id,)
        ).fetchone()
    return int(row["n"]) if row else 0
