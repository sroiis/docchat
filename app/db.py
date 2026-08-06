"""Persistence layer, backed by SQLite locally or Postgres in production.

The original docchat was fully in-memory: every restart wiped your index.
A real product persists data. This module is the single place that talks to
the database — users, documents, and the vector index live here.

Design notes
------------
- Uses SQLAlchemy Core so the same code runs on SQLite (dev) and Postgres
  (production). Which one is used is chosen by config.database_url:
      sqlite:///data/docchat.db          (local default)
      postgresql://user:pass@host/db     (Render / Fly / any Postgres)
- Every operation uses a short-lived connection (fine at this scale, keeps it
  thread-safe for FastAPI's threadpool).
- Documents are stored per-user; nothing is ever shared across users.
- Vectors are stored as BYTEA/BLOB next to the chunk text, so a fresh start can
  rebuild every user's index without recomputing embeddings.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import (
    Column,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
    create_engine,
    func,
    select,
)
from sqlalchemy import delete as sa_delete
from sqlalchemy.dialects import postgresql, sqlite

from .config import settings

metadata = MetaData()

users = Table(
    "users",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("email", String(320), nullable=False),
    Column("password_hash", String(255), nullable=False),
    Column("created_at", String(64), nullable=False),
    UniqueConstraint("email", name="uq_users_email"),
)

documents = Table(
    "documents",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("user_id", Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
    Column("doc_id", String(512), nullable=False),
    Column("file_name", String(512), nullable=False),
    Column("content", Text, nullable=False),
    Column("created_at", String(64), nullable=False),
    UniqueConstraint("user_id", "doc_id", name="uq_documents_user_doc"),
)

chunks = Table(
    "chunks",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("user_id", Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
    Column("doc_id", String(512), nullable=False),
    Column("chunk_id", String(1024), nullable=False),
    Column("pos", Integer, nullable=False),  # position within its document
    Column("text", Text, nullable=False),
    Column("vector", LargeBinary, nullable=False),
    UniqueConstraint("user_id", "chunk_id", name="uq_chunks_user_chunk"),
)

Index("idx_documents_user", documents.c.user_id)
Index("idx_chunks_user", chunks.c.user_id)
Index("idx_chunks_doc", chunks.c.user_id, chunks.c.doc_id)


def _db_url() -> str:
    """Resolve the engine URL: explicit database_url wins, else SQLite."""
    url = settings.database_url.strip()
    if url:
        # Render/Fly hand out postgres:// URLs; SQLAlchemy wants postgresql://
        return url.replace("postgres://", "postgresql://", 1)
    return f"sqlite:///{settings.db_path}"


engine = create_engine(_db_url(), pool_pre_ping=True)

# For SQLite (local dev) the file may not exist yet, so create tables on
# import. For Postgres, table creation happens in init_db() (app lifespan)
# to avoid assuming a live server at import time.
if engine.dialect.name == "sqlite":
    metadata.create_all(engine, checkfirst=True)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def init_db() -> None:
    """Create all tables if they don't exist."""
    metadata.create_all(engine)


# --- vector helpers -----------------------------------------------------------


def _pack(vector) -> bytes:
    import numpy as np

    return np.asarray(vector, dtype=np.float32).tobytes()


def _unpack(blob: bytes):
    import numpy as np

    return np.frombuffer(blob, dtype=np.float32)


# --- users -------------------------------------------------------------------


def create_user(email: str, password_hash: str) -> int:
    with engine.begin() as conn:
        result = conn.execute(
            users.insert().values(email=email, password_hash=password_hash, created_at=_now())
        )
        return int(result.inserted_primary_key[0])


def get_user_by_email(email: str) -> dict | None:
    with engine.connect() as conn:
        row = conn.execute(select(users).where(users.c.email == email)).mappings().first()
    return dict(row) if row else None


def get_user_by_id(user_id: int) -> dict | None:
    with engine.connect() as conn:
        row = conn.execute(select(users).where(users.c.id == user_id)).mappings().first()
    return dict(row) if row else None


def get_or_create_demo_user(email: str, password_hash: str):
    existing = get_user_by_email(email)
    if existing:
        return existing
    uid = create_user(email, password_hash)
    return get_user_by_id(uid)


def count_users() -> int:
    with engine.connect() as conn:
        return conn.execute(select(users.c.id).count()).scalar_one()


# --- documents ---------------------------------------------------------------


def add_document(user_id: int, doc_id: str, file_name: str, content: str) -> None:
    """Register a document row. Idempotent: same (user, doc_id) = update."""
    insert_stmt = _upsert_stmt(user_id, doc_id, file_name, content)
    with engine.begin() as conn:
        conn.execute(insert_stmt)


def _upsert_stmt(user_id: int, doc_id: str, file_name: str, content: str):
    """Build an INSERT ... ON CONFLICT DO UPDATE for the active dialect."""
    values = dict(user_id=user_id, doc_id=doc_id, file_name=file_name,
                  content=content, created_at=_now())
    if engine.dialect.name == "postgresql":
        stmt = postgresql.insert(documents).values(**values)
    else:
        stmt = sqlite.insert(documents).values(**values)
    return stmt.on_conflict_do_update(
        index_elements=[documents.c.user_id, documents.c.doc_id],
        set_={"file_name": file_name, "content": content},
    )


def delete_document(user_id: int, doc_id: str) -> bool:
    with engine.begin() as conn:
        result = conn.execute(
            sa_delete(documents).where(
                documents.c.user_id == user_id, documents.c.doc_id == doc_id
            )
        )
        conn.execute(
            sa_delete(chunks).where(
                chunks.c.user_id == user_id, chunks.c.doc_id == doc_id
            )
        )
        return result.rowcount > 0


def list_documents(user_id: int) -> list[dict]:
    subq = (
        select(func.count(chunks.c.id).label("n"))
        .where(chunks.c.user_id == user_id, chunks.c.doc_id == documents.c.doc_id)
        .scalar_subquery()
    )
    stmt = (
        select(
            documents.c.doc_id,
            documents.c.file_name,
            documents.c.created_at,
            subq.label("chunk_count"),
        )
        .where(documents.c.user_id == user_id)
        .order_by(documents.c.created_at)
    )
    with engine.connect() as conn:
        rows = conn.execute(stmt).mappings().all()
    return [dict(r) for r in rows]


def list_all_docs(user_id: int) -> dict[str, str]:
    """Map of doc_id -> content for all of a user's documents."""
    stmt = select(documents.c.doc_id, documents.c.content).where(
        documents.c.user_id == user_id
    )
    with engine.connect() as conn:
        rows = conn.execute(stmt).all()
    return {doc_id: content for doc_id, content in rows}


# --- chunks ------------------------------------------------------------------


def replace_chunks(user_id: int, chunks_: list[dict]) -> None:
    """Replace a user's entire index with the given chunk list.

    Each chunk dict: {doc_id, chunk_id, order, text, vector: np.ndarray}.
    Called after a full re-embed of a user's corpus.
    """
    with engine.begin() as conn:
        conn.execute(sa_delete(chunks).where(chunks.c.user_id == user_id))
        conn.execute(
            chunks.insert(),
            [
                {
                    "user_id": user_id,
                    "doc_id": c["doc_id"],
                    "chunk_id": c["chunk_id"],
                    "pos": c["order"],
                    "text": c["text"],
                    "vector": _pack(c["vector"]),
                }
                for c in chunks_
            ],
        )


def load_chunks(user_id: int) -> list[dict]:
    """Load every chunk+vector for a user, ordered by doc/position."""
    stmt = (
        select(
            chunks.c.doc_id,
            chunks.c.chunk_id,
            chunks.c.pos,
            chunks.c.text,
            chunks.c.vector,
        )
        .where(chunks.c.user_id == user_id)
        .order_by(chunks.c.doc_id, chunks.c.pos)
    )
    with engine.connect() as conn:
        rows = conn.execute(stmt).all()
    return [
        {
            "doc_id": r.doc_id,
            "chunk_id": r.chunk_id,
            "order": r.pos,
            "text": r.text,
            "vector": _unpack(r.vector),
        }
        for r in rows
    ]


def chunk_count(user_id: int) -> int:
    with engine.connect() as conn:
        return conn.execute(
            select(chunks.c.id.count()).where(chunks.c.user_id == user_id)
        ).scalar_one()
