"""End-to-end API tests via FastAPI's TestClient.

These cover the full surface: auth, document management, ask, and chat
streaming. The database is isolated to a temp file by conftest.py.
"""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture()
def client():
    return TestClient(app)


@pytest.fixture()
def authed_client(client):
    email = f"user-{uuid.uuid4().hex[:10]}@example.com"
    res = client.post(
        "/api/auth/register",
        json={"email": email, "password": "password123"},
    )
    assert res.status_code == 200, res.text
    token = res.json()["access_token"]
    client.headers["Authorization"] = f"Bearer {token}"
    return client


def test_health():
    client = TestClient(app)
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["llm_provider"] == "none"


def test_register_login_me():
    client = TestClient(app)
    res = client.post(
        "/api/auth/register",
        json={"email": "alice@example.com", "password": "supersecret1"},
    )
    assert res.status_code == 200
    token = res.json()["access_token"]

    # me() without a token must be rejected
    assert client.get("/api/auth/me").status_code == 401

    me = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200
    assert me.json()["email"] == "alice@example.com"

    login = client.post(
        "/api/auth/login",
        json={"email": "alice@example.com", "password": "supersecret1"},
    )
    assert login.status_code == 200

    bad = client.post(
        "/api/auth/login",
        json={"email": "alice@example.com", "password": "wrongpassword"},
    )
    assert bad.status_code == 401


def test_register_duplicate_email():
    client = TestClient(app)
    payload = {"email": "bob@example.com", "password": "supersecret1"}
    assert client.post("/api/auth/register", json=payload).status_code == 200
    assert client.post("/api/auth/register", json=payload).status_code == 409


def test_documents_are_user_scoped(authed_client, client):
    # User A ingests a doc
    res = authed_client.post(
        "/api/ingest", json={"documents": {"a.md": "alpha beta gamma delta epsilon"}}
    )
    assert res.status_code == 200
    assert res.json()["indexed_chunks"] >= 1

    docs = authed_client.get("/api/documents").json()["documents"]
    assert [d["doc_id"] for d in docs] == ["a.md"]

    # User B sees nothing
    res_b = client.post(
        "/api/auth/register",
        json={"email": "charlie@example.com", "password": "supersecret1"},
    )
    token_b = res_b.json()["access_token"]
    docs_b = client.get(
        "/api/documents", headers={"Authorization": f"Bearer {token_b}"}
    ).json()["documents"]
    assert docs_b == []


def test_upload_and_delete_document(authed_client):
    files = {"file": ("notes.txt", b"the quick brown fox jumps over the lazy dog", "text/plain")}
    res = authed_client.post("/api/documents/upload", files=files)
    assert res.status_code == 200
    assert res.json()["doc_id"] == "notes.txt"

    docs = authed_client.get("/api/documents").json()["documents"]
    assert [d["doc_id"] for d in docs] == ["notes.txt"]

    # Reject non-.md/.txt uploads
    bad = authed_client.post(
        "/api/documents/upload", files={"file": ("evil.exe", b"MZ", "application/x-msdownload")}
    )
    assert bad.status_code == 400

    delete = authed_client.delete(f"/api/documents/{'notes.txt'}")
    assert delete.status_code == 200
    assert authed_client.get("/api/documents").json()["documents"] == []


def test_ask_requires_auth():
    client = TestClient(app)
    res = client.post("/api/ask", json={"question": "hi", "k": 2})
    assert res.status_code == 401


def test_ask_returns_sources(authed_client):
    authed_client.post(
        "/api/ingest",
        json={"documents": {"payments.md": "Settlement moves money to a merchant bank."}},
    )
    res = authed_client.post(
        "/api/ask", json={"question": "How does money reach a merchant?", "k": 1}
    )
    assert res.status_code == 200
    body = res.json()
    assert body["sources"][0]["doc_id"] == "payments.md"
    assert body["confidence"] > 0


def test_chat_streams_events(authed_client):
    authed_client.post(
        "/api/ingest",
        json={"documents": {"food.md": "Yeast makes bread rise in a warm kitchen."}},
    )
    with authed_client.stream(
        "POST", "/api/chat", json={"question": "What makes bread rise?", "k": 1}
    ) as res:
        assert res.status_code == 200
        assert res.headers["content-type"].startswith("text/event-stream")
        text = "".join(res.iter_text())
    assert 'data: {"type": "sources"' in text
    assert '"type": "done"' in text


def test_chat_without_docs_still_streams(authed_client):
    with authed_client.stream(
        "POST", "/api/chat", json={"question": "anything?", "k": 2}
    ) as res:
        text = "".join(res.iter_text())
    assert '"type": "sources"' in text
    assert '"type": "done"' in text
