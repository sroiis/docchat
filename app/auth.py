"""Authentication: password hashing + JWT bearer tokens.

Kept dependency-light: PBKDF2 hashing via the stdlib (no bcrypt build pain)
and PyJWT for tokens. This is a clean, auditable security surface.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import time

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from . import db
from .config import settings

_ALGO = "HS256"
_ITERATIONS = 210_000  # OWASP-recommended for PBKDF2-SHA256


# --- password hashing (PBKDF2-SHA256) ----------------------------------------


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt), _ITERATIONS
    ).hex()
    return f"{_ITERATIONS}${salt}${digest}"


def verify_password(password: str, stored: str) -> bool:
    try:
        iterations, salt, expected = stored.split("$")
        digest = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            bytes.fromhex(salt),
            int(iterations),
        ).hex()
        return hmac.compare_digest(digest, expected)
    except (ValueError, TypeError):
        return False


# --- JWT ---------------------------------------------------------------------


def create_access_token(user_id: int) -> str:
    payload = {
        "sub": str(user_id),
        "iat": int(time.time()),
        "exp": int(time.time()) + settings.token_expiry_minutes * 60,
    }
    return jwt.encode(payload, settings.secret_key, algorithm=_ALGO)


def _decode(token: str) -> int:
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[_ALGO])
    except jwt.PyJWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        ) from exc
    return int(payload["sub"])


# --- FastAPI dependency ------------------------------------------------------

_bearer = HTTPBearer(auto_error=False)


def get_current_user(
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> dict:
    """Resolve the authenticated user from the Authorization header.

    When DOCCHAT_AUTH_ENABLED=false, any request acts as the demo user so
    the API is usable without a token for local demos.
    """
    if not settings.auth_enabled:
        return _require_user(_demo_user_id())

    if creds is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token",
        )
    return _require_user(_decode(creds.credentials))


def _demo_user_id() -> int:
    """Ensure the demo user exists and return its id."""
    demo = db.get_or_create_demo_user(settings.demo_user_email, hash_password("demo"))
    return int(demo["id"])


def _require_user(user_id: int) -> dict:
    user = db.get_user_by_id(user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User no longer exists",
        )
    return {"id": user["id"], "email": user["email"]}
