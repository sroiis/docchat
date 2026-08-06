"""Unit tests for auth primitives: password hashing and JWT tokens."""

from app.auth import _decode, create_access_token, hash_password, verify_password


def test_password_hash_roundtrip():
    hashed = hash_password("correct horse battery staple")
    assert hashed != "correct horse battery staple"
    assert verify_password("correct horse battery staple", hashed)
    assert not verify_password("wrong password", hashed)


def test_password_hash_is_salted():
    a = hash_password("same")
    b = hash_password("same")
    assert a != b  # random salts -> different hashes


def test_jwt_roundtrip():
    token = create_access_token(42)
    assert _decode(token) == 42


def test_jwt_returns_int_subject():
    # The token's "sub" is a string in the payload but decoded back to int.
    import jwt as _jwt

    from app.config import settings

    token = _jwt.encode({"sub": "7"}, settings.secret_key, algorithm="HS256")
    assert _decode(token) == 7
