"""
Basic login.

Username and password login returning a signed JWT. Every dashboard endpoint
needs that token in an `Authorization: Bearer <token>` header.

How the password is stored
    Never in plain text. It is stored as
    pbkdf2_hmac('sha256', password, salt, 200_000).

    The salt means two users with the same password get different hashes, so an
    attacker cannot crack them all at once from one rainbow table.

    The 200,000 iterations make each guess deliberately slow. Plain SHA-256 can
    be guessed billions of times a second on a GPU, and this brings that down to
    thousands.

    hmac.compare_digest compares in constant time, so the hash cannot be worked
    out byte by byte from how long the comparison took.

How the token works
    A JWT is three base64 parts: header, payload, signature. The payload holds
    the username and an expiry, and the signature is HMAC-SHA256 over the first
    two parts using the secret key. Anyone can read a JWT, so nothing secret goes
    in it, but nobody can forge one without the key. The signature and expiry are
    re-checked on every request.

What this is not
    The user store is a dict in memory. There is no registration, no password
    reset, no refresh tokens, no rate limiting on login and no HTTPS. It covers
    "only logged-in users can see the dashboard" and nothing more.
"""

from __future__ import annotations

import hashlib
import hmac
import os
from datetime import datetime, timedelta, timezone

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from backend import config, database

PBKDF2_ITERATIONS = 200_000
_bearer = HTTPBearer(auto_error=False)


def hash_password(password: str, salt: bytes | None = None) -> str:
    """Return 'salt_hex$hash_hex'. Generates a fresh random salt by default."""
    salt = salt or os.urandom(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode(), salt, PBKDF2_ITERATIONS
    )
    return f"{salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    """Check a candidate password against a stored hash, in constant time."""
    try:
        salt_hex, hash_hex = stored.split("$", 1)
        salt = bytes.fromhex(salt_hex)
    except (ValueError, AttributeError):
        return False
    candidate = hashlib.pbkdf2_hmac(
        "sha256", password.encode(), salt, PBKDF2_ITERATIONS
    )
    return hmac.compare_digest(candidate.hex(), hash_hex)


# --- User store ----------------------------------------------------------
# One demo account, hashed at import time. Swap this dict for a database table
# and nothing else in this file has to change.
USERS: dict[str, dict] = {
    config.DEMO_USERNAME: {
        "username": config.DEMO_USERNAME,
        "password_hash": hash_password(config.DEMO_PASSWORD),
        "role": "maintenance_engineer",
    }
}


def authenticate(username: str, password: str) -> dict | None:
    """Return the user record on success, None on failure."""
    user = USERS.get(username)
    if user is None:
        # Hash against a dummy value anyway, so a wrong username takes the same
        # time as a wrong password. Otherwise the response time tells an attacker
        # which usernames exist.
        verify_password(password, hash_password("dummy"))
        database.log_auth_event(username, False, "unknown user")
        return None

    if not verify_password(password, user["password_hash"]):
        database.log_auth_event(username, False, "bad password")
        return None

    database.log_auth_event(username, True)
    return user


def create_token(username: str) -> tuple[str, int]:
    """Return (jwt, seconds_until_expiry)."""
    ttl = timedelta(minutes=config.TOKEN_TTL_MINUTES)
    now = datetime.now(timezone.utc)
    payload = {
        "sub": username,          # subject: who this token is for
        "iat": int(now.timestamp()),          # issued at
        "exp": int((now + ttl).timestamp()),  # expiry, enforced by PyJWT
    }
    token = jwt.encode(payload, config.SECRET_KEY, algorithm=config.JWT_ALGORITHM)
    return token, int(ttl.total_seconds())


def decode_token(token: str) -> dict:
    """Check the signature and expiry. Raises 401 on anything that fails."""
    try:
        return jwt.decode(
            token, config.SECRET_KEY, algorithms=[config.JWT_ALGORITHM]
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "Session expired — please log in again.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "Invalid authentication token.",
            headers={"WWW-Authenticate": "Bearer"},
        )


def current_user(
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> dict:
    """
    FastAPI dependency. Put `user = Depends(current_user)` on an endpoint and it
    becomes login-only. FastAPI runs this first and rejects the request before
    the handler is reached.
    """
    if creds is None:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "Not authenticated.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    payload = decode_token(creds.credentials)
    username = payload.get("sub")
    user = USERS.get(username)
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "User no longer exists.")
    return user
