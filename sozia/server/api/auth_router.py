"""HTTP auth endpoints — /auth/register and /auth/login.

Users are persisted to a JSON file (SOZIA_USERS_FILE env var, default
"sozia_users.json"). Tokens remain in-memory only — they reset on restart
by design (clients will just re-login). Passwords are hashed with
PBKDF2-HMAC-SHA256 + a per-user salt; salt and digest are stored as hex.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import uuid
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

router = APIRouter(prefix="/auth")

# ---------------------------------------------------------------------------
# JSON persistence
# ---------------------------------------------------------------------------

_USERS_FILE = Path(
    os.environ.get("SOZIA_USERS_FILE")
    or Path(__file__).resolve().parents[3] / "sozia-user-data" / "sozia_users.json"
)
_USERS_FILE.parent.mkdir(parents=True, exist_ok=True)

# email → {"id", "email", "name", "salt", "dk"}  (salt/dk as hex strings)
_users: dict[str, dict] = {}

# token → user_id  (in-memory only; resets on server restart)
_tokens: dict[str, str] = {}


def _load_users() -> None:
    if _USERS_FILE.exists():
        data = json.loads(_USERS_FILE.read_text(encoding="utf-8"))
        _users.update(data)


def _save_users() -> None:
    _USERS_FILE.write_text(
        json.dumps(_users, indent=2, ensure_ascii=False), encoding="utf-8"
    )


_load_users()

_PBKDF2_ITERATIONS = 260_000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hash_password(password: str, salt: str) -> str:
    """Return hex digest of PBKDF2-HMAC-SHA256(password, salt)."""
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode(), bytes.fromhex(salt), _PBKDF2_ITERATIONS
    ).hex()


def _verify_password(password: str, salt: str, dk: str) -> bool:
    return secrets.compare_digest(_hash_password(password, salt), dk)


def _issue_token(user_id: str) -> str:
    token = secrets.token_hex(32)
    _tokens[token] = user_id
    return token


def _user_payload(user: dict) -> dict:
    return {"id": user["id"], "email": user["email"], "name": user["name"]}


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class RegisterBody(BaseModel):
    email: str
    password: str
    name: str


class LoginBody(BaseModel):
    email: str
    password: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/register")
async def register(body: RegisterBody):
    if not body.email or not body.password or not body.name:
        return JSONResponse(status_code=400, content={"error": "email, password and name are required"})

    if body.email in _users:
        return JSONResponse(status_code=409, content={"error": "Email already registered"})

    salt = os.urandom(32).hex()
    user: dict = {
        "id": str(uuid.uuid4()),
        "email": body.email,
        "name": body.name,
        "salt": salt,
        "dk": _hash_password(body.password, salt),
    }
    _users[body.email] = user
    _save_users()

    token = _issue_token(user["id"])
    return {"token": token, "user": _user_payload(user)}


@router.post("/login")
async def login(body: LoginBody):
    user = _users.get(body.email)
    if user is None or not _verify_password(body.password, user["salt"], user["dk"]):
        return JSONResponse(status_code=401, content={"error": "Invalid email or password"})

    token = _issue_token(user["id"])
    return {"token": token, "user": _user_payload(user)}
