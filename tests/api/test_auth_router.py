"""Tests for POST /auth/register and POST /auth/login."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import sozia.server.api.auth_router as auth_module
from sozia.server.api.auth_router import router


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def isolated_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset the in-memory stores and redirect the JSON file for every test."""
    monkeypatch.setattr(auth_module, "_USERS_FILE", tmp_path / "users.json")
    auth_module._users.clear()
    auth_module._tokens.clear()


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REGISTER = "/auth/register"
_LOGIN = "/auth/login"

_VALID = {"email": "ali@example.com", "password": "secret123", "name": "Ali"}


def _register(client: TestClient, payload: dict | None = None) -> dict:
    return client.post(_REGISTER, json=payload or _VALID).json()


# ---------------------------------------------------------------------------
# POST /auth/register — happy path
# ---------------------------------------------------------------------------


class TestRegister:
    def test_returns_200(self, client: TestClient) -> None:
        resp = client.post(_REGISTER, json=_VALID)
        assert resp.status_code == 200

    def test_response_contains_token(self, client: TestClient) -> None:
        data = _register(client)
        assert "token" in data
        assert len(data["token"]) > 0

    def test_response_user_fields(self, client: TestClient) -> None:
        data = _register(client)
        user = data["user"]
        assert user["email"] == _VALID["email"]
        assert user["name"] == _VALID["name"]
        assert "id" in user

    def test_user_id_is_uuid_string(self, client: TestClient) -> None:
        import uuid
        data = _register(client)
        uuid.UUID(data["user"]["id"])  # raises if invalid

    def test_user_persisted_to_json(self, client: TestClient, tmp_path: Path) -> None:
        _register(client)
        json_file = auth_module._USERS_FILE
        assert json_file.exists()
        import json
        stored = json.loads(json_file.read_text())
        assert _VALID["email"] in stored

    def test_password_not_stored_in_plaintext(self, client: TestClient) -> None:
        _register(client)
        stored = auth_module._users[_VALID["email"]]
        assert stored.get("password") is None
        assert "dk" in stored
        assert _VALID["password"] not in stored.values()

    def test_duplicate_email_returns_409(self, client: TestClient) -> None:
        _register(client)
        resp = client.post(_REGISTER, json=_VALID)
        assert resp.status_code == 409
        assert "error" in resp.json()

    def test_missing_name_returns_422(self, client: TestClient) -> None:
        resp = client.post(_REGISTER, json={"email": "a@b.com", "password": "x"})
        assert resp.status_code == 422

    def test_missing_email_returns_422(self, client: TestClient) -> None:
        resp = client.post(_REGISTER, json={"password": "x", "name": "Ali"})
        assert resp.status_code == 422

    def test_missing_password_returns_422(self, client: TestClient) -> None:
        resp = client.post(_REGISTER, json={"email": "a@b.com", "name": "Ali"})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST /auth/login — happy path
# ---------------------------------------------------------------------------


class TestLogin:
    def test_returns_200_after_register(self, client: TestClient) -> None:
        _register(client)
        resp = client.post(_LOGIN, json={"email": _VALID["email"], "password": _VALID["password"]})
        assert resp.status_code == 200

    def test_response_contains_token(self, client: TestClient) -> None:
        _register(client)
        data = client.post(_LOGIN, json={"email": _VALID["email"], "password": _VALID["password"]}).json()
        assert "token" in data
        assert len(data["token"]) > 0

    def test_response_user_matches_registration(self, client: TestClient) -> None:
        reg = _register(client)
        login = client.post(_LOGIN, json={"email": _VALID["email"], "password": _VALID["password"]}).json()
        assert login["user"]["id"] == reg["user"]["id"]
        assert login["user"]["email"] == reg["user"]["email"]
        assert login["user"]["name"] == reg["user"]["name"]

    def test_each_login_issues_new_token(self, client: TestClient) -> None:
        _register(client)
        creds = {"email": _VALID["email"], "password": _VALID["password"]}
        t1 = client.post(_LOGIN, json=creds).json()["token"]
        t2 = client.post(_LOGIN, json=creds).json()["token"]
        assert t1 != t2

    def test_wrong_password_returns_401(self, client: TestClient) -> None:
        _register(client)
        resp = client.post(_LOGIN, json={"email": _VALID["email"], "password": "wrong"})
        assert resp.status_code == 401
        assert "error" in resp.json()

    def test_unknown_email_returns_401(self, client: TestClient) -> None:
        resp = client.post(_LOGIN, json={"email": "nobody@example.com", "password": "x"})
        assert resp.status_code == 401
        assert "error" in resp.json()

    def test_missing_email_returns_422(self, client: TestClient) -> None:
        resp = client.post(_LOGIN, json={"password": "x"})
        assert resp.status_code == 422

    def test_missing_password_returns_422(self, client: TestClient) -> None:
        resp = client.post(_LOGIN, json={"email": "a@b.com"})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Persistence — users survive a store reload
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_login_works_after_store_reload(self, client: TestClient) -> None:
        _register(client)

        # Simulate restart: clear in-memory dict and reload from JSON.
        auth_module._users.clear()
        auth_module._load_users()

        resp = client.post(_LOGIN, json={"email": _VALID["email"], "password": _VALID["password"]})
        assert resp.status_code == 200

    def test_second_user_does_not_overwrite_first(self, client: TestClient) -> None:
        _register(client)
        other = {"email": "other@example.com", "password": "pass2", "name": "Other"}
        client.post(_REGISTER, json=other)

        auth_module._users.clear()
        auth_module._load_users()

        assert _VALID["email"] in auth_module._users
        assert other["email"] in auth_module._users
