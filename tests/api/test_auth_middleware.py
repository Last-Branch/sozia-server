"""Tests for AuthMiddleware."""

from __future__ import annotations

import pytest

from sozia.server.api.auth_middleware import AuthMiddleware


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def valid_key() -> str:
    return "super-secret-key-abc123"


@pytest.fixture
def middleware(valid_key: str) -> AuthMiddleware:
    return AuthMiddleware(api_key=valid_key)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_rejects_empty_key(self) -> None:
        with pytest.raises(ValueError, match="api_key"):
            AuthMiddleware(api_key="")

    def test_accepts_non_empty_key(self, valid_key: str) -> None:
        m = AuthMiddleware(api_key=valid_key)
        assert m is not None


# ---------------------------------------------------------------------------
# authenticate()
# ---------------------------------------------------------------------------


class TestAuthenticate:
    def test_returns_true_for_correct_key(
        self, middleware: AuthMiddleware, valid_key: str
    ) -> None:
        assert middleware.authenticate(valid_key) is True

    def test_returns_false_for_wrong_key(self, middleware: AuthMiddleware) -> None:
        assert middleware.authenticate("wrong-key") is False

    def test_returns_false_for_empty_string(self, middleware: AuthMiddleware) -> None:
        assert middleware.authenticate("") is False

    def test_timing_safe_no_short_circuit(
        self, middleware: AuthMiddleware, valid_key: str
    ) -> None:
        """Authenticate must not raise even when strings differ in length."""
        # Different-length strings used to raise in naive implementations;
        # hmac.compare_digest handles this safely.
        assert middleware.authenticate("x") is False
        assert middleware.authenticate(valid_key + "extra") is False

    def test_case_sensitive(self, middleware: AuthMiddleware, valid_key: str) -> None:
        assert middleware.authenticate(valid_key.upper()) is False

    def test_correct_key_after_wrong_attempts(
        self, middleware: AuthMiddleware, valid_key: str
    ) -> None:
        middleware.authenticate("bad")
        middleware.authenticate("also-bad")
        assert middleware.authenticate(valid_key) is True


# ---------------------------------------------------------------------------
# from_env()
# ---------------------------------------------------------------------------


class TestFromEnv:
    def test_reads_from_env(
        self, monkeypatch: pytest.MonkeyPatch, valid_key: str
    ) -> None:
        monkeypatch.setenv("SOZIA_API_KEY", valid_key)
        m = AuthMiddleware.from_env()
        assert m.authenticate(valid_key) is True

    def test_raises_when_env_var_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SOZIA_API_KEY", raising=False)
        with pytest.raises(EnvironmentError, match="SOZIA_API_KEY"):
            AuthMiddleware.from_env()
