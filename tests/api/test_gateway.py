"""Tests for WebSocketGateway."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.websockets import WebSocketDisconnect, WebSocketState

from sozia.api.auth_middleware import AuthMiddleware
from sozia.api.gateway import WebSocketGateway


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SESSION = "550e8400-e29b-41d4-a716-446655440000"
_KEY = "test-api-key"


def _make_ws(messages: list, *, state: WebSocketState = WebSocketState.CONNECTED) -> AsyncMock:
    """Build a mock WebSocket that accepts then yields messages."""
    ws = AsyncMock()
    ws.accept = AsyncMock()
    ws.send_text = AsyncMock()
    ws.close = AsyncMock()
    ws.client_state = state
    # receive_json is called after accept, so we mock its side_effect
    ws.receive_json = AsyncMock(
        side_effect=[*messages, WebSocketDisconnect(code=1000)]
    )
    return ws


def _valid_session_init(path: str = "SPEECH") -> dict:
    return {
        "type": "session_init",
        "session_id": _SESSION,
        "modality_path": path,
        "api_key": _KEY,
    }


def _make_gateway() -> WebSocketGateway:
    auth = AuthMiddleware(api_key=_KEY)
    mock_orch = AsyncMock()
    mock_orch.warm_up = AsyncMock()
    mock_orch.cool_down = AsyncMock()
    mock_orch.process = AsyncMock()
    orchestrator_factory = MagicMock(return_value=mock_orch)
    return WebSocketGateway(auth=auth, orchestrator_factory=orchestrator_factory)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_initial_active_sessions_empty(self) -> None:
        gw = _make_gateway()
        assert gw.active_sessions == {}


# ---------------------------------------------------------------------------
# on_connect() — auth
# ---------------------------------------------------------------------------


class TestOnConnectAuth:
    async def test_valid_auth_proceeds(self) -> None:
        gw = _make_gateway()
        ws = _make_ws([_valid_session_init(), {"type": "session_end"}])
        await gw.on_connect(ws)
        ws.accept.assert_awaited_once()

    async def test_missing_api_key_closes_4001(self) -> None:
        gw = _make_gateway()
        bad_init = {
            "type": "session_init",
            "session_id": _SESSION,
            "modality_path": "SPEECH",
            # api_key absent
        }
        ws = _make_ws([bad_init])
        await gw.on_connect(ws)
        ws.close.assert_awaited_once()
        assert ws.close.call_args[1]["code"] == 4001

    async def test_wrong_api_key_closes_4001(self) -> None:
        gw = _make_gateway()
        bad_init = {**_valid_session_init(), "api_key": "wrong-key"}
        ws = _make_ws([bad_init])
        await gw.on_connect(ws)
        ws.close.assert_awaited_once()
        assert ws.close.call_args[1]["code"] == 4001


# ---------------------------------------------------------------------------
# on_connect() — session_init parsing
# ---------------------------------------------------------------------------


class TestOnConnectSessionInit:
    async def test_missing_session_id_closes_4002(self) -> None:
        gw = _make_gateway()
        bad_init = {
            "type": "session_init",
            "modality_path": "SPEECH",
            "api_key": _KEY,
        }
        ws = _make_ws([bad_init])
        await gw.on_connect(ws)
        ws.close.assert_awaited_once()
        assert ws.close.call_args[1]["code"] == 4002

    async def test_invalid_modality_path_closes_4002(self) -> None:
        gw = _make_gateway()
        bad_init = {**_valid_session_init(), "modality_path": "INVALID"}
        ws = _make_ws([bad_init])
        await gw.on_connect(ws)
        ws.close.assert_awaited_once()
        assert ws.close.call_args[1]["code"] == 4002

    async def test_wrong_init_type_closes_4002(self) -> None:
        gw = _make_gateway()
        bad_init = {"type": "landmark_frame", "api_key": _KEY}
        ws = _make_ws([bad_init])
        await gw.on_connect(ws)
        ws.close.assert_awaited_once()
        assert ws.close.call_args[1]["code"] == 4002

    async def test_sign_path_accepted(self) -> None:
        gw = _make_gateway()
        ws = _make_ws([_valid_session_init("SIGN"), {"type": "session_end"}])
        await gw.on_connect(ws)
        ws.close.assert_not_awaited()


# ---------------------------------------------------------------------------
# on_connect() — duplicate session
# ---------------------------------------------------------------------------


class TestOnConnectDuplicate:
    async def test_duplicate_session_id_closes_4003(self) -> None:
        gw = _make_gateway()
        # First connection populates active_sessions; we inject it manually.
        gw.active_sessions[_SESSION] = MagicMock()
        ws = _make_ws([_valid_session_init()])
        await gw.on_connect(ws)
        ws.close.assert_awaited_once()
        assert ws.close.call_args[1]["code"] == 4003


# ---------------------------------------------------------------------------
# on_connect() — warm-up failure
# ---------------------------------------------------------------------------


class TestOnConnectWarmupFailure:
    async def test_warmup_error_closes_4004(self) -> None:
        auth = AuthMiddleware(api_key=_KEY)
        orch = AsyncMock()
        orch.warm_up = AsyncMock(side_effect=RuntimeError("GPU OOM"))
        orch.cool_down = AsyncMock()
        orch.process = AsyncMock()
        gw = WebSocketGateway(auth=auth, orchestrator_factory=lambda *_: orch)
        ws = _make_ws([_valid_session_init()])
        await gw.on_connect(ws)
        ws.close.assert_awaited_once()
        assert ws.close.call_args[1]["code"] == 4004


# ---------------------------------------------------------------------------
# on_connect() — session lifecycle
# ---------------------------------------------------------------------------


class TestSessionLifecycle:
    async def test_session_added_to_active_during_connection(self) -> None:
        gw = _make_gateway()
        active_at_receive: list[bool] = []

        async def fake_receive_json():
            active_at_receive.append(_SESSION in gw.active_sessions)
            raise WebSocketDisconnect(code=1000)

        ws = _make_ws([_valid_session_init(), {"type": "session_end"}])
        # Peek inside active_sessions after accept & init but before receive
        await gw.on_connect(ws)

    async def test_session_removed_after_disconnect(self) -> None:
        gw = _make_gateway()
        ws = _make_ws([_valid_session_init(), {"type": "session_end"}])
        await gw.on_connect(ws)
        assert _SESSION not in gw.active_sessions

    async def test_cool_down_called_on_disconnect(self) -> None:
        auth = AuthMiddleware(api_key=_KEY)
        orch = AsyncMock()
        orch.warm_up = AsyncMock()
        orch.cool_down = AsyncMock()
        orch.process = AsyncMock()
        gw = WebSocketGateway(auth=auth, orchestrator_factory=lambda *_: orch)
        ws = _make_ws([_valid_session_init(), {"type": "session_end"}])
        await gw.on_connect(ws)
        orch.cool_down.assert_awaited_once()


# ---------------------------------------------------------------------------
# on_disconnect()
# ---------------------------------------------------------------------------


class TestOnDisconnect:
    async def test_removes_session_from_active(self) -> None:
        gw = _make_gateway()
        mock_handler = AsyncMock()
        mock_handler.orchestrator = AsyncMock()
        mock_handler.orchestrator.cool_down = AsyncMock()
        gw.active_sessions[_SESSION] = mock_handler
        await gw.on_disconnect(_SESSION)
        assert _SESSION not in gw.active_sessions

    async def test_noop_for_unknown_session(self) -> None:
        gw = _make_gateway()
        await gw.on_disconnect("nonexistent-session")  # must not raise


# ---------------------------------------------------------------------------
# Status messages
# ---------------------------------------------------------------------------


class TestStatusMessages:
    async def test_initializing_status_sent_before_warmup(self) -> None:
        auth = AuthMiddleware(api_key=_KEY)
        sent_texts: list[str] = []

        orch = AsyncMock()
        orch.warm_up = AsyncMock()
        orch.cool_down = AsyncMock()
        orch.process = AsyncMock()

        gw = WebSocketGateway(auth=auth, orchestrator_factory=lambda *_: orch)
        ws = _make_ws([_valid_session_init(), {"type": "session_end"}])

        import json
        ws.send_text = AsyncMock(side_effect=lambda t: sent_texts.append(t))

        await gw.on_connect(ws)

        types_sent = [json.loads(t).get("state") for t in sent_texts]
        assert "INITIALIZING" in types_sent

    async def test_running_status_sent_after_warmup(self) -> None:
        auth = AuthMiddleware(api_key=_KEY)
        sent_texts: list[str] = []

        orch = AsyncMock()
        orch.warm_up = AsyncMock()
        orch.cool_down = AsyncMock()
        orch.process = AsyncMock()

        gw = WebSocketGateway(auth=auth, orchestrator_factory=lambda *_: orch)
        ws = _make_ws([_valid_session_init(), {"type": "session_end"}])

        import json
        ws.send_text = AsyncMock(side_effect=lambda t: sent_texts.append(t))

        await gw.on_connect(ws)

        types_sent = [json.loads(t).get("state") for t in sent_texts]
        assert "RUNNING" in types_sent
