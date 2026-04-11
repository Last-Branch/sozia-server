"""Integration tests for the full WebSocket stack using FastAPI TestClient.

These tests exercise the complete path from WebSocket accept through the
gateway → session handler → stub orchestrator → transcript segment send,
without loading any real ML models.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI, WebSocket
from fastapi.testclient import TestClient

from sozia.api.auth_middleware import AuthMiddleware
from sozia.api.gateway import WebSocketGateway

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_KEY = "integration-test-key"
_SESSION = "550e8400-e29b-41d4-a716-446655440000"
_SEGMENT_ID = "6ba7b810-9dad-11d1-80b4-00c04fd430c8"


# ---------------------------------------------------------------------------
# Stub orchestrator
# ---------------------------------------------------------------------------


def _stub_orchestrator() -> AsyncMock:
    orch = AsyncMock()
    orch.warm_up = AsyncMock()
    orch.cool_down = AsyncMock()
    orch.process = AsyncMock()
    return orch


# ---------------------------------------------------------------------------
# App builder helper (keeps route defined at module scope)
# ---------------------------------------------------------------------------


def _build_test_app(
    api_key: str = _KEY,
    orchestrator: AsyncMock | None = None,
) -> FastAPI:
    """Create a minimal FastAPI app wired to a stub gateway.

    The route handler is kept at module scope so FastAPI's dependency
    resolver can properly recognise the ``WebSocket`` type annotation.
    """
    auth = AuthMiddleware(api_key=api_key)
    orch = orchestrator or _stub_orchestrator()
    factory = MagicMock(return_value=orch)

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        gateway = WebSocketGateway(auth=auth, orchestrator_factory=factory)
        application.state.gateway = gateway
        yield

    application = FastAPI(lifespan=lifespan)
    application.add_api_websocket_route("/ws", _ws_handler)
    return application


async def _ws_handler(websocket: WebSocket) -> None:
    """Module-level handler so FastAPI resolves the WebSocket type correctly."""
    await websocket.app.state.gateway.on_connect(websocket)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def app() -> FastAPI:
    return _build_test_app()


@pytest.fixture
def client(app: FastAPI) -> TestClient:  # type: ignore[override]
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _session_init(path: str = "SPEECH") -> dict[str, Any]:
    return {
        "type": "session_init",
        "session_id": _SESSION,
        "modality_path": path,
        "api_key": _KEY,
    }


def _skip_status(ws, count: int = 2) -> None:
    """Drain ``count`` session_status messages before the test assertions."""
    for _ in range(count):
        msg = ws.receive_json()
        assert msg["type"] == "session_status"


# ---------------------------------------------------------------------------
# Happy-path connection flow
# ---------------------------------------------------------------------------


class TestHappyPath:
    def test_connection_accepted(self, client: TestClient) -> None:
        with client.websocket_connect("/ws") as ws:
            ws.send_json(_session_init())
            msg = ws.receive_json()
            assert msg["type"] == "session_status"
            assert msg["state"] == "INITIALIZING"

    def test_running_status_received(self, client: TestClient) -> None:
        with client.websocket_connect("/ws") as ws:
            ws.send_json(_session_init())
            states = []
            for _ in range(2):
                msg = ws.receive_json()
                if msg.get("type") == "session_status":
                    states.append(msg["state"])
            assert "RUNNING" in states

    def test_session_end_closes_cleanly(self, client: TestClient) -> None:
        with client.websocket_connect("/ws") as ws:
            ws.send_json(_session_init())
            _skip_status(ws)
            ws.send_json({"type": "session_end"})
            # No exception raised — connection closed cleanly.

    def test_sign_path_accepted(self, client: TestClient) -> None:
        with client.websocket_connect("/ws") as ws:
            ws.send_json(_session_init("SIGN"))
            msg = ws.receive_json()
            assert msg["type"] == "session_status"


# ---------------------------------------------------------------------------
# Auth failures
# ---------------------------------------------------------------------------


class TestAuthFailures:
    def test_wrong_key_receives_error_payload(self) -> None:
        app = _build_test_app()
        with TestClient(app) as c:
            # Server sends error JSON then closes — receive_json gets the error.
            with c.websocket_connect("/ws") as ws:
                ws.send_json({**_session_init(), "api_key": "wrong-key"})
                msg = ws.receive_json()
            assert msg["type"] == "error"
            assert msg["code"] == 4001

    def test_missing_key_closes_with_4001(self) -> None:
        app = _build_test_app()
        with TestClient(app) as c:
            with c.websocket_connect("/ws") as ws:
                init = _session_init()
                del init["api_key"]
                ws.send_json(init)
                msg = ws.receive_json()
            assert msg["code"] == 4001


# ---------------------------------------------------------------------------
# Session_init validation failures
# ---------------------------------------------------------------------------


class TestSessionInitFailures:
    def test_bad_modality_path_closes_with_4002(self) -> None:
        app = _build_test_app()
        with TestClient(app) as c:
            with c.websocket_connect("/ws") as ws:
                ws.send_json({**_session_init(), "modality_path": "INVALID"})
                msg = ws.receive_json()
            assert msg["code"] == 4002

    def test_missing_session_id_closes_with_4002(self) -> None:
        app = _build_test_app()
        with TestClient(app) as c:
            with c.websocket_connect("/ws") as ws:
                init = _session_init()
                del init["session_id"]
                ws.send_json(init)
                msg = ws.receive_json()
            assert msg["code"] == 4002


# ---------------------------------------------------------------------------
# Feature message dispatch
# ---------------------------------------------------------------------------


class TestFeatureDispatch:
    def test_audio_chunk_dispatched_to_orchestrator(self) -> None:
        orch = _stub_orchestrator()
        app = _build_test_app(orchestrator=orch)

        audio_msg = {
            "type": "audio_feature_chunk",
            "session_id": _SESSION,
            "timestamp_ms": 1000,
            "features": [[float(i) for i in range(13)] for _ in range(10)],
            "feature_type": "mfcc",
            "sample_rate_hz": 16000,
            "chunk_duration_ms": 500,
        }

        with TestClient(app) as c:
            with c.websocket_connect("/ws") as ws:
                ws.send_json(_session_init())
                _skip_status(ws)
                ws.send_json(audio_msg)
                ws.send_json({"type": "session_end"})

        orch.process.assert_awaited_once()

    def test_landmark_frame_dispatched_to_orchestrator(self) -> None:
        orch = _stub_orchestrator()
        app = _build_test_app(orchestrator=orch)

        landmark_msg = {
            "type": "landmark_frame",
            "session_id": _SESSION,
            "timestamp_ms": 1000,
            "face_landmarks": [[0.5, 0.5, 0.0]] * 83,
            "left_hand_landmarks": None,
            "right_hand_landmarks": None,
            "pose_landmarks": None,
        }

        with TestClient(app) as c:
            with c.websocket_connect("/ws") as ws:
                ws.send_json(_session_init("SIGN"))
                _skip_status(ws)
                ws.send_json(landmark_msg)
                ws.send_json({"type": "session_end"})

        orch.process.assert_awaited_once()

    def test_pipeline_health_does_not_call_process(self) -> None:
        orch = _stub_orchestrator()
        app = _build_test_app(orchestrator=orch)

        health_msg = {
            "type": "pipeline_health",
            "session_id": _SESSION,
            "pipeline": "audio",
            "available": True,
            "fps": None,
            "snr": 20.0,
            "face_detected": None,
            "last_updated_ms": 500,
        }

        with TestClient(app) as c:
            with c.websocket_connect("/ws") as ws:
                ws.send_json(_session_init())
                _skip_status(ws)
                ws.send_json(health_msg)
                ws.send_json({"type": "session_end"})

        orch.process.assert_not_awaited()

    def test_cool_down_called_after_session_end(self) -> None:
        orch = _stub_orchestrator()
        app = _build_test_app(orchestrator=orch)

        with TestClient(app) as c:
            with c.websocket_connect("/ws") as ws:
                ws.send_json(_session_init())
                _skip_status(ws)
                ws.send_json({"type": "session_end"})

        orch.cool_down.assert_awaited_once()
