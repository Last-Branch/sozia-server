"""Tests for SessionHandler."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock

from starlette.websockets import WebSocketDisconnect

from sozia.common.models import (
    AudioFeatureChunk,
    LandmarkFrame,
    ModalityPath,
    ModalityType,
    PipelineHealth,
    SegmentStatus,
    TranscriptSegment,
)
from sozia.api.session_handler import SessionHandler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SESSION = "550e8400-e29b-41d4-a716-446655440000"
_SEGMENT_ID = "6ba7b810-9dad-11d1-80b4-00c04fd430c8"


def _make_segment(
    status: SegmentStatus = SegmentStatus.PARTIAL,
    replaces: str | None = None,
) -> TranscriptSegment:
    return TranscriptSegment(
        segment_id=_SEGMENT_ID,
        session_id=_SESSION,
        status=status,
        text="merhaba",
        source=ModalityType.ASR,
        confidence=0.85,
        timestamp_ms=1000,
        duration_ms=500,
        created_at_ms=1_700_000_000_000,
        replaces_segment_id=replaces,
    )


def _make_landmark_msg() -> dict[str, Any]:
    return {
        "type": "landmark_frame",
        "session_id": _SESSION,
        "timestamp_ms": 1000,
        "face_landmarks": [[0.5, 0.5, 0.0]] * 83,
        "left_hand_landmarks": None,
        "right_hand_landmarks": None,
        "pose_landmarks": None,
    }


def _make_audio_msg() -> dict[str, Any]:
    return {
        "type": "audio_feature_chunk",
        "session_id": _SESSION,
        "timestamp_ms": 2000,
        "features": [[float(i) for i in range(13)] for _ in range(10)],
        "feature_type": "mfcc",
        "sample_rate_hz": 16000,
        "chunk_duration_ms": 500,
    }


def _make_health_msg() -> dict[str, Any]:
    return {
        "type": "pipeline_health",
        "session_id": _SESSION,
        "pipeline": "audio",
        "available": True,
        "fps": None,
        "snr": 15.0,
        "face_detected": None,
        "last_updated_ms": 1000,
    }


def _make_ws(*messages: dict[str, Any]) -> AsyncMock:
    """Build a mock WebSocket that yields messages then raises WebSocketDisconnect."""
    ws = AsyncMock()
    ws.send_text = AsyncMock()
    side_effects = [*messages, WebSocketDisconnect(code=1000)]
    ws.receive_json = AsyncMock(side_effect=side_effects)
    return ws


def _make_handler(
    ws: AsyncMock | None = None, path: ModalityPath = ModalityPath.SPEECH
) -> tuple[SessionHandler, AsyncMock]:
    ws = ws or _make_ws({"type": "session_end"})
    orchestrator = AsyncMock()
    orchestrator.process = AsyncMock()
    handler = SessionHandler(
        session_id=_SESSION,
        modality_path=path,
        websocket=ws,
        orchestrator=orchestrator,
    )
    return handler, orchestrator


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_attributes_set(self) -> None:
        ws = _make_ws()
        handler, _ = _make_handler(ws)
        assert handler.session_id == _SESSION
        assert handler.modality_path == ModalityPath.SPEECH
        assert handler.latest_health == {}

    def test_initial_health_empty(self) -> None:
        handler, _ = _make_handler()
        assert handler.latest_health == {}


# ---------------------------------------------------------------------------
# send_segment()
# ---------------------------------------------------------------------------


class TestSendSegment:
    async def test_sends_json_text(self) -> None:
        ws = AsyncMock()
        ws.send_text = AsyncMock()
        handler, _ = _make_handler(ws)
        seg = _make_segment()
        await handler.send_segment(seg)
        ws.send_text.assert_awaited_once()
        raw = ws.send_text.call_args[0][0]
        payload = json.loads(raw)
        assert payload["type"] == "transcript_segment"

    async def test_enums_serialized_as_strings(self) -> None:
        ws = AsyncMock()
        ws.send_text = AsyncMock()
        handler, _ = _make_handler(ws)
        seg = _make_segment()
        await handler.send_segment(seg)
        raw = ws.send_text.call_args[0][0]
        payload = json.loads(raw)
        assert payload["status"] == "PARTIAL"
        assert payload["source"] == "ASR"

    async def test_all_fields_present(self) -> None:
        ws = AsyncMock()
        ws.send_text = AsyncMock()
        handler, _ = _make_handler(ws)
        seg = _make_segment(status=SegmentStatus.FINAL, replaces=_SEGMENT_ID)
        await handler.send_segment(seg)
        raw = ws.send_text.call_args[0][0]
        payload = json.loads(raw)
        assert payload["segment_id"] == _SEGMENT_ID
        assert payload["session_id"] == _SESSION
        assert payload["confidence"] == 0.85
        assert payload["replaces_segment_id"] == _SEGMENT_ID

    async def test_replaces_segment_id_none_included(self) -> None:
        ws = AsyncMock()
        ws.send_text = AsyncMock()
        handler, _ = _make_handler(ws)
        seg = _make_segment()
        await handler.send_segment(seg)
        raw = ws.send_text.call_args[0][0]
        payload = json.loads(raw)
        assert "replaces_segment_id" in payload
        assert payload["replaces_segment_id"] is None


# ---------------------------------------------------------------------------
# receive_loop() — dispatch
# ---------------------------------------------------------------------------


class TestReceiveLoop:
    async def test_session_end_stops_loop(self) -> None:
        ws = _make_ws({"type": "session_end"})
        handler, orch = _make_handler(ws)
        await handler.receive_loop()
        orch.process.assert_not_awaited()

    async def test_disconnect_stops_loop_cleanly(self) -> None:
        ws = _make_ws()  # immediately raises WebSocketDisconnect
        handler, orch = _make_handler(ws)
        await handler.receive_loop()  # must not raise
        orch.process.assert_not_awaited()

    async def test_landmark_frame_calls_process(self) -> None:
        ws = _make_ws(_make_landmark_msg(), {"type": "session_end"})
        handler, orch = _make_handler(ws, ModalityPath.SIGN)
        await handler.receive_loop()
        orch.process.assert_awaited_once()
        _, kwargs = orch.process.call_args
        features = (
            orch.process.call_args[0][1]
            if orch.process.call_args[0]
            else orch.process.call_args.args[1]
        )
        assert isinstance(features, LandmarkFrame)

    async def test_audio_chunk_calls_process(self) -> None:
        ws = _make_ws(_make_audio_msg(), {"type": "session_end"})
        handler, orch = _make_handler(ws, ModalityPath.SPEECH)
        await handler.receive_loop()
        orch.process.assert_awaited_once()
        features = orch.process.call_args.args[1]
        assert isinstance(features, AudioFeatureChunk)

    async def test_pipeline_health_updates_latest_health(self) -> None:
        ws = _make_ws(_make_health_msg(), {"type": "session_end"})
        handler, orch = _make_handler(ws)
        await handler.receive_loop()
        assert "audio" in handler.latest_health
        h = handler.latest_health["audio"]
        assert isinstance(h, PipelineHealth)
        assert h.available is True

    async def test_pipeline_health_not_forwarded_to_process(self) -> None:
        ws = _make_ws(_make_health_msg(), {"type": "session_end"})
        handler, orch = _make_handler(ws)
        await handler.receive_loop()
        orch.process.assert_not_awaited()

    async def test_unknown_type_ignored(self) -> None:
        ws = _make_ws({"type": "unknown_gibberish", "data": 1}, {"type": "session_end"})
        handler, orch = _make_handler(ws)
        await handler.receive_loop()  # must not raise
        orch.process.assert_not_awaited()

    async def test_process_receives_health_list(self) -> None:
        ws = _make_ws(_make_health_msg(), _make_audio_msg(), {"type": "session_end"})
        handler, orch = _make_handler(ws, ModalityPath.SPEECH)
        await handler.receive_loop()
        call_args = orch.process.call_args
        health_arg = call_args.args[2]  # third positional arg
        assert isinstance(health_arg, list)
        assert len(health_arg) == 1
        assert isinstance(health_arg[0], PipelineHealth)

    async def test_multiple_feature_messages_dispatched(self) -> None:
        ws = _make_ws(_make_audio_msg(), _make_audio_msg(), {"type": "session_end"})
        handler, orch = _make_handler(ws, ModalityPath.SPEECH)
        await handler.receive_loop()
        assert orch.process.await_count == 2

    async def test_landmark_session_id_stamped_from_handler(self) -> None:
        """Client-supplied session_id in payload must be ignored; handler's wins."""
        msg = {**_make_landmark_msg(), "session_id": "attacker-session-id"}
        ws = _make_ws(msg, {"type": "session_end"})
        handler, orch = _make_handler(ws, ModalityPath.SIGN)
        await handler.receive_loop()
        frame: LandmarkFrame = orch.process.call_args.args[1]
        assert frame.session_id == _SESSION

    async def test_audio_session_id_stamped_from_handler(self) -> None:
        """Client-supplied session_id in payload must be ignored; handler's wins."""
        msg = {**_make_audio_msg(), "session_id": "attacker-session-id"}
        ws = _make_ws(msg, {"type": "session_end"})
        handler, orch = _make_handler(ws, ModalityPath.SPEECH)
        await handler.receive_loop()
        chunk: AudioFeatureChunk = orch.process.call_args.args[1]
        assert chunk.session_id == _SESSION
