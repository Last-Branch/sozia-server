"""SessionHandler — per-connection WebSocket session manager.

One SessionHandler is created per accepted WebSocket connection. It owns
the receive loop, deserializes inbound messages, and forwards them to the
FusionOrchestrator. Outbound TranscriptSegments are serialised and sent
back over the same connection.

Dependency rule R7: only sozia.server.fusion produces TranscriptSegments.
SessionHandler only sends them.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

from starlette.websockets import WebSocketDisconnect

from sozia.common.models import (
    AudioFeatureChunk,
    LandmarkFrame,
    ModalityPath,
    PipelineHealth,
    TranscriptSegment,
)

if TYPE_CHECKING:
    from fastapi import WebSocket

    from sozia.server.fusion.orchestrator import FusionOrchestrator


class SessionHandler:
    """Manages one WebSocket session end-to-end.

    Args:
        session_id: UUID v4 of the session (from ``session_init``).
        modality_path: The active inference pipeline (SPEECH or SIGN).
        websocket: The accepted FastAPI/Starlette WebSocket object.
        orchestrator: The FusionOrchestrator that handles inference.
    """

    def __init__(
        self,
        session_id: str,
        modality_path: ModalityPath,
        websocket: WebSocket,
        orchestrator: FusionOrchestrator,
    ) -> None:
        self.session_id = session_id
        self.modality_path = modality_path
        self._ws = websocket
        self._orchestrator = orchestrator
        self.latest_health: dict[str, PipelineHealth] = {}

    # ------------------------------------------------------------------
    # Outbound
    # ------------------------------------------------------------------

    async def send_segment(self, segment: TranscriptSegment) -> None:
        """Serialise ``segment`` and send it over the WebSocket.

        Enum fields are serialised as their string values.  The payload
        includes a ``"type": "transcript_segment"`` discriminator.

        Args:
            segment: The TranscriptSegment to send.
        """
        payload = dataclasses.asdict(segment)
        # Serialise enum values to their string representation.
        payload["status"] = segment.status.value
        payload["source"] = segment.source.value
        payload["type"] = "transcript_segment"
        await self._ws.send_text(json.dumps(payload))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Release orchestrator resources for this session.

        Called by the gateway on both graceful disconnect and server shutdown.
        """
        await self._orchestrator.cool_down()

    # ------------------------------------------------------------------
    # Inbound loop
    # ------------------------------------------------------------------

    async def receive_loop(self) -> None:
        """Read messages from the WebSocket and dispatch by type.

        Runs until the client sends ``{"type": "session_end"}`` or the
        connection is closed (``WebSocketDisconnect``).  Unknown message
        types are silently ignored.
        """
        while True:
            try:
                data = await self._ws.receive_json()
            except WebSocketDisconnect:
                break

            msg_type = data.get("type")

            if msg_type == "session_end":
                break
            elif msg_type == "landmark_frame":
                await self._handle_landmark(data)
            elif msg_type == "audio_feature_chunk":
                await self._handle_audio(data)
            elif msg_type == "pipeline_health":
                self._handle_health(data)
            # Unknown types are silently ignored per spec.

    # ------------------------------------------------------------------
    # Dispatch helpers
    # ------------------------------------------------------------------

    async def _handle_landmark(self, data: dict) -> None:
        frame = LandmarkFrame(
            session_id=self.session_id,
            timestamp_ms=data["timestamp_ms"],
            face_landmarks=data.get("face_landmarks"),
            left_hand_landmarks=data.get("left_hand_landmarks"),
            right_hand_landmarks=data.get("right_hand_landmarks"),
            pose_landmarks=data.get("pose_landmarks"),
        )
        await self._orchestrator.process(
            self.session_id,
            frame,
            list(self.latest_health.values()),
            self.modality_path,
            self.send_segment,
        )

    async def _handle_audio(self, data: dict) -> None:
        features = data["features"]
        rows = len(features)
        cols = len(features[0]) if rows else 0
        flat_max = max((max(row) for row in features), default=float("-inf"))
        logger.info(
            "audio_chunk received: shape=(%d, %d) max=%.3f feature_type=%s",
            rows, cols, flat_max, data.get("feature_type", "?"),
        )
        chunk = AudioFeatureChunk(
            session_id=self.session_id,
            timestamp_ms=data["timestamp_ms"],
            features=features,
            feature_type=data["feature_type"],
            sample_rate_hz=data["sample_rate_hz"],
            chunk_duration_ms=data["chunk_duration_ms"],
        )
        await self._orchestrator.process(
            self.session_id,
            chunk,
            list(self.latest_health.values()),
            self.modality_path,
            self.send_segment,
        )

    def _handle_health(self, data: dict) -> None:
        health = PipelineHealth(
            session_id=data["session_id"],
            pipeline=data["pipeline"],
            available=data["available"],
            fps=data.get("fps"),
            snr=data.get("snr"),
            face_detected=data.get("face_detected"),
            last_updated_ms=data["last_updated_ms"],
        )
        self.latest_health[health.pipeline] = health
