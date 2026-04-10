"""DegradedModeHandler — fallback when one modality is unavailable.

When a pipeline becomes unavailable (device lost, no face detected, low SNR)
or a modality times out, this handler decides whether to fall back to
single-modality output with reduced confidence — or, if every pipeline is
down, to signal that the session must terminate.
"""

from __future__ import annotations

import time
import uuid
from enum import Enum

from sozia.common.models import (
    ModalityResult,
    PipelineHealth,
    SegmentStatus,
    TranscriptSegment,
)

# Default confidence penalty applied to degraded (single-modality) output.
_DEFAULT_CONFIDENCE_PENALTY = 0.15

# Default minimum SNR (dB) for the audio pipeline to be considered usable.
_DEFAULT_MIN_AUDIO_SNR = 5.0


class DegradedStatus(Enum):
    """Overall health state of the session's input pipelines.

    Used by the API/gateway layer to decide whether to keep streaming
    (NORMAL), fall back to a single modality with reduced confidence
    (DEGRADED), or close the session (FAILED).
    """

    NORMAL = "normal"
    DEGRADED = "degraded"
    FAILED = "failed"


class DegradedModeHandler:
    """Decides when to enter degraded mode and produces fallback segments."""

    def __init__(
        self,
        *,
        confidence_penalty: float = _DEFAULT_CONFIDENCE_PENALTY,
        min_audio_snr: float = _DEFAULT_MIN_AUDIO_SNR,
    ) -> None:
        self._confidence_penalty = confidence_penalty
        self._min_audio_snr = min_audio_snr

    # ------------------------------------------------------------------
    # Status evaluation
    # ------------------------------------------------------------------

    def evaluate(self, health: list[PipelineHealth]) -> DegradedStatus:
        """Evaluate aggregate pipeline health into a 3-state status.

        - NORMAL — every reported pipeline is healthy.
        - DEGRADED — at least one pipeline is healthy; at least one is not.
        - FAILED — no healthy pipelines; the session should be terminated.

        An empty health list is treated as FAILED (nothing to listen to).
        """
        if not health:
            return DegradedStatus.FAILED

        degraded_count = sum(1 for h in health if self.should_fallback(h))

        if degraded_count == 0:
            return DegradedStatus.NORMAL
        if degraded_count == len(health):
            return DegradedStatus.FAILED
        return DegradedStatus.DEGRADED

    def get_advisory_message(
        self, health: list[PipelineHealth],
    ) -> str | None:
        """Return a user-facing string describing the degraded state.

        Returns ``None`` when every pipeline is healthy. Emits distinct
        strings for common failure modes so the client can render them
        directly.
        """
        if not health:
            return "All input pipelines unavailable — session ending."

        messages: list[str] = []
        for h in health:
            msg = self._advisory_for(h)
            if msg is not None:
                messages.append(msg)

        if not messages:
            return None
        return " ".join(messages)

    def _advisory_for(self, health: PipelineHealth) -> str | None:
        if not health.available:
            if health.pipeline == "audio":
                return "Microphone unavailable — speech recognition paused."
            if health.pipeline == "video":
                return "Camera unavailable — visual recognition paused."
            return f"{health.pipeline.capitalize()} pipeline unavailable."

        if health.pipeline == "video" and health.face_detected is False:
            return "Camera obstructed — visual recognition paused."

        if (
            health.pipeline == "audio"
            and health.snr is not None
            and health.snr < self._min_audio_snr
        ):
            return "Low audio level — speech recognition degraded."

        return None

    # ------------------------------------------------------------------
    # Per-pipeline fallback check (used by the orchestrator)
    # ------------------------------------------------------------------

    def should_fallback(self, health: PipelineHealth) -> bool:
        """Return True if the pipeline health indicates degraded state.

        Triggers:
          - ``available`` is False (device lost, permission denied).
          - Video pipeline: ``face_detected`` is False.
          - Audio pipeline: ``snr`` below minimum threshold.
        """
        if not health.available:
            return True

        if health.pipeline == "video" and health.face_detected is False:
            return True

        return (
            health.pipeline == "audio"
            and health.snr is not None
            and health.snr < self._min_audio_snr
        )

    # ------------------------------------------------------------------
    # Segment production
    # ------------------------------------------------------------------

    def handle_degraded_input(
        self,
        session_id: str,
        result: ModalityResult,
    ) -> TranscriptSegment | None:
        """Produce a degraded-mode segment from a single modality result.

        Applies a confidence penalty since only one modality contributed.
        Returns None if the penalised confidence drops to zero or below.

        Args:
            session_id: Active session identifier.
            result: The single available modality result.

        Returns:
            A PARTIAL TranscriptSegment with reduced confidence, or None.
        """
        penalised = result.confidence - self._confidence_penalty
        if penalised <= 0.0:
            return None

        return TranscriptSegment(
            segment_id=str(uuid.uuid4()),
            session_id=session_id,
            status=SegmentStatus.PARTIAL,
            text=result.text,
            source=result.modality_type,
            confidence=round(penalised, 4),
            timestamp_ms=result.timestamp_ms,
            duration_ms=result.duration_ms,
            created_at_ms=int(time.time() * 1000),
            replaces_segment_id=None,
        )
