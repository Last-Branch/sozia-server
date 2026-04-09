"""DegradedModeHandler — fallback when one modality is unavailable.

When a pipeline becomes unavailable (device lost, no face detected, low SNR)
or a modality times out, this handler decides whether to fall back to
single-modality output with reduced confidence.
"""

from __future__ import annotations

import time
import uuid

from sozia.common.models import (
    ModalityResult,
    ModalityType,
    PipelineHealth,
    SegmentStatus,
    TranscriptSegment,
)

# Confidence penalty applied to degraded (single-modality) output.
_DEGRADED_CONFIDENCE_PENALTY = 0.15

# Minimum SNR (dB) for the audio pipeline to be considered usable.
_MIN_AUDIO_SNR = 5.0


class DegradedModeHandler:
    """Decides when to enter degraded mode and produces fallback segments."""

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

        if (
            health.pipeline == "audio"
            and health.snr is not None
            and health.snr < _MIN_AUDIO_SNR
        ):
            return True

        return False

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
        penalised = result.confidence - _DEGRADED_CONFIDENCE_PENALTY
        if penalised <= 0.0:
            return None

        return TranscriptSegment(
            segment_id=uuid.uuid4().hex,
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
