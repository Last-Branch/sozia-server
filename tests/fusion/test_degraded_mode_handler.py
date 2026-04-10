"""Tests for DegradedModeHandler."""

from __future__ import annotations

from sozia.common.models import (
    ModalityResult,
    ModalityType,
    PipelineHealth,
    SegmentStatus,
)
from sozia.fusion.degraded_mode_handler import DegradedModeHandler

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SESSION = "550e8400-e29b-41d4-a716-446655440000"


def _audio_health(
    available: bool = True, snr: float | None = 25.0,
) -> PipelineHealth:
    return PipelineHealth(
        session_id=_SESSION,
        pipeline="audio",
        available=available,
        fps=None,
        snr=snr,
        face_detected=None,
        last_updated_ms=1000,
    )


def _video_health(
    available: bool = True, face_detected: bool | None = True,
) -> PipelineHealth:
    return PipelineHealth(
        session_id=_SESSION,
        pipeline="video",
        available=available,
        fps=30.0,
        snr=None,
        face_detected=face_detected,
        last_updated_ms=1000,
    )


def _result(
    modality: ModalityType = ModalityType.ASR,
    text: str = "merhaba",
    confidence: float = 0.80,
) -> ModalityResult:
    return ModalityResult(
        modality_type=modality,
        text=text,
        confidence=confidence,
        timestamp_ms=1000,
        duration_ms=500,
        inference_latency_ms=200,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestShouldFallback:
    def test_healthy_audio_no_fallback(self):
        handler = DegradedModeHandler()
        assert handler.should_fallback(_audio_health()) is False

    def test_healthy_video_no_fallback(self):
        handler = DegradedModeHandler()
        assert handler.should_fallback(_video_health()) is False

    def test_unavailable_audio_triggers_fallback(self):
        handler = DegradedModeHandler()
        assert handler.should_fallback(_audio_health(available=False)) is True

    def test_unavailable_video_triggers_fallback(self):
        handler = DegradedModeHandler()
        assert handler.should_fallback(_video_health(available=False)) is True

    def test_no_face_detected_triggers_fallback(self):
        handler = DegradedModeHandler()
        assert handler.should_fallback(_video_health(face_detected=False)) is True

    def test_face_detected_none_no_fallback(self):
        """face_detected=None means detection isn't applicable, not degraded."""
        handler = DegradedModeHandler()
        assert handler.should_fallback(_video_health(face_detected=None)) is False

    def test_low_snr_triggers_fallback(self):
        handler = DegradedModeHandler()
        assert handler.should_fallback(_audio_health(snr=2.0)) is True

    def test_good_snr_no_fallback(self):
        handler = DegradedModeHandler()
        assert handler.should_fallback(_audio_health(snr=20.0)) is False

    def test_snr_at_threshold_no_fallback(self):
        handler = DegradedModeHandler()
        assert handler.should_fallback(_audio_health(snr=5.0)) is False


class TestHandleDegradedInput:
    def test_returns_partial_segment(self):
        handler = DegradedModeHandler()
        seg = handler.handle_degraded_input(_SESSION, _result())

        assert seg is not None
        assert seg.status == SegmentStatus.PARTIAL
        assert seg.text == "merhaba"
        assert seg.session_id == _SESSION
        assert seg.replaces_segment_id is None

    def test_confidence_penalised(self):
        handler = DegradedModeHandler()
        seg = handler.handle_degraded_input(
            _SESSION, _result(confidence=0.80),
        )
        # 0.80 - 0.15 = 0.65
        assert seg is not None
        assert abs(seg.confidence - 0.65) < 0.001

    def test_low_confidence_returns_none(self):
        """If penalty drops confidence to zero, return None."""
        handler = DegradedModeHandler()
        seg = handler.handle_degraded_input(
            _SESSION, _result(confidence=0.10),
        )
        assert seg is None

    def test_exactly_penalty_returns_none(self):
        """Confidence exactly equal to penalty → 0.0 → None."""
        handler = DegradedModeHandler()
        seg = handler.handle_degraded_input(
            _SESSION, _result(confidence=0.15),
        )
        assert seg is None

    def test_preserves_source_modality(self):
        handler = DegradedModeHandler()
        seg = handler.handle_degraded_input(
            _SESSION, _result(modality=ModalityType.LIP_READING),
        )
        assert seg is not None
        assert seg.source == ModalityType.LIP_READING

    def test_preserves_timing_fields(self):
        handler = DegradedModeHandler()
        seg = handler.handle_degraded_input(
            _SESSION, _result(),
        )
        assert seg is not None
        assert seg.timestamp_ms == 1000
        assert seg.duration_ms == 500
