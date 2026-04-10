"""Tests for DegradedModeHandler."""

from __future__ import annotations

from sozia.common.models import (
    ModalityResult,
    ModalityType,
    PipelineHealth,
    SegmentStatus,
)
from sozia.fusion.degraded_mode_handler import (
    DegradedModeHandler,
    DegradedStatus,
)

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


class TestEvaluate:
    def test_empty_health_is_failed(self):
        handler = DegradedModeHandler()
        assert handler.evaluate([]) == DegradedStatus.FAILED

    def test_all_healthy_is_normal(self):
        handler = DegradedModeHandler()
        status = handler.evaluate([_audio_health(), _video_health()])
        assert status == DegradedStatus.NORMAL

    def test_one_down_is_degraded(self):
        handler = DegradedModeHandler()
        status = handler.evaluate(
            [_audio_health(available=False), _video_health()],
        )
        assert status == DegradedStatus.DEGRADED

    def test_all_down_is_failed(self):
        handler = DegradedModeHandler()
        status = handler.evaluate(
            [
                _audio_health(available=False),
                _video_health(available=False),
            ],
        )
        assert status == DegradedStatus.FAILED

    def test_video_obstructed_counts_as_degraded(self):
        handler = DegradedModeHandler()
        status = handler.evaluate(
            [_audio_health(), _video_health(face_detected=False)],
        )
        assert status == DegradedStatus.DEGRADED


class TestAdvisoryMessage:
    def test_all_healthy_returns_none(self):
        handler = DegradedModeHandler()
        assert handler.get_advisory_message(
            [_audio_health(), _video_health()],
        ) is None

    def test_empty_health_returns_terminal_message(self):
        handler = DegradedModeHandler()
        msg = handler.get_advisory_message([])
        assert msg is not None
        assert "session ending" in msg.lower()

    def test_camera_obstructed_message(self):
        handler = DegradedModeHandler()
        msg = handler.get_advisory_message(
            [_video_health(face_detected=False)],
        )
        assert msg is not None
        assert "camera obstructed" in msg.lower()

    def test_microphone_unavailable_message(self):
        handler = DegradedModeHandler()
        msg = handler.get_advisory_message(
            [_audio_health(available=False)],
        )
        assert msg is not None
        assert "microphone unavailable" in msg.lower()

    def test_low_snr_message(self):
        handler = DegradedModeHandler()
        msg = handler.get_advisory_message([_audio_health(snr=2.0)])
        assert msg is not None
        assert "low audio" in msg.lower()

    def test_camera_unavailable_message(self):
        handler = DegradedModeHandler()
        msg = handler.get_advisory_message(
            [_video_health(available=False)],
        )
        assert msg is not None
        assert "camera unavailable" in msg.lower()


class TestSegmentIdFormat:
    def test_degraded_segment_id_is_dashed_uuid_v4(self):
        import uuid as _uuid

        handler = DegradedModeHandler()
        seg = handler.handle_degraded_input(_SESSION, _result())
        assert seg is not None
        assert "-" in seg.segment_id
        assert _uuid.UUID(seg.segment_id).version == 4
