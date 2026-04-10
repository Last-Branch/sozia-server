"""Tests for SpeechFusionPolicy."""

from __future__ import annotations

from sozia.common.models import (
    ModalityResult,
    ModalityType,
    PipelineHealth,
    SegmentStatus,
)
from sozia.fusion.speech_fusion_policy import SpeechFusionPolicy

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SESSION = "550e8400-e29b-41d4-a716-446655440000"


def _asr(text: str = "merhaba", confidence: float = 0.85) -> ModalityResult:
    return ModalityResult(
        modality_type=ModalityType.ASR,
        text=text,
        confidence=confidence,
        timestamp_ms=1000,
        duration_ms=500,
        inference_latency_ms=200,
    )


def _lip(text: str = "merhaba dünya", confidence: float = 0.70) -> ModalityResult:
    return ModalityResult(
        modality_type=ModalityType.LIP_READING,
        text=text,
        confidence=confidence,
        timestamp_ms=1000,
        duration_ms=800,
        inference_latency_ms=600,
    )


def _health() -> list[PipelineHealth]:
    return [PipelineHealth(
        session_id=_SESSION,
        pipeline="audio",
        available=True,
        fps=None,
        snr=25.0,
        face_detected=None,
        last_updated_ms=1000,
    )]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestShouldSuppress:
    def test_below_threshold_suppressed(self):
        policy = SpeechFusionPolicy()
        assert policy.should_suppress(_asr(confidence=0.20)) is True

    def test_above_threshold_not_suppressed(self):
        policy = SpeechFusionPolicy()
        assert policy.should_suppress(_asr(confidence=0.50)) is False

    def test_exact_threshold_not_suppressed(self):
        policy = SpeechFusionPolicy()
        assert policy.should_suppress(_asr(confidence=0.30)) is False

    def test_confidence_threshold_value(self):
        policy = SpeechFusionPolicy()
        assert policy.get_confidence_threshold() == 0.30


class TestFuseEmpty:
    def test_empty_results_returns_empty(self):
        policy = SpeechFusionPolicy()
        assert policy.fuse([], _health()) == []

    def test_all_suppressed_returns_empty(self):
        policy = SpeechFusionPolicy()
        result = policy.fuse([_asr(confidence=0.10)], _health())
        assert result == []


class TestFuseAsrOnly:
    def test_asr_only_emits_partial(self):
        policy = SpeechFusionPolicy()
        segments = policy.fuse([_asr()], _health())

        assert len(segments) == 1
        seg = segments[0]
        assert seg.status == SegmentStatus.PARTIAL
        assert seg.source == ModalityType.ASR
        assert seg.text == "merhaba"
        assert seg.session_id == _SESSION
        assert seg.replaces_segment_id is None

    def test_asr_only_confidence_passed_through(self):
        policy = SpeechFusionPolicy()
        segments = policy.fuse([_asr(confidence=0.90)], _health())
        assert segments[0].confidence == 0.90


class TestFuseLipOnly:
    def test_lip_only_emits_partial(self):
        policy = SpeechFusionPolicy()
        segments = policy.fuse([_lip()], _health())

        assert len(segments) == 1
        seg = segments[0]
        assert seg.status == SegmentStatus.PARTIAL
        assert seg.source == ModalityType.LIP_READING
        assert seg.text == "merhaba dünya"


class TestFuseBothModalities:
    def test_both_emits_final(self):
        policy = SpeechFusionPolicy()
        segments = policy.fuse(
            [_asr(confidence=0.70), _lip(confidence=0.85)], _health(),
        )

        assert len(segments) == 1
        seg = segments[0]
        assert seg.status == SegmentStatus.FINAL

    def test_weighted_confidence(self):
        policy = SpeechFusionPolicy()
        segments = policy.fuse(
            [_asr(confidence=0.90), _lip(confidence=0.80)], _health(),
        )
        # 0.65 * 0.90 + 0.35 * 0.80 = 0.585 + 0.28 = 0.865
        assert abs(segments[0].confidence - 0.865) < 0.001

    def test_higher_confidence_lip_text_wins(self):
        policy = SpeechFusionPolicy()
        segments = policy.fuse(
            [
                _asr(text="merhaba", confidence=0.70),
                _lip(text="merhaba dünya", confidence=0.85),
            ],
            _health(),
        )
        assert segments[0].text == "merhaba dünya"
        assert segments[0].source == ModalityType.LIP_READING

    def test_higher_confidence_asr_text_wins(self):
        policy = SpeechFusionPolicy()
        segments = policy.fuse(
            [
                _asr(text="merhaba", confidence=0.90),
                _lip(text="merhaba dünya", confidence=0.60),
            ],
            _health(),
        )
        assert segments[0].text == "merhaba"
        assert segments[0].source == ModalityType.ASR

    def test_asr_text_fallback_when_lip_empty(self):
        policy = SpeechFusionPolicy()
        segments = policy.fuse(
            [_asr(text="merhaba"), _lip(text="")], _health(),
        )
        assert segments[0].text == "merhaba"
        assert segments[0].source == ModalityType.ASR

    def test_duration_takes_max(self):
        policy = SpeechFusionPolicy()
        segments = policy.fuse([_asr(), _lip()], _health())
        assert segments[0].duration_ms == 800


class TestFuseReplacesSegmentId:
    def test_partial_then_final_links_via_replaces(self):
        policy = SpeechFusionPolicy()

        # Step 1: ASR only → PARTIAL
        partials = policy.fuse([_asr()], _health())
        partial_id = partials[0].segment_id

        # Step 2: ASR + lip → FINAL should reference the PARTIAL
        finals = policy.fuse([_asr(), _lip()], _health())
        assert finals[0].replaces_segment_id == partial_id

    def test_final_without_prior_partial_has_no_replaces(self):
        policy = SpeechFusionPolicy()
        segments = policy.fuse([_asr(), _lip()], _health())
        assert segments[0].replaces_segment_id is None

    def test_replaces_consumed_only_once(self):
        policy = SpeechFusionPolicy()

        policy.fuse([_asr()], _health())
        policy.fuse([_asr(), _lip()], _health())

        # Second FINAL should not have a replaces_segment_id
        segments = policy.fuse([_asr(), _lip()], _health())
        assert segments[0].replaces_segment_id is None


class TestFuseSuppressedModality:
    def test_low_confidence_lip_ignored_asr_emits_partial(self):
        policy = SpeechFusionPolicy()
        segments = policy.fuse(
            [_asr(confidence=0.80), _lip(confidence=0.10)], _health(),
        )
        assert len(segments) == 1
        assert segments[0].status == SegmentStatus.PARTIAL
        assert segments[0].source == ModalityType.ASR

    def test_low_confidence_asr_ignored_lip_emits_partial(self):
        policy = SpeechFusionPolicy()
        segments = policy.fuse(
            [_asr(confidence=0.10), _lip(confidence=0.80)], _health(),
        )
        assert len(segments) == 1
        assert segments[0].status == SegmentStatus.PARTIAL
        assert segments[0].source == ModalityType.LIP_READING


class TestSegmentIdFormat:
    def test_segment_id_is_dashed_uuid_v4(self):
        """segment_id must be a canonical UUID v4 string (with dashes)."""
        import uuid as _uuid

        policy = SpeechFusionPolicy()
        segments = policy.fuse([_asr()], _health())
        assert len(segments) == 1

        seg_id = segments[0].segment_id
        assert "-" in seg_id, "segment_id must keep UUID dashes"
        # Parsing as UUID confirms canonical format.
        parsed = _uuid.UUID(seg_id)
        assert parsed.version == 4
