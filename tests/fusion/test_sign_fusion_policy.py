"""Tests for SignFusionPolicy."""

from __future__ import annotations

from sozia.common.models import (
    ModalityResult,
    ModalityType,
    PipelineHealth,
    SegmentStatus,
)
from sozia.server.fusion.sign_fusion_policy import SignFusionPolicy

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SESSION = "550e8400-e29b-41d4-a716-446655440000"


def _tsl(text: str = "MERHABA DUNYA", confidence: float = 0.80) -> ModalityResult:
    return ModalityResult(
        modality_type=ModalityType.TSL_RECOGNITION,
        text=text,
        confidence=confidence,
        timestamp_ms=1000,
        duration_ms=600,
        inference_latency_ms=400,
    )


def _gloss(text: str = "Merhaba dünya.", confidence: float = 0.85) -> ModalityResult:
    return ModalityResult(
        modality_type=ModalityType.GLOSS_TO_TEXT,
        text=text,
        confidence=confidence,
        timestamp_ms=1000,
        duration_ms=600,
        inference_latency_ms=1500,
    )


def _health() -> list[PipelineHealth]:
    return [PipelineHealth(
        session_id=_SESSION,
        pipeline="video",
        available=True,
        fps=30.0,
        snr=None,
        face_detected=True,
        last_updated_ms=1000,
    )]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestShouldSuppress:
    def test_below_threshold_suppressed(self):
        policy = SignFusionPolicy()
        assert policy.should_suppress(_tsl(confidence=0.20)) is True

    def test_above_threshold_not_suppressed(self):
        policy = SignFusionPolicy()
        assert policy.should_suppress(_tsl(confidence=0.50)) is False

    def test_confidence_threshold_value(self):
        policy = SignFusionPolicy()
        assert policy.get_confidence_threshold() == 0.30


class TestFuseEmpty:
    def test_empty_results_returns_empty(self):
        policy = SignFusionPolicy()
        assert policy.fuse([], _health()) == []

    def test_all_suppressed_returns_empty(self):
        policy = SignFusionPolicy()
        assert policy.fuse([_tsl(confidence=0.10)], _health()) == []


class TestFuseTslOnly:
    def test_tsl_only_emits_partial_with_raw_gloss(self):
        policy = SignFusionPolicy()
        segments = policy.fuse([_tsl()], _health())

        assert len(segments) == 1
        seg = segments[0]
        assert seg.status == SegmentStatus.PARTIAL
        assert seg.source == ModalityType.TSL_RECOGNITION
        assert seg.text == "MERHABA DUNYA"
        assert seg.session_id == _SESSION
        assert seg.replaces_segment_id is None

    def test_tsl_confidence_passed_through(self):
        policy = SignFusionPolicy()
        segments = policy.fuse([_tsl(confidence=0.75)], _health())
        assert segments[0].confidence == 0.75


class TestFuseGlossToText:
    def test_gloss_emits_final(self):
        policy = SignFusionPolicy()
        segments = policy.fuse([_gloss()], _health())

        assert len(segments) == 1
        seg = segments[0]
        assert seg.status == SegmentStatus.FINAL
        assert seg.source == ModalityType.GLOSS_TO_TEXT
        assert seg.text == "Merhaba dünya."

    def test_gloss_confidence_used_directly(self):
        """No weighted merge — GlossToText confidence is used as-is."""
        policy = SignFusionPolicy()
        segments = policy.fuse([_gloss(confidence=0.90)], _health())
        assert segments[0].confidence == 0.90

    def test_both_present_gloss_wins(self):
        """When both TSL and GlossToText are valid, FINAL with GlossToText."""
        policy = SignFusionPolicy()
        segments = policy.fuse([_tsl(), _gloss()], _health())

        assert len(segments) == 1
        assert segments[0].status == SegmentStatus.FINAL
        assert segments[0].text == "Merhaba dünya."


class TestFuseReplacesSegmentId:
    def test_partial_then_final_links_via_replaces(self):
        policy = SignFusionPolicy()

        # Step 1: TSL only → PARTIAL
        partials = policy.fuse([_tsl()], _health())
        partial_id = partials[0].segment_id

        # Step 2: GlossToText → FINAL should reference the PARTIAL
        finals = policy.fuse([_gloss()], _health())
        assert finals[0].replaces_segment_id == partial_id

    def test_final_without_prior_partial_has_no_replaces(self):
        policy = SignFusionPolicy()
        segments = policy.fuse([_gloss()], _health())
        assert segments[0].replaces_segment_id is None

    def test_replaces_consumed_only_once(self):
        policy = SignFusionPolicy()

        policy.fuse([_tsl()], _health())
        policy.fuse([_gloss()], _health())

        # Second FINAL should have no replaces
        segments = policy.fuse([_gloss()], _health())
        assert segments[0].replaces_segment_id is None


class TestFuseSuppressedModality:
    def test_low_confidence_gloss_ignored_tsl_emits_partial(self):
        policy = SignFusionPolicy()
        segments = policy.fuse(
            [_tsl(confidence=0.80), _gloss(confidence=0.10)], _health(),
        )
        assert len(segments) == 1
        assert segments[0].status == SegmentStatus.PARTIAL
        assert segments[0].source == ModalityType.TSL_RECOGNITION

    def test_low_confidence_tsl_with_valid_gloss_emits_final(self):
        policy = SignFusionPolicy()
        segments = policy.fuse(
            [_tsl(confidence=0.10), _gloss(confidence=0.85)], _health(),
        )
        assert len(segments) == 1
        assert segments[0].status == SegmentStatus.FINAL
        assert segments[0].source == ModalityType.GLOSS_TO_TEXT

    def test_unrelated_modality_returns_empty(self):
        """Sign policy must ignore non-sign modalities (ASR/LipReading)."""
        policy = SignFusionPolicy()
        asr_like = ModalityResult(
            modality_type=ModalityType.ASR,
            text="merhaba",
            confidence=0.80,
            timestamp_ms=1000,
            duration_ms=500,
            inference_latency_ms=200,
        )
        assert policy.fuse([asr_like], _health()) == []


class TestSegmentIdFormat:
    def test_segment_id_is_dashed_uuid_v4(self):
        """segment_id must be a canonical UUID v4 string (with dashes)."""
        import uuid as _uuid

        policy = SignFusionPolicy()
        segments = policy.fuse([_tsl()], _health())
        assert len(segments) == 1

        seg_id = segments[0].segment_id
        assert "-" in seg_id
        assert _uuid.UUID(seg_id).version == 4
