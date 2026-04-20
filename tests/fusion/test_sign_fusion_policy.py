"""Tests for SignFusionPolicy."""

from __future__ import annotations

import uuid

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
_PREV_PARTIAL_ID = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"


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
    return [
        PipelineHealth(
            session_id=_SESSION,
            pipeline="video",
            available=True,
            fps=30.0,
            snr=None,
            face_detected=True,
            last_updated_ms=1000,
        )
    ]


# ---------------------------------------------------------------------------
# TSL suppress gate (threshold = 0.55)
# ---------------------------------------------------------------------------


class TestShouldSuppress:
    def test_below_threshold_suppressed(self):
        policy = SignFusionPolicy()
        assert policy.should_suppress(_tsl(confidence=0.20)) is True

    def test_at_threshold_not_suppressed(self):
        policy = SignFusionPolicy()
        assert policy.should_suppress(_tsl(confidence=0.55)) is False

    def test_above_threshold_not_suppressed(self):
        policy = SignFusionPolicy()
        assert policy.should_suppress(_tsl(confidence=0.70)) is False

    def test_tsl_confidence_threshold_value(self):
        policy = SignFusionPolicy()
        assert policy.get_confidence_threshold() == 0.55

    def test_custom_tsl_threshold_respected(self):
        policy = SignFusionPolicy(confidence_threshold=0.60)
        assert policy.should_suppress(_tsl(confidence=0.58)) is True
        assert policy.should_suppress(_tsl(confidence=0.61)) is False


# ---------------------------------------------------------------------------
# LLM suppress gate (threshold = 0.40)
# ---------------------------------------------------------------------------


class TestShouldSuppressLlm:
    def test_below_llm_threshold_suppressed(self):
        policy = SignFusionPolicy()
        assert policy.should_suppress_llm(_gloss(confidence=0.30)) is True

    def test_at_llm_threshold_not_suppressed(self):
        policy = SignFusionPolicy()
        assert policy.should_suppress_llm(_gloss(confidence=0.40)) is False

    def test_above_llm_threshold_not_suppressed(self):
        policy = SignFusionPolicy()
        assert policy.should_suppress_llm(_gloss(confidence=0.85)) is False

    def test_llm_threshold_value(self):
        policy = SignFusionPolicy()
        assert policy.get_llm_confidence_threshold() == 0.40

    def test_custom_llm_threshold_respected(self):
        policy = SignFusionPolicy(llm_confidence_threshold=0.50)
        assert policy.should_suppress_llm(_gloss(confidence=0.45)) is True
        assert policy.should_suppress_llm(_gloss(confidence=0.55)) is False

    def test_llm_threshold_lower_than_tsl_threshold(self):
        policy = SignFusionPolicy()
        # LLM conf-exp scores are lower than TSL softmax; verify ordering.
        assert policy.get_llm_confidence_threshold() < policy.get_confidence_threshold()


# ---------------------------------------------------------------------------
# emit_partial helper
# ---------------------------------------------------------------------------


class TestEmitPartial:
    def test_returns_partial_segment(self):
        policy = SignFusionPolicy()
        seg = policy.emit_partial(_SESSION, "MERHABA DUNYA", _tsl(), replaces_id=None)
        assert seg.status == SegmentStatus.PARTIAL

    def test_text_is_gloss_phrase_not_raw_result(self):
        # Phrase passed in may be multi-word accumulated phrase.
        policy = SignFusionPolicy()
        seg = policy.emit_partial(
            _SESSION, "YEMEK OKUL", _tsl(text="OKUL"), replaces_id=None
        )
        assert seg.text == "YEMEK OKUL"

    def test_source_is_tsl_recognition(self):
        policy = SignFusionPolicy()
        seg = policy.emit_partial(_SESSION, "X", _tsl(), replaces_id=None)
        assert seg.source == ModalityType.TSL_RECOGNITION

    def test_confidence_from_tsl_result(self):
        policy = SignFusionPolicy()
        seg = policy.emit_partial(
            _SESSION, "X", _tsl(confidence=0.72), replaces_id=None
        )
        assert seg.confidence == 0.72

    def test_session_id_set(self):
        policy = SignFusionPolicy()
        seg = policy.emit_partial(_SESSION, "X", _tsl(), replaces_id=None)
        assert seg.session_id == _SESSION

    def test_segment_id_is_valid_uuid_v4(self):
        policy = SignFusionPolicy()
        seg = policy.emit_partial(_SESSION, "X", _tsl(), replaces_id=None)
        assert uuid.UUID(seg.segment_id).version == 4

    def test_replaces_id_none_when_no_prior_partial(self):
        policy = SignFusionPolicy()
        seg = policy.emit_partial(_SESSION, "X", _tsl(), replaces_id=None)
        assert seg.replaces_segment_id is None

    def test_replaces_id_forwarded_when_provided(self):
        policy = SignFusionPolicy()
        seg = policy.emit_partial(_SESSION, "X", _tsl(), replaces_id=_PREV_PARTIAL_ID)
        assert seg.replaces_segment_id == _PREV_PARTIAL_ID

    def test_two_calls_return_different_segment_ids(self):
        policy = SignFusionPolicy()
        s1 = policy.emit_partial(_SESSION, "A", _tsl(), replaces_id=None)
        s2 = policy.emit_partial(_SESSION, "B", _tsl(), replaces_id=None)
        assert s1.segment_id != s2.segment_id


# ---------------------------------------------------------------------------
# emit_final_from_llm helper
# ---------------------------------------------------------------------------


class TestEmitFinalFromLlm:
    def test_returns_final_segment(self):
        policy = SignFusionPolicy()
        seg = policy.emit_final_from_llm(
            _SESSION, "MERHABA", _gloss(), replaces_id=None
        )
        assert seg.status == SegmentStatus.FINAL

    def test_text_is_llm_output(self):
        policy = SignFusionPolicy()
        seg = policy.emit_final_from_llm(
            _SESSION, "MERHABA", _gloss(text="Merhaba dünya."), replaces_id=None
        )
        assert seg.text == "Merhaba dünya."

    def test_source_is_gloss_to_text(self):
        policy = SignFusionPolicy()
        seg = policy.emit_final_from_llm(_SESSION, "X", _gloss(), replaces_id=None)
        assert seg.source == ModalityType.GLOSS_TO_TEXT

    def test_confidence_from_llm_result(self):
        policy = SignFusionPolicy()
        seg = policy.emit_final_from_llm(
            _SESSION, "X", _gloss(confidence=0.73), replaces_id=None
        )
        assert seg.confidence == 0.73

    def test_replaces_id_forwarded(self):
        policy = SignFusionPolicy()
        seg = policy.emit_final_from_llm(
            _SESSION, "X", _gloss(), replaces_id=_PREV_PARTIAL_ID
        )
        assert seg.replaces_segment_id == _PREV_PARTIAL_ID

    def test_replaces_id_none_when_omitted(self):
        policy = SignFusionPolicy()
        seg = policy.emit_final_from_llm(_SESSION, "X", _gloss(), replaces_id=None)
        assert seg.replaces_segment_id is None


# ---------------------------------------------------------------------------
# emit_final_from_gloss helper (timeout / low-confidence LLM fallback)
# ---------------------------------------------------------------------------


class TestEmitFinalFromGloss:
    def test_returns_final_segment(self):
        policy = SignFusionPolicy()
        seg = policy.emit_final_from_gloss(
            _SESSION,
            "YEMEK OKUL",
            aggregate_conf=0.60,
            timestamp_ms=1000,
            duration_ms=2000,
            replaces_id=None,
        )
        assert seg.status == SegmentStatus.FINAL

    def test_text_is_raw_gloss_phrase(self):
        policy = SignFusionPolicy()
        seg = policy.emit_final_from_gloss(
            _SESSION,
            "YEMEK OKUL",
            aggregate_conf=0.60,
            timestamp_ms=1000,
            duration_ms=2000,
            replaces_id=None,
        )
        assert seg.text == "YEMEK OKUL"

    def test_source_is_tsl_recognition(self):
        # Raw-gloss fallback still originates from TSL, not GlossToText.
        policy = SignFusionPolicy()
        seg = policy.emit_final_from_gloss(
            _SESSION,
            "X",
            aggregate_conf=0.60,
            timestamp_ms=1000,
            duration_ms=2000,
            replaces_id=None,
        )
        assert seg.source == ModalityType.TSL_RECOGNITION

    def test_aggregate_confidence_used(self):
        policy = SignFusionPolicy()
        seg = policy.emit_final_from_gloss(
            _SESSION,
            "X",
            aggregate_conf=0.58,
            timestamp_ms=1000,
            duration_ms=2000,
            replaces_id=None,
        )
        assert seg.confidence == 0.58

    def test_timestamp_and_duration_forwarded(self):
        policy = SignFusionPolicy()
        seg = policy.emit_final_from_gloss(
            _SESSION,
            "X",
            aggregate_conf=0.60,
            timestamp_ms=5000,
            duration_ms=3000,
            replaces_id=None,
        )
        assert seg.timestamp_ms == 5000
        assert seg.duration_ms == 3000

    def test_replaces_id_forwarded(self):
        policy = SignFusionPolicy()
        seg = policy.emit_final_from_gloss(
            _SESSION,
            "X",
            aggregate_conf=0.60,
            timestamp_ms=1000,
            duration_ms=2000,
            replaces_id=_PREV_PARTIAL_ID,
        )
        assert seg.replaces_segment_id == _PREV_PARTIAL_ID


# ---------------------------------------------------------------------------
# fuse() — kept for backward compat / integration; no longer tracks partial IDs
# ---------------------------------------------------------------------------


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
        policy = SignFusionPolicy()
        segments = policy.fuse([_gloss(confidence=0.90)], _health())
        assert segments[0].confidence == 0.90

    def test_both_present_gloss_wins(self):
        policy = SignFusionPolicy()
        segments = policy.fuse([_tsl(), _gloss()], _health())

        assert len(segments) == 1
        assert segments[0].status == SegmentStatus.FINAL
        assert segments[0].text == "Merhaba dünya."


class TestFuseSuppressedModality:
    def test_low_confidence_gloss_ignored_tsl_emits_partial(self):
        policy = SignFusionPolicy()
        segments = policy.fuse(
            [_tsl(confidence=0.80), _gloss(confidence=0.10)],
            _health(),
        )
        assert len(segments) == 1
        assert segments[0].status == SegmentStatus.PARTIAL
        assert segments[0].source == ModalityType.TSL_RECOGNITION

    def test_low_confidence_tsl_with_valid_gloss_emits_final(self):
        policy = SignFusionPolicy()
        segments = policy.fuse(
            [_tsl(confidence=0.10), _gloss(confidence=0.85)],
            _health(),
        )
        assert len(segments) == 1
        assert segments[0].status == SegmentStatus.FINAL
        assert segments[0].source == ModalityType.GLOSS_TO_TEXT

    def test_unrelated_modality_returns_empty(self):
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

    def test_gloss_between_llm_and_tsl_thresholds_not_suppressed(self):
        # Confidence 0.45: above LLM threshold (0.40) but below TSL threshold (0.55).
        # fuse() uses should_suppress_llm() for gloss results, so it should pass.
        policy = SignFusionPolicy()
        segments = policy.fuse([_gloss(confidence=0.45)], _health())
        assert len(segments) == 1
        assert segments[0].status == SegmentStatus.FINAL


class TestSegmentIdFormat:
    def test_segment_id_is_dashed_uuid_v4(self):
        policy = SignFusionPolicy()
        segments = policy.fuse([_tsl()], _health())
        assert len(segments) == 1

        seg_id = segments[0].segment_id
        assert "-" in seg_id
        assert uuid.UUID(seg_id).version == 4
