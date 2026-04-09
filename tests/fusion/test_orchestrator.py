"""Tests for FusionOrchestrator."""

from __future__ import annotations

import pytest

from sozia.common.models import (
    ModalityPath,
    ModalityResult,
    ModalityType,
    PipelineHealth,
    SegmentStatus,
)
from sozia.fusion.degraded_mode_handler import DegradedModeHandler
from sozia.fusion.orchestrator import FusionOrchestrator
from sozia.fusion.sign_fusion_policy import SignFusionPolicy
from sozia.fusion.speech_fusion_policy import SpeechFusionPolicy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SESSION = "550e8400-e29b-41d4-a716-446655440000"


def _asr(confidence: float = 0.85) -> ModalityResult:
    return ModalityResult(
        modality_type=ModalityType.ASR,
        text="merhaba",
        confidence=confidence,
        timestamp_ms=1000,
        duration_ms=500,
        inference_latency_ms=200,
    )


def _lip(confidence: float = 0.70) -> ModalityResult:
    return ModalityResult(
        modality_type=ModalityType.LIP_READING,
        text="merhaba dünya",
        confidence=confidence,
        timestamp_ms=1000,
        duration_ms=800,
        inference_latency_ms=600,
    )


def _tsl(confidence: float = 0.80) -> ModalityResult:
    return ModalityResult(
        modality_type=ModalityType.TSL_RECOGNITION,
        text="MERHABA",
        confidence=confidence,
        timestamp_ms=1000,
        duration_ms=600,
        inference_latency_ms=400,
    )


def _gloss(confidence: float = 0.85) -> ModalityResult:
    return ModalityResult(
        modality_type=ModalityType.GLOSS_TO_TEXT,
        text="Merhaba.",
        confidence=confidence,
        timestamp_ms=1000,
        duration_ms=600,
        inference_latency_ms=1500,
    )


def _audio_health(available: bool = True, snr: float = 25.0) -> PipelineHealth:
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
    available: bool = True, face_detected: bool = True,
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


def _make_orchestrator() -> FusionOrchestrator:
    orch = FusionOrchestrator()
    orch.register_policy(ModalityPath.SPEECH, SpeechFusionPolicy())
    orch.register_policy(ModalityPath.SIGN, SignFusionPolicy())
    return orch


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_raises_on_unregistered_path(self):
        orch = FusionOrchestrator()
        with pytest.raises(ValueError, match="SPEECH"):
            orch.process_features(
                _SESSION, [_asr()], [_audio_health()], ModalityPath.SPEECH,
            )

    def test_register_policy_succeeds(self):
        orch = FusionOrchestrator()
        orch.register_policy(ModalityPath.SPEECH, SpeechFusionPolicy())
        # Should not raise anymore.
        segments = orch.process_features(
            _SESSION, [_asr()], [_audio_health()], ModalityPath.SPEECH,
        )
        assert len(segments) == 1


class TestSpeechPathRouting:
    def test_asr_only_produces_partial(self):
        orch = _make_orchestrator()
        segments = orch.process_features(
            _SESSION, [_asr()], [_audio_health()], ModalityPath.SPEECH,
        )
        assert len(segments) == 1
        assert segments[0].status == SegmentStatus.PARTIAL
        assert segments[0].source == ModalityType.ASR

    def test_asr_plus_lip_produces_final(self):
        orch = _make_orchestrator()
        segments = orch.process_features(
            _SESSION, [_asr(), _lip()], [_audio_health()], ModalityPath.SPEECH,
        )
        assert len(segments) == 1
        assert segments[0].status == SegmentStatus.FINAL


class TestSignPathRouting:
    def test_tsl_only_produces_partial(self):
        orch = _make_orchestrator()
        segments = orch.process_features(
            _SESSION, [_tsl()], [_video_health()], ModalityPath.SIGN,
        )
        assert len(segments) == 1
        assert segments[0].status == SegmentStatus.PARTIAL
        assert segments[0].text == "MERHABA"

    def test_gloss_produces_final(self):
        orch = _make_orchestrator()
        segments = orch.process_features(
            _SESSION, [_tsl(), _gloss()], [_video_health()], ModalityPath.SIGN,
        )
        assert len(segments) == 1
        assert segments[0].status == SegmentStatus.FINAL
        assert segments[0].text == "Merhaba."


class TestDegradedMode:
    def test_degraded_audio_single_result_uses_handler(self):
        orch = _make_orchestrator()
        segments = orch.process_features(
            _SESSION,
            [_asr(confidence=0.80)],
            [_audio_health(available=False)],
            ModalityPath.SPEECH,
        )
        assert len(segments) == 1
        seg = segments[0]
        assert seg.status == SegmentStatus.PARTIAL
        # Confidence should be penalised (0.80 - 0.15 = 0.65)
        assert abs(seg.confidence - 0.65) < 0.001

    def test_degraded_video_single_result_uses_handler(self):
        orch = _make_orchestrator()
        segments = orch.process_features(
            _SESSION,
            [_tsl(confidence=0.80)],
            [_video_health(face_detected=False)],
            ModalityPath.SIGN,
        )
        assert len(segments) == 1
        assert abs(segments[0].confidence - 0.65) < 0.001

    def test_degraded_but_two_results_uses_normal_policy(self):
        """With two results, degraded handler is skipped even if health is bad."""
        orch = _make_orchestrator()
        segments = orch.process_features(
            _SESSION,
            [_asr(), _lip()],
            [_audio_health(available=False)],
            ModalityPath.SPEECH,
        )
        assert len(segments) == 1
        assert segments[0].status == SegmentStatus.FINAL

    def test_degraded_low_confidence_returns_empty(self):
        orch = _make_orchestrator()
        segments = orch.process_features(
            _SESSION,
            [_asr(confidence=0.10)],
            [_audio_health(available=False)],
            ModalityPath.SPEECH,
        )
        assert segments == []

    def test_healthy_single_result_uses_normal_policy(self):
        """Healthy pipeline with one result → normal policy, not degraded."""
        orch = _make_orchestrator()
        segments = orch.process_features(
            _SESSION, [_asr()], [_audio_health()], ModalityPath.SPEECH,
        )
        assert len(segments) == 1
        # Normal confidence, no penalty
        assert segments[0].confidence == 0.85


class TestHandleTimeout:
    def test_timeout_emits_partial_from_primary(self):
        orch = _make_orchestrator()
        segments = orch.handle_timeout(
            _SESSION, _asr(), [_audio_health()], ModalityPath.SPEECH,
        )
        assert len(segments) == 1
        assert segments[0].status == SegmentStatus.PARTIAL

    def test_timeout_raises_on_unregistered_path(self):
        orch = FusionOrchestrator()
        with pytest.raises(ValueError, match="SIGN"):
            orch.handle_timeout(
                _SESSION, _tsl(), [_video_health()], ModalityPath.SIGN,
            )
