"""Integration tests wiring mock inference engines into the fusion layer.

These tests exercise the full contract between ``sozia.inference`` and
``sozia.fusion`` without loading real ML models. A minimal in-memory
``InferenceEngine`` subclass returns canned ``ModalityResult`` objects, which
flow through the registered ``FusionStrategy`` via the ``FusionOrchestrator``.

The goal is to catch contract drift between the two layers Aylin owns —
e.g. a ``ModalityResult`` field rename or a policy assuming fields the
engines never populate.
"""

from __future__ import annotations

import asyncio

import pytest

from sozia.common.interfaces import InferenceEngine
from sozia.common.models import (
    ModalityPath,
    ModalityResult,
    ModalityType,
    ModelConfig,
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


class _StubEngine(InferenceEngine):
    """Minimal InferenceEngine that returns a pre-canned ModalityResult.

    Does not load any real weights — satisfies R2 (only inference loads
    models) by declaring itself as inference layer, while keeping the test
    hermetic.
    """

    def __init__(self, result: ModalityResult, model_id: str = "stub") -> None:
        self._result = result
        self._model_id = model_id
        self._loaded = False

    async def load_model(self, config: ModelConfig) -> None:
        self._loaded = True
        self._model_id = config.model_id

    async def predict(
        self, features, timeout_ms: int = 2200,
    ) -> ModalityResult:
        if not self._loaded:
            from sozia.common.interfaces import ModelNotLoadedError
            raise ModelNotLoadedError(self._model_id)
        return self._result

    async def unload_model(self) -> None:
        self._loaded = False

    def is_loaded(self) -> bool:
        return self._loaded

    def get_model_id(self) -> str:
        return self._model_id if self._loaded else ""


def _result(
    modality: ModalityType,
    text: str,
    confidence: float = 0.85,
    timestamp_ms: int = 1000,
    duration_ms: int = 500,
    inference_latency_ms: int = 200,
) -> ModalityResult:
    return ModalityResult(
        modality_type=modality,
        text=text,
        confidence=confidence,
        timestamp_ms=timestamp_ms,
        duration_ms=duration_ms,
        inference_latency_ms=inference_latency_ms,
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


def _config(model_id: str) -> ModelConfig:
    return ModelConfig(
        model_id=model_id,
        weights_path="/tmp/noop",
        device="cpu",
        params={},
    )


def _build_orchestrator() -> FusionOrchestrator:
    orch = FusionOrchestrator(degraded_handler=DegradedModeHandler())
    orch.register_policy(ModalityPath.SPEECH, SpeechFusionPolicy())
    orch.register_policy(ModalityPath.SIGN, SignFusionPolicy())
    return orch


# ---------------------------------------------------------------------------
# Speech path: ASR → PARTIAL, then ASR + Lip → FINAL
# ---------------------------------------------------------------------------


class TestSpeechPipelineIntegration:
    def test_partial_then_final_replaces_partial(self):
        """ASR alone → PARTIAL; lip arrives → FINAL with replaces_segment_id."""
        asr = _StubEngine(_result(ModalityType.ASR, "merhaba", 0.85))
        lip = _StubEngine(
            _result(ModalityType.LIP_READING, "merhaba dünya", 0.70),
        )
        asyncio.run(asr.load_model(_config("whisper-small-tr")))
        asyncio.run(lip.load_model(_config("lip-reading-v1")))

        asr_out = asyncio.run(asr.predict(features=None))
        lip_out = asyncio.run(lip.predict(features=None))

        orch = _build_orchestrator()

        # First pass: ASR only → PARTIAL.
        partials = orch.process_features(
            _SESSION, [asr_out], [_audio_health()], ModalityPath.SPEECH,
        )
        assert len(partials) == 1
        assert partials[0].status == SegmentStatus.PARTIAL
        partial_id = partials[0].segment_id

        # Second pass: both modalities → FINAL replacing PARTIAL.
        finals = orch.process_features(
            _SESSION,
            [asr_out, lip_out],
            [_audio_health()],
            ModalityPath.SPEECH,
        )
        assert len(finals) == 1
        assert finals[0].status == SegmentStatus.FINAL
        assert finals[0].replaces_segment_id == partial_id
        # Lip text takes precedence when both are present.
        assert finals[0].text == "merhaba dünya"

    def test_low_confidence_asr_suppressed(self):
        asr = _StubEngine(_result(ModalityType.ASR, "noise", 0.10))
        asyncio.run(asr.load_model(_config("whisper-small-tr")))
        asr_out = asyncio.run(asr.predict(features=None))

        orch = _build_orchestrator()
        segments = orch.process_features(
            _SESSION, [asr_out], [_audio_health()], ModalityPath.SPEECH,
        )
        assert segments == []


# ---------------------------------------------------------------------------
# Sign path: TSL → PARTIAL, then GlossToText → FINAL
# ---------------------------------------------------------------------------


class TestSignPipelineIntegration:
    def test_gloss_replaces_tsl_partial(self):
        tsl = _StubEngine(
            _result(ModalityType.TSL_RECOGNITION, "MERHABA DUNYA", 0.80),
        )
        gloss = _StubEngine(
            _result(
                ModalityType.GLOSS_TO_TEXT,
                "Merhaba dünya.",
                0.88,
                inference_latency_ms=1500,
            ),
        )
        asyncio.run(tsl.load_model(_config("tsl-gru-v1")))
        asyncio.run(gloss.load_model(_config("gemma2-9b-lora")))

        tsl_out = asyncio.run(tsl.predict(features=None))
        gloss_out = asyncio.run(gloss.predict(features=None))

        orch = _build_orchestrator()

        # TSL alone → PARTIAL.
        partials = orch.process_features(
            _SESSION, [tsl_out], [_video_health()], ModalityPath.SIGN,
        )
        assert len(partials) == 1
        assert partials[0].status == SegmentStatus.PARTIAL
        assert partials[0].text == "MERHABA DUNYA"
        partial_id = partials[0].segment_id

        # GlossToText arrives → FINAL replaces PARTIAL, text fully replaced.
        finals = orch.process_features(
            _SESSION,
            [tsl_out, gloss_out],
            [_video_health()],
            ModalityPath.SIGN,
        )
        assert len(finals) == 1
        assert finals[0].status == SegmentStatus.FINAL
        assert finals[0].replaces_segment_id == partial_id
        assert finals[0].text == "Merhaba dünya."


# ---------------------------------------------------------------------------
# Degraded mode: one pipeline unhealthy, single modality falls back
# ---------------------------------------------------------------------------


class TestDegradedIntegration:
    def test_no_face_detected_falls_back_to_audio(self):
        """Video pipeline reports no face → fusion must still emit audio."""
        asr = _StubEngine(_result(ModalityType.ASR, "merhaba", 0.80))
        asyncio.run(asr.load_model(_config("whisper-small-tr")))
        asr_out = asyncio.run(asr.predict(features=None))

        orch = _build_orchestrator()
        # Audio healthy, video degraded (no face), only ASR result.
        segments = orch.process_features(
            _SESSION,
            [asr_out],
            [_audio_health(), _video_health(face_detected=False)],
            ModalityPath.SPEECH,
        )
        assert len(segments) == 1
        seg = segments[0]
        assert seg.status == SegmentStatus.PARTIAL
        # Degraded handler applies confidence penalty (0.80 - 0.15 = 0.65).
        assert abs(seg.confidence - 0.65) < 0.001

    def test_low_snr_penalises_asr(self):
        asr = _StubEngine(_result(ModalityType.ASR, "fısıltı", 0.70))
        asyncio.run(asr.load_model(_config("whisper-small-tr")))
        asr_out = asyncio.run(asr.predict(features=None))

        orch = _build_orchestrator()
        segments = orch.process_features(
            _SESSION,
            [asr_out],
            [_audio_health(snr=2.0)],
            ModalityPath.SPEECH,
        )
        assert len(segments) == 1
        # 0.70 - 0.15 = 0.55
        assert abs(segments[0].confidence - 0.55) < 0.001


# ---------------------------------------------------------------------------
# Engine lifecycle + fusion contract
# ---------------------------------------------------------------------------


class TestEngineLifecycleContract:
    def test_predict_before_load_raises(self):
        from sozia.common.interfaces import ModelNotLoadedError

        engine = _StubEngine(_result(ModalityType.ASR, "x"))
        with pytest.raises(ModelNotLoadedError):
            asyncio.run(engine.predict(features=None))

    def test_unload_resets_state(self):
        engine = _StubEngine(_result(ModalityType.ASR, "x"))
        asyncio.run(engine.load_model(_config("m")))
        assert engine.is_loaded() is True
        assert engine.get_model_id() == "m"

        asyncio.run(engine.unload_model())
        assert engine.is_loaded() is False
        assert engine.get_model_id() == ""

    def test_custom_policy_threshold_is_honoured(self):
        """Refactored policies expose constants — verify override flows end-to-end."""
        strict = SpeechFusionPolicy(confidence_threshold=0.90)
        orch = FusionOrchestrator()
        orch.register_policy(ModalityPath.SPEECH, strict)

        engine = _StubEngine(_result(ModalityType.ASR, "merhaba", 0.85))
        asyncio.run(engine.load_model(_config("whisper-small-tr")))
        out = asyncio.run(engine.predict(features=None))

        segments = orch.process_features(
            _SESSION, [out], [_audio_health()], ModalityPath.SPEECH,
        )
        # 0.85 < 0.90 → suppressed.
        assert segments == []
