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
from sozia.server.fusion.degraded_mode_handler import DegradedModeHandler
from sozia.server.fusion.orchestrator import FusionOrchestrator
from sozia.server.fusion.sign_fusion_policy import SignFusionPolicy
from sozia.server.fusion.speech_fusion_policy import SpeechFusionPolicy

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
    async def test_partial_then_final_replaces_partial(self):
        """ASR alone → PARTIAL; lip arrives → FINAL with replaces_segment_id."""
        asr = _StubEngine(_result(ModalityType.ASR, "merhaba", 0.70))
        lip = _StubEngine(
            _result(ModalityType.LIP_READING, "merhaba dünya", 0.85),
        )
        await asr.load_model(_config("whisper-small-tr"))
        await lip.load_model(_config("lip-reading-v1"))

        asr_out = await asr.predict(features=None)
        lip_out = await lip.predict(features=None)

        orch = _build_orchestrator()

        # First pass: ASR only → PARTIAL.
        partials = await orch.process_features(
            _SESSION, [asr_out], [_audio_health()], ModalityPath.SPEECH,
        )
        assert len(partials) == 1
        assert partials[0].status == SegmentStatus.PARTIAL
        partial_id = partials[0].segment_id

        # Second pass: both modalities → FINAL replacing PARTIAL.
        finals = await orch.process_features(
            _SESSION,
            [asr_out, lip_out],
            [_audio_health()],
            ModalityPath.SPEECH,
        )
        assert len(finals) == 1
        assert finals[0].status == SegmentStatus.FINAL
        assert finals[0].replaces_segment_id == partial_id
        # Lip has higher confidence → lip text wins.
        assert finals[0].text == "merhaba dünya"

    async def test_low_confidence_asr_suppressed(self):
        asr = _StubEngine(_result(ModalityType.ASR, "noise", 0.10))
        await asr.load_model(_config("whisper-small-tr"))
        asr_out = await asr.predict(features=None)

        orch = _build_orchestrator()
        segments = await orch.process_features(
            _SESSION, [asr_out], [_audio_health()], ModalityPath.SPEECH,
        )
        assert segments == []


# ---------------------------------------------------------------------------
# Sign path: TSL → PARTIAL, then GlossToText → FINAL
# ---------------------------------------------------------------------------


class TestSignPipelineIntegration:
    async def test_gloss_replaces_tsl_partial(self):
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
        await tsl.load_model(_config("tsl-gru-v1"))
        await gloss.load_model(_config("gemma2-9b-lora"))

        tsl_out = await tsl.predict(features=None)
        gloss_out = await gloss.predict(features=None)

        orch = _build_orchestrator()

        # TSL alone → PARTIAL.
        partials = await orch.process_features(
            _SESSION, [tsl_out], [_video_health()], ModalityPath.SIGN,
        )
        assert len(partials) == 1
        assert partials[0].status == SegmentStatus.PARTIAL
        assert partials[0].text == "MERHABA DUNYA"
        partial_id = partials[0].segment_id

        # GlossToText arrives → FINAL replaces PARTIAL, text fully replaced.
        finals = await orch.process_features(
            _SESSION,
            [tsl_out, gloss_out],
            [_video_health()],
            ModalityPath.SIGN,
        )
        assert len(finals) == 1
        assert finals[0].status == SegmentStatus.FINAL
        # replaces_segment_id linking is now orchestrator's responsibility
        # (GlossAccumulator tracks it); tested in test_orchestrator_process.py.
        assert finals[0].text == "Merhaba dünya."


# ---------------------------------------------------------------------------
# Degraded mode: one pipeline unhealthy, single modality falls back
# ---------------------------------------------------------------------------


class TestDegradedIntegration:
    async def test_no_face_detected_falls_back_to_audio(self):
        """Video pipeline reports no face → fusion must still emit audio."""
        asr = _StubEngine(_result(ModalityType.ASR, "merhaba", 0.80))
        await asr.load_model(_config("whisper-small-tr"))
        asr_out = await asr.predict(features=None)

        orch = _build_orchestrator()
        # Audio healthy, video degraded (no face), only ASR result.
        segments = await orch.process_features(
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

    async def test_low_snr_penalises_asr(self):
        asr = _StubEngine(_result(ModalityType.ASR, "fısıltı", 0.70))
        await asr.load_model(_config("whisper-small-tr"))
        asr_out = await asr.predict(features=None)

        orch = _build_orchestrator()
        segments = await orch.process_features(
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
    async def test_predict_before_load_raises(self):
        from sozia.common.interfaces import ModelNotLoadedError

        engine = _StubEngine(_result(ModalityType.ASR, "x"))
        with pytest.raises(ModelNotLoadedError):
            await engine.predict(features=None)

    async def test_unload_resets_state(self):
        engine = _StubEngine(_result(ModalityType.ASR, "x"))
        await engine.load_model(_config("m"))
        assert engine.is_loaded() is True
        assert engine.get_model_id() == "m"

        await engine.unload_model()
        assert engine.is_loaded() is False
        assert engine.get_model_id() == ""

    async def test_custom_policy_threshold_is_honoured(self):
        """Refactored policies expose constants — verify override flows end-to-end."""
        strict = SpeechFusionPolicy(confidence_threshold=0.90)
        orch = FusionOrchestrator()
        orch.register_policy(ModalityPath.SPEECH, strict)

        engine = _StubEngine(_result(ModalityType.ASR, "merhaba", 0.85))
        await engine.load_model(_config("whisper-small-tr"))
        out = await engine.predict(features=None)

        segments = await orch.process_features(
            _SESSION, [out], [_audio_health()], ModalityPath.SPEECH,
        )
        # 0.85 < 0.90 → suppressed.
        assert segments == []


# ---------------------------------------------------------------------------
# warm_up / cool_down lifecycle
# ---------------------------------------------------------------------------


class TestWarmUpCoolDown:
    async def test_warm_up_loads_registered_engines(self):
        asr = _StubEngine(_result(ModalityType.ASR, "x"))
        lip = _StubEngine(_result(ModalityType.LIP_READING, "x"))
        assert not asr.is_loaded()
        assert not lip.is_loaded()

        orch = _build_orchestrator()
        orch.register_engine(
            ModalityPath.SPEECH, ModalityType.ASR, asr, _config("whisper-small-tr"),
        )
        orch.register_engine(
            ModalityPath.SPEECH, ModalityType.LIP_READING, lip, _config("lip-reading-v1"),
        )

        await orch.warm_up(ModalityPath.SPEECH)
        assert asr.is_loaded()
        assert lip.is_loaded()

    async def test_warm_up_is_idempotent(self):
        engine = _StubEngine(_result(ModalityType.ASR, "x"))
        orch = _build_orchestrator()
        orch.register_engine(
            ModalityPath.SPEECH, ModalityType.ASR, engine, _config("whisper-small-tr"),
        )

        await orch.warm_up(ModalityPath.SPEECH)
        await orch.warm_up(ModalityPath.SPEECH)  # second call is a no-op
        assert engine.is_loaded()

    async def test_warm_up_only_touches_given_path(self):
        asr = _StubEngine(_result(ModalityType.ASR, "x"))
        tsl = _StubEngine(_result(ModalityType.TSL_RECOGNITION, "x"))

        orch = _build_orchestrator()
        orch.register_engine(ModalityPath.SPEECH, ModalityType.ASR, asr, _config("whisper"))
        orch.register_engine(ModalityPath.SIGN, ModalityType.TSL_RECOGNITION, tsl, _config("tsl-gru"))

        await orch.warm_up(ModalityPath.SPEECH)
        assert asr.is_loaded()
        assert not tsl.is_loaded()

    async def test_warm_up_with_no_engines_is_noop(self):
        orch = _build_orchestrator()
        await orch.warm_up(ModalityPath.SPEECH)  # must not raise

    async def test_cool_down_unloads_all_paths(self):
        asr = _StubEngine(_result(ModalityType.ASR, "x"))
        tsl = _StubEngine(_result(ModalityType.TSL_RECOGNITION, "x"))

        orch = _build_orchestrator()
        orch.register_engine(ModalityPath.SPEECH, ModalityType.ASR, asr, _config("whisper"))
        orch.register_engine(ModalityPath.SIGN, ModalityType.TSL_RECOGNITION, tsl, _config("tsl-gru"))

        await orch.warm_up(ModalityPath.SPEECH)
        await orch.warm_up(ModalityPath.SIGN)
        assert asr.is_loaded()
        assert tsl.is_loaded()

        await orch.cool_down()
        assert not asr.is_loaded()
        assert not tsl.is_loaded()

    async def test_cool_down_without_warm_up_is_noop(self):
        orch = _build_orchestrator()
        await orch.cool_down()  # must not raise
