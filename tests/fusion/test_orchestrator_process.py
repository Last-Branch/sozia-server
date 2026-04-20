"""Tests for FusionOrchestrator.process() and updated register_engine()."""

from __future__ import annotations


from sozia.common.interfaces import InferenceEngine, InferenceTimeoutError
from sozia.common.models import (
    FACE_LANDMARK_COUNT,
    HAND_LANDMARK_COUNT,
    POSE_LANDMARK_COUNT,
    AudioFeatureChunk,
    LandmarkFrame,
    ModalityPath,
    ModalityResult,
    ModalityType,
    ModelConfig,
    PipelineHealth,
    SegmentStatus,
    TranscriptSegment,
)
from sozia.server.fusion.orchestrator import FusionOrchestrator
from sozia.server.fusion.sign_fusion_policy import SignFusionPolicy
from sozia.server.fusion.speech_fusion_policy import SpeechFusionPolicy

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_SESSION = "550e8400-e29b-41d4-a716-446655440000"


def _cfg(model_id: str) -> ModelConfig:
    return ModelConfig(
        model_id=model_id, weights_path="/tmp/noop", device="cpu", params={}
    )


def _result(
    modality: ModalityType,
    text: str = "test",
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


def _video_health(available: bool = True, face_detected: bool = True) -> PipelineHealth:
    return PipelineHealth(
        session_id=_SESSION,
        pipeline="video",
        available=available,
        fps=30.0,
        snr=None,
        face_detected=face_detected,
        last_updated_ms=1000,
    )


def _audio_chunk() -> AudioFeatureChunk:
    return AudioFeatureChunk(
        session_id=_SESSION,
        timestamp_ms=0,
        features=[[1.0, 1.5, 2.0]] * 100,
        feature_type="mfcc",
        sample_rate_hz=16000,
        chunk_duration_ms=500,
    )


def _landmark_frame(
    session_id: str = _SESSION,
    timestamp_ms: int = 0,
) -> LandmarkFrame:
    n = POSE_LANDMARK_COUNT
    return LandmarkFrame(
        session_id=session_id,
        timestamp_ms=timestamp_ms,
        face_landmarks=[[0.1, 0.2, 0.3]] * FACE_LANDMARK_COUNT,
        left_hand_landmarks=[[0.5, 0.5, 0.5]] * HAND_LANDMARK_COUNT,
        right_hand_landmarks=[[0.5, 0.5, 0.5]] * HAND_LANDMARK_COUNT,
        pose_landmarks=[[i / n, (n - 1 - i) / n, float(i)] for i in range(n)],
    )


class _StubEngine(InferenceEngine):
    """Minimal stub that returns a pre-canned result."""

    def __init__(
        self,
        result: ModalityResult | None = None,
        raise_timeout: bool = False,
        model_id: str = "stub",
    ) -> None:
        self._result = result
        self._raise_timeout = raise_timeout
        self._model_id = model_id
        self._loaded = False

    async def load_model(self, config: ModelConfig) -> None:
        self._loaded = True
        self._model_id = config.model_id

    async def predict(self, features, timeout_ms: int = 2200) -> ModalityResult:
        from sozia.common.interfaces import ModelNotLoadedError

        if not self._loaded:
            raise ModelNotLoadedError(self._model_id)
        if self._raise_timeout:
            raise InferenceTimeoutError("stub timeout")
        return self._result

    async def unload_model(self) -> None:
        self._loaded = False

    def is_loaded(self) -> bool:
        return self._loaded

    def get_model_id(self) -> str:
        return self._model_id if self._loaded else ""


def _make_orchestrator(
    window_size: int = 1,
    mel_max_frames: int = 1500,
    with_speech: bool = False,
    with_sign: bool = False,
    asr_engine: _StubEngine | None = None,
    lip_engine: _StubEngine | None = None,
    tsl_engine: _StubEngine | None = None,
    gloss_engine: _StubEngine | None = None,
) -> FusionOrchestrator:
    orch = FusionOrchestrator(window_size=window_size, mel_max_frames=mel_max_frames)
    orch.register_policy(ModalityPath.SPEECH, SpeechFusionPolicy())
    orch.register_policy(ModalityPath.SIGN, SignFusionPolicy())

    if with_speech or asr_engine:
        _asr = asr_engine or _StubEngine(
            _result(ModalityType.ASR, "merhaba", 0.85), model_id="whisper"
        )
        _asr._loaded = True
        orch.register_engine(
            ModalityPath.SPEECH,
            ModalityType.ASR,
            _asr,
            _cfg("whisper"),
        )

    if with_speech or lip_engine:
        _lip = lip_engine or _StubEngine(
            _result(ModalityType.LIP_READING, "merhaba dünya", 0.75), model_id="lip"
        )
        _lip._loaded = True
        orch.register_engine(
            ModalityPath.SPEECH,
            ModalityType.LIP_READING,
            _lip,
            _cfg("lip"),
        )

    if with_sign or tsl_engine:
        _tsl = tsl_engine or _StubEngine(
            _result(ModalityType.TSL_RECOGNITION, "MERHABA", 0.80), model_id="tsl"
        )
        _tsl._loaded = True
        orch.register_engine(
            ModalityPath.SIGN,
            ModalityType.TSL_RECOGNITION,
            _tsl,
            _cfg("tsl"),
        )

    if with_sign or gloss_engine:
        _gloss = gloss_engine or _StubEngine(
            _result(ModalityType.GLOSS_TO_TEXT, "Merhaba.", 0.88), model_id="gemma"
        )
        _gloss._loaded = True
        orch.register_engine(
            ModalityPath.SIGN,
            ModalityType.GLOSS_TO_TEXT,
            _gloss,
            _cfg("gemma"),
        )

    return orch


# ---------------------------------------------------------------------------
# register_engine() API
# ---------------------------------------------------------------------------


class TestRegisterEngine:
    def test_register_engine_stores_by_modality_type(self):
        orch = FusionOrchestrator()
        engine = _StubEngine(_result(ModalityType.ASR, "x"))
        orch.register_engine(
            ModalityPath.SPEECH,
            ModalityType.ASR,
            engine,
            _cfg("whisper"),
        )
        path_engines = orch._engines.get(ModalityPath.SPEECH, {})
        assert ModalityType.ASR in path_engines

    def test_register_engine_second_type_same_path(self):
        orch = FusionOrchestrator()
        asr = _StubEngine(_result(ModalityType.ASR, "x"))
        lip = _StubEngine(_result(ModalityType.LIP_READING, "y"))
        orch.register_engine(ModalityPath.SPEECH, ModalityType.ASR, asr, _cfg("a"))
        orch.register_engine(
            ModalityPath.SPEECH, ModalityType.LIP_READING, lip, _cfg("b")
        )
        path_engines = orch._engines[ModalityPath.SPEECH]
        assert ModalityType.ASR in path_engines
        assert ModalityType.LIP_READING in path_engines

    def test_register_engine_overwrites_same_modality_type(self):
        orch = FusionOrchestrator()
        first = _StubEngine(_result(ModalityType.ASR, "first"))
        second = _StubEngine(_result(ModalityType.ASR, "second"))
        orch.register_engine(ModalityPath.SPEECH, ModalityType.ASR, first, _cfg("a"))
        orch.register_engine(ModalityPath.SPEECH, ModalityType.ASR, second, _cfg("b"))
        stored_engine, _ = orch._engines[ModalityPath.SPEECH][ModalityType.ASR]
        assert stored_engine is second


# ---------------------------------------------------------------------------
# warm_up / cool_down — updated structure
# ---------------------------------------------------------------------------


class TestWarmUpCoolDownUpdated:
    async def test_warm_up_loads_all_registered_engines_by_type(self):
        asr = _StubEngine(_result(ModalityType.ASR, "x"))
        lip = _StubEngine(_result(ModalityType.LIP_READING, "y"))

        orch = FusionOrchestrator()
        orch.register_engine(ModalityPath.SPEECH, ModalityType.ASR, asr, _cfg("w"))
        orch.register_engine(
            ModalityPath.SPEECH, ModalityType.LIP_READING, lip, _cfg("l")
        )

        await orch.warm_up(ModalityPath.SPEECH)
        assert asr.is_loaded()
        assert lip.is_loaded()

    async def test_cool_down_unloads_all(self):
        asr = _StubEngine(_result(ModalityType.ASR, "x"))
        tsl = _StubEngine(_result(ModalityType.TSL_RECOGNITION, "x"))

        orch = FusionOrchestrator()
        orch.register_engine(ModalityPath.SPEECH, ModalityType.ASR, asr, _cfg("w"))
        orch.register_engine(
            ModalityPath.SIGN, ModalityType.TSL_RECOGNITION, tsl, _cfg("t")
        )

        await orch.warm_up(ModalityPath.SPEECH)
        await orch.warm_up(ModalityPath.SIGN)
        await orch.cool_down()
        assert not asr.is_loaded()
        assert not tsl.is_loaded()


# ---------------------------------------------------------------------------
# process() — Path A (SPEECH)
# ---------------------------------------------------------------------------


class TestProcessSpeech:
    async def test_audio_chunk_emits_partial_via_send_fn(self):
        # Speech audio sets is_speaking=True. 30 landmark frames trigger lip → PARTIAL.
        orch = _make_orchestrator(with_speech=True)

        sent: list[TranscriptSegment] = []
        await orch.process(
            _SESSION, _audio_chunk(), [_audio_health()], ModalityPath.SPEECH, sent.append
        )
        for _ in range(30):
            await orch.process(
                _SESSION, _landmark_frame(), [_video_health()], ModalityPath.SPEECH, sent.append
            )

        assert any(s.status == SegmentStatus.PARTIAL for s in sent)

    async def test_audio_chunk_with_cached_face_emits_final(self):
        # mel_max_frames=1 forces accumulator to flush on the first speech chunk.
        # Lip-reading emits PARTIAL; ASR emits FINAL after flush.
        asr = _StubEngine(_result(ModalityType.ASR, "merhaba", 0.75), model_id="w")
        lip = _StubEngine(
            _result(ModalityType.LIP_READING, "merhaba dünya", 0.85), model_id="l"
        )
        asr._loaded = True
        lip._loaded = True

        orch = FusionOrchestrator(mel_max_frames=1)
        orch.register_policy(ModalityPath.SPEECH, SpeechFusionPolicy())
        orch.register_engine(ModalityPath.SPEECH, ModalityType.ASR, asr, _cfg("w"))
        orch.register_engine(
            ModalityPath.SPEECH, ModalityType.LIP_READING, lip, _cfg("l")
        )

        sent: list[TranscriptSegment] = []
        await orch.process(
            _SESSION,
            _landmark_frame(),
            [_video_health()],
            ModalityPath.SPEECH,
            sent.append,
        )
        assert sent == []  # face caching only

        await orch.process(
            _SESSION,
            _audio_chunk(),
            [_audio_health()],
            ModalityPath.SPEECH,
            sent.append,
        )
        assert any(s.status == SegmentStatus.FINAL for s in sent)

    async def test_asr_final_emits_after_accumulator_flush(self):
        # Without a cached face, lip-reading is skipped. ASR fires as FINAL
        # when the accumulator flushes (mel_max_frames=1 forces it on first chunk).
        asr = _StubEngine(_result(ModalityType.ASR, "merhaba", 0.85), model_id="w")
        lip = _StubEngine(_result(ModalityType.LIP_READING, "x", 0.75), model_id="l")
        asr._loaded = True
        lip._loaded = True

        orch = FusionOrchestrator(mel_max_frames=1)
        orch.register_policy(ModalityPath.SPEECH, SpeechFusionPolicy())
        orch.register_engine(ModalityPath.SPEECH, ModalityType.ASR, asr, _cfg("w"))
        orch.register_engine(
            ModalityPath.SPEECH, ModalityType.LIP_READING, lip, _cfg("l")
        )

        sent: list[TranscriptSegment] = []
        await orch.process(
            _SESSION,
            _audio_chunk(),
            [_audio_health()],
            ModalityPath.SPEECH,
            sent.append,
        )
        # Lip-reading skipped (no face cache). ASR fires as FINAL from accumulator.
        assert len(sent) == 1
        assert sent[0].status == SegmentStatus.FINAL
        assert sent[0].source == ModalityType.ASR

    async def test_asr_timeout_produces_no_segments(self):
        asr = _StubEngine(None, raise_timeout=True, model_id="w")
        asr._loaded = True

        orch = FusionOrchestrator()
        orch.register_policy(ModalityPath.SPEECH, SpeechFusionPolicy())
        orch.register_engine(ModalityPath.SPEECH, ModalityType.ASR, asr, _cfg("w"))

        sent: list[TranscriptSegment] = []
        await orch.process(
            _SESSION,
            _audio_chunk(),
            [_audio_health()],
            ModalityPath.SPEECH,
            sent.append,
        )
        assert sent == []

    async def test_lip_timeout_asr_still_emits_final(self):
        # Lip-reading times out → no PARTIAL. ASR fires FINAL from accumulator.
        asr = _StubEngine(_result(ModalityType.ASR, "merhaba", 0.80), model_id="w")
        lip = _StubEngine(None, raise_timeout=True, model_id="l")
        asr._loaded = True
        lip._loaded = True

        orch = FusionOrchestrator(mel_max_frames=1)
        orch.register_policy(ModalityPath.SPEECH, SpeechFusionPolicy())
        orch.register_engine(ModalityPath.SPEECH, ModalityType.ASR, asr, _cfg("w"))
        orch.register_engine(
            ModalityPath.SPEECH, ModalityType.LIP_READING, lip, _cfg("l")
        )

        await orch.process(
            _SESSION,
            _landmark_frame(),
            [_video_health()],
            ModalityPath.SPEECH,
            lambda s: None,
        )

        sent: list[TranscriptSegment] = []
        await orch.process(
            _SESSION,
            _audio_chunk(),
            [_audio_health()],
            ModalityPath.SPEECH,
            sent.append,
        )
        assert len(sent) == 1
        assert sent[0].status == SegmentStatus.FINAL
        assert sent[0].source == ModalityType.ASR

    async def test_suppressed_asr_emits_nothing(self):
        asr = _StubEngine(_result(ModalityType.ASR, "noise", 0.10), model_id="w")
        asr._loaded = True

        orch = FusionOrchestrator()
        orch.register_policy(ModalityPath.SPEECH, SpeechFusionPolicy())
        orch.register_engine(ModalityPath.SPEECH, ModalityType.ASR, asr, _cfg("w"))

        sent: list[TranscriptSegment] = []
        await orch.process(
            _SESSION,
            _audio_chunk(),
            [_audio_health()],
            ModalityPath.SPEECH,
            sent.append,
        )
        assert sent == []

    async def test_landmark_frame_on_speech_path_caches_only(self):
        """LandmarkFrame on SPEECH path updates face cache, no inference triggered."""
        orch = _make_orchestrator(with_speech=True)
        sent: list[TranscriptSegment] = []
        await orch.process(
            _SESSION,
            _landmark_frame(),
            [_video_health()],
            ModalityPath.SPEECH,
            sent.append,
        )
        assert sent == []

    async def test_segments_carry_timestamp_from_audio_chunk(self):
        """Orchestrator stamps timestamp_ms/duration_ms from AudioFeatureChunk."""
        chunk = AudioFeatureChunk(
            session_id=_SESSION,
            timestamp_ms=5000,
            features=[[1.0, 1.5]] * 100,
            feature_type="mfcc",
            sample_rate_hz=16000,
            chunk_duration_ms=750,
        )
        asr = _StubEngine(_result(ModalityType.ASR, "test", 0.85), model_id="w")
        asr._loaded = True

        # mel_max_frames=1 forces accumulator flush so ASR fires on first chunk.
        orch = FusionOrchestrator(mel_max_frames=1)
        orch.register_policy(ModalityPath.SPEECH, SpeechFusionPolicy())
        orch.register_engine(ModalityPath.SPEECH, ModalityType.ASR, asr, _cfg("w"))

        sent: list[TranscriptSegment] = []
        await orch.process(
            _SESSION, chunk, [_audio_health()], ModalityPath.SPEECH, sent.append
        )

        assert len(sent) >= 1
        for seg in sent:
            assert seg.timestamp_ms == 5000
            assert seg.duration_ms == 750

    async def test_lip_then_asr_emits_fused_final_replacing_partial(self):
        # Speech audio → is_speaking. 30 landmarks → lip PARTIAL. ASR flush → FUSED FINAL.
        asr = _StubEngine(_result(ModalityType.ASR, "merhaba", 0.80), model_id="w")
        lip = _StubEngine(_result(ModalityType.LIP_READING, "merhaba", 0.85), model_id="l")
        asr._loaded = True
        lip._loaded = True

        orch = FusionOrchestrator()  # large mel_max_frames — no auto flush
        orch.register_policy(ModalityPath.SPEECH, SpeechFusionPolicy())
        orch.register_engine(ModalityPath.SPEECH, ModalityType.ASR, asr, _cfg("w"))
        orch.register_engine(ModalityPath.SPEECH, ModalityType.LIP_READING, lip, _cfg("l"))

        sent: list[TranscriptSegment] = []
        await orch.process(_SESSION, _audio_chunk(), [_audio_health()], ModalityPath.SPEECH, sent.append)
        for _ in range(30):
            await orch.process(_SESSION, _landmark_frame(), [_video_health()], ModalityPath.SPEECH, sent.append)
        await orch.flush_speech_pending(_SESSION, [_audio_health()], sent.append)

        statuses = [s.status for s in sent]
        assert SegmentStatus.PARTIAL in statuses
        assert SegmentStatus.FINAL in statuses
        final = next(s for s in sent if s.status == SegmentStatus.FINAL)
        partial = next(s for s in sent if s.status == SegmentStatus.PARTIAL)
        assert final.replaces_segment_id == partial.segment_id

    async def test_asr_timeout_with_lip_cached_promotes_to_final(self):
        # Speech audio → is_speaking. 30 landmarks → lip PARTIAL cached. ASR times out → LIP FINAL.
        asr = _StubEngine(None, raise_timeout=True, model_id="w")
        lip = _StubEngine(_result(ModalityType.LIP_READING, "merhaba", 0.85), model_id="l")
        asr._loaded = True
        lip._loaded = True

        orch = FusionOrchestrator()
        orch.register_policy(ModalityPath.SPEECH, SpeechFusionPolicy())
        orch.register_engine(ModalityPath.SPEECH, ModalityType.ASR, asr, _cfg("w"))
        orch.register_engine(ModalityPath.SPEECH, ModalityType.LIP_READING, lip, _cfg("l"))

        sent: list[TranscriptSegment] = []
        await orch.process(_SESSION, _audio_chunk(), [_audio_health()], ModalityPath.SPEECH, sent.append)
        for _ in range(30):
            await orch.process(_SESSION, _landmark_frame(), [_video_health()], ModalityPath.SPEECH, sent.append)
        await orch.flush_speech_pending(_SESSION, [_audio_health()], sent.append)

        assert any(s.status == SegmentStatus.PARTIAL for s in sent)
        final_segs = [s for s in sent if s.status == SegmentStatus.FINAL]
        assert len(final_segs) == 1
        assert final_segs[0].source == ModalityType.LIP_READING
        partial = next(s for s in sent if s.status == SegmentStatus.PARTIAL)
        assert final_segs[0].replaces_segment_id == partial.segment_id

    async def test_lip_partial_fires_after_30_face_frames_during_speech(self):
        """Lip PARTIAL fires once 30 face frames accumulate during a speech period."""
        lip = _StubEngine(
            _result(ModalityType.LIP_READING, "hello", 0.80), model_id="l"
        )
        lip._loaded = True

        orch = FusionOrchestrator()
        orch.register_policy(ModalityPath.SPEECH, SpeechFusionPolicy())
        orch.register_engine(ModalityPath.SPEECH, ModalityType.LIP_READING, lip, _cfg("l"))

        sent: list[TranscriptSegment] = []
        # Speech audio marks is_speaking=True.
        await orch.process(_SESSION, _audio_chunk(), [_audio_health()], ModalityPath.SPEECH, sent.append)
        assert sent == []  # no lip yet

        # 29 frames — not enough to fire.
        for _ in range(29):
            await orch.process(_SESSION, _landmark_frame(), [_video_health()], ModalityPath.SPEECH, sent.append)
        assert sent == []

        # 30th frame triggers lip inference → PARTIAL.
        await orch.process(_SESSION, _landmark_frame(), [_video_health()], ModalityPath.SPEECH, sent.append)
        assert len(sent) == 1
        assert sent[0].status == SegmentStatus.PARTIAL
        assert sent[0].timestamp_ms > 0

    async def test_landmark_frames_during_silence_do_not_trigger_lip(self):
        # Silence audio chunk → is_speaking=False. Landmark frames must not accumulate.
        lip = _StubEngine(_result(ModalityType.LIP_READING, "hello", 0.80), model_id="l")
        lip._loaded = True

        orch = FusionOrchestrator()
        orch.register_policy(ModalityPath.SPEECH, SpeechFusionPolicy())
        orch.register_engine(ModalityPath.SPEECH, ModalityType.LIP_READING, lip, _cfg("l"))

        silence_chunk = AudioFeatureChunk(
            session_id=_SESSION,
            timestamp_ms=0,
            features=[[-1.0, -1.0]] * 100,
            feature_type="mfcc",
            sample_rate_hz=16000,
            chunk_duration_ms=500,
        )

        sent: list[TranscriptSegment] = []
        await orch.process(_SESSION, silence_chunk, [_audio_health()], ModalityPath.SPEECH, sent.append)
        for _ in range(30):
            await orch.process(_SESSION, _landmark_frame(), [_video_health()], ModalityPath.SPEECH, sent.append)

        assert sent == []

    async def test_silence_after_speech_clears_face_buffer(self):
        # Partial face buffer accumulated during speech is discarded on silence.
        lip = _StubEngine(_result(ModalityType.LIP_READING, "hello", 0.80), model_id="l")
        lip._loaded = True

        orch = FusionOrchestrator()
        orch.register_policy(ModalityPath.SPEECH, SpeechFusionPolicy())
        orch.register_engine(ModalityPath.SPEECH, ModalityType.LIP_READING, lip, _cfg("l"))

        silence_chunk = AudioFeatureChunk(
            session_id=_SESSION,
            timestamp_ms=0,
            features=[[-1.0, -1.0]] * 100,
            feature_type="mfcc",
            sample_rate_hz=16000,
            chunk_duration_ms=500,
        )

        sent: list[TranscriptSegment] = []
        # Accumulate 20 face frames during speech.
        await orch.process(_SESSION, _audio_chunk(), [_audio_health()], ModalityPath.SPEECH, sent.append)
        for _ in range(20):
            await orch.process(_SESSION, _landmark_frame(), [_video_health()], ModalityPath.SPEECH, sent.append)

        # Silence clears the buffer.
        await orch.process(_SESSION, silence_chunk, [_audio_health()], ModalityPath.SPEECH, sent.append)

        # 30 more landmark frames during silence — must not trigger lip.
        for _ in range(30):
            await orch.process(_SESSION, _landmark_frame(), [_video_health()], ModalityPath.SPEECH, sent.append)

        assert sent == []


# ---------------------------------------------------------------------------
# process() — Path B (SIGN)
# ---------------------------------------------------------------------------


class TestProcessSign:
    async def test_partial_window_emits_nothing(self):
        orch = _make_orchestrator(window_size=5, with_sign=True)
        sent: list[TranscriptSegment] = []
        for i in range(4):
            await orch.process(
                _SESSION,
                _landmark_frame(timestamp_ms=i),
                [_video_health()],
                ModalityPath.SIGN,
                sent.append,
            )
        assert sent == []

    async def test_full_window_emits_partial_from_tsl(self):
        tsl = _StubEngine(
            _result(ModalityType.TSL_RECOGNITION, "MERHABA", 0.80), model_id="tsl"
        )
        tsl._loaded = True
        gloss = _StubEngine(
            _result(ModalityType.GLOSS_TO_TEXT, "Merhaba.", 0.88), model_id="gemma"
        )
        gloss._loaded = True

        orch = FusionOrchestrator(window_size=3)
        orch.register_policy(ModalityPath.SIGN, SignFusionPolicy())
        orch.register_engine(
            ModalityPath.SIGN, ModalityType.TSL_RECOGNITION, tsl, _cfg("tsl")
        )
        orch.register_engine(
            ModalityPath.SIGN, ModalityType.GLOSS_TO_TEXT, gloss, _cfg("gemma")
        )

        sent: list[TranscriptSegment] = []
        for i in range(3):
            await orch.process(
                _SESSION,
                _landmark_frame(timestamp_ms=i),
                [_video_health()],
                ModalityPath.SIGN,
                sent.append,
            )

        # Should have: PARTIAL (TSL) + FINAL (gloss)
        assert len(sent) == 2
        statuses = {s.status for s in sent}
        assert SegmentStatus.PARTIAL in statuses
        assert SegmentStatus.FINAL in statuses

    async def test_full_window_partial_text_is_gloss(self):
        tsl = _StubEngine(
            _result(ModalityType.TSL_RECOGNITION, "MERHABA DUNYA", 0.80), model_id="tsl"
        )
        tsl._loaded = True
        gloss = _StubEngine(
            _result(ModalityType.GLOSS_TO_TEXT, "Merhaba dünya.", 0.88),
            model_id="gemma",
        )
        gloss._loaded = True

        orch = FusionOrchestrator(window_size=1)
        orch.register_policy(ModalityPath.SIGN, SignFusionPolicy())
        orch.register_engine(
            ModalityPath.SIGN, ModalityType.TSL_RECOGNITION, tsl, _cfg("tsl")
        )
        orch.register_engine(
            ModalityPath.SIGN, ModalityType.GLOSS_TO_TEXT, gloss, _cfg("gemma")
        )

        sent: list[TranscriptSegment] = []
        await orch.process(
            _SESSION,
            _landmark_frame(),
            [_video_health()],
            ModalityPath.SIGN,
            sent.append,
        )
        partial = next(s for s in sent if s.status == SegmentStatus.PARTIAL)
        final = next(s for s in sent if s.status == SegmentStatus.FINAL)
        assert partial.text == "MERHABA DUNYA"
        assert final.text == "Merhaba dünya."
        # replaces_segment_id linking tested after orchestrator _process_sign rewrite.

    async def test_tsl_timeout_emits_nothing(self):
        tsl = _StubEngine(None, raise_timeout=True, model_id="tsl")
        tsl._loaded = True

        orch = FusionOrchestrator(window_size=1)
        orch.register_policy(ModalityPath.SIGN, SignFusionPolicy())
        orch.register_engine(
            ModalityPath.SIGN, ModalityType.TSL_RECOGNITION, tsl, _cfg("tsl")
        )

        sent: list[TranscriptSegment] = []
        await orch.process(
            _SESSION,
            _landmark_frame(),
            [_video_health()],
            ModalityPath.SIGN,
            sent.append,
        )
        assert sent == []

    async def test_gloss_timeout_promotes_tsl_partial_to_final(self):
        tsl = _StubEngine(
            _result(ModalityType.TSL_RECOGNITION, "MERHABA", 0.80), model_id="tsl"
        )
        tsl._loaded = True
        gloss = _StubEngine(None, raise_timeout=True, model_id="gemma")
        gloss._loaded = True

        orch = FusionOrchestrator(window_size=1)
        orch.register_policy(ModalityPath.SIGN, SignFusionPolicy())
        orch.register_engine(
            ModalityPath.SIGN, ModalityType.TSL_RECOGNITION, tsl, _cfg("tsl")
        )
        orch.register_engine(
            ModalityPath.SIGN, ModalityType.GLOSS_TO_TEXT, gloss, _cfg("gemma")
        )

        sent: list[TranscriptSegment] = []
        await orch.process(
            _SESSION,
            _landmark_frame(),
            [_video_health()],
            ModalityPath.SIGN,
            sent.append,
        )

        # PARTIAL (TSL gloss) + FINAL (promoted from TSL on gloss timeout)
        assert len(sent) == 2
        partial = next(s for s in sent if s.status == SegmentStatus.PARTIAL)
        final = next(s for s in sent if s.status == SegmentStatus.FINAL)
        # FINAL replaces the PARTIAL
        assert final.replaces_segment_id == partial.segment_id
        # FINAL carries TSL raw gloss text (fallback)
        assert final.text == "MERHABA"
        # Confidence penalised
        assert final.confidence < 0.80

    async def test_suppressed_tsl_emits_nothing(self):
        tsl = _StubEngine(
            _result(ModalityType.TSL_RECOGNITION, "x", 0.10), model_id="tsl"
        )
        tsl._loaded = True

        orch = FusionOrchestrator(window_size=1)
        orch.register_policy(ModalityPath.SIGN, SignFusionPolicy())
        orch.register_engine(
            ModalityPath.SIGN, ModalityType.TSL_RECOGNITION, tsl, _cfg("tsl")
        )

        sent: list[TranscriptSegment] = []
        await orch.process(
            _SESSION,
            _landmark_frame(),
            [_video_health()],
            ModalityPath.SIGN,
            sent.append,
        )
        assert sent == []

    async def test_audio_chunk_on_sign_path_is_noop(self):
        orch = _make_orchestrator(window_size=1, with_sign=True)
        sent: list[TranscriptSegment] = []
        await orch.process(
            _SESSION,
            _audio_chunk(),
            [_audio_health()],
            ModalityPath.SIGN,
            sent.append,
        )
        assert sent == []


# ---------------------------------------------------------------------------
# reset_session()
# ---------------------------------------------------------------------------


class TestResetSession:
    async def test_reset_clears_face_cache(self):
        """After reset, a new audio chunk sees no cached face landmarks."""
        orch = _make_orchestrator(with_speech=True)
        # Populate face cache.
        await orch.process(
            _SESSION,
            _landmark_frame(),
            [_video_health()],
            ModalityPath.SPEECH,
            lambda _: None,
        )
        orch.reset_session(_SESSION)
        # After reset, audio chunk should produce ASR-only output (no lip result).
        sent: list[TranscriptSegment] = []
        await orch.process(
            _SESSION,
            _audio_chunk(),
            [_audio_health()],
            ModalityPath.SPEECH,
            sent.append,
        )
        assert all(seg.source == ModalityType.ASR for seg in sent)

    async def test_reset_clears_accumulator_buffer(self):
        """After reset, frame count restarts from zero."""
        orch = _make_orchestrator(window_size=3, with_sign=True)
        # Push 2 of 3 frames (window not yet full).
        for i in range(2):
            await orch.process(
                _SESSION,
                _landmark_frame(timestamp_ms=i),
                [_video_health()],
                ModalityPath.SIGN,
                lambda _: None,
            )
        orch.reset_session(_SESSION)
        # Push 2 more — should still not hit the window (buffer was cleared).
        sent: list[TranscriptSegment] = []
        for i in range(2):
            await orch.process(
                _SESSION,
                _landmark_frame(timestamp_ms=i + 10),
                [_video_health()],
                ModalityPath.SIGN,
                sent.append,
            )
        assert sent == []

    def test_reset_unknown_session_is_safe(self):
        """reset_session on a session with no state does not raise."""
        orch = FusionOrchestrator()
        orch.reset_session("nonexistent-session")  # must not raise
