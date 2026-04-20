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
from sozia.server.fusion.activity_detector import ADEvent, ActivityDetector
from sozia.server.fusion.gloss_accumulator import GlossAccumulator
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


class _MockAD:
    """Stub ActivityDetector that returns a fixed sequence of ADEvents."""

    def __init__(self, events: list[ADEvent], default: ADEvent = ADEvent.IDLE) -> None:
        self._events = iter(events)
        self._default = default

    def update(self, frame: LandmarkFrame) -> ADEvent:
        return next(self._events, self._default)

    def reset(self, session_id: str) -> None:
        pass

    def is_active(self, session_id: str) -> bool:
        return False


def _make_orchestrator(
    window_size: int = 1,
    mel_max_frames: int = 1500,
    with_speech: bool = False,
    with_sign: bool = False,
    asr_engine: _StubEngine | None = None,
    lip_engine: _StubEngine | None = None,
    tsl_engine: _StubEngine | None = None,
    gloss_engine: _StubEngine | None = None,
    activity_detector: ActivityDetector | None = None,
    gloss_accumulator: GlossAccumulator | None = None,
) -> FusionOrchestrator:
    orch = FusionOrchestrator(
        window_size=window_size,
        mel_max_frames=mel_max_frames,
        activity_detector=activity_detector,
        gloss_accumulator=gloss_accumulator,
    )
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
# process() — Path B (SIGN) — AD-gated multi-word accumulation flow
# ---------------------------------------------------------------------------


def _sign_orch(
    tsl_result: ModalityResult | None = None,
    gloss_result: ModalityResult | None = None,
    tsl_timeout: bool = False,
    gloss_timeout: bool = False,
    ad_events: list[ADEvent] | None = None,
    window_size: int = 1,
) -> FusionOrchestrator:
    """Build an orchestrator wired for sign-path tests with a controlled AD."""
    tsl = _StubEngine(
        tsl_result or _result(ModalityType.TSL_RECOGNITION, "MERHABA", 0.80),
        raise_timeout=tsl_timeout,
        model_id="tsl",
    )
    tsl._loaded = True

    mock_ad = _MockAD(ad_events or [ADEvent.ACTIVE])
    orch = FusionOrchestrator(window_size=window_size, activity_detector=mock_ad)
    orch.register_policy(ModalityPath.SIGN, SignFusionPolicy())
    orch.register_engine(
        ModalityPath.SIGN, ModalityType.TSL_RECOGNITION, tsl, _cfg("tsl")
    )

    if gloss_result is not None or gloss_timeout:
        g = _StubEngine(gloss_result, raise_timeout=gloss_timeout, model_id="gemma")
        g._loaded = True
        orch.register_engine(
            ModalityPath.SIGN, ModalityType.GLOSS_TO_TEXT, g, _cfg("gemma")
        )

    return orch


class TestProcessSign:
    async def test_idle_event_produces_nothing(self):
        """AD returning IDLE means no frames reach the accumulator."""
        orch = _sign_orch(ad_events=[ADEvent.IDLE, ADEvent.IDLE, ADEvent.IDLE])
        sent: list[TranscriptSegment] = []
        for i in range(3):
            await orch.process(
                _SESSION,
                _landmark_frame(timestamp_ms=i),
                [_video_health()],
                ModalityPath.SIGN,
                sent.append,
            )
        assert sent == []

    async def test_active_window_emits_partial(self):
        """ACTIVE event + full window → TSL predict → PARTIAL emitted."""
        orch = _sign_orch(
            tsl_result=_result(ModalityType.TSL_RECOGNITION, "MERHABA", 0.80),
            ad_events=[ADEvent.ACTIVE],
        )
        sent: list[TranscriptSegment] = []
        await orch.process(
            _SESSION,
            _landmark_frame(),
            [_video_health()],
            ModalityPath.SIGN,
            sent.append,
        )
        assert len(sent) == 1
        assert sent[0].status == SegmentStatus.PARTIAL
        assert sent[0].text == "MERHABA"

    async def test_started_event_same_as_active(self):
        """STARTED (rising edge) feeds the window just like ACTIVE."""
        orch = _sign_orch(
            tsl_result=_result(ModalityType.TSL_RECOGNITION, "GİT", 0.75),
            ad_events=[ADEvent.STARTED],
        )
        sent: list[TranscriptSegment] = []
        await orch.process(
            _SESSION,
            _landmark_frame(),
            [_video_health()],
            ModalityPath.SIGN,
            sent.append,
        )
        assert len(sent) == 1
        assert sent[0].status == SegmentStatus.PARTIAL
        assert sent[0].text == "GİT"

    async def test_two_active_windows_emit_growing_phrase(self):
        """Second ACTIVE window appends a new word → second PARTIAL replaces first."""
        tsl1 = _result(ModalityType.TSL_RECOGNITION, "YEMEK", 0.80)
        tsl2 = _result(ModalityType.TSL_RECOGNITION, "OKUL", 0.75)
        tsl_stub = _StubEngine(tsl1, model_id="tsl")
        tsl_stub._loaded = True

        mock_ad = _MockAD([ADEvent.ACTIVE, ADEvent.ACTIVE])
        orch = FusionOrchestrator(window_size=1, activity_detector=mock_ad)
        orch.register_policy(ModalityPath.SIGN, SignFusionPolicy())
        orch.register_engine(
            ModalityPath.SIGN, ModalityType.TSL_RECOGNITION, tsl_stub, _cfg("tsl")
        )

        sent: list[TranscriptSegment] = []
        await orch.process(
            _SESSION,
            _landmark_frame(timestamp_ms=0),
            [_video_health()],
            ModalityPath.SIGN,
            sent.append,
        )
        # Swap TSL result for second window.
        tsl_stub._result = tsl2
        await orch.process(
            _SESSION,
            _landmark_frame(timestamp_ms=1),
            [_video_health()],
            ModalityPath.SIGN,
            sent.append,
        )

        assert len(sent) == 2
        assert sent[0].text == "YEMEK"
        assert sent[1].text == "YEMEK OKUL"
        assert sent[1].replaces_segment_id == sent[0].segment_id

    async def test_dedup_same_gloss_no_second_partial(self):
        """Consecutive identical TSL glosses are deduped — no second PARTIAL."""
        tsl_result = _result(ModalityType.TSL_RECOGNITION, "YEMEK", 0.70)
        tsl_stub = _StubEngine(tsl_result, model_id="tsl")
        tsl_stub._loaded = True

        mock_ad = _MockAD([ADEvent.ACTIVE, ADEvent.ACTIVE])
        orch = FusionOrchestrator(window_size=1, activity_detector=mock_ad)
        orch.register_policy(ModalityPath.SIGN, SignFusionPolicy())
        orch.register_engine(
            ModalityPath.SIGN, ModalityType.TSL_RECOGNITION, tsl_stub, _cfg("tsl")
        )

        sent: list[TranscriptSegment] = []
        await orch.process(
            _SESSION,
            _landmark_frame(timestamp_ms=0),
            [_video_health()],
            ModalityPath.SIGN,
            sent.append,
        )
        await orch.process(
            _SESSION,
            _landmark_frame(timestamp_ms=1),
            [_video_health()],
            ModalityPath.SIGN,
            sent.append,
        )

        assert len(sent) == 1  # Second window deduped.

    async def test_ended_high_llm_confidence_emits_final_from_llm(self):
        """ACTIVE accumulates a word, ENDED triggers GlossToText FINAL."""
        gloss_res = _result(ModalityType.GLOSS_TO_TEXT, "Merhaba.", 0.88)
        orch = _sign_orch(
            tsl_result=_result(ModalityType.TSL_RECOGNITION, "MERHABA", 0.80),
            gloss_result=gloss_res,
            ad_events=[ADEvent.ACTIVE, ADEvent.ENDED],
        )
        sent: list[TranscriptSegment] = []
        # Frame 1: ACTIVE → window fills → TSL → PARTIAL
        await orch.process(
            _SESSION,
            _landmark_frame(timestamp_ms=0),
            [_video_health()],
            ModalityPath.SIGN,
            sent.append,
        )
        # Frame 2: ENDED → flush → GlossToText → FINAL
        await orch.process(
            _SESSION,
            _landmark_frame(timestamp_ms=1),
            [_video_health()],
            ModalityPath.SIGN,
            sent.append,
        )

        assert len(sent) == 2
        partial = next(s for s in sent if s.status == SegmentStatus.PARTIAL)
        final = next(s for s in sent if s.status == SegmentStatus.FINAL)
        assert final.text == "Merhaba."
        assert final.source == ModalityType.GLOSS_TO_TEXT
        assert final.replaces_segment_id == partial.segment_id

    async def test_ended_low_llm_confidence_emits_final_from_gloss(self):
        """LLM confidence below 0.40 → raw gloss becomes FINAL."""
        gloss_res = _result(ModalityType.GLOSS_TO_TEXT, "Merhaba.", confidence=0.20)
        orch = _sign_orch(
            tsl_result=_result(ModalityType.TSL_RECOGNITION, "MERHABA", 0.80),
            gloss_result=gloss_res,
            ad_events=[ADEvent.ACTIVE, ADEvent.ENDED],
        )
        sent: list[TranscriptSegment] = []
        await orch.process(
            _SESSION,
            _landmark_frame(timestamp_ms=0),
            [_video_health()],
            ModalityPath.SIGN,
            sent.append,
        )
        await orch.process(
            _SESSION,
            _landmark_frame(timestamp_ms=1),
            [_video_health()],
            ModalityPath.SIGN,
            sent.append,
        )

        final = next(s for s in sent if s.status == SegmentStatus.FINAL)
        assert final.text == "MERHABA"
        assert final.source == ModalityType.TSL_RECOGNITION

    async def test_ended_gloss_timeout_emits_penalised_final(self):
        """GlossToText timeout on ENDED → raw gloss FINAL with confidence penalty."""
        orch = _sign_orch(
            tsl_result=_result(ModalityType.TSL_RECOGNITION, "MERHABA", 0.80),
            gloss_timeout=True,
            ad_events=[ADEvent.ACTIVE, ADEvent.ENDED],
        )
        sent: list[TranscriptSegment] = []
        await orch.process(
            _SESSION,
            _landmark_frame(timestamp_ms=0),
            [_video_health()],
            ModalityPath.SIGN,
            sent.append,
        )
        await orch.process(
            _SESSION,
            _landmark_frame(timestamp_ms=1),
            [_video_health()],
            ModalityPath.SIGN,
            sent.append,
        )

        final = next(s for s in sent if s.status == SegmentStatus.FINAL)
        assert final.text == "MERHABA"
        assert final.confidence < 0.80  # Penalised.
        partial = next(s for s in sent if s.status == SegmentStatus.PARTIAL)
        assert final.replaces_segment_id == partial.segment_id

    async def test_ended_empty_accumulator_produces_nothing(self):
        """ENDED with no accumulated glosses emits nothing."""
        orch = _sign_orch(
            gloss_result=_result(ModalityType.GLOSS_TO_TEXT, "x", 0.88),
            ad_events=[ADEvent.ENDED],
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

    async def test_tsl_timeout_on_active_discards_window(self):
        """TSL timeout during ACTIVE — no PARTIAL emitted, no crash."""
        orch = _sign_orch(tsl_timeout=True, ad_events=[ADEvent.ACTIVE])
        sent: list[TranscriptSegment] = []
        await orch.process(
            _SESSION,
            _landmark_frame(),
            [_video_health()],
            ModalityPath.SIGN,
            sent.append,
        )
        assert sent == []

    async def test_suppressed_tsl_produces_nothing(self):
        """TSL confidence below 0.55 → suppressed, no PARTIAL."""
        orch = _sign_orch(
            tsl_result=_result(ModalityType.TSL_RECOGNITION, "noise", 0.10),
            ad_events=[ADEvent.ACTIVE],
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

    async def test_window_not_full_produces_nothing(self):
        """ACTIVE but window not yet full — no inference, no PARTIAL."""
        orch = _sign_orch(ad_events=[ADEvent.ACTIVE, ADEvent.ACTIVE], window_size=3)
        sent: list[TranscriptSegment] = []
        for i in range(2):
            await orch.process(
                _SESSION,
                _landmark_frame(timestamp_ms=i),
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
        """After reset, frame count restarts from zero (verified via ACTIVE events)."""
        # AD always returns ACTIVE so frames reach the accumulator.
        mock_ad = _MockAD([], default=ADEvent.ACTIVE)
        tsl = _StubEngine(
            _result(ModalityType.TSL_RECOGNITION, "MERHABA", 0.80), model_id="tsl"
        )
        tsl._loaded = True

        orch = FusionOrchestrator(window_size=3, activity_detector=mock_ad)
        orch.register_policy(ModalityPath.SIGN, SignFusionPolicy())
        orch.register_engine(
            ModalityPath.SIGN, ModalityType.TSL_RECOGNITION, tsl, _cfg("tsl")
        )

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

        # After reset, 2 more frames should still not fill the window.
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

    async def test_reset_clears_gloss_accumulator(self):
        """After reset, GlossAccumulator state is cleared for the session."""
        from sozia.server.fusion.gloss_accumulator import GlossAccumulator, GlossEntry

        gloss_acc = GlossAccumulator()
        orch = FusionOrchestrator(gloss_accumulator=gloss_acc)

        # Manually seed a gloss entry for the session.
        gloss_acc.push(_SESSION, GlossEntry("MERHABA", 0.80, 1000, 500))
        assert not gloss_acc.is_empty(_SESSION)

        orch.reset_session(_SESSION)
        assert gloss_acc.is_empty(_SESSION)

    def test_reset_unknown_session_is_safe(self):
        """reset_session on a session with no state does not raise."""
        orch = FusionOrchestrator()
        orch.reset_session("nonexistent-session")  # must not raise
