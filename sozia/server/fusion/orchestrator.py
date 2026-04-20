"""FusionOrchestrator — central coordinator for the fusion layer.

Receives inference results, routes them to the correct FusionStrategy based
on ModalityPath, and handles timeout fallback via DegradedModeHandler.
Also drives per-session model warm-up and cool-down by iterating over the
``InferenceEngine`` instances registered for each modality path.

Rule R3: coordinates inference engines but never calls models directly —
it only touches the ``InferenceEngine`` interface (``load_model``,
``unload_model``), never the underlying ML library.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
from typing import TYPE_CHECKING, Awaitable, Callable

logger = logging.getLogger(__name__)

import numpy as np

from sozia.common.interfaces import InferenceTimeoutError
from sozia.common.models import (
    AudioFeatureChunk,
    LandmarkFrame,
    ModalityPath,
    ModalityResult,
    ModalityType,
    ModelConfig,
    PipelineHealth,
    TranscriptSegment,
)
from sozia.server.fusion.activity_detector import ADEvent, ActivityDetector
from sozia.server.fusion.degraded_mode_handler import DegradedModeHandler
from sozia.server.fusion.frame_accumulator import FrameAccumulator
from sozia.server.fusion.gloss_accumulator import GlossAccumulator, GlossEntry
from sozia.server.fusion.mel_accumulator import MelAccumulator
from sozia.server.fusion.sign_fusion_policy import SignFusionPolicy
from sozia.server.fusion.speech_fusion_policy import SpeechFusionPolicy

if TYPE_CHECKING:
    from sozia.common.interfaces import FusionStrategy, InferenceEngine

# Additive confidence deduction applied when GlossToText times out and we
# promote the raw TSL gloss to a FINAL segment.
# Note: this is a subtraction (confidence -= N), not a multiplicative factor
# like DegradedModeHandler's degraded_penalty_factor.
_GLOSS_TIMEOUT_CONFIDENCE_DEDUCTION = 0.15

# Mirrors MelAccumulator threshold; gates face-frame collection for lip-reading.
_SPEECH_ENERGY_THRESHOLD = 0.3

# Number of face frames to accumulate before running lip inference.
_LIP_WINDOW_FRAMES = 30


class FusionOrchestrator:
    """Routes inference results to the appropriate fusion policy.

    Usage::

        orchestrator = FusionOrchestrator()
        orchestrator.register_policy(ModalityPath.SPEECH, speech_policy)
        orchestrator.register_engine(
            ModalityPath.SPEECH, ModalityType.ASR, asr_engine, asr_cfg,
        )
        orchestrator.register_engine(
            ModalityPath.SPEECH, ModalityType.LIP_READING, lip_engine, lip_cfg,
        )

        await orchestrator.warm_up(ModalityPath.SPEECH)
        await orchestrator.process(
            session_id, audio_chunk, health, ModalityPath.SPEECH, send_fn,
        )
        await orchestrator.cool_down()
    """

    def __init__(
        self,
        degraded_handler: DegradedModeHandler | None = None,
        window_size: int = 30,
        mel_max_frames: int = 1500,
        *,
        activity_detector: ActivityDetector | None = None,
        gloss_accumulator: GlossAccumulator | None = None,
    ) -> None:
        self._policies: dict[ModalityPath, FusionStrategy] = {}
        # Keyed by (path, modality_type) → (engine, config).
        self._engines: dict[
            ModalityPath, dict[ModalityType, tuple[InferenceEngine, ModelConfig]]
        ] = {}
        self._degraded: DegradedModeHandler = degraded_handler or DegradedModeHandler()
        self._accumulator = FrameAccumulator(window_size=window_size)
        self._mel_accumulator = MelAccumulator(max_frames=mel_max_frames)
        self._activity_detector: ActivityDetector = activity_detector or ActivityDetector()
        self._gloss_accumulator: GlossAccumulator = gloss_accumulator or GlossAccumulator()
        # Per-session face landmark cache (SPEECH path — latest frame from client).
        self._face_cache: dict[str, np.ndarray] = {}
        # Face frames accumulated during speech periods for lip-reading batches.
        self._face_frame_buffer: dict[str, list[np.ndarray]] = {}
        # Whether the last audio chunk detected speech energy.
        self._is_speaking: dict[str, bool] = {}
        # Most recent lip-reading result per session; fused with the next ASR flush.
        self._last_lip_result: dict[str, ModalityResult] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_policy(
        self,
        path: ModalityPath,
        policy: FusionStrategy,
    ) -> None:
        """Wire a fusion policy for a modality path."""
        self._policies[path] = policy

    def register_engine(
        self,
        path: ModalityPath,
        modality_type: ModalityType,
        engine: InferenceEngine,
        config: ModelConfig,
    ) -> None:
        """Attach an inference engine to a modality path keyed by ModalityType.

        The orchestrator records the (engine, config) pair but does not
        load the model until ``warm_up(path)`` is called. Registering the
        same ``modality_type`` twice on the same ``path`` overwrites the
        previous entry.

        Args:
            path: Which inference pipeline this engine belongs to.
            modality_type: The ModalityType this engine produces
                (e.g. ``ModalityType.ASR``). Used by ``process()`` to route
                features to the correct engine without string-matching model IDs.
            engine: A concrete InferenceEngine instance.
            config: ModelConfig passed to ``engine.load_model()`` during warm-up.
        """
        self._engines.setdefault(path, {})[modality_type] = (engine, config)

    def reset_session(self, session_id: str) -> None:
        """Discard all per-session state for ``session_id``.

        Clears the face landmark cache and any buffered landmark frames in the
        accumulator. Safe to call even if no data has been accumulated yet.

        In the current MVP each FusionOrchestrator instance serves a single
        session, so ``cool_down()`` followed by object disposal is the normal
        teardown path. ``reset_session`` exists for callers that manage
        session lifecycle externally (e.g. a future shared-orchestrator design
        per DEV-06) and for explicit mid-session resets.

        Args:
            session_id: The session whose cached state should be discarded.
        """
        self._face_cache.pop(session_id, None)
        self._face_frame_buffer.pop(session_id, None)
        self._is_speaking.pop(session_id, None)
        self._last_lip_result.pop(session_id, None)
        self._accumulator.reset(session_id)
        self._mel_accumulator.reset(session_id)
        self._activity_detector.reset(session_id)
        self._gloss_accumulator.reset(session_id)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def warm_up(self, path: ModalityPath) -> None:
        """Load every engine registered for ``path``.

        Engines that are already loaded are skipped so ``warm_up`` is
        idempotent per session. Load calls run concurrently to minimise
        cold-start latency; errors from individual engines propagate so
        the caller can decide whether to abort session setup.
        """
        engines = self._engines.get(path, {})
        await asyncio.gather(
            *(
                engine.load_model(config)
                for engine, config in engines.values()
                if not engine.is_loaded()
            )
        )

    async def cool_down(self) -> None:
        """Unload every registered engine across all paths.

        Safe to call even if no engines were warmed. Unload calls run
        concurrently and exceptions are suppressed so one failing engine
        does not block the others from releasing their memory.
        """
        tasks: list[asyncio.Task] = []
        for path_engines in self._engines.values():
            for engine, _ in path_engines.values():
                if engine.is_loaded():
                    tasks.append(asyncio.create_task(engine.unload_model()))
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    # ------------------------------------------------------------------
    # High-level public entry point
    # ------------------------------------------------------------------

    async def process(
        self,
        session_id: str,
        features: LandmarkFrame | AudioFeatureChunk,
        health: list[PipelineHealth],
        modality_path: ModalityPath,
        send_fn: Callable[[TranscriptSegment], Awaitable[None] | None],
    ) -> None:
        """Handle one incoming DTO: convert to numpy, run engines, emit segments.

        This is the primary entry point for SessionHandler. It dispatches to
        the correct internal handler based on ``modality_path`` and the DTO
        type, then calls ``send_fn`` for every TranscriptSegment produced.

        Path A (SPEECH):
          - ``LandmarkFrame`` — caches face landmarks; no inference triggered.
          - ``AudioFeatureChunk`` — runs ASR (+ LipReading in parallel if face
            landmarks are cached); emits PARTIAL from ASR, FINAL when both
            modalities succeed.

        Path B (SIGN):
          - ``LandmarkFrame`` — accumulates frames; when the window is full
            runs TslRecognitionEngine (PARTIAL), then GlossToTextEngine (FINAL).
            On GlossToText timeout the TSL gloss is promoted to FINAL with a
            confidence penalty.

        Args:
            session_id: Active session identifier.
            features: One inbound DTO from the WebSocket gateway.
            health: Current pipeline health signals from the client.
            modality_path: Which pipeline is active for this session.
            send_fn: Coroutine or plain callable invoked once per outbound segment.
        """
        if modality_path == ModalityPath.SPEECH:
            await self._process_speech(session_id, features, health, send_fn)
        else:
            await self._process_sign(session_id, features, health, send_fn)

    # ------------------------------------------------------------------
    # Internal pipeline handlers
    # ------------------------------------------------------------------

    async def flush_stale_speech(
        self,
        session_id: str,
        stale_after_s: float,
        health: list[PipelineHealth],
        send_fn: Callable[[TranscriptSegment], Awaitable[None] | None],
    ) -> None:
        """Run ASR on pending mel frames if no new chunk arrived in ``stale_after_s`` s.

        Called by the session watchdog every second so utterance-end silence
        is detected even when the client stops sending frames entirely.
        """
        asr_batch = self._mel_accumulator.take_if_stale(session_id, stale_after_s)
        if asr_batch is None:
            return
        await self._run_asr(session_id, asr_batch, health, send_fn)

    async def flush_speech_pending(
        self,
        session_id: str,
        health: list[PipelineHealth],
        send_fn: Callable[[TranscriptSegment], Awaitable[None] | None],
    ) -> None:
        """Run ASR on any buffered mel frames and emit a FINAL segment.

        Called on clean session teardown (``session_end``) so partial
        utterances are not silently discarded. No-op if nothing is buffered.
        """
        asr_batch = self._mel_accumulator.take_pending(session_id)
        if asr_batch is None:
            return
        path_engines = self._engines.get(ModalityPath.SPEECH, {})
        asr_entry = path_engines.get(ModalityType.ASR)
        if asr_entry is None:
            return
        await self._run_asr(session_id, asr_batch, health, send_fn)

    async def _process_speech(
        self,
        session_id: str,
        features: LandmarkFrame | AudioFeatureChunk,
        health: list[PipelineHealth],
        send_fn: Callable[[TranscriptSegment], Awaitable[None] | None],
    ) -> None:
        if isinstance(features, LandmarkFrame):
            if features.face_landmarks is not None:
                face = np.asarray(features.face_landmarks, dtype=np.float32)
                self._face_cache[session_id] = face
                # Accumulate face frames only during speech periods.
                if self._is_speaking.get(session_id, False):
                    self._face_frame_buffer.setdefault(session_id, []).append(face)
                    if len(self._face_frame_buffer[session_id]) >= _LIP_WINDOW_FRAMES:
                        await self._run_lip(session_id, health, send_fn)
            return

        # AudioFeatureChunk — two independent paths:
        #   Lip-reading: fires every _LIP_WINDOW_FRAMES face frames during speech.
        #   ASR: fires when MelAccumulator flushes (silence or 15 s) → FINAL.
        audio_np = np.asarray(features.features, dtype=np.float32)
        is_speech = float(np.max(audio_np)) > _SPEECH_ENERGY_THRESHOLD
        self._is_speaking[session_id] = is_speech

        if not is_speech:
            # Drop accumulated face frames so stale frames don't cross utterances.
            self._face_frame_buffer.pop(session_id, None)

        path_engines = self._engines.get(ModalityPath.SPEECH, {})
        asr_entry = path_engines.get(ModalityType.ASR)

        # --- ASR (accumulated, utterance-boundary) → FINAL ------------------
        asr_batch = self._mel_accumulator.push(session_id, audio_np)
        logger.info(
            "mel_accumulator: pending=%d frames, flushed=%s",
            self._mel_accumulator.pending_frames(session_id),
            asr_batch is not None,
        )
        if asr_batch is not None and asr_entry is not None:
            await self._run_asr(session_id, asr_batch, health, send_fn, features)

    async def _run_lip(
        self,
        session_id: str,
        health: list[PipelineHealth],
        send_fn: Callable[[TranscriptSegment], Awaitable[None] | None],
    ) -> None:
        buf = self._face_frame_buffer.get(session_id, [])
        if len(buf) < _LIP_WINDOW_FRAMES:
            return
        batch = np.stack(buf[:_LIP_WINDOW_FRAMES])  # (30, 83, 3)
        self._face_frame_buffer[session_id] = buf[_LIP_WINDOW_FRAMES:]

        path_engines = self._engines.get(ModalityPath.SPEECH, {})
        lip_entry = path_engines.get(ModalityType.LIP_READING)
        if lip_entry is None:
            return
        lip_engine, _ = lip_entry
        try:
            lip_result = await lip_engine.predict(batch)
            lip_result = dataclasses.replace(
                lip_result,
                timestamp_ms=int(time.time() * 1000),
                duration_ms=0,
            )
            policy = self._policies.get(ModalityPath.SPEECH)
            if isinstance(policy, SpeechFusionPolicy):
                if not policy.should_suppress(lip_result):
                    self._last_lip_result[session_id] = lip_result
                lip_segments = policy.emit_lip_partial(lip_result, session_id)
                logger.info(
                    "LIP result: text=%r confidence=%.4f → %s",
                    lip_result.text,
                    lip_result.confidence,
                    "PARTIAL emitted" if lip_segments else "suppressed",
                )
            else:
                lip_segments = await self.process_features(
                    session_id, [lip_result], health, ModalityPath.SPEECH
                )
            for seg in lip_segments:
                await _maybe_await(send_fn(seg))
        except InferenceTimeoutError:
            logger.warning("LIP inference timed out for session %s", session_id)

    async def _run_asr(
        self,
        session_id: str,
        asr_batch: np.ndarray,
        health: list[PipelineHealth],
        send_fn: Callable[[TranscriptSegment], Awaitable[None] | None],
        features: AudioFeatureChunk | None = None,
    ) -> None:
        if asr_batch.shape[0] < 100:
            logger.debug("asr_batch too short (%d frames), skipping", asr_batch.shape[0])
            return
        path_engines = self._engines.get(ModalityPath.SPEECH, {})
        asr_entry = path_engines.get(ModalityType.ASR)
        if asr_entry is None:
            return
        asr_engine, _ = asr_entry
        try:
            asr_result = await asr_engine.predict(asr_batch, timeout_ms=3000)
            logger.info(
                "ASR result: text=%r confidence=%.4f latency=%dms",
                asr_result.text,
                asr_result.confidence,
                asr_result.inference_latency_ms,
            )
            if asr_result.confidence < 0.30:
                logger.info(
                    "ASR result suppressed (confidence=%.4f < 0.30)", asr_result.confidence
                )
                return
            asr_result = dataclasses.replace(
                asr_result,
                timestamp_ms=features.timestamp_ms if features is not None else int(time.time() * 1000),
                duration_ms=features.chunk_duration_ms if features is not None else 0,
            )
            policy = self._policies.get(ModalityPath.SPEECH)
            lip_cached = self._last_lip_result.pop(session_id, None)
            if isinstance(policy, SpeechFusionPolicy):
                if lip_cached is not None:
                    logger.info(
                        "ASR flush: lip cached (text=%r conf=%.4f) → FUSED FINAL",
                        lip_cached.text,
                        lip_cached.confidence,
                    )
                    asr_segments = policy.emit_fused_final(asr_result, lip_cached, session_id)
                else:
                    logger.info("ASR flush: no lip cached → standalone ASR FINAL")
                    asr_segments = policy.emit_asr_final(asr_result, session_id)
            else:
                asr_segments = await self.process_features(
                    session_id, [asr_result], health, ModalityPath.SPEECH
                )
            for seg in asr_segments:
                await _maybe_await(send_fn(seg))
        except InferenceTimeoutError:
            lip_cached = self._last_lip_result.pop(session_id, None)
            if lip_cached is not None:
                logger.info(
                    "ASR timeout: promoting lip PARTIAL to FINAL (text=%r conf=%.4f)",
                    lip_cached.text,
                    lip_cached.confidence,
                )
                policy = self._policies.get(ModalityPath.SPEECH)
                if isinstance(policy, SpeechFusionPolicy):
                    for seg in policy.promote_partial_to_final(session_id, lip_cached):
                        await _maybe_await(send_fn(seg))
            else:
                logger.warning("ASR timeout: no lip cached, segment lost for session %s", session_id)

    async def _process_sign(
        self,
        session_id: str,
        features: LandmarkFrame | AudioFeatureChunk,
        health: list[PipelineHealth],
        send_fn: Callable[[TranscriptSegment], Awaitable[None] | None],
    ) -> None:
        if not isinstance(features, LandmarkFrame):
            return  # SIGN path only accepts LandmarkFrame.

        event = self._activity_detector.update(features)
        logger.info("AD event: session=%s event=%s", session_id, event.value)

        path_engines = self._engines.get(ModalityPath.SIGN, {})
        tsl_entry = path_engines.get(ModalityType.TSL_RECOGNITION)
        gloss_entry = path_engines.get(ModalityType.GLOSS_TO_TEXT)

        if event in (ADEvent.STARTED, ADEvent.ACTIVE):
            batch = self._accumulator.add(features)
            if batch is None:
                return  # Window not full yet.

            if tsl_entry is None:
                return

            sign_policy = self._policies.get(ModalityPath.SIGN)
            if not isinstance(sign_policy, SignFusionPolicy):
                return

            tsl_engine, _ = tsl_entry
            try:
                tsl_result = await tsl_engine.predict(batch)
            except InferenceTimeoutError:
                logger.warning("TSL inference timed out — window discarded")
                return

            logger.info(
                "TSL result: gloss=%r confidence=%.4f latency=%dms",
                tsl_result.text,
                tsl_result.confidence,
                tsl_result.inference_latency_ms,
            )

            if sign_policy.should_suppress(tsl_result):
                return

            entry = GlossEntry(
                text=tsl_result.text,
                confidence=tsl_result.confidence,
                timestamp_ms=tsl_result.timestamp_ms,
                duration_ms=tsl_result.duration_ms,
            )
            changed = self._gloss_accumulator.push(session_id, entry)
            if not changed:
                return  # Deduped — no new information, skip PARTIAL update.

            phrase = self._gloss_accumulator.peek(session_id)
            prev_id = self._gloss_accumulator.pop_partial_id(session_id)
            seg = sign_policy.emit_partial(session_id, phrase, tsl_result, replaces_id=prev_id)
            self._gloss_accumulator.set_partial_id(session_id, seg.segment_id)
            logger.info("Emitting PARTIAL: text=%r replaces=%s", phrase, prev_id)
            await _maybe_await(send_fn(seg))

        elif event == ADEvent.ENDED:
            if self._gloss_accumulator.is_empty(session_id):
                return

            sign_policy = self._policies.get(ModalityPath.SIGN)
            if not isinstance(sign_policy, SignFusionPolicy):
                return

            # Pop partial_id before flush clears it internally.
            prev_id = self._gloss_accumulator.pop_partial_id(session_id)
            entries = self._gloss_accumulator.flush(session_id)
            phrase = " ".join(e.text for e in entries)
            aggregate_conf = min(e.confidence for e in entries)
            timestamp_ms = entries[0].timestamp_ms
            duration_ms = sum(e.duration_ms for e in entries)

            logger.info(
                "AD ENDED: flushing phrase=%r agg_conf=%.4f",
                phrase,
                aggregate_conf,
            )

            seg: TranscriptSegment
            if gloss_entry is not None:
                gloss_engine, _ = gloss_entry
                try:
                    llm_result = await gloss_engine.predict(phrase)
                    logger.info(
                        "GlossToText result: text=%r confidence=%.4f latency=%dms",
                        llm_result.text,
                        llm_result.confidence,
                        llm_result.inference_latency_ms,
                    )
                    if sign_policy.should_suppress_llm(llm_result):
                        seg = sign_policy.emit_final_from_gloss(
                            session_id, phrase, aggregate_conf,
                            timestamp_ms, duration_ms, prev_id,
                        )
                    else:
                        seg = sign_policy.emit_final_from_llm(
                            session_id, phrase, llm_result, prev_id,
                        )
                except InferenceTimeoutError:
                    logger.warning("GlossToText timed out — using raw gloss as FINAL")
                    penalised = max(
                        0.0, aggregate_conf - _GLOSS_TIMEOUT_CONFIDENCE_DEDUCTION
                    )
                    seg = sign_policy.emit_final_from_gloss(
                        session_id, phrase, penalised,
                        timestamp_ms, duration_ms, prev_id,
                    )
            else:
                seg = sign_policy.emit_final_from_gloss(
                    session_id, phrase, aggregate_conf,
                    timestamp_ms, duration_ms, prev_id,
                )

            self._accumulator.reset(session_id)
            logger.info("Emitting FINAL: text=%r replaces=%s", seg.text, prev_id)
            await _maybe_await(send_fn(seg))

        # ADEvent.IDLE — no output.

    # ------------------------------------------------------------------
    # Internal fusion helper (kept public for existing test compatibility)
    # ------------------------------------------------------------------

    async def process_features(
        self,
        session_id: str,
        results: list[ModalityResult],
        health: list[PipelineHealth],
        modality_path: ModalityPath,
    ) -> list[TranscriptSegment]:
        """Route pre-computed inference results to the registered policy.

        If any pipeline in ``health`` signals degraded state and only one
        modality result is present, delegates to DegradedModeHandler instead.

        Args:
            session_id: Active session identifier.
            results: Inference results from one or more engines.
            health: Current pipeline health signals.
            modality_path: Which pipeline (SPEECH or SIGN) is active.

        Returns:
            List of TranscriptSegment objects to stream to the client.
            May be empty if all results are suppressed.

        Raises:
            ValueError: No policy registered for ``modality_path``.
        """
        if modality_path not in self._policies:
            raise ValueError(f"No fusion policy registered for {modality_path.value}")

        # Check degraded mode: if any health signal triggers fallback and
        # only one result is available, use the degraded handler.
        if len(results) == 1 and self._is_degraded(health):
            seg = self._degraded.handle_degraded_input(session_id, results[0])
            return [seg] if seg is not None else []

        policy = self._policies[modality_path]
        return policy.fuse(results, health)

    async def handle_timeout(
        self,
        session_id: str,
        result: ModalityResult,
        health: list[PipelineHealth],
        modality_path: ModalityPath,
    ) -> list[TranscriptSegment]:
        """Emit a PARTIAL when the slower modality times out.

        Called when the secondary modality exceeds its latency budget.
        The primary modality's result is fused alone via the registered
        policy (producing a PARTIAL segment).

        Args:
            session_id: Active session identifier.
            result: The primary (fast) modality result that is available.
            health: Current pipeline health signals.
            modality_path: Which pipeline is active.

        Returns:
            List of TranscriptSegment objects (typically one PARTIAL).
        """
        if modality_path not in self._policies:
            raise ValueError(f"No fusion policy registered for {modality_path.value}")

        policy = self._policies[modality_path]
        return policy.fuse([result], health)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_degraded(self, health: list[PipelineHealth]) -> bool:
        """Return True if any health signal indicates degraded state."""
        return any(self._degraded.should_fallback(h) for h in health)


# ---------------------------------------------------------------------------
# Module-level utility
# ---------------------------------------------------------------------------


async def _maybe_await(value: Awaitable | None) -> None:
    """Await ``value`` if it is awaitable, otherwise discard it."""
    if asyncio.isfuture(value) or asyncio.iscoroutine(value):
        await value
