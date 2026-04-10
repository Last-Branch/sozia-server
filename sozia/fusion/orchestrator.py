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
from typing import TYPE_CHECKING

from sozia.common.models import (
    ModalityPath,
    ModalityResult,
    ModelConfig,
    PipelineHealth,
    TranscriptSegment,
)
from sozia.fusion.degraded_mode_handler import DegradedModeHandler

if TYPE_CHECKING:
    from sozia.common.interfaces import FusionStrategy, InferenceEngine


class FusionOrchestrator:
    """Routes inference results to the appropriate fusion policy.

    Usage::

        orchestrator = FusionOrchestrator()
        orchestrator.register_policy(ModalityPath.SPEECH, speech_policy)
        orchestrator.register_engine(ModalityPath.SPEECH, asr_engine, asr_cfg)
        orchestrator.register_engine(ModalityPath.SPEECH, lip_engine, lip_cfg)

        await orchestrator.warm_up(ModalityPath.SPEECH)
        segments = await orchestrator.process_features(
            session_id, results, health, ModalityPath.SPEECH,
        )
        await orchestrator.cool_down()
    """

    def __init__(
        self,
        degraded_handler: DegradedModeHandler | None = None,
    ) -> None:
        self._policies: dict[ModalityPath, FusionStrategy] = {}
        self._engines: dict[
            ModalityPath, list[tuple[InferenceEngine, ModelConfig]],
        ] = {}
        self._degraded: DegradedModeHandler = (
            degraded_handler or DegradedModeHandler()
        )

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_policy(
        self, path: ModalityPath, policy: FusionStrategy,
    ) -> None:
        """Wire a fusion policy for a modality path."""
        self._policies[path] = policy

    def register_engine(
        self,
        path: ModalityPath,
        engine: InferenceEngine,
        config: ModelConfig,
    ) -> None:
        """Attach an inference engine to a modality path for warm-up.

        The orchestrator records the (engine, config) pair but does not
        load the model until ``warm_up(path)`` is called. Engines stay
        attached across sessions; ``cool_down()`` unloads them.
        """
        self._engines.setdefault(path, []).append((engine, config))

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
        engines = self._engines.get(path, [])
        await asyncio.gather(
            *(
                engine.load_model(config)
                for engine, config in engines
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
        for engines in self._engines.values():
            for engine, _ in engines:
                if engine.is_loaded():
                    tasks.append(asyncio.create_task(engine.unload_model()))
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def process_features(
        self,
        session_id: str,
        results: list[ModalityResult],
        health: list[PipelineHealth],
        modality_path: ModalityPath,
    ) -> list[TranscriptSegment]:
        """Route inference results to the registered policy.

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
            raise ValueError(
                f"No fusion policy registered for {modality_path.value}"
            )

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
            raise ValueError(
                f"No fusion policy registered for {modality_path.value}"
            )

        policy = self._policies[modality_path]
        return policy.fuse([result], health)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_degraded(self, health: list[PipelineHealth]) -> bool:
        """Return True if any health signal indicates degraded state."""
        return any(self._degraded.should_fallback(h) for h in health)
