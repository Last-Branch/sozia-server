"""FusionOrchestrator — central coordinator for the fusion layer.

Receives inference results, routes them to the correct FusionStrategy based
on ModalityPath, and handles timeout fallback via DegradedModeHandler.

Rule R3: coordinates inference engines but never calls models directly.
"""

from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING

from sozia.common.models import (
    ModalityPath,
    ModalityResult,
    ModalityType,
    PipelineHealth,
    SegmentStatus,
    TranscriptSegment,
)
from sozia.fusion.degraded_mode_handler import DegradedModeHandler

if TYPE_CHECKING:
    from sozia.common.interfaces import FusionStrategy


class FusionOrchestrator:
    """Routes inference results to the appropriate fusion policy.

    Usage::

        orchestrator = FusionOrchestrator()
        orchestrator.register_policy(ModalityPath.SPEECH, speech_policy)
        orchestrator.register_policy(ModalityPath.SIGN, sign_policy)

        segment = orchestrator.process_features(session_id, results, health, path)
    """

    def __init__(
        self,
        degraded_handler: DegradedModeHandler | None = None,
    ) -> None:
        self._policies: dict[ModalityPath, FusionStrategy] = {}
        self._degraded: DegradedModeHandler = degraded_handler or DegradedModeHandler()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_policy(
        self, path: ModalityPath, policy: FusionStrategy,
    ) -> None:
        """Wire a fusion policy for a modality path."""
        self._policies[path] = policy

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def process_features(
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

    def handle_timeout(
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
