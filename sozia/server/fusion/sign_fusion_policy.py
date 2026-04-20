"""SignFusionPolicy — fusion strategy for the sign language pipeline.

Sign path uses optimistic-then-revise:
  1. TSL Recognition emits raw gloss words as PARTIAL segments (accumulated
     by GlossAccumulator, replaced word-by-word).
  2. GlossToText converts the full accumulated phrase to natural Turkish and
     emits a FINAL segment that replaces the last PARTIAL.

The policy owns no per-session state — partial-ID bookkeeping lives in
GlossAccumulator. The orchestrator calls the helper methods directly instead
of going through fuse(); fuse() is kept for integration-test backward compat.

Two separate confidence thresholds:
  - TSL softmax:         _DEFAULT_CONFIDENCE_THRESHOLD   = 0.55
  - GlossToText exp(LM): _DEFAULT_LLM_CONFIDENCE_THRESHOLD = 0.40
    (beam sequences_scores are length-normalised log-probs → exp() yields
    lower numeric values than softmax, so a lower gate makes sense)
"""

from __future__ import annotations

import time
import uuid

from sozia.common.interfaces import FusionStrategy
from sozia.common.models import (
    ModalityResult,
    ModalityType,
    PipelineHealth,
    SegmentStatus,
    TranscriptSegment,
)

_DEFAULT_CONFIDENCE_THRESHOLD = 0.55
_DEFAULT_LLM_CONFIDENCE_THRESHOLD = 0.40


class SignFusionPolicy(FusionStrategy):
    """FusionStrategy for the TSL Recognition + GlossToText pipeline."""

    def __init__(
        self,
        *,
        confidence_threshold: float = _DEFAULT_CONFIDENCE_THRESHOLD,
        llm_confidence_threshold: float = _DEFAULT_LLM_CONFIDENCE_THRESHOLD,
    ) -> None:
        self._confidence_threshold = confidence_threshold
        self._llm_confidence_threshold = llm_confidence_threshold

    # ------------------------------------------------------------------
    # FusionStrategy interface (backward compat — orchestrator uses helpers)
    # ------------------------------------------------------------------

    def fuse(
        self,
        results: list[ModalityResult],
        health: list[PipelineHealth],
    ) -> list[TranscriptSegment]:
        if not results:
            return []

        tsl = next(
            (r for r in results
             if r.modality_type == ModalityType.TSL_RECOGNITION
             and not self.should_suppress(r)),
            None,
        )
        gloss = next(
            (r for r in results
             if r.modality_type == ModalityType.GLOSS_TO_TEXT
             and not self.should_suppress_llm(r)),
            None,
        )

        session_id = health[0].session_id if health else ""

        if gloss:
            return [self.emit_final_from_llm(session_id, gloss.text, gloss, replaces_id=None)]
        if tsl:
            return [self.emit_partial(session_id, tsl.text, tsl, replaces_id=None)]
        return []

    # ------------------------------------------------------------------
    # Suppress gates
    # ------------------------------------------------------------------

    def should_suppress(self, result: ModalityResult) -> bool:
        """True if TSL softmax confidence is below the TSL gate (0.55)."""
        return result.confidence < self._confidence_threshold

    def should_suppress_llm(self, result: ModalityResult) -> bool:
        """True if GlossToText beam confidence is below the LLM gate (0.40)."""
        return result.confidence < self._llm_confidence_threshold

    def get_confidence_threshold(self) -> float:
        return self._confidence_threshold

    def get_llm_confidence_threshold(self) -> float:
        return self._llm_confidence_threshold

    # ------------------------------------------------------------------
    # Segment-building helpers (called directly by FusionOrchestrator)
    # ------------------------------------------------------------------

    def emit_partial(
        self,
        session_id: str,
        gloss_phrase: str,
        tsl_result: ModalityResult,
        replaces_id: str | None,
    ) -> TranscriptSegment:
        """Build a PARTIAL segment for the current accumulated gloss phrase."""
        return _make_segment(
            session_id=session_id,
            status=SegmentStatus.PARTIAL,
            text=gloss_phrase,
            source=ModalityType.TSL_RECOGNITION,
            confidence=tsl_result.confidence,
            timestamp_ms=tsl_result.timestamp_ms,
            duration_ms=tsl_result.duration_ms,
            replaces_segment_id=replaces_id,
        )

    def emit_final_from_llm(
        self,
        session_id: str,
        gloss_phrase: str,
        llm_result: ModalityResult,
        replaces_id: str | None,
    ) -> TranscriptSegment:
        """Build a FINAL segment from a successful GlossToText translation."""
        return _make_segment(
            session_id=session_id,
            status=SegmentStatus.FINAL,
            text=llm_result.text,
            source=ModalityType.GLOSS_TO_TEXT,
            confidence=llm_result.confidence,
            timestamp_ms=llm_result.timestamp_ms,
            duration_ms=llm_result.duration_ms,
            replaces_segment_id=replaces_id,
        )

    def emit_final_from_gloss(
        self,
        session_id: str,
        gloss_phrase: str,
        aggregate_conf: float,
        timestamp_ms: int,
        duration_ms: int,
        replaces_id: str | None,
    ) -> TranscriptSegment:
        """Build a FINAL segment from the raw gloss phrase (LLM timeout / low-conf fallback)."""
        return _make_segment(
            session_id=session_id,
            status=SegmentStatus.FINAL,
            text=gloss_phrase,
            source=ModalityType.TSL_RECOGNITION,
            confidence=aggregate_conf,
            timestamp_ms=timestamp_ms,
            duration_ms=duration_ms,
            replaces_segment_id=replaces_id,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_segment(
    *,
    session_id: str,
    status: SegmentStatus,
    text: str,
    source: ModalityType,
    confidence: float,
    timestamp_ms: int,
    duration_ms: int,
    replaces_segment_id: str | None,
) -> TranscriptSegment:
    return TranscriptSegment(
        segment_id=str(uuid.uuid4()),
        session_id=session_id,
        status=status,
        text=text,
        source=source,
        confidence=confidence,
        timestamp_ms=timestamp_ms,
        duration_ms=duration_ms,
        created_at_ms=int(time.time() * 1000),
        replaces_segment_id=replaces_segment_id,
    )
