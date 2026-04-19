"""SignFusionPolicy — fusion strategy for the sign language pipeline.

Sign path uses optimistic-then-revise:
  1. TSL Recognition emits a raw gloss string (e.g. "MERHABA DUNYA") as a
     PARTIAL segment.
  2. GlossToText converts it to natural Turkish (e.g. "Merhaba dünya.") and
     emits a FINAL segment that replaces the PARTIAL via replaces_segment_id.

Unlike the speech path there is no weighted merge — GlossToText output fully
replaces the gloss. If GlossToText is suppressed (low confidence), the raw
gloss PARTIAL remains as the final output.
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


class SignFusionPolicy(FusionStrategy):
    """FusionStrategy for the TSL Recognition + GlossToText pipeline."""

    def __init__(
        self,
        *,
        confidence_threshold: float = _DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> None:
        self._confidence_threshold = confidence_threshold
        self._partial_ids: dict[str, str] = {}

    # ------------------------------------------------------------------
    # FusionStrategy interface
    # ------------------------------------------------------------------

    def fuse(
        self,
        results: list[ModalityResult],
        health: list[PipelineHealth],
    ) -> list[TranscriptSegment]:
        if not results:
            return []

        valid = [r for r in results if not self.should_suppress(r)]
        if not valid:
            return []

        tsl = next(
            (r for r in valid if r.modality_type == ModalityType.TSL_RECOGNITION),
            None,
        )
        gloss = next(
            (r for r in valid if r.modality_type == ModalityType.GLOSS_TO_TEXT),
            None,
        )

        session_id = health[0].session_id if health else ""

        if gloss:
            # GlossToText available → FINAL replacing the PARTIAL.
            replaces = self._partial_ids.pop(session_id, None)
            return [_make_segment(
                session_id=session_id,
                status=SegmentStatus.FINAL,
                text=gloss.text,
                source=ModalityType.GLOSS_TO_TEXT,
                confidence=gloss.confidence,
                timestamp_ms=gloss.timestamp_ms,
                duration_ms=gloss.duration_ms,
                replaces_segment_id=replaces,
            )]

        if tsl:
            # Only TSL available → PARTIAL with raw gloss.
            seg_id = str(uuid.uuid4())
            self._partial_ids[session_id] = seg_id
            return [_make_segment(
                session_id=session_id,
                status=SegmentStatus.PARTIAL,
                text=tsl.text,
                source=ModalityType.TSL_RECOGNITION,
                confidence=tsl.confidence,
                timestamp_ms=tsl.timestamp_ms,
                duration_ms=tsl.duration_ms,
                replaces_segment_id=None,
                segment_id=seg_id,
            )]

        return []

    def should_suppress(self, result: ModalityResult) -> bool:
        return result.confidence < self._confidence_threshold

    def get_confidence_threshold(self) -> float:
        return self._confidence_threshold


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
    segment_id: str | None = None,
) -> TranscriptSegment:
    return TranscriptSegment(
        segment_id=segment_id or str(uuid.uuid4()),
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
