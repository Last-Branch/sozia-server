"""SpeechFusionPolicy — fusion strategy for the speech pipeline.

Speech path uses optimistic-then-revise:
  1. ASR result arrives first (fast, ≤800ms) → emit PARTIAL segment.
  2. Lip-reading arrives later (≤1200ms) → weighted merge with ASR → emit
     FINAL segment that replaces the PARTIAL via replaces_segment_id.

If only one modality is available, it is emitted directly. Results with
confidence below the threshold (0.30) are suppressed.
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

_CONFIDENCE_THRESHOLD = 0.30

# Weights for the weighted-average confidence merge.
_ASR_WEIGHT = 0.65
_LIP_WEIGHT = 0.35


class SpeechFusionPolicy(FusionStrategy):
    """FusionStrategy for the ASR + Lip-Reading speech pipeline.

    Maintains a mapping of session_id → last PARTIAL segment_id so that
    FINAL segments can reference the PARTIAL they replace.
    """

    def __init__(self) -> None:
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

        asr = next((r for r in valid if r.modality_type == ModalityType.ASR), None)
        lip = next(
            (r for r in valid if r.modality_type == ModalityType.LIP_READING), None,
        )

        session_id = health[0].session_id if health else ""

        if asr and lip:
            # Both modalities → weighted merge → FINAL replacing the PARTIAL.
            merged_text = lip.text if lip.text else asr.text
            merged_confidence = (
                _ASR_WEIGHT * asr.confidence + _LIP_WEIGHT * lip.confidence
            )
            replaces = self._partial_ids.pop(session_id, None)

            return [self._make_segment(
                session_id=session_id,
                status=SegmentStatus.FINAL,
                text=merged_text,
                source=ModalityType.LIP_READING,
                confidence=min(1.0, merged_confidence),
                timestamp_ms=asr.timestamp_ms,
                duration_ms=max(asr.duration_ms, lip.duration_ms),
                replaces_segment_id=replaces,
            )]

        # Single modality → PARTIAL.
        single = asr or lip
        seg_id = uuid.uuid4().hex
        self._partial_ids[session_id] = seg_id

        return [self._make_segment(
            session_id=session_id,
            status=SegmentStatus.PARTIAL,
            text=single.text,
            source=single.modality_type,
            confidence=single.confidence,
            timestamp_ms=single.timestamp_ms,
            duration_ms=single.duration_ms,
            replaces_segment_id=None,
            segment_id=seg_id,
        )]

    def should_suppress(self, result: ModalityResult) -> bool:
        return result.confidence < _CONFIDENCE_THRESHOLD

    def get_confidence_threshold(self) -> float:
        return _CONFIDENCE_THRESHOLD

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
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
            segment_id=segment_id or uuid.uuid4().hex,
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
