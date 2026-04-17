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

_DEFAULT_CONFIDENCE_THRESHOLD = 0.30

# Default weights for the weighted-average confidence merge.
_DEFAULT_ASR_WEIGHT = 0.65
_DEFAULT_LIP_WEIGHT = 0.35


class SpeechFusionPolicy(FusionStrategy):
    """FusionStrategy for the ASR + Lip-Reading speech pipeline.

    Maintains a mapping of session_id → last PARTIAL segment_id so that
    FINAL segments can reference the PARTIAL they replace.

    Weights and threshold default to tuned production values but can be
    overridden for experimentation or testing.
    """

    def __init__(
        self,
        *,
        asr_weight: float = _DEFAULT_ASR_WEIGHT,
        lip_weight: float = _DEFAULT_LIP_WEIGHT,
        confidence_threshold: float = _DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> None:
        self._asr_weight = asr_weight
        self._lip_weight = lip_weight
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

        asr = next((r for r in valid if r.modality_type == ModalityType.ASR), None)
        lip = next(
            (r for r in valid if r.modality_type == ModalityType.LIP_READING), None,
        )

        session_id = health[0].session_id if health else ""

        if asr and lip:
            # Both modalities → weighted merge → FINAL replacing the PARTIAL.
            # Prefer the text from whichever hypothesis has higher confidence;
            # if lip text is empty, fall back to ASR regardless.
            if lip.text and lip.confidence >= asr.confidence:
                merged_text = lip.text
                merged_source = ModalityType.LIP_READING
            else:
                merged_text = asr.text
                merged_source = ModalityType.ASR
            merged_confidence = (
                self._asr_weight * asr.confidence
                + self._lip_weight * lip.confidence
            )
            replaces = self._partial_ids.pop(session_id, None)

            return [self._make_segment(
                session_id=session_id,
                status=SegmentStatus.FINAL,
                text=merged_text,
                source=merged_source,
                confidence=min(1.0, merged_confidence),
                timestamp_ms=asr.timestamp_ms,
                duration_ms=max(asr.duration_ms, lip.duration_ms),
                replaces_segment_id=replaces,
            )]

        # Single modality → PARTIAL, replacing the previous PARTIAL so the
        # client always shows the latest lip-reading word rather than stacking.
        single = asr or lip
        previous_id = self._partial_ids.get(session_id)
        seg_id = str(uuid.uuid4())
        self._partial_ids[session_id] = seg_id

        return [self._make_segment(
            session_id=session_id,
            status=SegmentStatus.PARTIAL,
            text=single.text,
            source=single.modality_type,
            confidence=single.confidence,
            timestamp_ms=single.timestamp_ms,
            duration_ms=single.duration_ms,
            replaces_segment_id=previous_id,
            segment_id=seg_id,
        )]

    def emit_asr_final(
        self,
        result: ModalityResult,
        session_id: str,
    ) -> list[TranscriptSegment]:
        """Emit a FINAL segment from an accumulated ASR result.

        Called when the MelAccumulator flushes after an utterance boundary.
        The result covers a full sentence so it goes straight to FINAL,
        replacing the last lip-reading PARTIAL stored for this session.
        """
        if self.should_suppress(result):
            return []
        replaces = self._partial_ids.pop(session_id, None)
        return [self._make_segment(
            session_id=session_id,
            status=SegmentStatus.FINAL,
            text=result.text,
            source=ModalityType.ASR,
            confidence=result.confidence,
            timestamp_ms=result.timestamp_ms,
            duration_ms=result.duration_ms,
            replaces_segment_id=replaces,
        )]

    def should_suppress(self, result: ModalityResult) -> bool:
        return result.confidence < self._confidence_threshold

    def get_confidence_threshold(self) -> float:
        return self._confidence_threshold

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
