"""MelAccumulator — per-session mel-spectrogram buffer for the speech pipeline.

WhisperAsrEngine produces better, sentence-level transcriptions when given a
full utterance rather than 500 ms chunks. MelAccumulator holds incoming mel
frames per session and flushes when one of two conditions is met:

  1. Silence detected after speech (utterance boundary).
  2. Buffer reaches ``max_frames`` (15 s ceiling at 100 Hz).

Lip-reading continues to fire on every short chunk (PARTIAL updates).
ASR fires on each flush (FINAL output).
"""

from __future__ import annotations

import numpy as np

# log10-mel floor for speech detection. Client sends Math.log10(energy + 1e-10);
# speech peaks above −1, background noise stays below −3. -2.0 sits in the gap
# and prevents weak-signal flushes that cause Whisper hallucinations.
_SPEECH_ENERGY_THRESHOLD: float = -2.0

# Ignore silence flushes shorter than this — avoids firing on a single quiet
# frame in the middle of continuous speech.
_MIN_FLUSH_FRAMES: int = 50  # 500 ms at 100 Hz


class MelAccumulator:
    """Buffers mel-spectrogram chunks per session; emits on utterance boundary.

    Usage::

        acc = MelAccumulator(max_frames=1500)
        batch = acc.push(session_id, chunk_features)
        if batch is not None:
            result = await asr_engine.predict(batch)  # shape (T_total, N)

    Args:
        max_frames: Hard ceiling (frames). Default 1500 = 15 s at 100 Hz.
        speech_threshold: log10-mel peak below which a chunk is silence.
        min_flush_frames: Minimum accumulated frames before a silence flush
            is honoured. Prevents spurious flushes on inter-word pauses.
    """

    def __init__(
        self,
        max_frames: int = 1500,
        speech_threshold: float = _SPEECH_ENERGY_THRESHOLD,
        min_flush_frames: int = _MIN_FLUSH_FRAMES,
    ) -> None:
        self._max_frames = max_frames
        self._speech_threshold = speech_threshold
        self._min_flush_frames = min_flush_frames

        # Per-session state.
        self._buffers: dict[str, list[np.ndarray]] = {}
        self._has_seen_speech: dict[str, bool] = {}
        self._frame_counts: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def push(self, session_id: str, chunk: np.ndarray) -> np.ndarray | None:
        """Append mel chunk and return the full buffer when ready to fire.

        Args:
            session_id: Active session identifier.
            chunk: Raw log10-mel array from the client, shape ``(T, N)``
                (AudioFeatureChunk.features is always ``[T, D]``).

        Returns:
            Concatenated buffer of shape ``(T_total, N)`` on flush;
            ``None`` otherwise.
        """
        is_speech = float(np.max(chunk)) > self._speech_threshold

        buf = self._buffers.setdefault(session_id, [])
        seen = self._has_seen_speech.get(session_id, False)
        count = self._frame_counts.get(session_id, 0)

        if is_speech:
            self._has_seen_speech[session_id] = True
            buf.append(chunk)
            # Client always sends (T, N) — AudioFeatureChunk.features is [T, D].
            # WhisperAsrEngine._prepare_mel handles orientation; we just count rows.
            count += chunk.shape[0]
            self._frame_counts[session_id] = count

            if count >= self._max_frames:
                return self._flush(session_id)

        else:
            # Silence after speech + minimum frames met → flush.
            if seen and count >= self._min_flush_frames:
                return self._flush(session_id)
            # Silence before speech, or pause too short → discard frame.

        return None

    def reset(self, session_id: str) -> None:
        """Discard all buffered state for a session.

        Safe to call even if no frames have been accumulated yet.
        """
        self._buffers.pop(session_id, None)
        self._has_seen_speech.pop(session_id, None)
        self._frame_counts.pop(session_id, None)

    def pending_frames(self, session_id: str) -> int:
        """Return buffered frame count for a session (0 if none)."""
        return self._frame_counts.get(session_id, 0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _flush(self, session_id: str) -> np.ndarray:
        buf = self._buffers.pop(session_id, [])
        self._has_seen_speech[session_id] = False
        self._frame_counts[session_id] = 0

        if not buf:
            return np.zeros((0, 1), dtype=np.float32)

        # Client sends (T, N); stack along the time axis (dim 0).
        return np.concatenate(buf, axis=0)
