"""GlossAccumulator — per-session TSL gloss phrase buffer.

Collects TSL gloss words across multiple rolling-window inferences and builds
a phrase to send to GlossToTextEngine once signing activity ends.

Dedup rule: if the incoming word matches the last buffered word, the higher-
confidence entry is kept. This collapses runs of overlapping-window predictions
that produce the same gloss (e.g. three windows all returning "YEMEK" → one
"YEMEK" entry at the best confidence).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GlossEntry:
    """Immutable record of a single TSL recognition result."""

    text: str
    confidence: float
    timestamp_ms: int
    duration_ms: int


class GlossAccumulator:
    """Buffers GlossEntry objects per session with consecutive-repeat dedup.

    Args:
        max_words: Maximum number of words to buffer per session before
            rejecting new entries. Default 10.
    """

    def __init__(self, max_words: int = 10) -> None:
        self._max_words = max_words
        self._buffers: dict[str, list[GlossEntry]] = {}
        self._partial_ids: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Buffer operations
    # ------------------------------------------------------------------

    def push(self, session_id: str, entry: GlossEntry) -> bool:
        """Add a gloss entry to the session buffer.

        Dedup rule: if ``entry.text`` matches the last buffered word, the
        entry with higher confidence wins. If the incoming entry wins, it
        replaces the existing one and True is returned. If the existing entry
        wins, the incoming one is dropped and False is returned.

        If the buffer is at ``max_words`` and the entry is a new word, it is
        dropped and False is returned.

        Args:
            session_id: Session to push into.
            entry: The GlossEntry to add.

        Returns:
            True if the buffer state changed; False if the entry was dropped.
        """
        buf = self._buffers.setdefault(session_id, [])

        if buf and buf[-1].text == entry.text:
            if entry.confidence > buf[-1].confidence:
                self._buffers[session_id] = buf[:-1] + [entry]
                return True
            return False

        if len(buf) >= self._max_words:
            return False

        buf.append(entry)
        return True

    def peek(self, session_id: str) -> str:
        """Return the accumulated phrase as a space-joined string."""
        return " ".join(e.text for e in self._buffers.get(session_id, []))

    def entries(self, session_id: str) -> list[GlossEntry]:
        """Return a copy of the current buffer."""
        return list(self._buffers.get(session_id, []))

    def flush(self, session_id: str) -> list[GlossEntry]:
        """Return all buffered entries and clear the buffer and partial ID."""
        entries = list(self._buffers.pop(session_id, []))
        self._partial_ids.pop(session_id, None)
        return entries

    def is_empty(self, session_id: str) -> bool:
        """Return True if no entries are buffered for ``session_id``."""
        return not self._buffers.get(session_id)

    def reset(self, session_id: str) -> None:
        """Clear buffer and partial ID for ``session_id``. Safe on unknown sessions."""
        self._buffers.pop(session_id, None)
        self._partial_ids.pop(session_id, None)

    # ------------------------------------------------------------------
    # Partial segment ID tracking
    # ------------------------------------------------------------------

    def set_partial_id(self, session_id: str, segment_id: str) -> None:
        """Store the segment ID of the currently emitted PARTIAL."""
        self._partial_ids[session_id] = segment_id

    def pop_partial_id(self, session_id: str) -> str | None:
        """Return and remove the stored PARTIAL segment ID, or None."""
        return self._partial_ids.pop(session_id, None)
