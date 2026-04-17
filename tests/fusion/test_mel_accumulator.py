"""Tests for MelAccumulator — utterance-boundary mel buffering."""

from __future__ import annotations

import numpy as np
import pytest

from sozia.server.fusion.mel_accumulator import MelAccumulator

SESSION = "test-session"

# Helpers: speech chunks peak above -4; silence peaks below -4.
def _speech(t: int = 50, n: int = 128) -> np.ndarray:
    arr = np.full((t, n), -10.0, dtype=np.float32)
    arr[0, 0] = -2.0  # one bin well above threshold
    return arr


def _silence(t: int = 50, n: int = 128) -> np.ndarray:
    return np.full((t, n), -10.0, dtype=np.float32)


# ---------------------------------------------------------------------------
# Basic accumulation
# ---------------------------------------------------------------------------


class TestMelAccumulatorBasic:
    def test_returns_none_while_buffering(self):
        acc = MelAccumulator(min_flush_frames=200)
        assert acc.push(SESSION, _speech()) is None
        assert acc.push(SESSION, _speech()) is None

    def test_flush_on_silence_after_speech(self):
        acc = MelAccumulator(min_flush_frames=50)
        acc.push(SESSION, _speech(t=50))
        result = acc.push(SESSION, _silence())
        assert result is not None
        assert result.shape[0] == 50  # only the speech chunk

    def test_silence_before_speech_is_ignored(self):
        acc = MelAccumulator()
        assert acc.push(SESSION, _silence()) is None
        assert acc.push(SESSION, _silence()) is None

    def test_flush_on_max_frames(self):
        acc = MelAccumulator(max_frames=100)
        # Two 60-frame speech chunks → should flush when total ≥ 100
        acc.push(SESSION, _speech(t=60))
        result = acc.push(SESSION, _speech(t=60))
        assert result is not None
        assert result.shape[0] == 120

    def test_buffer_resets_after_flush(self):
        acc = MelAccumulator(min_flush_frames=50)
        acc.push(SESSION, _speech(t=50))
        acc.push(SESSION, _silence())  # flush
        # New speech after flush should not carry over frames
        assert acc.pending_frames(SESSION) == 0

    def test_pending_frames_tracks_count(self):
        acc = MelAccumulator(min_flush_frames=200)
        acc.push(SESSION, _speech(t=50))
        assert acc.pending_frames(SESSION) == 50
        acc.push(SESSION, _speech(t=30))
        assert acc.pending_frames(SESSION) == 80

    def test_short_pause_does_not_flush(self):
        # min_flush_frames=200; only 50 frames buffered → silence should not flush
        acc = MelAccumulator(min_flush_frames=200)
        acc.push(SESSION, _speech(t=50))
        result = acc.push(SESSION, _silence())
        assert result is None
        assert acc.pending_frames(SESSION) == 50


# ---------------------------------------------------------------------------
# Multi-session isolation
# ---------------------------------------------------------------------------


class TestMelAccumulatorMultiSession:
    def test_sessions_are_isolated(self):
        acc = MelAccumulator(min_flush_frames=50)
        acc.push("s1", _speech(t=50))
        result_s2 = acc.push("s2", _silence())
        assert result_s2 is None  # s2 never saw speech

    def test_flush_in_one_session_does_not_affect_other(self):
        acc = MelAccumulator(min_flush_frames=50)
        acc.push("s1", _speech(t=50))
        acc.push("s1", _silence())  # flush s1
        acc.push("s2", _speech(t=50))
        assert acc.pending_frames("s2") == 50


# ---------------------------------------------------------------------------
# Output shape and content
# ---------------------------------------------------------------------------


class TestMelAccumulatorOutput:
    def test_output_concatenates_time_axis(self):
        acc = MelAccumulator(min_flush_frames=50)
        acc.push(SESSION, _speech(t=30))
        acc.push(SESSION, _speech(t=40))
        result = acc.push(SESSION, _silence())
        assert result is not None
        assert result.shape == (70, 128)

    def test_speech_content_preserved(self):
        acc = MelAccumulator(min_flush_frames=50)
        chunk = _speech(t=50)
        chunk[5, 10] = 999.0  # sentinel
        acc.push(SESSION, chunk)
        result = acc.push(SESSION, _silence())
        assert result is not None
        assert result[5, 10] == pytest.approx(999.0)


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


class TestMelAccumulatorReset:
    def test_reset_clears_buffer(self):
        acc = MelAccumulator(min_flush_frames=200)
        acc.push(SESSION, _speech(t=50))
        acc.reset(SESSION)
        assert acc.pending_frames(SESSION) == 0

    def test_reset_prevents_stale_flush(self):
        acc = MelAccumulator(min_flush_frames=50)
        acc.push(SESSION, _speech(t=50))
        acc.reset(SESSION)
        result = acc.push(SESSION, _silence())
        assert result is None  # speech flag was cleared

    def test_reset_on_unknown_session_is_safe(self):
        acc = MelAccumulator()
        acc.reset("nonexistent")  # must not raise
