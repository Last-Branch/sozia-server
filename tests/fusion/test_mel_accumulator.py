"""Tests for MelAccumulator — utterance-boundary mel buffering."""

from __future__ import annotations

import numpy as np
import pytest

from sozia.server.fusion.mel_accumulator import MelAccumulator

SESSION = "test-session"


def _speech(t: int = 50, n: int = 128) -> np.ndarray:
    arr = np.full((t, n), -10.0, dtype=np.float32)
    arr[0, 0] = 0.0  # one bin well above -1.0 threshold
    return arr


def _silence(t: int = 50, n: int = 128) -> np.ndarray:
    return np.full((t, n), -10.0, dtype=np.float32)


# ---------------------------------------------------------------------------
# Basic accumulation
# ---------------------------------------------------------------------------


class TestMelAccumulatorBasic:
    def test_returns_none_while_buffering(self):
        acc = MelAccumulator()
        assert acc.push(SESSION, _speech()) is None
        assert acc.push(SESSION, _speech()) is None

    def test_flush_on_1s_silence_after_speech(self):
        acc = MelAccumulator(min_silence_frames=100)
        acc.push(SESSION, _speech(t=50))
        result = acc.push(SESSION, _silence(t=100))
        assert result is not None
        assert result.shape[0] == 50  # only speech frames

    def test_silence_before_speech_is_ignored(self):
        acc = MelAccumulator()
        assert acc.push(SESSION, _silence()) is None
        assert acc.push(SESSION, _silence()) is None

    def test_flush_on_max_frames(self):
        acc = MelAccumulator(max_frames=100)
        acc.push(SESSION, _speech(t=60))
        result = acc.push(SESSION, _speech(t=60))
        assert result is not None
        assert result.shape[0] == 120

    def test_buffer_resets_after_flush(self):
        acc = MelAccumulator(min_silence_frames=50)
        acc.push(SESSION, _speech(t=50))
        acc.push(SESSION, _silence(t=50))
        assert acc.pending_frames(SESSION) == 0

    def test_pending_frames_tracks_count(self):
        acc = MelAccumulator()
        acc.push(SESSION, _speech(t=50))
        assert acc.pending_frames(SESSION) == 50
        acc.push(SESSION, _speech(t=30))
        assert acc.pending_frames(SESSION) == 80

    def test_short_silence_does_not_flush(self):
        # 50 frames silence < min_silence_frames=100 → no flush
        acc = MelAccumulator(min_silence_frames=100)
        acc.push(SESSION, _speech(t=50))
        result = acc.push(SESSION, _silence(t=50))
        assert result is None
        assert acc.pending_frames(SESSION) == 50

    def test_consecutive_silence_accumulates(self):
        # Two silent chunks totalling >= min_silence_frames → flush
        acc = MelAccumulator(min_silence_frames=100)
        acc.push(SESSION, _speech(t=50))
        acc.push(SESSION, _silence(t=60))  # 60 < 100, no flush
        result = acc.push(SESSION, _silence(t=60))  # 120 >= 100 → flush
        assert result is not None

    def test_speech_resets_silence_counter(self):
        # Speech chunk resets the consecutive-silence counter
        acc = MelAccumulator(min_silence_frames=100)
        acc.push(SESSION, _speech(t=50))
        acc.push(SESSION, _silence(t=60))  # 60 frames silence
        acc.push(SESSION, _speech(t=20))   # resets silence to 0
        result = acc.push(SESSION, _silence(t=60))  # only 60 again → no flush
        assert result is None

    def test_flush_on_silence_even_after_short_speech(self):
        # Even a small amount of speech triggers flush on 1 s silence
        acc = MelAccumulator(min_silence_frames=50)
        acc.push(SESSION, _speech(t=10))
        result = acc.push(SESSION, _silence(t=50))
        assert result is not None
        assert result.shape[0] == 10


# ---------------------------------------------------------------------------
# Multi-session isolation
# ---------------------------------------------------------------------------


class TestMelAccumulatorMultiSession:
    def test_sessions_are_isolated(self):
        acc = MelAccumulator(min_silence_frames=50)
        acc.push("s1", _speech(t=50))
        result_s2 = acc.push("s2", _silence(t=50))
        assert result_s2 is None  # s2 never saw speech

    def test_flush_in_one_session_does_not_affect_other(self):
        acc = MelAccumulator(min_silence_frames=50)
        acc.push("s1", _speech(t=50))
        acc.push("s1", _silence(t=50))  # flush s1
        acc.push("s2", _speech(t=50))
        assert acc.pending_frames("s2") == 50


# ---------------------------------------------------------------------------
# Output shape and content
# ---------------------------------------------------------------------------


class TestMelAccumulatorOutput:
    def test_output_concatenates_time_axis(self):
        acc = MelAccumulator(min_silence_frames=50)
        acc.push(SESSION, _speech(t=30))
        acc.push(SESSION, _speech(t=40))
        result = acc.push(SESSION, _silence(t=50))
        assert result is not None
        assert result.shape == (70, 128)

    def test_speech_content_preserved(self):
        acc = MelAccumulator(min_silence_frames=50)
        chunk = _speech(t=50)
        chunk[5, 10] = 999.0  # sentinel
        acc.push(SESSION, chunk)
        result = acc.push(SESSION, _silence(t=50))
        assert result is not None
        assert result[5, 10] == pytest.approx(999.0)


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


class TestMelAccumulatorReset:
    def test_reset_clears_buffer(self):
        acc = MelAccumulator()
        acc.push(SESSION, _speech(t=50))
        acc.reset(SESSION)
        assert acc.pending_frames(SESSION) == 0

    def test_reset_prevents_stale_flush(self):
        acc = MelAccumulator(min_silence_frames=50)
        acc.push(SESSION, _speech(t=50))
        acc.reset(SESSION)
        result = acc.push(SESSION, _silence(t=50))
        assert result is None  # speech flag was cleared

    def test_reset_clears_silence_counter(self):
        # After reset, accumulated silence from before should not carry over
        acc = MelAccumulator(min_silence_frames=100)
        acc.push(SESSION, _speech(t=50))
        acc.push(SESSION, _silence(t=60))  # partial silence, no flush
        acc.reset(SESSION)
        acc.push(SESSION, _speech(t=50))
        result = acc.push(SESSION, _silence(t=60))  # 60 < 100, fresh counter
        assert result is None

    def test_reset_on_unknown_session_is_safe(self):
        acc = MelAccumulator()
        acc.reset("nonexistent")  # must not raise
