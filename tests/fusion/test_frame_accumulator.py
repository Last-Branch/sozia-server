"""Tests for FrameAccumulator."""

from __future__ import annotations

import numpy as np
import pytest

from sozia.common.models import (
    FACE_LANDMARK_COUNT,
    HAND_LANDMARK_COUNT,
    POSE_LANDMARK_COUNT,
    LandmarkFrame,
)
from sozia.fusion.frame_accumulator import FEATURE_DIM, FrameAccumulator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SESSION = "550e8400-e29b-41d4-a716-446655440000"
_SESSION_B = "660e8400-e29b-41d4-a716-446655440000"


def _pose() -> list[list[float]]:
    # x, y in [0.0, 1.0]; z is unconstrained (MediaPipe relative depth).
    n = POSE_LANDMARK_COUNT
    return [[i / n, (n - 1 - i) / n, float(i)] for i in range(n)]


def _face() -> list[list[float]]:
    return [[0.1, 0.2, 0.3]] * FACE_LANDMARK_COUNT


def _hand() -> list[list[float]]:
    return [[0.5, 0.5, 0.5]] * HAND_LANDMARK_COUNT


def _frame(
    session_id: str = _SESSION,
    timestamp_ms: int = 0,
    with_pose: bool = True,
    with_face: bool = True,
    with_left: bool = True,
    with_right: bool = True,
) -> LandmarkFrame:
    # At least one array must be non-null.
    return LandmarkFrame(
        session_id=session_id,
        timestamp_ms=timestamp_ms,
        face_landmarks=_face() if with_face else None,
        left_hand_landmarks=_hand() if with_left else None,
        right_hand_landmarks=_hand() if with_right else None,
        pose_landmarks=_pose() if with_pose else None,
    )


# ---------------------------------------------------------------------------
# FEATURE_DIM constant
# ---------------------------------------------------------------------------


class TestFeatureDim:
    def test_feature_dim_matches_landmark_counts(self):
        expected = (
            POSE_LANDMARK_COUNT * 3
            + FACE_LANDMARK_COUNT * 3
            + HAND_LANDMARK_COUNT * 3
            + HAND_LANDMARK_COUNT * 3
        )
        assert FEATURE_DIM == expected

    def test_feature_dim_value(self):
        # pose(99) + face(249) + left(63) + right(63) = 474
        assert FEATURE_DIM == 474


# ---------------------------------------------------------------------------
# Window fill / partial fill
# ---------------------------------------------------------------------------


class TestWindowFill:
    def test_partial_fill_returns_none(self):
        acc = FrameAccumulator(window_size=5)
        for i in range(4):
            result = acc.add(_frame(timestamp_ms=i))
            assert result is None

    def test_exact_fill_returns_array(self):
        acc = FrameAccumulator(window_size=5)
        result = None
        for i in range(5):
            result = acc.add(_frame(timestamp_ms=i))
        assert result is not None
        assert isinstance(result, np.ndarray)

    def test_returned_shape_is_window_size_by_feature_dim(self):
        window = 10
        acc = FrameAccumulator(window_size=window)
        result = None
        for i in range(window):
            result = acc.add(_frame(timestamp_ms=i))
        assert result.shape == (window, FEATURE_DIM)

    def test_buffer_resets_after_emit(self):
        acc = FrameAccumulator(window_size=3)
        for i in range(3):
            acc.add(_frame(timestamp_ms=i))
        # Buffer should have been reset; next 2 frames → None
        assert acc.add(_frame(timestamp_ms=10)) is None
        assert acc.add(_frame(timestamp_ms=11)) is None

    def test_second_window_also_emits(self):
        acc = FrameAccumulator(window_size=3)
        for i in range(3):
            acc.add(_frame(timestamp_ms=i))
        # Second batch
        result = None
        for i in range(3):
            result = acc.add(_frame(timestamp_ms=10 + i))
        assert result is not None
        assert result.shape == (3, FEATURE_DIM)


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_clears_buffer(self):
        acc = FrameAccumulator(window_size=5)
        for i in range(3):
            acc.add(_frame(timestamp_ms=i))
        assert acc.pending_count(_SESSION) == 3

        acc.reset(_SESSION)
        assert acc.pending_count(_SESSION) == 0

    def test_reset_after_full_window_is_noop(self):
        acc = FrameAccumulator(window_size=3)
        for i in range(3):
            acc.add(_frame(timestamp_ms=i))
        # After emit, buffer is already empty; reset should still work without error
        acc.reset(_SESSION)
        assert acc.pending_count(_SESSION) == 0

    def test_reset_unknown_session_is_noop(self):
        acc = FrameAccumulator(window_size=5)
        acc.reset("does-not-exist")  # must not raise


# ---------------------------------------------------------------------------
# pending_count
# ---------------------------------------------------------------------------


class TestPendingCount:
    def test_pending_count_starts_at_zero(self):
        acc = FrameAccumulator(window_size=5)
        assert acc.pending_count(_SESSION) == 0

    def test_pending_count_increments(self):
        acc = FrameAccumulator(window_size=5)
        for i in range(3):
            acc.add(_frame(timestamp_ms=i))
        assert acc.pending_count(_SESSION) == 3

    def test_pending_count_resets_after_emit(self):
        acc = FrameAccumulator(window_size=3)
        for i in range(3):
            acc.add(_frame(timestamp_ms=i))
        assert acc.pending_count(_SESSION) == 0


# ---------------------------------------------------------------------------
# Session isolation
# ---------------------------------------------------------------------------


class TestSessionIsolation:
    def test_two_sessions_buffered_independently(self):
        acc = FrameAccumulator(window_size=5)
        acc.add(_frame(session_id=_SESSION, timestamp_ms=0))
        acc.add(_frame(session_id=_SESSION_B, timestamp_ms=0))
        assert acc.pending_count(_SESSION) == 1
        assert acc.pending_count(_SESSION_B) == 1

    def test_reset_session_a_does_not_affect_session_b(self):
        acc = FrameAccumulator(window_size=5)
        for i in range(3):
            acc.add(_frame(session_id=_SESSION, timestamp_ms=i))
        for i in range(2):
            acc.add(_frame(session_id=_SESSION_B, timestamp_ms=i))
        acc.reset(_SESSION)
        assert acc.pending_count(_SESSION) == 0
        assert acc.pending_count(_SESSION_B) == 2

    def test_session_b_fills_independently(self):
        acc = FrameAccumulator(window_size=3)
        # Fill session B to window.
        result = None
        for i in range(3):
            result = acc.add(_frame(session_id=_SESSION_B, timestamp_ms=i))
        assert result is not None
        # Session A is unaffected.
        assert acc.pending_count(_SESSION) == 0


# ---------------------------------------------------------------------------
# Numpy values — zero-filling absent groups
# ---------------------------------------------------------------------------


class TestNumpyValues:
    def test_absent_pose_yields_zeros(self):
        acc = FrameAccumulator(window_size=1)
        frame = _frame(with_pose=False)
        result = acc.add(frame)
        # Pose occupies first POSE_LANDMARK_COUNT * 3 dims.
        pose_slice = result[0, : POSE_LANDMARK_COUNT * 3]
        assert np.all(pose_slice == 0.0)

    def test_absent_face_yields_zeros(self):
        acc = FrameAccumulator(window_size=1)
        frame = _frame(with_face=False)
        result = acc.add(frame)
        pose_end = POSE_LANDMARK_COUNT * 3
        face_slice = result[0, pose_end : pose_end + FACE_LANDMARK_COUNT * 3]
        assert np.all(face_slice == 0.0)

    def test_absent_left_hand_yields_zeros(self):
        acc = FrameAccumulator(window_size=1)
        frame = _frame(with_left=False)
        result = acc.add(frame)
        left_start = (POSE_LANDMARK_COUNT + FACE_LANDMARK_COUNT) * 3
        left_slice = result[0, left_start : left_start + HAND_LANDMARK_COUNT * 3]
        assert np.all(left_slice == 0.0)

    def test_absent_right_hand_yields_zeros(self):
        acc = FrameAccumulator(window_size=1)
        frame = _frame(with_right=False)
        result = acc.add(frame)
        right_start = (POSE_LANDMARK_COUNT + FACE_LANDMARK_COUNT + HAND_LANDMARK_COUNT) * 3
        right_slice = result[0, right_start : right_start + HAND_LANDMARK_COUNT * 3]
        assert np.all(right_slice == 0.0)

    def test_present_pose_values_match(self):
        acc = FrameAccumulator(window_size=1)
        frame = _frame(with_face=False, with_left=False, with_right=False)
        result = acc.add(frame)
        expected = np.asarray(_pose(), dtype=np.float32).ravel()
        np.testing.assert_array_almost_equal(result[0, : POSE_LANDMARK_COUNT * 3], expected)

    def test_dtype_is_float32(self):
        acc = FrameAccumulator(window_size=1)
        result = acc.add(_frame())
        assert result.dtype == np.float32
