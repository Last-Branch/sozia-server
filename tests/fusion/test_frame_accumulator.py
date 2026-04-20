"""Tests for FrameAccumulator — rolling-window variant."""

from __future__ import annotations

import numpy as np
import pytest

from sozia.common.models import (
    FACE_LANDMARK_COUNT,
    HAND_LANDMARK_COUNT,
    POSE_LANDMARK_COUNT,
    LandmarkFrame,
)
from sozia.server.fusion.frame_accumulator import FEATURE_DIM, FrameAccumulator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SESSION = "550e8400-e29b-41d4-a716-446655440000"
_SESSION_B = "660e8400-e29b-41d4-a716-446655440000"


def _pose() -> list[list[float]]:
    n = POSE_LANDMARK_COUNT
    # LandmarkFrame validates exactly 3 coordinates per point; _flatten_pose
    # pads visibility=1.0 internally when the 4th component is absent.
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
    return LandmarkFrame(
        session_id=session_id,
        timestamp_ms=timestamp_ms,
        face_landmarks=_face() if with_face else None,
        left_hand_landmarks=_hand() if with_left else None,
        right_hand_landmarks=_hand() if with_right else None,
        pose_landmarks=_pose() if with_pose else None,
    )


# ---------------------------------------------------------------------------
# FEATURE_DIM constant — 507 dims (pose*4 + face*3 + hand*3 + hand*3)
# ---------------------------------------------------------------------------


class TestFeatureDim:
    def test_feature_dim_composition(self):
        expected = (
            POSE_LANDMARK_COUNT * 4  # x, y, z, visibility
            + FACE_LANDMARK_COUNT * 3
            + HAND_LANDMARK_COUNT * 3  # left
            + HAND_LANDMARK_COUNT * 3  # right
        )
        assert FEATURE_DIM == expected

    def test_feature_dim_value(self):
        # pose(132) + face(249) + left(63) + right(63) = 507
        assert FEATURE_DIM == 507


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


class TestConstructor:
    def test_default_window_size(self):
        acc = FrameAccumulator()
        assert acc._window_size == 30

    def test_default_stride_is_10_for_large_window(self):
        acc = FrameAccumulator()  # window=30 → stride=min(10, 30)=10
        assert acc._stride == 10

    def test_default_stride_clamped_for_small_window(self):
        acc = FrameAccumulator(window_size=3)  # stride=min(10, 3)=3 (tumbling)
        assert acc._stride == 3

    def test_small_window_no_stride_arg(self):
        acc = FrameAccumulator(window_size=1)  # must not raise; stride=1
        assert acc._stride == 1

    def test_custom_window_and_stride(self):
        acc = FrameAccumulator(window_size=20, stride=5)
        assert acc._window_size == 20
        assert acc._stride == 5

    def test_stride_equal_to_window_size_is_valid(self):
        acc = FrameAccumulator(window_size=5, stride=5)
        assert acc._stride == 5

    def test_explicit_stride_greater_than_window_size_raises(self):
        with pytest.raises(ValueError):
            FrameAccumulator(window_size=5, stride=6)

    def test_explicit_stride_zero_raises(self):
        with pytest.raises(ValueError):
            FrameAccumulator(window_size=5, stride=0)


# ---------------------------------------------------------------------------
# First window fill — behaviour up to the first emit
# ---------------------------------------------------------------------------


class TestFirstWindowFill:
    def test_partial_fill_returns_none(self):
        acc = FrameAccumulator(window_size=5, stride=2)
        for i in range(4):
            assert acc.add(_frame(timestamp_ms=i)) is None

    def test_exact_fill_returns_array(self):
        acc = FrameAccumulator(window_size=5, stride=2)
        result = None
        for i in range(5):
            result = acc.add(_frame(timestamp_ms=i))
        assert result is not None
        assert isinstance(result, np.ndarray)

    def test_returned_shape_is_window_size_by_feature_dim(self):
        window = 10
        acc = FrameAccumulator(window_size=window, stride=3)
        result = None
        for i in range(window):
            result = acc.add(_frame(timestamp_ms=i))
        assert result.shape == (window, FEATURE_DIM)


# ---------------------------------------------------------------------------
# Rolling-window stride behaviour
# ---------------------------------------------------------------------------


class TestRollingWindow:
    def test_buffer_retains_overlap_after_first_emit(self):
        # window=5, stride=2 → 5-2=3 frames retained
        acc = FrameAccumulator(window_size=5, stride=2)
        for i in range(5):
            acc.add(_frame(timestamp_ms=i))
        assert acc.pending_count(_SESSION) == 3

    def test_second_emit_arrives_after_stride_more_frames(self):
        acc = FrameAccumulator(window_size=5, stride=2)
        for i in range(5):
            acc.add(_frame(timestamp_ms=i))
        # 1 frame → still short (buf = 4)
        assert acc.add(_frame(timestamp_ms=5)) is None
        # 2nd frame → emit (buf = 5)
        result = acc.add(_frame(timestamp_ms=6))
        assert result is not None
        assert result.shape == (5, FEATURE_DIM)

    def test_pending_count_after_second_emit(self):
        acc = FrameAccumulator(window_size=5, stride=2)
        for i in range(7):  # 5 → first emit; +2 → second emit
            acc.add(_frame(timestamp_ms=i))
        # 5 - 2 = 3 retained after second emit
        assert acc.pending_count(_SESSION) == 3

    def test_stride_equal_to_window_size_clears_buffer(self):
        # Tumbling window: stride == window_size, no overlap
        acc = FrameAccumulator(window_size=3, stride=3)
        for i in range(3):
            acc.add(_frame(timestamp_ms=i))
        assert acc.pending_count(_SESSION) == 0

    def test_stride_equal_to_window_size_second_window_independent(self):
        acc = FrameAccumulator(window_size=3, stride=3)
        for i in range(3):
            acc.add(_frame(timestamp_ms=i))
        result = None
        for i in range(3):
            result = acc.add(_frame(timestamp_ms=10 + i))
        assert result is not None
        assert result.shape == (3, FEATURE_DIM)

    def test_many_consecutive_windows_emit_correctly(self):
        acc = FrameAccumulator(window_size=4, stride=2)
        emits = []
        for i in range(12):
            r = acc.add(_frame(timestamp_ms=i))
            if r is not None:
                emits.append(r)
        # Windows at frames [0-3], [2-5], [4-7], [6-9], [8-11]
        assert len(emits) == 5
        for r in emits:
            assert r.shape == (4, FEATURE_DIM)

    def test_no_emit_below_window_after_multiple_strides(self):
        # Ensure we don't get spurious emits as buffer grows
        acc = FrameAccumulator(window_size=5, stride=2)
        results = [acc.add(_frame(timestamp_ms=i)) for i in range(4)]
        assert all(r is None for r in results)


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_clears_buffer(self):
        acc = FrameAccumulator(window_size=5, stride=2)
        for i in range(3):
            acc.add(_frame(timestamp_ms=i))
        assert acc.pending_count(_SESSION) == 3

        acc.reset(_SESSION)
        assert acc.pending_count(_SESSION) == 0

    def test_reset_after_emit_clears_retained_overlap(self):
        acc = FrameAccumulator(window_size=5, stride=2)
        for i in range(5):
            acc.add(_frame(timestamp_ms=i))
        # 3 frames retained after emit
        assert acc.pending_count(_SESSION) == 3

        acc.reset(_SESSION)
        assert acc.pending_count(_SESSION) == 0

    def test_reset_unknown_session_is_noop(self):
        acc = FrameAccumulator(window_size=5)
        acc.reset("does-not-exist")  # must not raise

    def test_add_after_reset_starts_fresh(self):
        acc = FrameAccumulator(window_size=3, stride=1)
        for i in range(3):
            acc.add(_frame(timestamp_ms=i))
        acc.reset(_SESSION)
        # After reset, need another full window
        for i in range(2):
            assert acc.add(_frame(timestamp_ms=10 + i)) is None
        result = acc.add(_frame(timestamp_ms=12))
        assert result is not None


# ---------------------------------------------------------------------------
# pending_count
# ---------------------------------------------------------------------------


class TestPendingCount:
    def test_starts_at_zero(self):
        acc = FrameAccumulator(window_size=5)
        assert acc.pending_count(_SESSION) == 0

    def test_increments_with_each_add(self):
        acc = FrameAccumulator(window_size=5, stride=2)
        for i in range(3):
            acc.add(_frame(timestamp_ms=i))
        assert acc.pending_count(_SESSION) == 3

    def test_drops_to_overlap_after_emit(self):
        acc = FrameAccumulator(window_size=5, stride=2)
        for i in range(5):
            acc.add(_frame(timestamp_ms=i))
        assert acc.pending_count(_SESSION) == 3  # 5 - 2


# ---------------------------------------------------------------------------
# Session isolation
# ---------------------------------------------------------------------------


class TestSessionIsolation:
    def test_two_sessions_buffered_independently(self):
        acc = FrameAccumulator(window_size=5, stride=2)
        acc.add(_frame(session_id=_SESSION, timestamp_ms=0))
        acc.add(_frame(session_id=_SESSION_B, timestamp_ms=0))
        assert acc.pending_count(_SESSION) == 1
        assert acc.pending_count(_SESSION_B) == 1

    def test_reset_session_a_does_not_affect_session_b(self):
        acc = FrameAccumulator(window_size=5, stride=2)
        for i in range(3):
            acc.add(_frame(session_id=_SESSION, timestamp_ms=i))
        for i in range(2):
            acc.add(_frame(session_id=_SESSION_B, timestamp_ms=i))
        acc.reset(_SESSION)
        assert acc.pending_count(_SESSION) == 0
        assert acc.pending_count(_SESSION_B) == 2

    def test_session_b_fills_independently(self):
        acc = FrameAccumulator(window_size=3, stride=1)
        result = None
        for i in range(3):
            result = acc.add(_frame(session_id=_SESSION_B, timestamp_ms=i))
        assert result is not None
        assert acc.pending_count(_SESSION) == 0

    def test_emit_in_session_b_does_not_affect_session_a_buffer(self):
        acc = FrameAccumulator(window_size=3, stride=1)
        for i in range(2):
            acc.add(_frame(session_id=_SESSION, timestamp_ms=i))
        for i in range(3):
            acc.add(_frame(session_id=_SESSION_B, timestamp_ms=i))
        # session A has 2 frames; session B emitted and has 2 retained
        assert acc.pending_count(_SESSION) == 2
        assert acc.pending_count(_SESSION_B) == 2


# ---------------------------------------------------------------------------
# Numpy values — zero-filling absent groups, dtype, pose visibility
# ---------------------------------------------------------------------------


def _single_frame_acc() -> FrameAccumulator:
    """One-shot accumulator for testing feature vector values."""
    return FrameAccumulator(window_size=1)


class TestNumpyValues:
    def test_absent_pose_yields_zeros(self):
        result = _single_frame_acc().add(_frame(with_pose=False))
        pose_slice = result[0, : POSE_LANDMARK_COUNT * 4]
        assert np.all(pose_slice == 0.0)

    def test_absent_face_yields_zeros(self):
        result = _single_frame_acc().add(_frame(with_face=False))
        pose_end = POSE_LANDMARK_COUNT * 4
        face_slice = result[0, pose_end : pose_end + FACE_LANDMARK_COUNT * 3]
        assert np.all(face_slice == 0.0)

    def test_absent_left_hand_yields_zeros(self):
        result = _single_frame_acc().add(_frame(with_left=False))
        left_start = POSE_LANDMARK_COUNT * 4 + FACE_LANDMARK_COUNT * 3
        left_slice = result[0, left_start : left_start + HAND_LANDMARK_COUNT * 3]
        assert np.all(left_slice == 0.0)

    def test_absent_right_hand_yields_zeros(self):
        result = _single_frame_acc().add(_frame(with_right=False))
        right_start = (
            POSE_LANDMARK_COUNT * 4 + FACE_LANDMARK_COUNT * 3 + HAND_LANDMARK_COUNT * 3
        )
        right_slice = result[0, right_start : right_start + HAND_LANDMARK_COUNT * 3]
        assert np.all(right_slice == 0.0)

    def test_pose_xyz_values_match(self):
        # _flatten_pose interleaves visibility=1.0 after each (x, y, z).
        frame = _frame(with_face=False, with_left=False, with_right=False)
        result = _single_frame_acc().add(frame)
        raw = np.asarray(_pose(), dtype=np.float32)  # shape (33, 3)
        xyz_indices = [i * 4 + c for i in range(POSE_LANDMARK_COUNT) for c in range(3)]
        np.testing.assert_array_almost_equal(result[0, xyz_indices], raw.ravel())

    def test_pose_visibility_padded_to_1(self):
        # _flatten_pose pads visibility=1.0 when only (x, y, z) are provided.
        frame = _frame(with_face=False, with_left=False, with_right=False)
        result = _single_frame_acc().add(frame)
        visibility_indices = [3 + i * 4 for i in range(POSE_LANDMARK_COUNT)]
        assert np.all(result[0, visibility_indices] == 1.0)

    def test_dtype_is_float32(self):
        result = _single_frame_acc().add(_frame())
        assert result.dtype == np.float32
