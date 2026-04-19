"""FrameAccumulator — per-session landmark frame buffer for the sign pipeline.

TslRecognitionEngine.predict() expects a (T, feature_dim) numpy array — a
multi-frame sequence. The WebSocket gateway delivers one LandmarkFrame per
message. FrameAccumulator buffers frames per session and emits a numpy batch
when the window threshold is reached (R7: buffering and inference triggering
stay inside sozia.server.fusion, not the gateway).

Feature vector layout per frame (concatenated; zero-filled if absent):
    pose:       33 × 4 = 132 dims  (x, y, z, visibility)
    face:       83 × 3 = 249 dims
    left_hand:  21 × 3 =  63 dims
    right_hand: 21 × 3 =  63 dims
    ────────────────────────────
    total:               507 dims

Pose landmarks carry a 4th visibility component matching the sozia-research
training pipeline and the client extraction output.
"""

from __future__ import annotations

import numpy as np

from sozia.common.models import (
    FACE_LANDMARK_COUNT,
    HAND_LANDMARK_COUNT,
    POSE_LANDMARK_COUNT,
    LandmarkFrame,
)

# Flat feature vector length per frame — derived from LandmarkFrame structure.
FEATURE_DIM: int = (
    POSE_LANDMARK_COUNT * 4  # x, y, z, visibility
    + FACE_LANDMARK_COUNT * 3
    + HAND_LANDMARK_COUNT * 3  # left hand
    + HAND_LANDMARK_COUNT * 3  # right hand
)


def _flatten_or_zeros(
    arr: list[list[float]] | None, count: int, components: int = 3
) -> np.ndarray:
    """Return flattened landmark array or a zero vector if absent."""
    if arr is None:
        return np.zeros(count * components, dtype=np.float32)
    return np.asarray(arr, dtype=np.float32).ravel()


def _flatten_pose(arr: list[list[float]] | None) -> np.ndarray:
    """Return 33×4 pose vector. Pads visibility=1.0 if client sends only x,y,z."""
    if arr is None:
        return np.zeros(POSE_LANDMARK_COUNT * 4, dtype=np.float32)
    rows = []
    for lm in arr:
        rows.extend(lm[:3])
        rows.append(lm[3] if len(lm) >= 4 else 1.0)
    return np.array(rows, dtype=np.float32)


def _frame_to_row(frame: LandmarkFrame) -> np.ndarray:
    """Convert a single LandmarkFrame to a 1-D feature vector of length FEATURE_DIM."""
    return np.concatenate([
        _flatten_pose(frame.pose_landmarks),
        _flatten_or_zeros(frame.face_landmarks, FACE_LANDMARK_COUNT),
        _flatten_or_zeros(frame.left_hand_landmarks, HAND_LANDMARK_COUNT),
        _flatten_or_zeros(frame.right_hand_landmarks, HAND_LANDMARK_COUNT),
    ])


class FrameAccumulator:
    """Buffers LandmarkFrame objects per session and emits numpy batches.

    Usage::

        accumulator = FrameAccumulator(window_size=30)
        batch = accumulator.add(frame)  # returns None until window is full
        if batch is not None:
            result = await tsl_engine.predict(batch)  # shape (30, 474)

    Args:
        window_size: Number of frames to collect before emitting a batch.
            Defaults to 30, matching a typical ~1 s window at 30 fps.
    """

    def __init__(self, window_size: int = 30) -> None:
        self._window_size = window_size
        self._buffers: dict[str, list[LandmarkFrame]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, frame: LandmarkFrame) -> np.ndarray | None:
        """Append a frame to the session buffer.

        Args:
            frame: A LandmarkFrame received from the WebSocket gateway.

        Returns:
            A numpy array of shape ``(window_size, FEATURE_DIM)`` when the
            buffer reaches ``window_size``, otherwise ``None``.
        """
        buf = self._buffers.setdefault(frame.session_id, [])
        buf.append(frame)

        if len(buf) >= self._window_size:
            batch = self._to_numpy(buf)
            self._buffers[frame.session_id] = []
            return batch

        return None

    def reset(self, session_id: str) -> None:
        """Discard all buffered frames for the given session.

        Safe to call even if no frames have been accumulated yet.

        Args:
            session_id: The session whose buffer should be cleared.
        """
        self._buffers.pop(session_id, None)

    def pending_count(self, session_id: str) -> int:
        """Return the number of frames currently buffered for a session.

        Args:
            session_id: Session to query.

        Returns:
            Integer count in range ``[0, window_size)``.
        """
        return len(self._buffers.get(session_id, []))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _to_numpy(self, frames: list[LandmarkFrame]) -> np.ndarray:
        """Stack a list of LandmarkFrames into shape ``(T, FEATURE_DIM)``."""
        rows = [_frame_to_row(f) for f in frames]
        return np.stack(rows, axis=0)
