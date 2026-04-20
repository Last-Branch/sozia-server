"""ActivityDetector — per-session signing activity state machine.

Determines whether a signer is actively signing by requiring both hand
presence AND wrist motion above a velocity threshold. Hysteresis prevents
per-frame flicker: ``activate_frames`` consecutive active frames are needed
to enter the ACTIVE state, and ``deactivate_frames`` consecutive inactive
frames to leave it.

Sits upstream of FrameAccumulator in the sign pipeline.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sozia.common.models import LandmarkFrame

# Pose landmark indices for wrists (MediaPipe convention).
_LEFT_WRIST_IDX = 15
_RIGHT_WRIST_IDX = 16

_DEFAULT_VELOCITY_THRESHOLD: float = 0.005
_DEFAULT_ACTIVATE_FRAMES: int = 3
_DEFAULT_DEACTIVATE_FRAMES: int = 5


class ADEvent(Enum):
    IDLE = "IDLE"  # inactive, no state change
    STARTED = "STARTED"  # rising edge: inactive → active
    ACTIVE = "ACTIVE"  # active, no state change
    ENDED = "ENDED"  # falling edge: active → inactive


class ActivityDetector:
    """Per-session state machine for detecting signing activity.

    Args:
        velocity_threshold: Minimum wrist displacement (normalised coords) per
            frame to count as motion. Default 0.005.
        activate_frames: Consecutive active frames required to enter ACTIVE
            state. Default 3.
        deactivate_frames: Consecutive inactive frames required to leave ACTIVE
            state. Default 5.
    """

    def __init__(
        self,
        velocity_threshold: float = _DEFAULT_VELOCITY_THRESHOLD,
        activate_frames: int = _DEFAULT_ACTIVATE_FRAMES,
        deactivate_frames: int = _DEFAULT_DEACTIVATE_FRAMES,
    ) -> None:
        if activate_frames < 1:
            raise ValueError(f"activate_frames must be >= 1, got {activate_frames}")
        if deactivate_frames < 1:
            raise ValueError(f"deactivate_frames must be >= 1, got {deactivate_frames}")
        self._velocity_threshold = velocity_threshold
        self._activate_frames = activate_frames
        self._deactivate_frames = deactivate_frames

        # Per-session per-side previous wrist positions: sid → {"left": (x,y), "right": (x,y)}
        self._prev_wrists: dict[str, dict[str, tuple[float, float]]] = {}
        self._active_count: dict[str, int] = {}
        self._inactive_count: dict[str, int] = {}
        self._is_active: dict[str, bool] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, frame: LandmarkFrame) -> ADEvent:
        """Process one LandmarkFrame and return the current AD event.

        Args:
            frame: Incoming frame from the WebSocket gateway.

        Returns:
            ADEvent reflecting the current activity state transition.
        """
        sid = frame.session_id
        has_hands = (
            frame.left_hand_landmarks is not None
            or frame.right_hand_landmarks is not None
        )

        current_wrists = self._get_all_wrists(frame)
        prev_wrists = self._prev_wrists.get(sid, {})

        # Max velocity across both wrists; each side compared against its own prev.
        velocity = 0.0
        for side, pos in current_wrists.items():
            prev = prev_wrists.get(side)
            if prev is not None:
                velocity = max(velocity, math.hypot(pos[0] - prev[0], pos[1] - prev[1]))

        # Update previous wrist positions; clear when nothing visible to avoid stale
        # velocity spike when the signer re-enters the frame after an absence.
        if current_wrists:
            self._prev_wrists[sid] = current_wrists
        else:
            self._prev_wrists.pop(sid, None)

        frame_active = has_hands and velocity > self._velocity_threshold

        is_active = self._is_active.get(sid, False)
        active_count = self._active_count.get(sid, 0)
        inactive_count = self._inactive_count.get(sid, 0)

        if frame_active:
            active_count += 1
            inactive_count = 0
            self._active_count[sid] = active_count
            self._inactive_count[sid] = 0

            if not is_active and active_count >= self._activate_frames:
                self._is_active[sid] = True
                self._active_count[sid] = 0
                return ADEvent.STARTED

            return ADEvent.ACTIVE if is_active else ADEvent.IDLE

        else:
            inactive_count += 1
            active_count = 0
            self._active_count[sid] = 0
            self._inactive_count[sid] = inactive_count

            if is_active and inactive_count >= self._deactivate_frames:
                self._is_active[sid] = False
                self._inactive_count[sid] = 0
                return ADEvent.ENDED

            return ADEvent.ACTIVE if is_active else ADEvent.IDLE

    def is_active(self, session_id: str) -> bool:
        """Return whether ``session_id`` is currently in the ACTIVE state."""
        return self._is_active.get(session_id, False)

    def reset(self, session_id: str) -> None:
        """Clear all state for ``session_id``. Safe to call on unknown sessions."""
        self._prev_wrists.pop(session_id, None)
        self._active_count.pop(session_id, None)
        self._inactive_count.pop(session_id, None)
        self._is_active.pop(session_id, None)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_all_wrists(frame: LandmarkFrame) -> dict[str, tuple[float, float]]:
        """Extract available wrist positions from a frame.

        Returns a dict with keys ``"left"`` and/or ``"right"``. Prefers pose
        landmarks (indices 15/16) over hand landmarks (index 0) when pose is
        present. Returns an empty dict when no wrist data is available.
        """
        wrists: dict[str, tuple[float, float]] = {}

        if frame.pose_landmarks is not None:
            pose = frame.pose_landmarks
            if len(pose) > _LEFT_WRIST_IDX and len(pose[_LEFT_WRIST_IDX]) >= 2:
                lw = pose[_LEFT_WRIST_IDX]
                wrists["left"] = (float(lw[0]), float(lw[1]))
            if len(pose) > _RIGHT_WRIST_IDX and len(pose[_RIGHT_WRIST_IDX]) >= 2:
                rw = pose[_RIGHT_WRIST_IDX]
                wrists["right"] = (float(rw[0]), float(rw[1]))
        else:
            if (
                frame.left_hand_landmarks is not None
                and len(frame.left_hand_landmarks[0]) >= 2
            ):
                hw = frame.left_hand_landmarks[0]
                wrists["left"] = (float(hw[0]), float(hw[1]))
            if (
                frame.right_hand_landmarks is not None
                and len(frame.right_hand_landmarks[0]) >= 2
            ):
                hw = frame.right_hand_landmarks[0]
                wrists["right"] = (float(hw[0]), float(hw[1]))

        return wrists
