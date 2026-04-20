"""Tests for ActivityDetector — RED phase (implementation does not exist yet)."""

from __future__ import annotations


from sozia.common.models import (
    HAND_LANDMARK_COUNT,
    POSE_LANDMARK_COUNT,
    LandmarkFrame,
)
from sozia.server.fusion.activity_detector import ADEvent, ActivityDetector

# ---------------------------------------------------------------------------
# Session ID fixtures
# ---------------------------------------------------------------------------

_SESSION = "aaaaaaaa-0000-4000-a000-000000000001"
_SESSION_B = "bbbbbbbb-0000-4000-b000-000000000002"

# ---------------------------------------------------------------------------
# Frame-construction helpers
# ---------------------------------------------------------------------------


def _pose_at(wx: float = 0.5, wy: float = 0.5) -> list[list[float]]:
    """Build a minimal 33-point pose array.

    Indices 15 (left wrist) and 16 (right wrist) are set to (wx, wy, 0.0).
    All other points are (0.0, 0.0, 0.0).
    """
    points = [[0.0, 0.0, 0.0]] * POSE_LANDMARK_COUNT
    # list is shared — replace with distinct lists to avoid aliasing
    points = [list(p) for p in points]
    points[15] = [wx, wy, 0.0]
    points[16] = [wx, wy, 0.0]
    return points


def _hand_at(wx: float = 0.5, wy: float = 0.5) -> list[list[float]]:
    """Build a 21-point hand array with wrist (index 0) at (wx, wy, 0.0)."""
    points = [[0.0, 0.0, 0.0]] * HAND_LANDMARK_COUNT
    points = [list(p) for p in points]
    points[0] = [wx, wy, 0.0]
    return points


def _make_frame(
    session_id: str = _SESSION,
    timestamp_ms: int = 0,
    pose: list[list[float]] | None = None,
    left_hand: list[list[float]] | None = None,
    right_hand: list[list[float]] | None = None,
) -> LandmarkFrame:
    """Construct a minimal LandmarkFrame.

    At least one landmark array must be non-null (LandmarkFrame invariant).
    When all three explicit args are None, pose defaults to _pose_at() so the
    frame is valid.
    """
    if pose is None and left_hand is None and right_hand is None:
        pose = _pose_at()
    return LandmarkFrame(
        session_id=session_id,
        timestamp_ms=timestamp_ms,
        face_landmarks=None,
        left_hand_landmarks=left_hand,
        right_hand_landmarks=right_hand,
        pose_landmarks=pose,
    )


# ---------------------------------------------------------------------------
# Helper: feed N frames with the same wrist position and hands present
# ---------------------------------------------------------------------------


def _feed_active_frames(
    detector: ActivityDetector,
    n: int,
    session_id: str = _SESSION,
    wx: float = 0.5,
    wy: float = 0.5,
    start_ts: int = 0,
    step: float = 0.01,
) -> list[ADEvent]:
    """Feed *n* frames that have hands present and increasing wrist position.

    Each successive frame moves the wrist by *step* in both x and y so that
    velocity > default threshold (0.005).
    """
    events = []
    for i in range(n):
        x = wx + i * step
        y = wy + i * step
        frame = _make_frame(
            session_id=session_id,
            timestamp_ms=start_ts + i,
            pose=_pose_at(x, y),
            left_hand=_hand_at(x, y),
        )
        events.append(detector.update(frame))
    return events


# ===========================================================================
# 1. IDLE when no hands present
# ===========================================================================


class TestIdleNoHands:
    def test_returns_idle_with_no_hand_landmarks(self):
        detector = ActivityDetector()
        for i in range(10):
            frame = _make_frame(
                timestamp_ms=i,
                pose=_pose_at(0.1 * i, 0.1 * i),  # pose present, no hands
            )
            event = detector.update(frame)
            assert event == ADEvent.IDLE, f"frame {i}: expected IDLE, got {event}"

    def test_never_becomes_active_without_hands(self):
        detector = ActivityDetector(activate_frames=2)
        events = [
            detector.update(
                _make_frame(timestamp_ms=i, pose=_pose_at(float(i), float(i)))
            )
            for i in range(20)
        ]
        assert all(e == ADEvent.IDLE for e in events)


# ===========================================================================
# 2. IDLE when hands present but no motion
# ===========================================================================


class TestIdleNoMotion:
    def test_static_wrist_position_stays_idle(self):
        detector = ActivityDetector(velocity_threshold=0.005)
        # Same wrist position every frame — velocity is 0.
        hand = _hand_at(0.5, 0.5)
        pose = _pose_at(0.5, 0.5)
        for i in range(10):
            frame = _make_frame(timestamp_ms=i, pose=pose, left_hand=hand)
            event = detector.update(frame)
            assert event == ADEvent.IDLE, f"frame {i}: expected IDLE, got {event}"

    def test_sub_threshold_motion_stays_idle(self):
        """Velocity below threshold should not activate."""
        detector = ActivityDetector(velocity_threshold=0.1, activate_frames=2)
        # Move wrist by 0.001 per frame — well below 0.1 threshold.
        for i in range(10):
            x = 0.5 + i * 0.001
            frame = _make_frame(
                timestamp_ms=i,
                pose=_pose_at(x, x),
                left_hand=_hand_at(x, x),
            )
            event = detector.update(frame)
            assert event == ADEvent.IDLE


# ===========================================================================
# 3. Rising edge — STARTED fires on exactly the Nth consecutive active frame
# ===========================================================================


class TestRisingEdge:
    def test_started_fires_on_nth_active_frame(self):
        n = 3
        detector = ActivityDetector(activate_frames=n, deactivate_frames=5)

        # First frame: no previous position → velocity = 0 → IDLE.
        first = _make_frame(
            timestamp_ms=0, pose=_pose_at(0.1, 0.1), left_hand=_hand_at(0.1, 0.1)
        )
        assert detector.update(first) == ADEvent.IDLE

        # Subsequent frames with motion.  STARTED must appear on frame index n
        # (counting from 1 after the anchor frame).
        events = []
        for i in range(1, n + 1):
            x = 0.1 + i * 0.05
            frame = _make_frame(
                timestamp_ms=i,
                pose=_pose_at(x, x),
                left_hand=_hand_at(x, x),
            )
            events.append(detector.update(frame))

        # Frames 1 .. n-1 should be IDLE (not yet enough consecutive frames)
        for idx, ev in enumerate(events[:-1]):
            assert ev == ADEvent.IDLE, f"premature activation at sub-frame {idx + 1}"

        # Frame n should be STARTED.
        assert events[-1] == ADEvent.STARTED

    def test_started_requires_consecutive_frames(self):
        """A gap in activity resets the consecutive counter."""
        detector = ActivityDetector(activate_frames=3, deactivate_frames=100)

        def active_frame(ts: int, x: float) -> LandmarkFrame:
            return _make_frame(
                timestamp_ms=ts, pose=_pose_at(x, x), left_hand=_hand_at(x, x)
            )

        def idle_frame(ts: int) -> LandmarkFrame:
            # No hands — no activity.
            return _make_frame(timestamp_ms=ts, pose=_pose_at(0.5, 0.5))

        # 2 active, 1 idle, 2 active → counter reset → should not fire STARTED.
        events = [
            detector.update(active_frame(0, 0.10)),
            detector.update(active_frame(1, 0.15)),
            detector.update(idle_frame(2)),
            detector.update(active_frame(3, 0.25)),
            detector.update(active_frame(4, 0.30)),
        ]
        assert ADEvent.STARTED not in events


# ===========================================================================
# 4. ACTIVE continues after STARTED
# ===========================================================================


class TestActiveState:
    def test_events_after_started_are_active(self):
        n = 2
        detector = ActivityDetector(activate_frames=n, deactivate_frames=100)

        # Anchor frame.
        detector.update(
            _make_frame(
                timestamp_ms=0, pose=_pose_at(0.0, 0.0), left_hand=_hand_at(0.0, 0.0)
            )
        )

        # Reach STARTED.
        for i in range(1, n + 1):
            x = i * 0.05
            detector.update(
                _make_frame(
                    timestamp_ms=i, pose=_pose_at(x, x), left_hand=_hand_at(x, x)
                )
            )

        # Subsequent active frames.
        for i in range(n + 1, n + 6):
            x = i * 0.05
            frame = _make_frame(
                timestamp_ms=i, pose=_pose_at(x, x), left_hand=_hand_at(x, x)
            )
            event = detector.update(frame)
            assert event == ADEvent.ACTIVE, f"expected ACTIVE at frame {i}, got {event}"

    def test_is_active_returns_true_during_active_state(self):
        n = 2
        detector = ActivityDetector(activate_frames=n, deactivate_frames=100)
        assert not detector.is_active(_SESSION)

        # Bring detector to ACTIVE.
        detector.update(
            _make_frame(
                timestamp_ms=0, pose=_pose_at(0.0, 0.0), left_hand=_hand_at(0.0, 0.0)
            )
        )
        for i in range(1, n + 1):
            x = i * 0.05
            detector.update(
                _make_frame(
                    timestamp_ms=i, pose=_pose_at(x, x), left_hand=_hand_at(x, x)
                )
            )

        assert detector.is_active(_SESSION)


# ===========================================================================
# 5. Falling edge — ENDED fires on exactly the Nth consecutive inactive frame
# ===========================================================================


class TestFallingEdge:
    def _reach_active(self, detector: ActivityDetector, activate_frames: int) -> int:
        """Drive detector to ACTIVE state; return next timestamp to use."""
        detector.update(
            _make_frame(
                timestamp_ms=0, pose=_pose_at(0.0, 0.0), left_hand=_hand_at(0.0, 0.0)
            )
        )
        for i in range(1, activate_frames + 1):
            x = i * 0.05
            detector.update(
                _make_frame(
                    timestamp_ms=i, pose=_pose_at(x, x), left_hand=_hand_at(x, x)
                )
            )
        return activate_frames + 1

    def test_ended_fires_on_nth_inactive_frame(self):
        deactivate_n = 4
        activate_n = 2
        detector = ActivityDetector(
            activate_frames=activate_n, deactivate_frames=deactivate_n
        )
        next_ts = self._reach_active(detector, activate_n)

        # Feed deactivate_n frames with no hands.
        events = []
        for i in range(deactivate_n):
            frame = _make_frame(timestamp_ms=next_ts + i, pose=_pose_at(0.5, 0.5))
            events.append(detector.update(frame))

        # First deactivate_n - 1 events should be ACTIVE (still in grace period).
        for idx, ev in enumerate(events[:-1]):
            assert ev == ADEvent.ACTIVE, f"premature ENDED at inactive frame {idx + 1}"

        # Final frame triggers ENDED.
        assert events[-1] == ADEvent.ENDED

    def test_ended_requires_consecutive_inactive_frames(self):
        """A single active frame mid-deactivation resets the counter."""
        deactivate_n = 3
        activate_n = 2
        detector = ActivityDetector(
            activate_frames=activate_n, deactivate_frames=deactivate_n
        )
        next_ts = self._reach_active(detector, activate_n)

        def inactive(ts: int) -> LandmarkFrame:
            return _make_frame(timestamp_ms=ts, pose=_pose_at(0.5, 0.5))

        def active_frame(ts: int, x: float) -> LandmarkFrame:
            return _make_frame(
                timestamp_ms=ts, pose=_pose_at(x, x), left_hand=_hand_at(x, x)
            )

        # 2 inactive, 1 active (resets counter), 2 more inactive → ENDED not reached.
        events = [
            detector.update(inactive(next_ts + 0)),
            detector.update(inactive(next_ts + 1)),
            detector.update(active_frame(next_ts + 2, 0.9)),
            detector.update(inactive(next_ts + 3)),
            detector.update(inactive(next_ts + 4)),
        ]
        assert ADEvent.ENDED not in events


# ===========================================================================
# 6. IDLE after ENDED
# ===========================================================================


class TestIdleAfterEnded:
    def test_idle_events_follow_ended(self):
        activate_n = 2
        deactivate_n = 2
        detector = ActivityDetector(
            activate_frames=activate_n, deactivate_frames=deactivate_n
        )

        # Reach ACTIVE.
        detector.update(
            _make_frame(
                timestamp_ms=0, pose=_pose_at(0.0, 0.0), left_hand=_hand_at(0.0, 0.0)
            )
        )
        for i in range(1, activate_n + 1):
            x = i * 0.05
            detector.update(
                _make_frame(
                    timestamp_ms=i, pose=_pose_at(x, x), left_hand=_hand_at(x, x)
                )
            )

        # Reach ENDED.
        next_ts = activate_n + 1
        for i in range(deactivate_n):
            detector.update(
                _make_frame(timestamp_ms=next_ts + i, pose=_pose_at(0.5, 0.5))
            )
        next_ts += deactivate_n

        # Additional inactive frames should all be IDLE.
        for i in range(5):
            ev = detector.update(
                _make_frame(timestamp_ms=next_ts + i, pose=_pose_at(0.5, 0.5))
            )
            assert ev == ADEvent.IDLE, f"expected IDLE after ENDED, got {ev}"

    def test_is_active_returns_false_after_ended(self):
        activate_n = 2
        deactivate_n = 2
        detector = ActivityDetector(
            activate_frames=activate_n, deactivate_frames=deactivate_n
        )

        detector.update(
            _make_frame(
                timestamp_ms=0, pose=_pose_at(0.0, 0.0), left_hand=_hand_at(0.0, 0.0)
            )
        )
        for i in range(1, activate_n + 1):
            x = i * 0.05
            detector.update(
                _make_frame(
                    timestamp_ms=i, pose=_pose_at(x, x), left_hand=_hand_at(x, x)
                )
            )

        next_ts = activate_n + 1
        for i in range(deactivate_n):
            detector.update(
                _make_frame(timestamp_ms=next_ts + i, pose=_pose_at(0.5, 0.5))
            )

        assert not detector.is_active(_SESSION)


# ===========================================================================
# 7. Hysteresis prevents flicker
# ===========================================================================


class TestHysteresis:
    def test_single_inactive_frame_does_not_end_activity(self):
        activate_n = 2
        deactivate_n = 4  # grace period is 4 frames
        detector = ActivityDetector(
            activate_frames=activate_n, deactivate_frames=deactivate_n
        )

        # Reach ACTIVE state.
        detector.update(
            _make_frame(
                timestamp_ms=0, pose=_pose_at(0.0, 0.0), left_hand=_hand_at(0.0, 0.0)
            )
        )
        for i in range(1, activate_n + 1):
            x = i * 0.05
            detector.update(
                _make_frame(
                    timestamp_ms=i, pose=_pose_at(x, x), left_hand=_hand_at(x, x)
                )
            )

        next_ts = activate_n + 1

        # One inactive frame — should not trigger ENDED.
        ev = detector.update(_make_frame(timestamp_ms=next_ts, pose=_pose_at(0.5, 0.5)))
        assert ev != ADEvent.ENDED
        assert ev != ADEvent.IDLE

        # Resume activity — still ACTIVE (or STARTED again if implementation re-arms, but not IDLE/ENDED).
        next_ts += 1
        resume_x = 0.9
        ev = detector.update(
            _make_frame(
                timestamp_ms=next_ts,
                pose=_pose_at(resume_x, resume_x),
                left_hand=_hand_at(resume_x, resume_x),
            )
        )
        assert ev in (ADEvent.ACTIVE, ADEvent.STARTED)

    def test_deactivate_frames_minus_one_inactive_does_not_end(self):
        activate_n = 2
        deactivate_n = 5
        detector = ActivityDetector(
            activate_frames=activate_n, deactivate_frames=deactivate_n
        )

        detector.update(
            _make_frame(
                timestamp_ms=0, pose=_pose_at(0.0, 0.0), left_hand=_hand_at(0.0, 0.0)
            )
        )
        for i in range(1, activate_n + 1):
            x = i * 0.05
            detector.update(
                _make_frame(
                    timestamp_ms=i, pose=_pose_at(x, x), left_hand=_hand_at(x, x)
                )
            )

        next_ts = activate_n + 1
        events = []
        # Feed deactivate_n - 1 inactive frames → must NOT include ENDED.
        for i in range(deactivate_n - 1):
            ev = detector.update(
                _make_frame(timestamp_ms=next_ts + i, pose=_pose_at(0.5, 0.5))
            )
            events.append(ev)

        assert ADEvent.ENDED not in events


# ===========================================================================
# 8. Pose wrist fallback (use hand[0] when pose is None)
# ===========================================================================


class TestPoseWristFallback:
    def test_motion_detected_without_pose(self):
        """When pose is absent, hand[0] coords provide the wrist position."""
        activate_n = 2
        detector = ActivityDetector(
            velocity_threshold=0.005, activate_frames=activate_n, deactivate_frames=10
        )

        # First frame: hand present, no pose — velocity = 0 (no prior position).
        first = _make_frame(
            timestamp_ms=0,
            pose=None,
            left_hand=_hand_at(0.1, 0.1),
            right_hand=None,
        )
        assert detector.update(first) == ADEvent.IDLE

        # Subsequent frames with motion (no pose).
        events = []
        for i in range(1, activate_n + 1):
            x = 0.1 + i * 0.05
            frame = _make_frame(
                timestamp_ms=i,
                pose=None,
                left_hand=_hand_at(x, x),
                right_hand=None,
            )
            events.append(detector.update(frame))

        assert events[-1] == ADEvent.STARTED

    def test_right_hand_fallback_when_left_absent(self):
        """When left hand is absent, right hand[0] is used for wrist."""
        activate_n = 2
        detector = ActivityDetector(
            velocity_threshold=0.005, activate_frames=activate_n, deactivate_frames=10
        )

        detector.update(
            _make_frame(timestamp_ms=0, pose=None, right_hand=_hand_at(0.1, 0.1))
        )

        events = []
        for i in range(1, activate_n + 1):
            x = 0.1 + i * 0.05
            frame = _make_frame(timestamp_ms=i, pose=None, right_hand=_hand_at(x, x))
            events.append(detector.update(frame))

        assert events[-1] == ADEvent.STARTED


# ===========================================================================
# 9. Multi-session isolation
# ===========================================================================


class TestMultiSessionIsolation:
    def test_two_sessions_are_independent(self):
        detector = ActivityDetector(activate_frames=2, deactivate_frames=10)

        # Drive session A to ACTIVE.
        detector.update(
            _make_frame(
                session_id=_SESSION,
                timestamp_ms=0,
                pose=_pose_at(0.0, 0.0),
                left_hand=_hand_at(0.0, 0.0),
            )
        )
        for i in range(1, 3):
            x = i * 0.05
            detector.update(
                _make_frame(
                    session_id=_SESSION,
                    timestamp_ms=i,
                    pose=_pose_at(x, x),
                    left_hand=_hand_at(x, x),
                )
            )

        assert detector.is_active(_SESSION)
        # Session B has received no frames — must not be active.
        assert not detector.is_active(_SESSION_B)

    def test_session_b_activity_does_not_affect_session_a(self):
        detector = ActivityDetector(activate_frames=2, deactivate_frames=10)

        # Activate B.
        detector.update(
            _make_frame(
                session_id=_SESSION_B,
                timestamp_ms=0,
                pose=_pose_at(0.0, 0.0),
                left_hand=_hand_at(0.0, 0.0),
            )
        )
        for i in range(1, 3):
            x = i * 0.05
            detector.update(
                _make_frame(
                    session_id=_SESSION_B,
                    timestamp_ms=i,
                    pose=_pose_at(x, x),
                    left_hand=_hand_at(x, x),
                )
            )

        # A has never seen a frame.
        assert not detector.is_active(_SESSION)

    def test_reset_one_session_does_not_affect_other(self):
        activate_n = 2
        detector = ActivityDetector(activate_frames=activate_n, deactivate_frames=10)

        for sid in (_SESSION, _SESSION_B):
            detector.update(
                _make_frame(
                    session_id=sid,
                    timestamp_ms=0,
                    pose=_pose_at(0.0, 0.0),
                    left_hand=_hand_at(0.0, 0.0),
                )
            )
            for i in range(1, activate_n + 1):
                x = i * 0.05
                detector.update(
                    _make_frame(
                        session_id=sid,
                        timestamp_ms=i,
                        pose=_pose_at(x, x),
                        left_hand=_hand_at(x, x),
                    )
                )

        assert detector.is_active(_SESSION)
        assert detector.is_active(_SESSION_B)

        detector.reset(_SESSION)
        assert not detector.is_active(_SESSION)
        assert detector.is_active(_SESSION_B)


# ===========================================================================
# 10. reset() clears state
# ===========================================================================


class TestReset:
    def test_reset_returns_to_idle_state(self):
        activate_n = 2
        detector = ActivityDetector(activate_frames=activate_n, deactivate_frames=10)

        # Reach ACTIVE.
        detector.update(
            _make_frame(
                timestamp_ms=0, pose=_pose_at(0.0, 0.0), left_hand=_hand_at(0.0, 0.0)
            )
        )
        for i in range(1, activate_n + 1):
            x = i * 0.05
            detector.update(
                _make_frame(
                    timestamp_ms=i, pose=_pose_at(x, x), left_hand=_hand_at(x, x)
                )
            )

        assert detector.is_active(_SESSION)

        detector.reset(_SESSION)
        assert not detector.is_active(_SESSION)

    def test_first_frame_after_reset_is_idle(self):
        """After reset, velocity has no previous reference → IDLE."""
        activate_n = 1
        detector = ActivityDetector(activate_frames=activate_n, deactivate_frames=10)

        detector.update(
            _make_frame(
                timestamp_ms=0, pose=_pose_at(0.0, 0.0), left_hand=_hand_at(0.0, 0.0)
            )
        )
        detector.update(
            _make_frame(
                timestamp_ms=1, pose=_pose_at(0.5, 0.5), left_hand=_hand_at(0.5, 0.5)
            )
        )
        # Now ACTIVE (activate_frames=1).

        detector.reset(_SESSION)

        # First frame after reset: no prior position → velocity = 0.
        ev = detector.update(
            _make_frame(
                timestamp_ms=2, pose=_pose_at(0.9, 0.9), left_hand=_hand_at(0.9, 0.9)
            )
        )
        assert ev == ADEvent.IDLE

    def test_reset_unknown_session_is_noop(self):
        detector = ActivityDetector()
        detector.reset("does-not-exist")  # must not raise

    def test_reset_clears_consecutive_counter(self):
        """Resetting mid-countdown prevents a stale counter from firing."""
        activate_n = 3
        detector = ActivityDetector(activate_frames=activate_n, deactivate_frames=10)

        # Feed 2 active frames (one short of STARTED).
        detector.update(
            _make_frame(
                timestamp_ms=0, pose=_pose_at(0.0, 0.0), left_hand=_hand_at(0.0, 0.0)
            )
        )
        detector.update(
            _make_frame(
                timestamp_ms=1, pose=_pose_at(0.1, 0.1), left_hand=_hand_at(0.1, 0.1)
            )
        )

        detector.reset(_SESSION)

        # After reset, the accumulated count must be gone — one more active frame
        # alone must NOT trigger STARTED (needs activate_n fresh frames).
        ev = detector.update(
            _make_frame(
                timestamp_ms=2, pose=_pose_at(0.2, 0.2), left_hand=_hand_at(0.2, 0.2)
            )
        )
        assert ev == ADEvent.IDLE
