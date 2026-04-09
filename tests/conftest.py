"""Shared pytest fixtures for sozia-server tests."""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Session identifiers
# ---------------------------------------------------------------------------


@pytest.fixture
def valid_session_id() -> str:
    return "550e8400-e29b-41d4-a716-446655440000"


@pytest.fixture
def valid_segment_id() -> str:
    return "6ba7b810-9dad-11d1-80b4-00c04fd430c8"


# ---------------------------------------------------------------------------
# Landmark arrays
# ---------------------------------------------------------------------------


@pytest.fixture
def valid_face_landmarks() -> list[list[float]]:
    """83 face landmark points with x,y in [0,1] and unconstrained z."""
    return [[0.5, 0.5, 0.0]] * 83


@pytest.fixture
def valid_left_hand_landmarks() -> list[list[float]]:
    return [[0.5, 0.5, 0.0]] * 21


@pytest.fixture
def valid_right_hand_landmarks() -> list[list[float]]:
    return [[0.5, 0.5, 0.0]] * 21


@pytest.fixture
def valid_pose_landmarks() -> list[list[float]]:
    return [[0.5, 0.5, 0.0]] * 33


# ---------------------------------------------------------------------------
# Audio features
# ---------------------------------------------------------------------------


@pytest.fixture
def valid_audio_features() -> list[list[float]]:
    """10 frames × 13 MFCC dimensions."""
    return [[float(i) for i in range(13)] for _ in range(10)]
