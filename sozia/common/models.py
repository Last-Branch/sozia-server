"""Shared data structures for the Sozia system.

All types in this module cross the client-server boundary or are used by more
than one package. They carry no runtime logic — only structure, constraints,
and serialisation rules.

Dependency rule R3: this module must not import from sozia.server or sozia.client.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal


# ===========================================================================
# Enumerations
# ===========================================================================


class SessionState(str, Enum):
    """Lifecycle state of a transcription session.

    Driven by SessionController (client) and observed by the UI and server.
    """

    IDLE = "IDLE"
    INITIALIZING = "INITIALIZING"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    DEGRADED = "DEGRADED"
    ERROR = "ERROR"


class ModalityPath(str, Enum):
    """Which inference pipeline is active for the session.

    The system is modal-exclusive: only one path is active at a time.
    """

    SPEECH = "SPEECH"
    SIGN = "SIGN"


class ModalityType(str, Enum):
    """The specific inference modality that produced a result.

    Used for source labelling in transcript segments and for fusion logic.
    """

    ASR = "ASR"
    LIP_READING = "LIP_READING"
    TSL_RECOGNITION = "TSL_RECOGNITION"
    GLOSS_TO_TEXT = "GLOSS_TO_TEXT"


class SegmentStatus(str, Enum):
    """Whether a transcript segment is tentative or finalised.

    A PARTIAL segment may be replaced by a FINAL one via replaces_segment_id.
    A FINAL segment is never revised.
    """

    PARTIAL = "PARTIAL"
    FINAL = "FINAL"


# ===========================================================================
# Validation helpers (private)
# ===========================================================================


def _validate_non_empty_str(value: str, field: str) -> None:
    if not value:
        raise ValueError(f"{field} must be a non-empty string")


def _validate_non_negative_int(value: int, field: str) -> None:
    if value < 0:
        raise ValueError(f"{field} must be >= 0, got {value}")


def _validate_confidence(value: float, field: str = "confidence") -> None:
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"{field} must be in [0.0, 1.0], got {value}")


def _validate_landmark_array(
    arr: list[list[float]],
    expected_len: int,
    name: str,
    check_xy_unit_range: bool = False,
    components: int = 3,
) -> None:
    """Validate a landmark array's length and point dimensions.

    Args:
        arr: 2-D list of landmark points.
        expected_len: Required number of points.
        name: Field name for error messages.
        check_xy_unit_range: If True, enforce x and y coords in [0.0, 1.0].
            z is always unconstrained (MediaPipe relative depth).
        components: Expected number of values per point. 3 for face/hand
            ([x, y, z]), 4 for pose ([x, y, z, visibility]).
    """
    if len(arr) != expected_len:
        raise ValueError(f"{name} must have {expected_len} points, got {len(arr)}")
    for i, point in enumerate(arr):
        if len(point) != components:
            raise ValueError(
                f"{name}[{i}] must have {components} coordinates, got {len(point)}"
            )
        if check_xy_unit_range:
            x, y = point[0], point[1]
            if not (0.0 <= x <= 1.0):
                raise ValueError(f"{name}[{i}].x must be in [0.0, 1.0], got {x}")
            if not (0.0 <= y <= 1.0):
                raise ValueError(f"{name}[{i}].y must be in [0.0, 1.0], got {y}")


def _validate_rectangular_2d(
    features: list[list[float]], field: str = "features"
) -> None:
    """Validate that a 2-D list is non-empty and rectangular.

    Args:
        features: 2-D list of shape [T, D].
        field: Field name for error messages.

    Raises:
        ValueError: If T < 1, D < 1, or rows have inconsistent length.
    """
    if not features:
        raise ValueError(f"{field} must have at least one row (T >= 1)")
    d = len(features[0])
    if d == 0:
        raise ValueError(f"{field} rows must have at least one column (D >= 1)")
    for i, row in enumerate(features):
        if len(row) != d:
            raise ValueError(
                f"{field} is not rectangular: row 0 has {d} cols, row {i} has {len(row)}"
            )


# ===========================================================================
# Data Transfer Objects (DTOs)
# ===========================================================================


@dataclass(frozen=True)
class ModelConfig:
    """Configuration passed to InferenceEngine.load_model().

    Args:
        model_id: Unique identifier for the model (e.g. "whisper-small-tr").
        weights_path: Absolute path to model weight file(s).
        device: Compute device — "cpu" or "cuda".
        params: Engine-specific parameters (e.g. language, quantisation flags).
            The dict reference is frozen but the contents remain mutable.
    """

    model_id: str
    weights_path: str
    device: str
    params: dict

    def __post_init__(self) -> None:
        _validate_non_empty_str(self.model_id, "model_id")
        _validate_non_empty_str(self.weights_path, "weights_path")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError(f"device must be 'cpu' or 'cuda', got '{self.device}'")


@dataclass(frozen=True)
class ModalityResult:
    """Output of a single inference engine.

    Produced server-side and consumed by FusionOrchestrator.
    Not sent directly to the client — the fusion layer wraps it into a
    TranscriptSegment first.

    Args:
        modality_type: Which engine produced this result.
        text: Recognised text (UTF-8). May be empty string for silence.
        confidence: Confidence score in [0.0, 1.0].
        timestamp_ms: Position in the session timeline (ms since session start).
        duration_ms: Duration of the audio/video segment this result covers.
        inference_latency_ms: Wall-clock time the engine took. Used for latency
            budget monitoring against the 2200 ms cloud inference budget.
    """

    modality_type: ModalityType
    text: str
    confidence: float
    timestamp_ms: int
    duration_ms: int
    inference_latency_ms: int

    def __post_init__(self) -> None:
        _validate_confidence(self.confidence)
        _validate_non_negative_int(self.timestamp_ms, "timestamp_ms")
        _validate_non_negative_int(self.duration_ms, "duration_ms")
        _validate_non_negative_int(self.inference_latency_ms, "inference_latency_ms")


@dataclass(frozen=True)
class PipelineHealth:
    """Real-time health report for a client-side pipeline.

    Sent upstream periodically so DegradedModeHandler can adjust fusion
    strategy. Cross-field invariants enforce that audio-only and video-only
    fields are not mixed.

    Args:
        session_id: UUID v4 of the active session.
        pipeline: "audio" or "video".
        available: False if the pipeline cannot produce features (device lost,
            permission denied, etc.).
        fps: Frame rate — video pipeline only; None for audio.
        snr: Signal-to-noise ratio in dB — audio pipeline only; None for video.
        face_detected: Whether a face is currently detected — video only; None
            for audio.
        last_updated_ms: Milliseconds since session start.
    """

    session_id: str
    pipeline: Literal["audio", "video"]
    available: bool
    fps: float | None
    snr: float | None
    face_detected: bool | None
    last_updated_ms: int

    def __post_init__(self) -> None:
        _validate_non_empty_str(self.session_id, "session_id")
        if self.pipeline not in {"audio", "video"}:
            raise ValueError(
                f"pipeline must be 'audio' or 'video', got '{self.pipeline}'"
            )
        _validate_non_negative_int(self.last_updated_ms, "last_updated_ms")

        if self.pipeline == "audio":
            if self.fps is not None:
                raise ValueError("fps must be None for audio pipeline")
            if self.face_detected is not None:
                raise ValueError("face_detected must be None for audio pipeline")
        else:  # video
            if self.snr is not None:
                raise ValueError("snr must be None for video pipeline")


# Landmark count constants — sourced from sozia-research extraction pipeline.
# face: 83-point linguistically-relevant subset (not the full 478-point mesh).
# See tsl_recognition/config.py FACE_LANDMARK_INDICES for the exact subset.
FACE_LANDMARK_COUNT = 83
HAND_LANDMARK_COUNT = 21
POSE_LANDMARK_COUNT = 33


@dataclass(frozen=True)
class LandmarkFrame:
    """A single timestamped frame of body landmarks extracted by MediaPipe.

    Transmitted upstream from client to server via WebSocket.

    Invariant: at least one landmark array must be non-null. A frame with all
    nulls is never transmitted.

    Coordinate conventions:
        - x, y: normalised to [0.0, 1.0] relative to frame dimensions.
        - z: MediaPipe relative depth — not normalised, may be any finite float.

    Args:
        session_id: UUID v4 of the active session.
        timestamp_ms: Milliseconds since session start. Must be >= 0 and
            monotonically increasing within a session (enforced by SessionHandler).
        face_landmarks: 83 × [x, y, z] face mesh points (subset). None if not
            detected.
        left_hand_landmarks: 21 × [x, y, z]. None if not detected.
        right_hand_landmarks: 21 × [x, y, z]. None if not detected.
        pose_landmarks: 33 × [x, y, z, visibility]. None if not detected.
    """

    session_id: str
    timestamp_ms: int
    face_landmarks: list[list[float]] | None
    left_hand_landmarks: list[list[float]] | None
    right_hand_landmarks: list[list[float]] | None
    pose_landmarks: list[list[float]] | None

    def __post_init__(self) -> None:
        _validate_non_empty_str(self.session_id, "session_id")
        _validate_non_negative_int(self.timestamp_ms, "timestamp_ms")

        if all(
            arr is None
            for arr in (
                self.face_landmarks,
                self.left_hand_landmarks,
                self.right_hand_landmarks,
                self.pose_landmarks,
            )
        ):
            raise ValueError("at least one landmark array must be non-null")

        if self.face_landmarks is not None:
            _validate_landmark_array(
                self.face_landmarks,
                FACE_LANDMARK_COUNT,
                "face_landmarks",
                check_xy_unit_range=False,
            )
        if self.left_hand_landmarks is not None:
            _validate_landmark_array(
                self.left_hand_landmarks,
                HAND_LANDMARK_COUNT,
                "left_hand_landmarks",
                check_xy_unit_range=False,
            )
        if self.right_hand_landmarks is not None:
            _validate_landmark_array(
                self.right_hand_landmarks,
                HAND_LANDMARK_COUNT,
                "right_hand_landmarks",
                check_xy_unit_range=False,
            )
        if self.pose_landmarks is not None:
            _validate_landmark_array(
                self.pose_landmarks,
                POSE_LANDMARK_COUNT,
                "pose_landmarks",
                check_xy_unit_range=False,
                components=4,
            )


_ALLOWED_FEATURE_TYPES = frozenset({"mfcc", "mel_spectrogram"})


@dataclass(frozen=True)
class AudioFeatureChunk:
    """A timestamped chunk of preprocessed audio features.

    Transmitted upstream from client to server. Never contains raw PCM audio.

    Args:
        session_id: UUID v4 of the active session.
        timestamp_ms: Milliseconds since session start.
        features: 2-D array of shape [T, D] — T temporal frames, D feature
            dimensions. T >= 1, D >= 1, must be rectangular.
        feature_type: "mfcc" or "mel_spectrogram".
        sample_rate_hz: Original audio sample rate in Hz (e.g. 16000).
        chunk_duration_ms: Duration of audio this chunk covers in ms (> 0).
    """

    session_id: str
    timestamp_ms: int
    features: list[list[float]]
    feature_type: Literal["mfcc", "mel_spectrogram"]
    sample_rate_hz: int
    chunk_duration_ms: int

    def __post_init__(self) -> None:
        _validate_non_empty_str(self.session_id, "session_id")
        _validate_non_negative_int(self.timestamp_ms, "timestamp_ms")
        _validate_rectangular_2d(self.features)
        if self.feature_type not in _ALLOWED_FEATURE_TYPES:
            raise ValueError(
                f"feature_type must be one of {_ALLOWED_FEATURE_TYPES}, "
                f"got '{self.feature_type}'"
            )
        if self.sample_rate_hz <= 0:
            raise ValueError(
                f"sample_rate_hz must be positive, got {self.sample_rate_hz}"
            )
        if self.chunk_duration_ms <= 0:
            raise ValueError(
                f"chunk_duration_ms must be > 0, got {self.chunk_duration_ms}"
            )


_VALID_ERROR_CODES = frozenset({4001, 4002, 4003, 4004})


@dataclass(frozen=True)
class SessionStatusMessage:
    """Outbound server→client session lifecycle notification.

    Sent by the gateway during connection setup and whenever the session
    state changes (e.g. INITIALIZING while models are loading, DEGRADED
    when a pipeline becomes unavailable). Always precedes or accompanies
    a TranscriptSegment stream — never replaces it.

    Wire format: JSON with ``type`` field set to ``"session_status"``.

    Serialisation note: ``state`` must be serialised as its string value
    (e.g. ``"INITIALIZING"``), not as the enum object.

    Args:
        session_id: UUID v4 of the active session.
        state: Current lifecycle state of the session.
        message: Human-readable detail for the client UI (e.g.
            "Loading speech models…"). May be empty.
    """

    session_id: str
    state: SessionState
    message: str

    def __post_init__(self) -> None:
        _validate_non_empty_str(self.session_id, "session_id")


@dataclass(frozen=True)
class ErrorMessage:
    """Outbound server→client error notification.

    Sent immediately before the server closes the WebSocket connection.
    The client should surface ``message`` to the user and treat the
    connection as terminated.

    Wire format: JSON with ``type`` field set to ``"error"``.

    Close code semantics:
        4001 — authentication failure (bad or missing api_key)
        4002 — malformed session_init payload
        4003 — duplicate session_id (session already active)
        4004 — model warm-up failed (server-side error)

    Args:
        session_id: UUID v4 of the session, or empty string if the
            connection failed before a session_id was established.
        code: WebSocket application close code in {4001, 4002, 4003, 4004}.
        message: Human-readable error description.
    """

    session_id: str
    code: int
    message: str

    def __post_init__(self) -> None:
        if self.code not in _VALID_ERROR_CODES:
            raise ValueError(
                f"code must be one of {sorted(_VALID_ERROR_CODES)}, got {self.code}"
            )
        _validate_non_empty_str(self.message, "message")


@dataclass(frozen=True)
class TranscriptSegment:
    """The primary output of the Sozia system.

    A time-aligned piece of transcript text with metadata. Produced by
    FusionOrchestrator, streamed to the client via WebSocket.

    If status is FINAL and replaces_segment_id is non-null, the client must
    locate the referenced PARTIAL segment and replace it. If the PARTIAL is not
    found (e.g. already expired), the FINAL segment is appended as a new entry.

    Args:
        segment_id: Globally unique UUID v4.
        session_id: UUID v4 of the active session.
        status: PARTIAL (tentative) or FINAL (will not be revised).
        text: Display-ready UTF-8 text. For PARTIAL sign segments this is raw
            gloss (e.g. "MERHABA DUNYA"); for FINAL it is natural Turkish.
        source: The primary modality that produced this text.
        confidence: Fused confidence score in [0.0, 1.0].
        timestamp_ms: Position in the subtitle timeline (ms since session start).
        duration_ms: How long this segment spans in ms.
        created_at_ms: Wall-clock Unix epoch milliseconds when the segment was
            created on the server.
        replaces_segment_id: If non-null, this segment revises the referenced
            PARTIAL. The client must replace the old segment with this one.
    """

    segment_id: str
    session_id: str
    status: SegmentStatus
    text: str
    source: ModalityType
    confidence: float
    timestamp_ms: int
    duration_ms: int
    created_at_ms: int
    replaces_segment_id: str | None

    def __post_init__(self) -> None:
        _validate_non_empty_str(self.segment_id, "segment_id")
        _validate_non_empty_str(self.session_id, "session_id")
        _validate_confidence(self.confidence)
        _validate_non_negative_int(self.timestamp_ms, "timestamp_ms")
        _validate_non_negative_int(self.duration_ms, "duration_ms")
        if self.replaces_segment_id is not None:
            _validate_non_empty_str(self.replaces_segment_id, "replaces_segment_id")
