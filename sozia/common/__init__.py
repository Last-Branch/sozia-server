"""sozia.common — shared types and interface contracts.

This package is the dependency leaf of the Sozia system (rule R3): it imports
from neither sozia.server nor sozia.client. Every other package may import
from here.
"""

from sozia.common.interfaces import (
    FeatureExtractor,
    FusionStrategy,
    InferenceEngine,
    InferenceTimeoutError,
    ModelNotLoadedError,
)
from sozia.common.models import (
    FACE_LANDMARK_COUNT,
    HAND_LANDMARK_COUNT,
    POSE_LANDMARK_COUNT,
    AudioFeatureChunk,
    ErrorMessage,
    LandmarkFrame,
    ModalityPath,
    ModalityResult,
    ModalityType,
    ModelConfig,
    PipelineHealth,
    SegmentStatus,
    SessionState,
    SessionStatusMessage,
    TranscriptSegment,
)

__all__ = [
    # Enums
    "SessionState",
    "ModalityPath",
    "ModalityType",
    "SegmentStatus",
    # DTOs
    "ModelConfig",
    "ModalityResult",
    "PipelineHealth",
    "LandmarkFrame",
    "AudioFeatureChunk",
    "TranscriptSegment",
    "SessionStatusMessage",
    "ErrorMessage",
    # Constants
    "FACE_LANDMARK_COUNT",
    "HAND_LANDMARK_COUNT",
    "POSE_LANDMARK_COUNT",
    # ABCs
    "InferenceEngine",
    "FusionStrategy",
    "FeatureExtractor",
    # Exceptions
    "InferenceTimeoutError",
    "ModelNotLoadedError",
]
