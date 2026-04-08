"""Abstract interface contracts for the Sozia system.

These ABCs decouple consumers from concrete implementations. Defined here in
sozia.common so both the server and client share the same contract.

Dependency rule R3: this module must not import from sozia.server or sozia.client.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sozia.common.models import (
        ModalityResult,
        ModelConfig,
        PipelineHealth,
        TranscriptSegment,
    )


# ===========================================================================
# Custom exceptions
# ===========================================================================


class InferenceTimeoutError(Exception):
    """Raised when an inference engine exceeds its allotted timeout_ms budget."""


class ModelNotLoadedError(Exception):
    """Raised when predict() is called before load_model() has completed."""


# ===========================================================================
# Abstract Base Classes
# ===========================================================================


class InferenceEngine(abc.ABC):
    """Abstract contract for every AI inference component on the server.

    Enables model-agnostic design (Trade-off 5) and hot-swap via ModelRegistry.
    All four engine classes implement this interface:
    WhisperAsrEngine, LipReadingEngine, TslRecognitionEngine, GlossToTextEngine.
    """

    @abc.abstractmethod
    async def load_model(self, config: ModelConfig) -> None:
        """Load model weights into memory or GPU.

        Args:
            config: ModelConfig containing model_id, weights_path, device, and
                engine-specific params.

        Raises:
            ValueError: If config is invalid for this engine.
        """

    @abc.abstractmethod
    async def predict(self, features, timeout_ms: int = 2200) -> ModalityResult:
        """Run inference on preprocessed features.

        Args:
            features: Input features. Type varies by engine — numpy array for
                most engines, plain string for GlossToTextEngine.
            timeout_ms: Maximum wall-clock time allowed. Defaults to 2200 ms,
                matching the cloud inference budget.

        Returns:
            A ModalityResult with predicted text, confidence, and timing info.

        Raises:
            InferenceTimeoutError: Inference did not complete within timeout_ms.
            ModelNotLoadedError: No model is currently loaded.
        """

    @abc.abstractmethod
    async def unload_model(self) -> None:
        """Release model weights and free GPU memory."""

    @abc.abstractmethod
    def is_loaded(self) -> bool:
        """Return True if a model is currently loaded and ready for inference."""

    @abc.abstractmethod
    def get_model_id(self) -> str:
        """Return the identifier of the currently loaded model, or '' if none."""


class FusionStrategy(abc.ABC):
    """Abstract contract for fusion policies.

    Allows different fusion algorithms for the Speech path vs. the Sign path,
    and future experimentation with alternative strategies.
    Implemented by SpeechFusionPolicy and SignFusionPolicy in sozia.server.fusion.
    """

    @abc.abstractmethod
    def fuse(
        self,
        results: list[ModalityResult],
        health: list[PipelineHealth],
    ) -> list[TranscriptSegment]:
        """Accept inference results and pipeline health signals, return segments.

        Typically produces one PARTIAL segment immediately and one FINAL segment
        when the slower modality completes.

        Args:
            results: Inference results from one or more engines.
            health: Current pipeline health signals from the client.

        Returns:
            A list of TranscriptSegment objects to stream to the client.
        """

    @abc.abstractmethod
    def should_suppress(self, result: ModalityResult) -> bool:
        """Return True if the result's confidence is below the minimum threshold.

        Suppressed results are excluded from fusion and not shown to the user.

        Args:
            result: A ModalityResult to evaluate.
        """

    @abc.abstractmethod
    def get_confidence_threshold(self) -> float:
        """Return the current minimum confidence threshold (typically 0.30)."""


class FeatureExtractor(abc.ABC):
    """Abstract contract for client-side feature extraction.

    Ensures AudioFeatureExtractor and LandmarkExtractor share a uniform
    interface. Implemented by client pipeline packages (sozia.client.pipeline.*).
    Defined here so the server can reference the contract without importing
    client code.
    """

    @abc.abstractmethod
    def extract(self, raw_input) -> object | None:
        """Process a raw media buffer and return extracted features.

        Args:
            raw_input: Raw media handle (audio buffer or video frame).

        Returns:
            Extracted feature array or None if extraction fails (e.g. no face
            detected in the frame).
        """

    @abc.abstractmethod
    def is_ready(self) -> bool:
        """Return True if the extractor is initialised and ready to process input."""
