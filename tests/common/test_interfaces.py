"""Unit tests for sozia.common.interfaces — ABCs and custom exceptions."""

from __future__ import annotations

import inspect

import pytest

from sozia.common.interfaces import (
    FeatureExtractor,
    FusionStrategy,
    InferenceEngine,
    InferenceTimeoutError,
    ModelNotLoadedError,
)
from sozia.common.models import (
    ModalityResult,
    ModalityType,
    ModelConfig,
    PipelineHealth,
    TranscriptSegment,
)


# ===========================================================================
# Custom exceptions
# ===========================================================================


class TestInferenceTimeoutError:
    def test_is_exception_subclass(self):
        assert issubclass(InferenceTimeoutError, Exception)

    def test_message_preserved(self):
        err = InferenceTimeoutError("timed out after 2200ms")
        assert "2200" in str(err)

    def test_can_be_raised_and_caught(self):
        with pytest.raises(InferenceTimeoutError):
            raise InferenceTimeoutError("timeout")


class TestModelNotLoadedError:
    def test_is_exception_subclass(self):
        assert issubclass(ModelNotLoadedError, Exception)

    def test_message_preserved(self):
        err = ModelNotLoadedError("no model loaded")
        assert "no model" in str(err)

    def test_can_be_raised_and_caught(self):
        with pytest.raises(ModelNotLoadedError):
            raise ModelNotLoadedError("no model")


# ===========================================================================
# InferenceEngine ABC
# ===========================================================================


class TestInferenceEngineABC:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            InferenceEngine()  # type: ignore[abstract]

    def test_minimal_subclass_can_be_instantiated(self):
        class FakeEngine(InferenceEngine):
            async def load_model(self, config: ModelConfig) -> None:
                pass

            async def predict(self, features, timeout_ms: int = 2200) -> ModalityResult:
                return ModalityResult(
                    modality_type=ModalityType.ASR,
                    text="",
                    confidence=0.0,
                    timestamp_ms=0,
                    duration_ms=0,
                    inference_latency_ms=0,
                )

            async def unload_model(self) -> None:
                pass

            def is_loaded(self) -> bool:
                return False

            def get_model_id(self) -> str:
                return ""

        engine = FakeEngine()
        assert isinstance(engine, InferenceEngine)

    def test_subclass_missing_method_cannot_instantiate(self):
        class IncompleteEngine(InferenceEngine):
            async def load_model(self, config: ModelConfig) -> None:
                pass

            # predict, unload_model, is_loaded, get_model_id missing

        with pytest.raises(TypeError):
            IncompleteEngine()  # type: ignore[abstract]

    def test_async_methods(self):
        assert inspect.iscoroutinefunction(InferenceEngine.load_model)
        assert inspect.iscoroutinefunction(InferenceEngine.predict)
        assert inspect.iscoroutinefunction(InferenceEngine.unload_model)

    def test_sync_methods(self):
        assert not inspect.iscoroutinefunction(InferenceEngine.is_loaded)
        assert not inspect.iscoroutinefunction(InferenceEngine.get_model_id)


# ===========================================================================
# FusionStrategy ABC
# ===========================================================================


class TestFusionStrategyABC:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            FusionStrategy()  # type: ignore[abstract]

    def test_minimal_subclass_can_be_instantiated(self):
        class FakePolicy(FusionStrategy):
            def fuse(
                self,
                results: list[ModalityResult],
                health: list[PipelineHealth],
            ) -> list[TranscriptSegment]:
                return []

            def should_suppress(self, result: ModalityResult) -> bool:
                return result.confidence < 0.30

            def get_confidence_threshold(self) -> float:
                return 0.30

        policy = FakePolicy()
        assert isinstance(policy, FusionStrategy)

    def test_subclass_missing_method_cannot_instantiate(self):
        class IncompletePolicy(FusionStrategy):
            def fuse(self, results, health):
                return []

            # should_suppress and get_confidence_threshold missing

        with pytest.raises(TypeError):
            IncompletePolicy()  # type: ignore[abstract]

    def test_all_methods_are_sync(self):
        assert not inspect.iscoroutinefunction(FusionStrategy.fuse)
        assert not inspect.iscoroutinefunction(FusionStrategy.should_suppress)
        assert not inspect.iscoroutinefunction(FusionStrategy.get_confidence_threshold)


# ===========================================================================
# FeatureExtractor ABC
# ===========================================================================


class TestFeatureExtractorABC:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            FeatureExtractor()  # type: ignore[abstract]

    def test_minimal_subclass_can_be_instantiated(self):
        class FakeExtractor(FeatureExtractor):
            def extract(self, raw_input) -> object | None:
                return None

            def is_ready(self) -> bool:
                return True

        extractor = FakeExtractor()
        assert isinstance(extractor, FeatureExtractor)

    def test_subclass_missing_method_cannot_instantiate(self):
        class IncompleteExtractor(FeatureExtractor):
            def extract(self, raw_input):
                return None

            # is_ready missing

        with pytest.raises(TypeError):
            IncompleteExtractor()  # type: ignore[abstract]

    def test_all_methods_are_sync(self):
        assert not inspect.iscoroutinefunction(FeatureExtractor.extract)
        assert not inspect.iscoroutinefunction(FeatureExtractor.is_ready)
