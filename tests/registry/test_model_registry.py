"""Unit tests for sozia.registry.ModelRegistry.

TDD phase: RED — all tests must fail before the implementation exists.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from sozia.common import InferenceEngine, ModelConfig, ModelNotLoadedError
from sozia.common.models import ModalityResult, ModalityType
from sozia.registry import EngineClassNotRegisteredError, ModelRegistry
from sozia.registry.model_registry import reset_instance


# ===========================================================================
# Shared test helpers
# ===========================================================================


def _make_config(model_id: str) -> ModelConfig:  # noqa: D103
    return ModelConfig(
        model_id=model_id,
        weights_path="/tmp/fake.pt",
        device="cpu",
        params={},
    )


class StubEngine(InferenceEngine):
    """Minimal no-op InferenceEngine that records call counts."""

    def __init__(self) -> None:
        self.load_model_call_count = 0
        self.unload_model_call_count = 0
        self._loaded = False
        self._model_id = ""

    async def load_model(self, config: ModelConfig) -> None:
        self.load_model_call_count += 1
        self._loaded = True
        self._model_id = config.model_id

    async def predict(
        self, features: np.ndarray, timeout_ms: int = 2200
    ) -> ModalityResult:
        return ModalityResult(
            modality_type=ModalityType.ASR,
            text="",
            confidence=0.0,
            timestamp_ms=0,
            duration_ms=0,
            inference_latency_ms=0,
        )

    async def unload_model(self) -> None:
        self.unload_model_call_count += 1
        self._loaded = False
        self._model_id = ""

    def is_loaded(self) -> bool:
        return self._loaded

    def get_model_id(self) -> str:
        return self._model_id


class AnotherStubEngine(InferenceEngine):
    """Second stub for prefix-collision tests."""

    def __init__(self) -> None:
        self.load_model_call_count = 0

    async def load_model(self, config: ModelConfig) -> None:
        self.load_model_call_count += 1

    async def predict(
        self, features: np.ndarray, timeout_ms: int = 2200
    ) -> ModalityResult:
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


# ===========================================================================
# Fixture: fresh registry per test
# ===========================================================================


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Ensure singleton state is clean before and after every test."""
    reset_instance()
    yield
    reset_instance()


# ===========================================================================
# Singleton behaviour
# ===========================================================================


class TestSingleton:
    def test_get_instance_returns_same_object(self):
        a = ModelRegistry.get_instance()
        b = ModelRegistry.get_instance()
        assert a is b

    def test_reset_instance_clears_singleton(self):
        first = ModelRegistry.get_instance()
        reset_instance()
        second = ModelRegistry.get_instance()
        assert first is not second

    def test_direct_instantiation_isolated_from_singleton(self):
        singleton = ModelRegistry.get_instance()
        direct = ModelRegistry()
        assert singleton is not direct


# ===========================================================================
# register_engine_class
# ===========================================================================


class TestRegisterEngineClass:
    def test_stores_prefix_to_class_mapping(self):
        registry = ModelRegistry()
        registry.register_engine_class("whisper", StubEngine)
        assert registry.engine_classes["whisper"] is StubEngine

    def test_overwriting_prefix_replaces_class(self):
        registry = ModelRegistry()
        registry.register_engine_class("whisper", StubEngine)
        registry.register_engine_class("whisper", AnotherStubEngine)
        assert registry.engine_classes["whisper"] is AnotherStubEngine


# ===========================================================================
# load_model
# ===========================================================================


class TestLoadModel:
    async def test_instantiates_matching_prefix_class(self):
        registry = ModelRegistry()
        registry.register_engine_class("whisper", StubEngine)
        config = _make_config("whisper-small-tr")
        await registry.load_model(config)
        engine = registry.engines["whisper-small-tr"]
        assert isinstance(engine, StubEngine)

    async def test_calls_engine_load_model_exactly_once(self):
        registry = ModelRegistry()
        registry.register_engine_class("whisper", StubEngine)
        config = _make_config("whisper-small-tr")
        await registry.load_model(config)
        engine = registry.engines["whisper-small-tr"]
        assert engine.load_model_call_count == 1

    async def test_stores_in_engines_and_configs(self):
        registry = ModelRegistry()
        registry.register_engine_class("whisper", StubEngine)
        config = _make_config("whisper-small-tr")
        await registry.load_model(config)
        assert "whisper-small-tr" in registry.engines
        assert registry.configs["whisper-small-tr"] is config

    async def test_longest_prefix_wins(self):
        """'whisper-small' should beat 'whisper' for id 'whisper-small-tr'."""
        registry = ModelRegistry()
        registry.register_engine_class("whisper", StubEngine)
        registry.register_engine_class("whisper-small", AnotherStubEngine)
        config = _make_config("whisper-small-tr")
        await registry.load_model(config)
        engine = registry.engines["whisper-small-tr"]
        assert isinstance(engine, AnotherStubEngine)

    async def test_raises_engine_class_not_registered_when_no_prefix_matches(self):
        registry = ModelRegistry()
        config = _make_config("unknown-model")
        with pytest.raises(EngineClassNotRegisteredError):
            await registry.load_model(config)

    async def test_idempotent_second_call_is_noop(self):
        """Calling load_model twice with the same model_id must not reload."""
        registry = ModelRegistry()
        registry.register_engine_class("whisper", StubEngine)
        config = _make_config("whisper-small-tr")
        await registry.load_model(config)
        engine_first = registry.engines["whisper-small-tr"]
        await registry.load_model(config)
        engine_second = registry.engines["whisper-small-tr"]
        assert engine_first is engine_second
        assert engine_first.load_model_call_count == 1


# ===========================================================================
# get_engine
# ===========================================================================


class TestGetEngine:
    async def test_returns_stored_engine(self):
        registry = ModelRegistry()
        registry.register_engine_class("whisper", StubEngine)
        config = _make_config("whisper-small-tr")
        await registry.load_model(config)
        engine = registry.get_engine("whisper-small-tr")
        assert isinstance(engine, StubEngine)

    def test_raises_model_not_loaded_error_for_unknown_id(self):
        registry = ModelRegistry()
        with pytest.raises(ModelNotLoadedError):
            registry.get_engine("does-not-exist")


# ===========================================================================
# unload_model
# ===========================================================================


class TestUnloadModel:
    async def test_calls_engine_unload_once_and_removes_from_engines_and_configs(self):
        registry = ModelRegistry()
        registry.register_engine_class("whisper", StubEngine)
        config = _make_config("whisper-small-tr")
        await registry.load_model(config)
        engine = registry.engines["whisper-small-tr"]
        await registry.unload_model("whisper-small-tr")
        assert engine.unload_model_call_count == 1
        assert "whisper-small-tr" not in registry.engines
        assert "whisper-small-tr" not in registry.configs

    async def test_raises_model_not_loaded_error_for_unknown_id(self):
        registry = ModelRegistry()
        with pytest.raises(ModelNotLoadedError):
            await registry.unload_model("does-not-exist")


# ===========================================================================
# unload_all
# ===========================================================================


class TestUnloadAll:
    async def test_engines_empty_after_unload_all(self):
        registry = ModelRegistry()
        registry.register_engine_class("whisper", StubEngine)
        registry.register_engine_class("gemma", StubEngine)
        await registry.load_model(_make_config("whisper-small-tr"))
        await registry.load_model(_make_config("gemma-9b-gloss-tr"))
        await registry.unload_all()
        assert registry.engines == {}

    async def test_engine_classes_preserved_after_unload_all(self):
        registry = ModelRegistry()
        registry.register_engine_class("whisper", StubEngine)
        await registry.load_model(_make_config("whisper-small-tr"))
        await registry.unload_all()
        assert "whisper" in registry.engine_classes

    async def test_each_engine_unload_called_once(self):
        registry = ModelRegistry()
        registry.register_engine_class("whisper", StubEngine)
        registry.register_engine_class("gemma", StubEngine)
        await registry.load_model(_make_config("whisper-small-tr"))
        await registry.load_model(_make_config("gemma-9b-gloss-tr"))
        engine_a = registry.engines["whisper-small-tr"]
        engine_b = registry.engines["gemma-9b-gloss-tr"]
        await registry.unload_all()
        assert engine_a.unload_model_call_count == 1
        assert engine_b.unload_model_call_count == 1


# ===========================================================================
# is_loaded / list_loaded
# ===========================================================================


class TestIsLoadedAndListLoaded:
    async def test_is_loaded_true_for_loaded_id(self):
        registry = ModelRegistry()
        registry.register_engine_class("whisper", StubEngine)
        await registry.load_model(_make_config("whisper-small-tr"))
        assert registry.is_loaded("whisper-small-tr") is True

    def test_is_loaded_false_for_unknown_id(self):
        registry = ModelRegistry()
        assert registry.is_loaded("does-not-exist") is False

    async def test_list_loaded_returns_all_loaded_ids(self):
        registry = ModelRegistry()
        registry.register_engine_class("whisper", StubEngine)
        registry.register_engine_class("gemma", StubEngine)
        await registry.load_model(_make_config("whisper-small-tr"))
        await registry.load_model(_make_config("gemma-9b-gloss-tr"))
        loaded = registry.list_loaded()
        assert set(loaded) == {"whisper-small-tr", "gemma-9b-gloss-tr"}


# ===========================================================================
# Dependency rule R2 smoke test
# ===========================================================================


class TestDependencyRules:
    def test_importing_registry_does_not_pull_inference_or_client(self):
        """Rule R2: sozia.registry must not import sozia.inference.* or sozia.client.*."""
        import importlib

        # Evict sozia.registry so the import below is a genuine cold load.
        # Without this, the test only checks whatever is already in sys.modules
        # (which includes sozia.inference.* after inference tests run first).
        evicted = {
            k: sys.modules.pop(k)
            for k in list(sys.modules)
            if k.startswith("sozia.registry")
        }
        before = set(sys.modules.keys())
        try:
            importlib.import_module("sozia.registry")
            added = set(sys.modules.keys()) - before
        finally:
            sys.modules.update(evicted)

        forbidden_prefixes = ("sozia.inference", "sozia.client")
        for mod_name in added:
            for prefix in forbidden_prefixes:
                assert not mod_name.startswith(prefix), (
                    f"Forbidden module '{mod_name}' was pulled in by sozia.registry"
                )
