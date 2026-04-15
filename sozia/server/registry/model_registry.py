"""ModelRegistry — singleton registry for InferenceEngine instances.

Dependency rules (non-negotiable):
- May import from sozia.common only.
- Engine classes must NOT be imported at module scope; they arrive at runtime
  via register_engine_class().

Rule R6: Inference engines are accessed only through ModelRegistry.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sozia.common.interfaces import InferenceEngine, ModelNotLoadedError

if TYPE_CHECKING:
    from sozia.common.models import ModelConfig


# ===========================================================================
# Custom exception
# ===========================================================================


class EngineClassNotRegisteredError(Exception):
    """Raised when load_model cannot find a registered prefix for a model_id."""


# ===========================================================================
# ModelRegistry
# ===========================================================================

_instance: ModelRegistry | None = None


class ModelRegistry:
    """Singleton registry that manages InferenceEngine lifecycle.

    Usage::

        registry = ModelRegistry.get_instance()
        registry.register_engine_class("whisper", WhisperAsrEngine)
        await registry.load_model(config)
        engine = registry.get_engine("whisper-small-tr")
    """

    def __init__(self) -> None:
        self.engines: dict[str, InferenceEngine] = {}
        self.configs: dict[str, ModelConfig] = {}
        self.engine_classes: dict[str, type[InferenceEngine]] = {}

    # ------------------------------------------------------------------
    # Singleton access
    # ------------------------------------------------------------------

    @classmethod
    def get_instance(cls) -> ModelRegistry:
        """Return the process-wide singleton, creating it on first call."""
        global _instance
        if _instance is None:
            _instance = cls()
        return _instance

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_engine_class(self, prefix: str, cls: type[InferenceEngine]) -> None:
        """Register a model_id prefix → engine class mapping.

        Args:
            prefix: The string prefix that identifies the engine family
                (e.g. ``"whisper"``).
            cls: The concrete InferenceEngine subclass to instantiate when
                a model_id starts with this prefix.
        """
        self.engine_classes[prefix] = cls

    # ------------------------------------------------------------------
    # Load / unload
    # ------------------------------------------------------------------

    async def load_model(self, config: ModelConfig) -> None:
        """Instantiate and load an engine for the given config.

        Resolves the engine class by finding the longest registered prefix
        that ``config.model_id`` starts with.  Idempotent: if the model_id
        is already loaded, this is a no-op.

        Args:
            config: ModelConfig whose ``model_id`` is used for prefix lookup.

        Raises:
            EngineClassNotRegisteredError: No registered prefix matches
                ``config.model_id``.
        """
        if config.model_id in self.engines:
            return

        cls = self._resolve_class(config.model_id)
        engine: InferenceEngine = cls()
        await engine.load_model(config)
        self.engines[config.model_id] = engine
        self.configs[config.model_id] = config

    async def unload_model(self, model_id: str) -> None:
        """Unload the engine for ``model_id`` and release resources.

        Args:
            model_id: Identifier of the model to unload.

        Raises:
            ModelNotLoadedError: ``model_id`` is not currently loaded.
        """
        if model_id not in self.engines:
            raise ModelNotLoadedError(f"Model '{model_id}' is not loaded.")
        await self.engines[model_id].unload_model()
        del self.engines[model_id]
        del self.configs[model_id]

    async def unload_all(self) -> None:
        """Unload every engine; preserve ``engine_classes`` for re-use."""
        for model_id in list(self.engines):
            await self.unload_model(model_id)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_engine(self, model_id: str) -> InferenceEngine:
        """Return the loaded engine for ``model_id``.

        Args:
            model_id: Identifier of the requested engine.

        Raises:
            ModelNotLoadedError: ``model_id`` is not currently loaded.
        """
        if model_id not in self.engines:
            raise ModelNotLoadedError(f"Model '{model_id}' is not loaded.")
        return self.engines[model_id]

    def is_loaded(self, model_id: str) -> bool:
        """Return True if ``model_id`` is currently in the engines dict."""
        return model_id in self.engines

    def list_loaded(self) -> list[str]:
        """Return all currently loaded model ids."""
        return list(self.engines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_class(self, model_id: str) -> type[InferenceEngine]:
        """Pick the longest registered prefix matching ``model_id``.

        Args:
            model_id: The model identifier to look up.

        Raises:
            EngineClassNotRegisteredError: No prefix matches ``model_id``.
        """
        matching = [p for p in self.engine_classes if model_id.startswith(p)]
        if not matching:
            raise EngineClassNotRegisteredError(
                f"No engine class registered for model_id '{model_id}'. "
                f"Registered prefixes: {list(self.engine_classes)}"
            )
        best = max(matching, key=len)
        return self.engine_classes[best]


# ===========================================================================
# Module-level singleton reset (for test teardown)
# ===========================================================================


def reset_instance() -> None:
    """Clear the process-wide singleton.  Call in test teardown only."""
    global _instance
    _instance = None
