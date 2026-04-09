"""sozia.registry — ModelRegistry and related exceptions.

Dependency rule R6: inference engines are accessed only through ModelRegistry,
never instantiated directly by gateway or orchestrator.
"""

from sozia.registry.model_registry import EngineClassNotRegisteredError, ModelRegistry

# reset_instance is intentionally omitted from __all__ — it is a test-only
# helper and should be imported directly from sozia.registry.model_registry.

__all__ = [
    "ModelRegistry",
    "EngineClassNotRegisteredError",
]
