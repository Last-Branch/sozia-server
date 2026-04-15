"""Smoke tests for the sozia.common package surface.

Verifies __all__ completeness and enforces dependency rule R3.
"""

from __future__ import annotations

import importlib
import sozia.common as common


class TestPublicSurface:
    def test_all_symbols_importable(self):
        for name in common.__all__:
            assert hasattr(common, name), f"{name} in __all__ but not importable"

    def test_r3_no_server_imports(self):
        """sozia.common must not import from sozia.server."""
        import sys

        evicted = {
            k: sys.modules.pop(k)
            for k in list(sys.modules)
            if k.startswith("sozia.common")
        }
        before = set(sys.modules.keys())
        try:
            importlib.import_module("sozia.common")
            added = set(sys.modules.keys()) - before
        finally:
            sys.modules.update(evicted)

        for mod_name in added:
            if mod_name.startswith("sozia.server"):
                raise AssertionError(
                    f"sozia.common transitively imports sozia.server ({mod_name}) — R3 violation"
                )

    def test_r3_no_client_imports(self):
        """sozia.common must not import from sozia.client."""
        import sys

        for mod_name in sys.modules:
            if mod_name.startswith("sozia.client"):
                raise AssertionError(
                    f"sozia.common transitively imports sozia.client ({mod_name}) — R3 violation"
                )
