"""AuthMiddleware — API key authentication for WebSocket connections.

The client sends the API key inside the ``session_init`` JSON payload because
browsers cannot set custom HTTP headers on WebSocket upgrade requests.

Timing-safe comparison is done via ``hmac.compare_digest`` to prevent
timing-based side-channel attacks.
"""

from __future__ import annotations

import hmac
import os


class AuthMiddleware:
    """Validates per-connection API keys.

    Args:
        api_key: The expected API key.  Must be non-empty.

    Raises:
        ValueError: If ``api_key`` is empty.
    """

    def __init__(self, api_key: str) -> None:
        if not api_key:
            raise ValueError("api_key must be a non-empty string")
        self._api_key = api_key

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls, env_var: str = "SOZIA_API_KEY") -> AuthMiddleware:
        """Construct from an environment variable.

        Args:
            env_var: Name of the environment variable holding the key.
                Defaults to ``"SOZIA_API_KEY"``.

        Raises:
            EnvironmentError: The environment variable is not set.
        """
        key = os.environ.get(env_var)
        if not key:
            raise EnvironmentError(
                f"{env_var} environment variable is not set. "
                "Set it to a strong random string before starting the server."
            )
        return cls(api_key=key)

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------

    def authenticate(self, provided_key: str) -> bool:
        """Return True if ``provided_key`` matches the configured API key.

        Uses ``hmac.compare_digest`` to prevent timing-based attacks.
        Returns False for any empty or mismatched input without raising.

        Args:
            provided_key: The key extracted from the ``session_init`` payload.
        """
        if not provided_key:
            return False
        return hmac.compare_digest(
            self._api_key.encode(),
            provided_key.encode(),
        )
