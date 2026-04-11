"""WebSocketGateway — entry point for all WebSocket connections.

One gateway instance runs for the life of the FastAPI application. It
accepts connections, authenticates them, negotiates ``session_init``, and
delegates message handling to per-session ``SessionHandler`` instances.

Close code semantics (per LLD):
    4001 — authentication failure (bad or missing api_key)
    4002 — malformed session_init payload
    4003 — duplicate session_id (session already active)
    4004 — model warm-up failed (server-side error)
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
from typing import TYPE_CHECKING, Any, Callable

from starlette.websockets import WebSocketDisconnect

from sozia.api.auth_middleware import AuthMiddleware
from sozia.api.session_handler import SessionHandler
from sozia.common.models import (
    ModalityPath,
    SessionState,
    SessionStatusMessage,
)

if TYPE_CHECKING:
    from fastapi import WebSocket

    from sozia.fusion.orchestrator import FusionOrchestrator

logger = logging.getLogger(__name__)


class WebSocketGateway:
    """Manages all active WebSocket sessions.

    Args:
        auth: The AuthMiddleware used to validate API keys.
        orchestrator_factory: Callable that returns a new FusionOrchestrator
            for each accepted session.  Receives ``(session_id, modality_path)``
            as arguments.
    """

    def __init__(
        self,
        auth: AuthMiddleware,
        orchestrator_factory: Callable[..., FusionOrchestrator],
    ) -> None:
        self._auth = auth
        self._orchestrator_factory = orchestrator_factory
        self.active_sessions: dict[str, SessionHandler] = {}
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def on_connect(self, websocket: WebSocket) -> None:
        """Handle the full lifecycle of a single WebSocket connection.

        Steps:
        1. Accept the connection.
        2. Read the first message and expect ``session_init``.
        3. Authenticate via api_key.
        4. Validate the session_init payload.
        5. Reject duplicate sessions.
        6. Create orchestrator and warm up.
        7. Run the receive loop until disconnect.
        8. Clean up on exit.

        Close codes: 4001 auth, 4002 bad init, 4003 duplicate, 4004 warm-up.
        """
        await websocket.accept()

        # Step 2 — read session_init (5-second deadline to prevent idle DoS).
        try:
            init_msg = await asyncio.wait_for(websocket.receive_json(), timeout=5.0)
        except asyncio.TimeoutError:
            await self._error_close(websocket, "", 4002)
            return
        except WebSocketDisconnect:
            return

        # Step 3 — authenticate.
        provided_key = init_msg.get("api_key", "")
        if not self._auth.authenticate(provided_key):
            await self._error_close(websocket, init_msg.get("session_id", ""), 4001)
            return

        # Step 4 — parse session_init.
        session_id, modality_path = self._parse_session_init(init_msg)
        if session_id is None or modality_path is None:
            await self._error_close(websocket, init_msg.get("session_id", ""), 4002)
            return

        # Step 5 — reject duplicates.
        async with self._lock:
            if session_id in self.active_sessions:
                await self._error_close(websocket, session_id, 4003)
                return

            # Reserve the slot immediately so a concurrent connection with
            # the same session_id cannot race past the duplicate check.
            self.active_sessions[session_id] = None  # type: ignore[assignment]

        # Step 6 — create orchestrator and warm up.
        orchestrator = self._orchestrator_factory(session_id, modality_path)

        # Send INITIALIZING before warm-up so the client knows we're working.
        await self._send_status(websocket, session_id, SessionState.INITIALIZING)

        try:
            await orchestrator.warm_up(modality_path)
        except Exception as exc:
            logger.exception("Warm-up failed for session %s: %s", session_id, exc)
            async with self._lock:
                self.active_sessions.pop(session_id, None)
            await self._error_close(websocket, session_id, 4004)
            return

        # Step 7 — run the session.
        handler = SessionHandler(
            session_id=session_id,
            modality_path=modality_path,
            websocket=websocket,
            orchestrator=orchestrator,
        )
        async with self._lock:
            self.active_sessions[session_id] = handler

        await self._send_status(websocket, session_id, SessionState.RUNNING)

        try:
            await handler.receive_loop()
        finally:
            # Step 8 — clean up regardless of how the loop ended.
            await self.on_disconnect(session_id)

    async def shutdown_all_sessions(self) -> None:
        """Close every active session during server shutdown.

        Drains ``active_sessions`` and calls ``close()`` on each handler so
        that per-session orchestrators release GPU memory before
        ``ModelRegistry.unload_all()`` is called.
        """
        async with self._lock:
            handlers = list(self.active_sessions.values())
            self.active_sessions.clear()

        for handler in handlers:
            if handler is not None:
                try:
                    await handler.close()
                except Exception:
                    logger.exception("close failed during server shutdown")

    async def on_disconnect(self, session_id: str) -> None:
        """Tear down the session identified by ``session_id``.

        Removes it from ``active_sessions`` and calls ``cool_down()`` on
        the orchestrator.  Safe to call even if the session is unknown.

        Args:
            session_id: The session to remove.
        """
        async with self._lock:
            handler = self.active_sessions.pop(session_id, None)

        if handler is not None:
            try:
                await handler.close()
            except Exception:
                logger.exception("cool_down failed for session %s", session_id)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_session_init(
        msg: dict[str, Any],
    ) -> tuple[str | None, ModalityPath | None]:
        """Extract and validate session_id + modality_path from session_init.

        Returns ``(None, None)`` if the message is invalid.
        """
        if msg.get("type") != "session_init":
            return None, None

        session_id = msg.get("session_id")
        if not session_id or not isinstance(session_id, str):
            return None, None

        raw_path = msg.get("modality_path")
        try:
            modality_path = ModalityPath(raw_path)
        except (ValueError, KeyError):
            return None, None

        return session_id, modality_path

    @staticmethod
    async def _send_status(
        websocket: WebSocket,
        session_id: str,
        state: SessionState,
        message: str = "",
    ) -> None:
        """Send a SessionStatusMessage JSON frame to the client.

        Args:
            websocket: The active WebSocket connection.
            session_id: Current session identifier.
            state: The lifecycle state to report.
            message: Optional human-readable detail for the client UI.
        """
        status = SessionStatusMessage(
            session_id=session_id,
            state=state,
            message=message,
        )
        payload = dataclasses.asdict(status)
        payload["type"] = "session_status"
        payload["state"] = state.value
        await websocket.send_text(json.dumps(payload))

    @staticmethod
    async def _error_close(
        websocket: WebSocket,
        session_id: str,
        code: int,
    ) -> None:
        """Send an error payload then close the WebSocket with ``code``.

        Args:
            websocket: The WebSocket to close.
            session_id: Session identifier (may be empty if init failed early).
            code: Application-level close code (4001–4004).
        """
        _MESSAGES = {
            4001: "Authentication failed — invalid or missing api_key.",
            4002: "Malformed session_init payload.",
            4003: "Session already active for this session_id.",
            4004: "Model warm-up failed — server error.",
        }
        error_payload = json.dumps(
            {
                "type": "error",
                "session_id": session_id,
                "code": code,
                "message": _MESSAGES.get(code, "Unknown error."),
            }
        )
        try:
            await websocket.send_text(error_payload)
        except Exception:
            pass
        await websocket.close(code=code)
