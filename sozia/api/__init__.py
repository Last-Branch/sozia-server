"""sozia.api — WebSocket gateway layer.

Public surface:
    AuthMiddleware    — API key authentication for WebSocket connections.
    SessionHandler    — Per-session WebSocket receive/send loop.
    WebSocketGateway  — Manages all active sessions.
"""

from sozia.api.auth_middleware import AuthMiddleware
from sozia.api.gateway import WebSocketGateway
from sozia.api.session_handler import SessionHandler

__all__ = ["AuthMiddleware", "SessionHandler", "WebSocketGateway"]
