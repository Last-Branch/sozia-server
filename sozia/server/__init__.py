"""Sozia server entry point.

Builds the FastAPI application, registers the ``/ws`` WebSocket endpoint,
and wires up the startup/shutdown lifecycle:

  startup  — create auth + gateway; register all four engine classes on the
             ModelRegistry singleton so it can resolve prefixes on warm_up.
  shutdown — call ``registry.unload_all()`` to release GPU memory cleanly.

Configuration is driven entirely by environment variables (see README).
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, WebSocket

from sozia.server.api.auth_middleware import AuthMiddleware
from sozia.server.api.gateway import WebSocketGateway
from sozia.common.models import ModalityPath, ModalityType, ModelConfig
from sozia.server.fusion.degraded_mode_handler import DegradedModeHandler
from sozia.server.fusion.orchestrator import FusionOrchestrator
from sozia.server.fusion.sign_fusion_policy import SignFusionPolicy
from sozia.server.fusion.speech_fusion_policy import SpeechFusionPolicy
from sozia.server.inference.sign.gloss_to_text_engine import GlossToTextEngine
from sozia.server.inference.sign.tsl_recognition_engine import TslRecognitionEngine
from sozia.server.inference.speech.lip_reading_engine import LipReadingEngine
from sozia.server.inference.speech.whisper_asr_engine import WhisperAsrEngine
from sozia.server.registry.model_registry import ModelRegistry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment variable names (public so tests can monkeypatch by name)
# ---------------------------------------------------------------------------

ENV_API_KEY = "SOZIA_API_KEY"
ENV_DEVICE = "SOZIA_DEVICE"
ENV_WHISPER_WEIGHTS = "SOZIA_WHISPER_WEIGHTS"
ENV_LIP_WEIGHTS = "SOZIA_LIP_WEIGHTS"
ENV_LIP_VOCAB = "SOZIA_LIP_VOCAB"
ENV_TSL_WEIGHTS = "SOZIA_TSL_WEIGHTS"
ENV_TSL_SCALER = "SOZIA_TSL_SCALER"
ENV_GLOSS_WEIGHTS = "SOZIA_GLOSS_WEIGHTS"

_DEFAULT_DEVICE = "cpu"


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _device() -> str:
    return os.environ.get(ENV_DEVICE, _DEFAULT_DEVICE)


def _weights(env_var: str) -> str | None:
    return os.environ.get(env_var) or None


def _speech_configs(device: str) -> tuple[ModelConfig | None, ModelConfig | None]:
    """Return (asr_config, lip_config); None where weights are not configured."""
    asr_path = _weights(ENV_WHISPER_WEIGHTS)
    lip_path = _weights(ENV_LIP_WEIGHTS)
    lip_vocab = _weights(ENV_LIP_VOCAB)

    asr_cfg = (
        ModelConfig(
            model_id="whisper-small-tr",
            weights_path=asr_path,
            device=device,
            params={},
        )
        if asr_path
        else None
    )
    lip_params: dict = {}
    if lip_vocab:
        lip_params["vocab_path"] = lip_vocab
    lip_cfg = (
        ModelConfig(
            model_id="lip-reading-v1",
            weights_path=lip_path,
            device=device,
            params=lip_params,
        )
        if lip_path
        else None
    )
    return asr_cfg, lip_cfg


def _sign_configs(device: str) -> tuple[ModelConfig | None, ModelConfig | None]:
    """Return (tsl_config, gloss_config); None where weights are not configured."""
    tsl_path = _weights(ENV_TSL_WEIGHTS)
    gloss_path = _weights(ENV_GLOSS_WEIGHTS)
    tsl_scaler = _weights(ENV_TSL_SCALER)

    tsl_params: dict = {}
    if tsl_scaler:
        tsl_params["scaler_path"] = tsl_scaler

    tsl_cfg = (
        ModelConfig(
            model_id="tsl-gru-v1",
            weights_path=tsl_path,
            device=device,
            params=tsl_params,
        )
        if tsl_path
        else None
    )
    gloss_cfg = (
        ModelConfig(
            model_id="gemma-9b-gloss-tr",
            weights_path=gloss_path,
            device=device,
            params={},
        )
        if gloss_path
        else None
    )
    return tsl_cfg, gloss_cfg


# ---------------------------------------------------------------------------
# Orchestrator factory
# ---------------------------------------------------------------------------


def _make_orchestrator_factory(device: str):
    """Return a callable ``(session_id, modality_path) -> FusionOrchestrator``.

    Each call produces a new orchestrator pre-wired with policies and engine
    instances for the requested path.  The engines are instantiated here but
    not loaded; ``warm_up()`` triggers the actual weight loading.

    Note: engine instances are created fresh per-session because the GPU
    memory is owned by each engine object.  The ModelRegistry is used for
    prefix resolution (Rule R6 — engine classes registered at startup).

    Args:
        device: Compute device string (``"cpu"`` or ``"cuda"``).
    """

    def factory(session_id: str, modality_path: ModalityPath) -> FusionOrchestrator:
        orchestrator = FusionOrchestrator(degraded_handler=DegradedModeHandler())
        orchestrator.register_policy(ModalityPath.SPEECH, SpeechFusionPolicy())
        orchestrator.register_policy(ModalityPath.SIGN, SignFusionPolicy())

        if modality_path == ModalityPath.SPEECH:
            asr_cfg, lip_cfg = _speech_configs(device)
            if asr_cfg:
                orchestrator.register_engine(
                    ModalityPath.SPEECH,
                    ModalityType.ASR,
                    WhisperAsrEngine(),
                    asr_cfg,
                )
            if lip_cfg:
                orchestrator.register_engine(
                    ModalityPath.SPEECH,
                    ModalityType.LIP_READING,
                    LipReadingEngine(),
                    lip_cfg,
                )
        else:
            tsl_cfg, gloss_cfg = _sign_configs(device)
            if tsl_cfg:
                orchestrator.register_engine(
                    ModalityPath.SIGN,
                    ModalityType.TSL_RECOGNITION,
                    TslRecognitionEngine(),
                    tsl_cfg,
                )
            if gloss_cfg:
                orchestrator.register_engine(
                    ModalityPath.SIGN,
                    ModalityType.GLOSS_TO_TEXT,
                    GlossToTextEngine(),
                    gloss_cfg,
                )

        return orchestrator

    return factory


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def create_app(auth: AuthMiddleware | None = None) -> FastAPI:
    """Build and return the configured FastAPI application.

    Accepts an optional ``auth`` override so tests can inject a
    pre-configured middleware without needing ``SOZIA_API_KEY`` set.

    Args:
        auth: If provided, used as-is.  If None, ``AuthMiddleware.from_env()``
            is called during the lifespan startup event.
    """
    registry = ModelRegistry.get_instance()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncGenerator[None, None]:
        device = _device()

        # Resolve auth — either injected (tests) or from environment.
        resolved_auth = auth if auth is not None else AuthMiddleware.from_env()

        # Register engine classes so registry can resolve prefixes.
        registry.register_engine_class("whisper", WhisperAsrEngine)
        registry.register_engine_class("lip-reading", LipReadingEngine)
        registry.register_engine_class("tsl-gru", TslRecognitionEngine)
        registry.register_engine_class("gemma", GlossToTextEngine)
        logger.info("Engine classes registered. Device: %s", device)

        gateway = WebSocketGateway(
            auth=resolved_auth,
            orchestrator_factory=_make_orchestrator_factory(device),
        )
        application.state.gateway = gateway

        yield

        # Shutdown — close per-session orchestrators first so their engines
        # release GPU memory, then clean up anything the registry still holds.
        await gateway.shutdown_all_sessions()
        await registry.unload_all()
        logger.info("All models unloaded. Server shutting down.")

    application = FastAPI(title="Sozia Server", lifespan=lifespan)

    @application.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        await websocket.app.state.gateway.on_connect(websocket)

    return application


# ---------------------------------------------------------------------------
# Module-level app instance — used by ``uvicorn sozia.server:app``.
# Auth is resolved from SOZIA_API_KEY during lifespan startup, not here.
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)

app = create_app()
