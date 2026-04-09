"""WhisperAsrEngine — Whisper-based automatic speech recognition.

Wraps OpenAI Whisper (small) for Turkish ASR. Receives preprocessed audio
features (mel spectrogram) from the client, runs encoder-decoder inference,
and returns a ModalityResult.

Latency budget: ≤ 800 ms.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

import numpy as np
import torch

from sozia.common.interfaces import (
    InferenceEngine,
    InferenceTimeoutError,
    ModelNotLoadedError,
)
from sozia.common.models import ModalityResult, ModalityType

if TYPE_CHECKING:
    from sozia.common.models import ModelConfig

_DEFAULT_TIMEOUT_MS = 800


class WhisperAsrEngine(InferenceEngine):
    """InferenceEngine implementation backed by OpenAI Whisper.

    Expected ``ModelConfig.params`` keys (all optional):
        language (str): Target language code, default ``"tr"``.
        beam_size (int): Beam search width, default ``5``.
        task (str): ``"transcribe"`` or ``"translate"``, default ``"transcribe"``.
    """

    def __init__(self) -> None:
        self._model: torch.nn.Module | None = None
        self._model_id: str = ""
        self._device: torch.device = torch.device("cpu")
        self._language: str = "tr"
        self._beam_size: int = 5
        self._task: str = "transcribe"
        self._tokenizer = None
        self._decoder = None

    # ------------------------------------------------------------------
    # InferenceEngine interface
    # ------------------------------------------------------------------

    async def load_model(self, config: ModelConfig) -> None:
        import whisper  # lazy import — not needed at module scope

        self._device = torch.device(config.device)
        self._language = config.params.get("language", "tr")
        self._beam_size = config.params.get("beam_size", 5)
        self._task = config.params.get("task", "transcribe")

        model_size = config.params.get("model_size", "small")
        model = await asyncio.to_thread(
            whisper.load_model, model_size, device=config.device,
            download_root=config.weights_path,
        )
        self._model = model
        self._model_id = config.model_id

    async def predict(
        self,
        features: np.ndarray,
        timeout_ms: int = _DEFAULT_TIMEOUT_MS,
    ) -> ModalityResult:
        """Run ASR inference on preprocessed audio features.

        Args:
            features: Mel-spectrogram numpy array of shape ``(80, T)`` or
                ``(T, 80)`` (auto-transposed). Alternatively a 2-D MFCC
                array — the engine pads/trims to Whisper's expected 80-mel
                format internally.
            timeout_ms: Maximum wall-clock time in ms (default 800).

        Returns:
            ModalityResult with recognised text, confidence, and timing.
        """
        if self._model is None:
            raise ModelNotLoadedError("WhisperAsrEngine: no model loaded.")

        mel = self._prepare_mel(features)
        start = time.perf_counter()

        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(self._run_inference, mel),
                timeout=timeout_ms / 1000.0,
            )
        except asyncio.TimeoutError:
            raise InferenceTimeoutError(
                f"WhisperAsrEngine exceeded {timeout_ms}ms budget."
            )

        latency_ms = int((time.perf_counter() - start) * 1000)
        text: str = result.get("text", "").strip()
        confidence = self._extract_confidence(result)

        return ModalityResult(
            modality_type=ModalityType.ASR,
            text=text,
            confidence=confidence,
            timestamp_ms=0,
            duration_ms=0,
            inference_latency_ms=latency_ms,
        )

    async def unload_model(self) -> None:
        self._model = None
        self._model_id = ""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def is_loaded(self) -> bool:
        return self._model is not None

    def get_model_id(self) -> str:
        return self._model_id

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_mel(self, features: np.ndarray) -> torch.Tensor:
        """Convert incoming feature array to a Whisper-compatible mel tensor.

        Whisper expects a float32 tensor of shape ``(80, 3000)`` (80 mel bins,
        30 s at 100 Hz). This helper handles common input shapes and pads /
        trims to the expected length.
        """
        arr = np.asarray(features, dtype=np.float32)

        if arr.ndim == 1:
            raise ValueError(
                f"WhisperAsrEngine: expected 2-D features, got shape {arr.shape}"
            )

        # If shape is (T, 80), transpose to (80, T).
        if arr.ndim == 2 and arr.shape[1] <= 80 and arr.shape[0] > 80:
            arr = arr.T

        n_mels = arr.shape[0]
        t_len = arr.shape[1] if arr.ndim == 2 else arr.shape[0]

        # Pad to 80 mel bins if fewer (e.g. 13-dim MFCC → zero-pad upper bins).
        if n_mels < 80:
            pad = np.zeros((80 - n_mels, t_len), dtype=np.float32)
            arr = np.vstack([arr, pad])

        # Whisper expects exactly 3000 time frames (30 s * 100 Hz).
        target_t = 3000
        if arr.shape[1] < target_t:
            arr = np.pad(arr, ((0, 0), (0, target_t - arr.shape[1])))
        elif arr.shape[1] > target_t:
            arr = arr[:, :target_t]

        return torch.from_numpy(arr).to(self._device)

    def _run_inference(self, mel: torch.Tensor) -> dict:
        """Synchronous Whisper decode — called inside ``to_thread``."""
        import whisper

        options = whisper.DecodingOptions(
            language=self._language,
            beam_size=self._beam_size,
            task=self._task,
            without_timestamps=True,
        )
        with torch.no_grad():
            result = whisper.decode(self._model, mel, options)

        # whisper.decode returns a DecodingResult or list thereof.
        if isinstance(result, list):
            result = result[0]

        return {
            "text": result.text,
            "avg_logprob": getattr(result, "avg_logprob", -1.0),
            "no_speech_prob": getattr(result, "no_speech_prob", 0.0),
        }

    @staticmethod
    def _extract_confidence(result: dict) -> float:
        """Derive a [0, 1] confidence score from Whisper's log-probabilities."""
        avg_logprob = result.get("avg_logprob", -1.0)
        no_speech = result.get("no_speech_prob", 0.0)

        # avg_logprob is negative; closer to 0 → higher confidence.
        # Map roughly: -0.0 → 1.0, -1.0 → ~0.37, -3.0 → ~0.05
        import math

        raw = math.exp(avg_logprob)
        # Penalise high no-speech probability.
        confidence = raw * (1.0 - no_speech)
        return max(0.0, min(1.0, confidence))
