"""WhisperAsrEngine — HuggingFace Whisper-based automatic speech recognition.

Wraps a fine-tuned WhisperForConditionalGeneration model (ogulcanakca/whisper-small-tr)
for Turkish ASR. Receives preprocessed mel-spectrogram features from the client,
runs encoder-decoder inference, and returns a ModalityResult.

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
_CACHE_CLEAR_INTERVAL = 10


class WhisperAsrEngine(InferenceEngine):
    """InferenceEngine backed by a HuggingFace WhisperForConditionalGeneration.

    ``ModelConfig.weights_path`` must point to the model directory containing
    ``config.json``, ``model.safetensors``, and tokenizer files.

    Expected ``ModelConfig.params`` keys (all optional):
        language (str): Target language code, default ``"tr"``.
        task (str): ``"transcribe"`` or ``"translate"``, default ``"transcribe"``.
        num_beams (int): Beam search width, default ``1`` (greedy).
    """

    def __init__(self) -> None:
        self._model = None
        self._processor = None
        self._model_id: str = ""
        self._device: torch.device = torch.device("cpu")
        self._language: str = "tr"
        self._task: str = "transcribe"
        self._num_beams: int = 1
        self._inference_count: int = 0

    # ------------------------------------------------------------------
    # InferenceEngine interface
    # ------------------------------------------------------------------

    async def load_model(self, config: ModelConfig) -> None:

        self._device = torch.device(config.device)
        self._language = config.params.get("language", "tr")
        self._task = config.params.get("task", "transcribe")
        self._num_beams = config.params.get("num_beams", 1)

        processor, model = await asyncio.to_thread(
            self._load_from_directory, config.weights_path, config.device
        )
        self._processor = processor
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
                ``(T, 80)`` (auto-transposed). Padded/trimmed to ``(80, 3000)``.
            timeout_ms: Maximum wall-clock time in ms (default 800).

        Returns:
            ModalityResult with recognised text, confidence, and timing.
        """
        if self._model is None or self._processor is None:
            raise ModelNotLoadedError("WhisperAsrEngine: no model loaded.")

        mel = self._prepare_mel(features)
        start = time.perf_counter()

        try:
            text, confidence = await asyncio.wait_for(
                asyncio.to_thread(self._run_inference, mel),
                timeout=timeout_ms / 1000.0,
            )
        except asyncio.TimeoutError:
            raise InferenceTimeoutError(
                f"WhisperAsrEngine exceeded {timeout_ms}ms budget."
            )
        finally:
            # On timeout the thread is still running and holds its own reference
            # to `mel` via the call frame; this del only drops the event-loop
            # copy. GPU memory is released when the thread returns naturally.
            del mel

        latency_ms = int((time.perf_counter() - start) * 1000)

        self._inference_count += 1
        if (
            self._inference_count % _CACHE_CLEAR_INTERVAL == 0
            and torch.cuda.is_available()
        ):
            torch.cuda.empty_cache()

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
        self._processor = None
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

    @staticmethod
    def _load_from_directory(model_dir: str, device: str):
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        processor = WhisperProcessor.from_pretrained(model_dir)
        model = WhisperForConditionalGeneration.from_pretrained(model_dir)
        model = model.to(device)
        model.eval()
        return processor, model

    def _prepare_mel(self, features: np.ndarray) -> torch.Tensor:
        """Convert incoming feature array to a Whisper-compatible mel tensor.

        Returns a float32 tensor of shape ``(1, 80, 3000)`` (batch=1).
        """
        arr = np.asarray(features, dtype=np.float32)

        if arr.ndim == 1:
            raise ValueError(
                f"WhisperAsrEngine: expected 2-D features, got shape {arr.shape}"
            )

        # Transpose (T, 80) → (80, T)
        if arr.ndim == 2 and arr.shape[1] <= 80 and arr.shape[0] > 80:
            arr = arr.T

        n_mels, t_len = arr.shape[0], arr.shape[1]

        # Zero-pad to 80 mel bins if fewer (e.g. 13-dim MFCC input)
        if n_mels < 80:
            arr = np.vstack([arr, np.zeros((80 - n_mels, t_len), dtype=np.float32)])

        # Whisper expects exactly 3000 time frames (30 s × 100 Hz)
        target_t = 3000
        if arr.shape[1] < target_t:
            arr = np.pad(arr, ((0, 0), (0, target_t - arr.shape[1])))
        elif arr.shape[1] > target_t:
            arr = arr[:, :target_t]

        # Global log-mel normalisation matching WhisperFeatureExtractor.
        # Must be applied after the full segment is assembled — frame-level
        # normalisation on the client produces inconsistent scale.
        max_val = float(arr.max())
        arr = np.clip(arr, max_val - 8.0, max_val)
        arr = (arr + 4.0) / 4.0

        return torch.from_numpy(arr).unsqueeze(0).to(self._device)

    def _run_inference(self, mel: torch.Tensor) -> tuple[str, float]:
        """Synchronous HF Whisper decode — called inside ``to_thread``."""
        forced_ids = self._processor.get_decoder_prompt_ids(
            language=self._language, task=self._task
        )
        with torch.inference_mode():
            output = self._model.generate(
                mel,
                forced_decoder_ids=forced_ids,
                num_beams=self._num_beams,
                return_dict_in_generate=True,
                output_scores=True,
            )
            text = self._processor.tokenizer.batch_decode(
                output.sequences, skip_special_tokens=True
            )[0].strip()
            confidence = self._extract_confidence(output)

        # Explicitly release GPU tensors held in the generate output (scores
        # tuple, sequences) before returning to the event loop.
        del output
        return text, confidence

    def _extract_confidence(self, output) -> float:
        """Derive a [0, 1] confidence from generation sequence score."""
        import math

        if hasattr(output, "sequences_scores") and output.sequences_scores is not None:
            # sequences_scores is the sum of log-probs normalised by length
            score = output.sequences_scores[0].item()
        elif output.scores:
            # Fallback: mean of per-step max log-prob (CPU to avoid CUDA intermediates).
            log_probs = [
                torch.log_softmax(s.cpu().float(), dim=-1).max(dim=-1).values.item()
                for s in output.scores
            ]
            score = sum(log_probs) / len(log_probs) if log_probs else -1.0
        else:
            score = -1.0

        return max(0.0, min(1.0, math.exp(score)))
