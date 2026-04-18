"""LipReadingEngine — visual speech recognition from facial landmarks.

Receives a sequence of 83-point facial landmark frames (extracted by
MediaPipe Face Mesh on the client), runs them through a CNN-GRU word
classifier, and returns a ModalityResult with the predicted word label.

Architecture (word classifier, not CTC):
    Conv1d temporal smoother → GRU encoder → last real-frame hidden state
    → BN → ReLU → Dropout → Linear(num_classes)

The model is trained offline on face landmarks from the TSL datasets
(AUTSL / BosphorusSign22k) using weak supervision: sign class names
(Turkish words) serve as word-level labels.

In the speech pipeline this is the *secondary* (slower) modality: ASR emits
PARTIAL immediately, lip-reading arrives later and the fusion layer produces
FINAL via weighted merge.

Latency budget: ≤ 1200 ms.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn

from sozia.common.interfaces import (
    InferenceEngine,
    InferenceTimeoutError,
    ModelNotLoadedError,
)
from sozia.common.models import ModalityResult, ModalityType

if TYPE_CHECKING:
    from sozia.common.models import ModelConfig

_DEFAULT_TIMEOUT_MS = 1200
_CACHE_CLEAR_INTERVAL = 10

# Face landmark subset used by Sozia (83 points × 3 coords = 249 features).
_FACE_FEATURE_DIM = 249


class LipReadingModel(nn.Module):
    """GRU word classifier for lip-motion features (ActionGRU small).

    Mirrors the ActionGRU architecture trained in sozia-research:
        GRU encoder (4 layers, hidden=256, input=249)
        → last real-frame hidden state
        → head: Linear(256,512) → BN → ReLU → Dropout
                → Linear(512,256) → BN → ReLU → Dropout
                → Linear(256, num_classes)

    No Conv1d frontend — the GRU receives raw 249-dim landmark features.
    """

    def __init__(
        self,
        input_dim: int = _FACE_FEATURE_DIM,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_classes: int = 226,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: ``(batch, seq_len, input_dim)``
            lengths: Actual sequence lengths ``(batch,)``. When provided,
                the last *real* frame's hidden state is used as the sequence
                representation, avoiding zero-pad contamination.

        Returns:
            Class logits of shape ``(batch, num_classes)``.
        """
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x,
                lengths.cpu().clamp(min=1),
                batch_first=True,
                enforce_sorted=False,
            )
            packed_out, _ = self.gru(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
            idx = (lengths - 1).clamp(min=0).to(out.device)
            last = out[torch.arange(out.size(0), device=out.device), idx]
        else:
            out, _ = self.gru(x)
            last = out[:, -1, :]

        return self.head(last)


class LipReadingEngine(InferenceEngine):
    """InferenceEngine implementation for word-level lip-motion classification.

    Expected ``ModelConfig.params`` keys (all optional):
        hidden_dim (int): GRU hidden size, default ``256``.
        num_layers (int): GRU layers, default ``3``.
        num_classes (int): Number of word classes, default ``226`` (AUTSL).
        max_seq_len (int): Max input frames, default ``150``.
        vocab_path (str): Path to newline-delimited vocabulary file (class
            index → word label, one word per line).
    """

    def __init__(self) -> None:
        self._model: LipReadingModel | None = None
        self._model_id: str = ""
        self._device: torch.device = torch.device("cpu")
        self._max_seq_len: int = 150
        self._vocab: list[str] | None = None
        self._inference_count: int = 0

    # ------------------------------------------------------------------
    # InferenceEngine interface
    # ------------------------------------------------------------------

    async def load_model(self, config: ModelConfig) -> None:
        device = torch.device(config.device)
        hidden_dim = config.params.get("hidden_dim", 256)
        num_layers = config.params.get("num_layers", 4)
        num_classes = config.params.get("num_classes", 226)
        self._max_seq_len = config.params.get("max_seq_len", 150)

        model = LipReadingModel(
            input_dim=_FACE_FEATURE_DIM,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
        )

        state_dict = await asyncio.to_thread(
            torch.load,
            config.weights_path,
            map_location=device,
            weights_only=True,
        )
        model.load_state_dict(state_dict)
        model.to(device)
        # Normalize to fp32: state dicts saved with AMP have fp16 GRU/Linear weights
        # but fp32 BatchNorm running_mean/running_var (PyTorch never saves BN stats as
        # fp16). The dtype mismatch causes a RuntimeError in BatchNorm at inference.
        model.float()
        model.eval()

        self._model = model
        self._device = device
        self._model_id = config.model_id

        vocab_path = config.params.get("vocab_path")
        if vocab_path:
            self._vocab = await asyncio.to_thread(self._load_vocab, vocab_path)

    async def predict(
        self,
        features: np.ndarray,
        timeout_ms: int = _DEFAULT_TIMEOUT_MS,
    ) -> ModalityResult:
        """Run lip-reading inference on facial landmark frames.

        Args:
            features: Numpy array of shape ``(T, 249)`` — T frames of 83
                landmarks × 3 coordinates each. Accepts ``(T, 83, 3)`` and
                single-frame inputs ``(83, 3)`` which are reshaped automatically.
            timeout_ms: Maximum wall-clock time in ms (default 1200).

        Returns:
            ModalityResult with predicted word label and softmax confidence.
        """
        if self._model is None:
            raise ModelNotLoadedError("LipReadingEngine: no model loaded.")

        tensor, length = self._prepare_input(features)
        start = time.perf_counter()

        try:
            logits = await asyncio.wait_for(
                asyncio.to_thread(self._run_inference, tensor, length),
                timeout=timeout_ms / 1000.0,
            )
        except asyncio.TimeoutError:
            raise InferenceTimeoutError(
                f"LipReadingEngine exceeded {timeout_ms}ms budget."
            )
        finally:
            # On timeout the thread still holds a reference to `tensor` via its
            # call frame; this del only drops the event-loop copy. GPU memory is
            # released when the thread returns naturally.
            del tensor

        latency_ms = int((time.perf_counter() - start) * 1000)
        text, confidence = self._decode_output(logits)

        # _inference_count is only incremented from the asyncio event loop (after
        # await), so no locking is needed. One counter per engine instance.
        self._inference_count += 1
        if (
            self._inference_count % _CACHE_CLEAR_INTERVAL == 0
            and torch.cuda.is_available()
        ):
            torch.cuda.empty_cache()

        return ModalityResult(
            modality_type=ModalityType.LIP_READING,
            text=text,
            confidence=confidence,
            timestamp_ms=0,
            duration_ms=0,
            inference_latency_ms=latency_ms,
        )

    async def unload_model(self) -> None:
        self._model = None
        self._model_id = ""
        self._vocab = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def is_loaded(self) -> bool:
        return self._model is not None

    def get_model_id(self) -> str:
        return self._model_id

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_input(
        self,
        features: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalise input shape and convert to batched tensor.

        Accepted input shapes:
            ``(T, 249)``     — T frames already flattened
            ``(T, 83, 3)``   — T frames as landmark arrays (reshaped to flat)
            ``(83, 3)``      — single frame (expanded to ``(1, 249)``)
        """
        arr = np.asarray(features, dtype=np.float32)

        if arr.ndim == 3:
            # (T, 83, 3) → (T, 249)
            arr = arr.reshape(arr.shape[0], -1)
        elif arr.shape == (83, 3):
            # Single frame (83, 3) → (1, 249)
            arr = arr.reshape(1, -1)

        if arr.ndim != 2 or arr.shape[1] != _FACE_FEATURE_DIM:
            raise ValueError(
                f"LipReadingEngine: expected features of shape (T, {_FACE_FEATURE_DIM}), "
                f"got {arr.shape}"
            )

        seq_len = arr.shape[0]
        if seq_len > self._max_seq_len:
            arr = arr[: self._max_seq_len]
            seq_len = self._max_seq_len

        # Pad to max_seq_len.
        if seq_len < self._max_seq_len:
            pad = np.zeros(
                (self._max_seq_len - seq_len, _FACE_FEATURE_DIM),
                dtype=np.float32,
            )
            arr = np.vstack([arr, pad])

        tensor = torch.from_numpy(arr).unsqueeze(0).to(self._device)
        length = torch.tensor([seq_len], dtype=torch.long)
        return tensor, length

    def _run_inference(
        self,
        tensor: torch.Tensor,
        length: torch.Tensor,
    ) -> torch.Tensor:
        """Synchronous forward pass — called inside ``to_thread``.

        Returns logits on CPU so the CUDA allocation is released before the
        result crosses back to the event loop.
        """
        with torch.inference_mode():
            logits = self._model(tensor, lengths=length)
        return logits.cpu()

    def _decode_output(
        self,
        logits: torch.Tensor,
    ) -> tuple[str, float]:
        """Argmax-decode classifier logits into a word label and confidence.

        Args:
            logits: ``(1, num_classes)`` raw logits from the model.

        Returns:
            ``(text, confidence)`` where ``text`` is the predicted word
            (or class index string when no vocab is loaded) and
            ``confidence`` is the softmax probability at the predicted class.
        """
        # logits: (1, num_classes)
        probs = torch.softmax(logits[0], dim=-1)  # (num_classes,)
        class_idx = int(torch.argmax(probs).item())
        confidence = float(probs[class_idx].item())

        if self._vocab and class_idx < len(self._vocab):
            text = self._vocab[class_idx]
        else:
            text = str(class_idx)

        return text, max(0.0, min(1.0, confidence))

    @staticmethod
    def _load_vocab(path: str) -> list[str]:
        """Load a newline-delimited vocabulary file.

        Line number = class index. Each line is the word label for that class.
        """
        with open(path, encoding="utf-8") as f:
            return [line.strip() for line in f]
