"""LipReadingEngine — visual speech recognition from facial landmarks.

Receives a sequence of 83-point facial landmark frames (extracted by
MediaPipe Face Mesh on the client), runs them through a CNN/RNN encoder-
decoder, and returns a ModalityResult with recognised text.

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

# Face landmark subset used by Sozia (83 points × 3 coords = 249 features).
_FACE_FEATURE_DIM = 249


class LipReadingModel(nn.Module):
    """Lightweight CNN-GRU encoder with CTC-style linear head.

    Architecture:
        1-D Conv block (temporal smoothing) → GRU encoder → FC → output vocab
    The model is intentionally small so that inference fits within 1200 ms on
    CPU.  Weights are trained offline and loaded via ``state_dict``.
    """

    def __init__(
        self,
        input_dim: int = _FACE_FEATURE_DIM,
        hidden_dim: int = 256,
        num_layers: int = 3,
        vocab_size: int = 1024,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: ``(batch, seq_len, input_dim)``
            lengths: actual sequence lengths ``(batch,)``

        Returns:
            Logits of shape ``(batch, seq_len, vocab_size)``.
        """
        # Conv expects (batch, channels, seq_len)
        out = self.conv(x.transpose(1, 2)).transpose(1, 2)

        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                out, lengths.cpu().clamp(min=1),
                batch_first=True, enforce_sorted=False,
            )
            packed_out, _ = self.gru(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        else:
            out, _ = self.gru(out)

        return self.fc(out)


class LipReadingEngine(InferenceEngine):
    """InferenceEngine implementation for visual speech recognition.

    Expected ``ModelConfig.params`` keys (all optional):
        hidden_dim (int): GRU hidden size, default ``256``.
        num_layers (int): GRU layers, default ``3``.
        vocab_size (int): Output vocabulary size, default ``1024``.
        max_seq_len (int): Max input frames, default ``150``.
        vocab_path (str): Path to token→text vocabulary file.
    """

    def __init__(self) -> None:
        self._model: LipReadingModel | None = None
        self._model_id: str = ""
        self._device: torch.device = torch.device("cpu")
        self._max_seq_len: int = 150
        self._vocab: list[str] | None = None

    # ------------------------------------------------------------------
    # InferenceEngine interface
    # ------------------------------------------------------------------

    async def load_model(self, config: ModelConfig) -> None:
        device = torch.device(config.device)
        hidden_dim = config.params.get("hidden_dim", 256)
        num_layers = config.params.get("num_layers", 3)
        vocab_size = config.params.get("vocab_size", 1024)
        self._max_seq_len = config.params.get("max_seq_len", 150)

        model = LipReadingModel(
            input_dim=_FACE_FEATURE_DIM,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            vocab_size=vocab_size,
        )

        state_dict = await asyncio.to_thread(
            torch.load, config.weights_path,
            map_location=device, weights_only=True,
        )
        model.load_state_dict(state_dict)
        model.to(device)
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
                landmarks × 3 coordinates each. Alternatively ``(T, 83, 3)``
                which is reshaped automatically.
            timeout_ms: Maximum wall-clock time in ms (default 1200).

        Returns:
            ModalityResult with recognised text and confidence.
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

        latency_ms = int((time.perf_counter() - start) * 1000)
        text, confidence = self._decode_output(logits, length)

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
        self, features: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalise input shape and convert to batched tensor."""
        arr = np.asarray(features, dtype=np.float32)

        if arr.ndim == 3:
            # (T, 83, 3) → (T, 249)
            arr = arr.reshape(arr.shape[0], -1)

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
        self, tensor: torch.Tensor, length: torch.Tensor,
    ) -> torch.Tensor:
        """Synchronous forward pass — called inside ``to_thread``."""
        with torch.no_grad():
            return self._model(tensor, lengths=length)

    def _decode_output(
        self, logits: torch.Tensor, length: torch.Tensor,
    ) -> tuple[str, float]:
        """Greedy-decode logits into text and derive a confidence score."""
        # logits: (1, seq_len, vocab_size)
        probs = torch.softmax(logits[0], dim=-1)
        pred_ids = torch.argmax(probs, dim=-1)  # (seq_len,)
        max_probs = probs.gather(1, pred_ids.unsqueeze(-1)).squeeze(-1)

        actual_len = int(length[0].item())
        pred_ids = pred_ids[:actual_len]
        max_probs = max_probs[:actual_len]

        # CTC blank-collapse: remove consecutive duplicates and blank (id=0).
        collapsed: list[int] = []
        conf_scores: list[float] = []
        prev = -1
        for i, tok_id in enumerate(pred_ids.tolist()):
            if tok_id != 0 and tok_id != prev:
                collapsed.append(tok_id)
                conf_scores.append(max_probs[i].item())
            prev = tok_id

        if self._vocab and collapsed:
            text = " ".join(
                self._vocab[tid] if tid < len(self._vocab) else f"<{tid}>"
                for tid in collapsed
            )
        else:
            text = " ".join(str(tid) for tid in collapsed)

        confidence = float(np.mean(conf_scores)) if conf_scores else 0.0
        return text, max(0.0, min(1.0, confidence))

    @staticmethod
    def _load_vocab(path: str) -> list[str]:
        """Load a newline-delimited vocabulary file."""
        with open(path, encoding="utf-8") as f:
            return [line.strip() for line in f]
