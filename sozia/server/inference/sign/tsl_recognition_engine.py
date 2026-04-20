"""TslRecognitionEngine — Turkish Sign Language recognition via GRU classifier.

Receives per-frame landmark features (507-dim: pose 132 + face 249 + hands 126)
accumulated by the client, runs them through a trained GRU model, and returns
the top-1 gloss label (e.g. "MERHABA DUNYA") as a ModalityResult.

In the sign pipeline this is the *primary* (faster) modality: its output is
emitted as a PARTIAL segment, later replaced by GlossToTextEngine's natural
Turkish FINAL.

Latency budget: ≤ 1000 ms.
"""

from __future__ import annotations

import asyncio
import json
import pickle
import time
from pathlib import Path
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

_DEFAULT_TIMEOUT_MS = 1000

# 507-dim: pose(33×4) + face(83×3) + left_hand(21×3) + right_hand(21×3)
_FEATURE_DIM = 507


class TslRecognitionEngine(InferenceEngine):
    """InferenceEngine wrapping the GRU-based sign language classifier.

    The underlying model architecture and training pipeline live in
    ``sozia-research/tsl_recognition``. This engine loads a trained checkpoint
    produced by that pipeline and exposes it behind the standard
    InferenceEngine interface.

    Expected ``ModelConfig`` layout:
        weights_path: Path to the ``run_*`` directory containing
            ``best_model.pt`` and ``config.json``.
        device: ``"cpu"`` or ``"cuda"``.
        params (optional overrides):
            normalize (bool): Apply StandardScaler, default ``True``.
            apply_interpolation (bool): Interpolate missing landmarks,
                default ``True``.
            scaler_path (str): Explicit path to a ``.pkl`` scaler file.
                Takes priority over the candidate search in the run dir.
    """

    def __init__(self) -> None:
        self._model: nn.Module | None = None
        self._model_id: str = ""
        self._device: torch.device = torch.device("cpu")
        self._scaler = None
        self._actions: np.ndarray | None = None
        self._max_seq_len: int = 150
        self._sequence_handling: str = "truncate"
        self._normalize: bool = True
        self._apply_interpolation: bool = True
        self._feature_dim: int = _FEATURE_DIM

    # ------------------------------------------------------------------
    # InferenceEngine interface
    # ------------------------------------------------------------------

    async def load_model(self, config: ModelConfig) -> None:
        run_dir = Path(config.weights_path)
        device = torch.device(config.device)
        self._normalize = config.params.get("normalize", True)
        self._apply_interpolation = config.params.get("apply_interpolation", True)
        scaler_path_str = config.params.get("scaler_path")
        scaler_path = Path(scaler_path_str) if scaler_path_str else None

        run_data = await asyncio.to_thread(self._load_run, run_dir, device, scaler_path)

        self._model = run_data["model"]
        self._scaler = run_data["scaler"]
        self._actions = run_data["actions"]
        self._max_seq_len = run_data["max_len"]
        self._sequence_handling = run_data["sequence_handling"]
        self._normalize = run_data["normalize"]
        self._feature_dim = run_data["feature_dim"]
        self._device = device
        self._model_id = config.model_id

    async def predict(
        self,
        features: np.ndarray,
        timeout_ms: int = _DEFAULT_TIMEOUT_MS,
    ) -> ModalityResult:
        """Classify a landmark sequence into a sign gloss label.

        Args:
            features: Numpy array of shape ``(T, 507)`` — T frames of
                flattened body landmarks.
            timeout_ms: Maximum wall-clock time in ms (default 1000).

        Returns:
            ModalityResult whose ``text`` field contains the predicted gloss
            (e.g. ``"MERHABA"``).
        """
        if self._model is None:
            raise ModelNotLoadedError("TslRecognitionEngine: no model loaded.")

        kp_array, actual_length = self._preprocess(features)
        start = time.perf_counter()

        try:
            top_preds, probs = await asyncio.wait_for(
                asyncio.to_thread(
                    self._run_inference,
                    kp_array,
                    actual_length,
                ),
                timeout=timeout_ms / 1000.0,
            )
        except asyncio.TimeoutError:
            raise InferenceTimeoutError(
                f"TslRecognitionEngine exceeded {timeout_ms}ms budget."
            )

        latency_ms = int((time.perf_counter() - start) * 1000)
        gloss, confidence = top_preds[0]

        return ModalityResult(
            modality_type=ModalityType.TSL_RECOGNITION,
            text=gloss,
            confidence=max(0.0, min(1.0, confidence)),
            timestamp_ms=0,
            duration_ms=0,
            inference_latency_ms=latency_ms,
        )

    async def unload_model(self) -> None:
        self._model = None
        self._model_id = ""
        self._scaler = None
        self._actions = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def is_loaded(self) -> bool:
        return self._model is not None

    def get_model_id(self) -> str:
        return self._model_id

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _preprocess(
        self,
        features: np.ndarray,
    ) -> tuple[np.ndarray, int]:
        """Preprocess landmark frames identically to the training pipeline."""
        arr = np.asarray(features, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(
                f"TslRecognitionEngine: expected 2-D array (T, {self._feature_dim}), "
                f"got shape {arr.shape}"
            )

        n_frames = len(arr)

        # Optional missing-landmark interpolation.
        if self._apply_interpolation and arr.shape[1] == self._feature_dim:
            try:
                from sozia.server.inference.sign._interpolation import (
                    interpolate_missing_keypoints,
                )

                arr = interpolate_missing_keypoints(arr).astype(np.float32)
            except ImportError:
                pass

        # Normalise with the scaler fitted at training time.
        if self._normalize and self._scaler is not None:
            arr = self._scaler.transform(arr)

        # Truncate or sample long sequences.
        if n_frames > self._max_seq_len:
            if self._sequence_handling == "uniform_sample":
                indices = (
                    np.linspace(
                        0,
                        n_frames - 1,
                        self._max_seq_len,
                    )
                    .round()
                    .astype(int)
                )
                arr = arr[indices]
            else:
                arr = arr[: self._max_seq_len]
            n_frames = self._max_seq_len

        actual_length = n_frames

        # Pad short sequences.
        if n_frames < self._max_seq_len:
            pad = np.zeros(
                (self._max_seq_len - n_frames, arr.shape[1]),
                dtype=np.float32,
            )
            arr = np.vstack([arr, pad])

        return arr, actual_length

    def _run_inference(
        self,
        kp_array: np.ndarray,
        actual_length: int,
    ) -> tuple[list[tuple[str, float]], np.ndarray]:
        """Synchronous GRU forward pass + softmax — called inside ``to_thread``."""
        x = torch.tensor(kp_array, dtype=torch.float32).unsqueeze(0).to(self._device)
        lengths = torch.tensor([actual_length], dtype=torch.long).to(self._device)

        with torch.no_grad():
            logits = self._model(x, lengths=lengths)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        top_k = min(5, len(probs))
        idx = np.argsort(probs)[::-1][:top_k]
        preds = [(str(self._actions[i]), float(probs[i])) for i in idx]
        return preds, probs

    @staticmethod
    def _load_run(
        run_dir: Path, device: torch.device, scaler_path: Path | None = None
    ) -> dict:
        """Load model, scaler, and metadata from a training run directory.

        Mirrors ``sozia-research/tsl_recognition/evaluation/inference._load_run``
        but without any cv2/GUI dependencies.
        """
        config_path = run_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"No config.json in {run_dir}")

        with open(config_path) as f:
            meta = json.load(f)

        model_arch: str = meta["model_arch"]
        model_size: str = meta["model_size"]
        feature_dim: int = int(meta["feature_dim"])
        dropout: float = float(meta.get("dropout") or 0.4)
        split_mode: str = meta.get("split_mode", "signer")
        classes: list[str] = meta["classes_to_process"]
        num_classes = len(classes)
        max_len: int = int(meta["max_sequence_length"])
        seq_handling: str = meta.get("sequence_handling", "truncate")
        normalize: bool = bool(meta.get("normalize_features", True))

        # Build model via inline factory (avoids importing sozia-research).
        model = _build_gru_model(
            input_size=feature_dim,
            num_classes=num_classes,
            model_size=model_size,
            dropout=dropout,
        )
        ckpt_path = run_dir / "best_model.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"No best_model.pt in {run_dir}")
        state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        # Load scaler.
        scaler = None
        if normalize:
            scaler_candidates: list[Path] = []
            if scaler_path is not None:
                scaler_candidates.append(scaler_path)
            scaler_candidates.extend(
                [
                    run_dir / f"scaler_{split_mode}.pkl",
                    run_dir / "scaler.pkl",
                ]
            )
            # Also check dataset processed dir if metadata has dataset name.
            dataset_name = meta.get("dataset", "bosphorus")
            data_root = Path(meta.get("data_root", ""))
            if data_root.exists():
                scaler_candidates.append(
                    data_root / dataset_name / "processed" / f"scaler_{split_mode}.pkl"
                )

            for sp in scaler_candidates:
                if sp.exists():
                    with open(sp, "rb") as f:
                        scaler = pickle.load(f)  # noqa: S301
                    break

        return {
            "model": model,
            "scaler": scaler,
            "actions": np.array(classes),
            "max_len": max_len,
            "sequence_handling": seq_handling,
            "normalize": normalize,
            "feature_dim": feature_dim,
        }


# ---------------------------------------------------------------------------
# Inline GRU model (mirrors sozia-research ActionGRU to avoid cross-repo
# import). Kept minimal — only the forward path needed for inference.
# ---------------------------------------------------------------------------


def _build_gru_model(
    input_size: int,
    num_classes: int,
    model_size: str = "small",
    dropout: float = 0.4,
) -> nn.Module:
    """Construct an ActionGRU-compatible model for inference."""
    from sozia.server.inference.sign._gru_model import ActionGRU

    return ActionGRU(
        input_size=input_size,
        num_classes=num_classes,
        model_size=model_size,
        dropout=dropout,
    )
