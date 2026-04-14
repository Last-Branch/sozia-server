"""Tests for TslRecognitionEngine."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from sozia.common.interfaces import InferenceTimeoutError, ModelNotLoadedError
from sozia.common.models import ModelConfig, ModalityType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FEATURE_DIM = 507
_NUM_CLASSES = 10
_MAX_SEQ_LEN = 150


class _FakeScaler:
    """Picklable scaler stub — identity transform."""

    def transform(self, x):
        return x


def _make_config(run_dir: str, model_id: str = "tsl-gru-v1", params: dict | None = None) -> ModelConfig:
    return ModelConfig(
        model_id=model_id,
        weights_path=run_dir,
        device="cpu",
        params=params or {},
    )


def _make_features(n_frames: int = 50) -> np.ndarray:
    return np.random.randn(n_frames, _FEATURE_DIM).astype(np.float32)


def _create_fake_run_dir(tmp_path: Path) -> Path:
    """Create a minimal run directory with config.json and best_model.pt."""
    from sozia.inference.sign._gru_model import ActionGRU

    run_dir = tmp_path / "run_001"
    run_dir.mkdir()

    classes = [f"CLASS_{i}" for i in range(_NUM_CLASSES)]
    config = {
        "model_arch": "gru",
        "model_size": "small",
        "feature_dim": _FEATURE_DIM,
        "dropout": 0.4,
        "split_mode": "signer",
        "classes_to_process": classes,
        "max_sequence_length": _MAX_SEQ_LEN,
        "sequence_handling": "truncate",
        "normalize_features": False,
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f)

    model = ActionGRU(
        input_size=_FEATURE_DIM,
        num_classes=_NUM_CLASSES,
        model_size="small",
        dropout=0.4,
    )
    torch.save(model.state_dict(), run_dir / "best_model.pt")

    return run_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTslRecognitionEngineState:
    async def test_not_loaded_initially(self):
        from sozia.inference.sign.tsl_recognition_engine import TslRecognitionEngine

        engine = TslRecognitionEngine()
        assert engine.is_loaded() is False
        assert engine.get_model_id() == ""

    async def test_load_and_unload_lifecycle(self, tmp_path):
        from sozia.inference.sign.tsl_recognition_engine import TslRecognitionEngine

        run_dir = _create_fake_run_dir(tmp_path)
        engine = TslRecognitionEngine()
        await engine.load_model(_make_config(str(run_dir)))

        assert engine.is_loaded() is True
        assert engine.get_model_id() == "tsl-gru-v1"

        await engine.unload_model()
        assert engine.is_loaded() is False
        assert engine.get_model_id() == ""


class TestTslRecognitionEnginePredict:
    async def test_predict_raises_when_not_loaded(self):
        from sozia.inference.sign.tsl_recognition_engine import TslRecognitionEngine

        engine = TslRecognitionEngine()
        with pytest.raises(ModelNotLoadedError):
            await engine.predict(_make_features())

    async def test_predict_returns_modality_result(self, tmp_path):
        from sozia.inference.sign.tsl_recognition_engine import TslRecognitionEngine

        run_dir = _create_fake_run_dir(tmp_path)
        engine = TslRecognitionEngine()
        await engine.load_model(_make_config(str(run_dir)))

        result = await engine.predict(_make_features())
        assert result.modality_type == ModalityType.TSL_RECOGNITION
        assert isinstance(result.text, str)
        assert result.text.startswith("CLASS_")
        assert 0.0 <= result.confidence <= 1.0
        assert result.inference_latency_ms >= 0

    async def test_predict_handles_long_sequences(self, tmp_path):
        from sozia.inference.sign.tsl_recognition_engine import TslRecognitionEngine

        run_dir = _create_fake_run_dir(tmp_path)
        engine = TslRecognitionEngine()
        await engine.load_model(_make_config(str(run_dir)))

        result = await engine.predict(_make_features(n_frames=300))
        assert result.modality_type == ModalityType.TSL_RECOGNITION

    async def test_predict_handles_short_sequences(self, tmp_path):
        from sozia.inference.sign.tsl_recognition_engine import TslRecognitionEngine

        run_dir = _create_fake_run_dir(tmp_path)
        engine = TslRecognitionEngine()
        await engine.load_model(_make_config(str(run_dir)))

        result = await engine.predict(_make_features(n_frames=5))
        assert result.modality_type == ModalityType.TSL_RECOGNITION


class TestTslRecognitionEnginePreprocess:
    async def test_rejects_1d_input(self):
        from sozia.inference.sign.tsl_recognition_engine import TslRecognitionEngine

        engine = TslRecognitionEngine()
        engine._model = MagicMock()
        with pytest.raises(ValueError, match="2-D"):
            engine._preprocess(np.zeros(100, dtype=np.float32))

    async def test_pad_short_sequence(self):
        from sozia.inference.sign.tsl_recognition_engine import TslRecognitionEngine

        engine = TslRecognitionEngine()
        engine._model = MagicMock()
        engine._max_seq_len = 150
        engine._normalize = False
        engine._apply_interpolation = False
        engine._feature_dim = _FEATURE_DIM

        arr, length = engine._preprocess(_make_features(10))
        assert arr.shape == (150, _FEATURE_DIM)
        assert length == 10

    async def test_truncate_long_sequence(self):
        from sozia.inference.sign.tsl_recognition_engine import TslRecognitionEngine

        engine = TslRecognitionEngine()
        engine._model = MagicMock()
        engine._max_seq_len = 150
        engine._normalize = False
        engine._apply_interpolation = False
        engine._feature_dim = _FEATURE_DIM
        engine._sequence_handling = "truncate"

        arr, length = engine._preprocess(_make_features(200))
        assert arr.shape == (150, _FEATURE_DIM)
        assert length == 150


class TestTslRecognitionEngineScalerPath:
    async def test_explicit_scaler_path_is_used(self, tmp_path):
        """scaler_path in params takes priority over run-dir candidate search."""
        import json
        import pickle

        from sozia.inference.sign.tsl_recognition_engine import TslRecognitionEngine

        run_dir = _create_fake_run_dir(tmp_path)

        # Write a picklable stub scaler to a separate dir (mimics scalers/AUTSL/).
        scaler_dir = tmp_path / "scalers" / "AUTSL"
        scaler_dir.mkdir(parents=True)
        scaler_file = scaler_dir / "scaler_signer.pkl"
        with open(scaler_file, "wb") as f:
            pickle.dump(_FakeScaler(), f)

        # Enable normalisation in the run dir config.
        config_path = run_dir / "config.json"
        with open(config_path) as f:
            meta = json.load(f)
        meta["normalize_features"] = True
        with open(config_path, "w") as f:
            json.dump(meta, f)

        engine = TslRecognitionEngine()
        await engine.load_model(
            _make_config(str(run_dir), params={"scaler_path": str(scaler_file)})
        )

        assert engine._scaler is not None
        assert engine.is_loaded()

    async def test_missing_explicit_scaler_path_falls_back(self, tmp_path):
        """If scaler_path points to a non-existent file, fall back to candidates."""
        from sozia.inference.sign.tsl_recognition_engine import TslRecognitionEngine

        run_dir = _create_fake_run_dir(tmp_path)
        engine = TslRecognitionEngine()
        # Fake path that doesn't exist — engine should load fine (scaler stays None).
        await engine.load_model(
            _make_config(str(run_dir), params={"scaler_path": "/nonexistent/scaler.pkl"})
        )
        assert engine.is_loaded()
        assert engine._scaler is None


class TestTslRecognitionEngineLoadErrors:
    async def test_missing_config_json(self, tmp_path):
        from sozia.inference.sign.tsl_recognition_engine import TslRecognitionEngine

        empty_dir = tmp_path / "empty_run"
        empty_dir.mkdir()

        engine = TslRecognitionEngine()
        with pytest.raises(FileNotFoundError, match="config.json"):
            await engine.load_model(_make_config(str(empty_dir)))

    async def test_missing_checkpoint(self, tmp_path):
        from sozia.inference.sign.tsl_recognition_engine import TslRecognitionEngine

        run_dir = tmp_path / "run_no_ckpt"
        run_dir.mkdir()
        config = {
            "model_arch": "gru", "model_size": "small",
            "feature_dim": _FEATURE_DIM, "classes_to_process": ["A", "B"],
            "max_sequence_length": 150,
        }
        with open(run_dir / "config.json", "w") as f:
            json.dump(config, f)

        engine = TslRecognitionEngine()
        with pytest.raises(FileNotFoundError, match="best_model.pt"):
            await engine.load_model(_make_config(str(run_dir)))


class TestGruModel:
    """Smoke test the inline GRU model."""

    def test_forward_shape(self):
        from sozia.inference.sign._gru_model import ActionGRU

        model = ActionGRU(
            input_size=_FEATURE_DIM, num_classes=_NUM_CLASSES,
            model_size="small", dropout=0.4,
        )
        x = torch.randn(2, 50, _FEATURE_DIM)
        lengths = torch.tensor([50, 30])
        out = model(x, lengths=lengths)
        assert out.shape == (2, _NUM_CLASSES)

    def test_forward_without_lengths(self):
        from sozia.inference.sign._gru_model import ActionGRU

        model = ActionGRU(
            input_size=_FEATURE_DIM, num_classes=_NUM_CLASSES,
        )
        model.eval()  # BatchNorm requires eval mode for batch_size=1
        x = torch.randn(1, 50, _FEATURE_DIM)
        out = model(x)
        assert out.shape == (1, _NUM_CLASSES)
