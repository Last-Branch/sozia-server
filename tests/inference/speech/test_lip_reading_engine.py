"""Tests for LipReadingEngine."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from sozia.common.interfaces import InferenceTimeoutError, ModelNotLoadedError
from sozia.common.models import ModelConfig, ModalityType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FACE_DIM = 249  # 83 points × 3 coords


def _make_config(model_id: str = "lipreading-cnn-gru-v1") -> ModelConfig:
    return ModelConfig(
        model_id=model_id,
        weights_path="/tmp/lipreading.pt",
        device="cpu",
        params={"hidden_dim": 64, "num_layers": 1, "vocab_size": 32},
    )


def _make_landmarks(n_frames: int = 30) -> np.ndarray:
    """Fake facial landmarks of shape (T, 249)."""
    return np.random.randn(n_frames, _FACE_DIM).astype(np.float32)


def _make_landmarks_3d(n_frames: int = 30) -> np.ndarray:
    """Fake facial landmarks of shape (T, 83, 3)."""
    return np.random.randn(n_frames, 83, 3).astype(np.float32)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLipReadingEngineState:
    async def test_not_loaded_initially(self):
        from sozia.server.inference.speech.lip_reading_engine import LipReadingEngine

        engine = LipReadingEngine()
        assert engine.is_loaded() is False
        assert engine.get_model_id() == ""

    async def test_load_and_unload_lifecycle(self):
        from sozia.server.inference.speech.lip_reading_engine import (
            LipReadingEngine,
            LipReadingModel,
        )

        model = LipReadingModel(
            input_dim=_FACE_DIM, hidden_dim=64, num_layers=1, vocab_size=32,
        )
        state_dict = model.state_dict()

        with patch("torch.load", return_value=state_dict):
            engine = LipReadingEngine()
            await engine.load_model(_make_config())
            assert engine.is_loaded() is True
            assert engine.get_model_id() == "lipreading-cnn-gru-v1"

            await engine.unload_model()
            assert engine.is_loaded() is False
            assert engine.get_model_id() == ""


class TestLipReadingEnginePredict:
    async def test_predict_raises_when_not_loaded(self):
        from sozia.server.inference.speech.lip_reading_engine import LipReadingEngine

        engine = LipReadingEngine()
        with pytest.raises(ModelNotLoadedError):
            await engine.predict(_make_landmarks())

    async def test_predict_returns_modality_result(self):
        from sozia.server.inference.speech.lip_reading_engine import (
            LipReadingEngine,
            LipReadingModel,
        )

        model = LipReadingModel(
            input_dim=_FACE_DIM, hidden_dim=64, num_layers=1, vocab_size=32,
        )
        state_dict = model.state_dict()

        with patch("torch.load", return_value=state_dict):
            engine = LipReadingEngine()
            await engine.load_model(_make_config())

        result = await engine.predict(_make_landmarks())
        assert result.modality_type == ModalityType.LIP_READING
        assert isinstance(result.text, str)
        assert 0.0 <= result.confidence <= 1.0
        assert result.inference_latency_ms >= 0

    async def test_predict_accepts_3d_landmarks(self):
        from sozia.server.inference.speech.lip_reading_engine import (
            LipReadingEngine,
            LipReadingModel,
        )

        model = LipReadingModel(
            input_dim=_FACE_DIM, hidden_dim=64, num_layers=1, vocab_size=32,
        )
        state_dict = model.state_dict()

        with patch("torch.load", return_value=state_dict):
            engine = LipReadingEngine()
            await engine.load_model(_make_config())

        result = await engine.predict(_make_landmarks_3d())
        assert result.modality_type == ModalityType.LIP_READING


class TestLipReadingEngineInput:
    async def test_rejects_wrong_feature_dim(self):
        from sozia.server.inference.speech.lip_reading_engine import LipReadingEngine

        engine = LipReadingEngine()
        engine._model = MagicMock()
        with pytest.raises(ValueError, match="249"):
            engine._prepare_input(np.zeros((30, 100), dtype=np.float32))

    async def test_pads_short_sequences(self):
        from sozia.server.inference.speech.lip_reading_engine import LipReadingEngine

        engine = LipReadingEngine()
        engine._model = MagicMock()
        engine._device = torch.device("cpu")
        engine._max_seq_len = 150
        tensor, length = engine._prepare_input(_make_landmarks(10))
        assert tensor.shape == (1, 150, _FACE_DIM)
        assert length.item() == 10

    async def test_truncates_long_sequences(self):
        from sozia.server.inference.speech.lip_reading_engine import LipReadingEngine

        engine = LipReadingEngine()
        engine._model = MagicMock()
        engine._device = torch.device("cpu")
        engine._max_seq_len = 150
        tensor, length = engine._prepare_input(_make_landmarks(200))
        assert tensor.shape == (1, 150, _FACE_DIM)
        assert length.item() == 150


class TestLipReadingModel:
    """Smoke test the model architecture itself."""

    def test_forward_shape(self):
        from sozia.server.inference.speech.lip_reading_engine import LipReadingModel

        model = LipReadingModel(
            input_dim=_FACE_DIM, hidden_dim=64, num_layers=1, vocab_size=32,
        )
        x = torch.randn(2, 50, _FACE_DIM)
        lengths = torch.tensor([50, 30])
        out = model(x, lengths=lengths)
        assert out.shape == (2, 50, 32)

    def test_forward_without_lengths(self):
        from sozia.server.inference.speech.lip_reading_engine import LipReadingModel

        model = LipReadingModel(
            input_dim=_FACE_DIM, hidden_dim=64, num_layers=1, vocab_size=32,
        )
        x = torch.randn(1, 50, _FACE_DIM)
        out = model(x)
        assert out.shape == (1, 50, 32)
