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
_NUM_CLASSES = 32


def _make_config(model_id: str = "lipreading-gru-v1") -> ModelConfig:
    return ModelConfig(
        model_id=model_id,
        weights_path="/tmp/lipreading.pt",
        device="cpu",
        params={"hidden_dim": 64, "num_layers": 1, "num_classes": _NUM_CLASSES},
    )


def _make_landmarks(n_frames: int = 30) -> np.ndarray:
    """Fake facial landmarks of shape (T, 249)."""
    return np.random.randn(n_frames, _FACE_DIM).astype(np.float32)


def _make_landmarks_3d(n_frames: int = 30) -> np.ndarray:
    """Fake facial landmarks of shape (T, 83, 3)."""
    return np.random.randn(n_frames, 83, 3).astype(np.float32)


# ---------------------------------------------------------------------------
# Model architecture tests
# ---------------------------------------------------------------------------


class TestLipReadingModel:
    """Smoke-test the model architecture."""

    def test_forward_output_shape(self):
        """Classifier outputs (batch, num_classes) — one prediction per sequence."""
        from sozia.server.inference.speech.lip_reading_engine import LipReadingModel

        model = LipReadingModel(
            input_dim=_FACE_DIM,
            hidden_dim=64,
            num_layers=1,
            num_classes=_NUM_CLASSES,
        )
        x = torch.randn(2, 50, _FACE_DIM)
        lengths = torch.tensor([50, 30])
        out = model(x, lengths=lengths)
        assert out.shape == (2, _NUM_CLASSES)

    def test_forward_without_lengths(self):
        """Without lengths, last sequence step is used. eval() required for batch_size=1."""
        from sozia.server.inference.speech.lip_reading_engine import LipReadingModel

        model = LipReadingModel(
            input_dim=_FACE_DIM,
            hidden_dim=64,
            num_layers=1,
            num_classes=_NUM_CLASSES,
        )
        model.eval()
        x = torch.randn(1, 50, _FACE_DIM)
        out = model(x)
        assert out.shape == (1, _NUM_CLASSES)

    def test_forward_batch_size_one(self):
        """eval() required: BatchNorm1d needs running stats for single-sample batches."""
        from sozia.server.inference.speech.lip_reading_engine import LipReadingModel

        model = LipReadingModel(
            input_dim=_FACE_DIM,
            hidden_dim=64,
            num_layers=1,
            num_classes=_NUM_CLASSES,
        )
        model.eval()
        x = torch.randn(1, 30, _FACE_DIM)
        lengths = torch.tensor([30])
        out = model(x, lengths=lengths)
        assert out.shape == (1, _NUM_CLASSES)


# ---------------------------------------------------------------------------
# Engine lifecycle tests
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
            input_dim=_FACE_DIM,
            hidden_dim=64,
            num_layers=1,
            num_classes=_NUM_CLASSES,
        )
        state_dict = model.state_dict()

        with patch("torch.load", return_value=state_dict):
            engine = LipReadingEngine()
            await engine.load_model(_make_config())
            assert engine.is_loaded() is True
            assert engine.get_model_id() == "lipreading-gru-v1"

            await engine.unload_model()
            assert engine.is_loaded() is False
            assert engine.get_model_id() == ""


# ---------------------------------------------------------------------------
# Predict tests
# ---------------------------------------------------------------------------


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
            input_dim=_FACE_DIM,
            hidden_dim=64,
            num_layers=1,
            num_classes=_NUM_CLASSES,
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
            input_dim=_FACE_DIM,
            hidden_dim=64,
            num_layers=1,
            num_classes=_NUM_CLASSES,
        )
        state_dict = model.state_dict()

        with patch("torch.load", return_value=state_dict):
            engine = LipReadingEngine()
            await engine.load_model(_make_config())

        result = await engine.predict(_make_landmarks_3d())
        assert result.modality_type == ModalityType.LIP_READING

    async def test_predict_with_vocab_returns_word_text(self):
        """When vocab is loaded, text should be the word at the predicted class index."""
        from sozia.server.inference.speech.lip_reading_engine import (
            LipReadingEngine,
            LipReadingModel,
        )

        model = LipReadingModel(
            input_dim=_FACE_DIM,
            hidden_dim=64,
            num_layers=1,
            num_classes=_NUM_CLASSES,
        )
        state_dict = model.state_dict()

        with patch("torch.load", return_value=state_dict):
            engine = LipReadingEngine()
            await engine.load_model(_make_config())

        # Inject a vocab so we can verify word lookup.
        engine._vocab = [f"word_{i}" for i in range(_NUM_CLASSES)]

        result = await engine.predict(_make_landmarks())
        assert result.text.startswith("word_")

    async def test_predict_without_vocab_returns_class_index_string(self):
        from sozia.server.inference.speech.lip_reading_engine import (
            LipReadingEngine,
            LipReadingModel,
        )

        model = LipReadingModel(
            input_dim=_FACE_DIM,
            hidden_dim=64,
            num_layers=1,
            num_classes=_NUM_CLASSES,
        )
        state_dict = model.state_dict()

        with patch("torch.load", return_value=state_dict):
            engine = LipReadingEngine()
            await engine.load_model(_make_config())

        # No vocab — should fall back to class index string.
        engine._vocab = None
        result = await engine.predict(_make_landmarks())
        assert result.text.isdigit()

    async def test_predict_timeout_raises(self):
        import time as _time

        from sozia.server.inference.speech.lip_reading_engine import LipReadingEngine

        engine = LipReadingEngine()
        engine._model = MagicMock()
        engine._device = torch.device("cpu")
        engine._max_seq_len = 150

        def _slow_inference(*_):
            # Synchronous sleep — mimics a slow model inside to_thread.
            _time.sleep(10)
            return torch.randn(1, _NUM_CLASSES)

        with patch.object(engine, "_run_inference", side_effect=_slow_inference):
            with pytest.raises(InferenceTimeoutError):
                await engine.predict(_make_landmarks(), timeout_ms=1)


# ---------------------------------------------------------------------------
# Input preparation tests
# ---------------------------------------------------------------------------


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

    async def test_accepts_single_frame_83x3(self):
        """Single frame (83, 3) from _face_cache is expanded to (1, 249) — DEV-07 MVP path."""
        from sozia.server.inference.speech.lip_reading_engine import LipReadingEngine

        engine = LipReadingEngine()
        engine._model = MagicMock()
        engine._device = torch.device("cpu")
        engine._max_seq_len = 150
        single_frame = np.random.randn(83, 3).astype(np.float32)
        tensor, length = engine._prepare_input(single_frame)
        assert tensor.shape == (1, 150, _FACE_DIM)
        assert length.item() == 1


# ---------------------------------------------------------------------------
# Decode tests
# ---------------------------------------------------------------------------


class TestLipReadingEngineDecode:
    def test_decode_with_vocab_returns_word(self):
        from sozia.server.inference.speech.lip_reading_engine import LipReadingEngine

        engine = LipReadingEngine()
        vocab = [f"word_{i}" for i in range(_NUM_CLASSES)]
        engine._vocab = vocab

        # Force class 5 to be highest probability.
        logits = torch.full((1, _NUM_CLASSES), -1.0)
        logits[0, 5] = 10.0
        text, confidence = engine._decode_output(logits)

        assert text == "word_5"
        assert confidence > 0.9

    def test_decode_without_vocab_returns_index_string(self):
        from sozia.server.inference.speech.lip_reading_engine import LipReadingEngine

        engine = LipReadingEngine()
        engine._vocab = None

        logits = torch.full((1, _NUM_CLASSES), -1.0)
        logits[0, 3] = 10.0
        text, confidence = engine._decode_output(logits)

        assert text == "3"
        assert confidence > 0.9

    def test_decode_confidence_is_clamped(self):
        from sozia.server.inference.speech.lip_reading_engine import LipReadingEngine

        engine = LipReadingEngine()
        engine._vocab = None

        # Even with extreme logits, confidence stays in [0, 1].
        logits = torch.zeros(1, _NUM_CLASSES)
        _, confidence = engine._decode_output(logits)
        assert 0.0 <= confidence <= 1.0
