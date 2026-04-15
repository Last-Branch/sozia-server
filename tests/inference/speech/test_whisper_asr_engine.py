"""Tests for WhisperAsrEngine."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from sozia.common.interfaces import InferenceTimeoutError, ModelNotLoadedError
from sozia.common.models import ModelConfig, ModalityType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(model_id: str = "whisper-small-tr") -> ModelConfig:
    return ModelConfig(
        model_id=model_id,
        weights_path="/tmp/whisper",
        device="cpu",
        params={"model_size": "small", "language": "tr"},
    )


def _make_mel_features(n_mels: int = 80, t_frames: int = 100) -> np.ndarray:
    """Fake mel spectrogram of shape (n_mels, t_frames)."""
    return np.random.randn(n_mels, t_frames).astype(np.float32)


@dataclass
class FakeDecodingResult:
    text: str = "merhaba dünya"
    avg_logprob: float = -0.3
    no_speech_prob: float = 0.01


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestWhisperAsrEngineState:
    """State management: is_loaded, get_model_id, load/unload lifecycle."""

    async def test_not_loaded_initially(self):
        from sozia.server.inference.speech.whisper_asr_engine import WhisperAsrEngine

        engine = WhisperAsrEngine()
        assert engine.is_loaded() is False
        assert engine.get_model_id() == ""

    async def test_load_sets_state(self):
        from sozia.server.inference.speech.whisper_asr_engine import WhisperAsrEngine

        mock_whisper = MagicMock()
        mock_whisper.load_model.return_value = MagicMock()
        with patch.dict("sys.modules", {"whisper": mock_whisper}):
            engine = WhisperAsrEngine()
            await engine.load_model(_make_config())
            assert engine.is_loaded() is True
            assert engine.get_model_id() == "whisper-small-tr"

    async def test_unload_clears_state(self):
        from sozia.server.inference.speech.whisper_asr_engine import WhisperAsrEngine

        mock_whisper = MagicMock()
        mock_whisper.load_model.return_value = MagicMock()
        with patch.dict("sys.modules", {"whisper": mock_whisper}):
            engine = WhisperAsrEngine()
            await engine.load_model(_make_config())
            await engine.unload_model()
            assert engine.is_loaded() is False
            assert engine.get_model_id() == ""


class TestWhisperAsrEnginePredict:
    """Predict behaviour with mocked whisper backend."""

    async def test_predict_raises_when_not_loaded(self):
        from sozia.server.inference.speech.whisper_asr_engine import WhisperAsrEngine

        engine = WhisperAsrEngine()
        with pytest.raises(ModelNotLoadedError):
            await engine.predict(np.zeros((80, 100), dtype=np.float32))

    async def test_predict_returns_modality_result(self):
        from sozia.server.inference.speech.whisper_asr_engine import WhisperAsrEngine

        mock_whisper = MagicMock()
        mock_whisper.load_model.return_value = MagicMock()
        fake_result = FakeDecodingResult()
        mock_whisper.decode.return_value = fake_result
        mock_whisper.DecodingOptions.return_value = MagicMock()

        with patch.dict("sys.modules", {"whisper": mock_whisper}):
            engine = WhisperAsrEngine()
            await engine.load_model(_make_config())
            result = await engine.predict(_make_mel_features())

        assert result.modality_type == ModalityType.ASR
        assert result.text == "merhaba dünya"
        assert 0.0 <= result.confidence <= 1.0
        assert result.inference_latency_ms >= 0

    async def test_predict_handles_list_decode_result(self):
        from sozia.server.inference.speech.whisper_asr_engine import WhisperAsrEngine

        mock_whisper = MagicMock()
        mock_whisper.load_model.return_value = MagicMock()
        mock_whisper.decode.return_value = [FakeDecodingResult(text="test")]
        mock_whisper.DecodingOptions.return_value = MagicMock()

        with patch.dict("sys.modules", {"whisper": mock_whisper}):
            engine = WhisperAsrEngine()
            await engine.load_model(_make_config())
            result = await engine.predict(_make_mel_features())
        assert result.text == "test"


class TestWhisperAsrEngineMelPrep:
    """Feature preparation edge cases."""

    async def test_transpose_t_by_80_input(self):
        from sozia.server.inference.speech.whisper_asr_engine import WhisperAsrEngine

        engine = WhisperAsrEngine()
        engine._model = MagicMock()
        engine._device = "cpu"
        # (T=200, 80) → should be transposed to (80, T)
        features = np.random.randn(200, 80).astype(np.float32)
        mel = engine._prepare_mel(features)
        assert mel.shape == (80, 3000)

    async def test_pads_short_mel_bins(self):
        from sozia.server.inference.speech.whisper_asr_engine import WhisperAsrEngine

        engine = WhisperAsrEngine()
        engine._model = MagicMock()
        engine._device = "cpu"
        # 13-dim MFCC
        features = np.random.randn(13, 100).astype(np.float32)
        mel = engine._prepare_mel(features)
        assert mel.shape == (80, 3000)

    async def test_rejects_1d_input(self):
        from sozia.server.inference.speech.whisper_asr_engine import WhisperAsrEngine

        engine = WhisperAsrEngine()
        engine._model = MagicMock()
        engine._device = "cpu"
        with pytest.raises(ValueError, match="2-D"):
            engine._prepare_mel(np.zeros(100, dtype=np.float32))


class TestWhisperAsrEngineConfidence:
    """Confidence extraction from log-probabilities."""

    def test_high_logprob_gives_high_confidence(self):
        from sozia.server.inference.speech.whisper_asr_engine import WhisperAsrEngine

        c = WhisperAsrEngine._extract_confidence(
            {"avg_logprob": -0.1, "no_speech_prob": 0.0}
        )
        assert c > 0.8

    def test_low_logprob_gives_low_confidence(self):
        from sozia.server.inference.speech.whisper_asr_engine import WhisperAsrEngine

        c = WhisperAsrEngine._extract_confidence(
            {"avg_logprob": -3.0, "no_speech_prob": 0.0}
        )
        assert c < 0.1

    def test_high_no_speech_penalises(self):
        from sozia.server.inference.speech.whisper_asr_engine import WhisperAsrEngine

        c = WhisperAsrEngine._extract_confidence(
            {"avg_logprob": -0.1, "no_speech_prob": 0.9}
        )
        assert c < 0.15
