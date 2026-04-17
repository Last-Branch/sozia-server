"""Tests for WhisperAsrEngine (HuggingFace transformers backend)."""

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


def _make_config(model_id: str = "whisper-small-tr") -> ModelConfig:
    return ModelConfig(
        model_id=model_id,
        weights_path="/tmp/whisper-small-tr",
        device="cpu",
        params={"language": "tr"},
    )


def _make_mel_features(n_mels: int = 80, t_frames: int = 100) -> np.ndarray:
    return np.random.randn(n_mels, t_frames).astype(np.float32)


def _make_mock_engine() -> tuple:
    """Return (engine, mock_model, mock_processor) with load_model already called."""
    from sozia.server.inference.speech.whisper_asr_engine import WhisperAsrEngine

    mock_processor = MagicMock()
    mock_processor.get_decoder_prompt_ids.return_value = [[1, 2]]
    mock_processor.tokenizer.batch_decode.return_value = ["merhaba dünya"]

    mock_model = MagicMock()
    mock_output = MagicMock()
    mock_output.sequences = torch.zeros(1, 10, dtype=torch.long)
    mock_output.sequences_scores = torch.tensor([-0.3])
    mock_output.scores = None
    mock_model.generate.return_value = mock_output

    with patch.object(
        WhisperAsrEngine,
        "_load_from_directory",
        return_value=(mock_processor, mock_model),
    ):
        engine = WhisperAsrEngine()
        import asyncio

        asyncio.get_event_loop().run_until_complete(engine.load_model(_make_config()))

    return engine, mock_model, mock_processor


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------


class TestWhisperAsrEngineState:
    async def test_not_loaded_initially(self):
        from sozia.server.inference.speech.whisper_asr_engine import WhisperAsrEngine

        engine = WhisperAsrEngine()
        assert engine.is_loaded() is False
        assert engine.get_model_id() == ""

    async def test_load_sets_state(self):
        from sozia.server.inference.speech.whisper_asr_engine import WhisperAsrEngine

        with patch.object(
            WhisperAsrEngine,
            "_load_from_directory",
            return_value=(MagicMock(), MagicMock()),
        ):
            engine = WhisperAsrEngine()
            await engine.load_model(_make_config())
            assert engine.is_loaded() is True
            assert engine.get_model_id() == "whisper-small-tr"

    async def test_unload_clears_state(self):
        from sozia.server.inference.speech.whisper_asr_engine import WhisperAsrEngine

        with patch.object(
            WhisperAsrEngine,
            "_load_from_directory",
            return_value=(MagicMock(), MagicMock()),
        ):
            engine = WhisperAsrEngine()
            await engine.load_model(_make_config())
            await engine.unload_model()
            assert engine.is_loaded() is False
            assert engine.get_model_id() == ""


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------


class TestWhisperAsrEnginePredict:
    async def test_predict_raises_when_not_loaded(self):
        from sozia.server.inference.speech.whisper_asr_engine import WhisperAsrEngine

        engine = WhisperAsrEngine()
        with pytest.raises(ModelNotLoadedError):
            await engine.predict(np.zeros((80, 100), dtype=np.float32))

    async def test_predict_returns_modality_result(self):
        from sozia.server.inference.speech.whisper_asr_engine import WhisperAsrEngine

        mock_processor = MagicMock()
        mock_processor.get_decoder_prompt_ids.return_value = [[1, 2]]
        mock_processor.tokenizer.batch_decode.return_value = ["merhaba dünya"]

        mock_output = MagicMock()
        mock_output.sequences = torch.zeros(1, 10, dtype=torch.long)
        mock_output.sequences_scores = torch.tensor([-0.3])
        mock_output.scores = None
        mock_model = MagicMock()
        mock_model.generate.return_value = mock_output

        with patch.object(
            WhisperAsrEngine,
            "_load_from_directory",
            return_value=(mock_processor, mock_model),
        ):
            engine = WhisperAsrEngine()
            await engine.load_model(_make_config())
            result = await engine.predict(_make_mel_features())

        assert result.modality_type == ModalityType.ASR
        assert result.text == "merhaba dünya"
        assert 0.0 <= result.confidence <= 1.0
        assert result.inference_latency_ms >= 0

    async def test_predict_timeout_raises(self):
        import time as _time

        from sozia.server.inference.speech.whisper_asr_engine import WhisperAsrEngine

        mock_processor = MagicMock()
        mock_processor.get_decoder_prompt_ids.return_value = []

        def _slow(*_a, **_kw):
            _time.sleep(10)

        mock_model = MagicMock()
        mock_model.generate.side_effect = _slow

        with patch.object(
            WhisperAsrEngine,
            "_load_from_directory",
            return_value=(mock_processor, mock_model),
        ):
            engine = WhisperAsrEngine()
            await engine.load_model(_make_config())

        with pytest.raises(InferenceTimeoutError):
            await engine.predict(_make_mel_features(), timeout_ms=1)


# ---------------------------------------------------------------------------
# Mel preparation
# ---------------------------------------------------------------------------


class TestWhisperAsrEngineMelPrep:
    def _make_engine(self):
        from sozia.server.inference.speech.whisper_asr_engine import WhisperAsrEngine

        engine = WhisperAsrEngine()
        engine._model = MagicMock()
        engine._processor = MagicMock()
        engine._device = torch.device("cpu")
        return engine

    def test_output_shape_is_batched(self):
        engine = self._make_engine()
        mel = engine._prepare_mel(np.random.randn(80, 200).astype(np.float32))
        assert mel.shape == (1, 80, 3000)

    def test_transpose_t_by_80_input(self):
        engine = self._make_engine()
        mel = engine._prepare_mel(np.random.randn(200, 80).astype(np.float32))
        assert mel.shape == (1, 80, 3000)

    def test_pads_short_mel_bins(self):
        engine = self._make_engine()
        mel = engine._prepare_mel(np.random.randn(13, 100).astype(np.float32))
        assert mel.shape == (1, 80, 3000)

    def test_trims_long_time_axis(self):
        engine = self._make_engine()
        mel = engine._prepare_mel(np.random.randn(80, 5000).astype(np.float32))
        assert mel.shape == (1, 80, 3000)

    def test_rejects_1d_input(self):
        engine = self._make_engine()
        with pytest.raises(ValueError, match="2-D"):
            engine._prepare_mel(np.zeros(100, dtype=np.float32))


# ---------------------------------------------------------------------------
# Confidence extraction
# ---------------------------------------------------------------------------


class TestWhisperAsrEngineConfidence:
    def _make_output(self, seq_score: float | None = None, scores=None):
        out = MagicMock()
        out.sequences_scores = (
            torch.tensor([seq_score]) if seq_score is not None else None
        )
        out.scores = scores
        return out

    def test_high_score_gives_high_confidence(self):
        from sozia.server.inference.speech.whisper_asr_engine import WhisperAsrEngine

        engine = WhisperAsrEngine()
        c = engine._extract_confidence(self._make_output(seq_score=-0.1))
        assert c > 0.8

    def test_low_score_gives_low_confidence(self):
        from sozia.server.inference.speech.whisper_asr_engine import WhisperAsrEngine

        engine = WhisperAsrEngine()
        c = engine._extract_confidence(self._make_output(seq_score=-5.0))
        assert c < 0.1

    def test_fallback_to_per_step_scores(self):
        from sozia.server.inference.speech.whisper_asr_engine import WhisperAsrEngine

        engine = WhisperAsrEngine()
        step_scores = [torch.randn(1, 50_000) for _ in range(5)]
        c = engine._extract_confidence(
            self._make_output(seq_score=None, scores=step_scores)
        )
        assert 0.0 <= c <= 1.0

    def test_confidence_always_clamped(self):
        from sozia.server.inference.speech.whisper_asr_engine import WhisperAsrEngine

        engine = WhisperAsrEngine()
        c = engine._extract_confidence(self._make_output(seq_score=0.0))
        assert 0.0 <= c <= 1.0
