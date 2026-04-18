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


def _make_mock_model(dtype: torch.dtype = torch.float32) -> MagicMock:
    """Return a MagicMock model with parameters() and generate() wired up."""
    mock_model = MagicMock()
    mock_model.parameters.return_value = iter(
        [torch.nn.Parameter(torch.zeros(1, dtype=dtype))]
    )
    mock_output = MagicMock()
    mock_output.sequences = torch.zeros(1, 10, dtype=torch.long)
    mock_output.sequences_scores = torch.tensor([-0.3])
    mock_output.scores = None
    mock_model.generate.return_value = mock_output
    return mock_model


def _make_mock_engine() -> tuple:
    """Return (engine, mock_model, mock_processor) with load_model already called."""
    from sozia.server.inference.speech.whisper_asr_engine import WhisperAsrEngine

    mock_processor = MagicMock()
    mock_processor.get_decoder_prompt_ids.return_value = [[1, 2]]
    mock_processor.tokenizer.batch_decode.return_value = ["merhaba dünya"]

    mock_model = _make_mock_model()

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
        mock_processor.tokenizer.batch_decode.return_value = ["merhaba dünya"]
        mock_model = _make_mock_model()

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

        def _slow(*_a, **_kw):
            _time.sleep(10)

        mock_model = _make_mock_model()
        mock_model.generate.side_effect = _slow

        with patch.object(
            WhisperAsrEngine,
            "_load_from_directory",
            return_value=(MagicMock(), mock_model),
        ):
            engine = WhisperAsrEngine()
            await engine.load_model(_make_config())

        with pytest.raises(InferenceTimeoutError):
            await engine.predict(_make_mel_features(), timeout_ms=1)


# ---------------------------------------------------------------------------
# Mel preparation
# ---------------------------------------------------------------------------


class TestWhisperAsrEngineMelPrep:
    def _make_engine(self, dtype: torch.dtype = torch.float32, n_mels: int = 80):
        from sozia.server.inference.speech.whisper_asr_engine import WhisperAsrEngine

        engine = WhisperAsrEngine()
        engine._model = _make_mock_model(dtype)
        engine._processor = MagicMock()
        engine._device = torch.device("cpu")
        engine._n_mels = n_mels
        return engine

    def test_output_shape_is_batched(self):
        engine = self._make_engine()
        mel, mask = engine._prepare_mel(np.random.randn(80, 200).astype(np.float32))
        assert mel.shape == (1, 80, 3000)

    def test_output_dtype_matches_model(self):
        engine = self._make_engine(dtype=torch.float16)
        mel, _ = engine._prepare_mel(np.random.randn(50, 80).astype(np.float32))
        assert mel.dtype == torch.float16

    def test_128_bin_model_large_v3(self):
        engine = self._make_engine(n_mels=128)
        mel, _ = engine._prepare_mel(np.random.randn(50, 128).astype(np.float32))
        assert mel.shape == (1, 128, 3000)

    def test_transpose_short_chunk_client_format(self):
        # Real client format: 500ms chunk → 50 frames × 80 mel bins → (50, 80).
        # The previous condition (shape[0] > 80) failed here — regression guard.
        engine = self._make_engine()
        mel, _ = engine._prepare_mel(np.random.randn(50, 80).astype(np.float32))
        assert mel.shape == (1, 80, 3000)

    def test_transpose_long_chunk_t_by_80_input(self):
        engine = self._make_engine()
        mel, _ = engine._prepare_mel(np.random.randn(200, 80).astype(np.float32))
        assert mel.shape == (1, 80, 3000)

    def test_pads_short_mel_bins(self):
        engine = self._make_engine()
        mel, _ = engine._prepare_mel(np.random.randn(13, 100).astype(np.float32))
        assert mel.shape == (1, 80, 3000)

    def test_trims_long_time_axis(self):
        engine = self._make_engine()
        mel, _ = engine._prepare_mel(np.random.randn(80, 5000).astype(np.float32))
        assert mel.shape == (1, 80, 3000)

    def test_rejects_1d_input(self):
        engine = self._make_engine()
        with pytest.raises(ValueError, match="2-D"):
            engine._prepare_mel(np.zeros(100, dtype=np.float32))

    def test_attention_mask_shape(self):
        engine = self._make_engine()
        _, mask = engine._prepare_mel(np.random.randn(80, 200).astype(np.float32))
        assert mask.shape == (1, 3000)
        assert mask.dtype == torch.long

    def test_attention_mask_marks_real_frames(self):
        engine = self._make_engine()
        # (N, T) orientation: 80 bins × 200 real frames, padded to 3000
        _, mask = engine._prepare_mel(np.random.randn(80, 200).astype(np.float32))
        assert mask[0, :200].sum().item() == 200
        assert mask[0, 200:].sum().item() == 0

    def test_attention_mask_full_chunk(self):
        # 3000-frame chunk should produce all-ones mask — no padding needed
        engine = self._make_engine()
        _, mask = engine._prepare_mel(np.random.randn(80, 3000).astype(np.float32))
        assert mask.sum().item() == 3000

    def test_normalisation_applied_globally(self):
        engine = self._make_engine()
        # Known array: all values 0.0 except one cell = 8.0 → max_val = 8.0
        # clip(x, 0, 8) → (x+4)/4 → range [1.0, 3.0]
        arr = np.zeros((80, 100), dtype=np.float32)
        arr[0, 0] = 8.0
        mel, _ = engine._prepare_mel(arr)
        tensor_vals = mel[0].numpy()
        assert float(tensor_vals.max()) == pytest.approx(3.0, abs=1e-5)
        assert float(tensor_vals.min()) == pytest.approx(1.0, abs=1e-5)

    def test_normalisation_span_is_exactly_two(self):
        # clip window is 8 units wide; after (x+4)/4 the span collapses to 8/4 = 2.0
        engine = self._make_engine()
        rng = np.random.default_rng(42)
        arr = rng.uniform(-10, 10, (80, 3000)).astype(np.float32)
        mel, _ = engine._prepare_mel(arr)
        span = mel.max().item() - mel.min().item()
        assert span == pytest.approx(2.0, abs=1e-4)


# ---------------------------------------------------------------------------
# VAD gate
# ---------------------------------------------------------------------------


class TestWhisperAsrEngineVad:
    def test_silence_skips_inference(self):
        from sozia.server.inference.speech.whisper_asr_engine import WhisperAsrEngine

        mock_model = _make_mock_model()
        mock_processor = MagicMock()
        mock_processor.tokenizer.batch_decode.return_value = ["hallucination"]

        with patch.object(
            WhisperAsrEngine,
            "_load_from_directory",
            return_value=(mock_processor, mock_model),
        ):
            engine = WhisperAsrEngine()
            import asyncio

            asyncio.get_event_loop().run_until_complete(
                engine.load_model(_make_config())
            )

        # All-−10 array is pure silence (log10-mel floor)
        silent = np.full((80, 200), -10.0, dtype=np.float32)
        result = asyncio.get_event_loop().run_until_complete(engine.predict(silent))
        assert result.text == ""
        assert result.confidence == 0.0
        mock_model.generate.assert_not_called()

    def test_speech_runs_inference(self):
        from sozia.server.inference.speech.whisper_asr_engine import WhisperAsrEngine

        mock_processor = MagicMock()
        mock_processor.tokenizer.batch_decode.return_value = ["merhaba"]
        mock_model = _make_mock_model()

        with patch.object(
            WhisperAsrEngine,
            "_load_from_directory",
            return_value=(mock_processor, mock_model),
        ):
            engine = WhisperAsrEngine()
            import asyncio

            asyncio.get_event_loop().run_until_complete(
                engine.load_model(_make_config())
            )

        # Array with a peak above the threshold → should run inference
        speech = np.full((80, 200), -10.0, dtype=np.float32)
        speech[0, 0] = 0.0
        result = asyncio.get_event_loop().run_until_complete(engine.predict(speech))
        assert result.text == "merhaba"
        mock_model.generate.assert_called_once()

    def test_has_speech_content_threshold(self):
        from sozia.server.inference.speech.whisper_asr_engine import WhisperAsrEngine

        assert WhisperAsrEngine._has_speech_content(np.array([[-10.0, -8.0]])) is False
        assert WhisperAsrEngine._has_speech_content(np.array([[0.0, -10.0]])) is True


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
