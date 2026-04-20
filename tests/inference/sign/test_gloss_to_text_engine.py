"""Tests for GlossToTextEngine."""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import pytest
import torch

from sozia.common.interfaces import InferenceTimeoutError, ModelNotLoadedError
from sozia.common.models import ModelConfig, ModalityType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(model_id: str = "gemma-9b-gloss-tr") -> ModelConfig:
    return ModelConfig(
        model_id=model_id,
        weights_path="/tmp/final_adapter",
        device="cpu",
        params={
            "base_model_id": "google/gemma-2-9b-it",
            "load_in_4bit": False,
        },
    )


def _make_generate_output(
    sequences: torch.Tensor,
    sequences_scores: torch.Tensor | None = None,
) -> MagicMock:
    """Build a mock that mimics HuggingFace GenerateBeamDecoderOnlyOutput."""
    out = MagicMock()
    out.sequences = sequences
    out.sequences_scores = sequences_scores
    return out


def _loaded_engine():
    """Return a GlossToTextEngine with model/tokenizer mocked out."""
    from sozia.server.inference.sign.gloss_to_text_engine import GlossToTextEngine

    engine = GlossToTextEngine()
    engine._model_id = "gemma-9b-gloss-tr"
    engine._base_model_id = "google/gemma-2-9b-it"
    engine._device = torch.device("cpu")

    tok_output = MagicMock()
    tok_output.input_ids = torch.tensor([[1, 2, 3]])  # input_len = 3
    tok_output.attention_mask = torch.tensor([[1, 1, 1]])
    engine._tokenizer = MagicMock(return_value=tok_output)
    engine._tokenizer.decode.return_value = "merhaba dünya"  # pre-polish
    return engine


# ---------------------------------------------------------------------------
# Turkish text utility tests
# ---------------------------------------------------------------------------


class TestTurkishUtils:
    def test_turkish_lower(self):
        from sozia.server.inference.sign.gloss_to_text_engine import _turkish_lower

        assert _turkish_lower("İSTANBUL") == "istanbul"
        assert _turkish_lower("IŞIK") == "ışık"

    def test_turkish_capitalize(self):
        from sozia.server.inference.sign.gloss_to_text_engine import _turkish_capitalize

        assert _turkish_capitalize("istanbul") == "İstanbul"
        assert _turkish_capitalize("ışık") == "Işık"
        assert _turkish_capitalize("") == ""

    def test_polish_turkish_adds_period(self):
        from sozia.server.inference.sign.gloss_to_text_engine import _polish_turkish

        assert _polish_turkish("merhaba dünya") == "Merhaba dünya."

    def test_polish_turkish_capitalises_after_period(self):
        from sozia.server.inference.sign.gloss_to_text_engine import _polish_turkish

        result = _polish_turkish("merhaba. nasılsın")
        assert result == "Merhaba. Nasılsın."

    def test_polish_turkish_empty(self):
        from sozia.server.inference.sign.gloss_to_text_engine import _polish_turkish

        assert _polish_turkish("") == ""
        assert _polish_turkish("  ") == ""


class TestFormatChat:
    def test_gemma_format(self):
        from sozia.server.inference.sign.gloss_to_text_engine import _format_chat

        result = _format_chat("google/gemma-2-9b-it", "Translate:", "merhaba")
        assert "<start_of_turn>user" in result
        assert "Gloss: merhaba" in result

    def test_llama_format(self):
        from sozia.server.inference.sign.gloss_to_text_engine import _format_chat

        result = _format_chat("meta-llama/llama-3", "Translate:", "merhaba")
        assert "<|begin_of_text|>" in result

    def test_chatml_format(self):
        from sozia.server.inference.sign.gloss_to_text_engine import _format_chat

        result = _format_chat("trendyol/model", "Translate:", "merhaba")
        assert "<|im_start|>user" in result


# ---------------------------------------------------------------------------
# _extract_confidence helper
# ---------------------------------------------------------------------------


class TestExtractConfidence:
    def test_sequences_scores_converted_via_exp(self):
        from sozia.server.inference.sign.gloss_to_text_engine import _extract_confidence

        score = -0.5
        output = _make_generate_output(
            sequences=torch.tensor([[1]]),
            sequences_scores=torch.tensor([score]),
        )
        expected = math.exp(score)
        assert abs(_extract_confidence(output) - expected) < 1e-5

    def test_confidence_clamped_to_1_for_positive_score(self):
        from sozia.server.inference.sign.gloss_to_text_engine import _extract_confidence

        output = _make_generate_output(
            sequences=torch.tensor([[1]]),
            sequences_scores=torch.tensor([2.0]),  # exp(2) > 1
        )
        assert _extract_confidence(output) == 1.0

    def test_confidence_clamped_to_0_for_very_negative_score(self):
        from sozia.server.inference.sign.gloss_to_text_engine import _extract_confidence

        output = _make_generate_output(
            sequences=torch.tensor([[1]]),
            sequences_scores=torch.tensor([-100.0]),  # exp(-100) ≈ 0
        )
        assert _extract_confidence(output) == pytest.approx(0.0, abs=1e-6)

    def test_fallback_when_sequences_scores_is_none(self):
        from sozia.server.inference.sign.gloss_to_text_engine import _extract_confidence

        output = _make_generate_output(
            sequences=torch.tensor([[1]]),
            sequences_scores=None,
        )
        # Fallback: exp(-1.0) — any value in [0, 1] is acceptable.
        result = _extract_confidence(output)
        assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# Engine state tests
# ---------------------------------------------------------------------------


class TestGlossToTextEngineState:
    async def test_not_loaded_initially(self):
        from sozia.server.inference.sign.gloss_to_text_engine import GlossToTextEngine

        engine = GlossToTextEngine()
        assert engine.is_loaded() is False
        assert engine.get_model_id() == ""

    async def test_unload_clears_state(self):
        from sozia.server.inference.sign.gloss_to_text_engine import GlossToTextEngine

        engine = GlossToTextEngine()
        engine._model = MagicMock()
        engine._tokenizer = MagicMock()
        engine._model_id = "test"

        await engine.unload_model()
        assert engine.is_loaded() is False
        assert engine.get_model_id() == ""


# ---------------------------------------------------------------------------
# predict() — confidence wired from beam sequences_scores
# ---------------------------------------------------------------------------


class TestGlossToTextEnginePredict:
    async def test_predict_raises_when_not_loaded(self):
        from sozia.server.inference.sign.gloss_to_text_engine import GlossToTextEngine

        engine = GlossToTextEngine()
        with pytest.raises(ModelNotLoadedError):
            await engine.predict("MERHABA")

    async def test_predict_returns_modality_result(self):
        engine = _loaded_engine()
        engine._model = MagicMock()
        engine._model.generate.return_value = _make_generate_output(
            sequences=torch.tensor([[1, 2, 3, 4, 5, 6]]),
            sequences_scores=torch.tensor([-0.3]),
        )

        result = await engine.predict("MERHABA DUNYA")
        assert result.modality_type == ModalityType.GLOSS_TO_TEXT
        assert "Merhaba" in result.text
        assert 0.0 <= result.confidence <= 1.0
        assert result.inference_latency_ms >= 0

    async def test_confidence_derived_from_sequences_scores(self):
        """Real beam score → exp(score) replaces the old 0.85 heuristic."""
        engine = _loaded_engine()
        known_score = -0.4
        engine._model = MagicMock()
        engine._model.generate.return_value = _make_generate_output(
            sequences=torch.tensor([[1, 2, 3, 4]]),
            sequences_scores=torch.tensor([known_score]),
        )

        result = await engine.predict("MERHABA")
        expected_conf = math.exp(known_score)
        assert abs(result.confidence - expected_conf) < 1e-4

    async def test_confidence_is_not_hardcoded_085(self):
        """Ensure the old 0.85 literal is gone — two calls with different scores differ."""
        engine = _loaded_engine()
        mock_model = MagicMock()
        engine._model = mock_model

        mock_model.generate.return_value = _make_generate_output(
            sequences=torch.tensor([[1, 2, 3, 4]]),
            sequences_scores=torch.tensor([-0.2]),
        )
        r1 = await engine.predict("A")

        mock_model.generate.return_value = _make_generate_output(
            sequences=torch.tensor([[1, 2, 3, 4]]),
            sequences_scores=torch.tensor([-1.5]),
        )
        r2 = await engine.predict("B")

        assert r1.confidence != r2.confidence

    async def test_generate_kwargs_include_return_dict_and_output_scores(self):
        """Verify generate() is called with the flags needed for sequences_scores."""
        engine = _loaded_engine()
        engine._model = MagicMock()
        engine._model.generate.return_value = _make_generate_output(
            sequences=torch.tensor([[1, 2, 3, 4]]),
            sequences_scores=torch.tensor([-0.5]),
        )

        await engine.predict("X")

        call_kwargs = engine._model.generate.call_args.kwargs
        assert call_kwargs.get("return_dict_in_generate") is True
        assert call_kwargs.get("output_scores") is True

    async def test_empty_output_gives_zero_confidence(self):
        engine = _loaded_engine()
        engine._tokenizer.decode.return_value = ""
        engine._model = MagicMock()
        engine._model.generate.return_value = _make_generate_output(
            sequences=torch.tensor([[1, 2, 3]]),
            sequences_scores=torch.tensor([-0.3]),
        )

        result = await engine.predict("MERHABA")
        assert result.confidence == 0.0
        assert result.text == ""

    async def test_predict_timeout_raises_inference_timeout_error(self):
        import asyncio
        from unittest.mock import patch

        engine = _loaded_engine()
        engine._model = MagicMock()

        async def _slow_coro(*_args, **_kwargs):
            await asyncio.sleep(10)

        with patch(
            "sozia.server.inference.sign.gloss_to_text_engine.asyncio.to_thread",
            side_effect=_slow_coro,
        ):
            with pytest.raises(InferenceTimeoutError):
                await engine.predict("MERHABA", timeout_ms=1)
