"""Tests for GlossToTextEngine."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

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


class TestGlossToTextEnginePredict:
    async def test_predict_raises_when_not_loaded(self):
        from sozia.server.inference.sign.gloss_to_text_engine import GlossToTextEngine

        engine = GlossToTextEngine()
        with pytest.raises(ModelNotLoadedError):
            await engine.predict("MERHABA")

    async def test_predict_returns_modality_result(self):
        from sozia.server.inference.sign.gloss_to_text_engine import GlossToTextEngine

        engine = GlossToTextEngine()
        engine._model_id = "gemma-9b-gloss-tr"
        engine._base_model_id = "google/gemma-2-9b-it"
        engine._device = torch.device("cpu")

        # Mock tokenizer — must return an object with .input_ids / .attention_mask
        tok_output = MagicMock()
        tok_output.input_ids = torch.tensor([[1, 2, 3]])
        tok_output.attention_mask = torch.tensor([[1, 1, 1]])
        mock_tokenizer = MagicMock(return_value=tok_output)
        mock_tokenizer.decode.return_value = "Merhaba dünya."
        engine._tokenizer = mock_tokenizer

        # Mock model
        mock_model = MagicMock()
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5, 6]])
        engine._model = mock_model

        result = await engine.predict("MERHABA DUNYA")
        assert result.modality_type == ModalityType.GLOSS_TO_TEXT
        assert "Merhaba" in result.text
        assert 0.0 <= result.confidence <= 1.0
        assert result.inference_latency_ms >= 0

    async def test_predict_empty_output_gives_zero_confidence(self):
        from sozia.server.inference.sign.gloss_to_text_engine import GlossToTextEngine

        engine = GlossToTextEngine()
        engine._model_id = "gemma-9b-gloss-tr"
        engine._base_model_id = "google/gemma-2-9b-it"
        engine._device = torch.device("cpu")

        tok_output = MagicMock()
        tok_output.input_ids = torch.tensor([[1, 2, 3]])
        tok_output.attention_mask = torch.tensor([[1, 1, 1]])
        mock_tokenizer = MagicMock(return_value=tok_output)
        mock_tokenizer.decode.return_value = ""
        engine._tokenizer = mock_tokenizer

        mock_model = MagicMock()
        mock_model.generate.return_value = torch.tensor([[1, 2, 3]])
        engine._model = mock_model

        result = await engine.predict("MERHABA")
        assert result.confidence == 0.0
        assert result.text == ""
