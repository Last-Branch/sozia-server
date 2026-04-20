"""GlossToTextEngine — gloss-to-natural-Turkish translation via LLM.

Takes a raw gloss string (e.g. ``"MERHABA DUNYA"``) produced by
TslRecognitionEngine and converts it to natural Turkish text
(e.g. ``"Merhaba dünya."``) using a Gemma-2-9B-it model fine-tuned with LoRA.

In the sign pipeline this is the *secondary* (slower) modality: the TSL
engine's gloss is emitted as PARTIAL, and this engine's output replaces it
as FINAL via ``replaces_segment_id``.

Latency budget: ≤ 2000 ms.
"""

from __future__ import annotations

import asyncio
import math
import re
import time
from typing import TYPE_CHECKING, Any

import torch

from sozia.common.interfaces import (
    InferenceEngine,
    InferenceTimeoutError,
    ModelNotLoadedError,
)
from sozia.common.models import ModalityResult, ModalityType

if TYPE_CHECKING:
    from sozia.common.models import ModelConfig

_DEFAULT_TIMEOUT_MS = 2000

# Best prompt strategy from sozia-research benchmarks (P3_EN, 87.19% success).
_DEFAULT_INSTRUCTION = (
    "Act as a system that translates the following lowercase and tagged Turkish Sign "
    "Language transcriptions into natural and understandable Turkish, considering the "
    "structure of sign language. Do not add any unnecessary words; just accurately "
    "express the message from the gloss in Turkish."
)


class GlossToTextEngine(InferenceEngine):
    """InferenceEngine for gloss → natural Turkish translation.

    Expected ``ModelConfig`` layout:
        weights_path: Path to the LoRA adapter directory (``final_adapter/``).
        device: ``"cpu"`` or ``"cuda"``.
        params (optional):
            base_model_id (str): HuggingFace model ID, default
                ``"google/gemma-2-9b-it"``.
            instruction (str): System prompt, default P3_EN.
            max_new_tokens (int): Generation limit, default ``128``.
            beam_size (int): Beam width, default ``5``.
            repetition_penalty (float): default ``1.15``.
            use_autocast (bool): Wrap generation in bfloat16 autocast,
                default ``True``.
            load_in_4bit (bool): Use BitsAndBytes 4-bit quantisation,
                default ``True``.
    """

    def __init__(self) -> None:
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._model_id: str = ""
        self._device: torch.device = torch.device("cpu")
        self._instruction: str = _DEFAULT_INSTRUCTION
        self._max_new_tokens: int = 128
        self._beam_size: int = 5
        self._repetition_penalty: float = 1.15
        self._use_autocast: bool = True
        self._base_model_id: str = "google/gemma-2-9b-it"

    # ------------------------------------------------------------------
    # InferenceEngine interface
    # ------------------------------------------------------------------

    async def load_model(self, config: ModelConfig) -> None:
        self._base_model_id = config.params.get(
            "base_model_id",
            "google/gemma-2-9b-it",
        )
        self._instruction = config.params.get("instruction", _DEFAULT_INSTRUCTION)
        self._max_new_tokens = config.params.get("max_new_tokens", 128)
        self._beam_size = config.params.get("beam_size", 5)
        self._repetition_penalty = config.params.get("repetition_penalty", 1.15)
        self._use_autocast = config.params.get("use_autocast", True)
        load_in_4bit = config.params.get("load_in_4bit", True)
        merged = config.params.get("merged", False)

        model, tokenizer = await asyncio.to_thread(
            self._load_model_sync,
            base_model_id=self._base_model_id,
            adapter_path=config.weights_path,
            device=config.device,
            load_in_4bit=load_in_4bit,
            merged=merged,
        )

        self._model = model
        self._tokenizer = tokenizer
        self._device = torch.device(config.device)
        self._model_id = config.model_id

    async def predict(
        self,
        features: str,
        timeout_ms: int = _DEFAULT_TIMEOUT_MS,
    ) -> ModalityResult:
        """Translate a gloss string into natural Turkish.

        Args:
            features: Raw gloss string (e.g. ``"MERHABA DUNYA"``).
            timeout_ms: Maximum wall-clock time in ms (default 2000).

        Returns:
            ModalityResult with natural Turkish text and confidence.
        """
        if self._model is None or self._tokenizer is None:
            raise ModelNotLoadedError("GlossToTextEngine: no model loaded.")

        gloss = _turkish_lower(str(features))
        prompt = _format_chat(self._base_model_id, self._instruction, gloss)

        start = time.perf_counter()

        try:
            raw_text, confidence = await asyncio.wait_for(
                asyncio.to_thread(self._generate, prompt),
                timeout=timeout_ms / 1000.0,
            )
        except asyncio.TimeoutError:
            raise InferenceTimeoutError(
                f"GlossToTextEngine exceeded {timeout_ms}ms budget."
            )

        latency_ms = int((time.perf_counter() - start) * 1000)
        text = _polish_turkish(raw_text)

        if not text:
            confidence = 0.0

        return ModalityResult(
            modality_type=ModalityType.GLOSS_TO_TEXT,
            text=text,
            confidence=confidence,
            timestamp_ms=0,
            duration_ms=0,
            inference_latency_ms=latency_ms,
        )

    async def unload_model(self) -> None:
        self._model = None
        self._tokenizer = None
        self._model_id = ""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def is_loaded(self) -> bool:
        return self._model is not None

    def get_model_id(self) -> str:
        return self._model_id

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_model_sync(
        base_model_id: str,
        adapter_path: str,
        device: str,
        load_in_4bit: bool,
        merged: bool = False,
    ) -> tuple:
        """Synchronous model + tokenizer loading (runs in thread)."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_path = adapter_path if merged else base_model_id
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token

        load_kwargs: dict[str, Any] = {
            "device_map": "auto" if device == "cuda" else None,
            "dtype": torch.bfloat16,
            "attn_implementation": "sdpa",
        }

        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)

        if not merged:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter_path)

        model.eval()
        return model, tokenizer

    def _generate(self, prompt: str) -> tuple[str, float]:
        """Synchronous generation — runs in thread."""
        inputs = self._tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(self._device)
        attention_mask = inputs.attention_mask.to(self._device)
        input_len = input_ids.shape[1]

        gen_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": self._max_new_tokens,
            "num_beams": self._beam_size,
            "repetition_penalty": self._repetition_penalty,
            "do_sample": False,
            "return_dict_in_generate": True,
            "output_scores": True,
        }

        with torch.no_grad():
            if self._use_autocast and self._device.type == "cuda":
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    outputs = self._model.generate(**gen_kwargs)
            else:
                outputs = self._model.generate(**gen_kwargs)

        new_tokens = outputs.sequences[0][input_len:]
        text = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        confidence = _extract_confidence(outputs)
        del outputs
        return text, confidence


# ---------------------------------------------------------------------------
# Confidence extraction (mirrors WhisperAsrEngine._extract_confidence)
# ---------------------------------------------------------------------------


def _extract_confidence(output) -> float:
    """Derive a [0, 1] confidence from beam generation output.

    Uses ``sequences_scores`` (length-normalised sum of log-probs) when
    available — the natural result of ``num_beams > 1`` with
    ``return_dict_in_generate=True, output_scores=True``.
    Falls back to ``exp(-1.0)`` ≈ 0.37 if the field is absent.
    """
    if hasattr(output, "sequences_scores") and output.sequences_scores is not None:
        score = float(output.sequences_scores[0])
    else:
        score = -1.0
    return max(0.0, min(1.0, math.exp(score)))


# ---------------------------------------------------------------------------
# Turkish text utilities (inlined from sozia-research/gloss_to_text/utils.py)
# ---------------------------------------------------------------------------


def _turkish_lower(text: str) -> str:
    """Lowercase with correct Turkish I/İ mappings."""
    return text.replace("İ", "i").replace("I", "ı").lower()


def _turkish_capitalize(text: str) -> str:
    if not text:
        return ""
    first = text[0]
    if first == "i":
        return "İ" + text[1:]
    if first == "ı":
        return "I" + text[1:]
    return first.upper() + text[1:]


def _polish_turkish(text: str) -> str:
    """Post-process model output for natural Turkish."""
    text = text.strip()
    if not text:
        return ""

    text = _turkish_capitalize(text)

    def _cap_match(m: re.Match) -> str:
        return m.group(1) + _turkish_capitalize(m.group(2))

    text = re.sub(r"([.!?]\s+)([a-zığüşöç])", _cap_match, text)

    def _cap_proper(m: re.Match) -> str:
        return _turkish_capitalize(m.group(0))

    text = re.sub(r"\b[a-zığüşöç]+'[a-zığüşöç]*\b", _cap_proper, text)

    if text[-1] not in ".!?":
        text += "."

    return text


def _format_chat(model_name: str, instruction: str, gloss: str) -> str:
    """Build a Gemma chat-template prompt."""
    m = model_name.lower()
    if "gemma" in m:
        return (
            f"<start_of_turn>user\n{instruction}\n\nGloss: {gloss}"
            f"<end_of_turn>\n<start_of_turn>model\n"
        )
    if "llama" in m:
        return (
            f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{instruction}\n\nGloss: {gloss}"
            f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    # ChatML (Trendyol etc.)
    return (
        f"<|im_start|>user\n{instruction}\n\nGloss: {gloss}"
        f"<|im_end|>\n<|im_start|>assistant\n"
    )
