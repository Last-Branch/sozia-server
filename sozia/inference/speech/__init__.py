"""sozia.inference.speech — speech-path inference engines."""

from sozia.inference.speech.lip_reading_engine import LipReadingEngine
from sozia.inference.speech.whisper_asr_engine import WhisperAsrEngine

__all__ = ["WhisperAsrEngine", "LipReadingEngine"]
