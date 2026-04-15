"""sozia.inference.speech — speech-path inference engines."""

from sozia.server.inference.speech.lip_reading_engine import LipReadingEngine
from sozia.server.inference.speech.whisper_asr_engine import WhisperAsrEngine

__all__ = ["WhisperAsrEngine", "LipReadingEngine"]
