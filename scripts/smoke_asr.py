"""ASR smoke test — verifies WhisperAsrEngine end-to-end with real weights.

Runs three probes:
  1. Silence  → engine must return empty text, confidence near 0.
  2. Reference mel from WhisperFeatureExtractor on a synthetic 440 Hz tone
     → engine must return a ModalityResult without crashing.
  3. If --audio is given, transcribes a real WAV and prints the result.

Usage on remote:
    conda activate sozia-server
    python scripts/smoke_asr.py --weights /home/monica/mehmet-altintas/Sozia/weights/whisper-large-v3-turbo
    python scripts/smoke_asr.py --weights /path/to/weights --audio /path/to/speech.wav
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time

import numpy as np


def _make_silence_mel(n_mels: int = 128, t_frames: int = 200) -> np.ndarray:
    """Log10-mel array that looks like silence to the energy VAD (all values below -4)."""
    return np.full((t_frames, n_mels), -10.0, dtype=np.float32)


def _make_tone_mel(weights_path: str, n_mels: int = 128, duration_s: float = 3.0, sample_rate: int = 16_000) -> np.ndarray:
    """Compute log10-mel from a 440 Hz sine wave using WhisperFeatureExtractor.

    This is the ground-truth mel pipeline — identical to what Whisper was trained on.
    If the server pipeline is correct, feeding this array into WhisperAsrEngine should
    not crash and should produce a valid (if nonsensical) ModalityResult.
    """
    from transformers import WhisperFeatureExtractor

    samples = np.sin(2 * np.pi * 440 * np.arange(int(duration_s * sample_rate)) / sample_rate).astype(np.float32)

    extractor = WhisperFeatureExtractor.from_pretrained(weights_path)
    # Returns (1, n_mels, 3000) as a tensor-like object; grab the numpy array.
    result = extractor(samples, sampling_rate=sample_rate, return_tensors="np")
    mel_nd = result.input_features[0]  # shape (n_mels, 3000) — already normalised

    # Convert back to raw (un-normalised) log10-mel so _prepare_mel can apply its own
    # normalisation, matching the client's output.  Reverse: (x * 4.0) - 4.0
    raw = mel_nd * 4.0 - 4.0  # (n_mels, 3000)
    return raw.T  # (3000, n_mels) — client sends (T, N)


def _make_wav_mel(wav_path: str, weights_path: str, n_mels: int = 128, sample_rate: int = 16_000) -> np.ndarray:
    """Load a WAV file, compute mel via WhisperFeatureExtractor, return (T, N)."""
    import soundfile as sf
    from transformers import WhisperFeatureExtractor

    audio, sr = sf.read(wav_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != sample_rate:
        raise ValueError(f"WAV sample rate {sr} != {sample_rate}. Resample first.")

    extractor = WhisperFeatureExtractor.from_pretrained(weights_path)
    result = extractor(audio, sampling_rate=sample_rate, return_tensors="np")
    mel_nd = result.input_features[0]  # (n_mels, 3000)
    raw = mel_nd * 4.0 - 4.0
    return raw.T  # (T, N)


async def run(weights_path: str, audio_path: str | None) -> None:
    import torch
    from sozia.common.models import ModelConfig
    from sozia.server.inference.speech.whisper_asr_engine import WhisperAsrEngine

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    engine = WhisperAsrEngine()
    config = ModelConfig(
        model_id="whisper-smoke-test",
        weights_path=weights_path,
        device=device,
        params={"language": "tr", "task": "transcribe", "num_beams": 1},
    )

    print("Loading model …")
    t0 = time.perf_counter()
    await engine.load_model(config)
    print(f"  Loaded in {(time.perf_counter() - t0)*1000:.0f} ms")

    n_mels = engine._n_mels
    print(f"  n_mels from model config: {n_mels}")

    # --- Probe 1: silence ---------------------------------------------------
    print("\n[1] Silence probe")
    silence = _make_silence_mel(n_mels=n_mels)
    result = await engine.predict(silence)
    print(f"  text='{result.text}'  confidence={result.confidence:.4f}  latency={result.inference_latency_ms} ms")
    if result.text != "":
        print("  WARN: expected empty text for silence — possible VAD threshold issue")
    else:
        print("  OK: returned empty text as expected")

    # --- Probe 2: synthetic tone mel ----------------------------------------
    print("\n[2] Synthetic tone probe (440 Hz, 3 s via WhisperFeatureExtractor)")
    tone_mel = _make_tone_mel(weights_path, n_mels=n_mels)
    result = await engine.predict(tone_mel, timeout_ms=10_000)
    print(f"  text='{result.text}'  confidence={result.confidence:.4f}  latency={result.inference_latency_ms} ms")
    if result.confidence > 0.0 and result.confidence < 0.99:
        print("  OK: confidence is non-trivial (output_scores=True is working)")
    elif result.confidence == 0.0 and result.text == "":
        print("  OK (suppressed as silence/noise)")
    else:
        print(f"  NOTE: confidence={result.confidence:.4f} — check _extract_confidence")

    # --- Probe 3: real audio ------------------------------------------------
    if audio_path:
        print(f"\n[3] Real audio probe: {audio_path}")
        wav_mel = _make_wav_mel(audio_path, weights_path, n_mels=n_mels)
        result = await engine.predict(wav_mel, timeout_ms=15_000)
        print(f"  text='{result.text}'  confidence={result.confidence:.4f}  latency={result.inference_latency_ms} ms")

    await engine.unload_model()
    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smoke-test WhisperAsrEngine with real weights.")
    parser.add_argument("--weights", required=True, help="Path to model directory (config.json + safetensors)")
    parser.add_argument("--audio", default=None, help="Optional WAV file (16 kHz mono) for a real transcription test")
    args = parser.parse_args()

    asyncio.run(run(args.weights, args.audio))
