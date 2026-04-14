"""Tests for sozia.server config helpers."""

from __future__ import annotations

import pytest

from sozia.server import (
    ENV_GLOSS_WEIGHTS,
    ENV_LIP_WEIGHTS,
    ENV_TSL_SCALER,
    ENV_TSL_WEIGHTS,
    ENV_WHISPER_WEIGHTS,
    _sign_configs,
    _speech_configs,
)


class TestSpeechConfigs:
    def test_both_none_when_env_unset(self, monkeypatch):
        monkeypatch.delenv(ENV_WHISPER_WEIGHTS, raising=False)
        monkeypatch.delenv(ENV_LIP_WEIGHTS, raising=False)

        asr, lip = _speech_configs("cpu")
        assert asr is None
        assert lip is None

    def test_asr_config_built_from_env(self, monkeypatch):
        monkeypatch.setenv(ENV_WHISPER_WEIGHTS, "/models/whisper.pt")
        monkeypatch.delenv(ENV_LIP_WEIGHTS, raising=False)

        asr, lip = _speech_configs("cpu")
        assert asr is not None
        assert asr.weights_path == "/models/whisper.pt"
        assert asr.device == "cpu"
        assert asr.model_id == "whisper-small-tr"
        assert lip is None

    def test_lip_config_built_from_env(self, monkeypatch):
        monkeypatch.delenv(ENV_WHISPER_WEIGHTS, raising=False)
        monkeypatch.setenv(ENV_LIP_WEIGHTS, "/models/lip.pt")

        asr, lip = _speech_configs("cuda")
        assert asr is None
        assert lip is not None
        assert lip.weights_path == "/models/lip.pt"
        assert lip.device == "cuda"

    def test_both_configs_when_both_set(self, monkeypatch):
        monkeypatch.setenv(ENV_WHISPER_WEIGHTS, "/models/whisper.pt")
        monkeypatch.setenv(ENV_LIP_WEIGHTS, "/models/lip.pt")

        asr, lip = _speech_configs("cpu")
        assert asr is not None
        assert lip is not None


class TestSignConfigs:
    def test_both_none_when_env_unset(self, monkeypatch):
        monkeypatch.delenv(ENV_TSL_WEIGHTS, raising=False)
        monkeypatch.delenv(ENV_TSL_SCALER, raising=False)
        monkeypatch.delenv(ENV_GLOSS_WEIGHTS, raising=False)

        tsl, gloss = _sign_configs("cpu")
        assert tsl is None
        assert gloss is None

    def test_tsl_config_without_scaler(self, monkeypatch):
        monkeypatch.setenv(ENV_TSL_WEIGHTS, "/models/run_autsl")
        monkeypatch.delenv(ENV_TSL_SCALER, raising=False)
        monkeypatch.delenv(ENV_GLOSS_WEIGHTS, raising=False)

        tsl, gloss = _sign_configs("cpu")
        assert tsl is not None
        assert tsl.weights_path == "/models/run_autsl"
        assert tsl.params == {}
        assert gloss is None

    def test_tsl_config_with_scaler(self, monkeypatch):
        monkeypatch.setenv(ENV_TSL_WEIGHTS, "/models/run_autsl")
        monkeypatch.setenv(ENV_TSL_SCALER, "/models/scalers/AUTSL/scaler_signer.pkl")
        monkeypatch.delenv(ENV_GLOSS_WEIGHTS, raising=False)

        tsl, _ = _sign_configs("cpu")
        assert tsl is not None
        assert tsl.params["scaler_path"] == "/models/scalers/AUTSL/scaler_signer.pkl"

    def test_gloss_config_built_from_env(self, monkeypatch):
        monkeypatch.delenv(ENV_TSL_WEIGHTS, raising=False)
        monkeypatch.delenv(ENV_TSL_SCALER, raising=False)
        monkeypatch.setenv(ENV_GLOSS_WEIGHTS, "/models/gemma-9b-gloss-tr")

        tsl, gloss = _sign_configs("cuda")
        assert tsl is None
        assert gloss is not None
        assert gloss.weights_path == "/models/gemma-9b-gloss-tr"
        assert gloss.device == "cuda"

    def test_both_configs_when_both_set(self, monkeypatch):
        monkeypatch.setenv(ENV_TSL_WEIGHTS, "/models/run_autsl")
        monkeypatch.setenv(ENV_TSL_SCALER, "/models/scalers/AUTSL/scaler_signer.pkl")
        monkeypatch.setenv(ENV_GLOSS_WEIGHTS, "/models/gemma-9b-gloss-tr")

        tsl, gloss = _sign_configs("cuda")
        assert tsl is not None
        assert gloss is not None
        assert tsl.params["scaler_path"] == "/models/scalers/AUTSL/scaler_signer.pkl"
