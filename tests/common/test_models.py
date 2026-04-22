"""Unit tests for sozia.common.models — enums and DTOs.

Run with: pytest tests/common/test_models.py -v
"""

from __future__ import annotations

import pytest

from sozia.common.models import (
    AudioFeatureChunk,
    ErrorMessage,
    LandmarkFrame,
    ModalityPath,
    ModalityResult,
    ModalityType,
    ModelConfig,
    PipelineHealth,
    SegmentStatus,
    SessionState,
    SessionStatusMessage,
    TranscriptSegment,
)


# ===========================================================================
# Enums
# ===========================================================================


class TestSessionState:
    def test_all_values_present(self):
        names = {s.name for s in SessionState}
        assert names == {
            "IDLE",
            "INITIALIZING",
            "RUNNING",
            "PAUSED",
            "DEGRADED",
            "ERROR",
        }

    def test_string_values_match_names(self):
        for state in SessionState:
            assert state.value == state.name

    def test_iterable_count(self):
        assert len(list(SessionState)) == 6


class TestModalityPath:
    def test_all_values_present(self):
        assert {p.name for p in ModalityPath} == {"SPEECH", "SIGN"}

    def test_string_values_match_names(self):
        for path in ModalityPath:
            assert path.value == path.name


class TestModalityType:
    def test_all_values_present(self):
        assert {t.name for t in ModalityType} == {
            "ASR",
            "LIP_READING",
            "TSL_RECOGNITION",
            "GLOSS_TO_TEXT",
        }

    def test_string_values_match_names(self):
        for modality in ModalityType:
            assert modality.value == modality.name


class TestSegmentStatus:
    def test_all_values_present(self):
        assert {s.name for s in SegmentStatus} == {"PARTIAL", "FINAL"}

    def test_string_values_match_names(self):
        for status in SegmentStatus:
            assert status.value == status.name


# ===========================================================================
# ModelConfig
# ===========================================================================


class TestModelConfig:
    def test_happy_path(self):
        cfg = ModelConfig(
            model_id="whisper-small",
            weights_path="/models/whisper.pt",
            device="cuda",
            params={"language": "tr"},
        )
        assert cfg.model_id == "whisper-small"
        assert cfg.device == "cuda"

    def test_frozen(self):
        cfg = ModelConfig(model_id="m", weights_path="/p", device="cpu", params={})
        with pytest.raises(Exception):  # FrozenInstanceError
            cfg.model_id = "other"  # type: ignore[misc]

    def test_empty_params_accepted(self):
        cfg = ModelConfig(model_id="m", weights_path="/p", device="cpu", params={})
        assert cfg.params == {}

    def test_cpu_device_accepted(self):
        cfg = ModelConfig(model_id="m", weights_path="/p", device="cpu", params={})
        assert cfg.device == "cpu"

    def test_empty_model_id_raises(self):
        with pytest.raises(ValueError):
            ModelConfig(model_id="", weights_path="/p", device="cpu", params={})

    def test_empty_weights_path_raises(self):
        with pytest.raises(ValueError):
            ModelConfig(model_id="m", weights_path="", device="cpu", params={})

    def test_invalid_device_raises(self):
        with pytest.raises(ValueError):
            ModelConfig(model_id="m", weights_path="/p", device="gpu", params={})


# ===========================================================================
# ModalityResult
# ===========================================================================


class TestModalityResult:
    def _make(self, **kwargs):
        defaults = dict(
            modality_type=ModalityType.ASR,
            text="merhaba",
            confidence=0.9,
            timestamp_ms=100,
            duration_ms=500,
            inference_latency_ms=120,
        )
        defaults.update(kwargs)
        return ModalityResult(**defaults)

    def test_happy_path(self):
        r = self._make()
        assert r.text == "merhaba"
        assert r.confidence == 0.9

    def test_empty_text_allowed(self):
        r = self._make(text="")
        assert r.text == ""

    def test_frozen(self):
        r = self._make()
        with pytest.raises(Exception):
            r.text = "other"  # type: ignore[misc]

    def test_confidence_above_one_raises(self):
        with pytest.raises(ValueError):
            self._make(confidence=1.01)

    def test_confidence_below_zero_raises(self):
        with pytest.raises(ValueError):
            self._make(confidence=-0.01)

    def test_confidence_boundary_zero(self):
        r = self._make(confidence=0.0)
        assert r.confidence == 0.0

    def test_confidence_boundary_one(self):
        r = self._make(confidence=1.0)
        assert r.confidence == 1.0

    def test_negative_timestamp_raises(self):
        with pytest.raises(ValueError):
            self._make(timestamp_ms=-1)

    def test_negative_duration_raises(self):
        with pytest.raises(ValueError):
            self._make(duration_ms=-1)

    def test_negative_latency_raises(self):
        with pytest.raises(ValueError):
            self._make(inference_latency_ms=-1)


# ===========================================================================
# PipelineHealth
# ===========================================================================


class TestPipelineHealth:
    def _audio(self, **kwargs):
        defaults = dict(
            session_id="550e8400-e29b-41d4-a716-446655440000",
            pipeline="audio",
            available=True,
            fps=None,
            snr=12.5,
            face_detected=None,
            last_updated_ms=100,
        )
        defaults.update(kwargs)
        return PipelineHealth(**defaults)

    def _video(self, **kwargs):
        defaults = dict(
            session_id="550e8400-e29b-41d4-a716-446655440000",
            pipeline="video",
            available=True,
            fps=30.0,
            snr=None,
            face_detected=True,
            last_updated_ms=100,
        )
        defaults.update(kwargs)
        return PipelineHealth(**defaults)

    def test_audio_happy_path(self):
        h = self._audio()
        assert h.pipeline == "audio"
        assert h.snr == 12.5
        assert h.fps is None
        assert h.face_detected is None

    def test_video_happy_path(self):
        h = self._video()
        assert h.pipeline == "video"
        assert h.fps == 30.0
        assert h.snr is None

    def test_frozen(self):
        h = self._audio()
        with pytest.raises(Exception):
            h.available = False  # type: ignore[misc]

    def test_invalid_pipeline_raises(self):
        with pytest.raises(ValueError):
            self._audio(pipeline="camera")

    def test_negative_last_updated_raises(self):
        with pytest.raises(ValueError):
            self._audio(last_updated_ms=-1)

    def test_empty_session_id_raises(self):
        with pytest.raises(ValueError):
            self._audio(session_id="")

    # cross-field rules
    def test_audio_with_fps_raises(self):
        with pytest.raises(ValueError):
            self._audio(fps=30.0)

    def test_audio_with_face_detected_raises(self):
        with pytest.raises(ValueError):
            self._audio(face_detected=True)

    def test_video_with_snr_raises(self):
        with pytest.raises(ValueError):
            self._video(snr=10.0)


# ===========================================================================
# LandmarkFrame
# ===========================================================================


class TestLandmarkFrame:
    def _make(self, **kwargs):
        face = [[0.5, 0.5, 0.0]] * 83
        hand = [[0.5, 0.5, 0.0]] * 21
        pose = [[0.5, 0.5, 0.0, 0.9]] * 33
        defaults = dict(
            session_id="550e8400-e29b-41d4-a716-446655440000",
            timestamp_ms=0,
            face_landmarks=face,
            left_hand_landmarks=hand,
            right_hand_landmarks=hand,
            pose_landmarks=pose,
        )
        defaults.update(kwargs)
        return LandmarkFrame(**defaults)

    def test_happy_path_all_arrays(self):
        f = self._make()
        assert len(f.face_landmarks) == 83

    def test_happy_path_pose_only(self):
        f = self._make(
            face_landmarks=None, left_hand_landmarks=None, right_hand_landmarks=None
        )
        assert f.pose_landmarks is not None

    def test_all_none_raises(self):
        with pytest.raises(ValueError):
            self._make(
                face_landmarks=None,
                left_hand_landmarks=None,
                right_hand_landmarks=None,
                pose_landmarks=None,
            )

    def test_frozen(self):
        f = self._make()
        with pytest.raises(Exception):
            f.timestamp_ms = 999  # type: ignore[misc]

    def test_empty_session_id_raises(self):
        with pytest.raises(ValueError):
            self._make(session_id="")

    def test_negative_timestamp_raises(self):
        with pytest.raises(ValueError):
            self._make(timestamp_ms=-1)

    def test_face_wrong_length_raises(self):
        with pytest.raises(ValueError):
            self._make(face_landmarks=[[0.5, 0.5, 0.0]] * 10)

    def test_hand_wrong_length_raises(self):
        with pytest.raises(ValueError):
            self._make(left_hand_landmarks=[[0.5, 0.5, 0.0]] * 5)

    def test_pose_wrong_length_raises(self):
        with pytest.raises(ValueError):
            self._make(pose_landmarks=[[0.5, 0.5, 0.0, 0.9]] * 10)

    def test_pose_missing_visibility_raises(self):
        with pytest.raises(ValueError):
            self._make(pose_landmarks=[[0.5, 0.5, 0.0]] * 33)

    def test_point_wrong_dims_raises(self):
        bad = [[0.5, 0.5]] * 83  # missing z
        with pytest.raises(ValueError):
            self._make(face_landmarks=bad)

    def test_face_xy_out_of_range_allowed(self):
        # MediaPipe produces x/y outside [0, 1] when a landmark exits the frame.
        ok = [[1.5, 0.5, 0.0]] * 83
        f = self._make(face_landmarks=ok)
        assert f.face_landmarks[0][0] == 1.5

    def test_face_z_out_of_range_allowed(self):
        # z is MediaPipe relative depth — not constrained to [0, 1]
        ok = [[0.5, 0.5, -2.5]] * 83
        f = self._make(face_landmarks=ok)
        assert f.face_landmarks[0][2] == -2.5

    def test_hand_xy_out_of_range_allowed(self):
        # MediaPipe produces out-of-range coords when a hand exits the frame.
        ok = [[0.5, 1.2, 0.0]] * 21
        f = self._make(left_hand_landmarks=ok)
        assert f.left_hand_landmarks[0][1] == pytest.approx(1.2)


# ===========================================================================
# AudioFeatureChunk
# ===========================================================================


class TestAudioFeatureChunk:
    def _make(self, **kwargs):
        defaults = dict(
            session_id="550e8400-e29b-41d4-a716-446655440000",
            timestamp_ms=0,
            features=[[float(i) for i in range(13)] for _ in range(10)],
            feature_type="mfcc",
            sample_rate_hz=16000,
            chunk_duration_ms=500,
        )
        defaults.update(kwargs)
        return AudioFeatureChunk(**defaults)

    def test_happy_path(self):
        c = self._make()
        assert c.sample_rate_hz == 16000
        assert len(c.features) == 10

    def test_mel_spectrogram_accepted(self):
        c = self._make(feature_type="mel_spectrogram")
        assert c.feature_type == "mel_spectrogram"

    def test_frozen(self):
        c = self._make()
        with pytest.raises(Exception):
            c.sample_rate_hz = 8000  # type: ignore[misc]

    def test_empty_session_id_raises(self):
        with pytest.raises(ValueError):
            self._make(session_id="")

    def test_empty_features_raises(self):
        with pytest.raises(ValueError):
            self._make(features=[])

    def test_empty_feature_row_raises(self):
        with pytest.raises(ValueError):
            self._make(features=[[]])

    def test_ragged_features_raises(self):
        with pytest.raises(ValueError):
            self._make(features=[[1.0, 2.0], [1.0]])

    def test_invalid_feature_type_raises(self):
        with pytest.raises(ValueError):
            self._make(feature_type="raw_pcm")

    def test_zero_sample_rate_raises(self):
        with pytest.raises(ValueError):
            self._make(sample_rate_hz=0)

    def test_negative_sample_rate_raises(self):
        with pytest.raises(ValueError):
            self._make(sample_rate_hz=-1)

    def test_zero_chunk_duration_raises(self):
        with pytest.raises(ValueError):
            self._make(chunk_duration_ms=0)


# ===========================================================================
# TranscriptSegment
# ===========================================================================


class TestTranscriptSegment:
    def _make(self, **kwargs):
        defaults = dict(
            segment_id="6ba7b810-9dad-11d1-80b4-00c04fd430c8",
            session_id="550e8400-e29b-41d4-a716-446655440000",
            status=SegmentStatus.PARTIAL,
            text="merhaba",
            source=ModalityType.ASR,
            confidence=0.85,
            timestamp_ms=0,
            duration_ms=500,
            created_at_ms=1_700_000_000_000,
            replaces_segment_id=None,
        )
        defaults.update(kwargs)
        return TranscriptSegment(**defaults)

    def test_happy_path_partial(self):
        s = self._make()
        assert s.status == SegmentStatus.PARTIAL
        assert s.replaces_segment_id is None

    def test_happy_path_final_with_replaces(self):
        s = self._make(
            status=SegmentStatus.FINAL,
            replaces_segment_id="550e8400-e29b-41d4-a716-446655440000",
        )
        assert s.status == SegmentStatus.FINAL
        assert s.replaces_segment_id is not None

    def test_final_without_replaces_allowed(self):
        s = self._make(status=SegmentStatus.FINAL, replaces_segment_id=None)
        assert s.status == SegmentStatus.FINAL

    def test_frozen(self):
        s = self._make()
        with pytest.raises(Exception):
            s.text = "other"  # type: ignore[misc]

    def test_empty_segment_id_raises(self):
        with pytest.raises(ValueError):
            self._make(segment_id="")

    def test_empty_session_id_raises(self):
        with pytest.raises(ValueError):
            self._make(session_id="")

    def test_confidence_out_of_range_raises(self):
        with pytest.raises(ValueError):
            self._make(confidence=1.5)

    def test_negative_timestamp_raises(self):
        with pytest.raises(ValueError):
            self._make(timestamp_ms=-1)

    def test_negative_duration_raises(self):
        with pytest.raises(ValueError):
            self._make(duration_ms=-1)

    def test_large_created_at_ms_accepted(self):
        # Unix epoch in ms — should be a large integer
        s = self._make(created_at_ms=1_700_000_000_000)
        assert s.created_at_ms == 1_700_000_000_000

    def test_empty_text_allowed(self):
        s = self._make(text="")
        assert s.text == ""


# ===========================================================================
# SessionStatusMessage
# ===========================================================================


class TestSessionStatusMessage:
    def _make(self, **kwargs):
        defaults = dict(
            session_id="550e8400-e29b-41d4-a716-446655440000",
            state=SessionState.INITIALIZING,
            message="Loading speech models...",
        )
        defaults.update(kwargs)
        return SessionStatusMessage(**defaults)

    def test_happy_path(self):
        msg = self._make()
        assert msg.session_id == "550e8400-e29b-41d4-a716-446655440000"
        assert msg.state == SessionState.INITIALIZING
        assert msg.message == "Loading speech models..."

    def test_all_session_states_accepted(self):
        for state in SessionState:
            msg = self._make(state=state)
            assert msg.state == state

    def test_empty_message_allowed(self):
        msg = self._make(message="")
        assert msg.message == ""

    def test_frozen(self):
        msg = self._make()
        with pytest.raises(Exception):
            msg.state = SessionState.RUNNING  # type: ignore[misc]

    def test_empty_session_id_raises(self):
        with pytest.raises(ValueError):
            self._make(session_id="")

    def test_running_state(self):
        msg = self._make(state=SessionState.RUNNING, message="Session active.")
        assert msg.state == SessionState.RUNNING

    def test_degraded_state(self):
        msg = self._make(
            state=SessionState.DEGRADED,
            message="Camera obstructed — visual recognition paused.",
        )
        assert msg.state == SessionState.DEGRADED

    def test_error_state(self):
        msg = self._make(state=SessionState.ERROR, message="Inference engine failed.")
        assert msg.state == SessionState.ERROR


# ===========================================================================
# ErrorMessage
# ===========================================================================


class TestErrorMessage:
    def _make(self, **kwargs):
        defaults = dict(
            session_id="550e8400-e29b-41d4-a716-446655440000",
            code=4001,
            message="Authentication failed.",
        )
        defaults.update(kwargs)
        return ErrorMessage(**defaults)

    def test_happy_path(self):
        err = self._make()
        assert err.code == 4001
        assert err.message == "Authentication failed."

    def test_all_valid_codes_accepted(self):
        for code in (4001, 4002, 4003, 4004):
            err = self._make(code=code)
            assert err.code == code

    def test_frozen(self):
        err = self._make()
        with pytest.raises(Exception):
            err.code = 4002  # type: ignore[misc]

    def test_invalid_code_raises(self):
        with pytest.raises(ValueError):
            self._make(code=4000)

    def test_code_above_range_raises(self):
        with pytest.raises(ValueError):
            self._make(code=4005)

    def test_standard_ws_code_raises(self):
        with pytest.raises(ValueError):
            self._make(code=1000)

    def test_empty_message_raises(self):
        with pytest.raises(ValueError):
            self._make(message="")

    def test_empty_session_id_allowed(self):
        # session_id may be empty when the connection failed before
        # a session_id was established (e.g. auth failure on first connect)
        err = self._make(session_id="", code=4001)
        assert err.session_id == ""

    def test_code_4002_bad_init(self):
        err = self._make(code=4002, message="Missing modality_path in session_init.")
        assert err.code == 4002

    def test_code_4003_duplicate_session(self):
        err = self._make(code=4003, message="Session ID already active.")
        assert err.code == 4003

    def test_code_4004_warmup_failed(self):
        err = self._make(code=4004, message="Model warm-up failed.")
        assert err.code == 4004
