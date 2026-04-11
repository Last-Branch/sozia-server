# sozia-server

Backend for the Sozia transcription system. Takes anonymized features from the client, runs inference, and streams transcript segments back over WebSocket.

---

## Requirements

- Python 3.11+
- conda (recommended) or a standard venv

---

## Installation

```bash
conda create -n sozia-server python=3.11
conda activate sozia-server
pip install -e ".[dev]"
```

---

## Configuration

All config comes from environment variables.

| Variable | Required | Default | Description |
|---|---|---|---|
| `SOZIA_API_KEY` | yes | — | Secret included in every `session_init` payload. Use something random in production. |
| `SOZIA_DEVICE` | no | `cpu` | `"cpu"` or `"cuda"`. |
| `SOZIA_WHISPER_WEIGHTS` | no | — | Path to Whisper ASR weights. Speech path is disabled if unset. |
| `SOZIA_LIP_WEIGHTS` | no | — | Path to lip-reading weights. Optional — speech path works without it. |
| `SOZIA_TSL_WEIGHTS` | no | — | Path to TSL recognition weights. Sign path is disabled if unset. |
| `SOZIA_GLOSS_WEIGHTS` | no | — | Path to Gemma-9B LoRA weights. Sign path falls back to raw gloss if unset. |

### Local development

Create a `.env` in the project root (already in `.gitignore`):

```dotenv
SOZIA_API_KEY=dev-only-key-change-in-production
SOZIA_DEVICE=cpu

# Omit these if you don't have weights locally
# SOZIA_WHISPER_WEIGHTS=/path/to/whisper-small-tr.pt
# SOZIA_LIP_WEIGHTS=/path/to/lip-reading-v1.pt
# SOZIA_TSL_WEIGHTS=/path/to/tsl-gru-v1.pt
# SOZIA_GLOSS_WEIGHTS=/path/to/gemma-9b-gloss-tr/
```

Load it before starting:

```bash
export $(grep -v '^#' .env | xargs)
```

Or add `python-dotenv` as a dev dependency.

---

## Running the server

```bash
uvicorn sozia.server:app --host 0.0.0.0 --port 8000 --reload
```

On a single GPU, keep workers at 1 — multiple workers each try to load the full model set and will run out of memory:

```bash
uvicorn sozia.server:app --host 0.0.0.0 --port 8000 --workers 1
```

---

## WebSocket protocol

Connect to `ws://<host>:8000/ws`.

### session_init

First message from the client:

```json
{
  "type": "session_init",
  "session_id": "<uuid-v4>",
  "modality_path": "SPEECH",
  "api_key": "<SOZIA_API_KEY>"
}
```

`modality_path` is `"SPEECH"` or `"SIGN"`.

### Status frames

The server sends two status frames after init:

```json
{ "type": "session_status", "session_id": "...", "state": "INITIALIZING", "message": "" }
{ "type": "session_status", "session_id": "...", "state": "RUNNING",       "message": "" }
```

### Feature messages from client

LandmarkFrame (SIGN path, or face cache for SPEECH):
```json
{
  "type": "landmark_frame",
  "session_id": "...",
  "timestamp_ms": 1000,
  "face_landmarks": [[x,y,z], ...],
  "left_hand_landmarks": null,
  "right_hand_landmarks": null,
  "pose_landmarks": null
}
```

AudioFeatureChunk (SPEECH path):
```json
{
  "type": "audio_feature_chunk",
  "session_id": "...",
  "timestamp_ms": 2000,
  "features": [[...], ...],
  "feature_type": "mfcc",
  "sample_rate_hz": 16000,
  "chunk_duration_ms": 500
}
```

PipelineHealth (periodic, either path):
```json
{
  "type": "pipeline_health",
  "session_id": "...",
  "pipeline": "audio",
  "available": true,
  "fps": null,
  "snr": 18.5,
  "face_detected": null,
  "last_updated_ms": 1500
}
```

### TranscriptSegment from server

```json
{
  "type": "transcript_segment",
  "segment_id": "<uuid>",
  "session_id": "...",
  "status": "PARTIAL",
  "text": "merhaba dünya",
  "source": "ASR",
  "confidence": 0.87,
  "timestamp_ms": 1000,
  "duration_ms": 500,
  "created_at_ms": 1700000000000,
  "replaces_segment_id": null
}
```

### Ending a session

```json
{ "type": "session_end" }
```

### Close codes

| Code | Meaning |
|------|---------|
| 4001 | Auth failed — bad or missing `api_key` |
| 4002 | Malformed `session_init` |
| 4003 | Duplicate `session_id` |
| 4004 | Model warm-up failed |

---

## Development

```bash
pytest              # run tests with coverage
pytest tests/api/   # API layer only
mypy sozia/         # type check
```

---

## Model weights

Weights aren't in this repo. Get them from the shared Google Drive or the GitHub release assets in `sozia-research`, then set the env vars to the local paths.

A `download_models.sh` script is coming.
