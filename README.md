# sozia-server

Cloud backend for the Sozia real-time multimodal transcription system.
Receives anonymized features from the client, runs AI inference, fuses results,
and streams `TranscriptSegment` objects back over WebSocket.

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

All runtime configuration is read from environment variables at server startup.

| Variable | Required | Default | Description |
|---|---|---|---|
| `SOZIA_API_KEY` | **Yes** | — | Shared secret clients include in every `session_init` payload. Use a strong random string in production. |
| `SOZIA_DEVICE` | No | `cpu` | Compute device for inference: `"cpu"` or `"cuda"`. |
| `SOZIA_WHISPER_WEIGHTS` | No | — | Absolute path to Whisper ASR weight file. Speech path disabled if unset. |
| `SOZIA_LIP_WEIGHTS` | No | — | Absolute path to LipReading weight file. Supplementary — speech path continues without it. |
| `SOZIA_TSL_WEIGHTS` | No | — | Absolute path to TSL recognition GRU weight file. Sign path disabled if unset. |
| `SOZIA_GLOSS_WEIGHTS` | No | — | Absolute path to Gemma-9B LoRA weight directory. Sign path continues (with raw gloss fallback) without it. |

### Local development — `.env` file

Create a `.env` file in the project root (already in `.gitignore`):

```dotenv
SOZIA_API_KEY=dev-only-key-change-in-production
SOZIA_DEVICE=cpu

# Optional — omit if you don't have model weights locally
# SOZIA_WHISPER_WEIGHTS=/path/to/whisper-small-tr.pt
# SOZIA_LIP_WEIGHTS=/path/to/lip-reading-v1.pt
# SOZIA_TSL_WEIGHTS=/path/to/tsl-gru-v1.pt
# SOZIA_GLOSS_WEIGHTS=/path/to/gemma-9b-gloss-tr/
```

Load it before starting the server:

```bash
export $(grep -v '^#' .env | xargs)
```

Or use `python-dotenv` if you add it as a dev dependency.

---

## Running the server

```bash
uvicorn sozia.server:app --host 0.0.0.0 --port 8000 --reload
```

For production (multi-worker with GPU):

```bash
uvicorn sozia.server:app --host 0.0.0.0 --port 8000 --workers 1
```

> **Note:** Use `--workers 1` when running on a single GPU. Multiple workers each
> try to load the full model set, which will exhaust GPU memory.

---

## WebSocket protocol

Connect to `ws://<host>:8000/ws`.

### 1. session_init (first message from client)

```json
{
  "type": "session_init",
  "session_id": "<uuid-v4>",
  "modality_path": "SPEECH",
  "api_key": "<SOZIA_API_KEY>"
}
```

`modality_path` is either `"SPEECH"` or `"SIGN"`.

### 2. Status messages from server

After `session_init`, the server sends `session_status` frames:

```json
{ "type": "session_status", "session_id": "...", "state": "INITIALIZING", "message": "" }
{ "type": "session_status", "session_id": "...", "state": "RUNNING",       "message": "" }
```

### 3. Feature messages from client

**LandmarkFrame** (for SIGN path, or face cache for SPEECH):
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

**AudioFeatureChunk** (for SPEECH path):
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

**PipelineHealth** (periodic, either path):
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

### 4. TranscriptSegment from server

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

### 5. Ending a session

```json
{ "type": "session_end" }
```

### Close codes

| Code | Meaning |
|------|---------|
| 4001 | Auth failed — bad or missing `api_key` |
| 4002 | Malformed `session_init` payload |
| 4003 | Duplicate `session_id` — already active |
| 4004 | Model warm-up failed (server error) |

---

## Development

```bash
# Run tests with coverage
pytest

# Run only the API layer tests
pytest tests/api/

# Type check
mypy sozia/
```

---

## Model weights

Weights are **not committed to this repository**. Download them from the shared
Google Drive or GitHub Release assets in `sozia-research`, then point the env
vars at the local paths.

A `download_models.sh` helper script will be added in a future release.
