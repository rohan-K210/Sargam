# Sargam ML Pipeline

A modular Python pipeline for analyzing Indian classical music audio and producing structured Sargam-oriented note information.

The pipeline is organized into six sequential stages:

1. **Pitch and tonic detection** — estimates the pitch contour, detects or accepts the tonic (Sa), and maps pitches to swaras.
2. **Timing** — detects note onsets, segments note events and estimates tempo/laya.
3. **Gamaka analysis** — examines pitch contours for ornaments such as meend, andolan, gamak and kan using a hybrid rule/model approach.
4. **Dynamics** — estimates relative note dynamics and attack characteristics.
5. **Silence and breath analysis** — classifies gaps and identifies possible breath events.
6. **Timbre classification** — estimates instrument class using a downloaded model when available, with a heuristic fallback.

This directory is the modular ML implementation. The separate browser-facing Flask application is in [`../sargam_backend/`](../sargam_backend/). See the [repository overview](../README.md) for the distinction between the two components.

## Requirements

- Python 3.9 or later
- PyTorch and torchaudio compatible with your Python and platform
- The packages listed in `requirements.txt`

## Setup

From this directory:

```bash
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

For development and tests:

```bash
pip install -r requirements-dev.txt
pytest
```

## Run the pipeline

Run the pipeline on a local audio file:

```python
from src.pipeline.orchestrator import create_pipeline

pipeline = create_pipeline()
result = pipeline.run("path/to/audio.wav")

if result["success"]:
    print(result["data"])
else:
    print(result["error"])
```

The pipeline accepts an optional tonic frequency. Source separation can be requested with `separate_source=True`; this uses the optional HDemucs implementation and falls back to the original audio if separation cannot be performed.

```python
result = pipeline.run(
    "path/to/audio.wav",
    tonic_hz=261.63,
    separate_source=True,
)
```

## Local Flask API

Start the API from this directory:

```bash
python -m api.flask_app
```

The server listens on port `5000` by default.

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/run/predict-local` | Analyze base64-encoded audio using the local pipeline |

Example JSON request:

```json
{
  "audio_base64": "<base64-encoded audio bytes>",
  "tonic_hz": 261.63,
  "return_student_view": true
}
```

The response contains structured analysis including tonic, tempo, instrument estimate, note events, breath information and (by default) a simplified student view.

## Configuration and model files

- `config/model_config.yaml` — audio preprocessing and model/pipeline settings.
- `config/swara_mapping.yaml` — swara mapping configuration.
- `config/dataset_config.yaml` — Saraga dataset download, annotation and split settings.
- `data/` — dataset download and preparation utilities.
- The gamaka and timbre stages can load model artifacts from the configured Hugging Face repository (`RK14x86/sargam-models`). If an artifact cannot be loaded, the gamaka stage uses rule-based analysis and the timbre stage uses a heuristic fallback.
- HDemucs source separation is optional and requires compatible PyTorch/torchaudio support and model availability.

## Tests

Run the test suite from this directory:

```bash
pytest
```

## Limitations

This is an experimental music-analysis pipeline. Results can be affected by audio quality, overlapping instruments, expressive pitch movement and model availability. Treat generated note and ornament estimates as analysis suggestions that require musical review.
