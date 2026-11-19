# Sargam 🎵

**Audio-to-Sargam notation for Indian classical music.**

Sargam is an experimental project exploring how audio analysis and machine learning can help musicians turn recorded melodies into Indian classical note notation (Sa, Re, Ga, Ma, Pa, Dha, Ni).

The repository currently contains two related Python implementations:

| Component | Purpose |
|---|---|
| [`sargam_backend/`](sargam_backend/) | Browser-based Sargam Sheet Generator: upload audio, analyze melody, generate notation and play synthesized output |
| [`sargam/`](sargam/) | Modular six-stage ML pipeline with a local Flask API and student-oriented analysis output |

These are separate implementations with different entry points and dependency sets. The sections below explain how to run each one.

## Sargam Sheet Generator

The web application in `sargam_backend/` accepts an audio recording and produces a timed Sargam sheet.

### Features

- Upload audio files in MP3, WAV, M4A or FLAC format (50 MB upload limit).
- Estimate the tonic (Sa), inspect candidate tonic values and optionally provide a manual Sa frequency.
- Detect melody pitches and map them to Sargam swaras, including komal and tivra variants.
- Estimate note timing, phrases, beats and downbeats.
- Detect ornaments such as meend and gamak.
- Optionally separate vocals and accompaniment with Spleeter when the dependency is available.
- Optionally use Gemini 2.5 Flash to refine the time-block analysis.
- Enter song lyrics manually and align the supplied text with the generated notation. **The application does not automatically transcribe lyrics from audio.**
- Play notation blocks or the generated sequence using browser-based bansuri synthesis.
- Optionally generate sung Sargam audio through Gemini's audio generation integration.

Some analysis features are optional and depend on installed libraries or API availability. The application reports whether Spleeter and madmom are available; beat tracking has a librosa fallback.

### Run the web application locally

Python 3.10 is the recommended environment for the backend's optional audio dependencies.

```bash
cd sargam_backend

python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
python app.py
```

Open [http://localhost:7860](http://localhost:7860).

The Dockerfile in this directory installs the system audio utilities and starts the Flask application:

```bash
cd sargam_backend
docker build -t sargam-sheet-generator .
docker run --rm -p 7860:7860 -e GEMINI_API_KEY=your_key sargam-sheet-generator
```

The Gemini key is optional. Without it, Gemini-enhanced analysis and singing generation are unavailable; the non-Gemini analysis path remains available.

### Backend configuration

| Variable | Required | Purpose |
|---|---:|---|
| `GEMINI_API_KEY` | No | Enables Gemini-enhanced analysis and Sargam singing generation |
| `PORT` | No | HTTP port; defaults to `7860` |
| `FLASK_DEBUG` | No | Flask debug mode; defaults to `true` in the current application |

### HTTP API

The Flask application exposes these routes:

| Method | Route | Description |
|---|---|---|
| GET | `/api/health` | Health status and optional dependency availability |
| GET | `/api/config` | Public runtime configuration and upload/duration limits |
| POST | `/api/sa-candidates` | Detect candidate tonic (Sa) values from an uploaded audio file |
| POST | `/api/analyze` | Analyze uploaded audio and return Sargam notation and timing blocks |
| POST | `/api/sing` | Generate sung audio for a Sargam passage using Gemini |
| POST | `/api/sing-all` | Generate sung audio for multiple Sargam lines |

Audio-analysis endpoints use `multipart/form-data` with an `audio` file. The `/api/analyze` endpoint also accepts optional `song_name`, `lyrics`, `sa_hz` and `use_gemini` fields. See the route docstrings in [`sargam_backend/app.py`](sargam_backend/app.py) for request and response details.

## Modular ML pipeline

The separate implementation in `sargam/` is organized as a six-stage pipeline:

1. **Pitch and tonic:** estimate pitch, detect the tonic and map the pitch contour to swaras.
2. **Timing:** detect onsets, segment notes and estimate tempo/laya.
3. **Gamaka:** analyze pitch contours for ornaments such as meend, andolan, gamak and kan.
4. **Dynamics:** estimate note dynamics and attack characteristics.
5. **Silence and breath:** classify gaps and identify possible breath events.
6. **Timbre:** estimate the instrument class using a trained model when available, with a heuristic fallback.

The pipeline also contains audio preprocessing utilities, configurable YAML settings, swara mapping, optional HDemucs-based source separation and tests.

### Run the ML pipeline

```bash
cd sargam

python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

Run the pipeline on an audio file:

```bash
python -c "from src.pipeline.orchestrator import create_pipeline; p = create_pipeline(); print(p.run('path/to/audio.wav'))"
```

To request melodic-source isolation where the optional HDemucs dependencies and model are available:

```python
from src.pipeline.orchestrator import create_pipeline

pipeline = create_pipeline()
result = pipeline.run("path/to/audio.wav", separate_source=True)
print(result)
```

### Local Flask API

Start the API from inside the `sargam/` directory:

```bash
python -m api.flask_app
```

The API listens on port `5000` by default.

| Method | Route | Description |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/run/predict-local` | Run the local ML pipeline on base64-encoded audio |

Example request body for `/run/predict-local`:

```json
{
  "audio_base64": "<base64-encoded audio bytes>",
  "tonic_hz": 261.63,
  "return_student_view": true
}
```

The response includes the detected tonic, tempo, instrument estimate, note events, breath information and an optional simplified student view.

### Models and configuration

- Pipeline settings are in [`sargam/config/model_config.yaml`](sargam/config/model_config.yaml).
- Swara definitions are in [`sargam/config/swara_mapping.yaml`](sargam/config/swara_mapping.yaml).
- Dataset settings and Saraga download/split utilities are in [`sargam/config/`](sargam/config/) and [`sargam/data/`](sargam/data/).
- The gamaka and timbre components are configured to look for model files in the Hugging Face repository `RK14x86/sargam-models`. If those models are unavailable, the code uses its implemented fallback paths where applicable.
- HDemucs source separation is optional and is only requested when `separate_source=True`.

Run the pipeline tests from the `sargam/` directory:

```bash
pip install -r requirements-dev.txt
pytest
```

## Repository layout

```text
Sargam/
├── README.md
├── sargam_backend/          # Browser-facing Flask sheet generator
│   ├── app.py
│   ├── melody.py
│   ├── sargam_converter.py
│   ├── gemini_singer.py
│   ├── ornaments.py
│   ├── templates/
│   ├── requirements.txt
│   └── Dockerfile
└── sargam/                  # Modular ML pipeline
    ├── api/                 # Local Flask API
    ├── config/              # Model, audio and dataset settings
    ├── data/                # Dataset utilities and data
    ├── src/
    │   ├── features/
    │   ├── mapping/
    │   ├── models/
    │   ├── pipeline/
    │   └── utils/
    ├── tests/
    ├── requirements.txt
    └── pyproject.toml
```

## Current limitations

Sargam is a work in progress, not a guaranteed transcription system. Pitch tracking and ornament classification can be affected by recording quality, overlapping instruments, percussion and expressive pitch movement. Generated notation should be reviewed by a musician.

- Lyrics are optional user-provided text; they are not automatically recognized from the recording.
- Spleeter, madmom, Gemini and downloaded ML models are optional integrations and may not be available in every environment.
- The repository does not currently publish verified user, download or usage statistics.

## License

The nested ML package declares an MIT license in its package metadata. A single repository-wide license file is not currently present; check the relevant component and dataset/model licenses before redistribution.
