---
title: Sargam Sheet Generator
emoji: 🎵
colorFrom: purple
colorTo: gold
sdk: docker
pinned: false
app_port: 7860
---

# Sargam Sheet Generator

A browser-based tool that analyzes an uploaded audio recording and generates Indian Sargam notation for musicians, especially flute learners.

This is the Flask web application in `sargam_backend/`. The repository also contains a separate modular ML pipeline in [`../sargam/`](../sargam/).

## What it does

1. Upload an audio file (MP3, WAV, M4A or FLAC; maximum upload size 50 MB).
2. Optionally provide a song title and lyrics.
3. Detect candidate tonic (Sa) values or enter a manual Sa frequency.
4. Analyze melody pitch, note timing, phrases, tempo and ornaments.
5. Generate a Sargam sheet with time blocks.
6. Play individual blocks or the sequence using browser-based bansuri synthesis.

Optional integrations:

- **Spleeter** can separate vocals and accompaniment when installed and available.
- **madmom** can provide beat/downbeat tracking; a librosa-based fallback is used when it is unavailable.
- **Gemini 2.5 Flash** can optionally enhance the time-block analysis when `GEMINI_API_KEY` is configured.
- **Gemini audio generation** can optionally generate sung Sargam audio.

Lyrics are supplied by the user. The application does not perform automatic speech recognition or lyric transcription from the audio.

The melody analysis currently limits processing to the first 60 seconds of a recording.

## Run locally

Python 3.10 is recommended, particularly for optional Spleeter/madmom dependencies.

```bash
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
python app.py
```

Open [http://localhost:7860](http://localhost:7860).

## Run with Docker

```bash
docker build -t sargam-sheet-generator .
docker run --rm -p 7860:7860 -e GEMINI_API_KEY=your_key sargam-sheet-generator
```

The Gemini key is optional. Without it, Gemini-enhanced analysis and singing generation are unavailable.

## Configuration

| Variable | Required | Description |
|---|---:|---|
| `GEMINI_API_KEY` | No | Enables optional Gemini analysis and audio generation |
| `PORT` | No | HTTP port; defaults to `7860` |
| `FLASK_DEBUG` | No | Flask debug mode; set to `false` in deployed environments |

## API

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | Health and optional dependency status |
| GET | `/api/config` | Runtime configuration and limits |
| POST | `/api/sa-candidates` | Detect candidate tonic values |
| POST | `/api/analyze` | Analyze audio and return Sargam notation |
| POST | `/api/sing` | Generate sung audio for one Sargam passage |
| POST | `/api/sing-all` | Generate sung audio for multiple lines |

The audio upload routes accept `multipart/form-data` with an `audio` file. The analysis route also accepts optional `song_name`, `lyrics`, `sa_hz` and `use_gemini` fields. See the route docstrings in `app.py` for details.

## Dependencies and optional features

- Flask and Flask-CORS serve the UI and API.
- librosa, NumPy and SciPy provide audio processing.
- Spleeter, madmom and TensorFlow are optional integrations with Python-version/platform constraints.
- google-genai and python-dotenv support optional Gemini features.

The health/config endpoints report whether optional Spleeter and madmom integrations are available in the current runtime.

## Deployment

The repository includes a Dockerfile for container-based deployment and a Procfile for process-based platforms. Configure `PORT` through the hosting platform and set `FLASK_DEBUG=false` for deployed environments.

## Related implementation

See the [modular ML pipeline](../sargam/) for the separate six-stage architecture covering pitch/tonic, timing, gamaka, dynamics, breath detection and timbre classification.
