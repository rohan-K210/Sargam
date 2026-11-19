# Sargam ML Pipeline

AI-powered system that converts Indian classical music audio (especially bansuri flute) into accurate sargam notation.

## Features

- **Layer 1**: Pitch detection (CREPE/PYIN) + Tonic (Sa) detection + Octave classification
- **Layer 2**: Onset/offset detection + Tempo/Laya estimation
- **Layer 3**: Gamaka detection (meend, andolan, kan, murki, gamak) - hybrid rules + ML
- **Layer 4**: Dynamics (RMS envelope) + Attack classification
- **Layer 5**: Silence/Breath detection (bansuri-specific)
- **Layer 6**: Instrument timbre classification

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Download pretrained models
python scripts/download_models.py

# Download Saraga dataset (optional, for training/evaluation)
python data/download_saraga.py
python data/prepare_splits.py

# Run pipeline on audio file
python -c "
from src.pipeline.orchestrator import create_pipeline
pipeline = create_pipeline()
result = pipeline.run('audio.wav')
print(result)
"
```

## Flask API

```bash
# Start server
python -m api.flask_app

# POST /run/predict-local
# Body: {"audio_base64": "...", "tonic_hz": 261.63}
```

## Project Structure

```
sargam/
├── config/           # YAML configs
├── data/             # Dataset download & splits
├── src/
│   ├── pipeline/     # 6-layer pipeline
│   ├── features/     # Audio, pitch, spectral features
│   ├── models/       # Tonic, onset, gamaka, timbre models
│   ├── mapping/      # Hz → Svara mapping
│   └── utils/        # I/O, caching
├── scripts/          # Training, evaluation, benchmarking
├── tests/            # Unit tests
└── api/              # Flask endpoints
```

## Output Format

```json
{
  "tonic": "C#",
  "tonic_hz": 277.18,
  "tempo": "Madhya",
  "tempo_bpm": 120,
  "instrument": "bansuri",
  "notes": [
    {
      "swara": "Ga",
      "variant": "komal",
      "octave": "Madhya",
      "onset_sec": 1.24,
      "duration_sec": 0.48,
      "ornament": {"type": "meend", "confidence": 0.87},
      "dynamics": "forte",
      "attack": "breathy"
    }
  ]
}
```

## For Flute Students

- `error_cents` per note shows intonation accuracy
- `finger_hint` suggests hole positions
- `student_view` provides simplified notation + practice focus