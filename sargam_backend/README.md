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

Convert any song into Indian Sargam notation (Sa Re Ga Ma Pa Dha Ni) for flute players.

## How to use
1. Upload an audio file (MP3, WAV, M4A, FLAC — max 50MB)
2. Optionally enter song name and lyrics
3. Click "Generate Sargam Sheet"
4. Play back any block or the full song in synthesized bansuri

## Features
- Source separation with Spleeter (vocals + accompaniment)
- Beat tracking with madmom
- Indian ornament detection (meend/gamak)
- Sa (tonic) auto-detection with manual override
- Web Audio API bansuri synthesis with 5 harmonics + breath chiff
- Gemini 2.5 Flash enhanced analysis (optional)

## Environment Variables
| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | No | Google Gemini API key for enhanced analysis |

## Technical Stack
- **Backend**: Python Flask
- **Audio**: librosa, Spleeter, madmom
- **AI**: Google Gemini 2.5 Flash
- **Frontend**: Vanilla JS + CSS + Web Audio API
- **Deployment**: Hugging Face Spaces
