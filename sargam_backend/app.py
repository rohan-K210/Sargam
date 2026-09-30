"""
Sargam Sheet Generator - Flask Backend
API endpoints for audio analysis, sargam conversion, and Gemini TTS singing.
"""

import os
import uuid
import json
import base64
import tempfile
import json as _json
from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS


class NumpySafeEncoder(_json.JSONEncoder):
    """Flask JSON encoder that handles numpy types safely."""
    def default(self, obj):
        import numpy as np
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return super().default(obj)

# Load environment variables from .env file if present
load_dotenv()

from melody import (
    analyze_melody, sargam_to_notation_lines, create_time_blocks,
    detect_sa_candidates, SPLEETER_AVAILABLE, MADMOM_AVAILABLE,
    MAX_DURATION_SECONDS
)
from sargam_converter import build_sargam_sheet, generate_gemini_sargam_analysis_prompt
from gemini_singer import get_gemini_client, sing_sargam_line

app = Flask(__name__, static_folder='static', static_url_path='')
app.json_encoder = NumpySafeEncoder
CORS(app)

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload


def cleanup_file(filepath: str):
    """Safely remove a file if it exists."""
    try:
        if filepath and os.path.exists(filepath):
            os.remove(filepath)
    except Exception:
        pass


@app.route('/')
def index():
    """Serve the main frontend."""
    return send_from_directory(os.path.join(os.path.dirname(__file__), 'templates'), 'index.html')


@app.route('/api/health')
def health():
    """Health check endpoint with Spleeter/madmom availability."""
    client = get_gemini_client()
    return jsonify({
        'status': 'ok',
        'gemini_available': client is not None,
        'spleeter_available': SPLEETER_AVAILABLE,
        'madmom_available': MADMOM_AVAILABLE,
        'max_duration_seconds': MAX_DURATION_SECONDS,
    })


@app.route('/api/sa-candidates', methods=['POST'])
def sa_candidates():
    """
    Detect Sa (tonic) candidates from an audio file before full analysis.
    Useful for letting the user select Sa before generating the sheet.

    Expects: multipart/form-data with:
        - audio: Audio file

    Returns: JSON with list of Sa candidates sorted by confidence
    """
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    ext = os.path.splitext(audio_file.filename)[1] or '.wav'
    filename = f"{uuid.uuid4()}{ext}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    audio_file.save(filepath)

    try:
        import librosa
        y, sr = librosa.load(filepath, sr=None, duration=MAX_DURATION_SECONDS, mono=True)
        candidates = detect_sa_candidates(y, sr)
        cleanup_file(filepath)
        return jsonify({
            'success': True,
            'candidates': candidates
        })
    except Exception as e:
        cleanup_file(filepath)
        return jsonify({'error': f'Sa detection failed: {str(e)}'}), 500


@app.route('/api/analyze', methods=['POST'])
def analyze():
    """
    Analyze uploaded audio and convert to sargam notation.

    Expects: multipart/form-data with:
        - audio: Audio file (MP3, WAV, etc.)
        - song_name (optional): Name of the song
        - lyrics (optional): Lyrics text (lines separated by newlines)
        - sa_hz (optional): Override Sa frequency (float). If not provided, auto-detect.
        - use_gemini (optional): 'true' to use Gemini for enhanced analysis

    Returns: JSON with sargam notation and timing data
    """
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    song_name = request.form.get('song_name', '')
    lyrics = request.form.get('lyrics', '')
    use_gemini_analysis = request.form.get('use_gemini', 'false').lower() == 'true'

    # Optional Sa override
    sa_hz_str = request.form.get('sa_hz', '')
    sa_hz = None
    if sa_hz_str:
        try:
            sa_hz = float(sa_hz_str)
        except ValueError:
            return jsonify({'error': f'Invalid sa_hz value: {sa_hz_str}'}), 400

    ext = os.path.splitext(audio_file.filename)[1] or '.wav'
    filename = f"{uuid.uuid4()}{ext}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    audio_file.save(filepath)

    try:
        melody_result = analyze_melody(filepath, sa_hz=sa_hz)

        if not melody_result['notes']:
            result = {
                'success': False,
                'error': 'Could not detect any notes in the audio. Please try a clearer recording.',
                'sa_frequency': melody_result['sa_frequency'],
                'sa_note': melody_result['sa_note'],
                'sa_candidates': melody_result.get('sa_candidates', []),
                'spleeter_available': SPLEETER_AVAILABLE,
                'madmom_available': MADMOM_AVAILABLE,
            }
            cleanup_file(filepath)
            return jsonify(result), 422

        lyrics_lines = []
        if lyrics:
            lyrics_lines = [l.strip() for l in lyrics.split('\n') if l.strip()]

        notation_lines = sargam_to_notation_lines(
            melody_result['notes'],
            lyrics_lines if lyrics_lines else None
        )

        # Create full song time blocks (no gaps)
        time_blocks = create_time_blocks(
            melody_result['notes'],
            melody_result['duration_seconds'],
            lyrics_lines if lyrics_lines else None
        )

        phrases = []
        for phrase_group in melody_result['phrases']:
            phrase_notes = [n.to_display() for n in phrase_group]
            if phrase_notes:
                phrases.append({
                    'notes': [n.formatted for n in phrase_group],
                    'display': '  '.join(phrase_notes),
                    'start_time': round(phrase_group[0].start_time, 2),
                    'end_time': round(phrase_group[-1].end_time, 2)
                })

        cleanup_file(filepath)

        gemini_result = None

        # If Gemini analysis is enabled and available, try to enhance time blocks
        if use_gemini_analysis:
            client = get_gemini_client()
            if client:
                try:
                    enhanced_prompt = generate_gemini_sargam_analysis_prompt(
                        [{'start_time': n.start_time, 'end_time': n.end_time,
                          'note': n.formatted, 'display': n.to_display(),
                          'beat_number': getattr(n, 'beat_number', -1),
                          'position_in_bar': getattr(n, 'position_in_bar', -1),
                          'ornament': getattr(n, 'ornament', '')}
                         for n in melody_result['notes']],
                        lyrics, song_name,
                        sa_candidates=melody_result.get('sa_candidates', []),
                        beat_times=melody_result.get('beat_times', []),
                        downbeat_times=melody_result.get('downbeat_times', []),
                        ornaments_data=melody_result.get('ornaments', None),
                        duration_seconds=melody_result['duration_seconds']
                    )
                    response = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=enhanced_prompt
                    )
                    resp_text = response.text
                    if '```json' in resp_text:
                        resp_text = resp_text.split('```json')[1].split('```')[0].strip()
                    elif '```' in resp_text:
                        resp_text = resp_text.split('```')[1].split('```')[0].strip()
                    gemini_result = json.loads(resp_text)
                except Exception as e:
                    print(f"Gemini analysis error: {e}")

        # If Gemini returned enhanced blocks with full coverage, use those instead
        final_blocks = time_blocks
        if gemini_result and 'blocks' in gemini_result and len(gemini_result['blocks']) > 0:
            final_blocks = gemini_result['blocks']

        sa_info = f"{melody_result['sa_note']} ({melody_result['sa_frequency']:.1f} Hz)"
        text_sheet = build_sargam_sheet(notation_lines, song_name, sa_info)

        return jsonify({
            'success': True,
            'song_name': song_name,
            'sa_frequency': melody_result['sa_frequency'],
            'sa_note': melody_result['sa_note'],
            'sa_info': sa_info,
            'sa_candidates': melody_result.get('sa_candidates', []),
            'notation_lines': notation_lines,
            'phrases': phrases,
            'time_blocks': final_blocks,
            'text_sheet': text_sheet,
            'duration_seconds': melody_result['duration_seconds'],
            'gemini_enhanced': gemini_result,
            'spleeter_available': SPLEETER_AVAILABLE,
            'madmom_available': MADMOM_AVAILABLE,
            'beat_times': melody_result.get('beat_times', []),
            'downbeat_times': melody_result.get('downbeat_times', []),
            'ornaments': melody_result.get('ornaments', None),
            'truncated': melody_result.get('truncated', False),
            'max_duration_seconds': melody_result.get('max_duration_seconds', 60),
        })

    except Exception as e:
        cleanup_file(filepath)
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500


@app.route('/api/sing', methods=['POST'])
def sing():
    """Generate singing audio for sargam notation using Gemini TTS."""
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    sargam_text = data.get('sargam_text', '')
    lyrics = data.get('lyrics', '')
    voice = data.get('voice', 'Kore')

    if not sargam_text:
        return jsonify({'error': 'No sargam text provided'}), 400

    audio_bytes = sing_sargam_line(sargam_text, lyrics, voice)

    if audio_bytes is None:
        return jsonify({'error': 'Gemini TTS failed. Check API key.'}), 500

    audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')

    return jsonify({
        'success': True,
        'audio_base64': audio_b64,
        'mime_type': 'audio/wav',
        'sargam_text': sargam_text
    })


@app.route('/api/sing-all', methods=['POST'])
def sing_all():
    """Generate singing audio for all lines."""
    data = request.get_json()
    if not data or 'lines' not in data:
        return jsonify({'error': 'No lines provided'}), 400

    lines = data['lines']
    voice = data.get('voice', 'Kore')

    results = []
    for i, line in enumerate(lines):
        sargam = line.get('sargam', '')
        lyrics = line.get('lyrics', '')
        audio_bytes = sing_sargam_line(sargam, lyrics, voice)
        audio_b64 = base64.b64encode(audio_bytes).decode('utf-8') if audio_bytes else None

        results.append({
            'index': i,
            'lyrics': lyrics,
            'sargam': sargam,
            'audio_base64': audio_b64
        })

    return jsonify({'success': True, 'lines': results})


@app.route('/api/config')
def get_config():
    """Return public configuration."""
    gemini_available = get_gemini_client() is not None
    return jsonify({
        'gemini_available': gemini_available,
        'spleeter_available': SPLEETER_AVAILABLE,
        'madmom_available': MADMOM_AVAILABLE,
        'max_upload_size_mb': 50,
        'max_duration_seconds': MAX_DURATION_SECONDS,
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 7860))
    debug = os.environ.get('FLASK_DEBUG', 'true').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug)
