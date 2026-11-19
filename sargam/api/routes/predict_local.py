from flask import Blueprint, request, jsonify
import base64
import numpy as np
import io
import soundfile as sf
from src.pipeline.orchestrator import create_pipeline
from src.utils.io import load_audio_base64

bp = Blueprint('predict_local', __name__)

_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = create_pipeline()
    return _pipeline


def format_student_view(data: dict) -> dict:
    """Format output for student-friendly display."""
    notes = data.get('notes', [])
    
    # Simplified notation string
    notation_parts = []
    for n in notes:
        svara = n.get('swara', '')
        variant = n.get('variant', '')
        octave = n.get('octave', '')
        ornament = n.get('ornament', {}).get('type', 'none')
        
        display = svara
        if variant == 'komal':
            display += '♭'
        elif variant == 'tivra':
            display += '♯'
        
        if octave == 'Mandra':
            display = display.lower()
        elif octave == 'Taar':
            display = display + "'"
        
        if ornament != 'none':
            display += f"({ornament[:3]})"
        
        notation_parts.append(display)
    
    # Error analysis
    errors = []
    for n in notes:
        err = n.get('error_cents', 0)
        if abs(err) > 20:
            svara = n.get('swara', '')
            variant = n.get('variant', '')
            direction = "flat" if err < 0 else "sharp"
            errors.append(f"{svara}{variant}: {abs(err):.0f}¢ {direction}")
    
    # Practice focus
    focus = []
    if errors:
        focus.append(f"Intonation: {', '.join(errors[:3])}")
    
    ornaments_weak = [n for n in notes if n.get('ornament', {}).get('type') in ['andolan', 'gamak'] 
                      and n.get('ornament', {}).get('confidence', 0) < 0.6]
    if ornaments_weak:
        focus.append(f"Ornaments: practice {ornaments_weak[0].get('ornament', {}).get('type')}")
    
    return {
        'notation': ' | '.join(notation_parts),
        'errors': errors[:5],
        'practice_focus': focus,
        'breath_marks': [b['end_sec'] for b in data.get('breaths', [])],
        'summary': {
            'total_notes': len(notes),
            'duration_sec': data.get('audio_info', {}).get('duration_sec', 0),
            'raga_hint': data.get('tonic', '')
        }
    }


@bp.route("/run/predict-local", methods=["POST"])
def predict_local():
    try:
        data = request.get_json()
        if not data or 'audio_base64' not in data:
            return jsonify({'success': False, 'error': 'Missing audio_base64'}), 400
        
        audio_b64 = data['audio_base64']
        tonic_hz = data.get('tonic_hz')
        
        # Decode audio
        audio_bytes = base64.b64decode(audio_b64)
        audio, sr = sf.read(io.BytesIO(audio_bytes))
        
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)
        
        # Run pipeline
        pipeline = get_pipeline()
        
        # Save temp and run (orchestrator expects file path)
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            sf.write(tmp.name, audio, sr)
            result = pipeline.run(tmp.name, tonic_hz)
        
        if not result['success']:
            return jsonify({'success': False, 'error': result['error']}), 500
        
        output = result['data']
        
        # Add student view if requested
        if data.get('return_student_view', True):
            output['student_view'] = format_student_view(output)
        
        return jsonify({'success': True, **output})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500