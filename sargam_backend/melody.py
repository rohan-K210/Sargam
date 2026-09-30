"""
Melody analysis module for Sargam Sheet Generator.
Uses Spleeter for source separation, madmom for beat tracking,
librosa.pyin for robust pitch detection, and maps to Indian sargam notes.
"""

import os
import warnings
import tempfile
import numpy as np
from typing import List, Dict, Optional, Tuple

warnings.filterwarnings('ignore', category=UserWarning, module='librosa')

# Optional imports for enhanced features
_SPLEETER = None
try:
    from spleeter.separator import Separator
    _SPLEETER = Separator('spleeter:2stems', multiprocess=False)
    SPLEETER_AVAILABLE = True
except Exception:
    SPLEETER_AVAILABLE = False

_MADMOM_BEATS = None
_MADMOM_DOWNBEATS = None
try:
    import madmom
    from madmom.features.beats import DBNBeatTrackingProcessor, RNNBeatProcessor
    from madmom.features.downbeats import DBNDownBeatTrackingProcessor, RNNDownBeatProcessor
    _MADMOM_BEATS = lambda: (DBNBeatTrackingProcessor(fps=100), RNNBeatProcessor())
    _MADMOM_DOWNBEATS = lambda: (DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=100), RNNDownBeatProcessor())
    MADMOM_AVAILABLE = True
except Exception:
    MADMOM_AVAILABLE = False

import librosa
from ornaments import detect_meend, detect_gamak

# Sargam note definitions relative to Sa (the tonic)
# These are the 12 sargam notes in order (like the 12 semitones in Western music)
SARGAM_NOTES = [
    ("Sa", 0),       # 0 semitones from Sa
    ("re", 1),       # Komal Re (flat)
    ("Re", 2),       # Shuddha Re
    ("ga", 3),       # Komal Ga (flat)
    ("Ga", 4),       # Shuddha Ga
    ("Ma", 5),       # Shuddha Ma
    ("Ma'", 6),      # Tivra Ma (sharp)
    ("Pa", 7),       # Pa
    ("dha", 8),      # Komal Dha (flat)
    ("Dha", 9),      # Shuddha Dha
    ("ni", 10),      # Komal Ni (flat)
    ("Ni", 11),      # Shuddha Ni
]

# Western note names (0 = C)
WESTERN_NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Maximum duration to process (first 60 seconds)
MAX_DURATION_SECONDS = 60


def get_spleeter() -> Optional['Separator']:
    """Get the shared Spleeter Separator singleton."""
    global _SPLEETER
    return _SPLEETER


def separate_stems(filepath: str, output_dir: str = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int]:
    """
    Use Spleeter to separate audio into vocals and accompaniment stems.
    
    Args:
        filepath: Path to audio file
        output_dir: Directory for Spleeter output (auto temp if None)
    
    Returns:
        (vocals_audio, accompaniment_audio, sample_rate) or (None, None, 0) on failure
    """
    separator = get_spleeter()
    if separator is None:
        return None, None, 0
    
    own_dir = False
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix='spleeter_')
        own_dir = True
    
    try:
        # Spleeter separates to output_dir/{filename}/vocals.wav etc.
        separator.separate_to_file(filepath, output_dir)
        base = os.path.splitext(os.path.basename(filepath))[0]
        stem_dir = os.path.join(output_dir, base)
        
        vocals_path = os.path.join(stem_dir, 'vocals.wav')
        accomp_path = os.path.join(stem_dir, 'accompaniment.wav')
        
        vocals_y = None
        accomp_y = None
        sr = 44100
        
        if os.path.exists(vocals_path):
            vocals_y, sr = librosa.load(vocals_path, sr=None, duration=MAX_DURATION_SECONDS, mono=True)
        if os.path.exists(accomp_path):
            accomp_y, _ = librosa.load(accomp_path, sr=None, duration=MAX_DURATION_SECONDS, mono=True)
        
        # Clean up Spleeter output files
        try:
            for p in [vocals_path, accomp_path]:
                if os.path.exists(p):
                    os.remove(p)
            if os.path.isdir(stem_dir):
                os.rmdir(stem_dir)
        except Exception:
            pass
        
        if own_dir:
            try:
                os.rmdir(output_dir)
            except Exception:
                pass
        
        return vocals_y, accomp_y, sr
    except Exception as e:
        print(f"Spleeter separation error: {e}")
        # Cleanup on failure
        if own_dir:
            try:
                import shutil
                shutil.rmtree(output_dir, ignore_errors=True)
            except Exception:
                pass
        return None, None, 0


def detect_beats_madmom(y: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Use madmom to detect beat positions and downbeat positions.
    
    Returns:
        (beat_times, downbeat_times) arrays in seconds
    """
    if not MADMOM_AVAILABLE:
        # Fallback: simple energy-based beat tracking
        return _detect_beats_fallback(y, sr)
    
    try:
        # madmom expects a specific preprocessing: load with its own audio loader
        # But we can use the raw waveform
        proc_beats, rnn = _MADMOM_BEATS()
        proc_downbeats, rnn_down = _MADMOM_DOWNBEATS()
        
        # Compute beat activation features (madmom expects raw audio at 44100 Hz)
        act = rnn(y.astype(np.float32))
        beats = proc_beats(act)
        
        # Downbeat detection
        act_down = rnn_down(y.astype(np.float32))
        downbeats = proc_downbeats(act_down)
        # downbeats is shape (N, 2) where col 0 = time, col 1 = beat position (1,2,3,4)
        downbeat_times = downbeats[downbeats[:, 1] == 1, 0] if len(downbeats) > 0 else np.array([])
        
        return beats, downbeat_times
    except Exception as e:
        print(f"madmom beat detection error: {e}")
        return _detect_beats_fallback(y, sr)


def _detect_beats_fallback(y: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
    """Fallback beat detection using librosa's onset detection."""
    try:
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units='time')
        # Simple downbeat: every 4th beat (common time)
        downbeats = beats[::4] if len(beats) >= 4 else np.array([])
        return beats, downbeats
    except Exception:
        return np.array([]), np.array([])


def quantize_to_beat_grid(onsets: np.ndarray, beat_times: np.ndarray, 
                           tolerance: float = 0.12) -> np.ndarray:
    """
    Quantize note onset times to the nearest beat position.
    
    Args:
        onsets: Array of onset times in seconds
        beat_times: Array of beat times in seconds
        tolerance: Max distance to snap to a beat (seconds)
    
    Returns:
        Array of quantized onset times (same length as onsets)
    """
    if len(beat_times) == 0 or len(onsets) == 0:
        return onsets
    
    quantized = np.copy(onsets)
    for i in range(len(onsets)):
        diff = np.abs(beat_times - onsets[i])
        min_diff = np.min(diff)
        if min_diff <= tolerance:
            quantized[i] = beat_times[np.argmin(diff)]
    
    return quantized


def detect_sa_from_audio(y: np.ndarray, sr: int) -> float:
    """
    Better tonic detection using multiple approaches.
    Returns the frequency of the detected Sa.
    """
    hop_length = 512
    fmin = librosa.note_to_hz('C2')
    fmax = librosa.note_to_hz('C6')
    
    try:
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, sr=sr, fmin=fmin, fmax=fmax,
            hop_length=hop_length, fill_na=np.nan
        )
    except Exception:
        return 261.63  # Default Sa = C4
    
    # Filter out NaN values and low confidence
    valid_mask = ~np.isnan(f0) & (voiced_probs > 0.3)
    valid_pitches = f0[valid_mask]
    
    if len(valid_pitches) < 10:
        return 261.63
    
    # Find the most common pitch using KDE-like approach
    pitch_midi = librosa.hz_to_midi(valid_pitches)
    pitch_rounded = np.round(pitch_midi * 2) / 2
    
    unique, counts = np.unique(pitch_rounded, return_counts=True)
    best_idx = np.argmax(counts)
    best_midi = unique[best_idx]
    
    # Try this as Pa (fifth), derive Sa
    sa_from_pa = librosa.midi_to_hz(best_midi - 7)
    # Try this as Sa directly
    sa_direct = librosa.midi_to_hz(best_midi)
    
    # Choose the lower one (Sa is usually the lower tonic)
    sa = min(sa_from_pa, sa_direct)
    
    # Ensure Sa is in a reasonable range (C3 to C5)
    sa_hz = float(sa)
    if sa_hz < 130 or sa_hz > 523:
        sa_hz = 261.63
    
    return sa_hz


def detect_sa_candidates(y: np.ndarray, sr: int, top_k: int = 5) -> List[Dict]:
    """
    Detect top-K tonic (Sa) candidates with confidence scores.
    
    Uses pitch histogram analysis to find the most likely Sa frequencies,
    returning multiple candidates ranked by confidence.
    
    Returns:
        List of dicts: [{'frequency': float, 'note_name': str, 'confidence': float}, ...]
        At most top_k items, sorted by confidence descending.
    """
    hop_length = 512
    fmin = librosa.note_to_hz('C2')
    fmax = librosa.note_to_hz('C6')
    
    try:
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, sr=sr, fmin=fmin, fmax=fmax,
            hop_length=hop_length, fill_na=np.nan
        )
    except Exception:
        return [{'frequency': 261.63, 'note_name': 'C4', 'confidence': 1.0}]
    
    valid_mask = ~np.isnan(f0) & (voiced_probs > 0.3)
    valid_pitches = f0[valid_mask]
    
    if len(valid_pitches) < 10:
        return [{'frequency': 261.63, 'note_name': 'C4', 'confidence': 1.0}]
    
    # Build pitch histogram at semitone resolution
    pitch_midi = librosa.hz_to_midi(valid_pitches)
    
    candidates = []
    
    # Method 1: Most common pitch as Sa (direct)
    for offset in [0, 7]:  # Try as Sa or Pa
        shifted = pitch_midi - offset
        rounded = np.round(shifted * 2) / 2
        unique, counts = np.unique(rounded, return_counts=True)
        
        for idx in np.argsort(counts)[-5:][::-1]:
            sa_midi = unique[idx]
            sa_hz = librosa.midi_to_hz(sa_midi)
            if 130 <= sa_hz <= 523:
                confidence = counts[idx] / len(valid_pitches)
                note_name = librosa.midi_to_note(sa_midi)
                # Penalize extreme values slightly
                if sa_hz < 150 or sa_hz > 400:
                    confidence *= 0.9
                candidates.append({
                    'frequency': float(sa_hz),
                    'note_name': note_name,
                    'confidence': float(confidence)
                })
    
    # Method 2: Spectral peak analysis
    try:
        spec = np.abs(librosa.stft(y, hop_length=hop_length))
        freqs = librosa.fft_frequencies(sr=sr, hop_length=hop_length)
        
        # Find prominent spectral peaks
        spec_mean = np.mean(spec, axis=1)
        threshold = np.percentile(spec_mean, 90)
        peaks = np.where(spec_mean > threshold)[0]
        
        for peak_idx in peaks[-10:]:
            peak_freq = freqs[peak_idx]
            if 130 <= peak_freq <= 523:
                sa_midi = librosa.hz_to_midi(peak_freq)
                note_name = librosa.midi_to_note(sa_midi)
                candidates.append({
                    'frequency': float(peak_freq),
                    'note_name': note_name,
                    'confidence': float(spec_mean[peak_idx] / np.max(spec_mean) * 0.5)
                })
    except Exception:
        pass
    
    # Deduplicate: merge candidates within 1 semitone
    merged = []
    for cand in sorted(candidates, key=lambda x: x['confidence'], reverse=True):
        freq = cand['frequency']
        existing = False
        for m in merged:
            if abs(12 * np.log2(m['frequency'] / freq)) < 1.0:
                if cand['confidence'] > m['confidence']:
                    m['frequency'] = cand['frequency']
                    m['note_name'] = cand['note_name']
                    m['confidence'] = max(m['confidence'], cand['confidence'])
                existing = True
                break
        if not existing:
            merged.append(cand)
    
    # Normalize confidences so top is 1.0
    merged = sorted(merged, key=lambda x: x['confidence'], reverse=True)[:top_k]
    if merged:
        max_conf = merged[0]['confidence']
        for m in merged:
            m['confidence'] = m['confidence'] / max_conf if max_conf > 0 else 0.5
    
    return merged


def hz_to_sargam(frequency: float, sa_hz: float) -> Optional[Tuple[str, int]]:
    """
    Convert a frequency to the nearest sargam note name and octave offset.
    
    Returns: (note_name, octave_offset) where octave_offset is:
        0 = middle octave (Madhya Saptak)
        -1 = lower octave (Mandra Saptak)
        +1 = higher octave (Tar Saptak)
    """
    if frequency <= 0 or sa_hz <= 0:
        return None
    
    semitone_distance = 12 * np.log2(frequency / sa_hz)
    rounded_distance = round(semitone_distance)
    
    octave_offset = 0
    note_distance = rounded_distance
    
    while note_distance < 0:
        note_distance += 12
        octave_offset -= 1
    while note_distance > 11:
        note_distance -= 12
        octave_offset += 1
    
    note_name = SARGAM_NOTES[note_distance][0]
    
    return (note_name, octave_offset)


def format_sargam_note(note_name: str, octave_offset: int) -> str:
    """Format a sargam note with proper octave indicators."""
    if octave_offset == 0:
        return note_name
    elif octave_offset == 1:
        return f"{note_name}'"
    elif octave_offset == -1:
        return f".{note_name}"
    elif octave_offset > 1:
        apostrophes = "'" * octave_offset
        return f"{note_name}{apostrophes}"
    else:
        dots = "." * abs(octave_offset)
        return f"{dots}{note_name}"


class NoteEvent:
    """Represents a detected note with timing information."""
    def __init__(self, note: str, octave: int, start_time: float, end_time: float, 
                 formatted: str, dynamics: str = "",
                 beat_number: int = -1, position_in_bar: int = -1,
                 ornament: str = "", accomp_energy: float = 0.0):
        self.note = note
        self.octave = int(octave)
        self.start_time = float(start_time)
        self.end_time = float(end_time)
        self.duration = float(end_time - start_time)
        self.formatted = formatted
        self.dynamics = dynamics
        self.beat_number = int(beat_number)
        self.position_in_bar = int(position_in_bar)
        self.ornament = ornament  # 'meend', 'gamak', or ''
        self.accomp_energy = float(accomp_energy)
    
    def is_long(self) -> bool:
        return self.duration > 0.3
    
    def to_display(self) -> str:
        result = self.formatted
        if self.ornament == 'meend':
            result = f"{result}~"
        elif self.ornament == 'gamak':
            result = f"{result}~~~"
        if self.is_long():
            result = result.upper()
        if self.dynamics:
            result = f"{result} {self.dynamics}"
        return result


def detect_accompaniment_energy(accomp_y: np.ndarray, sr: int, 
                                 hop_length: int, times: np.ndarray) -> np.ndarray:
    """
    Compute accompaniment energy per time frame for block classification.
    
    Returns:
        Array of energy values per frame, normalized 0–1
    """
    if accomp_y is None:
        return np.zeros(len(times))
    
    # Compute frame-level RMS energy
    frame_length = hop_length * 2
    energy = librosa.feature.rms(y=accomp_y, frame_length=frame_length, 
                                  hop_length=hop_length)[0]
    
    # Interpolate to match times array length
    energy_times = librosa.times_like(energy, sr=sr, hop_length=hop_length)
    energy_interp = np.interp(times, energy_times[:len(energy)], energy[:len(energy_times)])
    
    # Normalize to 0–1
    if np.max(energy_interp) > 0:
        energy_interp = energy_interp / np.max(energy_interp)
    
    return energy_interp


def analyze_melody(filepath: str, sa_hz: Optional[float] = None) -> Dict:
    """
    Analyze audio file and extract melody as sargam notes.
    
    Uses Spleeter to separate vocals/accompaniment, madmom for beat tracking,
    and pYIN for pitch detection on the vocal stem.
    
    Args:
        filepath: Path to audio file
        sa_hz: Optional tonic frequency (Sa). If None, auto-detect.
    
    Returns:
        Dict with sa info, notes, phrases, ornaments, beat info, Sa candidates
    """
    # --- Step 1: Load audio (limit to 60s) ---
    y, sr = librosa.load(filepath, sr=None, duration=MAX_DURATION_SECONDS, mono=True)
    
    total_duration = len(y) / sr
    
    # --- Step 2: Source separation ---
    vocals_y, accomp_y, sep_sr = separate_stems(filepath)
    
    # Use the stems if available, otherwise fall back to HPSS harmonic
    if vocals_y is not None:
        pitch_source = vocals_y
        # Resample to original sr if needed
        if sep_sr != sr:
            pitch_source = librosa.resample(vocals_y, orig_sr=sep_sr, target_sr=sr)
            if len(pitch_source) < len(y):
                pitch_source = np.pad(pitch_source, (0, len(y) - len(pitch_source)))
            else:
                pitch_source = pitch_source[:len(y)]
    else:
        # Fallback: librosa HPSS to extract harmonic (vocal-like) component
        try:
            y_harmonic, _ = librosa.effects.hpss(y, margin=2.0)
            pitch_source = y_harmonic
            print("Using HPSS harmonic component as vocal proxy")
        except Exception:
            pitch_source = y
            print("HPSS failed, using mixed audio")
    
    if accomp_y is not None:
        if sep_sr != sr:
            accomp_y_resampled = librosa.resample(accomp_y, orig_sr=sep_sr, target_sr=sr)
            if len(accomp_y_resampled) < len(y):
                accomp_y = np.pad(accomp_y_resampled, (0, len(y) - len(accomp_y_resampled)))
            else:
                accomp_y = accomp_y_resampled[:len(y)]
    else:
        accomp_y = None
    
    # --- Step 3: Detect Sa if not provided ---
    if sa_hz is None:
        sa_hz = detect_sa_from_audio(pitch_source, sr)
    
    # Get Sa candidates
    sa_candidates = detect_sa_candidates(pitch_source, sr)
    
    # Western note name for Sa
    sa_midi = librosa.hz_to_midi(sa_hz)
    sa_western_note = librosa.midi_to_note(sa_midi)
    
    # --- Step 4: Beat tracking ---
    beat_times, downbeat_times = detect_beats_madmom(y, sr)
    
    # --- Step 5: Pitch detection on vocal stem ---
    hop_length = 512
    fmin = max(65, sa_hz / 4)
    fmax = min(2000, sa_hz * 4)
    
    try:
        f0, voiced_flag, voiced_probs = librosa.pyin(
            pitch_source, sr=sr, fmin=fmin, fmax=fmax,
            hop_length=hop_length, fill_na=np.nan
        )
    except Exception:
        f0_full, magnitudes = librosa.piptrack(y=pitch_source, sr=sr, hop_length=hop_length,
                                                fmin=fmin, fmax=fmax)
        f0 = np.zeros(f0_full.shape[1])
        for i in range(f0_full.shape[1]):
            idx = magnitudes[:, i].argmax()
            f0[i] = f0_full[idx, i] if magnitudes[idx, i] > 0.02 else np.nan
        voiced_probs = np.ones_like(f0)
    
    times = librosa.times_like(f0, sr=sr, hop_length=hop_length)
    
    # --- Step 6: Detect ornaments ---
    ornaments_meend = detect_meend(f0, times)
    ornaments_gamak = detect_gamak(f0, times)
    
    # Build ornament lookup: for each time range, what ornament is active
    def get_ornament_at_time(t: float) -> str:
        for o in ornaments_meend:
            if o['start_time'] <= t <= o['end_time']:
                return 'meend'
        for o in ornaments_gamak:
            if o['start_time'] <= t <= o['end_time']:
                return 'gamak'
        return ''
    
    # --- Step 7: Accompaniment energy ---
    accomp_energy = detect_accompaniment_energy(accomp_y, sr, hop_length, times)
    
    # --- Step 8: Convert pitch frames to sargam notes ---
    raw_notes = []
    
    for i in range(len(f0)):
        if np.isnan(f0[i]) or voiced_probs[i] < 0.2:
            continue
        result = hz_to_sargam(f0[i], sa_hz)
        if result:
            note_name, octave = result
            formatted = format_sargam_note(note_name, octave)
            ornament = get_ornament_at_time(times[i])
            
            # Determine beat info
            beat_number = -1
            position_in_bar = -1
            if len(beat_times) > 0:
                beat_diff = np.abs(beat_times - times[i])
                nearest_beat = np.argmin(beat_diff)
                if beat_diff[nearest_beat] < 0.15:
                    beat_number = nearest_beat
                    # Position in bar (assuming 4/4 time)
                    position_in_bar = (nearest_beat % 4) + 1
            
            raw_notes.append({
                'time': times[i],
                'note': note_name,
                'octave': octave,
                'formatted': formatted,
                'freq': f0[i],
                'ornament': ornament,
                'beat_number': beat_number,
                'position_in_bar': position_in_bar,
                'accomp_energy': float(accomp_energy[i]) if i < len(accomp_energy) else 0.0
            })
    
    if not raw_notes:
        return {
            'sa_frequency': float(sa_hz),
            'sa_note': sa_western_note,
            'sa_candidates': sa_candidates,
            'notes': [],
            'phrases': [],
            'ornaments': {'meend': ornaments_meend, 'gamak': ornaments_gamak},
            'beat_times': beat_times.tolist() if hasattr(beat_times, 'tolist') else list(beat_times),
            'downbeat_times': downbeat_times.tolist() if hasattr(downbeat_times, 'tolist') else list(downbeat_times),
            'duration_seconds': total_duration,
            'spleeter_available': SPLEETER_AVAILABLE,
            'madmom_available': MADMOM_AVAILABLE,
        }
    
    # --- Step 9: Quantize onsets to beat grid ---
    onset_times = np.array([r['time'] for r in raw_notes])
    quantized_times = quantize_to_beat_grid(onset_times, beat_times)
    
    # Update raw notes with quantized times
    for i, q_time in enumerate(quantized_times):
        raw_notes[i]['time'] = q_time
    
    # --- Step 10: Group consecutive same notes into NoteEvents ---
    note_events = []
    min_gap = 0.05
    
    current_note = None
    current_start = 0
    current_ornament = ''
    current_beat = -1
    current_pos = -1
    current_energy = 0.0
    
    for n in raw_notes:
        key = n['formatted']
        if current_note is None:
            current_note = key
            current_start = n['time']
            current_ornament = n['ornament']
            current_beat = n['beat_number']
            current_pos = n['position_in_bar']
            current_energy = n['accomp_energy']
            continue
        
        if key == current_note:
            # Extend the note — update ornament if later ornament is present
            if n['ornament']:
                current_ornament = n['ornament']
            continue
        else:
            end_time = n['time']
            if end_time - current_start >= min_gap:
                orig = next((x for x in raw_notes if x['formatted'] == current_note), None)
                if orig:
                    note_events.append(NoteEvent(
                        note=orig['note'],
                        octave=orig['octave'],
                        start_time=current_start,
                        end_time=end_time,
                        formatted=orig['formatted'],
                        ornament=current_ornament,
                        beat_number=current_beat,
                        position_in_bar=current_pos,
                        accomp_energy=current_energy
                    ))
            current_note = key
            current_start = n['time']
            current_ornament = n['ornament']
            current_beat = n['beat_number']
            current_pos = n['position_in_bar']
            current_energy = n['accomp_energy']
    
    if current_note and len(raw_notes) > 0:
        orig = next((x for x in raw_notes if x['formatted'] == current_note), None)
        if orig:
            end_time = raw_notes[-1]['time']
            if end_time - current_start >= min_gap:
                note_events.append(NoteEvent(
                    note=orig['note'],
                    octave=orig['octave'],
                    start_time=current_start,
                    end_time=end_time,
                    formatted=orig['formatted'],
                    ornament=current_ornament,
                    beat_number=current_beat,
                    position_in_bar=current_pos,
                    accomp_energy=current_energy
                ))
    
    if not note_events:
        return {
            'sa_frequency': float(sa_hz),
            'sa_note': sa_western_note,
            'sa_candidates': sa_candidates,
            'notes': [],
            'phrases': [],
            'ornaments': {'meend': ornaments_meend, 'gamak': ornaments_gamak},
            'beat_times': beat_times.tolist() if hasattr(beat_times, 'tolist') else list(beat_times),
            'downbeat_times': downbeat_times.tolist() if hasattr(downbeat_times, 'tolist') else list(downbeat_times),
            'duration_seconds': total_duration,
            'spleeter_available': SPLEETER_AVAILABLE,
            'madmom_available': MADMOM_AVAILABLE,
        }
    
    # Assign dynamics based on duration
    durations = [n.duration for n in note_events]
    avg_dur = np.mean(durations) if durations else 0.1
    std_dur = np.std(durations) if len(durations) > 1 else 0
    
    for note in note_events:
        if std_dur > 0:
            z = (note.duration - avg_dur) / std_dur
            if z > 0.5:
                note.dynamics = "(loud)"
            elif z < -0.5:
                note.dynamics = "(soft)"
    
    # Group into phrases based on timing gaps
    phrases = []
    current_phrase = []
    last_end = 0
    
    for note in note_events:
        if not current_phrase:
            current_phrase.append(note)
            last_end = note.end_time
        elif note.start_time - last_end > 0.5:
            if current_phrase:
                phrases.append(current_phrase)
            current_phrase = [note]
            last_end = note.end_time
        else:
            current_phrase.append(note)
            last_end = note.end_time
    
    if current_phrase:
        phrases.append(current_phrase)
    
    return {
        'sa_frequency': float(sa_hz),
        'sa_note': sa_western_note,
        'sa_candidates': sa_candidates,
        'notes': note_events,
        'phrases': phrases,
        'ornaments': {'meend': ornaments_meend, 'gamak': ornaments_gamak},
        'beat_times': beat_times.tolist() if hasattr(beat_times, 'tolist') else list(beat_times),
        'downbeat_times': downbeat_times.tolist() if hasattr(downbeat_times, 'tolist') else list(downbeat_times),
        'duration_seconds': total_duration,
        'truncated': total_duration >= MAX_DURATION_SECONDS,
        'max_duration_seconds': MAX_DURATION_SECONDS,
        'spleeter_available': SPLEETER_AVAILABLE,
        'madmom_available': MADMOM_AVAILABLE,
    }


def create_time_blocks(notes: List[NoteEvent], total_duration: float,
                         lyrics_lines: List[str] = None) -> List[Dict]:
    """
    Divide the entire song into continuous time blocks from start to finish.
    Uses note variety/density and accompaniment energy for block classification.
    """
    if not notes:
        return [{
            'start_time': 0.0,
            'end_time': round(total_duration, 2),
            'type': 'Instrumental',
            'sargam': '',
            'notes': [],
            'lyrics': ''
        }]
    
    # First pass: find gaps between notes to create boundaries
    gap_threshold = 0.5
    boundaries = [0.0]
    
    for i in range(1, len(notes)):
        gap = notes[i].start_time - notes[i - 1].end_time
        if gap > gap_threshold:
            boundaries.append(round(notes[i - 1].end_time, 2))
            boundaries.append(round(notes[i].start_time, 2))
    
    boundaries.append(round(total_duration, 2))
    
    clean_boundaries = []
    for b in boundaries:
        if not clean_boundaries or abs(b - clean_boundaries[-1]) > 0.05:
            clean_boundaries.append(b)
    
    # Second pass: create blocks from boundaries
    raw_blocks = []
    
    for i in range(len(clean_boundaries) - 1):
        block_start = clean_boundaries[i]
        block_end = clean_boundaries[i + 1]
        
        if block_end - block_start < 0.1:
            continue
        
        block_notes = []
        for n in notes:
            if n.start_time < block_end and n.end_time > block_start:
                clipped_start = max(n.start_time, block_start)
                clipped_end = min(n.end_time, block_end)
                block_notes.append({
                    'note': n.formatted,
                    'display': n.to_display(),
                    'start_time': round(clipped_start, 2),
                    'end_time': round(clipped_end, 2),
                    'duration': round(clipped_end - clipped_start, 3),
                    'beat_number': n.beat_number,
                    'position_in_bar': n.position_in_bar,
                    'ornament': n.ornament,
                })
        
        if not block_notes:
            block_type = 'Instrumental'
            sargam_str = ''
        else:
            unique_notes = set(n['note'] for n in block_notes)
            duration = block_end - block_start
            note_density = len(block_notes) / max(duration, 0.1)
            
            if len(unique_notes) >= 3 and note_density >= 0.8:
                block_type = 'Vocal'
            elif len(unique_notes) >= 5:
                block_type = 'Vocal'
            else:
                block_type = 'Instrumental'
            
            sargam_str = '  '.join([n['display'] for n in block_notes])
        
        raw_blocks.append({
            'start_time': round(block_start, 2),
            'end_time': round(block_end, 2),
            'type': block_type,
            'sargam': sargam_str,
            'notes': block_notes,
            'lyrics': ''
        })
    
    # Third pass: merge very small blocks or blocks with too few notes
    merged_blocks = []
    min_notes_per_block = 3
    
    for block in raw_blocks:
        if not merged_blocks:
            merged_blocks.append(block)
            continue
        
        last = merged_blocks[-1]
        note_count = len(block['notes'])
        last_note_count = len(last['notes'])
        
        # Merge if current block has very few notes OR is very short
        if note_count < min_notes_per_block and block['type'] == last['type']:
            # Merge into previous block
            last['end_time'] = block['end_time']
            last['notes'].extend(block['notes'])
            if block['sargam']:
                if last['sargam']:
                    last['sargam'] += '  ' + block['sargam']
                else:
                    last['sargam'] = block['sargam']
            if block['lyrics']:
                if last['lyrics']:
                    last['lyrics'] += ' ' + block['lyrics']
                else:
                    last['lyrics'] = block['lyrics']
        elif last_note_count < min_notes_per_block and block['type'] == last['type']:
            # Merge previous into current
            block['start_time'] = last['start_time']
            block['notes'] = last['notes'] + block['notes']
            if last['sargam']:
                block['sargam'] = last['sargam'] + '  ' + block['sargam']
            if last['lyrics']:
                block['lyrics'] = last['lyrics'] + ' ' + block['lyrics']
            merged_blocks[-1] = block
        else:
            merged_blocks.append(block)
    
    blocks = merged_blocks
    
    # Distribute lyrics across vocal blocks
    if lyrics_lines and blocks:
        vocal_blocks = [b for b in blocks if b['type'] == 'Vocal']
        if vocal_blocks:
            lines_per_block = max(1, len(lyrics_lines) // len(vocal_blocks))
            lyric_idx = 0
            for b in blocks:
                if b['type'] == 'Vocal' and lyric_idx < len(lyrics_lines):
                    block_lyrics = lyrics_lines[lyric_idx:lyric_idx + lines_per_block]
                    b['lyrics'] = ' '.join(block_lyrics)
                    lyric_idx += lines_per_block
    
    return blocks


def sargam_to_notation_lines(notes: List[NoteEvent], lyrics_lines: List[str] = None) -> List[Dict]:
    """Convert note events to display-ready lines with timing."""
    if not notes:
        return []
    
    if lyrics_lines:
        total_lines = len(lyrics_lines)
        notes_per_line = max(1, len(notes) // total_lines)
        
        lines = []
        for i, lyric in enumerate(lyrics_lines):
            start_idx = i * notes_per_line
            end_idx = min(start_idx + notes_per_line, len(notes)) if i < total_lines - 1 else len(notes)
            
            line_notes = notes[start_idx:end_idx]
            if not line_notes:
                continue
            
            line = {
                'lyrics': lyric.strip(),
                'sargam': '  '.join([n.to_display() for n in line_notes]),
                'notes': [{
                    'note': n.formatted,
                    'display': n.to_display(),
                    'start_time': round(n.start_time, 2),
                    'end_time': round(n.end_time, 2),
                    'ornament': n.ornament,
                    'beat_number': n.beat_number,
                } for n in line_notes],
                'start_time': round(line_notes[0].start_time, 2),
                'end_time': round(line_notes[-1].end_time, 2)
            }
            lines.append(line)
        
        return lines
    
    lines = []
    phrase_size = 6
    
    for i in range(0, len(notes), phrase_size):
        line_notes = notes[i:i + phrase_size]
        if not line_notes:
            continue
        
        line = {
            'lyrics': '',
            'sargam': '  '.join([n.to_display() for n in line_notes]),
            'notes': [{
                'note': n.formatted,
                'display': n.to_display(),
                'start_time': round(n.start_time, 2),
                'end_time': round(n.end_time, 2),
                'ornament': n.ornament,
                'beat_number': n.beat_number,
            } for n in line_notes],
            'start_time': round(line_notes[0].start_time, 2),
            'end_time': round(line_notes[-1].end_time, 2)
        }
        lines.append(line)
    
    return lines
