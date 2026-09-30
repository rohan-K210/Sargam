"""
Ornament detection for Indian classical music.
Detects meend (glides) and gamak (oscillations) from pYIN pitch contour.
"""

import numpy as np
from typing import List, Dict


def detect_meend(f0: np.ndarray, times: np.ndarray,
                 min_duration: float = 0.1,
                 min_pitch_change: float = 1.5) -> List[Dict]:
    """
    Detect meend (smooth pitch glides between notes).
    """
    ornaments = []
    n = len(f0)
    i = 0
    
    while i < n:
        if np.isnan(f0[i]):
            i += 1
            continue
        
        start_idx = i
        start_freq = f0[i]
        start_time = times[i]
        
        # Find continuous voiced segment
        j = i + 1
        while j < n and not np.isnan(f0[j]):
            j += 1
        
        if j > i + 1:
            segment_f0 = f0[i:j]
            segment_times = times[i:j]
            duration = segment_times[-1] - segment_times[0]
            
            if duration >= min_duration:
                # Calculate pitch change in semitones
                pitch_start = 12 * np.log2(segment_f0[0] / 261.63)  # relative to C4
                pitch_end = 12 * np.log2(segment_f0[-1] / 261.63)
                pitch_change = abs(pitch_end - pitch_start)
                
                if pitch_change >= min_pitch_change:
                    ornaments.append({
                        'start_time': float(segment_times[0]),
                        'end_time': float(segment_times[-1]),
                        'start_freq': float(segment_f0[0]),
                        'end_freq': float(segment_f0[-1]),
                        'pitch_change_semitones': float(pitch_change),
                        'type': 'meend'
                    })
        
        i = j + 1
    
    return ornaments


def detect_gamak(f0: np.ndarray, times: np.ndarray,
                 min_oscillations: int = 3,
                 freq_range_semitones: float = 0.5) -> List[Dict]:
    """
    Detect gamak (rapid pitch oscillations around a note).
    """
    ornaments = []
    n = len(f0)
    window_size = 10  # frames
    
    for i in range(window_size, n - window_size):
        if np.isnan(f0[i]):
            continue
        
        window_f0 = f0[i-window_size:i+window_size]
        valid = window_f0[~np.isnan(window_f0)]
        
        if len(valid) < window_size:
            continue
        
        # Calculate pitch in semitones
        pitches = 12 * np.log2(valid / 261.63)
        
        # Find peaks and valleys (oscillations)
        diffs = np.diff(pitches)
        sign_changes = np.where(np.diff(np.sign(diffs)))[0]
        
        if len(sign_changes) >= min_oscillations * 2:
            pitch_range = np.max(pitches) - np.min(pitches)
            if pitch_range <= freq_range_semitones * 2:
                center_pitch = np.median(pitches)
                ornaments.append({
                    'start_time': float(times[i-window_size]),
                    'end_time': float(times[i+window_size]),
                    'center_freq': float(261.63 * 2**(center_pitch/12)),
                    'pitch_range_semitones': float(pitch_range),
                    'oscillation_count': len(sign_changes) // 2,
                    'type': 'gamak'
                })
    
    # Merge nearby gamaks
    merged = []
    for o in ornaments:
        if merged and o['start_time'] - merged[-1]['end_time'] < 0.1:
            merged[-1]['end_time'] = o['end_time']
        else:
            merged.append(o)
    
    return merged


def annotate_ornaments(note_events: List, ornaments: List) -> List:
    """
    Add ornament markers to note events for display.
    Meend: add '~' between notes (e.g., "Sa~Re")
    Gamak: add '~~~' after note (e.g., "Ga~~~")
    """
    # For now, return note_events as-is
    # Full integration happens in melody.py
    return note_events