import numpy as np
from scipy.signal import find_peaks
from typing import Tuple, Optional
import yaml


def detect_tonic_from_pitch(
    frequencies: np.ndarray,
    confidence: np.ndarray,
    voiced_flag: np.ndarray,
    config: dict = None
) -> Tuple[float, float, str]:
    """
    Detect tonic (Sa) from pitch contour using histogram peak picking.
    
    Returns:
        tonic_hz: Detected tonic frequency
        tonic_confidence: Confidence score (0-1)
        tonic_note: Note name (e.g., "C#")
    """
    if config is None:
        config = {}
    
    histogram_bins = config.get('histogram_bins', 240)
    min_salience_ratio = config.get('min_salience_ratio', 0.05)
    refinement_window = config.get('refinement_window_cents', 50)
    validation_threshold = config.get('validation_threshold', 0.8)
    
    # Filter voiced frames with good confidence
    valid = voiced_flag & (confidence > 0.5)
    if not np.any(valid):
        return 440.0, 0.0, "A4"
    
    voiced_freqs = frequencies[valid]
    voiced_conf = confidence[valid]
    
    # Convert to cents relative to A4 (440 Hz)
    cents = 1200 * np.log2(voiced_freqs / 440.0)
    
    # Build weighted histogram
    hist_range = (-1200, 1200)  # ±1 octave
    hist, bin_edges = np.histogram(
        cents, bins=histogram_bins, range=hist_range, weights=voiced_conf
    )
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Find peaks
    min_height = np.max(hist) * min_salience_ratio
    peaks, properties = find_peaks(hist, height=min_height, distance=20)
    
    if len(peaks) == 0:
        return 440.0, 0.0, "A4"
    
    # Select highest peak
    best_peak_idx = peaks[np.argmax(hist[peaks])]
    tonic_cents = bin_centers[best_peak_idx]
    
    # Refine: weighted mean of frames near peak
    peak_low = tonic_cents - refinement_window
    peak_high = tonic_cents + refinement_window
    mask = (cents >= peak_low) & (cents <= peak_high)
    if np.any(mask):
        tonic_cents = np.average(cents[mask], weights=voiced_conf[mask])
    
    # Convert back to Hz
    tonic_hz = 440.0 * (2.0 ** (tonic_cents / 1200.0))
    
    # Confidence: peak salience / total salience
    tonic_confidence = float(hist[best_peak_idx] / np.sum(hist))
    
    # Note name
    tonic_note = hz_to_note_name(tonic_hz)
    
    return tonic_hz, tonic_confidence, tonic_note


def hz_to_note_name(hz: float) -> str:
    """Convert Hz to note name (e.g., C#4)."""
    if hz <= 0:
        return "A4"
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    midi = int(round(69 + 12 * np.log2(hz / 440.0)))
    note = note_names[midi % 12]
    octave = midi // 12 - 1
    return f"{note}{octave}"


def validate_tonic(
    tonic_hz: float,
    frequencies: np.ndarray,
    confidence: np.ndarray,
    voiced_flag: np.ndarray,
    swara_mapper,
    threshold: float = 0.8
) -> Tuple[float, bool]:
    """
    Validate tonic by checking how many frames map to valid svaras.
    Returns adjusted tonic and validity flag.
    """
    # This would use the swara_mapper to check mapping quality
    # For now, return as-is
    return tonic_hz, True


def detect_tonic_initial_phrase(
    frequencies: np.ndarray,
    confidence: np.ndarray,
    voiced_flag: np.ndarray,
    times: np.ndarray,
    config: dict = None,
    phrase_duration: float = 5.0
) -> Tuple[float, float, str]:
    """
    Detect tonic from initial phrase (first N seconds).
    Useful for recordings where alap establishes tonic.
    """
    # Filter to first phrase_duration seconds
    mask = times <= phrase_duration
    return detect_tonic_from_pitch(
        frequencies[mask], confidence[mask], voiced_flag[mask], config
    )