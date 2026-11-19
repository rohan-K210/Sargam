import numpy as np
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Tuple


@dataclass
class SwaraNote:
    svara: str
    variant: str
    octave: str
    target_cents: float
    actual_cents: float
    error_cents: float
    confidence: float
    finger_hint: str = ""


class SwaraMapper:
    def __init__(self, config_path: str = "config/swara_mapping.yaml"):
        if not Path(config_path).is_absolute():
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / config_path
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.svara_cents = self.config['svara_cents']
        self.variant_labels = self.config['variant_labels']
        self.octave_boundaries = self.config['octave_boundaries']
        self.octave_labels = self.config['octave_labels']
        self.display_names = self.config['display_names']
        self.finger_hints = self.config['finger_hints']
    
    def hz_to_cents(self, hz: float, tonic_hz: float) -> float:
        """Convert frequency to cents relative to tonic."""
        if hz <= 0 or tonic_hz <= 0:
            return 0.0
        return 1200.0 * np.log2(hz / tonic_hz)
    
    def cents_to_octave(self, cents: float) -> str:
        """Determine octave from cents (fallback boundary check, unused by
        the main per-frame path, which derives octave from the nearest-svara
        octave shift instead)."""
        if cents < self.octave_boundaries['mandra_max']:
            return self.octave_labels['mandra']
        elif cents < self.octave_boundaries['madhya_max']:
            return self.octave_labels['madhya']
        else:
            return self.octave_labels['taar']

    def octave_shift_to_label(self, octave_shift: int) -> str:
        """Map the octave shift chosen during nearest-svara matching to a label."""
        if octave_shift < 0:
            return self.octave_labels['mandra']
        elif octave_shift > 0:
            return self.octave_labels['taar']
        else:
            return self.octave_labels['madhya']
    
    def cents_to_svara(self, cents: float) -> Tuple[str, str, float, int]:
        """
        Map cents to nearest svara.

        Returns:
            svara_name, variant, target_cents, octave_shift
        """
        # Find closest svara, searching across octave-shifted positions so the
        # same nearest-neighbor decision determines both svara and octave.
        min_dist = float('inf')
        best_svara = 'Sa'
        best_variant = ''
        best_target = 0.0
        best_octave_shift = 0

        for svara_key, target_cents in self.svara_cents.items():
            for octave_shift in [-1200, 0, 1200]:
                shifted_target = target_cents + octave_shift
                dist_shifted = abs(cents - shifted_target)
                if dist_shifted < min_dist:
                    min_dist = dist_shifted
                    best_svara = svara_key.split('_')[0]
                    best_variant = self.variant_labels.get(svara_key, '')
                    best_target = shifted_target
                    best_octave_shift = octave_shift

        return best_svara, best_variant, best_target, best_octave_shift
    
    def map_pitch_contour(
        self,
        frequencies: np.ndarray,
        confidence: np.ndarray,
        tonic_hz: float,
        smoothing_window: int = 5
    ) -> List[SwaraNote]:
        """Map pitch contour to swara sequence."""
        notes = []
        
        for freq, conf in zip(frequencies, confidence):
            if freq <= 0 or conf < 0.3:
                # Unvoiced or low confidence
                notes.append(SwaraNote(
                    svara='',
                    variant='',
                    octave='',
                    target_cents=0.0,
                    actual_cents=0.0,
                    error_cents=0.0,
                    confidence=float(conf)
                ))
                continue
            
            cents = self.hz_to_cents(freq, tonic_hz)
            svara, variant, target_cents, octave_shift = self.cents_to_svara(cents)
            octave = self.octave_shift_to_label(octave_shift)
            error = cents - target_cents
            
            notes.append(SwaraNote(
                svara=svara,
                variant=variant,
                octave=octave,
                target_cents=float(target_cents),
                actual_cents=float(cents),
                error_cents=float(error),
                confidence=float(conf),
                finger_hint=self.finger_hints.get(f"{svara}_{variant}" if variant else svara, '')
            ))
        
        # Smooth with median filter
        if smoothing_window > 1:
            notes = self._smooth_notes(notes, smoothing_window)
        
        return notes
    
    def _smooth_notes(self, notes: List[SwaraNote], window: int) -> List[SwaraNote]:
        """Apply median filtering to svara sequence."""
        if len(notes) < window:
            return notes
        
        smoothed = []
        half = window // 2
        
        for i in range(len(notes)):
            start = max(0, i - half)
            end = min(len(notes), i + half + 1)
            
            # Vote on the full (svara, variant, octave) combination as one
            # unit -- voting on each field independently can stitch together
            # a combination (e.g. "Sa" + "tivra") that never actually
            # co-occurred in any single frame.
            valid_combos = [
                (n.svara, n.variant, n.octave) for n in notes[start:end] if n.svara
            ]

            if valid_combos:
                svara, variant, octave = max(set(valid_combos), key=valid_combos.count)

                # Recompute target cents for smoothed svara
                svara_key = f"{svara}_{variant}" if variant else svara
                target_cents = self.svara_cents.get(svara_key, 0)
                # Adjust for octave
                if octave == self.octave_labels['mandra']:
                    target_cents -= 1200
                elif octave == self.octave_labels['taar']:
                    target_cents += 1200
                
                actual = notes[i].actual_cents
                error = actual - target_cents if actual else 0
                
                smoothed.append(SwaraNote(
                    svara=svara,
                    variant=variant,
                    octave=octave,
                    target_cents=float(target_cents),
                    actual_cents=float(actual),
                    error_cents=float(error),
                    confidence=notes[i].confidence,
                    finger_hint=self.finger_hints.get(svara_key, '')
                ))
            else:
                smoothed.append(notes[i])
        
        return smoothed
    
    def notes_to_json(self, notes: List[SwaraNote], times: np.ndarray) -> List[Dict]:
        """Convert notes to JSON-serializable format."""
        result = []
        for note, t in zip(notes, times):
            if not note.svara:
                continue
            result.append({
                'time_sec': float(t),
                'swara': note.svara,
                'variant': note.variant,
                'octave': note.octave,
                'display': self.display_names.get(f"{note.svara}_{note.variant}" if note.variant else note.svara, note.svara),
                'target_cents': note.target_cents,
                'actual_cents': note.actual_cents,
                'error_cents': note.error_cents,
                'confidence': note.confidence,
                'finger_hint': note.finger_hint
            })
        return result


def load_swara_config(config_path: str = "config/swara_mapping.yaml") -> dict:
    if not Path(config_path).is_absolute():
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / config_path
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)