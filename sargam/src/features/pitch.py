import numpy as np
import torch
import librosa
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import yaml


@dataclass
class PitchOutput:
    times: np.ndarray
    frequencies: np.ndarray
    confidence: np.ndarray
    voiced_flag: np.ndarray


class PitchDetector(ABC):
    @abstractmethod
    def predict(self, audio: np.ndarray, sr: int) -> PitchOutput:
        pass


class CrepePitchDetector(PitchDetector):
    def __init__(self, config: dict):
        self.config = config
        self.model_capacity = config.get('model_capacity', 'tiny')
        self.batch_size = config.get('batch_size', 1024)
        self.device = config.get('device', 'cpu')
        self.threshold = config.get('threshold', 0.5)
        self._model = None
    
    def _load_model(self):
        if self._model is None:
            import torchcrepe
            self._model = torchcrepe
    
    def predict(self, audio: np.ndarray, sr: int) -> PitchOutput:
        self._load_model()
        import torchcrepe
        
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
        
        pitch, periodicity = torchcrepe.predict(
            audio_tensor,
            sr,
            hop_length=160,
            fmin=50.0,
            fmax=2000.0,
            model=self.model_capacity,
            batch_size=self.batch_size,
            device=self.device,
            return_periodicity=True
        )
        
        pitch = pitch.squeeze(0).numpy()
        periodicity = periodicity.squeeze(0).numpy()
        
        voiced = periodicity >= self.threshold
        frequencies = np.where(voiced, pitch, 0.0)
        confidence = periodicity
        
        times = librosa.frames_to_time(
            np.arange(len(pitch)), sr=sr, hop_length=160
        )
        
        return PitchOutput(
            times=times,
            frequencies=frequencies,
            confidence=confidence,
            voiced_flag=voiced
        )


class PyinPitchDetector(PitchDetector):
    def __init__(self, config: dict):
        self.config = config
        self.fmin = config.get('fmin', 50.0)
        self.fmax = config.get('fmax', 2000.0)
        self.frame_length = config.get('frame_length', 2048)
        self.hop_length = config.get('hop_length', 160)
    
    def predict(self, audio: np.ndarray, sr: int) -> PitchOutput:
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio,
            fmin=self.fmin,
            fmax=self.fmax,
            frame_length=self.frame_length,
            hop_length=self.hop_length,
            sr=sr
        )
        
        times = librosa.frames_to_time(
            np.arange(len(f0)), sr=sr, hop_length=self.hop_length
        )
        
        frequencies = np.nan_to_num(f0, nan=0.0)
        confidence = np.nan_to_num(voiced_probs, nan=0.0)
        voiced = voiced_flag.astype(bool)
        
        return PitchOutput(
            times=times,
            frequencies=frequencies,
            confidence=confidence,
            voiced_flag=voiced
        )


class EssentiaPitchDetector(PitchDetector):
    def __init__(self, config: dict):
        self.config = config
        self.enabled = config.get('enabled', False)
    
    def predict(self, audio: np.ndarray, sr: int) -> PitchOutput:
        if not self.enabled:
            raise NotImplementedError("Essentia not enabled in config")
        raise NotImplementedError("Essentia detector not yet implemented")


def correct_octave_errors(output: PitchOutput, window: int = 9, tol_cents: float = 150.0) -> PitchOutput:
    """
    Correct isolated octave errors in a pitch contour: pitch trackers
    frequently lock onto a harmonic (2x/0.5x the true fundamental) for a few
    frames right after a silence/breath gap, before the true pitch stabilizes.
    A single such frame produces an implausible ~1200-cents jump that never
    happens in real playing, and downstream (note segmentation, ornament
    detection) reads it as either a wrong note or a huge fake "meend" slide.

    For each voiced frame, compares it to the median of nearby voiced frames
    (in log2/cents space) and snaps it back an integer number of octaves if
    that brings it close to its neighbors.
    """
    freqs = output.frequencies.copy()
    voiced = output.voiced_flag
    n = len(freqs)
    half = window // 2

    voiced_idx = np.where(voiced & (freqs > 0))[0]
    if len(voiced_idx) < 3:
        return output

    cents = np.zeros(n)
    cents[voiced_idx] = 1200.0 * np.log2(freqs[voiced_idx])

    for i in voiced_idx:
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        neighbor_idx = [j for j in range(lo, hi) if j != i and voiced[j] and freqs[j] > 0]
        if len(neighbor_idx) < 3:
            continue

        local_median = np.median(cents[neighbor_idx])
        diff = cents[i] - local_median
        k = round(diff / 1200.0)
        if k != 0 and abs(diff - k * 1200.0) < tol_cents:
            freqs[i] = freqs[i] / (2.0 ** k)
            cents[i] = 1200.0 * np.log2(freqs[i])

    return PitchOutput(
        times=output.times,
        frequencies=freqs,
        confidence=output.confidence,
        voiced_flag=output.voiced_flag
    )


def create_pitch_detector(detector_type: str, config: dict) -> PitchDetector:
    detectors = {
        'crepe': CrepePitchDetector,
        'pyin': PyinPitchDetector,
        'essentia': EssentiaPitchDetector,
    }
    
    if detector_type not in detectors:
        raise ValueError(f"Unknown detector: {detector_type}")
    
    return detectors[detector_type](config)


def load_pitch_config(config_path: str = "config/model_config.yaml") -> dict:
    if not Path(config_path).is_absolute():
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / config_path
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg.get('pitch', {})