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