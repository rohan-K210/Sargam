import json
import base64
import numpy as np
from pathlib import Path
from typing import Any, Dict, Union, Optional
import soundfile as sf
import io


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def load_audio_base64(base64_string: str, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    """Load audio from base64 encoded string."""
    audio_bytes = base64.b64decode(base64_string)
    audio, sr = sf.read(io.BytesIO(audio_bytes))
    
    if sr != target_sr:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    
    return audio.astype(np.float32), sr


def save_audio_base64(audio: np.ndarray, sr: int, format: str = "WAV") -> str:
    """Save audio to base64 encoded string."""
    buffer = io.BytesIO()
    sf.write(buffer, audio, sr, format=format)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def load_json(path: Union[str, Path]) -> Dict:
    """Load JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def save_json(data: Dict, path: Union[str, Path], indent: int = 2) -> None:
    """Save JSON file with numpy support."""
    with open(path, 'w') as f:
        json.dump(data, f, cls=NumpyEncoder, indent=indent)


def load_yaml(path: Union[str, Path]) -> Dict:
    """Load YAML file."""
    import yaml
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def save_yaml(data: Dict, path: Union[str, Path]) -> None:
    """Save YAML file."""
    import yaml
    with open(path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)


def audio_to_base64_file(audio_path: Union[str, Path]) -> str:
    """Convert audio file to base64 string."""
    with open(audio_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')