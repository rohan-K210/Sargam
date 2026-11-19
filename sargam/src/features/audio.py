import numpy as np
import librosa
import yaml
from pathlib import Path
from typing import Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class AudioInfo:
    """Metadata about processed audio."""
    original_sr: int
    original_duration: float
    processed_sr: int
    processed_duration: float
    n_channels: int
    was_trimmed: bool
    trim_start: float
    trim_end: float


def load_config(config_path: str = "config/model_config.yaml") -> dict:
    """Load model configuration."""
    # Resolve relative to project root
    if not Path(config_path).is_absolute():
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / config_path
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_audio(
    path: str,
    config: Optional[dict] = None,
    target_sr: int = 16000
) -> Tuple[np.ndarray, int, AudioInfo]:
    """
    Load and preprocess audio file.
    
    Returns:
        audio: Preprocessed mono audio at target_sr
        sr: Sample rate (target_sr)
        info: AudioInfo metadata
    """
    if config is None:
        config = load_config()
    
    audio_cfg = config.get('audio', {})
    target_sr = audio_cfg.get('sample_rate', target_sr)
    mono = audio_cfg.get('mono', True)
    trim_top_db = audio_cfg.get('trim_top_db', 30)
    preemphasis = audio_cfg.get('preemphasis', 0.97)
    normalization = audio_cfg.get('normalization', 'peak')
    target_dbfs = audio_cfg.get('target_dbfs', -1.0)
    
    # Load audio
    audio, orig_sr = librosa.load(path, sr=None, mono=mono)
    orig_duration = len(audio) / orig_sr
    n_channels = 1 if audio.ndim == 1 else audio.shape[0]
    
    # Resample if needed
    if orig_sr != target_sr:
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
    
    # Trim silence
    was_trimmed = False
    trim_start = 0.0
    trim_end = len(audio) / target_sr
    
    if trim_top_db > 0:
        audio_trimmed, (start_idx, end_idx) = librosa.effects.trim(
            audio, top_db=trim_top_db
        )
        if len(audio_trimmed) < len(audio):
            was_trimmed = True
            trim_start = start_idx / target_sr
            trim_end = end_idx / target_sr
            audio = audio_trimmed
    
    # Pre-emphasis
    if preemphasis > 0:
        audio = np.append(audio[0], audio[1:] - preemphasis * audio[:-1])
    
    # Normalization
    if normalization == 'peak':
        peak = np.max(np.abs(audio))
        if peak > 0:
            target_amp = 10 ** (target_dbfs / 20)
            audio = audio * (target_amp / peak)
    elif normalization == 'rms':
        rms = np.sqrt(np.mean(audio ** 2))
        if rms > 0:
            target_rms = 10 ** (target_dbfs / 20)
            audio = audio * (target_rms / rms)
    
    proc_duration = len(audio) / target_sr
    
    info = AudioInfo(
        original_sr=orig_sr,
        original_duration=orig_duration,
        processed_sr=target_sr,
        processed_duration=proc_duration,
        n_channels=n_channels,
        was_trimmed=was_trimmed,
        trim_start=trim_start,
        trim_end=trim_end
    )
    
    return audio.astype(np.float32), target_sr, info


def frame_audio(
    audio: np.ndarray,
    sr: int,
    config: Optional[dict] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Frame audio into overlapping windows.
    
    Returns:
        frames: (n_frames, frame_length)
        times: Frame center times in seconds
    """
    if config is None:
        config = load_config()
    
    audio_cfg = config.get('audio', {})
    frame_length = audio_cfg.get('frame_length', 2048)
    hop_length = audio_cfg.get('hop_length', 160)
    window = audio_cfg.get('window', 'hann')
    
    frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length).T
    times = librosa.frames_to_time(
        np.arange(frames.shape[0]), sr=sr, hop_length=hop_length
    )
    
    # Apply window
    if window == 'hann':
        win = np.hanning(frame_length)
    elif window == 'hamming':
        win = np.hamming(frame_length)
    else:
        win = np.ones(frame_length)
    
    frames = frames * win
    
    return frames, times


def chunk_audio(
    audio: np.ndarray,
    sr: int,
    max_duration: float = 30.0,
    overlap: float = 1.0
) -> Tuple[list, list]:
    """
    Split long audio into overlapping chunks.
    
    Returns:
        chunks: List of audio chunks
        chunk_times: List of (start_time, end_time) for each chunk
    """
    max_samples = int(max_duration * sr)
    overlap_samples = int(overlap * sr)
    
    if len(audio) <= max_samples:
        return [audio], [(0.0, len(audio) / sr)]
    
    chunks = []
    chunk_times = []
    
    start = 0
    while start < len(audio):
        end = min(start + max_samples, len(audio))
        chunks.append(audio[start:end])
        chunk_times.append((start / sr, end / sr))
        
        if end == len(audio):
            break
        start += max_samples - overlap_samples
    
    return chunks, chunk_times