"""
Source separation: isolates a melodic instrument (e.g. flute) from a full
mix (vocals + drums + bass + other instruments) before pitch tracking, using
torchaudio's pretrained HDemucs model.

Spleeter (used by the sibling sargam_backend project) depends on TensorFlow,
which doesn't support Python 3.12+. HDemucs is PyTorch-native, ships with
torchaudio (no extra heavy dependency), and is generally considered
higher-quality than Spleeter today, so it's used here instead.
"""
import numpy as np
import torch
import torchaudio

try:
    import torchaudio.pipelines as _pipelines
    SEPARATION_AVAILABLE = True
except Exception:
    SEPARATION_AVAILABLE = False

_MODEL = None
_MODEL_SR = None
_SOURCES = None


def _load_model():
    global _MODEL, _MODEL_SR, _SOURCES
    if _MODEL is None:
        bundle = _pipelines.HDEMUCS_HIGH_MUSDB_PLUS
        _MODEL = bundle.get_model()
        _MODEL.eval()
        _MODEL_SR = bundle.sample_rate
        _SOURCES = _MODEL.sources
    return _MODEL, _MODEL_SR, _SOURCES


def _separate_sources(model, mix, sample_rate, segment=10.0, overlap_sec=0.5):
    """
    Apply the model to long audio in overlapping chunks, weighted-blending
    the overlaps (triangular fade, normalized by total accumulated weight)
    so chunk boundaries don't produce audible clicks or amplitude dips.
    mix: (channels, samples) tensor.
    """
    channels, length = mix.shape[0], mix.shape[-1]
    chunk_samples = int(sample_rate * segment)
    overlap_samples = min(int(sample_rate * overlap_sec), chunk_samples // 2)
    hop_samples = chunk_samples - overlap_samples

    fade = torch.ones(chunk_samples)
    if overlap_samples > 0:
        ramp = torch.linspace(0, 1, overlap_samples)
        fade[:overlap_samples] = ramp
        fade[-overlap_samples:] = ramp.flip(0)

    n_sources = len(model.sources)
    final = torch.zeros(n_sources, channels, length)
    weight_sum = torch.zeros(length)

    start = 0
    while start < length:
        end = min(start + chunk_samples, length)
        chunk = mix[:, start:end]
        pad_len = chunk_samples - chunk.shape[-1]
        if pad_len > 0:
            chunk = torch.nn.functional.pad(chunk, (0, pad_len))

        with torch.no_grad():
            out = model(chunk.unsqueeze(0))[0]  # (n_sources, channels, chunk_samples)

        seg_len = end - start
        w = fade[:seg_len]
        final[:, :, start:end] += out[:, :, :seg_len] * w
        weight_sum[start:end] += w

        if end >= length:
            break
        start += hop_samples

    weight_sum = weight_sum.clamp(min=1e-8)
    final = final / weight_sum
    return final


def separate_stems(audio: np.ndarray, sr: int, segment: float = 10.0) -> dict:
    """
    Separate mono audio into stems: vocals, drums, bass, other.

    Args:
        audio: mono float32 array
        sr: sample rate of `audio`
        segment: chunk length in seconds for chunked processing (keeps
            memory bounded on long tracks)

    Returns:
        {stem_name: (audio_array, sample_rate)} at the model's native
        sample rate (44100), or {} if separation is unavailable.
    """
    if not SEPARATION_AVAILABLE:
        return {}

    model, model_sr, sources = _load_model()

    wav = torch.from_numpy(audio).float()
    if sr != model_sr:
        wav = torchaudio.functional.resample(wav, sr, model_sr)

    # HDemucs expects stereo; duplicate the mono channel
    mix = wav.unsqueeze(0).repeat(2, 1)

    ref = mix.mean(0)
    mean, std = ref.mean(), ref.std()
    mix_norm = (mix - mean) / (std + 1e-8)

    sources_out = _separate_sources(model, mix_norm, model_sr, segment=segment)
    sources_out = sources_out * std + mean

    stems = {}
    for i, name in enumerate(sources):
        stem_audio = sources_out[i].mean(0).numpy()  # back to mono
        stems[name] = (stem_audio.astype(np.float32), model_sr)
    return stems


def isolate_melodic_instrument(audio: np.ndarray, sr: int, segment: float = 10.0) -> tuple:
    """
    Convenience wrapper for the flute-practice use case: returns the 'other'
    stem (melodic instruments minus vocals/drums/bass), which is where a
    flute line ends up after separation. Falls back to the original audio,
    unseparated, if separation isn't available.
    """
    stems = separate_stems(audio, sr, segment=segment)
    if not stems:
        return audio, sr, False
    stem_audio, stem_sr = stems.get('other', (None, None))
    if stem_audio is None:
        return audio, sr, False
    return stem_audio, stem_sr, True
