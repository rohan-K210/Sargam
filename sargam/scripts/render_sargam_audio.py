"""
Render a sargam pipeline JSON output back into audio, for listening tests.
Not part of the production pipeline -- a QA/preview tool.

Usage: python scripts/render_sargam_audio.py <sargam_output.json> <out.wav>
"""
import sys
import json
import numpy as np
from scipy.signal import butter, lfilter
import soundfile as sf

sys.path.insert(0, '.')
from src.mapping.swara_mapper import SwaraMapper

SR = 22050
GAP_CONNECT_THRESHOLD = 0.15
GLIDE_MAX = 0.12
DYN_AMP = {'piano': 0.30, 'mezzo': 0.62, 'forte': 1.0}
HARMONICS = [(1, 1.0), (2, 0.5), (3, 0.22), (4, 0.10), (5, 0.05), (6, 0.025)]
VIBRATO_HZ = 5.5
VIBRATO_CENTS = 12.0
VIBRATO_MIN_NOTE_SEC = 0.18


def note_freq(mapper, tonic_hz, swara, variant, octave):
    key = f'{swara}_{variant}' if variant else swara
    cents = mapper.svara_cents.get(key, 0)
    if octave == 'Mandra':
        cents -= 1200
    elif octave == 'Taar':
        cents += 1200
    return tonic_hz * (2 ** (cents / 1200.0))


def bandpass_noise(n_samples, sr, lo=1500, hi=3500):
    noise = np.random.randn(n_samples)
    b, a = butter(4, [lo / (sr / 2), hi / (sr / 2)], btype='band')
    return lfilter(b, a, noise)


def render(data, out_path):
    mapper = SwaraMapper()
    tonic_hz = data['tonic_hz']
    notes = sorted(data['notes'], key=lambda n: n['onset_sec'])
    total_dur = data['audio_info']['duration_sec'] + 2.0
    n_samples = int(total_dur * SR)

    freq_contour = np.full(n_samples, tonic_hz)
    amp_envelope = np.zeros(n_samples)
    breathy_mask = np.zeros(n_samples, dtype=bool)

    idxs = []
    for n in notes:
        s = int(n['onset_sec'] * SR)
        e = s + int(max(n['duration_sec'], 0.05) * SR)
        e = min(e, n_samples)
        if e <= s:
            continue
        f = note_freq(mapper, tonic_hz, n['swara'], n['variant'], n['octave'])
        a = DYN_AMP.get(n['dynamics'], 0.62)
        freq_contour[s:e] = f
        amp_envelope[s:e] = a
        if n.get('attack') == 'breathy':
            breathy_mask[s:e] = True
        idxs.append((s, e, f, a, n))

    # continuous vibrato on notes long enough to actually show it
    t = np.arange(n_samples) / SR
    vibrato_cents = VIBRATO_CENTS * np.sin(2 * np.pi * VIBRATO_HZ * t)
    for s, e, f, a, n in idxs:
        if n['duration_sec'] >= VIBRATO_MIN_NOTE_SEC:
            ramp_len = min(int(0.08 * SR), (e - s) // 2)
            depth = np.ones(e - s)
            if ramp_len > 0:
                depth[:ramp_len] = np.linspace(0, 1, ramp_len)
            freq_contour[s:e] *= 2 ** (vibrato_cents[s:e] * depth / 1200.0)

    # glide between connected consecutive notes; fade at real gaps
    for i in range(len(idxs) - 1):
        s0, e0, f0, a0, n0 = idxs[i]
        s1, e1, f1, a1, n1 = idxs[i + 1]
        gap = n1['onset_sec'] - (n0['onset_sec'] + n0['duration_sec'])
        if gap <= GAP_CONNECT_THRESHOLD:
            glide_dur = min(GLIDE_MAX, n0['duration_sec'] / 2, n1['duration_sec'] / 2)
            glide_len = max(int(glide_dur * SR), 8)
            mid = s1
            lo = max(s0, mid - glide_len // 2)
            hi = min(n_samples, mid + glide_len // 2)
            if hi > lo:
                alpha = np.linspace(0, 1, hi - lo)
                freq_contour[lo:hi] = np.exp(np.log(f0) * (1 - alpha) + np.log(f1) * alpha)
                amp_envelope[lo:hi] = a0 * (1 - alpha) + a1 * alpha
        else:
            fade = min(int(0.05 * SR), (e0 - s0) // 2, (e1 - s1) // 2)
            if fade > 0:
                amp_envelope[e0 - fade:e0] *= np.linspace(1, 0, fade)
                amp_envelope[s1:s1 + fade] *= np.linspace(0, 1, fade)

    if idxs:
        s0, e0, f0, a0, n0 = idxs[0]
        fade = min(int(0.06 * SR), e0 - s0)
        if fade > 0:
            amp_envelope[s0:s0 + fade] *= (np.linspace(0, 1, fade)) ** 1.5
        sN, eN, fN, aN, nN = idxs[-1]
        fade = min(int(0.1 * SR), eN - sN)
        if fade > 0:
            amp_envelope[eN - fade:eN] *= (np.linspace(1, 0, fade)) ** 1.5

    # gentle amplitude tremolo for a less static, more "breathing" tone
    tremolo = 1.0 + 0.05 * np.sin(2 * np.pi * 4.3 * t)
    amp_envelope = amp_envelope * tremolo

    phase = 2 * np.pi * np.cumsum(freq_contour) / SR
    tone = np.zeros(n_samples)
    for h, amp in HARMONICS:
        tone += amp * np.sin(phase * h)
    tone /= sum(a for _, a in HARMONICS)

    breath = bandpass_noise(n_samples, SR)
    breath_level = np.where(breathy_mask, 0.09, 0.035)
    sig = tone * amp_envelope + breath * breath_level * amp_envelope

    peak = np.max(np.abs(sig))
    if peak > 0:
        sig = sig / peak * 0.9

    sf.write(out_path, sig.astype(np.float32), SR)
    print(f'wrote {out_path}  ({n_samples / SR:.1f}s)')


if __name__ == '__main__':
    json_path = sys.argv[1] if len(sys.argv) > 1 else 'sargam_output.json'
    out_path = sys.argv[2] if len(sys.argv) > 2 else 'sargam_render.wav'
    with open(json_path) as f:
        data = json.load(f)
    render(data, out_path)
