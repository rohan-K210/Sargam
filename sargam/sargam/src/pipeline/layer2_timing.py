import numpy as np
import librosa
from src.pipeline.base import PipelineLayer, LayerOutput


class TimingLayer(PipelineLayer):
    def __init__(self, config: dict = None):
        super().__init__(config)
        onset_cfg = config.get('onset', {}) if config else {}
        self.method = onset_cfg.get('method', 'spectral_flux')
        flux_cfg = onset_cfg.get('spectral_flux', {})
        self.n_mels = flux_cfg.get('n_mels', 64)
        self.fmin = flux_cfg.get('fmin', 50)
        self.fmax = flux_cfg.get('fmax', 8000)
        self.pre_max = flux_cfg.get('pre_max', 3)
        self.post_max = flux_cfg.get('post_max', 3)
        self.delta = flux_cfg.get('delta', 0.1)
        self.wait = flux_cfg.get('wait', 10)
        self.min_note_duration = config.get('min_note_duration', 0.05) if config else 0.05
    
    def process(self, audio: np.ndarray, sr: int, context: dict) -> LayerOutput:
        # Get swara notes from previous layer
        swara_notes = context.get('swara_notes', [])
        if not swara_notes:
            return LayerOutput(False, None, {}, ["No swara notes from previous layer"])
        
        pitch_data = context.get('pitch')
        pitch_times = pitch_data.times if pitch_data else None
        
        # Onset detection
        onsets = self._detect_onsets(audio, sr)
        
        # Segment notes using svara changes with pitch times
        note_segments = self._segment_notes(swara_notes, onsets, pitch_times)
        
        # Tempo estimation
        tempo_bpm, laya = self._estimate_tempo(onsets, sr)
        
        return LayerOutput(
            success=True,
            data={
                'onsets': onsets,
                'note_segments': note_segments,
                'tempo_bpm': tempo_bpm,
                'laya': laya
            },
            metadata={'onset_method': self.method},
            errors=[]
        )
    
    def _detect_onsets(self, audio: np.ndarray, sr: int) -> np.ndarray:
        # Spectral flux on mel spectrogram
        onset_env = librosa.onset.onset_strength(
            y=audio, sr=sr,
            n_mels=self.n_mels, fmin=self.fmin, fmax=self.fmax
        )
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_env, sr=sr,
            pre_max=self.pre_max, post_max=self.post_max,
            delta=self.delta, wait=self.wait
        )
        return librosa.frames_to_time(onset_frames, sr=sr)
    
    def _segment_notes(self, swara_notes, onsets, pitch_times=None):
        # Segment notes based on svara changes with time from pitch_times
        segments = []
        current_svara = None
        current_variant = None
        current_octave = None
        start_idx = 0
        
        if pitch_times is None:
            pitch_times = np.arange(len(swara_notes)) * 0.01  # 10ms hop
        
        for i, note in enumerate(swara_notes):
            svara = note.svara
            variant = getattr(note, 'variant', '')
            octave = getattr(note, 'octave', '')
            
            if svara != current_svara:
                if current_svara is not None:
                    duration = pitch_times[i] - pitch_times[start_idx]
                    if duration >= self.min_note_duration:
                        segments.append({
                            'swara': current_svara,
                            'variant': current_variant,
                            'octave': current_octave,
                            'onset_sec': float(pitch_times[start_idx]),
                            'duration_sec': float(duration)
                        })
                current_svara = svara
                current_variant = variant
                current_octave = octave
                start_idx = i
        
        # Add final segment
        if current_svara is not None and start_idx < len(pitch_times) - 1:
            duration = pitch_times[-1] - pitch_times[start_idx]
            if duration >= self.min_note_duration:
                segments.append({
                    'swara': current_svara,
                    'variant': current_variant,
                    'octave': current_octave,
                    'onset_sec': float(pitch_times[start_idx]),
                    'duration_sec': float(duration)
                })
        
        return segments
    
    def _estimate_tempo(self, onsets, sr):
        if len(onsets) < 2:
            return 120.0, "Madhya"
        
        iois = np.diff(onsets)
        median_ioi = np.median(iois)
        tempo = 60.0 / median_ioi if median_ioi > 0 else 120.0
        
        # Laya classification
        if median_ioi > 1.5:
            laya = "Vilambit"
        elif median_ioi > 0.5:
            laya = "Madhya"
        else:
            laya = "Drut"
        
        return float(tempo), laya