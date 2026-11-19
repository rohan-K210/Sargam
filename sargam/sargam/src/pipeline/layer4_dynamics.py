import numpy as np
import librosa
from src.pipeline.base import PipelineLayer, LayerOutput


class DynamicsLayer(PipelineLayer):
    def __init__(self, config: dict = None):
        super().__init__(config)
        dyn_cfg = config.get('dynamics', {}) if config else {}
        self.n_bins = dyn_cfg.get('n_tertile_bins', 3)
        self.attack_window = dyn_cfg.get('attack_window_sec', 0.1)
        self.sustain_pos = dyn_cfg.get('sustain_position', 0.5)
        self.decay_window = dyn_cfg.get('decay_window_sec', 0.2)
        self.attack_thresh = dyn_cfg.get('attack_threshold_sharp', 0.5)
    
    def process(self, audio: np.ndarray, sr: int, context: dict) -> LayerOutput:
        note_segments = context.get('note_segments', [])
        if not note_segments:
            return LayerOutput(False, None, {}, ["No note segments"])
        
        # Compute RMS envelope
        frame_len = 512
        hop_len = 160
        rms = librosa.feature.rms(y=audio, frame_length=frame_len, hop_length=hop_len)[0]
        rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_len)
        
        # Global dynamics bins
        rms_vals = rms[rms > 0]
        if len(rms_vals) > 0:
            thresholds = np.percentile(rms_vals, [100/self.n_bins * i for i in range(1, self.n_bins)])
        else:
            thresholds = [0.1, 0.3]
        
        for segment in note_segments:
            onset = segment['onset_sec']
            duration = segment['duration_sec']
            offset = onset + duration
            
            # RMS in note segment
            mask = (rms_times >= onset) & (rms_times <= offset)
            note_rms = rms[mask] if np.any(mask) else np.array([0])
            
            # Dynamics level
            mean_rms = np.mean(note_rms)
            if mean_rms <= thresholds[0]:
                dyn_level = 'piano'
            elif mean_rms <= thresholds[1] if len(thresholds) > 1 else True:
                dyn_level = 'mezzo'
            else:
                dyn_level = 'forte'
            
            # Attack shape
            attack_samples = int(self.attack_window * sr)
            attack_end = min(int(onset * sr) + attack_samples, len(audio))
            attack_audio = audio[int(onset * sr):attack_end]
            if len(attack_audio) > 10:
                attack_rms = librosa.feature.rms(y=attack_audio, frame_length=256, hop_length=64)[0]
                attack_slope = np.polyfit(np.arange(len(attack_rms)), attack_rms, 1)[0] if len(attack_rms) > 1 else 0
                attack_type = 'sharp' if attack_slope > self.attack_thresh else 'breathy'
            else:
                attack_type = 'unknown'
            
            segment['dynamics'] = dyn_level
            segment['attack'] = attack_type
        
        return LayerOutput(
            success=True,
            data={'note_segments': note_segments},
            metadata={},
            errors=[]
        )