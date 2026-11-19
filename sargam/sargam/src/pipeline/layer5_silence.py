import numpy as np
import librosa
from src.pipeline.base import PipelineLayer, LayerOutput


class SilenceBreathLayer(PipelineLayer):
    def __init__(self, config: dict = None):
        super().__init__(config)
        sil_cfg = config.get('silence', {}) if config else {}
        self.short_gap_max = sil_cfg.get('short_gap_max_sec', 0.2)
        self.phrase_gap_max = sil_cfg.get('phrase_gap_max_sec', 1.0)
        breath_cfg = sil_cfg.get('breath', {})
        self.breath_min_dur = breath_cfg.get('min_duration_sec', 0.4)
        self.breath_max_dur = breath_cfg.get('max_duration_sec', 1.5)
        self.breath_min_centroid = breath_cfg.get('min_spectral_centroid_hz', 3000)
        self.pre_gap_decay = breath_cfg.get('pre_gap_decay_threshold', 0.3)
    
    def process(self, audio: np.ndarray, sr: int, context: dict) -> LayerOutput:
        note_segments = context.get('note_segments', [])
        if not note_segments:
            return LayerOutput(False, None, {}, ["No note segments"])
        
        # Compute spectral centroid for breath detection
        centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        centroid_times = librosa.frames_to_time(np.arange(len(centroid)), sr=sr)
        
        gaps = []
        breaths = []
        
        for i in range(len(note_segments) - 1):
            curr_end = note_segments[i]['onset_sec'] + note_segments[i]['duration_sec']
            next_start = note_segments[i + 1]['onset_sec']
            gap_dur = next_start - curr_end
            
            if gap_dur <= 0:
                continue
            
            gap_info = {
                'start_sec': curr_end,
                'end_sec': next_start,
                'duration_sec': gap_dur
            }
            
            # Classify gap
            if gap_dur < self.short_gap_max:
                gap_info['type'] = 'short_gap'
            elif gap_dur < self.phrase_gap_max:
                gap_info['type'] = 'phrase_gap'
            else:
                gap_info['type'] = 'section_gap'
            
            # Breath detection
            if (self.breath_min_dur <= gap_dur <= self.breath_max_dur):
                # Check spectral centroid in gap
                cent_mask = (centroid_times >= curr_end) & (centroid_times <= next_start)
                gap_centroid = np.mean(centroid[cent_mask]) if np.any(cent_mask) else 0
                
                # Check pre-gap decay
                pre_note = note_segments[i]
                pre_end_idx = int(pre_note['onset_sec'] * sr) + int(pre_note['duration_sec'] * sr)
                pre_audio = audio[max(0, pre_end_idx - int(0.2*sr)):pre_end_idx]
                pre_rms = np.sqrt(np.mean(pre_audio**2)) if len(pre_audio) > 0 else 0
                
                if gap_centroid > self.breath_min_centroid and pre_rms < self.pre_gap_decay:
                    gap_info['type'] = 'breath'
                    gap_info['confidence'] = min(gap_centroid / 5000, 1.0)
                    breaths.append(gap_info)
            
            gaps.append(gap_info)
        
        # Add breath marks to note segments
        for breath in breaths:
            # Find next note and mark it as post-breath
            for seg in note_segments:
                if seg['onset_sec'] >= breath['end_sec']:
                    seg['post_breath'] = True
                    break
        
        return LayerOutput(
            success=True,
            data={
                'note_segments': note_segments,
                'gaps': gaps,
                'breaths': breaths
            },
            metadata={},
            errors=[]
        )