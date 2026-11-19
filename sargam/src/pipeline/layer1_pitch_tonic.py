import numpy as np
from typing import Dict
from src.features.audio import load_config
from src.pipeline.base import PipelineLayer, LayerOutput
from src.features.pitch import create_pitch_detector, load_pitch_config, correct_octave_errors
from src.models.tonic_detector import detect_tonic_from_pitch, detect_tonic_initial_phrase, hz_to_note_name
from src.mapping.swara_mapper import SwaraMapper


class PitchTonicLayer(PipelineLayer):
    def __init__(self, config: dict = None):
        super().__init__(config)
        self.pitch_config = load_pitch_config()
        self.detector_type = self.pitch_config.get('primary', 'crepe')
        self.fallback_type = self.pitch_config.get('fallback', 'pyin')
        self.pitch_detector = create_pitch_detector(self.detector_type, self.pitch_config.get(self.detector_type, {}))
        self.fallback_detector = create_pitch_detector(self.fallback_type, self.pitch_config.get(self.fallback_type, {}))
        self.swara_mapper = SwaraMapper()
        self.tonic_config = config.get('tonic', {}) if config else {}
    
    def process(self, audio: np.ndarray, sr: int, context: dict) -> LayerOutput:
        # Pitch detection with fallback
        pitch_output = self._detect_pitch(audio, sr)
        if not pitch_output.success:
            return pitch_output
        
        pitch_data = correct_octave_errors(pitch_output.data)

        # Tonic detection
        tonic_hz = context.get('tonic_hz')
        if tonic_hz is None:
            tonic_hz, tonic_conf, tonic_note = detect_tonic_initial_phrase(
                pitch_data.frequencies,
                pitch_data.confidence,
                pitch_data.voiced_flag,
                pitch_data.times,
                self.tonic_config
            )
        else:
            tonic_conf = 1.0
            tonic_note = hz_to_note_name(tonic_hz)
        
        # Map to svaras
        swara_notes = self.swara_mapper.map_pitch_contour(
            pitch_data.frequencies,
            pitch_data.confidence,
            tonic_hz,
            smoothing_window=self.config.get('swara', {}).get('smoothing_window', 5)
        )
        
        return LayerOutput(
            success=True,
            data={
                'pitch': pitch_data,
                'tonic_hz': tonic_hz,
                'tonic_confidence': tonic_conf,
                'tonic_note': tonic_note,
                'swara_notes': swara_notes
            },
            metadata={
                'detector_used': self.detector_type,
                'fallback_available': self.fallback_type
            },
            errors=[]
        )
    
    def _detect_pitch(self, audio: np.ndarray, sr: int) -> LayerOutput:
        try:
            return LayerOutput(True, self.pitch_detector.predict(audio, sr), {}, [])
        except Exception as e:
            try:
                return LayerOutput(True, self.fallback_detector.predict(audio, sr), {}, [f"Primary failed: {e}"])
            except Exception as e2:
                return LayerOutput(False, None, {}, [f"Both detectors failed: {e}, {e2}"])