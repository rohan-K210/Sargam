import numpy as np
import yaml
from typing import Dict, Any, Optional
from src.pipeline.layer1_pitch_tonic import PitchTonicLayer
from src.pipeline.layer2_timing import TimingLayer
from src.pipeline.layer3_gamaka import GamakaLayer
from src.pipeline.layer4_dynamics import DynamicsLayer
from src.pipeline.layer5_silence import SilenceBreathLayer
from src.pipeline.layer6_timbre import TimbreLayer
from src.features.audio import load_audio, load_config
from src.utils.io import save_json


class SargamPipeline:
    def __init__(self, config_path: str = "config/model_config.yaml"):
        self.config = load_config(config_path)
        self.layers = self._build_layers()
    
    def _build_layers(self):
        return [
            ('pitch_tonic', PitchTonicLayer(self.config)),
            ('timing', TimingLayer(self.config)),
            ('gamaka', GamakaLayer(self.config)),
            ('dynamics', DynamicsLayer(self.config)),
            ('silence_breath', SilenceBreathLayer(self.config)),
            ('timbre', TimbreLayer(self.config)),
        ]
    
    def run(self, audio_path: str, tonic_hz: Optional[float] = None) -> Dict[str, Any]:
        # Load audio
        audio, sr, audio_info = load_audio(audio_path, self.config)
        
        context = {
            'audio': audio,
            'sr': sr,
            'audio_info': audio_info,
            'tonic_hz': tonic_hz
        }
        
        # Run each layer sequentially
        results = {}
        for name, layer in self.layers:
            output = layer(audio, sr, context)
            if not output.success:
                return {'success': False, 'error': f"Layer {name} failed: {output.errors}", 'partial': results}
            
            results[name] = output.data
            context.update(output.data)
        
        # Compile final output
        final_output = self._compile_output(results, audio_info)
        return {'success': True, 'data': final_output}
    
    def _compile_output(self, results: Dict, audio_info) -> Dict:
        note_segments = results.get('timing', {}).get('note_segments', [])
        
        notes_output = []
        for seg in note_segments:
            note = {
                'swara': seg.get('swara', ''),
                'variant': seg.get('variant', ''),
                'octave': seg.get('octave', ''),
                'onset_sec': seg.get('onset_sec', 0),
                'duration_sec': seg.get('duration_sec', 0),
                'ornament': seg.get('ornament', {'type': 'none'}),
                'dynamics': seg.get('dynamics', 'mezzo'),
                'attack': seg.get('attack', 'unknown')
            }
            notes_output.append(note)
        
        return {
            'tonic': results.get('pitch_tonic', {}).get('tonic_note', ''),
            'tonic_hz': results.get('pitch_tonic', {}).get('tonic_hz', 440),
            'tonic_confidence': results.get('pitch_tonic', {}).get('tonic_confidence', 0),
            'tempo': results.get('timing', {}).get('laya', 'Madhya'),
            'tempo_bpm': results.get('timing', {}).get('tempo_bpm', 120),
            'instrument': results.get('timbre', {}).get('instrument', 'unknown'),
            'instrument_confidence': results.get('timbre', {}).get('confidence', 0),
            'breaths': results.get('silence_breath', {}).get('breaths', []),
            'notes': notes_output,
            'audio_info': {
                'duration_sec': audio_info.processed_duration,
                'sample_rate': audio_info.processed_sr
            }
        }
    
    def run_to_json(self, audio_path: str, output_path: str, tonic_hz: Optional[float] = None):
        result = self.run(audio_path, tonic_hz)
        if result['success']:
            save_json(result['data'], output_path)
        return result


def create_pipeline(config_path: str = "config/model_config.yaml") -> SargamPipeline:
    return SargamPipeline(config_path)