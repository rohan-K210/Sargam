import os
import tempfile
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
    
    def run(
        self,
        audio_path: str,
        tonic_hz: Optional[float] = None,
        separate_source: bool = False,
    ) -> Dict[str, Any]:
        working_path, temp_stem_path, separation_used = self._maybe_separate(
            audio_path, separate_source
        )

        try:
            # Load audio
            audio, sr, audio_info = load_audio(working_path, self.config)

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
            final_output['source_separation_applied'] = separation_used
            return {'success': True, 'data': final_output}
        finally:
            if temp_stem_path and os.path.exists(temp_stem_path):
                os.remove(temp_stem_path)

    def _maybe_separate(self, audio_path: str, separate_source: bool):
        """
        If requested, isolate the melodic-instrument stem (vocals/drums/bass
        removed) from a full mix before the rest of the pipeline runs, using
        HDemucs. Writes the isolated stem to a temp WAV so it can flow
        through the existing path-based load_audio() unchanged. Falls back
        to the original audio on any failure (missing dependency, model
        load error, etc.) rather than breaking the pipeline.

        Returns: (path_to_analyze, temp_file_to_clean_up_or_None, was_separated)
        """
        if not separate_source:
            return audio_path, None, False

        try:
            import librosa
            import soundfile as sf
            from src.features.source_separation import isolate_melodic_instrument

            raw_audio, raw_sr = librosa.load(audio_path, sr=None, mono=True)
            stem_audio, stem_sr, was_separated = isolate_melodic_instrument(raw_audio, raw_sr)
            if not was_separated:
                return audio_path, None, False

            fd, temp_path = tempfile.mkstemp(suffix='.wav')
            os.close(fd)
            sf.write(temp_path, stem_audio, stem_sr)
            return temp_path, temp_path, True
        except Exception:
            return audio_path, None, False
    
    def _compile_output(self, results: Dict, audio_info) -> Dict:
        note_segments = results.get('timing', {}).get('note_segments', [])

        # Leading silence gets trimmed before analysis (see load_audio), so
        # note timestamps are relative to the trimmed audio. Add the trim
        # offset back so they line up with the original, untrimmed file.
        trim_offset = getattr(audio_info, 'trim_start', 0.0) or 0.0

        notes_output = []
        for seg in note_segments:
            note = {
                'swara': seg.get('swara', ''),
                'variant': seg.get('variant', ''),
                'octave': seg.get('octave', ''),
                'onset_sec': seg.get('onset_sec', 0) + trim_offset,
                'duration_sec': seg.get('duration_sec', 0),
                'ornament': seg.get('ornament', {'type': 'none'}),
                'dynamics': seg.get('dynamics', 'mezzo'),
                'attack': seg.get('attack', 'unknown')
            }
            notes_output.append(note)
        
        breaths = []
        for b in results.get('silence_breath', {}).get('breaths', []):
            breath = dict(b)
            if 'start_sec' in breath:
                breath['start_sec'] = breath['start_sec'] + trim_offset
            if 'end_sec' in breath:
                breath['end_sec'] = breath['end_sec'] + trim_offset
            breaths.append(breath)

        return {
            'tonic': results.get('pitch_tonic', {}).get('tonic_note', ''),
            'tonic_hz': float(results.get('pitch_tonic', {}).get('tonic_hz', 440)),
            'tonic_confidence': float(results.get('pitch_tonic', {}).get('tonic_confidence', 0)),
            'tempo': results.get('timing', {}).get('laya', 'Madhya'),
            'tempo_bpm': float(results.get('timing', {}).get('tempo_bpm', 120)),
            'instrument': results.get('timbre', {}).get('instrument', 'unknown'),
            'instrument_confidence': float(results.get('timbre', {}).get('confidence', 0)),
            'breaths': breaths,
            'notes': notes_output,
            'audio_info': {
                'duration_sec': float(audio_info.processed_duration),
                'sample_rate': int(audio_info.processed_sr)
            }
        }
    
    def run_to_json(self, audio_path: str, output_path: str, tonic_hz: Optional[float] = None):
        result = self.run(audio_path, tonic_hz)
        if result['success']:
            save_json(result['data'], output_path)
        return result


def create_pipeline(config_path: str = "config/model_config.yaml") -> SargamPipeline:
    return SargamPipeline(config_path)