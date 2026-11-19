import numpy as np
import os
import joblib
from scipy.signal import find_peaks
from src.pipeline.base import PipelineLayer, LayerOutput

try:
    from huggingface_hub import hf_hub_download
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False


class GamakaLayer(PipelineLayer):
    def __init__(self, config: dict = None):
        super().__init__(config)
        gamaka_cfg = config.get('gamaka', {}) if config else {}
        self.method = gamaka_cfg.get('method', 'hybrid')
        rule_cfg = gamaka_cfg.get('rule_thresholds', {})
        self.meend_slope = rule_cfg.get('meend_slope_cents_per_sec', 50)
        # No real flute slide moves faster than this; a steeper slope is a
        # pitch-tracking artifact (octave error, transient), not a real meend
        self.meend_slope_max = rule_cfg.get('meend_slope_max_cents_per_sec', 1200)
        self.andolan_freq_range = rule_cfg.get('andolan_freq_range', [4, 10])
        self.andolan_min_amp = rule_cfg.get('andolan_min_amp_cents', 20)
        self.gamak_freq = rule_cfg.get('gamak_freq_hz', 15)
        self.gamak_min_amp = rule_cfg.get('gamak_min_amp_cents', 60)
        self.kan_max_dur = rule_cfg.get('kan_max_duration_sec', 0.15)
        self.kan_energy_ratio = rule_cfg.get('kan_energy_ratio', 0.3)
        self.murki_min_changes = rule_cfg.get('murki_min_changes', 4)
        self.murki_max_window = rule_cfg.get('murki_max_window_sec', 0.5)
        self.classifier_enabled = gamaka_cfg.get('classifier', {}).get('enabled', False)
        
        # HF Hub model config
        self.model_repo = gamaka_cfg.get('model_repo', 'RK14x86/sargam-models')
        self.model_filename = gamaka_cfg.get('model_filename', 'gamaka/gamaka_classifier.joblib')
        self._model = None
        self._load_model()
    
    def _load_model(self):
        if self.classifier_enabled and HF_HUB_AVAILABLE:
            try:
                model_path = hf_hub_download(
                    repo_id=self.model_repo,
                    filename=self.model_filename,
                    token=os.getenv('HF_TOKEN')
                )
                self._model = joblib.load(model_path)
            except Exception:
                self._model = None
    
    def process(self, audio: np.ndarray, sr: int, context: dict) -> LayerOutput:
        note_segments = context.get('note_segments', [])
        pitch_data = context.get('pitch', None)
        
        if not note_segments or pitch_data is None:
            return LayerOutput(False, None, {}, ["Missing note segments or pitch data"])
        
        # Analyze each note for ornaments
        for segment in note_segments:
            ornament = self._detect_ornament(segment, pitch_data, sr)
            segment['ornament'] = ornament
        
        return LayerOutput(
            success=True,
            data={'note_segments': note_segments},
            metadata={'method': self.method},
            errors=[]
        )
    
    def _detect_ornament(self, segment, pitch_data, sr):
        onset = segment['onset_sec']
        duration = segment['duration_sec']
        offset = onset + duration
        
        # Extract pitch contour for this note
        mask = (pitch_data.times >= onset) & (pitch_data.times <= offset)
        note_times = pitch_data.times[mask]
        note_freqs = pitch_data.frequencies[mask]
        note_conf = pitch_data.confidence[mask]
        
        if len(note_freqs) < 3:
            return {'type': 'none', 'confidence': 0.0}
        
        # Convert to cents relative to segment tonic (skip unvoiced/zero-freq frames)
        voiced_mask = (note_freqs > 0) & (note_conf > 0.3)
        tonic = segment.get('tonic_hz', 440)
        cents = 1200 * np.log2(note_freqs[voiced_mask] / tonic)
        note_times = note_times[voiced_mask]
        
        if len(cents) < 3:
            return {'type': 'none', 'confidence': 0.0}
        
        # Try ML classifier first if available
        if self._model is not None:
            try:
                features = self._extract_contour_features(cents, note_times)
                pred = self._model.predict([features])[0]
                proba = self._model.predict_proba([features])[0]
                confidence = float(np.max(proba))
                return {'type': pred, 'confidence': confidence, 'method': 'ml'}
            except Exception:
                pass  # Fall back to rule-based
        
        # Rule-based detection
        ornament = self._rule_based_detection(cents, note_times, duration)
        return ornament
    
    def _extract_contour_features(self, cents, times):
        """Extract features for ML classifier."""
        features = {}
        
        if len(cents) < 3:
            return np.zeros(8)
        
        # Linear slope (meend)
        slope = np.polyfit(times, cents, 1)[0]
        features['slope'] = slope
        
        # Detrend
        trend = np.polyval(np.polyfit(times, cents, 1), times)
        detrended = cents - trend
        
        # Oscillation analysis
        if len(detrended) > 4:
            from scipy.fft import fft, fftfreq
            N = len(detrended)
            dt = times[1] - times[0] if len(times) > 1 else 0.01
            yf = fft(detrended)
            xf = fftfreq(N, dt)[:N//2]
            idx = np.argmax(np.abs(yf[:N//2]))
            features['osc_freq'] = abs(xf[idx])
            features['osc_amp'] = np.std(detrended)
        else:
            features['osc_freq'] = 0
            features['osc_amp'] = 0
        
        # Duration
        features['duration'] = times[-1] - times[0] if len(times) > 1 else 0
        
        # Direction changes
        diff = np.diff(cents)
        features['direction_changes'] = np.sum(np.diff(np.sign(diff)) != 0)
        
        return np.array(list(features.values()))
    
    def _rule_based_detection(self, cents, times, duration):
        # Linear fit for meend
        if len(cents) > 2:
            slope = np.polyfit(times, cents, 1)[0]  # cents/sec
        else:
            slope = 0
        
        # Detrend for oscillation analysis
        if len(cents) > 4:
            trend = np.polyval(np.polyfit(times, cents, 1), times)
            detrended = cents - trend
            # FFT for oscillation frequency
            from scipy.fft import fft, fftfreq
            N = len(detrended)
            if N > 4:
                yf = fft(detrended)
                xf = fftfreq(N, times[1] - times[0])[:N//2]
                idx = np.argmax(np.abs(yf[:N//2]))
                osc_freq = abs(xf[idx])
                osc_amp = np.std(detrended)
            else:
                osc_freq = 0
                osc_amp = 0
        else:
            osc_freq = 0
            osc_amp = 0
        
        # Detect ornament type. A slope beyond meend_slope_max is implausible
        # for real playing and is treated as a tracking artifact, not a slide.
        if self.meend_slope < abs(slope) <= self.meend_slope_max:
            return {'type': 'meend', 'direction': 'ascending' if slope > 0 else 'descending',
                    'slope_cents_per_sec': float(slope), 'confidence': min(abs(slope)/200, 1.0)}
        
        if (self.andolan_freq_range[0] <= osc_freq <= self.andolan_freq_range[1] 
            and osc_amp > self.andolan_min_amp):
            return {'type': 'andolan', 'frequency_hz': float(osc_freq),
                    'amplitude_cents': float(osc_amp), 'confidence': min(osc_amp/50, 1.0)}
        
        if osc_freq > self.gamak_freq and osc_amp > self.gamak_min_amp:
            return {'type': 'gamak', 'frequency_hz': float(osc_freq),
                    'amplitude_cents': float(osc_amp), 'confidence': min(osc_amp/100, 1.0)}
        
        # Kan: short duration, check first portion
        if duration < self.kan_max_dur:
            return {'type': 'kan', 'duration_sec': float(duration), 'confidence': 0.7}
        
        # Murki: many svara changes in short window
        # Would need svara sequence analysis
        
        return {'type': 'none', 'confidence': 0.0}