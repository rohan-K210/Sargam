import numpy as np
import librosa
import joblib
import os
from pathlib import Path
from src.pipeline.base import PipelineLayer, LayerOutput

try:
    from huggingface_hub import hf_hub_download
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False


class TimbreLayer(PipelineLayer):
    def __init__(self, config: dict = None):
        super().__init__(config)
        tim_cfg = config.get('timbre', {}) if config else {}
        self.model_type = tim_cfg.get('model_type', 'rf')
        self.model_repo = tim_cfg.get('model_repo', 'RK14x86/sargam-models')
        self.model_filename = tim_cfg.get('model_filename', 'timbre/timbre_classifier.joblib')
        self.classes = tim_cfg.get('classes', ['bansuri', 'violin', 'veena', 'sitar', 'vocals', 'other'])
        self.mfcc_n = tim_cfg.get('features', {}).get('mfcc_n', 13)
        self.include_deltas = tim_cfg.get('features', {}).get('include_deltas', True)
        self.include_spectral = tim_cfg.get('features', {}).get('include_spectral', True)
        self.model_path = tim_cfg.get('model_path', '')
        self._model = None
        self._load_model()

    def _load_model(self):
        if self._model is None:
            try:
                if HF_HUB_AVAILABLE:
                    model_path = hf_hub_download(
                        repo_id=self.model_repo,
                        filename=self.model_filename,
                        token=os.getenv('HF_TOKEN')
                    )
                    self._model = joblib.load(model_path)
                elif self.model_path and Path(self.model_path).exists():
                    self._model = joblib.load(self.model_path)
            except Exception:
                self._model = None
    
    def process(self, audio: np.ndarray, sr: int, context: dict) -> LayerOutput:
        features = self._extract_features(audio, sr)
        
        if self._model is None:
            # Fallback: simple heuristic
            instrument = self._heuristic_classify(features)
            confidence = 0.5
        else:
            try:
                probs = self._model.predict_proba([features])[0]
                pred_idx = np.argmax(probs)
                instrument = self.classes[pred_idx]
                confidence = float(probs[pred_idx])
            except Exception:
                instrument = self._heuristic_classify(features)
                confidence = 0.3
        
        return LayerOutput(
            success=True,
            data={
                'instrument': instrument,
                'confidence': confidence,
                'features': features.tolist() if isinstance(features, np.ndarray) else features
            },
            metadata={'model_loaded': self._model is not None},
            errors=[]
        )
    
    def _extract_features(self, audio: np.ndarray, sr: int):
        features = []
        
        # MFCCs
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=self.mfcc_n)
        features.extend(np.mean(mfcc, axis=1))
        features.extend(np.std(mfcc, axis=1))
        
        if self.include_deltas:
            delta = librosa.feature.delta(mfcc)
            delta2 = librosa.feature.delta(mfcc, order=2)
            features.extend(np.mean(delta, axis=1))
            features.extend(np.std(delta, axis=1))
            features.extend(np.mean(delta2, axis=1))
            features.extend(np.std(delta2, axis=1))
        
        if self.include_spectral:
            centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
            bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
            zcr = librosa.feature.zero_crossing_rate(y=audio)[0]
            
            for feat in [centroid, rolloff, bandwidth, zcr]:
                features.extend([np.mean(feat), np.std(feat)])
            
            # Chroma
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
            features.extend(np.mean(chroma, axis=1))
            features.extend(np.std(chroma, axis=1))
        
        return np.array(features, dtype=np.float32)
    
    def _heuristic_classify(self, features):
        # Simple heuristic: bansuri has high centroid, low bandwidth
        if len(features) > 30:
            centroid_idx = self.mfcc_n * 2 + (self.mfcc_n * 4 if self.include_deltas else 0) + 0
            if centroid_idx < len(features) and features[centroid_idx] > 2000:
                return 'bansuri'
        return 'other'