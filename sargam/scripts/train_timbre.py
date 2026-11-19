import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import librosa
import yaml


def extract_timbre_features(audio, sr, mfcc_n=13, include_deltas=True, include_spectral=True):
    features = []
    
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=mfcc_n)
    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))
    
    if include_deltas:
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        features.extend(np.mean(delta, axis=1))
        features.extend(np.std(delta, axis=1))
        features.extend(np.mean(delta2, axis=1))
        features.extend(np.std(delta2, axis=1))
    
    if include_spectral:
        centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
        bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
        zcr = librosa.feature.zero_crossing_rate(y=audio)[0]
        
        for feat in [centroid, rolloff, bandwidth, zcr]:
            features.extend([np.mean(feat), np.std(feat)])
        
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        features.extend(np.mean(chroma, axis=1))
        features.extend(np.std(chroma, axis=1))
    
    return np.array(features)


def generate_synthetic_timbre_data(n_per_class=500):
    """Generate synthetic timbre data for training."""
    np.random.seed(42)
    
    classes = ['bansuri', 'violin', 'veena', 'sitar', 'vocals', 'other']
    sr = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration))
    
    X, y = [], []
    
    for class_name in classes:
        for _ in range(n_per_class):
            if class_name == 'bansuri':
                # Pure tone + breath noise
                f0 = np.random.uniform(200, 800)
                audio = np.sin(2 * np.pi * f0 * t)
                audio += 0.1 * np.random.randn(len(t))
                audio *= np.exp(-t * 0.5)
            elif class_name == 'violin':
                # Rich harmonics
                f0 = np.random.uniform(150, 600)
                audio = np.zeros_like(t)
                for h in range(1, 8):
                    audio += (1/h) * np.sin(2 * np.pi * f0 * h * t)
                audio += 0.05 * np.random.randn(len(t))
            elif class_name == 'veena':
                # Plucked string - decaying
                f0 = np.random.uniform(100, 400)
                audio = np.sin(2 * np.pi * f0 * t) * np.exp(-t * 2)
            elif class_name == 'sitar':
                # Sympathetic strings + buzz
                f0 = np.random.uniform(150, 500)
                audio = np.sin(2 * np.pi * f0 * t)
                for h in [2, 3, 5]:
                    audio += 0.3 * np.sin(2 * np.pi * f0 * h * t)
                audio += 0.1 * np.random.randn(len(t))
            elif class_name == 'vocals':
                # Formant-like
                f0 = np.random.uniform(100, 300)
                audio = np.sin(2 * np.pi * f0 * t)
                # Add formants
                for f in [800, 1200, 2500]:
                    audio += 0.2 * np.sin(2 * np.pi * f * t)
            else:
                audio = np.random.randn(len(t)) * 0.1
            
            # Normalize
            audio = audio / (np.max(np.abs(audio)) + 1e-8)
            
            feat = extract_timbre_features(audio, sr)
            X.append(feat)
            y.append(class_name)
    
    return np.array(X), np.array(y)


def train_timbre_classifier(config_path: str = "config/model_config.yaml"):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    tim_cfg = config['timbre']
    model_path = Path(tim_cfg['model_path'])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("Generating synthetic timbre data...")
    X, y = generate_synthetic_timbre_data(800)
    
    print(f"Data shape: {X.shape}, Classes: {np.unique(y)}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    clf = RandomForestClassifier(n_estimators=300, max_depth=15, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    joblib.dump(clf, model_path)
    print(f"\nModel saved to {model_path}")


if __name__ == '__main__':
    train_timbre_classifier()