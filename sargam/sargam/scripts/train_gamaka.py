import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import yaml


def extract_gamaka_features(cents, times, sr=16000):
    """Extract features for gamaka classification."""
    features = {}
    
    if len(cents) < 3:
        return None
    
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
    
    # Number of direction changes
    diff = np.diff(cents)
    features['direction_changes'] = np.sum(np.diff(np.sign(diff)) != 0)
    
    return np.array(list(features.values()))


def generate_synthetic_data(n_samples=5000):
    """Generate synthetic gamaka data for training."""
    np.random.seed(42)
    X, y = [], []
    
    ornaments = ['none', 'meend', 'andolan', 'kan', 'murki', 'gamak']
    
    for _ in range(n_samples):
        ornament = np.random.choice(ornaments)
        duration = np.random.uniform(0.1, 2.0)
        times = np.linspace(0, duration, int(duration * 100))
        
        if ornament == 'none':
            cents = np.random.normal(0, 5, len(times))
        elif ornament == 'meend':
            slope = np.random.uniform(100, 500)
            cents = slope * times + np.random.normal(0, 10, len(times))
        elif ornament == 'andolan':
            freq = np.random.uniform(4, 10)
            amp = np.random.uniform(20, 60)
            cents = amp * np.sin(2 * np.pi * freq * times) + np.random.normal(0, 5, len(times))
        elif ornament == 'gamak':
            freq = np.random.uniform(15, 30)
            amp = np.random.uniform(60, 150)
            cents = amp * np.sin(2 * np.pi * freq * times) + np.random.normal(0, 10, len(times))
        elif ornament == 'kan':
            cents = np.zeros(len(times))
            if len(cents) > 5:
                cents[:3] = np.random.uniform(100, 300, 3)
        elif ornament == 'murki':
            cents = np.random.choice([-200, -100, 0, 100, 200], len(times))
        
        feat = extract_gamaka_features(cents, times)
        if feat is not None:
            X.append(feat)
            y.append(ornament)
    
    return np.array(X), np.array(y)


def train_gamaka_classifier(config_path: str = "config/model_config.yaml"):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model_path = Path(config['gamaka']['classifier']['model_path'])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("Generating synthetic training data...")
    X, y = generate_synthetic_data(10000)
    
    print(f"Data shape: {X.shape}, Classes: {np.unique(y)}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    joblib.dump(clf, model_path)
    print(f"\nModel saved to {model_path}")


if __name__ == '__main__':
    train_gamaka_classifier()