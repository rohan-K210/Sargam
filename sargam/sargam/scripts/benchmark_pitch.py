import numpy as np
import pandas as pd
import librosa
import mir_eval
from pathlib import Path
import yaml
from tqdm import tqdm
from src.features.pitch import create_pitch_detector, load_pitch_config
from src.features.audio import load_audio


def evaluate_pitch_detector(detector, audio, sr, gt_times, gt_freqs, gt_voiced):
    """Evaluate pitch detector against ground truth."""
    pred = detector.predict(audio, sr)
    
    # Interpolate to common time grid
    common_times = np.linspace(max(pred.times[0], gt_times[0]), 
                               min(pred.times[-1], gt_times[-1]), 
                               num=min(len(pred.times), len(gt_times)))
    
    pred_interp = np.interp(common_times, pred.times, pred.frequencies)
    pred_voiced_interp = np.interp(common_times, pred.times, pred.voiced_flag.astype(float)) > 0.5
    gt_interp = np.interp(common_times, gt_times, gt_freqs)
    gt_voiced_interp = np.interp(common_times, gt_times, gt_voiced.astype(float)) > 0.5
    
    # Raw pitch accuracy
    both_voiced = pred_voiced_interp & gt_voiced_interp
    if np.any(both_voiced):
        cents_diff = 1200 * np.log2(pred_interp[both_voiced] / gt_interp[both_voiced])
        accuracy_50 = np.mean(np.abs(cents_diff) <= 50)
        accuracy_100 = np.mean(np.abs(cents_diff) <= 100)
        rmse_cents = np.sqrt(np.mean(cents_diff**2))
    else:
        accuracy_50 = accuracy_100 = rmse_cents = 0
    
    # Voicing metrics
    tp = np.sum(pred_voiced_interp & gt_voiced_interp)
    fp = np.sum(pred_voiced_interp & ~gt_voiced_interp)
    fn = np.sum(~pred_voiced_interp & gt_voiced_interp)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'pitch_acc_50c': accuracy_50,
        'pitch_acc_100c': accuracy_100,
        'pitch_rmse_cents': rmse_cents,
        'voicing_precision': precision,
        'voicing_recall': recall,
        'voicing_f1': f1
    }


def load_ground_truth(pitch_path):
    """Load ground truth pitch from Saraga CSV."""
    import pandas as pd
    df = pd.read_csv(pitch_path)
    return df['time'].values, df['frequency'].values, df['voiced_flag'].values.astype(bool)


def run_benchmark(config_path: str = "config/dataset_config.yaml", max_files: int = 20):
    """Run pitch detector benchmark on Saraga test set."""
    # Resolve config path relative to project root
    project_root = Path(__file__).parent.parent
    config_path = project_root / config_path
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    splits_dir = Path(config['paths']['splits_dir'])
    test_df = pd.read_csv(splits_dir / 'test.csv')
    root = Path(config['paths']['root'])
    
    pitch_config = load_pitch_config()
    
    detectors = {
        'crepe': create_pitch_detector('crepe', pitch_config.get('crepe', {})),
        'pyin': create_pitch_detector('pyin', pitch_config.get('pyin', {})),
    }
    
    results = {name: [] for name in detectors}
    
    for idx, row in tqdm(test_df.head(max_files).iterrows(), total=min(max_files, len(test_df))):
        audio_path = root / row['audio_path']
        pitch_path = root / row['pitch_path']
        
        if not pitch_path.exists():
            continue
        
        try:
            audio, sr, _ = load_audio(str(audio_path))
            gt_times, gt_freqs, gt_voiced = load_ground_truth(str(pitch_path))
            
            for name, detector in detectors.items():
                metrics = evaluate_pitch_detector(detector, audio, sr, gt_times, gt_freqs, gt_voiced)
                metrics['file'] = row['base_name']
                metrics['raga'] = row['raga']
                results[name].append(metrics)
        except Exception as e:
            print(f"Error on {row['base_name']}: {e}")
    
    # Print summary
    for name, res_list in results.items():
        if not res_list:
            continue
        df = pd.DataFrame(res_list)
        print(f"\n=== {name.upper()} ===")
        print(f"Pitch Acc (±50¢): {df['pitch_acc_50c'].mean():.3f} ± {df['pitch_acc_50c'].std():.3f}")
        print(f"Pitch Acc (±100¢): {df['pitch_acc_100c'].mean():.3f} ± {df['pitch_acc_100c'].std():.3f}")
        print(f"Pitch RMSE (cents): {df['pitch_rmse_cents'].mean():.1f} ± {df['pitch_rmse_cents'].std():.1f}")
        print(f"Voicing F1: {df['voicing_f1'].mean():.3f} ± {df['voicing_f1'].std():.3f}")
        
        # Per-raga breakdown
        print("\nPer-raga:")
        for raga, group in df.groupby('raga'):
            print(f"  {raga}: Acc50={group['pitch_acc_50c'].mean():.3f}, F1={group['voicing_f1'].mean():.3f} (n={len(group)})")
    
    return results


if __name__ == '__main__':
    run_benchmark()