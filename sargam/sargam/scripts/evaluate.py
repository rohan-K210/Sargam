import numpy as np
import pandas as pd
import mir_eval
import yaml
from pathlib import Path
from tqdm import tqdm
from src.pipeline.orchestrator import create_pipeline
from src.features.audio import load_audio


def evaluate_pipeline(config_path: str = "config/dataset_config.yaml", max_files: int = 10):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    splits_dir = Path(config['paths']['splits_dir'])
    test_df = pd.read_csv(splits_dir / 'test.csv')
    root = Path(config['paths']['root'])
    
    pipeline = create_pipeline()
    
    results = []
    
    for idx, row in tqdm(test_df.head(max_files).iterrows(), total=min(max_files, len(test_df))):
        audio_path = root / row['audio_path']
        
        if not audio_path.exists():
            continue
        
        try:
            result = pipeline.run(str(audio_path))
            if not result['success']:
                continue
            
            data = result['data']
            
            # Load ground truth pitch
            pitch_path = root / row['pitch_path']
            gt_times, gt_freqs, gt_voiced = [], [], []
            if pitch_path.exists():
                import pandas as pd
                df = pd.read_csv(pitch_path)
                gt_times, gt_freqs, gt_voiced = df['time'].values, df['frequency'].values, df['voiced_flag'].values.astype(bool)
            
            # Load ground truth tonic
            tonic_path = root / row['tonic_path']
            gt_tonic = 440.0
            if tonic_path.exists():
                gt_tonic = float(open(tonic_path).read().strip())
            
            # Metrics
            pred_tonic = data.get('tonic_hz', 440)
            tonic_err_cents = 1200 * np.log2(pred_tonic / gt_tonic) if gt_tonic > 0 else 0
            tonic_acc = abs(tonic_err_cents) <= 100
            
            # Note segmentation (if annotations exist)
            ann_path = root / row['annotation_path']
            onset_f1 = 0
            if ann_path.exists():
                # Would load ground truth onsets and compare
                pass
            
            results.append({
                'file': row['base_name'],
                'raga': row['raga'],
                'tonic_error_cents': tonic_err_cents,
                'tonic_acc': tonic_acc,
                'tempo_bpm': data.get('tempo_bpm', 0),
                'instrument': data.get('instrument', ''),
                'num_notes': len(data.get('notes', [])),
            })
            
        except Exception as e:
            print(f"Error on {row['base_name']}: {e}")
    
    if results:
        df = pd.DataFrame(results)
        print("\n=== EVALUATION SUMMARY ===")
        print(f"Files evaluated: {len(df)}")
        print(f"Tonic accuracy (±100¢): {df['tonic_acc'].mean():.2%}")
        print(f"Tonic RMSE (cents): {np.sqrt((df['tonic_error_cents']**2).mean()):.1f}")
        print(f"\nPer-raga:")
        for raga, group in df.groupby('raga'):
            print(f"  {raga}: TonicAcc={group['tonic_acc'].mean():.2%} (n={len(group)})")
    
    return results


if __name__ == '__main__':
    evaluate_pipeline()