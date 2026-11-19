import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from sklearn.model_selection import train_test_split


def prepare_splits(config_path: str = "config/dataset_config.yaml"):
    """Create train/val/test splits stratified by raga."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    paths_cfg = config['paths']
    split_cfg = config['split']
    
    root = Path(paths_cfg['root'])
    splits_dir = Path(paths_cfg['splits_dir'])
    splits_dir.mkdir(parents=True, exist_ok=True)
    
    # Build metadata by scanning directories
    metadata = []
    
    # Try different config structures
    expected = paths_cfg.get('expected_files', {})
    
    # Define subsets to check
    subsets_to_check = ['carnatic', 'hindustani', 'saraga_mini']
    
    for subset in subsets_to_check:
        if subset not in expected:
            continue
            
        subset_cfg = expected[subset]
        audio_dir = root / subset_cfg.get('audio_dir', '')
        pitch_dir = root / subset_cfg.get('pitch_dir', '')
        tonic_dir = root / subset_cfg.get('tonic_dir', '')
        ann_dir = root / subset_cfg.get('annotation_dir', '')
        
        if not audio_dir.exists():
            print(f"Warning: {audio_dir} not found")
            continue
        
        for audio_file in audio_dir.glob('*'):
            if audio_file.suffix not in ['.mp3', '.wav', '.flac']:
                continue
            
            base = audio_file.stem
            
            # Find corresponding files
            pitch_file = pitch_dir / f"{base}.csv"
            tonic_file = tonic_dir / f"{base}.txt"
            ann_file = ann_dir / f"{base}.json"
            
            # Extract raga from filename or annotation
            raga = extract_raga(base, ann_file)
            
            metadata.append({
                'audio_path': str(audio_file.relative_to(root)),
                'pitch_path': str(pitch_file.relative_to(root)) if pitch_file.exists() else '',
                'tonic_path': str(tonic_file.relative_to(root)) if tonic_file.exists() else '',
                'annotation_path': str(ann_file.relative_to(root)) if ann_file.exists() else '',
                'raga': raga,
                'subset': subset,
                'base_name': base
            })
    
    if not metadata:
        print("No audio files found!")
        return
    
    df = pd.DataFrame(metadata)
    print(f"Found {len(df)} recordings")
    print(f"Ragas: {df['raga'].value_counts().to_dict()}")
    
    # Stratified split
    stratify_col = split_cfg.get('stratify_by', 'raga')
    if stratify_col not in df.columns or df[stratify_col].nunique() < 2:
        stratify_col = None
        min_samples = split_cfg.get('min_samples_per_raga', 2)
    if stratify_col:
        # Filter ragas with enough samples
        raga_counts = df[stratify_col].value_counts()
        valid_ragas = raga_counts[raga_counts >= min_samples].index
        df = df[df[stratify_col].isin(valid_ragas)]
        print(f"After filtering: {len(df)} recordings, {df[stratify_col].nunique()} ragas")
        stratify = df[stratify_col]
    else:
        stratify = None
    
    # Split
    train_ratio = split_cfg['train_ratio']
    val_ratio = split_cfg['val_ratio']
    test_ratio = split_cfg['test_ratio']
    seed = split_cfg['random_seed']
    
    train_df, temp_df = train_test_split(
        df, test_size=(val_ratio + test_ratio), 
        stratify=stratify, random_state=seed
    )
    
    val_stratify = train_df[stratify_col] if stratify_col else None
    val_df, test_df = train_test_split(
        temp_df, test_size=test_ratio/(val_ratio + test_ratio),
        stratify=val_stratify, random_state=seed
    )
    
    # Save splits
    train_df.to_csv(splits_dir / 'train.csv', index=False)
    val_df.to_csv(splits_dir / 'val.csv', index=False)
    test_df.to_csv(splits_dir / 'test.csv', index=False)
    
    # Save full metadata
    df.to_csv(paths_cfg['metadata_csv'], index=False)
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    print(f"Splits saved to {splits_dir}")


def extract_raga(base_name: str, ann_file: Path) -> str:
    """Extract raga from filename or annotation."""
    # Try annotation first
    if ann_file.exists():
        import json
        try:
            with open(ann_file) as f:
                ann = json.load(f)
            if 'raga' in ann:
                return ann['raga']
        except:
            pass
    
    # Try filename (common patterns)
    parts = base_name.split('_')
    for part in parts:
        if len(part) > 2 and part[0].isupper():
            return part
    
    return 'unknown'


if __name__ == '__main__':
    prepare_splits()