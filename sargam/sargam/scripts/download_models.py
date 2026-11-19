import os
import torch
from pathlib import Path


def download_crepe_models():
    """Pre-download CREPE model weights."""
    print("Downloading CREPE models...")
    import torchcrepe
    
    # This triggers download of all model capacities
    for capacity in ['tiny', 'small', 'medium', 'large']:
        try:
            torchcrepe.load.model(capacity, device='cpu')
            print(f"  CREPE-{capacity}: OK")
        except Exception as e:
            print(f"  CREPE-{capacity}: FAILED - {e}")


def download_all():
    """Download all required pretrained models."""
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    download_crepe_models()
    
    print("\nAll models downloaded. Cache location:", torch.hub.get_dir())


if __name__ == '__main__':
    download_all()