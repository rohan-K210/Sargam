import requests
import tarfile
import zipfile
from pathlib import Path
import yaml
from tqdm import tqdm


def download_file(url: str, dest: Path, chunk_size: int = 8192):
    """Download file with progress bar."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total = int(response.headers.get('content-length', 0))
    
    with open(dest, 'wb') as f, tqdm(total=total, unit='B', unit_scale=True, desc=dest.name) as pbar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))


def extract_archive(archive_path: Path, extract_dir: Path):
    """Extract tar.gz or zip archive."""
    extract_dir.mkdir(parents=True, exist_ok=True)
    
    if archive_path.suffixes[-2:] == ['.tar', '.gz'] or archive_path.suffix == '.tgz':
        with tarfile.open(archive_path, 'r:gz') as tar:
            tar.extractall(extract_dir)
    elif archive_path.suffix == '.zip':
        with zipfile.ZipFile(archive_path, 'r') as zipf:
            zipf.extractall(extract_dir)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path}")


def download_saraga(config_path: str = "config/dataset_config.yaml"):
    """Download Saraga dataset from Zenodo."""
    project_root = Path(__file__).parent.parent
    config_path = project_root / config_path
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    dl_cfg = config['download']
    paths_cfg = config['paths']
    
    root_dir = Path(paths_cfg['root'])
    root_dir.mkdir(parents=True, exist_ok=True)
    
    # Zenodo API to get file list
    record_id = dl_cfg['record_id']
    api_url = f"{dl_cfg['base_url']}/{record_id}"
    
    print(f"Fetching file list from Zenodo record {record_id}...")
    resp = requests.get(api_url)
    resp.raise_for_status()
    record = resp.json()
    
    # Find audio and annotation files
    files = record.get('files', [])
    
    # Handle both Saraga 1.5 and Saraga Mini structures
    record_id = dl_cfg['record_id']
    
    if record_id == '16411600':  # Saraga Mini
        # Files are: saraga audio.zip, saraga pitch.zip, saraga tonic.zip, saraga annotations.zip
        for file_info in files:
            filename = file_info['key']
            url = file_info['links']['self']
            dest = root_dir / filename
            
            if dest.exists():
                print(f"Skipping {filename} (already exists)")
                continue
            
            print(f"Downloading {filename}...")
            download_file(url, dest)
            
            if filename.endswith(('.tar.gz', '.tgz', '.zip')):
                print(f"Extracting {filename}...")
                extract_archive(dest, root_dir)
                dest.unlink()
    else:  # Saraga 1.5 full
        for subset in dl_cfg['subsets']:
            subset_files = [f for f in files if subset in f['key']]
            
            for file_info in subset_files:
                filename = file_info['key']
                url = file_info['links']['self']
                dest = root_dir / filename
                
                if dest.exists():
                    print(f"Skipping {filename} (already exists)")
                    continue
                
                print(f"Downloading {filename}...")
                download_file(url, dest)
                
                if filename.endswith(('.tar.gz', '.tgz', '.zip')):
                    print(f"Extracting {filename}...")
                    extract_archive(dest, root_dir)
                    dest.unlink()
    
    print("Download complete!")


if __name__ == '__main__':
    download_saraga()