import os
import shutil
import subprocess
from pathlib import Path

DATA_ROOT = Path("data/ami_mini")
RTTM_CACHE = Path("data/rttm_cache")
AUDIO_DIR = DATA_ROOT / "audio"

def setup_data():
    """Ensures audio and RTTMs are ready. Skips if already done."""
    
    # 1. Check if we already have data
    if RTTM_CACHE.exists() and any(RTTM_CACHE.iterdir()):
        print("✅ RTTM Cache found. Skipping Git clone.")
        return

    print("⚡️ Cache missing. Initializing Data Setup...")
    RTTM_CACHE.mkdir(parents=True, exist_ok=True)
    
    # 2. Clone Repo to Temp
    subprocess.run(["git", "clone", "https://github.com/pyannote/AMI-diarization-setup.git", "temp_repo"], check=True)
    
    # 3. Match and Move RTTMs
    # (Insert your sanitization logic here from Phase 5 Platinum)
    # ...
    
    print("✅ Data Setup Complete.")

def get_database_yml():
    """Generates the YAML config pointing to local files."""
    return f"""
Databases:
  AMI:
    - {AUDIO_DIR.absolute()}/{{uri}}.wav

Protocols:
  AMI:
    SpeakerDiarization:
      mini:
        train:
          uri: lists/train.txt
          annotation: {RTTM_CACHE.absolute()}/{{uri}}.rttm
        # ... (dev and test splits)
"""