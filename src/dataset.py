import os
import shutil
import subprocess
import random
from pathlib import Path

# --- CONFIGURATION ---
# Paths are relative to the repository root
LOCAL_DATA_ROOT = Path("data/ami_mini")
AUDIO_DIR = LOCAL_DATA_ROOT / "audio"
RTTM_DIR = LOCAL_DATA_ROOT / "annotations"
LIST_DIR = LOCAL_DATA_ROOT / "lists"
REPO_TEMP = Path("temp_ami_setup")

# Google Drive Destination (For the "Morning Launcher")
DRIVE_DEST_DIR = Path("/content/drive/MyDrive/Research_Proposal_Papers/Pyannnote_Objective_1++/datasets")
DRIVE_ZIP_NAME = "ami_mini.zip"

def setup_data(force=False):
    """
    1. Sanitizes Audio Filenames (removes .Mix-Headset)
    2. Fetches RTTMs from Git
    3. Creates Train/Dev/Test Splits
    4. Generates database.yml
    """
    
    # 1. Quick Check: Skip if already done
    if not force and RTTM_DIR.exists() and any(RTTM_DIR.iterdir()):
        print("✅ Data already prepared locally. Skipping setup.")
        return get_database_yml()

    print("⚡️ Initializing Data Setup (Sanitization & RTTM Fetch)...")
    
    # Clean up old folders
    for d in [RTTM_DIR, LIST_DIR, REPO_TEMP]:
        if d.exists(): shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    # --- A. SANITIZE AUDIO ---
    # We look for ANY .wav file in the data folder (nested or flat)
    print("🧹 Sanitizing Audio Filenames...")
    found_wavs = list(LOCAL_DATA_ROOT.rglob("*.wav"))
    
    # Filter for AMI files (avoid system sounds)
    ami_wavs = [f for f in found_wavs if "ami" in str(f).lower() or "mix" in str(f).lower()]
    
    clean_ids = []
    for f in ami_wavs:
        # Logic: "TS3005a.Mix-Headset.wav" -> "TS3005a"
        clean_id = f.name.split('.')[0] 
        target_path = AUDIO_DIR / f"{clean_id}.wav"
        
        # Only copy if it's not already there (avoids self-overwrite errors)
        if not target_path.exists():
            shutil.copy(f, target_path)
        
        clean_ids.append(clean_id)

    # Unique list of valid meetings
    unique_ids = sorted(list(set(clean_ids)))
    print(f"✅ Audio Ready: {len(unique_ids)} unique meetings in {AUDIO_DIR}")

    # --- B. CLONE REPO & MATCH RTTMS ---
    print("📥 Cloning AMI-diarization-setup repository...")
    subprocess.run(["git", "clone", "https://github.com/pyannote/AMI-diarization-setup.git", str(REPO_TEMP)], check=True)

    print("🔗 Matching RTTMs...")
    all_repo_rttms = list(REPO_TEMP.rglob("*.rttm"))
    valid_ids = []

    for fid in unique_ids:
        # Exact match (e.g. TS3005a.rttm)
        matches = [r for r in all_repo_rttms if fid == r.stem] 
        
        if matches:
            # Prefer 'only_words' version if available
            best_match = next((m for m in matches if "only_words" in str(m)), matches[0])
            shutil.copy(best_match, RTTM_DIR / f"{fid}.rttm")
            valid_ids.append(fid)
    
    print(f"✅ Matched {len(valid_ids)} RTTMs.")

    # --- C. CREATE SPLITS ---
    # Shuffle and split: 24 Train, 5 Dev, 5 Test (Adjust based on your dataset size)
    random.seed(42)
    random.shuffle(valid_ids)
    
    # Safe fallback if you have fewer than 34 files
    n = len(valid_ids)
    train_split = valid_ids[:max(1, int(n * 0.7))]
    dev_split = valid_ids[len(train_split):len(train_split) + max(1, int(n * 0.15))]
    test_split = valid_ids[len(train_split) + len(dev_split):]

    # Write split files
    with open(LIST_DIR / "train.txt", "w") as f: f.write("\n".join(train_split))
    with open(LIST_DIR / "dev.txt", "w") as f: f.write("\n".join(dev_split))
    with open(LIST_DIR / "test.txt", "w") as f: f.write("\n".join(test_split))

    print(f"📊 Splits Created: {len(train_split)} Train, {len(dev_split)} Dev, {len(test_split)} Test")

    # Clean up git repo
    shutil.rmtree(REPO_TEMP)
    return get_database_yml()

def get_database_yml():
    """Generates the content for database.yml"""
    # Note: We use absolute paths to avoid confusion
    return f"""
Databases:
  AMI:
    - {AUDIO_DIR.resolve()}/{{uri}}.wav

Protocols:
  AMI:
    SpeakerDiarization:
      mini:
        train:
          uri: {LIST_DIR.resolve()}/train.txt
          annotation: {RTTM_DIR.resolve()}/{{uri}}.rttm
        development:
          uri: {LIST_DIR.resolve()}/dev.txt
          annotation: {RTTM_DIR.resolve()}/{{uri}}.rttm
        test:
          uri: {LIST_DIR.resolve()}/test.txt
          annotation: {RTTM_DIR.resolve()}/{{uri}}.rttm
"""

def push_dataset_to_drive():
    """Zips the prepared data and sends it to Google Drive."""
    print("\n📦 PACKAGING DATASET FOR DRIVE...")
    
    # 1. Check if Drive is available (Colab) or manually mounted
    if not os.path.exists("/content/drive"):
        print("⚠️ Google Drive not found at /content/drive.")
        print("   If running locally, manually upload 'data/ami_mini.zip' to your Drive.")
        return

    # 2. Zip the folder
    # We zip 'data/ami_mini' into 'ami_mini.zip'
    zip_base_name = "ami_mini"
    root_dir = "data"
    base_dir = "ami_mini"
    
    shutil.make_archive(zip_base_name, 'zip', root_dir, base_dir)
    print(f"✅ Zipped: {zip_base_name}.zip")

    # 3. Copy to Drive
    DRIVE_DEST_DIR.mkdir(parents=True, exist_ok=True)
    dest_path = DRIVE_DEST_DIR / DRIVE_ZIP_NAME
    
    print(f"🚀 Uploading to: {dest_path} ...")
    shutil.copy(f"{zip_base_name}.zip", dest_path)
    print("✅ Upload Complete! Your Morning Launcher is ready.")

if __name__ == "__main__":
    # If run directly, perform setup and push
    setup_data(force=True)
    
    # Save the config locally just in case
    with open("database.yml", "w") as f:
        f.write(get_database_yml())
        
    # Push to Drive
    push_dataset_to_drive()