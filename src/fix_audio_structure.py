import os
import shutil
from pathlib import Path

# --- CONFIGURATION ---
# This matches the folder structure you showed in the screenshot
DATA_ROOT = Path("data/ami_mini/audio")

def flatten_audio_directory():
    print(f"🔍 Scanning for nested audio in: {DATA_ROOT.absolute()}")
    
    if not DATA_ROOT.exists():
        print("❌ Error: Audio directory not found.")
        return

    # 1. Find all .wav files recursively
    # rglob looks inside all subfolders (amicorpus/ES2002a/audio/...)
    all_wavs = list(DATA_ROOT.rglob("*.wav"))
    
    if not all_wavs:
        print("⚠️ No .wav files found anywhere!")
        return

    print(f"📦 Found {len(all_wavs)} nested audio files. Flattening now...")

    moved_count = 0
    for file_path in all_wavs:
        # Skip if the file is already in the right place
        if file_path.parent == DATA_ROOT:
            continue
            
        # Define new home: data/ami_mini/audio/ES2002a.Mix-Headset.wav
        target_path = DATA_ROOT / file_path.name
        
        try:
            shutil.move(str(file_path), str(target_path))
            moved_count += 1
        except Exception as e:
            print(f"   ❌ Error moving {file_path.name}: {e}")

    print(f"✅ Successfully moved {moved_count} files to root.")

    # 2. Cleanup: Remove the empty 'amicorpus' folder
    nested_dir = DATA_ROOT / "amicorpus"
    if nested_dir.exists():
        print("🧹 Removing empty 'amicorpus' folders...")
        shutil.rmtree(nested_dir)
        
    # 3. Final Verification
    final_count = len(list(DATA_ROOT.glob("*.wav")))
    print("-" * 40)
    print(f"🎉 FINAL STATUS: {final_count} audio files ready in {DATA_ROOT}")
    print("   You can now run 'python src/dataset.py'!")

if __name__ == "__main__":
    flatten_audio_directory()