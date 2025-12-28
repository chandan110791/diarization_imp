import os
import shutil
import subprocess
import sys
from pathlib import Path

# --- CONFIGURATION ---
# We download directly into the 'audio' folder expected by dataset.py
DESTINATION_DIR = Path("data/ami_mini/audio")
TEMP_REPO_DIR = Path("temp_ami_downloader")
REPO_URL = "https://github.com/pyannote/AMI-diarization-setup.git"

def check_dependencies():
    """Checks if 'sox' and 'git' are installed."""
    missing = []
    if not shutil.which("git"):
        missing.append("git")
    if not shutil.which("sox"):
        missing.append("sox")
    
    if missing:
        print("❌ CRITICAL ERROR: Missing system tools.")
        print(f"   Please install: {', '.join(missing)}")
        print("   On Ubuntu/WSL run: sudo apt-get install git sox -y")
        sys.exit(1)
    print("✅ System dependencies (git, sox) found.")

def download_ami_mini():
    print(f"🚀 STARTING DOWNLOAD: AMI Mini Corpus")
    print(f"   Target Directory: {DESTINATION_DIR.absolute()}")

    # 1. Clean/Create Destination
    if DESTINATION_DIR.exists():
        print(f"   ⚠️ directory {DESTINATION_DIR} exists. checking content...")
        if any(DESTINATION_DIR.iterdir()):
             print("   ⚠️ Directory is not empty. Assuming data is already downloaded.")
             return
    
    DESTINATION_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Clone the Setup Repo to a temp folder
    if TEMP_REPO_DIR.exists():
        shutil.rmtree(TEMP_REPO_DIR)
    
    print("📥 Cloning setup repository...")
    subprocess.run(["git", "clone", "-q", REPO_URL, str(TEMP_REPO_DIR)], check=True)

    # 3. Locate the download script
    # The script is located at: pyannote/download_ami_mini.sh inside the repo
    script_path = TEMP_REPO_DIR / "pyannote" / "download_ami_mini.sh"
    
    if not script_path.exists():
        print(f"❌ Error: Could not find download script at {script_path}")
        return

    # 4. Make it executable
    os.chmod(script_path, 0o755)

    # 5. Run the bash script
    print("📡 Downloading audio files (This may take a minute)...")
    try:
        # We run the script and pass the absolute path of our destination
        subprocess.run(
            ["bash", str(script_path.name), str(DESTINATION_DIR.resolve())], 
            cwd=script_path.parent, # Run from inside the directory so it finds relative files
            check=True
        )
        print("✅ Download finished successfully.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Download script failed with error: {e}")
        sys.exit(1)
    finally:
        # 6. Cleanup
        print("🧹 Cleaning up temp files...")
        if TEMP_REPO_DIR.exists():
            shutil.rmtree(TEMP_REPO_DIR)

    # 7. Final Verification
    wav_count = len(list(DESTINATION_DIR.glob("*.wav")))
    print(f"📊 SUMMARY: {wav_count} .wav files are now in {DESTINATION_DIR}")

if __name__ == "__main__":
    check_dependencies()
    download_ami_mini()