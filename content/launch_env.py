# content/launch_env.py
import os
import shutil
import sys
from google.colab import drive, userdata

# --- CONFIGURATION ---
REPO_URL = "https://github.com/chandan110791/diarization_imp.git"
REPO_NAME = "diarization_imp"  # Name of your repo folder
DRIVE_DATA_ZIP = "/content/drive/MyDrive/Research_Proposal_Papers/Pyannnote_Objective_1++/datasets/ami_mini.zip"
GITHUB_TOKEN_NAME = "GITHUB_TOKEN" # Name of secret in Colab

def boot_up():
    print("🚀 STARTING MORNING BOOT SEQUENCE...")

    # 1. Mount Drive (For Data & Checkpoints)
    if not os.path.exists('/content/drive'):
        drive.mount('/content/drive')
    
    # 2. Hydrate Audio Data (Fast NVMe Storage)
    # We check if data exists to avoid unpacking 2x
    if not os.path.exists("/content/ami_mini"):
        print("📦 Extracting Audio Data from Drive...")
        shutil.unpack_archive(DRIVE_DATA_ZIP, "/content")
    else:
        print("✅ Audio Data already present.")

    # 3. Pull/Clone Code from Git
    if os.path.exists(f"/content/{REPO_NAME}"):
        print("🔄 Pulling latest code updates...")
        os.chdir(f"/content/{REPO_NAME}")
        !git pull
    else:
        print("📥 Cloning Repository...")
        token = userdata.get(GITHUB_TOKEN_NAME)
        auth_url = REPO_URL.replace("https://", f"https://{token}@")
        !git clone {auth_url}
        os.chdir(f"/content/{REPO_NAME}")

    # 4. Smart Dependency Install
    # Checks if we need to install or if it's already done
    try:
        import pyannote.audio
        print("✅ Dependencies already installed.")
    except ImportError:
        print("🔧 Installing Dependencies from requirements.txt...")
        !pip install -r requirements.txt -q

    print(f"\n✅ SYSTEM READY. Working Directory: {os.getcwd()}")

if __name__ == "__main__":
    boot_up()