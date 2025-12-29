# DAILY SETUP: Mounts Drive, Unzips Data, Pulls Code, Installs Deps
import os
import shutil
from google.colab import drive, userdata

print("🚀 INITIALIZING DAILY WORKFLOW...")

# 1. MOUNT DRIVE (To access your Zipped Data)
if not os.path.exists('/content/drive'):
    drive.mount('/content/drive')

# 2. HYDRATE AUDIO DATA (Copy from Drive -> Colab Fast SSD)
ZIP_PATH = "/content/drive/MyDrive/Research_Proposal_Papers/Pyannnote_Objective_1++/datasets/ami_mini.zip"
LOCAL_DATA_DIR = "/content/ami_mini"

if not os.path.exists(LOCAL_DATA_DIR):
    if os.path.exists(ZIP_PATH):
        print("📦 Extracting Audio Data from Drive to NVMe...")
        shutil.unpack_archive(ZIP_PATH, "/content")
        print("✅ Audio Data Ready.")
    else:
        print(f"⚠️ WARNING: Zip file not found at {ZIP_PATH}")
        print("   (Did you run 'python src/dataset.py' locally to upload it?)")
else:
    print("✅ Audio Data already present.")

# 3. PULL CODE (From Your GitHub)
REPO_URL = "https://github.com/chandan110791/diarization_imp.git"
REPO_DIR = "/content/diarization_imp"

if not os.path.exists(REPO_DIR):
    print("📥 Cloning Public Repository...")
    !git clone $REPO_URL $REPO_DIR
else:
    print("🔄 Pulling Latest Changes...")
    %cd $REPO_DIR
    !git pull

# 4. INSTALL DEPENDENCIES
print("🔧 Installing Dependencies...")
%cd $REPO_DIR
!pip install -r requirements.txt -q

# 5. AUTHENTICATE HUGGING FACE (Required for WavLM download)
try:
    from google.colab import userdata
    os.environ["HF_TOKEN"] = userdata.get('HF_TOKEN')
    print("✅ Hugging Face Token Loaded.")
except:
    print("⚠️ HF_TOKEN not found in Secrets! Please add it in the sidebar.")

print("\n✨ SYSTEM READY. You can now run '!python src/train.py'")