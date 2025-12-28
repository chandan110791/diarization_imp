import os
import torch
import torchaudio
import pytorch_lightning as pl
from pathlib import Path
from pyannote.audio.tasks import Segmentation
from pytorch_lightning.callbacks import ModelCheckpoint
from pyannote.database import registry, FileFinder
from pyannote.core import Segment, Timeline

# Import your custom modules
from dataset import setup_data
from model import WavLMSegmentation

# --- CONFIGURATION ---
BATCH_SIZE = 16 
NUM_WORKERS = 4
MAX_EPOCHS = 2 
CHECKPOINT_DIR = Path("checkpoints")

# --- HELPER: The Fix for "NoneType" Error ---
def get_annotated(file):
    """
    Generates a 'UEM' timeline on the fly.
    Tells Pyannote to treat the ENTIRE file duration as valid for training.
    """
    # Use pre-fetched info if available, else fetch it
    if "torchaudio.info" in file:
        info = file["torchaudio.info"]
    else:
        info = torchaudio.info(file["audio"])
    
    duration = info.num_frames / info.sample_rate
    return Timeline([Segment(0, duration)])

def train():
    # 1. Setup Data & Config
    print("⚡️ Setting up Data...")
    config_yaml = setup_data(force=False)
    
    # Write config for Pyannote
    with open("database.yml", "w") as f:
        f.write(config_yaml)
    
    os.environ["PYANNOTE_DATABASE_CONFIG"] = str(Path("database.yml").absolute())
    registry.load_database(str(Path("database.yml").absolute()))
    
    # 2. Load Protocol with CUSTOM PREPROCESSORS (Critical!)
    # This injects the 'annotated' key that was missing
    preprocessors = {
        "audio": FileFinder(),
        "torchaudio.info": lambda f: torchaudio.info(f["audio"]),
        "annotated": get_annotated
    }
    
    # We pass the preprocessors here so the 'file' dictionary is populated correctly
    protocol = registry.get_protocol("AMI.SpeakerDiarization.mini", preprocessors=preprocessors)
    
    # 3. Define the Task
    print("📝 Configuring Segmentation Task...")
    task = Segmentation(
        protocol, 
        duration=5.0, 
        max_speakers_per_chunk=3, 
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        loss="bce" 
    )
    
    # 4. Initialize Model
    model = WavLMSegmentation()
    model.task = task 
    
    # 5. Setup Trainer
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(
        monitor="DiarizationErrorRate",
        mode="min",
        dirpath=CHECKPOINT_DIR,
        filename="wavlm-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1
    )
    
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = 1 if torch.cuda.is_available() else "auto"
    
    print(f"🚀 Starting Training on {accelerator.upper()}...")
    
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=MAX_EPOCHS,
        callbacks=[checkpoint_callback],
        default_root_dir=CHECKPOINT_DIR
    )
    
    # 6. TRAIN!
    trainer.fit(model)
    
    print(f"\n✅ Training Complete!")
    print(f"🏆 Best Model Saved at: {checkpoint_callback.best_model_path}")

if __name__ == "__main__":
    train()