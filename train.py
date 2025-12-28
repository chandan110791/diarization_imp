import os
import torch
import pytorch_lightning as pl
from pathlib import Path
from pyannote.audio.tasks import Segmentation
from pytorch_lightning.callbacks import ModelCheckpoint
from pyannote.database import registry

# Import your custom modules
from dataset import setup_data
from model import WavLMSegmentation

# --- CONFIGURATION ---
BATCH_SIZE = 16 # Reduce to 8 if you run out of GPU memory
NUM_WORKERS = 4
MAX_EPOCHS = 10 # 10 is usually enough for fine-tuning
CHECKPOINT_DIR = Path("checkpoints")

def train():
    # 1. Setup Data & Config
    print("⚡️ Setting up Data...")
    config_yaml = setup_data(force=False)
    
    # Write the config temporarily so Pyannote can read it
    with open("database.yml", "w") as f:
        f.write(config_yaml)
    
    os.environ["PYANNOTE_DATABASE_CONFIG"] = str(Path("database.yml").absolute())
    registry.load_database(str(Path("database.yml").absolute()))
    
    # 2. Load Protocol
    # We use the 'mini' protocol we defined in dataset.py
    protocol = registry.get_protocol("AMI.SpeakerDiarization.mini")
    
    # 3. Define the Task (Segmentation)
    # duration=5.0 means we chop audio into 5-second training chunks
    task = Segmentation(
        protocol, 
        duration=5.0, 
        max_speakers_per_chunk=3, 
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        loss="bce" # Binary Cross Entropy
    )
    
    # 4. Initialize Model
    model = WavLMSegmentation()
    model.task = task # Link task to model
    
    # 5. Setup Trainer
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        dirpath=CHECKPOINT_DIR,
        filename="wavlm-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1
    )
    
    # Auto-detect GPU
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