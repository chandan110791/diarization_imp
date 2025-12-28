import os
import argparse
import torch
import torchaudio
import pytorch_lightning as pl
from pathlib import Path
from pyannote.audio.tasks import Segmentation
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from pyannote.database import registry, FileFinder
from pyannote.core import Segment, Timeline

# Import your custom modules
from dataset import setup_data
from model import WavLMSegmentation

# --- HELPER: The Fix for Validation Crashes ---
def get_annotated(file):
    """
    Generates a 'UEM' timeline on the fly.
    Tells Pyannote to treat the ENTIRE file duration as valid for training/validation.
    """
    if "torchaudio.info" in file:
        info = file["torchaudio.info"]
    else:
        info = torchaudio.info(file["audio"])
    
    duration = info.num_frames / info.sample_rate
    return Timeline([Segment(0, duration)])

def train(max_epochs):
    # 1. Setup Data
    print("⚡️ Setting up Data...")
    config_yaml = setup_data(force=False)
    
    with open("database.yml", "w") as f:
        f.write(config_yaml)
    
    os.environ["PYANNOTE_DATABASE_CONFIG"] = str(Path("database.yml").absolute())
    registry.load_database(str(Path("database.yml").absolute()))
    
    # 2. Load Protocol with PREPROCESSORS
    # This prevents the "NoneType is not iterable" errors
    preprocessors = {
        "audio": FileFinder(),
        "torchaudio.info": lambda f: torchaudio.info(f["audio"]),
        "annotated": get_annotated
    }
    
    protocol = registry.get_protocol("AMI.SpeakerDiarization.mini", preprocessors=preprocessors)
    
    # 3. Define the Task
    print("📝 Configuring Segmentation Task...")
    task = Segmentation(
        protocol, 
        duration=5.0, 
        max_speakers_per_chunk=3, 
        batch_size=16,
        num_workers=4,
        loss="bce" 
    )
    
    # 4. Initialize Model
    model = WavLMSegmentation()
    model.task = task 
    
    # 5. Setup Trainer
    # Checkpoints will be saved in 'checkpoints/'
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(
        monitor="DiarizationErrorRate",
        mode="min",
        dirpath=checkpoint_dir,
        filename="wavlm-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        save_last=True
    )
    
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    
    print(f"🚀 Starting Training for {max_epochs} Epochs on {accelerator.upper()}...")
    
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=1,
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback, RichProgressBar()],
        default_root_dir=checkpoint_dir,
        # Gradient clipping prevents the model from exploding if learning rate is too high
        gradient_clip_val=0.5
    )
    
    # 6. TRAIN!
    trainer.fit(model)
    
    print(f"\n✅ Training Complete!")
    print(f"🏆 Best Model: {checkpoint_callback.best_model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs to train")
    args = parser.parse_args()
    
    train(args.epochs)