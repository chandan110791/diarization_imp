import os
import sys
import torch
import optuna
import re
import numpy as np
from pathlib import Path
from pyannote.audio import Inference
from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.database import registry, FileFinder
from pyannote.audio.utils.signal import Binarize
from pyannote.core import SlidingWindowFeature, Segment, Timeline

from model import WavLMSegmentation
from dataset import setup_data

# --- CONFIGURATION ---
CHECKPOINT_DIR = Path("checkpoints")
N_TRIALS = 10 

def get_best_checkpoint():
    print("🔍 Scanning for checkpoints...")
    if not CHECKPOINT_DIR.exists():
        raise FileNotFoundError("❌ No checkpoints found! Did you run training?")
    
    # Get all .ckpt files
    all_files = list(CHECKPOINT_DIR.glob("*.ckpt"))
    if not all_files:
        raise FileNotFoundError("❌ Checkpoint folder exists but is empty.")

    # 1. Parse all files
    valid_checkpoints = []
    print(f"   Found {len(all_files)} files:")
    
    for path in all_files:
        # Regex to safely find numbers. 
        # Matches "val_loss=" followed by digits/dots, stopping before ".ckpt"
        loss_match = re.search(r"val_loss=([0-9\.]+)", path.name)
        epoch_match = re.search(r"epoch=([0-9]+)", path.name)
        
        if loss_match and epoch_match:
            try:
                # Strip trailing dot if regex grabbed it (e.g. "0.00." -> "0.00")
                loss_str = loss_match.group(1).rstrip(".") 
                loss = float(loss_str)
                epoch = int(epoch_match.group(1))
                valid_checkpoints.append({'path': path, 'loss': loss, 'epoch': epoch})
                print(f"    • {path.name} -> Loss: {loss}, Epoch: {epoch}")
            except ValueError:
                print(f"    ⚠️ Skipping {path.name} (Parse Error)")

    if not valid_checkpoints:
        print("⚠️ No valid 'val_loss' filenames found. Using newest file.")
        return max(all_files, key=os.path.getmtime)

    # 2. Sort: Primary = Lowest Loss, Secondary = Highest Epoch
    # We sort by tuple (loss, -epoch) because Python sorts tuples element-wise
    best_ckpt_data = min(valid_checkpoints, key=lambda x: (x['loss'], -x['epoch']))
    
    print(f"\n🏆 WINNER: {best_ckpt_data['path'].name}")
    print(f"   (Loss: {best_ckpt_data['loss']:.4f}, Epoch: {best_ckpt_data['epoch']})")
    
    return best_ckpt_data['path']

def tune():
    # Force flush to ensure logs appear immediately
    sys.stdout.reconfigure(line_buffering=True)

    print("⚡️ Setting up Tuning Environment...")
    setup_data(force=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚙️  Using Device: {device}")

    # 1. Load Model
    ckpt_path = get_best_checkpoint()
    model = WavLMSegmentation.load_from_checkpoint(ckpt_path)
    model.eval()
    model.to(device)

    # 2. Prepare Data
    registry.load_database("database.yml")
    
    def get_annotated(file):
        import torchaudio
        info = torchaudio.info(file["audio"])
        duration = info.num_frames / info.sample_rate
        return Timeline([Segment(0, duration)])

    preprocessors = {
        "audio": FileFinder(),
        "annotated": get_annotated
    }
    
    protocol = registry.get_protocol("AMI.SpeakerDiarization.mini", preprocessors=preprocessors)
    dev_files = list(protocol.development())[:3] 
    
    print(f"📊 Tuning on {len(dev_files)} files.")

    # 3. Objective Function
    def objective(trial):
        print(f"\n🔄 Trial {trial.number + 1}/{N_TRIALS} started...")
        
        onset = trial.suggest_float("onset", 0.3, 0.9)
        # FORCE OFFSET TO BE VALID (Always lower than onset)
        offset = trial.suggest_float("offset", 0.1, onset - 0.01)
        
        min_duration_on = trial.suggest_float("min_duration_on", 0.0, 1.0)
        min_duration_off = trial.suggest_float("min_duration_off", 0.0, 1.0)
        
        binarizer = Binarize(
            onset=onset, 
            offset=offset, 
            min_duration_on=min_duration_on, 
            min_duration_off=min_duration_off
        )
        
        der_metric = DiarizationErrorRate()
        inference = Inference(model)

        for i, file in enumerate(dev_files):
            print(f"   Processing file {i+1}...", end="\r")
            
            # Crop to first 60s for speed during tuning (optional, remove for full precision)
            # scores = inference.crop(file, Segment(0, 60)) 
            scores = inference(file)

            if scores.data.ndim == 3:
                new_data = scores.data.reshape(-1, scores.data.shape[-1])
                scores = SlidingWindowFeature(new_data, scores.sliding_window)
            
            try:
                hypothesis = binarizer(scores)
                reference = file["annotation"]
                uem = file["annotated"]
                der_metric(reference, hypothesis, uem=uem)
            except Exception as e:
                print(f"\n   ⚠️ Failed on {file['uri']}: {e}")
                return 1.0

        val = abs(der_metric)
        print(f"\n   ✅ Trial {trial.number + 1} Result: DER = {val:.4f}")
        return val

    # 4. Run
    print("🚀 Starting Optuna Study...")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=N_TRIALS)

    print("\n🎉 TUNING COMPLETE!")
    print(f"🌟 Best Error Rate (DER): {study.best_value:.4f}")
    print("⚙️  Best Hyperparameters:", study.best_params)
    
    with open("best_params.txt", "w") as f:
        f.write(str(study.best_params))

if __name__ == "__main__":
    tune()