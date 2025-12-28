import torch
import os
import ast
import json
from pathlib import Path
from tqdm import tqdm
from pyannote.audio import Pipeline
from pyannote.audio.pipelines import SpeakerDiarization
from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.database import registry, FileFinder
from pyannote.core import Segment, Timeline
import re 

# Import custom modules
from src.model import WavLMSegmentation
from src.dataset import setup_data

# --- CONFIGURATION ---
CHECKPOINT_DIR = Path("checkpoints")
# Use your Hugging Face Token from environment or hardcode it if necessary
AUTH_TOKEN = os.environ.get("hf_token") 


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


def get_best_params():
    """Loads parameters from the Tuning phase, or uses defaults."""
    try:
        with open("best_params.txt", "r") as f:
            params = ast.literal_eval(f.read())
        print(f"⚙️  Loaded Tuned Params: {params}")
        return params
    except:
        print("⚠️  No tuning file found. Using Defaults (Expect lower performance).")
        return {"onset": 0.5, "offset": 0.5, "min_duration_on": 0.0, "min_duration_off": 0.0}

def run_benchmark():
    print("🚀 INITIALIZING FINAL BENCHMARK...")
    
    # 1. Setup Data
    setup_data(force=False)
    registry.load_database("database.yml")
    
    # Preprocessors to fix UEM/Timeline issues
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
    test_files = list(protocol.test())
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚙️  Running on {device}")

    # 2. Load Models
    print("⏳ Loading Pyannote Baseline...")
    pipeline_baseline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=AUTH_TOKEN
    ).to(device)

    print("⏳ Building Your WavLM Pipeline...")
    ckpt_path = get_best_checkpoint()
    print(f"   • Checkpoint: {ckpt_path.name}")
    
    custom_model = WavLMSegmentation.load_from_checkpoint(ckpt_path)
    custom_model.to(device)
    custom_model.eval()

    # Reconstruct pipeline using Baseline's clustering (Fair Comparison)
    pipeline_custom = SpeakerDiarization(
        segmentation=custom_model,
        embedding=pipeline_baseline.embedding,
        embedding_exclude_overlap=True,
        clustering="AgglomerativeClustering",
    )
    pipeline_custom.to(device)

    # 3. Apply Tuned Parameters
    best_params = get_best_params()
    
    # Map Optuna 'onset/offset' to the pipeline config structure
    pipeline_config = {
        "segmentation": {
            "min_duration_off": best_params.get("min_duration_off", 0.0),
            "threshold": best_params.get("onset", 0.5), # 'onset' acts as the main threshold here
        },
        "clustering": {
            "method": "centroid",
            "min_cluster_size": 12,
            "threshold": 0.707,
        }
    }
    pipeline_custom.instantiate(pipeline_config)

    # 4. Run Comparison
    print(f"\n⚔️  Benchmarking {len(test_files)} files...")
    metric_base = DiarizationErrorRate()
    metric_custom = DiarizationErrorRate()

    for file in tqdm(test_files):
        # Baseline
        try:
            hyp_base = pipeline_baseline(file)
            metric_base(file["annotation"], hyp_base, uem=file["annotated"])
        except Exception as e:
            print(f"   ⚠️ Baseline Failed on {file['uri']}: {e}")

        # Custom
        try:
            hyp_custom = pipeline_custom(file)
            metric_custom(file["annotation"], hyp_custom, uem=file["annotated"])
        except Exception as e:
            print(f"   ⚠️ Custom Failed on {file['uri']}: {e}")

    # 5. Report
    der_base = abs(metric_base) * 100
    der_custom = abs(metric_custom) * 100
    
    print("\n" + "="*40)
    print("📊 FINAL RESULTS (Global DER)")
    print("="*40)
    print(f"🔵 Baseline (SOTA):  {der_base:.2f}%")
    print(f"🟢 Your WavLM Model: {der_custom:.2f}%")
    print("-" * 40)
    
    diff = der_base - der_custom
    if diff > 0:
        print(f"🏆 SUCCESS: You beat the baseline by {diff:.2f}%!")
    else:
        print(f"📉 GAP: You are behind by {abs(diff):.2f}%.")
        print("   (Tip: Train for more epochs or tune 'min_duration_on' further)")

if __name__ == "__main__":
    run_benchmark()