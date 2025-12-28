import torch
import matplotlib.pyplot as plt
import os
from pyannote.audio import Pipeline
from pyannote.audio.pipelines import SpeakerDiarization
from pyannote.core import notebook, Segment, Timeline
from pyannote.database import registry, FileFinder
from pathlib import Path

# Import custom modules
from src.benchmark import get_best_checkpoint, get_best_params
from src.model import WavLMSegmentation
from src.dataset import setup_data

def visualize_comparison(file_index=0, crop_start=30, crop_end=60):
    """
    Plots Ground Truth vs Baseline vs Custom Model for a specific file.
    """
    print("👀 PREPARING VISUALIZATION...")
    
    # 1. Load Data
    # We assume setup_data() was run previously
    registry.load_database("database.yml")
    
    def get_annotated(file):
        import torchaudio
        info = torchaudio.info(file["audio"])
        duration = info.num_frames / info.sample_rate
        return Timeline([Segment(0, duration)])

    preprocessors = {"audio": FileFinder(), "annotated": get_annotated}
    protocol = registry.get_protocol("AMI.SpeakerDiarization.mini", preprocessors=preprocessors)
    test_files = list(protocol.test())
    
    if file_index >= len(test_files):
        print(f"❌ Error: Index {file_index} out of range (Max: {len(test_files)-1})")
        return

    target_file = test_files[file_index]
    print(f"📄 File: {target_file['uri']}")
    
    # 2. Load Models (Quick Load)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    auth_token = os.environ.get("HF_TOKEN")
    
    print("   Loading Baseline...")
    base_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=auth_token).to(device)
    
    print("   Loading Custom Model...")
    ckpt = get_best_checkpoint()
    custom_model = WavLMSegmentation.load_from_checkpoint(ckpt).to(device).eval()
    
    custom_pipeline = SpeakerDiarization(
        segmentation=custom_model,
        embedding=base_pipeline.embedding,
        embedding_exclude_overlap=True,
        clustering="AgglomerativeClustering",
    )
    custom_pipeline.to(device)
    
    # Apply Params
    params = get_best_params()
    pipeline_config = {
        "segmentation": {
            "min_duration_off": params.get("min_duration_off", 0.0),
            "threshold": params.get("onset", 0.5),
        },
        "clustering": {"method": "centroid", "min_cluster_size": 12, "threshold": 0.707}
    }
    custom_pipeline.instantiate(pipeline_config)

    # 3. Run Inference
    print("⚡️ Running Inference...")
    hyp_base = base_pipeline(target_file)
    hyp_custom = custom_pipeline(target_file)
    ref = target_file["annotation"]

    # 4. Plot
    print(f"\n🎨 PLOTTING ({crop_start}s to {crop_end}s)...")
    notebook.crop = Segment(crop_start, crop_end)
    
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 10), sharex=True)
    
    # Helper to plot on specific axis
    notebook.plot_annotation(ref, ax=axes[0], time=False)
    axes[0].set_title("1. Ground Truth (Reference)", fontweight='bold')
    
    notebook.plot_annotation(hyp_base, ax=axes[1], time=False)
    axes[1].set_title("2. Pyannote Baseline (SOTA)", fontweight='bold')
    
    notebook.plot_annotation(hyp_custom, ax=axes[2], time=True)
    axes[2].set_title("3. Your WavLM Model", fontweight='bold', color='green')
    
    plt.tight_layout()
    plt.show()
    
    # Reset crop
    notebook.crop = None