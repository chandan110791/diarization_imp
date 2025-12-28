import torch
from pyannote.audio import Pipeline
from src.model import WavLMSegmentation # Imported from your file
from src.dataset import setup_data, get_database_yml

# 1. Setup Data (Fast Check)
setup_data()

# 2. Load Config
with open("database.yml", "w") as f:
    f.write(get_database_yml())

# 3. Load Model
model = WavLMSegmentation.load_from_checkpoint("checkpoints/best.ckpt")

# 4. Run Benchmark (Code from Phase 5)
# ...