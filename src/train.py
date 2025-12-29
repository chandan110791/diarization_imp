# train_embedding.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from pyannote.audio.core.model import Model
from pyannote.audio.core.task import Specifications, Problem, Resolution

from model import WavLMEmbedding


class PyannoteWavLMEmbeddingAdapter(Model):
    """
    Wrap torch embedding model to be usable in pyannote SpeakerDiarization pipeline.

    SpeakerDiarization expects embedding=Inference(window="sliding", duration=..., step=...)
    where Inference wraps a pyannote.audio Model with:
      specifications.problem = REPRESENTATION
      specifications.resolution = CHUNK
      specifications.duration = <chunk duration>
    """
    def __init__(
        self,
        model_name: str = "microsoft/wavlm-base-plus",
        target_dim: int = 256,
        duration: float = 4.0,
        freeze_backbone: bool = False,
    ):
        super().__init__(sample_rate=16000, num_channels=1)

        self.wavlm_model = WavLMEmbedding(
            model_name=model_name,
            target_dim=target_dim,
            freeze_backbone=freeze_backbone,
            output_hidden_states=True,
        )

        self.specifications = Specifications(
            problem=Problem.REPRESENTATION,
            resolution=Resolution.CHUNK,
            duration=duration,
            classes=None,
        )

    def forward(self, waveforms: torch.Tensor, weights: torch.Tensor = None) -> torch.Tensor:
        # waveforms expected: (B, C, T) or (B, T)
        return self.wavlm_model(waveforms)


# -------------------------
# OPTIONAL: tiny training skeleton (if you want later)
# -------------------------
@dataclass
class TrainConfig:
    lr: float = 1e-4
    freeze_backbone: bool = False
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def build_model(cfg: TrainConfig, duration: float = 4.0) -> PyannoteWavLMEmbeddingAdapter:
    m = PyannoteWavLMEmbeddingAdapter(
        model_name="microsoft/wavlm-base-plus",
        target_dim=256,
        duration=duration,
        freeze_backbone=cfg.freeze_backbone,
    )
    m.to(torch.device(cfg.device))
    return m
