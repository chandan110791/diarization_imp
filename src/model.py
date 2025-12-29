# model.py
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WavLMModel


class WavLMEmbedding(nn.Module):
    """
    WavLM backbone + soft layer selection (weighted sum) + stats pooling + projection head.
    Returns L2-normalized embeddings (B, target_dim).

    Input:
      waveforms: (B, 1, T) or (B, T) at 16kHz
    Output:
      embeddings: (B, target_dim)
    """
    def __init__(
        self,
        model_name: str = "microsoft/wavlm-base-plus",
        target_dim: int = 256,
        freeze_backbone: bool = False,
        output_hidden_states: bool = True,
    ):
        super().__init__()

        # Load WavLM backbone
        self.backbone = WavLMModel.from_pretrained(
            model_name,
            output_hidden_states=output_hidden_states
        )

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # Number of hidden states returned = num_hidden_layers + 1 (embeddings)
        num_layers = self.backbone.config.num_hidden_layers + 1
        self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)

        self.hidden_size = self.backbone.config.hidden_size
        self.pooling_dim = self.hidden_size * 2  # mean + std

        self.projection = nn.Sequential(
            nn.Linear(self.pooling_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, target_dim),
        )

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        # (B, 1, T) -> (B, T)
        if waveforms.dim() == 3:
            waveforms = waveforms.squeeze(1)

        # Forward through backbone
        outputs = self.backbone(waveforms)
        if outputs.hidden_states is None:
            raise RuntimeError("WavLMModel did not return hidden_states. Ensure output_hidden_states=True.")

        # Stack all layers: (L, B, T, H)
        hidden_states = torch.stack(outputs.hidden_states, dim=0)

        # Soft layer selection
        norm_weights = F.softmax(self.layer_weights, dim=0)  # (L,)
        weighted = (hidden_states * norm_weights.view(-1, 1, 1, 1)).sum(dim=0)  # (B, T, H)

        # Stats pooling
        mean_stat = weighted.mean(dim=1)                          # (B, H)
        var = weighted.var(dim=1, unbiased=False)
        std_stat = (var + 1e-6).sqrt()    # (B, H)
        pooled = torch.cat([mean_stat, std_stat], dim=1)          # (B, 2H)


        # Projection + L2 norm
        emb = self.projection(pooled)                             # (B, D)
        emb = F.normalize(emb, p=2, dim=-1)
        return emb
