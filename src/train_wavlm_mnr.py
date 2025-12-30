# train_wavlm_mnr.py
from __future__ import annotations

import os
import json
import math
import random
import argparse
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader, Sampler

from transformers import WavLMModel


# -----------------------------
# Model: WavLM -> (optional layer mix) -> pooling -> projection -> L2 norm
# -----------------------------

class AttentiveStatsPooling(nn.Module):
    """
    ECAPA-style Attentive Statistical Pooling.
    Input:  (B, T, H)
    Output: (B, 2H)
    """
    def __init__(self, hidden_size: int, bottleneck: int = 128):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_size, bottleneck),
            nn.Tanh(),
            nn.Linear(bottleneck, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,H)
        w = self.attn(x)                     # (B,T,1)
        w = torch.softmax(w, dim=1)          # (B,T,1)
        mean = torch.sum(w * x, dim=1)       # (B,H)
        # weighted second moment
        second = torch.sum(w * (x ** 2), dim=1)
        var = (second - mean ** 2).clamp_min(1e-6)
        std = torch.sqrt(var)
        return torch.cat([mean, std], dim=1) # (B,2H)


class WavLMEmbedding(nn.Module):
    """
    WavLM speaker embedding extractor:
      - WavLM backbone (frozen)
      - layer mixing (optional learnable weights)
      - pooling (ASP or mean+std)
      - projection to target_dim
      - L2 normalize
    """
    def __init__(
        self,
        model_name: str = "microsoft/wavlm-base-plus",
        target_dim: int = 256,
        use_layer_mix: bool = True,
        pooling: str = "asp",  # "asp" or "stats"
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.backbone = WavLMModel.from_pretrained(model_name, output_hidden_states=True)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.hidden_size = self.backbone.config.hidden_size
        self.use_layer_mix = use_layer_mix

        num_layers = self.backbone.config.num_hidden_layers + 1  # + embeddings
        if self.use_layer_mix:
            self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        else:
            self.register_buffer("layer_weights", torch.ones(num_layers) / num_layers)

        if pooling.lower() == "asp":
            self.pool = AttentiveStatsPooling(self.hidden_size, bottleneck=128)
            pooled_dim = self.hidden_size * 2
        elif pooling.lower() == "stats":
            self.pool = None
            pooled_dim = self.hidden_size * 2
        else:
            raise ValueError("pooling must be 'asp' or 'stats'")

        self.pooling = pooling.lower()

        self.proj = nn.Sequential(
            nn.Linear(pooled_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, target_dim),
        )

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """
        waveforms: (B, 1, T) or (B, T)
        returns:   (B, target_dim) normalized
        """
        if waveforms.dim() == 3:
            waveforms = waveforms.squeeze(1)  # (B,T)

        out = self.backbone(waveforms)
        hs = torch.stack(out.hidden_states, dim=0)  # (L,B,T,H)

        w = F.softmax(self.layer_weights, dim=0).view(-1, 1, 1, 1)  # (L,1,1,1)
        x = torch.sum(w * hs, dim=0)  # (B,T,H)

        if self.pooling == "asp":
            pooled = self.pool(x)  # (B,2H)
        else:
            mean = x.mean(dim=1)
            var = x.var(dim=1, unbiased=False)
            std = torch.sqrt(var + 1e-6)
            pooled = torch.cat([mean, std], dim=1)  # (B,2H)

        emb = self.proj(pooled)
        emb = F.normalize(emb, p=2, dim=1)
        return emb


# -----------------------------
# Data: manifest JSONL -> load wav segments
# -----------------------------

@dataclass
class ChunkRow:
    uri: str
    audio: str
    start: float
    duration: float
    speaker: str


class ChunkDataset(Dataset):
    def __init__(self, rows: List[ChunkRow], sample_rate: int = 16000):
        self.rows = rows
        self.sr = sample_rate

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        return r


def load_manifest(path: str) -> List[ChunkRow]:
    rows: List[ChunkRow] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            rows.append(
                ChunkRow(
                    uri=obj["uri"],
                    audio=obj["audio"],
                    start=float(obj["start"]),
                    duration=float(obj["duration"]),
                    speaker=str(obj["speaker"]),
                )
            )
    if not rows:
        raise RuntimeError(f"Manifest is empty: {path}")
    return rows


def _read_audio_segment(path: str, start_s: float, dur_s: float, target_sr: int) -> torch.Tensor:
    """
    Returns waveform (1,T) at target_sr, exactly dur_s long (pad or trim).
    """
    wav, sr = torchaudio.load(path)  # (C,T)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)  # mono

    start = int(start_s * sr)
    length = int(dur_s * sr)
    end = min(start + length, wav.size(1))
    seg = wav[:, start:end]

    if sr != target_sr:
        seg = torchaudio.functional.resample(seg, sr, target_sr)

    target_len = int(dur_s * target_sr)
    if seg.size(1) < target_len:
        seg = F.pad(seg, (0, target_len - seg.size(1)))
    else:
        seg = seg[:, :target_len]

    return seg  # (1,T)


# -----------------------------
# Balanced batch sampling: N speakers x K segments per speaker
# -----------------------------

class BalancedSpeakerBatchSampler(Sampler[List[int]]):
    """
    Produces batches where each batch contains:
      - batch_speakers distinct speakers
      - segs_per_speaker samples per speaker
    """
    def __init__(
        self,
        speaker_to_indices: Dict[int, List[int]],
        batch_speakers: int,
        segs_per_speaker: int,
        steps_per_epoch: int,
        seed: int = 1337,
    ):
        self.speaker_to_indices = speaker_to_indices
        self.speakers = list(speaker_to_indices.keys())
        self.batch_speakers = batch_speakers
        self.segs_per_speaker = segs_per_speaker
        self.steps_per_epoch = steps_per_epoch
        self.rng = random.Random(seed)

        # filter speakers with enough samples
        self.speakers = [s for s in self.speakers if len(self.speaker_to_indices[s]) >= self.segs_per_speaker]
        if len(self.speakers) < self.batch_speakers:
            raise RuntimeError(
                f"Not enough speakers with >= {self.segs_per_speaker} segments "
                f"to form a batch of {self.batch_speakers} speakers."
            )

    def __len__(self):
        return self.steps_per_epoch

    def __iter__(self):
        for _ in range(self.steps_per_epoch):
            chosen = self.rng.sample(self.speakers, k=self.batch_speakers)
            batch = []
            for spk in chosen:
                inds = self.speaker_to_indices[spk]
                # sample with replacement to keep it simple
                picked = [inds[self.rng.randrange(len(inds))] for _ in range(self.segs_per_speaker)]
                batch.extend(picked)
            yield batch


def collate_fn(rows: List[ChunkRow], speaker_id_map: Dict[str, int], sample_rate: int) -> Tuple[torch.Tensor, torch.Tensor]:
    waves = []
    spk_ids = []
    for r in rows:
        w = _read_audio_segment(r.audio, r.start, r.duration, target_sr=sample_rate)  # (1,T)
        waves.append(w)
        spk_ids.append(speaker_id_map[r.speaker])

    wave_batch = torch.stack(waves, dim=0)                 # (B,1,T)
    spk_ids = torch.tensor(spk_ids, dtype=torch.long)      # (B,)
    return wave_batch, spk_ids


# -----------------------------
# MNR / InfoNCE loss (in-batch negatives)
# -----------------------------

def mnr_loss(emb: torch.Tensor, labels: torch.Tensor, tau: float = 0.07) -> torch.Tensor:
    """
    emb: (B,D) L2-normalized
    labels: (B,)
    For each anchor i, choose one positive j (same label) within batch.
    Negatives: all other samples.
    """
    B = emb.size(0)
    sim = (emb @ emb.t()) / tau  # (B,B)

    # mask self
    sim = sim.masked_fill(torch.eye(B, device=sim.device, dtype=torch.bool), torch.finfo(sim.dtype).min)

    # build a target index per anchor (pick a deterministic positive: first other occurrence)
    targets = torch.empty(B, device=sim.device, dtype=torch.long)
    for i in range(B):
        same = torch.where(labels == labels[i])[0]
        same = same[same != i]
        if len(same) == 0:
            # if batch sampling is correct, this should not happen (K>=2)
            raise RuntimeError("No positive in batch for some anchor. Ensure segs_per_speaker >= 2.")
        targets[i] = same[0]

    loss = F.cross_entropy(sim, targets)
    return loss


@torch.no_grad()
def dev_retrieval_accuracy(model: nn.Module, loader: DataLoader, device: torch.device, tau: float) -> float:
    """
    Quick proxy: for each anchor, nearest neighbor in batch should share speaker label.
    """
    model.eval()
    correct = 0
    total = 0

    for wave, y in loader:
        wave = wave.to(device)
        y = y.to(device)
        emb = model(wave)
        emb = F.normalize(emb, p=2, dim=1)

        sim = emb @ emb.t()
        sim.fill_diagonal_(-1e9)
        nn_idx = sim.argmax(dim=1)
        pred = y[nn_idx]
        correct += (pred == y).sum().item()
        total += y.numel()

    return correct / max(total, 1)


# -----------------------------
# Training script (Setting A)
# -----------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--train_manifest", required=True)
    ap.add_argument("--dev_manifest", required=True)
    ap.add_argument("--out_ckpt", required=True)

    ap.add_argument("--wavlm_model", default="microsoft/wavlm-base-plus")
    ap.add_argument("--target_dim", type=int, default=256)
    ap.add_argument("--duration", type=float, default=3.0)
    ap.add_argument("--sample_rate", type=int, default=16000)

    ap.add_argument("--pooling", choices=["asp", "stats"], default="asp")
    ap.add_argument("--use_layer_mix", action="store_true")  # default False unless specified
    ap.add_argument("--tau", type=float, default=0.07)

    # Batch construction: N speakers x K segments per speaker
    ap.add_argument("--batch_speakers", type=int, default=16)
    ap.add_argument("--segs_per_speaker", type=int, default=2)
    ap.add_argument("--steps_per_epoch", type=int, default=200)

    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--grad_clip", type=float, default=1.0)

    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--seed", type=int, default=1337)

    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] device:", device)

    # Load manifests
    train_rows = load_manifest(args.train_manifest)
    dev_rows = load_manifest(args.dev_manifest)

    # Sanity: enforce same duration in manifests as training duration (recommended)
    # (You can relax this later, but start strict.)
    for r in random.sample(train_rows, k=min(5, len(train_rows))):
        if abs(r.duration - args.duration) > 1e-6:
            print("[WARN] manifest duration != args.duration; consider regenerating manifest with same chunk_dur")

    # Build speaker map (global IDs across train)
    speakers = sorted({r.speaker for r in train_rows})
    spk_map = {s: i for i, s in enumerate(speakers)}
    print("[INFO] train speakers:", len(spk_map), "train rows:", len(train_rows), "dev rows:", len(dev_rows))

    # Build index per speaker for balanced sampling
    speaker_to_indices: Dict[int, List[int]] = {}
    for i, r in enumerate(train_rows):
        sid = spk_map[r.speaker]
        speaker_to_indices.setdefault(sid, []).append(i)

    # Dataset + Sampler
    train_ds = ChunkDataset(train_rows, sample_rate=args.sample_rate)

    sampler = BalancedSpeakerBatchSampler(
        speaker_to_indices=speaker_to_indices,
        batch_speakers=args.batch_speakers,
        segs_per_speaker=args.segs_per_speaker,
        steps_per_epoch=args.steps_per_epoch,
        seed=args.seed,
    )

    train_loader = DataLoader(
        train_ds,
        batch_sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=lambda batch: collate_fn(batch, spk_map, args.sample_rate),
    )

    # Dev loader (simple random batches; retrieval proxy)
    # Build a smaller speaker map for dev using train speaker IDs where possible
    # (dev speakers not in train are ignored for accuracy proxy)
    # Dev loader (proxy metric): build a DEV-local speaker map (AMI speaker ids are per-meeting)
    dev_speakers = sorted({r.speaker for r in dev_rows})
    dev_spk_map = {s: i for i, s in enumerate(dev_speakers)}
    dev_ds = ChunkDataset(dev_rows, sample_rate=args.sample_rate)

    dev_loader = DataLoader(
      dev_ds,
      batch_size=args.batch_speakers * args.segs_per_speaker,
      shuffle=True,
      num_workers=args.num_workers,
      pin_memory=(device.type == "cuda"),
      drop_last=True,
      collate_fn=lambda batch: collate_fn(batch, dev_spk_map, args.sample_rate),
    )

    if len(dev_ds) == 0:
      raise RuntimeError("Dev manifest is empty.")


    # Model (Setting A: freeze backbone)
    model = WavLMEmbedding(
        model_name=args.wavlm_model,
        target_dim=args.target_dim,
        use_layer_mix=args.use_layer_mix,
        pooling=args.pooling,
        freeze_backbone=True,
    ).to(device)

    # Optimizer only for trainable params (head + optional layer mix)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(args.fp16 and device.type == "cuda"))

    best_dev = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0

        for step, (wave, y) in enumerate(train_loader, start=1):
            wave = wave.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(args.fp16 and device.type == "cuda")):
                emb = model(wave)  # already L2 normalized
                loss = mnr_loss(emb, y, tau=args.tau)

            scaler.scale(loss).backward()

            if args.grad_clip and args.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(opt)
            scaler.update()

            running += loss.item()
            if step % 25 == 0:
                print(f"[TRAIN] epoch={epoch} step={step}/{args.steps_per_epoch} loss={running/step:.4f}", flush=True)

        # Dev proxy
        acc = dev_retrieval_accuracy(model, dev_loader, device=device, tau=args.tau)
        print(f"[DEV] epoch={epoch} retrieval@1={acc:.4f}")

        # Save best
        if acc > best_dev:
            best_dev = acc
            os.makedirs(os.path.dirname(args.out_ckpt), exist_ok=True)
            torch.save(model.state_dict(), args.out_ckpt)
            print(f"[SAVE] best checkpoint -> {args.out_ckpt} (retrieval@1={best_dev:.4f})")

    print("[DONE] training complete. Best dev retrieval@1:", best_dev)


if __name__ == "__main__":
    main()
