# benchmark.py
from __future__ import annotations

import os
import argparse
import torch
from pyannote.audio import Audio

from pyannote.database import get_protocol, FileFinder
from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.audio import Model, Pipeline
from pyannote.audio.pipelines import SpeakerDiarization

from train import PyannoteWavLMEmbeddingAdapter  # your train.py


SEGMENTATION_MODEL_ID = "pyannote/segmentation-3.0"
BASELINE_PIPELINE_ID = "pyannote/speaker-diarization-3.1"

# Your intended params (we will apply only those supported by your pipeline version)
DESIRED_PARAMS = {
    "segmentation": {"threshold": 0.444, "min_duration_off": 0.0},
    "clustering": {"method": "centroid", "min_cluster_size": 15, "threshold": 0.715},
}

import random, torch
import torch.nn.functional as F
from pyannote.audio import Audio

def probe_across_files(embedding_model, file_dicts, device, num_files=4, num_chunks_per_file=4, chunk_sec=5.0):
    audio = Audio(sample_rate=16000, mono=True)
    chosen = random.sample(file_dicts, k=min(num_files, len(file_dicts)))

    embs = []
    tags = []
    for f in chosen:
        waveform, sr = audio(f["audio"])
        T = waveform.shape[1]
        L = int(chunk_sec * sr)
        if T < L:
            continue
        for _ in range(num_chunks_per_file):
            s = random.randint(0, T - L)
            chunk = waveform[:, s:s+L]
            embs.append(chunk)
            tags.append(f["uri"])

    batch = torch.stack(embs, dim=0).to(device)  # (B,1,L)
    with torch.no_grad():
        E = F.normalize(embedding_model(batch), p=2, dim=1)

    # cosine matrix
    C = E @ E.T

    # compute same-uri vs different-uri cosine stats
    same = []
    diff = []
    for i in range(len(tags)):
        for j in range(i+1, len(tags)):
            (same if tags[i] == tags[j] else diff).append(C[i, j].item())

    print("\n[CHECK2++] Across-files cosine stats")
    print(" same-uri  : mean", sum(same)/len(same), "min", min(same), "max", max(same))
    print(" diff-uri  : mean", sum(diff)/len(diff), "min", min(diff), "max", max(diff))


import random
import torch.nn.functional as F

def probe_embedding_distribution(embedding_model, audio_path: str, device, num_chunks=12, chunk_sec=5.0):
    """
    Proper CHECK2: sample multiple chunks -> embeddings -> check diversity.
    If embeddings collapse, cosine similarities will be ~1.0 everywhere.
    """
    from pyannote.audio import Audio
    audio = Audio(sample_rate=16000, mono=True)
    waveform, sr = audio(audio_path)             # (1, T)
    T = waveform.shape[1]
    L = int(chunk_sec * sr)

    if T < L:
        raise RuntimeError(f"Audio too short for {chunk_sec}s probe: {audio_path}")

    # sample random start positions
    starts = [random.randint(0, T - L) for _ in range(num_chunks)]
    chunks = [waveform[:, s:s+L] for s in starts]   # list of (1, L)

    batch = torch.stack(chunks, dim=0).to(device)   # (B, 1, L)
    with torch.no_grad():
        emb = embedding_model(batch)                # (B, D)
        emb = F.normalize(emb, p=2, dim=1)

    # cosine similarity matrix (B x B)
    cos = emb @ emb.T
    offdiag = cos[~torch.eye(cos.size(0), dtype=torch.bool, device=cos.device)]

    print("\n[CHECK2+] Multi-chunk cosine similarity stats:")
    print("  cos mean:", offdiag.mean().item())
    print("  cos min :", offdiag.min().item())
    print("  cos max :", offdiag.max().item())
    print("  emb std (mean over dims):", emb.std(dim=0, unbiased=False).mean().item())

def make_embedding_spec(
    backend: str,
    device: torch.device,
    hf_token: str,
    wavlm_model: str = "microsoft/wavlm-base-plus",
    proj_dim: int = 256,
    wavlm_duration: float = 5.0,
):
    """
    pyannote.audio 3.x expects embedding= to be str/dict OR a pyannote.audio Model.
    """
    if backend == "ecapa":
        base = Pipeline.from_pretrained(BASELINE_PIPELINE_ID, use_auth_token=hf_token)
        base.to(device)
        return base.embedding

    if backend == "wavlm":
        m = PyannoteWavLMEmbeddingAdapter(
            model_name=wavlm_model,
            target_dim=proj_dim,
            duration=wavlm_duration,
            freeze_backbone=False,
        )
        m.to(device).eval()
        return m

    raise ValueError("backend must be one of: ecapa, wavlm")


def _subpipeline_leaf_keys(pipe: SpeakerDiarization, name: str) -> set[str]:
    sub = getattr(pipe, "_pipelines", {}).get(name, None)
    if sub is None:
        return set()
    try:
        return set(sub.parameters().keys())
    except Exception:
        return set()


def _filter_params_for_subpipeline(desired: dict, allowed_leaf: set[str]) -> dict:
    return {k: v for k, v in desired.items() if k in allowed_leaf}


def build_pipeline(device: torch.device, hf_token: str, embedding_spec):
    seg = Model.from_pretrained(SEGMENTATION_MODEL_ID, use_auth_token=hf_token).to(device)

    pipe = SpeakerDiarization(
        segmentation=seg,
        embedding=embedding_spec,
        embedding_exclude_overlap=True,
        clustering="AgglomerativeClustering",
    )
    pipe.to(device)

    print("[DEBUG] segmentation pipeline object:", pipe._pipelines["segmentation"])
    print("[DEBUG] segmentation leaf params:", pipe._pipelines["segmentation"].parameters())

    # Filter params according to what your installed version exposes
    seg_keys = _subpipeline_leaf_keys(pipe, "segmentation")
    clu_keys = _subpipeline_leaf_keys(pipe, "clustering")

    print("\n[DEBUG] segmentation leaf params:", sorted(seg_keys) if seg_keys else "(none)")
    print("[DEBUG] clustering leaf params:", sorted(clu_keys) if clu_keys else "(none)")

    seg_params = _filter_params_for_subpipeline(DESIRED_PARAMS.get("segmentation", {}), seg_keys)
    clu_params = _filter_params_for_subpipeline(DESIRED_PARAMS.get("clustering", {}), clu_keys)

    params_to_apply = {"segmentation": seg_params, "clustering": clu_params}
    print("\n[DEBUG] Applying params:", params_to_apply)

    # REQUIRED (your earlier error was because instantiate didn't happen)
    pipe.instantiate(params_to_apply)
    return pipe


def run_der(pipe, protocol_name: str, database_yml: str, use_fp16: bool,
            backend: str, embedding_spec=None, device=None, wavlm_duration: float = 5.0):
    # Make sure we load the right YAML
    os.environ["PYANNOTE_DATABASE_CONFIG"] = os.path.abspath(database_yml)

    preprocessors = {"audio": FileFinder()}
    protocol = get_protocol(protocol_name, preprocessors=preprocessors)

    test_files = list(protocol.test())
    if not test_files:
        raise RuntimeError("Protocol test() returned 0 files. Check your lists/test.txt paths.")

    # OPTIONAL: run across-file probe ONCE (before diarization loop)
    if backend == "wavlm" and embedding_spec is not None and device is not None:
        probe_across_files(embedding_spec, test_files, device=device, chunk_sec=wavlm_duration)

    # ---- sanity check: confirm audio is present and resolvable ----
    sample = test_files[0]
    print("\n[DEBUG] Sample keys:", sorted(sample.keys()))
    print("[DEBUG] Sample uri:", sample.get("uri", None))
    print("[DEBUG] Sample audio:", sample.get("audio", None))

    audio_path = sample.get("audio", None)
    if audio_path is None:
        raise RuntimeError("Missing 'audio' key. FileFinder didn't resolve paths. Check database.yml.")
    if isinstance(audio_path, str) and audio_path.startswith("/") and not os.path.exists(audio_path):
        raise FileNotFoundError(f"Resolved audio path does not exist: {audio_path}")

    metric = DiarizationErrorRate()

    device2 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autocast = torch.cuda.amp.autocast if (use_fp16 and device2.type == "cuda") else None

    for idx, f in enumerate(test_files):
        if autocast:
            with autocast():
                hyp = pipe(f)
        else:
            hyp = pipe(f)

        # UEM must be applied per file
        uem = f.get("annotated", None)
        if uem is None or len(uem) == 0:
            raise RuntimeError(f"Bad or empty UEM for uri={f.get('uri')}.")

        metric(f["annotation"], hyp, uem=uem)

        # CHECK3 prints per file (force flush so you see it during long runs)
        print(f"[CHECK3] {idx+1}/{len(test_files)} {f.get('uri')} predicted speakers: {len(hyp.labels())}", flush=True)

    return abs(metric)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default="AMI.SpeakerDiarization.mini")
    ap.add_argument("--database_yml", default="database.yml")
    ap.add_argument("--backend", choices=["ecapa", "wavlm"], required=True)
    ap.add_argument("--wavlm_model", default="microsoft/wavlm-base-plus")
    ap.add_argument("--proj_dim", type=int, default=256)
    ap.add_argument("--wavlm_duration", type=float, default=5.0)
    ap.add_argument("--fp16", action="store_true")
    args = ap.parse_args()

    hf_token = os.environ.get("HF_TOKEN", None)
    if not hf_token:
        raise RuntimeError("HF_TOKEN is not set. Run: export HF_TOKEN=hf_...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    embedding_spec = make_embedding_spec(
        backend=args.backend,
        device=device,
        hf_token=hf_token,
        wavlm_model=args.wavlm_model,
        proj_dim=args.proj_dim,
        wavlm_duration=args.wavlm_duration,
    )
    print(f"Embedding backend={args.backend} -> type={type(embedding_spec)}")

    pipe = build_pipeline(device=device, hf_token=hf_token, embedding_spec=embedding_spec)
	

    audio = Audio(sample_rate=16000, mono=True)

    # pick one test file to probe embeddings
    os.environ["PYANNOTE_DATABASE_CONFIG"] = os.path.abspath(args.database_yml)
    protocol = get_protocol(args.protocol, preprocessors={"audio": FileFinder()})
    probe = next(iter(protocol.test()))

    waveform, sr = audio(probe["audio"])  # waveform: (1, T)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.backend == "wavlm":
      probe_embedding_distribution(embedding_spec, probe["audio"], device=device, num_chunks=12, chunk_sec=args.wavlm_duration)


    # take a 5s chunk (match wavlm_duration) if possible
    chunk_len = int(5.0 * sr)
    wave = waveform[:, :chunk_len] if waveform.shape[1] >= chunk_len else waveform

    # make a batch: (B, C, T)
    batch = wave.unsqueeze(0).to(device)

    # Only for wavlm backend: check your embedding model directly
    if args.backend == "wavlm":
      emb = embedding_spec(batch)  # embedding_spec is your PyannoteWavLMEmbeddingAdapter
      norms = emb.norm(dim=1)
      print("\n[CHECK2] WavLM embedding norms:",
          "mean=", norms.mean().item(),
          "min=", norms.min().item(),
          "max=", norms.max().item())
      print("[CHECK2] Embedding variance (mean over dims):",
          emb.var(dim=0, unbiased=False).mean().item())


    der = run_der(
          pipe,
          args.protocol,
          args.database_yml,
          use_fp16=args.fp16,
          backend=args.backend,
          embedding_spec=embedding_spec,
          device=device,
          wavlm_duration=args.wavlm_duration)


    print("=" * 70)
    print(f"{args.backend} DER: {der:.2%}")
    print("=" * 70)


if __name__ == "__main__":
    main()
