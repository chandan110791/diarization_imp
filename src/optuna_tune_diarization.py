# optuna_tune_diarization.py
from __future__ import annotations

import os
import argparse
import math
import torch
import optuna

from pyannote.database import get_protocol, FileFinder
from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.audio import Model, Pipeline
from pyannote.audio.pipelines import SpeakerDiarization
from pyannote.database import get_protocol, FileFinder

# expects your class to exist in train.py
from train import PyannoteWavLMEmbeddingAdapter


SEGMENTATION_MODEL_ID = "pyannote/segmentation-3.0"
BASELINE_PIPELINE_ID = "pyannote/speaker-diarization-3.1"

from pyannote.database.util import load_uem

def build_protocol_with_uem(protocol_name: str, dev_uem_path: str, test_uem_path: str):
    base = get_protocol(protocol_name)

    preprocessors = dict(getattr(base, "preprocessors", {}) or {})
    preprocessors["audio"] = FileFinder()  # keep your audio injection

    dev_uem = load_uem(dev_uem_path)     # dict: uri -> Timeline
    test_uem = load_uem(test_uem_path)

    def annotated_preprocessor(file):
        uri = file["uri"]
        subset = file.get("subset", "").lower()  # train/dev/test (or development)
        if subset in ["dev", "development", "valid", "validation"]:
            u = dev_uem.get(uri, None)
        elif subset == "test":
            u = test_uem.get(uri, None)
        else:
            u = None

        # if still missing, fall back to reference extent (never empty)
        if u is None:
            u = file["annotation"].get_timeline().support()
        return u

    preprocessors["annotated"] = annotated_preprocessor

    return get_protocol(protocol_name, preprocessors=preprocessors)


def load_wavlm_adapter(
    device: torch.device,
    model_name: str,
    target_dim: int,
    duration: float,
    ckpt_path: str | None,
    freeze_backbone: bool = True,
):
    m = PyannoteWavLMEmbeddingAdapter(
        model_name=model_name,
        target_dim=target_dim,
        duration=duration,
        freeze_backbone=freeze_backbone,
    )
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        # support either plain state_dict or lightning-like dict
        state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

        # try to load robustly
        missing, unexpected = m.load_state_dict(state, strict=False)
        print("[INFO] Loaded ckpt with strict=False")
        if missing:
            print("[WARN] Missing keys:", missing[:10], "..." if len(missing) > 10 else "")
        if unexpected:
            print("[WARN] Unexpected keys:", unexpected[:10], "..." if len(unexpected) > 10 else "")

    m.to(device).eval()
    return m


def make_embedding_spec(
    backend: str,
    device: torch.device,
    hf_token: str,
    wavlm_model: str,
    proj_dim: int,
    wavlm_duration: float,
    wavlm_ckpt: str | None,
):
    if backend == "ecapa":
        base = Pipeline.from_pretrained(BASELINE_PIPELINE_ID, use_auth_token=hf_token)
        base.to(device)
        return base.embedding  # usually str/dict in 3.x

    if backend == "wavlm":
        return load_wavlm_adapter(
            device=device,
            model_name=wavlm_model,
            target_dim=proj_dim,
            duration=wavlm_duration,
            ckpt_path=wavlm_ckpt,
            freeze_backbone=True,  # Setting A fairness
        )

    raise ValueError("backend must be ecapa or wavlm")


def build_pipeline(device: torch.device, hf_token: str, embedding_spec):
    seg = Model.from_pretrained(SEGMENTATION_MODEL_ID, use_auth_token=hf_token).to(device)

    pipe = SpeakerDiarization(
        segmentation=seg,
        embedding=embedding_spec,
        embedding_exclude_overlap=True,
        clustering="AgglomerativeClustering",
    ).to(device)

    return pipe


def evaluate_der(pipe: SpeakerDiarization, files, params: dict, use_fp16: bool) -> float:
    # instantiate required
    pipe.instantiate(params)

    metric = DiarizationErrorRate()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autocast = torch.cuda.amp.autocast if (use_fp16 and device.type == "cuda") else None

    for f in files:
        if autocast:
            with autocast():
                hyp = pipe(f)
        else:
            hyp = pipe(f)

        uem = f.get("annotated", None)
        if uem is None or len(uem) == 0:
            raise RuntimeError(f"Bad/empty UEM for {f.get('uri')}.")

        metric(f["annotation"], hyp, uem=uem)

    return abs(metric)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--database_yml", required=True)
    ap.add_argument("--protocol", default="AMI.SpeakerDiarization.mini")
    ap.add_argument("--backend", choices=["ecapa", "wavlm"], required=True)

    ap.add_argument("--wavlm_model", default="microsoft/wavlm-base-plus")
    ap.add_argument("--proj_dim", type=int, default=256)
    ap.add_argument("--wavlm_duration", type=float, default=5.0)
    ap.add_argument("--wavlm_ckpt", default=None, help="Optional path to trained adapter checkpoint/state_dict")

    ap.add_argument("--trials", type=int, default=25)
    ap.add_argument("--max_dev_files", type=int, default=0, help="0 = all dev files; else subset for speed")
    ap.add_argument("--max_test_files", type=int, default=0, help="0 = all test files; else subset for speed")
    ap.add_argument("--fp16", action="store_true")

    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    hf_token = os.environ.get("HF_TOKEN", None)
    if not hf_token:
        raise RuntimeError("HF_TOKEN not set. export HF_TOKEN=hf_...")

    os.environ["PYANNOTE_DATABASE_CONFIG"] = os.path.abspath(args.database_yml)
    base = get_protocol(args.protocol)
    preprocessors = dict(getattr(base, "preprocessors", {}) or {})
    preprocessors["audio"] = FileFinder()
    protocol = build_protocol_with_uem(
        args.protocol,
        dev_uem_path="/content/diarization_imp/data/ami_mini/uems/dev.uem",
        test_uem_path="/content/diarization_imp/data/ami_mini/uems/test.uem",
    )
    dev_files = list(protocol.development())
    test_files = list(protocol.test())


    if args.max_dev_files and args.max_dev_files > 0:
        dev_files = dev_files[: args.max_dev_files]
    if args.max_test_files and args.max_test_files > 0:
        test_files = test_files[: args.max_test_files]

    print(f"[INFO] dev files: {len(dev_files)} | test files: {len(test_files)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embedding_spec = make_embedding_spec(
        backend=args.backend,
        device=device,
        hf_token=hf_token,
        wavlm_model=args.wavlm_model,
        proj_dim=args.proj_dim,
        wavlm_duration=args.wavlm_duration,
        wavlm_ckpt=args.wavlm_ckpt,
    )

    pipe = build_pipeline(device=device, hf_token=hf_token, embedding_spec=embedding_spec)

    # Determine which segmentation params exist in your installed version:
    # you reported only 'min_duration_off' exists.
    seg_param_keys = set(pipe._pipelines["segmentation"].parameters().keys())
    clu_param_keys = set(pipe._pipelines["clustering"].parameters().keys())
    print("[INFO] segmentation params:", sorted(seg_param_keys))
    print("[INFO] clustering params:", sorted(clu_param_keys))

    def objective(trial: optuna.Trial) -> float:
        # dev-only tuning
        params = {"segmentation": {}, "clustering": {}}

        # segmentation
        if "min_duration_off" in seg_param_keys:
            params["segmentation"]["min_duration_off"] = trial.suggest_float("min_duration_off", 0.0, 0.5)

        # clustering
        if "method" in clu_param_keys:
            params["clustering"]["method"] = "centroid"  # fixed
        if "min_cluster_size" in clu_param_keys:
            params["clustering"]["min_cluster_size"] = trial.suggest_int("min_cluster_size", 5, 30)
        if "threshold" in clu_param_keys:
            params["clustering"]["threshold"] = trial.suggest_float("clu_threshold", 0.30, 0.95)

        der = evaluate_der(pipe, dev_files, params=params, use_fp16=args.fp16)
        trial.set_user_attr("params", params)
        return der

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=args.trials)

    best_trial = study.best_trial
    best_params = best_trial.user_attrs["params"]

    print("\n" + "=" * 80)
    print("[BEST DEV] DER:", best_trial.value)
    print("[BEST DEV] Params:", best_params)
    print("=" * 80)

    # Evaluate once on test using the best dev params
    test_der = evaluate_der(pipe, test_files, params=best_params, use_fp16=args.fp16)
    print("\n" + "=" * 80)
    print(f"[TEST] backend={args.backend} DER: {test_der:.4f} ({test_der*100:.2f}%)")
    print("=" * 80)


if __name__ == "__main__":
    main()
