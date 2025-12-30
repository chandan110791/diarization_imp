# dataset_segments.py
from __future__ import annotations

import os
import json
import argparse
import random
from typing import List, Tuple, Dict

from pyannote.database import get_protocol, FileFinder
from pyannote.core import Segment, Timeline, Annotation


def _annotation_boundaries(annotation: Annotation) -> List[float]:
    """Collect all unique boundaries from an annotation's segments."""
    bounds = set()
    for seg, _track, _label in annotation.itertracks(yield_label=True):
        bounds.add(float(seg.start))
        bounds.add(float(seg.end))
    return sorted(bounds)


def _active_labels(annotation: Annotation, seg: Segment) -> List[str]:
    """Return list of active speaker labels in seg (may be empty)."""
    cropped = annotation.crop(seg, mode="intersection")
    # pyannote.core.Annotation supports `labels()`
    labels = list(cropped.labels())
    return labels


def single_speaker_regions(
    annotation: Annotation,
    collar: float = 0.0,
    min_region_dur: float = 0.0,
) -> List[Tuple[Segment, str]]:
    """
    Convert arbitrary RTTM annotation into non-overlapping regions labeled
    with exactly 1 active speaker. Adjacent regions with same speaker are merged.

    collar: trims each resulting region by collar seconds on both ends
            (helps avoid boundary noise).
    """
    bounds = _annotation_boundaries(annotation)
    if len(bounds) < 2:
        return []

    regions: List[Tuple[Segment, str]] = []

    # build atomic intervals between boundaries
    for t0, t1 in zip(bounds[:-1], bounds[1:]):
        if t1 <= t0:
            continue
        seg = Segment(t0, t1)
        labels = _active_labels(annotation, seg)
        if len(labels) != 1:
            continue

        spk = labels[0]
        # apply collar trimming
        s = seg.start + collar
        e = seg.end - collar
        if e <= s:
            continue
        seg2 = Segment(s, e)
        if seg2.duration < min_region_dur:
            continue

        regions.append((seg2, spk))

    # merge adjacent regions with same speaker (touching/near-touching)
    merged: List[Tuple[Segment, str]] = []
    eps = 1e-6
    for seg, spk in regions:
        if not merged:
            merged.append((seg, spk))
            continue
        last_seg, last_spk = merged[-1]
        if spk == last_spk and abs(seg.start - last_seg.end) <= eps:
            merged[-1] = (Segment(last_seg.start, seg.end), spk)
        else:
            merged.append((seg, spk))

    return merged


def sample_chunks_from_regions(
    uri: str,
    audio_path: str,
    regions: List[Tuple[Segment, str]],
    chunk_dur: float,
    max_chunks_per_file: int,
    seed: int,
) -> List[Dict]:
    """
    Sample fixed-duration chunks uniformly from single-speaker regions.
    Output rows are JSON-serializable dicts.
    """
    rng = random.Random(seed + hash(uri) % 10_000_000)
    rows: List[Dict] = []

    # collect candidate windows (region start range)
    candidates: List[Tuple[float, float, str]] = []
    for seg, spk in regions:
        if seg.duration < chunk_dur:
            continue
        # allowable start interval [seg.start, seg.end - chunk_dur]
        candidates.append((float(seg.start), float(seg.end - chunk_dur), spk))

    if not candidates:
        return rows

    # sample starts by choosing random candidate region each time
    for _ in range(max_chunks_per_file):
        a, b, spk = rng.choice(candidates)
        start = a if b <= a else rng.uniform(a, b)
        rows.append(
            {
                "uri": uri,
                "audio": audio_path,
                "start": float(start),
                "duration": float(chunk_dur),
                "speaker": spk,
            }
        )

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--database_yml", required=True, help="Path to database.yml")
    ap.add_argument("--protocol", default="AMI.SpeakerDiarization.mini")
    ap.add_argument("--split", choices=["train", "dev", "test"], default="train")

    ap.add_argument("--chunk_dur", type=float, default=3.0, help="Chunk duration in seconds")
    ap.add_argument("--max_chunks_per_file", type=int, default=20, help="Cap chunk samples per file")

    ap.add_argument("--collar", type=float, default=0.10, help="Trim regions by collar seconds")
    ap.add_argument("--min_region_dur", type=float, default=1.0, help="Min single-speaker region dur")

    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", required=True, help="Output JSONL path")

    args = ap.parse_args()

    os.environ["PYANNOTE_DATABASE_CONFIG"] = os.path.abspath(args.database_yml)
    protocol = get_protocol(args.protocol, preprocessors={"audio": FileFinder()})

    if args.split == "train":
        files = list(protocol.train())
    elif args.split == "dev":
        files = list(protocol.development())
    else:
        files = list(protocol.test())

    print(f"[INFO] Loaded {len(files)} files for split={args.split}")

    n_written = 0
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f_out:
        for idx, file in enumerate(files, start=1):
            uri = file["uri"]
            audio = str(file.get("audio", None))
            ann: Annotation = file["annotation"]

            if not audio:
                print(f"[WARN] {uri}: missing audio path, skipping.")
                continue

            regions = single_speaker_regions(ann, collar=args.collar, min_region_dur=args.min_region_dur)
            rows = sample_chunks_from_regions(
                uri=uri,
                audio_path=audio,
                regions=regions,
                chunk_dur=args.chunk_dur,
                max_chunks_per_file=args.max_chunks_per_file,
                seed=args.seed,
            )

            for r in rows:
                r["split"] = args.split
                f_out.write(json.dumps(r) + "\n")
                n_written += 1

            if idx % 10 == 0:
                print(f"[INFO] processed {idx}/{len(files)} files | wrote {n_written} chunks")

    print(f"[DONE] Wrote {n_written} chunks -> {args.out}")


if __name__ == "__main__":
    main()
