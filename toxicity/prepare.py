"""Prepare locally available Tamil-English and English offensive-language data."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import pandas as pd

from .data import clean_frame, group_key
from .evaluation import DEFAULT_CASES
from .model import partition, split_fold


def prepare(data_dir, extra=None):
    frames, sources = [], []
    for split in ("train", "dev", "test"):
        path = data_dir / f"tamil_offensive_full_{split}.csv"
        frame = pd.read_csv(path, sep="\t", header=None, quoting=csv.QUOTE_NONE,
                            dtype=str, keep_default_na=False).iloc[:, :2]
        frame.columns = ["comment", "label"]
        frame["split"], frame["source"] = split, "dravidian_codemix"
        # Do not pretend script detection can identify English vs romanized Tamil.
        frame["language"] = "tamil_english_corpus"
        frames.append(clean_frame(frame)[0])
        sources.append({"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    english_path = data_dir / "labeled_data.csv"
    english_raw = pd.read_csv(english_path, keep_default_na=False, dtype=str)
    english_raw["source"], english_raw["language"] = "davidson", "english"
    # Davidson class IDs are NOT Mozhi's binary labels: 0=hate, 1=offensive, 2=neither.
    english, _ = clean_frame(english_raw, "tweet", "class", {"0": 1, "1": 1, "2": 0})
    remainder, test = split_fold(english, 5)
    train, dev = split_fold(remainder, 4)
    for split, part in (("train", train), ("dev", dev), ("test", test)):
        part["split"] = split
        frames.append(part)
    sources.append({"file": english_path.name, "sha256": hashlib.sha256(english_path.read_bytes()).hexdigest()})
    if extra:
        frame, _ = clean_frame(pd.read_csv(extra, keep_default_na=False))
        if not frame.split.isin(["train", "dev", "test"]).all():
            raise ValueError("Extra reviewed data must have explicit train/dev/test assignments.")
        if "source" not in frame:
            frame["source"] = "reviewed_extra"
        if "language" not in frame:
            frame["language"] = "unspecified"
        frames.append(frame)
        sources.append({"file": extra.name, "sha256": hashlib.sha256(extra.read_bytes()).hexdigest()})
    data, info = clean_frame(pd.concat(frames, ignore_index=True))
    # Keep the development diagnostic examples out of every corpus split.
    cases = json.loads(DEFAULT_CASES.read_text(encoding="utf-8"))
    reserved = {group_key(case["comment"]) for case in cases}
    removed = int(data.group.isin(reserved).sum())
    data = data[~data.group.isin(reserved)].copy()
    parts, split_info = partition(data)
    for name, part in parts.items():
        part["split"] = "dev" if name in {"calibration", "validation"} else name
    data = pd.concat(parts.values(), ignore_index=True)
    info.update(name="DravidianCodeMix + Davidson (local files)", sources=sources,
                split=split_info, diagnostic_overlap_removed=removed, labels_verified=False,
                saved_rows=len(data), source_counts=data.groupby(["source", "label"]).size().unstack(fill_value=0).to_dict("index"),
                limitations=["Davidson English is heavily skewed toward offensive tweets and has documented bias.",
                             "Add reviewed ordinary English conversation and context examples before deployment.",
                             "Language metadata is corpus-level; English vs romanized Tamil within DravidianCodeMix is not identified.",
                             "TamHateX and sentiment datasets are not merged because their label policies differ."])
    return data, info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("../DravidianCodeMix"))
    parser.add_argument("--extra", type=Path, help="Additional reviewed comment,label,split,language,source CSV")
    parser.add_argument("--output", type=Path, default=Path("training_data.csv"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a new --output path to preserve your previous data.")
    data, info = prepare(args.data_dir, args.extra)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.output, index=False, encoding="utf-8")
    args.output.with_suffix(".info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "rows": len(data), "source_counts": info["source_counts"]}, indent=2))


if __name__ == "__main__":
    main()
