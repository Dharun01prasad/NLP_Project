"""Build an isolated multilingual training CSV; never change the MVP1 dataset."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path

import pandas as pd

from .data import clean_frame, group_key
from .evaluation import DEFAULT_CASES
from .model import partition

LATIN = re.compile(r"[A-Za-z]")
INDIC = re.compile(r"[\u0900-\u0d7f]")
LANGUAGES = {
    "tamil": ("tamil_offensive_full", "romanized_tamil"),
    "malayalam": ("mal_full_offensive", "romanized_malayalam"),
    "kannada": ("kannada_offensive", "romanized_kannada"),
}


def romanized(text: str) -> bool:
    return bool(LATIN.search(text)) and not INDIC.search(text)


def archive_rows(archive: Path, language: str) -> pd.DataFrame:
    prefix, tag = LANGUAGES[language]
    rows = []
    with zipfile.ZipFile(archive) as source:
        for split in ("train", "dev", "test"):
            matches = [name for name in source.namelist() if name.endswith(f"/{prefix}_{split}.csv")]
            if len(matches) != 1:
                raise ValueError(f"Expected one {prefix}_{split}.csv in {archive}; found {len(matches)}")
            content = source.read(matches[0]).decode("utf-8-sig")
            for line in content.splitlines():
                cells = line.split("\t")
                while cells and not cells[-1].strip():
                    cells.pop()
                if len(cells) < 2:
                    continue
                comment, label = " ".join(cells[:-1]).strip(), cells[-1].strip()
                if romanized(comment):
                    rows.append({"comment": comment, "label": label, "split": split,
                                 "source": f"dravidian_{language}", "language": tag})
    return pd.DataFrame(rows)


def hinglish_rows(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, keep_default_na=False)
    if not {"comment", "label"}.issubset(frame.columns):
        raise ValueError("Hinglish source needs comment,label columns")
    frame = frame[frame.comment.map(romanized)].copy()
    # Stable 70/15/15 split by normalized comment group, independent of row order.
    bucket = frame.comment.map(lambda value: int(hashlib.sha256(group_key(value).encode()).hexdigest()[:8], 16) % 100)
    frame["split"] = bucket.map(lambda value: "train" if value < 70 else "dev" if value < 85 else "test")
    frame["source"], frame["language"] = "textdetox_hinglish", "romanized_hindi"
    return frame


def prepare(base: Path, archive: Path, hinglish: Path):
    original = pd.read_csv(base, keep_default_na=False, low_memory=False)
    if not {"comment", "label", "split"}.issubset(original.columns):
        raise ValueError("Base training CSV needs comment,label,split")
    original["language"] = original.get("language", "unspecified")
    original["language"] = original.language.replace({"tamil_english_corpus": "romanized_tamil"})
    tamil_native = original.language.eq("romanized_tamil") & ~original.comment.map(romanized)
    original.loc[tamil_native, "language"] = "tamil_script"
    additions = [archive_rows(archive, language) for language in ("malayalam", "kannada")]
    additions.append(hinglish_rows(hinglish))
    combined, cleaning = clean_frame(pd.concat([original, *additions], ignore_index=True))
    reserved = {group_key(item["comment"]) for item in json.loads(DEFAULT_CASES.read_text(encoding="utf-8"))}
    combined = combined[~combined.group.isin(reserved)].copy()
    parts, split_info = partition(combined)
    for name, part in parts.items():
        part["split"] = "dev" if name in {"calibration", "validation"} else name
    data = pd.concat(parts.values(), ignore_index=True)
    info = {"name": "Mozhi multilingual candidate V2", "rows": len(data), "cleaning": cleaning,
            "split": split_info, "source_language_counts": data.groupby(["split", "language", "label"]).size().to_dict(),
            "sources": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (base, archive, hinglish)},
            "limitations": ["Corpus language tags are weak labels, especially for code mixed Latin comments.",
                            "TextDetox Hinglish slice has no original source listed; audit before presenting per-language claims.",
                            "Offensive-language datasets have different labeling policies; compare on held-out slices."]}
    info["source_language_counts"] = {"|".join(map(str, key)): int(count) for key, count in info["source_language_counts"].items()}
    return data, info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("training_data.csv"))
    parser.add_argument("--archive", type=Path, default=Path("runtime/data/source/v2/dravidian.zip"))
    parser.add_argument("--hinglish", type=Path, default=Path("runtime/data/source/v2/hinglish_toxicity.csv"))
    parser.add_argument("--output", type=Path, default=Path("runtime/data/v2/training.csv"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a new path to preserve the previous data")
    data, info = prepare(args.base, args.archive, args.hinglish)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.output, index=False)
    args.output.with_suffix(".info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "rows": len(data), "counts": info["source_language_counts"]}, indent=2))


if __name__ == "__main__":
    main()
