from __future__ import annotations

import csv
import hashlib
import io
import re
import unicodedata
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

import pandas as pd

SOURCE_URL = "https://zenodo.org/records/4750858/files/DravidianCodeMix-2020.zip?download=1"
SOURCE_PAGE = "https://zenodo.org/records/4750858"
SOURCE_MD5 = "7850be52919a387f5b36c7a09b05ad87"


def normalize(text: str) -> str:
    """Conservative cleaning: retain Tamil, emoji, negation and punctuation."""
    text = unicodedata.normalize("NFKC", str(text)).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)
    text = re.sub(r"(?<!\w)@[\w.]+", " USER ", text)
    return re.sub(r"\s+", " ", text).strip()


def word_tokens(text: str) -> list[str]:
    """Keep Unicode letters, numbers and combining marks in the same word.

    Python's regex \\w excludes marks such as Tamil vowel signs and pulli.
    """
    words, current = [], []
    for char in text:
        category = unicodedata.category(char)[0]
        if category in {"L", "N"} or (category == "M" and current):
            current.append(char)
        elif current:
            words.append("".join(current))
            current = []
    if current:
        words.append("".join(current))
    return words


def script_bucket(text: str) -> str:
    tamil = bool(re.search(r"[\u0b80-\u0bff]", text))
    latin = bool(re.search(r"[a-zA-Z]", text))
    if tamil:
        return "tamil_and_latin" if latin else "tamil_script"
    # Latin script alone does not distinguish English from romanized Tamil.
    return "latin_only" if latin else "other"


def group_key(text: str) -> str:
    # Conservatively group punctuation/repeated-letter variants to limit leakage.
    value = normalize(text)
    value = "".join(c for c in value if c.isspace() or unicodedata.category(c)[0] in {"L", "M", "N"})
    value = re.sub(r"([a-z])\1+", r"\1", value)
    return re.sub(r"\s+", "", value) or normalize(text)


def row_id(text: str) -> str:
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()[:20]


def label_key(value) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def binary_label(value, custom: dict | None = None) -> int | None:
    key = label_key(value)
    if custom:
        mapping = {label_key(k): v for k, v in custom.items()}
        if key not in mapping:
            raise ValueError(f"Unmapped label: {value!r}. Supply an explicit label mapping.")
        result = mapping[key]
        if result not in (0, 1, None):
            raise ValueError("Mappings must contain only 0, 1 or null (exclude).")
        return result
    if str(value).strip() in {"0", "0.0"} or key in {"nonoffensive", "notoffensive", "nontoxic"}:
        return 0
    if str(value).strip() in {"1", "1.0"} or key in {"toxic", "offensive", "abusive"}:
        return 1
    if key.startswith("offensiveuntarget") or key.startswith("offensivetargetedinsult"):
        return 1
    if key in {"nottamil", "notintendedlanguage", "otherlanguage"}:
        return None
    raise ValueError(f"Unknown label {value!r}. Expected 0/1 or explicit offensive-language labels. Sentiment labels are not toxicity labels.")


def clean_frame(raw: pd.DataFrame, text_column="comment", label_column="label", mapping=None):
    if text_column not in raw or label_column not in raw:
        raise ValueError(f"Required columns: {text_column!r}, {label_column!r}. Found: {list(raw.columns)}")
    rows, excluded = [], Counter()
    for _, item in raw.iterrows():
        text = "" if pd.isna(item[text_column]) else str(item[text_column]).strip()
        if not text or not normalize(text):
            excluded["empty_text"] += 1
            continue
        if len(text) > 10000:
            excluded["over_10000_characters"] += 1
            continue
        label = binary_label(item[label_column], mapping)
        if label is None:
            excluded["other_language"] += 1
            continue
        split = str(item.get("split", "")).lower().strip()
        split = {"validation": "dev", "val": "dev", "development": "dev"}.get(split, split)
        if split not in {"", "train", "dev", "test"}:
            raise ValueError(f"Unrecognized split {split!r}; use train/dev/test or omit split.")
        row = {"id": row_id(text), "comment": text, "label": label,
                     "original_label": str(item.get("original_label", item[label_column])),
                     "split": split, "group": group_key(text)}
        for column in ("language", "source"):
            if column in raw:
                row[column] = str(item[column]).strip() if pd.notna(item[column]) else "unspecified"
                row[column] = row[column] or "unspecified"
        rows.append(row)
    if not rows:
        raise ValueError("No usable labeled comments were found.")
    df = pd.DataFrame(rows)
    conflicts = set(df.groupby("id").label.nunique().loc[lambda x: x > 1].index)
    excluded["conflicting_duplicate_rows"] = int(df.id.isin(conflicts).sum())
    df = df.loc[~df.id.isin(conflicts)].copy()
    # Prefer the held-out copy if a comment appears in several official splits.
    df["priority"] = df.split.map({"test": 0, "dev": 1, "train": 2, "": 3})
    df = df.sort_values("priority", kind="stable")
    before = len(df)
    df = df.drop_duplicates("id").drop(columns="priority")
    excluded["duplicate_rows"] = before - len(df)
    if df.label.nunique() < 2:
        raise ValueError("The dataset needs both toxic and non-toxic examples.")
    return df.reset_index(drop=True), {
        "input_rows": len(raw), "usable_rows": len(df), "excluded": dict(excluded),
        "class_counts": {"non_toxic": int((df.label == 0).sum()), "toxic": int((df.label == 1).sum())},
    }


def read_upload(content: str, text_column="comment", label_column="label", mapping=None):
    content = content.lstrip("\ufeff")
    try:
        dialect = csv.Sniffer().sniff(content[:10000], delimiters=",\t;")
        raw = pd.read_csv(io.StringIO(content), sep=dialect.delimiter, dtype=str, keep_default_na=False)
    except (csv.Error, pd.errors.ParserError) as exc:
        raise ValueError("Could not parse CSV/TSV. Use a header and quote comments containing separators.") from exc
    return clean_frame(raw, text_column, label_column, mapping)


def fetch_research_dataset(cache_dir: Path):
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive = cache_dir / "DravidianCodeMix-2020.zip"
    if not archive.exists():
        request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "Mozhi-research/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = response.read(25_000_001)
        if len(payload) > 25_000_000:
            raise ValueError("Unexpectedly large dataset archive.")
        archive.write_bytes(payload)
    if hashlib.md5(archive.read_bytes()).hexdigest() != SOURCE_MD5:
        archive.unlink(missing_ok=True)
        raise ValueError("Source archive checksum mismatch. Please download again.")
    frames = []
    with zipfile.ZipFile(archive) as z:
        for split in ("train", "dev", "test"):
            name = f"DravidianCodeMix/tamil_offensive_full_{split}.csv"
            part = pd.read_csv(io.BytesIO(z.read(name)), sep="\t", header=None,
                               quoting=csv.QUOTE_NONE, keep_default_na=False, dtype=str)
            part = part.iloc[:, :2]
            part.columns = ["comment", "label"]
            part["split"] = split
            frames.append(part)
    df, info = clean_frame(pd.concat(frames, ignore_index=True))
    info.update({"name": "DravidianCodeMix · Tamil offensive-language splits", "source": SOURCE_PAGE,
                 "labels_verified": False, "checksum_md5": SOURCE_MD5})
    return df, info
