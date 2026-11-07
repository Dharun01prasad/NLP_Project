"""Download the two public source files for the optional multilingual V2 run.

Run this separately from training. Files stay under runtime/data, which Git ignores.
The Hinglish data is read through the Hugging Face dataset viewer API so that
the normal project requirements do not need a Parquet dependency.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
import urllib.parse
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DRAVIDIAN_URL = "https://zenodo.org/records/4750858/files/DravidianCodeMix-2020.zip?download=1"
DRAVIDIAN_MD5 = "7850be52919a387f5b36c7a09b05ad87"
HINGLISH_PAGE = "https://huggingface.co/datasets/textdetox/multilingual_toxicity_dataset"
HINGLISH_API = "https://datasets-server.huggingface.co/rows"
DATASET_ID = "textdetox/multilingual_toxicity_dataset"
PAGE_SIZE = 100


def read_url(url: str, limit: int = 25_000_000) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozhi-research/2.0"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                data = response.read(limit + 1)
            if len(data) > limit:
                raise ValueError(f"Source exceeded the {limit}-byte limit: {url}")
            return data
        except (TimeoutError, urllib.error.URLError):
            if attempt == 2:
                raise
            time.sleep(attempt + 1)
    raise RuntimeError("Unreachable")


def fetch_dravidian(directory: Path) -> dict:
    path = directory / "dravidian.zip"
    if not path.exists():
        payload = read_url(DRAVIDIAN_URL)
        path.write_bytes(payload)
    payload = path.read_bytes()
    if hashlib.md5(payload).hexdigest() != DRAVIDIAN_MD5:
        path.unlink(missing_ok=True)
        raise ValueError("DravidianCodeMix checksum mismatch; discarded the archive.")
    return {"path": str(path), "url": DRAVIDIAN_URL, "md5": DRAVIDIAN_MD5,
            "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}


def fetch_hinglish(directory: Path) -> dict:
    path = directory / "hinglish_toxicity.csv"
    if not path.exists():
        def page(offset: int) -> dict:
            query = urllib.parse.urlencode({"dataset": DATASET_ID, "config": "default",
                                            "split": "hin", "offset": offset, "length": PAGE_SIZE})
            return json.loads(read_url(f"{HINGLISH_API}?{query}", 2_000_000))

        first = page(0)
        total = first["num_rows_total"]
        if not 100 <= total <= 50_000:
            raise ValueError(f"Unexpected Hinglish row count: {total}")
        offsets = list(range(PAGE_SIZE, total, PAGE_SIZE))
        with ThreadPoolExecutor(max_workers=4) as pool:
            pages = [first, *pool.map(page, offsets)]
        rows = []
        for index, batch in enumerate(pages):
            for entry in batch["rows"]:
                row = entry["row"]
                if entry["row_idx"] != len(rows) or not isinstance(row["text"], str) or row["toxic"] not in (0, 1):
                    raise ValueError(f"Malformed Hinglish dataset row in page {index}")
                rows.append((row["text"], row["toxic"]))
        if len(rows) != total:
            raise ValueError("Hinglish dataset download was incomplete.")
        with path.open("w", encoding="utf-8", newline="") as output:
            writer = csv.writer(output)
            writer.writerow(("comment", "label"))
            writer.writerows(rows)
    payload = path.read_bytes()
    with path.open(encoding="utf-8", newline="") as source:
        count = sum(1 for _ in csv.DictReader(source))
    return {"path": str(path), "url": HINGLISH_PAGE, "split": "hin", "rows": count,
            "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload),
            "provenance_note": "TextDetox lists no original source for its Hinglish slice; audit labels before claims."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runtime/data/source/v2"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"dravidian_codemix": fetch_dravidian(args.output),
              "textdetox_hinglish": fetch_hinglish(args.output)}
    (args.output / "sources.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
