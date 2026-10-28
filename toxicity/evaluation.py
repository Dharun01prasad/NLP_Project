"""Read-only diagnostics; these examples are never automatically used for training."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import group_key, script_bucket
from .model import metrics, predict

DEFAULT_CASES = Path(__file__).resolve().parents[1] / "evaluation" / "smoke_cases.json"


def sliced_metrics(frame, probabilities, threshold):
    probabilities = np.asarray(probabilities)
    frame = frame.reset_index(drop=True).copy()
    frame["script"] = frame.comment.map(script_bucket)
    result = {"overall": metrics(frame.label, probabilities, threshold)}
    for column in ("language", "source", "script"):
        if column in frame:
            result[column] = {
                str(name): metrics(part.label, probabilities[part.index], threshold)
                for name, part in frame.groupby(column, dropna=False)
            }
    return result


def evaluate_cases(bundle, cases, training_groups=None):
    predictions = predict(bundle, [case["comment"] for case in cases])
    rows = []
    for case, prediction in zip(cases, predictions):
        expected = case.get("label")
        if expected not in (0, 1, None):
            raise ValueError("Diagnostic labels must be 0, 1, or null for context-dependent cases.")
        rows.append({**case, **prediction, "expected_label": expected,
                     "correct": None if expected is None else (prediction["label"] == "toxic") == bool(expected),
                     "training_overlap": None if training_groups is None else group_key(case["comment"]) in training_groups})
    scored = [row for row in rows if row["expected_label"] is not None]
    frame = pd.DataFrame([{**row, "label": row["expected_label"]} for row in scored])
    return {
        "model": bundle["report"]["selected_model"], "threshold": bundle["threshold"],
        "purpose": "Hand-authored development diagnostics, not an independent or human-validated accuracy benchmark.",
        "training_overlap_checked": training_groups is not None,
        "counts": {"total": len(rows), "scored": len(scored),
                   "failed": sum(row["correct"] is False for row in scored),
                   "training_overlap": None if training_groups is None else sum(row["training_overlap"] for row in rows)},
        "metrics": sliced_metrics(frame, [r["probability"] for r in scored], bundle["threshold"]) if scored else {},
        "results": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, default=Path("runtime"))
    parser.add_argument("--backend", choices=["auto", "baseline", "transformer"], default="auto")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output", type=Path, default=Path("runtime/reports/diagnostics.json"))
    parser.add_argument("--training-data", type=Path, help="Optional corpus for a normalized group overlap check")
    parser.add_argument("--fail-on-errors", action="store_true")
    args = parser.parse_args()
    from .inference import load_bundle
    bundle = load_bundle(args.runtime / "models", backend=args.backend)
    if bundle is None:
        parser.error("No model found in the specified runtime.")
    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    groups = None
    if args.training_data:
        data = pd.read_csv(args.training_data, keep_default_na=False, low_memory=False)
        if "split" in data:
            data = data[data.split.eq("train")]
        groups = set(data.comment.map(group_key))
    report = evaluate_cases(bundle, cases, groups)
    report["cases_sha256"] = hashlib.sha256(args.cases.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"counts": report["counts"], "output": str(args.output)}, indent=2))
    if args.fail_on_errors and report["counts"]["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
