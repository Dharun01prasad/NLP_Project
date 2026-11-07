"""Train the isolated V2 candidate on your GPU; MVP1 stays active by default."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from toxicity.language import train as train_language
from toxicity.train_transformer import train as train_toxicity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("runtime/data/v2/training.csv"))
    parser.add_argument("--output", type=Path, default=Path("runtime/v2_candidate"))
    parser.add_argument("--base-model", default="google/muril-base-cased")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--threshold-policy", choices=["balanced", "precision_first"], default="balanced")
    args = parser.parse_args()
    models = args.output / "models"
    if models.exists() and any(models.iterdir()):
        parser.error("Output models directory is not empty; choose another --output")
    report = train_toxicity(args.data, models, base_model=args.base_model,
                            epochs=args.epochs, batch_size=args.batch_size,
                            accumulation=args.accumulation, max_length=args.max_length,
                            device="cuda", threshold_policy=args.threshold_policy,
                            progress=lambda message: print(message, flush=True))
    language = train_language(args.data, models / "language.joblib")
    summary = {"toxicity_test": report["test_slices"], "language_test": language["test"],
               "activation": "Set MOZHI_RUNTIME to this output directory only after reviewing these metrics."}
    (args.output / "v2_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"summary": str(args.output / "v2_summary.json"),
                      "test_macro_f1": report["test_slices"]["overall"]["macro_f1"]}, indent=2))


if __name__ == "__main__":
    main()
