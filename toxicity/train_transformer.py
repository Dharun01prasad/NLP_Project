"""Fine-tune a multilingual encoder, calibrate it, and export for offline inference."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit

from .data import clean_frame, group_key
from .evaluation import DEFAULT_CASES, sliced_metrics
from .inference import transformer_text
from .model import SEED, partition, pick_threshold


def fit_temperature(margins, labels):
    margins, labels = np.asarray(margins), np.asarray(labels)
    def loss(temperature):
        scaled = margins / temperature
        return float(np.mean(np.logaddexp(0, scaled) - labels * scaled))
    result = minimize_scalar(loss, bounds=(0.05, 10.0), method="bounded")
    return float(result.x) if result.success else 1.0


def train(data_path, output, base_model="google/muril-base-cased", epochs=3,
          batch_size=8, accumulation=4, max_length=128, learning_rate=2e-5,
          device="auto", threshold_policy="balanced", progress=print):
    import torch
    import transformers
    from torch.utils.data import DataLoader, WeightedRandomSampler
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, set_seed

    if min(epochs, batch_size, accumulation) < 1 or not 8 <= max_length <= 512 or learning_rate <= 0:
        raise ValueError("Invalid epochs, batch size, accumulation, token limit or learning rate.")
    if threshold_policy not in {"balanced", "precision_first"}:
        raise ValueError("Unknown threshold policy.")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose an empty output directory; previous checkpoints are never overwritten.")
    raw = pd.read_csv(data_path, keep_default_na=False, low_memory=False)
    frame, cleaning = clean_frame(raw)
    cases = json.loads(DEFAULT_CASES.read_text(encoding="utf-8"))
    reserved = {group_key(case["comment"]) for case in cases}
    overlap = frame.group.isin(reserved)
    cleaning["diagnostic_overlap_removed"] = int(overlap.sum())
    frame = frame[~overlap].copy()
    parts, split_info = partition(frame)
    set_seed(SEED)
    torch.set_num_threads(2)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable. Select a GPU runtime in Colab or use --device cpu.")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model, num_labels=2, id2label={0: "non_toxic", 1: "toxic"},
        label2id={"non_toxic": 0, "toxic": 1}, trust_remote_code=False).to(device)
    if max_length > getattr(model.config, "max_position_embeddings", 512):
        raise ValueError("Token limit exceeds the base model's position limit.")
    if device == "cuda":
        model.gradient_checkpointing_enable()
    # Tokenize once without padding. Pad only within each batch to reduce GPU memory.
    encoded_parts = {}
    truncation = {}
    for name, part in parts.items():
        texts = [transformer_text(t) for t in part.comment]
        tokens = tokenizer(texts, truncation=True, max_length=max_length, padding=False)
        encoded_parts[name] = [
            {**{key: values[i] for key, values in tokens.items()}, "labels": int(label)}
            for i, label in enumerate(part.label)
        ]
        lengths = tokenizer(texts, truncation=False, padding=False, return_length=True)["length"]
        truncation[name] = int(sum(length > max_length for length in lengths))

    def collate(rows):
        return tokenizer.pad(rows, padding=True, return_tensors="pt")

    train_frame = parts["train"]
    # Equal source/class sampling prevents the English source's ~83% offensive prior
    # from dominating training. Calibration uses the untouched calibration distribution.
    strata = train_frame.label.astype(str)
    if "source" in train_frame:
        strata = train_frame.source.astype(str) + ":" + strata
    weights = (1 / strata.map(strata.value_counts())).to_numpy()
    sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), len(weights), replacement=True)
    train_loader = DataLoader(encoded_parts["train"], batch_size=batch_size, sampler=sampler, collate_fn=collate)

    def margins(name):
        model.eval()
        values = []
        loader = DataLoader(encoded_parts[name], batch_size=batch_size, collate_fn=collate)
        with torch.inference_mode():
            for batch in loader:
                batch.pop("labels")
                logits = model(**{k: v.to(device) for k, v in batch.items()}).logits.float().cpu().numpy()
                values.extend(logits[:, 1] - logits[:, 0])
        return np.asarray(values)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    steps = math.ceil(len(train_loader) / accumulation) * epochs
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, int(steps * 0.1), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    checkpoint = output / "transformer"
    checkpoint.mkdir(parents=True, exist_ok=True)
    best_score, best_epoch, history = -1.0, 0, []
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            # Correct normalization for the final incomplete accumulation group.
            group_size = min(accumulation, len(train_loader) - (step // accumulation) * accumulation)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
                loss = model(**batch).loss
            total_loss += float(loss.detach())
            scaler.scale(loss / group_size).backward()
            if (step + 1) % accumulation == 0 or step + 1 == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= previous_scale:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if (step + 1) % 100 == 0:
                progress(f"Epoch {epoch}/{epochs}, batch {step + 1}/{len(train_loader)}")
        val_p = expit(margins("validation"))
        threshold = pick_threshold(parts["validation"].label, val_p, threshold_policy)
        evaluation = sliced_metrics(parts["validation"], val_p, threshold)
        source_scores = [s["macro_f1"] for s in evaluation.get("source", {}).values()]
        score = float(np.mean(source_scores)) if source_scores else evaluation["overall"]["macro_f1"]
        history.append({"epoch": epoch, "training_loss": total_loss / len(train_loader),
                        "selection_macro_f1": score, "validation": evaluation, "threshold": threshold})
        progress(f"Epoch {epoch}: validation selection macro-F1 {score:.4f}")
        if score > best_score:
            best_score, best_epoch = score, epoch
            model.save_pretrained(checkpoint, safe_serialization=True)
            tokenizer.save_pretrained(checkpoint)
    # Release the optimizer state before loading the winning checkpoint.
    del optimizer, scheduler, model
    if device == "cuda":
        torch.cuda.empty_cache()
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, local_files_only=True, use_safetensors=True).to(device)
    temperature = fit_temperature(margins("calibration"), parts["calibration"].label)
    val_p = expit(margins("validation") / temperature)
    threshold = pick_threshold(parts["validation"].label, val_p, threshold_policy)
    validation = sliced_metrics(parts["validation"], val_p, threshold)
    # The test split is evaluated once, after checkpoint, temperature and threshold selection.
    test = sliced_metrics(parts["test"], expit(margins("test") / temperature), threshold)
    name = f"Fine-tuned {base_model}"
    info_path = Path(data_path).with_suffix(".info.json")
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {"name": Path(data_path).name}
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "dataset": info,
              "dataset_fingerprint": hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
              "cleaning": cleaning, "split": split_info, "selected_model": name, "selected_index": 0,
              "backend": "transformer", "base_model": base_model, "best_epoch": best_epoch,
              "selection_rule": "Highest mean validation macro-F1 across sources (overall if source absent)",
              "threshold_rule": threshold_policy, "temperature": temperature, "max_length": max_length,
              "human_validated": False, "score_meaning": "Predicted offensive-language likelihood; not severity or a hate-specific label",
              "training": {"epochs": epochs, "batch_size": batch_size, "accumulation": accumulation,
                           "learning_rate": learning_rate, "seed": SEED, "device": device,
                           "sampling": "Equal source/class strata with replacement; training only"},
              "truncated_rows": truncation, "epochs": history,
              "validation_slices": validation, "test_slices": test,
              "experiments": [{"name": name, "kind": "transformer", "augmented": False,
                               "threshold": threshold, "validation": validation["overall"],
                               "test": test["overall"], "robustness": {}}],
              "limitations": ["Source labels and diagnostic labels require independent human review.",
                              "Source-domain calibration may not transfer to real conversations.",
                              "Only the first max_length tokens are classified; longer inputs require review.",
                              "Latin script is not a reliable language identifier. Validate English and Thanglish separately.",
                              "Binary offensive language is not equivalent to identity-based hate speech."],
              "versions": {"torch": torch.__version__, "transformers": transformers.__version__}}
    (checkpoint / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    # Write the activation manifest last: incomplete training is never a deployable export.
    manifest = {"schema_version": 1, "labels": {"0": "non_toxic", "1": "toxic"},
                "temperature": temperature, "threshold": threshold, "max_length": max_length}
    (checkpoint / "mozhi.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    progress(f"Export complete: {checkpoint}. Test macro-F1: {test['overall']['macro_f1']:.4f}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("trained"))
    parser.add_argument("--base-model", default="google/muril-base-cased")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--threshold-policy", choices=["balanced", "precision_first"], default="balanced")
    args = parser.parse_args()
    train(args.data, args.output, args.base_model, args.epochs, args.batch_size, args.accumulation,
          args.max_length, args.learning_rate, args.device, args.threshold_policy,
          progress=lambda message: print(message, flush=True))


if __name__ == "__main__":
    main()
