from __future__ import annotations

import hashlib
import json
import os
import platform
import re
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from threadpoolctl import threadpool_limits

from .data import group_key, normalize, word_tokens

SEED = 42


def build_pipeline(kind="word_char"):
    word = TfidfVectorizer(preprocessor=normalize, lowercase=False, ngram_range=(1, 2),
                           sublinear_tf=True, min_df=2, max_features=35000,
                           tokenizer=word_tokens, token_pattern=None)
    features = word if kind == "word" else FeatureUnion([
        ("word", word),
        ("char", TfidfVectorizer(preprocessor=normalize, lowercase=False, analyzer="char",
                                 ngram_range=(3, 5), min_df=3, sublinear_tf=True,
                                 max_features=65000)),
    ])
    return Pipeline([("features", features), ("classifier", LogisticRegression(
        C=2.0, class_weight="balanced", max_iter=500, solver="liblinear", random_state=SEED))])


def split_fold(df: pd.DataFrame, folds: int):
    counts = df.groupby("label").group.nunique()
    if len(counts) < 2 or counts.min() < folds:
        raise ValueError(f"Need at least {folds} distinct comment groups per class for this split. Add more labeled data.")
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=SEED)
    a, b = next(splitter.split(df.comment, df.label, df.group))
    return df.iloc[a].copy(), df.iloc[b].copy()


def partition(df: pd.DataFrame):
    df = df.copy()
    if "group" not in df:
        df["group"] = df.comment.map(group_key)
    has_split = df.split.ne("")
    if has_split.any() and not has_split.all():
        raise ValueError("Either specify a split for every row, or omit splits for every row.")
    excluded = 0
    if has_split.all():
        if set(df.split.unique()) != {"train", "dev", "test"}:
            raise ValueError("Explicit splits must include train, dev, and test.")
        test = df[df.split == "test"].copy()
        dev = df[(df.split == "dev") & ~df.group.isin(test.group)].copy()
        train = df[(df.split == "train") & ~df.group.isin(set(test.group) | set(dev.group))].copy()
        excluded = len(df) - len(train) - len(dev) - len(test)
        strategy = "Official splits; overlapping normalized variant groups removed from train/dev"
    else:
        remainder, test = split_fold(df, 5)
        train, dev = split_fold(remainder, 4)
        strategy = "Seeded stratified group holdout, approximately 60/20/20"
    calibration, validation = split_fold(dev, 2)
    parts = {"train": train, "calibration": calibration, "validation": validation, "test": test}
    for name, part in parts.items():
        if part.label.nunique() != 2 or part.label.value_counts().min() < 3:
            raise ValueError(f"Too few examples in {name}. Use at least 60 examples per class for reliable splitting.")
    info = {"strategy": strategy, "overlap_rows_excluded": excluded,
            "counts": {name: {"rows": len(part), "toxic": int(part.label.sum())} for name, part in parts.items()}}
    return parts, info


def spelling_variant(text: str, mode="repeat") -> str:
    # Script-safe: only edit Latin words, never Tamil combining characters.
    match = re.search(r"\b[A-Za-z]{4,}\b", text)
    if not match:
        return text
    word = match.group()
    changed = word[0] * 3 + word[1:] if mode == "repeat" else ".".join(word)
    return text[:match.start()] + changed + text[match.end():]


def augment_training(df: pd.DataFrame):
    # Both classes get the same augmentation rate. Variants never leave training.
    sample = df.sample(frac=0.3, random_state=SEED).copy()
    sample["comment"] = sample.comment.map(spelling_variant)
    return pd.concat([df, sample], ignore_index=True).drop_duplicates("comment")


def metrics(y, probabilities, threshold):
    y = np.asarray(y, dtype=int)
    predicted = (np.asarray(probabilities) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {
        "macro_f1": float(f1_score(y, predicted, labels=[0, 1], average="macro", zero_division=0)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "false_positive_rate": float(fp / max(1, fp + tn)),
        "brier_score": float(brier_score_loss(y, probabilities)),
        "roc_auc": float(roc_auc_score(y, probabilities)) if len(np.unique(y)) == 2 else None,
        "average_precision": float(average_precision_score(y, probabilities)) if len(np.unique(y)) == 2 else None,
        "confusion": {"true_negative": int(tn), "false_positive": int(fp),
                      "false_negative": int(fn), "true_positive": int(tp)},
        "count": len(y),
    }


def pick_threshold(y, probabilities, policy="precision_first"):
    # Tune on validation only. Test labels never choose the operating threshold.
    if policy not in {"precision_first", "balanced"}:
        raise ValueError("Threshold policy must be precision_first or balanced.")
    candidates = []
    for threshold in np.linspace(0.05, 0.95, 91):
        predicted = np.asarray(probabilities) >= threshold
        f1 = f1_score(y, predicted, average="macro", zero_division=0)
        precision = precision_score(y, predicted, zero_division=0)
        candidates.append((policy == "balanced" or precision >= 0.75, f1, precision, float(threshold)))
    return max(candidates)[-1]


def fit_one(train, calibration, kind):
    pipeline = build_pipeline(kind)
    pipeline.fit(train.comment.tolist(), train.label.to_numpy())
    calibrated = CalibratedClassifierCV(FrozenEstimator(pipeline), method="sigmoid")
    calibrated.fit(calibration.comment.tolist(), calibration.label.to_numpy())
    return calibrated


def train_experiment(df, dataset_info, model_dir: Path, report_dir: Path, augment=True, progress=None):
    progress = progress or (lambda message: None)
    progress("Separating train, calibration, validation, and test groups")
    parts, split_info = partition(df)
    configs = [("word", "Word TF-IDF", False), ("word_char", "Word + character TF-IDF", False)]
    if augment:
        configs.append(("word_char", "Word + character TF-IDF · augmented", True))
    fitted, results = [], []
    with threadpool_limits(limits=2):
        for kind, name, use_augmentation in configs:
            progress(f"Training {name}")
            train = augment_training(parts["train"]) if use_augmentation else parts["train"]
            model = fit_one(train, parts["calibration"], kind)
            val_p = model.predict_proba(parts["validation"].comment.tolist())[:, 1]
            threshold = pick_threshold(parts["validation"].label, val_p)
            fitted.append(model)
            results.append({"name": name, "kind": kind, "augmented": use_augmentation,
                            "training_rows_with_variants": len(train), "threshold": threshold,
                            "validation": metrics(parts["validation"].label, val_p, threshold)})
        # Choose before inspecting test performance.
        best = max(range(len(results)), key=lambda i: (results[i]["validation"]["macro_f1"],
                                                      -results[i]["validation"]["false_positive_rate"]))
        progress("Evaluating held-out comments and synthetic spelling variants")
        for model, result in zip(fitted, results):
            test = parts["test"]
            p = model.predict_proba(test.comment.tolist())[:, 1]
            result["test"] = metrics(test.label, p, result["threshold"])
            result["robustness"] = {}
            for mode in ("repeat", "punctuation"):
                texts = test.comment.map(lambda text: spelling_variant(text, mode))
                changed = texts.ne(test.comment)
                if not changed.any():
                    result["robustness"][mode] = None
                    continue
                variant_p = model.predict_proba(texts[changed].tolist())[:, 1]
                original = p[changed.to_numpy()] >= result["threshold"]
                altered = variant_p >= result["threshold"]
                result["robustness"][mode] = {
                    "count": int(changed.sum()), "prediction_consistency": float(np.mean(original == altered)),
                    "accuracy_against_inherited_labels": float(np.mean(altered == test.loc[changed, "label"].to_numpy())),
                }
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = hashlib.sha256("\n".join(sorted(df.id + ":" + df.label.astype(str))).encode()).hexdigest()
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(), "dataset": dataset_info,
        "dataset_fingerprint": fingerprint, "split": split_info, "experiments": results,
        "selected_index": best, "selected_model": results[best]["name"],
        "selection_rule": "Highest validation macro-F1; test data never used for selection",
        "threshold_rule": "Prefer validation precision >= 0.75, then maximize validation macro-F1",
        "human_validated": False, "score_meaning": "Calibrated predicted offensive-language likelihood × 10; not abuse severity",
        "limitations": ["Published labels are unverified; test metrics measure agreement with those labels.",
                        "Calibration is fitted on source-domain data and may not transfer to new videos.",
                        "Synthetic variations inherit labels; robustness scores are a proxy, not human validation.",
                        "No personal-attack or gender-abuse category classifier is trained.",
                        "English-only and Tamil-script performance are not separately validated."],
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__},
    }
    bundle = {"model": fitted[best], "threshold": results[best]["threshold"], "report": report}
    temp_model = model_dir / "model.tmp.joblib"
    joblib.dump(bundle, temp_model, compress=3)
    os.replace(temp_model, model_dir / "model.joblib")
    temp_report = report_dir / "experiment.tmp.json"
    temp_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp_report, report_dir / "experiment.json")
    # Keep run history so before/after label-review experiments can be compared.
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    (report_dir / f"run_{run_id}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return bundle


def explain(bundle, texts, limit=4):
    # Linear contributions to the uncalibrated decision. They are associations, not reasons or spans.
    calibrated = bundle["model"].calibrated_classifiers_[0]
    base = calibrated.estimator
    pipeline = base.estimator if hasattr(base, "estimator") else base
    features = pipeline.named_steps["features"]
    classifier = pipeline.named_steps["classifier"]
    matrix = features.transform(texts)
    names = features.get_feature_names_out()
    result = []
    for row in matrix:
        contributions = row.data * classifier.coef_[0][row.indices]
        order = np.argsort(contributions)[::-1]
        evidence = []
        for i in order:
            if contributions[i] <= 0 or len(evidence) >= limit:
                break
            name = str(names[row.indices[i]])
            evidence.append({"feature": name.split("__", 1)[-1],
                             "type": "character" if name.startswith("char__") else "word",
                             "contribution": round(float(contributions[i]), 4)})
        result.append(evidence)
    return result


def predict(bundle, texts, include_evidence=True):
    if not texts:
        return []
    is_transformer = bundle.get("backend") == "transformer"
    if is_transformer:
        probabilities, truncated = bundle["model"].predict_details(texts)
    else:
        probabilities = bundle["model"].predict_proba(texts)[:, 1]
        truncated = [False] * len(texts)
    evidence = explain(bundle, texts) if include_evidence and not is_transformer else [[] for _ in texts]
    threshold = bundle["threshold"]
    return [{"comment": text, "probability": round(float(p), 5), "score": round(float(p) * 10, 2),
             "label": "toxic" if p >= threshold else "non_toxic",
             "uncertain": bool(cut or abs(float(p) - threshold) < 0.1),
             "truncated": bool(cut), "evidence": e}
            for text, p, e, cut in zip(texts, probabilities, evidence, truncated)]
