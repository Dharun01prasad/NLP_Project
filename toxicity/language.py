"""Conservative language estimates for native scripts and romanized comments."""
from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline
from wordfreq import zipf_frequency

NAMES = {"english": "English", "romanized_tamil": "Romanized Tamil", "romanized_hindi": "Romanized Hindi",
         "romanized_malayalam": "Romanized Malayalam", "romanized_kannada": "Romanized Kannada",
         "tamil_script": "Tamil", "hindi_script": "Hindi", "malayalam_script": "Malayalam",
         "kannada_script": "Kannada", "unknown": "Language unclear"}
SCRIPTS = (("hindi_script", r"[\u0900-\u097f]"), ("tamil_script", r"[\u0b80-\u0bff]"),
           ("malayalam_script", r"[\u0d00-\u0d7f]"), ("kannada_script", r"[\u0c80-\u0cff]"))
LATIN = re.compile(r"[A-Za-z]")


def clear_english(words: list[str]) -> bool:
    """Independent evidence for ordinary English despite noisy corpus tags.

    Proper names are often absent from word-frequency lists, so require both
    common English words and an English function word. This does not affect
    the toxicity classifier or force a label for ambiguous short text.
    """
    if len(words) < 2:
        return False
    tokens = [word.lower() for word in words]
    # A lowercase out-of-vocabulary word is often a romanized Indic term.
    # Keep those mixed comments with the trained classifier. Capitalized
    # person names may also be absent from the English frequency list.
    if any(word.islower() and len(word) >= 3 and zipf_frequency(word, "en") < 3.5
           for word in words):
        return False
    frequent = sum(zipf_frequency(word, "en") >= 5 for word in tokens)
    return frequent >= 2 and frequent / len(tokens) >= 0.4 and any(
        word in ENGLISH_STOP_WORDS for word in tokens)


def detect(text: str, model=None) -> dict:
    counts = [(tag, len(re.findall(pattern, text))) for tag, pattern in SCRIPTS]
    tag, count = max(counts, key=lambda pair: pair[1])
    if count >= 2:
        return {"language": tag, "language_label": NAMES[tag], "language_confidence": None}
    words = re.findall(r"[A-Za-z]+", text)
    # Short comments are common on YouTube. Two words carry useful evidence;
    # single Latin words remain ambiguous across romanized languages.
    if model is None or len(words) < 2 or sum(map(len, words)) < 6:
        return {"language": "unknown", "language_label": NAMES["unknown"], "language_confidence": None}
    if clear_english(words):
        return {"language": "english", "language_label": NAMES["english"],
                "language_confidence": None}
    values = model.predict_proba([text])[0]
    order = values.argsort()[::-1]
    best, second = float(values[order[0]]), float(values[order[1]])
    label = str(model.classes_[order[0]]) if best >= 0.55 and best - second >= 0.15 else "unknown"
    return {"language": label, "language_label": NAMES.get(label, NAMES["unknown"]),
            "language_confidence": round(best, 4)}


def load(path: Path):
    return joblib.load(path) if path.exists() else None


def train(data_path: Path, output: Path):
    """User-run companion training. Corpus tags are weak labels, not human verification."""
    if output.exists():
        raise ValueError(f"Output exists: {output}")
    data = pd.read_csv(data_path, keep_default_na=False)
    tags = set(NAMES) - {"unknown", "tamil_script", "hindi_script", "malayalam_script", "kannada_script"}
    data = data[data.language.isin(tags) & data.comment.map(lambda text: bool(LATIN.search(text)))].copy()
    train_rows, dev, test = (data[data.split.eq(split)] for split in ("train", "dev", "test"))
    if min(len(train_rows), len(dev), len(test)) < 100:
        raise ValueError("Too few language tagged rows in a split")
    model = Pipeline([("features", TfidfVectorizer(analyzer="char", ngram_range=(2, 5),
                       min_df=3, max_features=120000, sublinear_tf=True)),
                      ("classifier", LogisticRegression(max_iter=500, class_weight="balanced"))])
    model.fit(train_rows.comment, train_rows.language)
    # Rejection is evaluated separately, because corpus labels include mixed-language texts.
    def measure(frame):
        predictions = model.predict(frame.comment)
        accepted = [detect(text, model)["language"] for text in frame.comment]
        return {"rows": len(frame), "raw_accuracy": float(accuracy_score(frame.language, predictions)),
                "raw_macro_f1": float(f1_score(frame.language, predictions, average="macro", zero_division=0)),
                "accepted_fraction": sum(value != "unknown" for value in accepted) / len(accepted),
                "report": classification_report(frame.language, predictions, output_dict=True, zero_division=0)}
    report = {"dev": measure(dev), "test": measure(test), "warning": "Language labels are corpus proxies, not manually verified."}
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output)
    output.with_suffix(".report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
