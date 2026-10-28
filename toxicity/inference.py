"""Load a baseline or an explicitly exported, locally stored transformer."""
from __future__ import annotations

import json
import unicodedata
from pathlib import Path

import joblib
import numpy as np
from scipy.special import expit


def transformer_text(text):
    # Preserve case, negation, Tamil marks and punctuation for pretrained tokenizers.
    return unicodedata.normalize("NFKC", str(text)).strip()


class TransformerPredictor:
    def __init__(self, directory, temperature=1.0, max_length=128, batch_size=8):
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise ValueError("Install requirements-transformer.txt to use the exported transformer.") from exc
        self.torch = torch
        torch.set_num_threads(2)
        self.tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            directory, local_files_only=True, trust_remote_code=False, use_safetensors=True)
        if self.model.config.num_labels != 2:
            raise ValueError("Mozhi requires a fine-tuned two-class checkpoint, not a base language model.")
        self.model.eval()
        self.temperature, self.max_length, self.batch_size = temperature, max_length, batch_size

    def predict_details(self, texts):
        scores, truncated = [], []
        for offset in range(0, len(texts), self.batch_size):
            batch = [transformer_text(t) for t in texts[offset:offset + self.batch_size]]
            lengths = self.tokenizer(batch, truncation=False, padding=False, return_length=True)["length"]
            truncated.extend(length > self.max_length for length in lengths)
            encoded = self.tokenizer(batch, truncation=True, max_length=self.max_length,
                                     padding=True, return_tensors="pt")
            with self.torch.inference_mode():
                logits = self.model(**encoded).logits.cpu().numpy()
            scores.extend(expit((logits[:, 1] - logits[:, 0]) / self.temperature))
        return np.asarray(scores), truncated

    def predict_proba(self, texts):
        probabilities, _ = self.predict_details(texts)
        return np.column_stack([1 - probabilities, probabilities])


def load_bundle(model_dir, backend="auto"):
    model_dir = Path(model_dir)
    if backend not in {"auto", "baseline", "transformer"}:
        raise ValueError("MOZHI_MODEL_BACKEND must be auto, baseline or transformer.")
    directory = model_dir / "transformer"
    manifest = directory / "mozhi.json"
    if backend == "transformer" or (backend == "auto" and manifest.exists()):
        if not manifest.exists():
            raise ValueError(f"No exported transformer found at {manifest}")
        meta = json.loads(manifest.read_text(encoding="utf-8"))
        if meta.get("schema_version") != 1 or meta.get("labels") != {"0": "non_toxic", "1": "toxic"}:
            raise ValueError("Unsupported transformer export or incompatible label order.")
        temperature, threshold = float(meta["temperature"]), float(meta["threshold"])
        max_length = int(meta["max_length"])
        if not (0 < temperature <= 100 and 0 < threshold < 1 and 8 <= max_length <= 512):
            raise ValueError("Invalid transformer calibration or token limit.")
        report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
        return {"backend": "transformer", "threshold": threshold, "report": report,
                "model": TransformerPredictor(directory, temperature, max_length)}
    path = model_dir / "model.joblib"
    if not path.exists():
        return None
    bundle = joblib.load(path)
    bundle["backend"] = "baseline"
    return bundle
