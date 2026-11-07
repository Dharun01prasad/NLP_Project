"""V2 wiring and data checks. These tests never fit a model."""
import zipfile

import pandas as pd
import numpy as np

from toxicity.data import binary_label
from toxicity.language import detect
from toxicity.prepare_v2 import archive_rows, hinglish_rows, romanized
from toxicity.train_transformer import source_label_weights


def test_new_official_labels_and_romanized_filter():
    assert binary_label("not-malayalam") is None
    assert binary_label("not-Kannada") is None
    assert binary_label("Offensive_Targeted_Insult_Individual") == 1
    assert romanized("nalla video anu")
    assert not romanized("നല്ല video")


def test_archive_parser_keeps_embedded_tabs(tmp_path):
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as output:
        for split in ("train", "dev", "test"):
            output.writestr(f"DravidianCodeMix/kannada_offensive_{split}.csv",
                            "chennagide\tvideo\tNot_offensive\t\nಕನ್ನಡ\tOffensive_Untargetede\n")
    frame = archive_rows(archive, "kannada")
    assert len(frame) == 3
    assert set(frame.comment) == {"chennagide video"}
    assert set(frame.language) == {"romanized_kannada"}


def test_hinglish_split_is_stable(tmp_path):
    path = tmp_path / "hin.csv"
    pd.DataFrame([{"comment": "kya haal hai dost", "label": 0},
                  {"comment": "KYA HAAL HAI DOST!", "label": 0}]).to_csv(path, index=False)
    frame = hinglish_rows(path)
    assert len(set(frame.split)) == 1
    assert set(frame.language) == {"romanized_hindi"}


def test_language_abstains_without_model_and_recognizes_native_script():
    assert detect("Hi how are you")["language"] == "unknown"
    assert detect("नमस्ते दोस्त")["language"] == "hindi_script"
    assert detect("ನಮಸ್ಕಾರ ಸ್ನೇಹಿತ")["language"] == "kannada_script"
    assert detect("മലയാളം നല്ലത്")["language"] == "malayalam_script"


def test_two_word_language_uses_model_but_ambiguous_text_abstains():
    class Model:
        classes_ = np.array(["english", "romanized_tamil"])

        def __init__(self, scores):
            self.scores = scores

        def predict_proba(self, comments):
            return np.array([self.scores])

    assert detect("sootha saathu", Model([0.23, 0.77]))["language"] == "romanized_tamil"
    assert detect("qwerty asdfgh", Model([0.52, 0.48]))["language"] == "unknown"
    assert detect("hello", Model([0.05, 0.95]))["language"] == "unknown"


def test_clear_english_overrides_noisy_corpus_language_probability():
    class Model:
        classes_ = np.array(["english", "romanized_tamil"])

        def predict_proba(self, comments):
            return np.array([[0.07, 0.93]])

    model = Model()
    assert detect("Who is waiting for Hardik Pandya from India", model)["language"] == "english"
    assert detect("Only here for Hardik Pandya", model)["language"] == "english"
    assert detect("Haaland is good person", model)["language"] == "english"
    assert detect("Sootha saathu", model)["language"] == "romanized_tamil"
    assert detect("nalla video is good", model)["language"] == "romanized_tamil"


def test_tempered_sampler_does_not_equalize_tiny_and_large_strata():
    frame = pd.DataFrame({"source": ["large"] * 32 + ["small"] * 2,
                          "label": [0] * 34})
    weights = source_label_weights(frame)
    assert weights[32] > weights[0]  # minority examples are still upweighted
    assert weights[:32].sum() > weights[32:].sum()  # no 50/50 replay of two rows
