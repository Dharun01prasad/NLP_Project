# Mozhi — Tamil-English toxicity lab

A runnable NLP project for direct classification of Tamil-English YouTube comments. It includes a trained research model, a local FastAPI application, a dataset importer, training experiments, a Cleanlab review queue, and a Manifest V3 Chrome extension. Inference uses TF-IDF and Logistic Regression. No generative LLM, translation API, or API key is used.

For the English/Tamil/Thanglish reliability upgrade, see **[ROBUSTNESS.md](ROBUSTNESS.md)**
and **[the Colab training notebook](colab/Train_Mozhi.ipynb)**. This adds optional
multilingual transformer training and offline inference, a Tamil tokenizer fix, and
language/script diagnostics. The packaged TF-IDF checkpoint is unchanged; a full
multilingual model must be trained and evaluated before claiming improved accuracy.

## Start on Windows

1. Install **Python 3.12** from https://www.python.org/downloads/ if needed, including the Python launcher.
2. Extract this archive into a regular folder. Open that folder and double-click **`setup_windows.bat`**. The first installation needs internet access.
3. Double-click **`start_windows.bat`**. Leave its terminal running.
4. Open **http://127.0.0.1:8000** in Chrome.
5. In **Comment analysis**, select **Load examples**, then **Analyze comments**. You can also paste your own comments, one per line.

The packaged model is already trained. You do not need to download the training corpus to analyze comments. Its predictions are a research starting point and can be wrong.

On macOS/Linux, use Python 3.12 and run `bash start.sh`. Alternatively, use the commands below from the project root:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python run.py
```

If a local Python environment already exists but dependencies are missing, run the requirements installation again. All commands below assume that environment is activated.

## Install the Chrome extension

1. Start the Python backend as above.
2. Open `chrome://extensions` in Chrome.
3. Enable **Developer mode**.
4. Select **Load unpacked** and choose this project's **`extension`** folder, which contains `manifest.json`.
5. Open a YouTube video at a `/watch?v=...` URL. Scroll to the comments; expand replies if you want them included.
6. Open the Mozhi extension and select **Analyze loaded comments**.

The extension only analyzes comments currently present in the page, up to 500 and one million characters per request. Loaded replies are included. It does not claim to scrape every comment or provide a representative video-wide score. YouTube may change its markup; extraction selectors are in `extension/content.js`. Shorts and embedded players are outside the first version's scope. Closing the popup ends the visible session; rerun analysis when reopening it.

Permissions are limited to active-tab scripting and the fixed local backend at `127.0.0.1:8000`. No browsing history or account credentials are collected. Comment text is sent to your local backend when you select Analyze. Training data, explicit review decisions, and model artifacts are stored on your computer; analysis requests are not persisted by the application.

## Training and data quality

Open **Dataset studio → Load research dataset** to download the original DravidianCodeMix archive. The app checks its published MD5 checksum and reads the headerless, tab-separated `tamil_offensive_full_train/dev/test.csv` files. These split files are **text first, label second**. The archive's unsplit `tamil_offensive_full.csv` is **label first, text second** and is deliberately not used.

The raw corpus and its review comments are not redistributed in this project archive; the app downloads data from the original source. The packaged model and aggregate report were produced from that source. Load the corpus and run an audit to generate your review queue.

Automatic mapping is explicit:

| Original label | Binary label |
| --- | --- |
| `Not_offensive` | 0 — non-toxic |
| `Offensive_Untargetede` | 1 — toxic/offensive |
| `Offensive_Targeted_Insult_Individual` | 1 |
| `Offensive_Targeted_Insult_Group` | 1 |
| `Offensive_Targeted_Insult_Other` | 1 |
| `not-Tamil` | excluded |

This operational definition covers offensive language broadly. It does not mean all flagged comments are identity-based hate speech. Collapsing categories does not repair incorrect original annotations.

Alternatively, import a UTF-8 CSV/TSV with `comment,label` headers and labels 0/1. Custom column names and an explicit JSON label mapping can be configured in the interface. An optional `split` column accepts train/dev/test; provide all three splits or omit the column for all rows. Use a reasonably sized dataset: at least 60 distinct examples per class is a practical minimum for this implementation, and more is preferable. Sentiment labels such as positive/negative are rejected unless you supply your own explicit mapping; they should not be used as toxicity labels.

```csv
comment,label
"Indha video romba helpful, nandri",0
"An example explicitly labeled abusive by your annotation rules",1
```

That snippet illustrates the format, not a sufficient training dataset.

The importer removes empty/oversized text, exact normalized duplicates, and conflicting duplicate labels. Original test/dev copies take priority over training duplicates. Normalized groups collapse Latin repeated letters, punctuation, and spacing to reduce obvious variant leakage. This is a conservative heuristic; it does not detect all semantic duplicates. Grouping does not edit the text used by the classifier.

## Run experiments

Select **Experiments → Train & compare** after loading data. Three configurations are compared:

1. Word TF-IDF (unigrams/bigrams) + class-weighted Logistic Regression.
2. Word + character TF-IDF (3–5-character fragments) + Logistic Regression.
3. The combined model with deterministic spelling augmentation on a subset of training comments.

The train/calibration/validation/test partition is frozen when data is imported. For original data, official splits are retained and overlapping groups are removed from lower-priority partitions. With unsplit imports, the split is approximately 60/20/20 by group. Development data is split into calibration and validation subsets.

Only training rows fit the feature vocabulary and classifier. A sigmoid calibrator is fitted separately. Validation chooses a threshold, preferring precision >= 0.75 where possible, and then maximizing macro-F1. Validation macro-F1 selects the winning configuration. The selected model is chosen before test metrics are computed. This precision preference can reduce recall considerably; inspect both metrics rather than assuming it will catch all abuse. No final refit on test data is performed.

The report records macro-F1, precision, recall, false-positive rate, Brier score, ROC-AUC, average precision, confusion matrices, dataset fingerprint, split counts, and source. Synthetic repeated-letter and punctuation tests report prediction consistency. These variants inherit labels and have not been human-validated. Experiments use seed 42 and constrained numerical threads for repeatability. Reports from subsequent runs are retained in `runtime/reports/run_*.json` so before/after review results can be compared.

CLI equivalents:

```bash
python -m toxicity fetch
python -m toxicity train
python -m toxicity audit
# For the two non-augmented configurations only:
python -m toxicity train --no-augment
```

Run CLI jobs while the server is stopped, then restart it to load the resulting model. The web interface handles model refresh automatically.

## Review suspicious labels

1. Load the dataset, then select **Label review → Find suspicious labels**.
2. Cleanlab receives probabilities from three group-separated out-of-fold fits on **training rows only**. The vectorizer is also refitted inside every fold.
3. Review up to 100 ranked comments. Select **Keep** or **Change** based on your own Tamil understanding and agreed annotation rules.
4. Every explicit decision is recorded with the original label and time. Test/dev labels are not exposed for correction here.
5. Retrain after a review batch. The old model remains available, with a stale-data notice, until training succeeds.

These are suggested issues, not guaranteed annotation errors. Model disagreement alone does not justify changing a label. The system never automatically relabels your data. Loading a new corpus replaces the dataset and review queue; export anything you need first. Model-based auditing does not create a trustworthy test set: independently review an initial sample of about 200 comments before making real-world accuracy claims, and expand evaluation for the final report.

## Scores and explanations

Each comment gets `10 × calibrated probability` as a **predicted offensive-language likelihood score**. It is **not abuse severity**. Calibration reflects the published dataset and may not transfer to a new video's language, topic, or class balance.

The binary label uses the selected validation threshold. Predictions within 0.1 probability of this threshold are marked **Needs context**. This heuristic is not a statistical uncertainty interval and will miss confident errors. Feature chips show positive linear contributions from word/character features. They are learned associations, not verified toxic spans, causal explanations, or category labels.

The comment-section summary shows the number analyzed, percentage flagged, mean likelihood, and near-threshold count. It describes the sample actually analyzed. It does not measure general sentiment or the entire video's toxicity severity.

## What is and is not established

See **MODEL_CARD.md** for the measured baseline. It is not yet a robust final model. Important remaining work is independent Tamil-speaker evaluation, targeted training-label correction, performance across new videos and language varieties, and measured robustness to genuine slang and obfuscation. Standalone English performance is not established by training on code-mixed comments. Fine-grained abuse categories and severity prediction are not implemented.

The local API is available at `/docs`. It serves the web interface and extension using:

| Endpoint | Purpose |
| --- | --- |
| `GET /api/status` | Model, dataset, report, and active-job status |
| `POST /api/analyze` | Batch inference, 1–500 comments |
| `POST /api/datasets/research` | Fetch and validate the fixed source archive |
| `POST /api/datasets/import` | Import labeled CSV/TSV |
| `POST /api/train` | Start controlled training comparisons |
| `POST /api/audit` | Start training-label audit |
| `GET /api/jobs/{id}` | Poll job progress |
| `GET /api/reviews` | Get ranked training review candidates |
| `POST /api/reviews` | Save explicit label decisions |
| `GET /api/export/{report,dataset,audit}` | Download local artifacts |

Only one dataset/training/audit job runs at a time. The server binds to loopback and restricts cross-origin writes. This is a local academic tool, not a hosted multi-user service. Never load untrusted `.joblib` files: they can execute code. User imports accept text CSV/TSV only.

## Checks

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
node tests/extension.test.cjs
```

Tests cover parsing and label mapping, duplicate conflicts, group leakage, augmentation boundaries, training/calibration/inference, API validation, explicit review persistence, held-out protection, label-audit boundaries, and extension extraction/message behavior on fixtures. Browser checks cover the actual local interface at desktop and mobile widths. Live YouTube behavior still needs checking in your installed Chrome extension because YouTube can change its DOM.

## Project layout

```text
mozhi/
  run.py                  Local server entry point
  toxicity/               Data parsing, model training, auditing, API
  web/                    Local research interface
  extension/              Chrome Manifest V3 extension
  tests/                  Pipeline, API, and extension tests
  runtime/models/         Trained model artifact
  runtime/reports/        Experimental reports and review queue
  runtime/data/           Created when source data is downloaded/imported
```

## Sources

- DravidianCodeMix, Chakravarthi et al.: https://zenodo.org/records/4750858; paper https://doi.org/10.1007/s10579-022-09583-7. Cite the original authors when using the corpus. Source archive checksum: `7850be52919a387f5b36c7a09b05ad87` (MD5, published by Zenodo).
- Character TF-IDF/linear classification for Dravidian comments: https://aclanthology.org/2021.dravidianlangtech-1.37/ and https://aclanthology.org/2021.dravidianlangtech-1.5/.
- Cleanlab: https://docs.cleanlab.ai/; used for label-issue ranking with out-of-fold predictions.
- Chrome content scripts: https://developer.chrome.com/docs/extensions/develop/concepts/content-scripts.

See dependency licenses in their original packages. Source corpus annotations and their original classification scheme remain attributable to their authors.
