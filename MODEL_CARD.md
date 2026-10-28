# Model card — Mozhi research baseline

This card describes the original packaged TF-IDF checkpoint. The optional
multilingual transformer produces its own `runtime/models/transformer/report.json`.
See [ROBUSTNESS.md](ROBUSTNESS.md) for the diagnosed failures and Colab workflow.

## Model and training

- Direct Tamil-English offensive-language classification; no translation or generative LLM.
- Selected model: **Word + character TF-IDF · augmented**.
- Word unigrams/bigrams + character 3–5-grams, class-weighted Logistic Regression, and separately fitted sigmoid calibration.
- Selected by validation macro-F1, before examining test results.
- Training date: 2026-10-25. Seed: 42.
- Source: https://zenodo.org/records/4750858.
- 43919 original rows; 41622 after mapping, deduplication, conflicts, and overlap checks.
- **Published labels have not been independently verified.**

## Measured results

| Configuration | Validation macro-F1 | Test macro-F1 | Toxic precision | Toxic recall | False-positive rate |
| --- | --- | --- | --- | --- | --- |
| Word TF-IDF | 71.71% | 69.70% | 77.45% | 37.42% | 3.55% |
| Word + character TF-IDF | 74.53% | 73.84% | 78.74% | 45.71% | 4.03% |
| Word + character TF-IDF · augmented | 75.08% | 74.08% | 78.47% | 46.38% | 4.15% |

These metrics measure agreement with the original, unverified labels. They are not human-validated real-world accuracy. The selected operating threshold emphasizes precision and misses substantial offensive content.

## Splits

| Partition | Comments | Toxic labels |
| --- | --- | --- |
| train | 33218 | 8203 |
| calibration | 2094 | 509 |
| validation | 2093 | 508 |
| test | 4217 | 1037 |

Selected probability threshold: **0.60**. Calibration and validation use separate halves of the development partition. Original test rows and variant groups are excluded from training. Thresholds and the selected configuration are chosen on validation, not on test data.

## Test confusion matrix

| | Predicted non-toxic | Predicted toxic |
| --- | --- | --- |
| Labeled non-toxic | 3048 | 132 |
| Labeled toxic | 556 | 481 |

## Intended use

Academic experiments, manual inspection, and demonstration of a local Chrome extension. Model output should support interpretation and evaluation. It is not suitable as the sole basis for automatic moderation, punishment, or claims about individuals or communities.

## Output meaning

- 0–10 score = 10 times calibrated predicted probability of the operational offensive-language label.
- It does not measure abuse severity, sentiment, or friendliness.
- Near-threshold flags identify a simple margin around the chosen decision threshold, not formal uncertainty.
- Character/word feature chips are statistical associations, not verified offensive spans.
- No fine-grained gender-targeted or personal-attack categories are produced.

## Unresolved limitations

- Published labels are unverified; test metrics measure agreement with those labels.
- Calibration is fitted on source-domain data and may not transfer to new videos.
- Synthetic variations inherit labels; robustness scores are a proxy, not human validation.
- No personal-attack or gender-abuse category classifier is trained.
- English-only and Tamil-script performance are not separately validated.
- Grouping repeated letters, punctuation and spaces only catches obvious variations; semantic duplicate leakage may remain.
- A small initial manually verified test set is needed before claiming real-world quality.
- The Chrome extension analyzes loaded comments only; results do not represent all comments on a video.

## Reproduce

Use Python 3.12 and the pinned requirements. In the app, load the original dataset and run Train & compare, or run `python -m toxicity fetch` followed by `python -m toxicity train`. Stop the web server while using the CLI. The dataset fingerprint and machine-readable metrics are in `runtime/reports/experiment.json`.
