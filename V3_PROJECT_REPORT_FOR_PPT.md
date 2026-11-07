# Mozhi — Project Report and PPT Source (Latest V3 Candidate)

> **For Claude / slide author:** Build a 12–16 slide presentation from this document. Put the short “Slide content” points on slides and use the “Speaker context” paragraphs as presenter notes. Use a clean architecture diagram, one dataset table, and two results charts. Label every metric **held-out source-dataset performance**, not real-world YouTube accuracy. The latest completed model is the **V3 candidate** in `runtime/v3_candidate`; the app serves whichever runtime is selected at startup. Do not call the binary output a dedicated hate-speech classifier or a severity score.

## 1. Project overview

**Title:** Mozhi: Multilingual offensive-language analysis for code-mixed comments.

Mozhi is a local Python web application and Chrome extension that analyzes English and Indian-language social-media comments. The latest model was trained on English, Tamil/Tanglish, Romanized Hindi (Hinglish), Romanized Malayalam, and Romanized Kannada examples. It predicts a binary **offensive/toxic versus non-offensive** label and a probability. A separate language identifier supplies an estimated language label. The extension shows an emoji, percentage, and language beside comments currently rendered on a YouTube watch page. The web app accepts pasted comment batches and shows individual results and exports.

The project is a **research and review aid**. It does not automatically remove comments. Its `toxic` category merges several source definitions of offensive or toxic language and does **not** isolate identity-directed hate speech.

## 2. Problem statement and social relevance

### Slide content

- Online comment sections mix English, native Indian scripts, romanized languages, slang, emojis, and creative spellings.
- Offensive comments can harm participation and create a heavy manual-review burden.
- Language-specific or English-only models may overlook local expressions; blunt keyword rules can flag harmless discussion.
- Objective: give reviewers a local, explainable workflow for **prioritizing** comments while showing uncertainty and limitations.

### Speaker context

On Indian social platforms, one comment may contain English words and Tamil grammar written in Latin letters; another may be Hindi written as “tum kaise ho.” Similar spelling patterns can occur across languages. The problem is not just finding profanity. Quoted abuse, sarcasm, disagreement, and context change interpretation. A greeting such as “Hi how are you” should stay non-offensive; a direct attack such as “fuck off” should be flagged. “He told me to fuck off and I reported him” quotes abuse but is not itself a direct attack. Mozhi's purpose is to surface useful model estimates for review, not to make a final moderation decision.

### Project objectives

1. Detect broadly offensive language in short, code-mixed comments.
2. Compare a transparent traditional-ML baseline with an Indic-language transformer.
3. Show a probability-derived score, predicted label, context warning, and estimated language.
4. Evaluate on held-out data and inspect errors by source/language, rather than relying only on overall accuracy.
5. Keep inference local and make model/version selection explicit.

## 3. Scope and examples

| Example comment | Intended interpretation | Why it matters |
| --- | --- | --- |
| “Hi how are you” | Non-offensive | Ordinary greeting must not receive an alarming flag. |
| “Who is waiting for Hardik Pandya from India” | Non-offensive | Real sports discussion tests domain transfer. |
| “fuck off” | Offensive | Direct interpersonal abuse. |
| “He told me to fuck off and I reported him” | Non-offensive in this diagnostic policy | A quoted insult is different from attacking someone. |
| “why the hell did you come here” | Context-dependent | Profanity alone does not settle the label. |

These are **diagnostic examples**, not a statistically representative benchmark. The code keeps the diagnostic set out of future training splits. Some examples may still be misclassified; show errors honestly during the presentation.

## 4. System architecture and user workflow

### Slide content / diagram

```text
YouTube rendered comments ──> Chrome MV3 content script ──> local FastAPI API
Pasted web comments ────────> HTML/CSS/JavaScript web UI ──> POST /api/analyze
                                                          │
                                             selected toxicity checkpoint
                                             + separate language identifier
                                                          │
                              probability, 0–10 score, flag, uncertainty, language
                                                          │
                                  inline extension badge / web result / CSV export
```

### Speaker context

The extension is OFF by default. When enabled on a YouTube `/watch` page, it observes comments and replies currently present in the DOM, sends batches to the local backend at `127.0.0.1:8000`, and adds badges. It does **not** scrape every comment on the video. The web page permits up to 500 pasted comments per analysis. The API validates input and returns per-comment predictions. `score = 10 × predicted offensive-language probability`; the emoji communicates score bands, while the saved threshold produces the “Flagged” label. “Needs context” appears near the threshold or when text is truncated. A repeated input with a fixed checkpoint normally receives the same prediction; phrases are not hardcoded to toxicity labels.

### Tech stack

| Layer | Technology |
| --- | --- |
| Web frontend | HTML, CSS, vanilla JavaScript |
| Browser extension | Chrome Manifest V3, content script, popup, service worker, `chrome.storage.local` |
| Local backend | Python 3.12, FastAPI, Pydantic, Uvicorn |
| Baseline ML | scikit-learn, TF-IDF, Logistic Regression, sigmoid calibration |
| Transformer ML | PyTorch, Hugging Face Transformers, fine-tuned MuRIL, safetensors |
| Language estimate | Character-ngram TF-IDF + Logistic Regression, native-script detection, English word-frequency evidence |
| Data/storage | pandas/NumPy; local CSV, JSON, joblib, safetensors files under `runtime/` |
| Database | None |

No translation service, generative LLM, or cloud inference API is required for local use. GPU training can occur in a permitted environment; local transformer inference still requires PyTorch to load.

## 5. Datasets used in the latest model

### Slide content

| Source in V3 training CSV | Usable rows after filtering and de-duplication | Role and label mapping |
| --- | ---: | --- |
| DravidianCodeMix Tamil-English | 41,629 | Tamil/Tanglish; `Not_offensive` → 0, offensive categories → 1; unrelated-language rows removed. |
| Davidson English tweets | 24,550 | English; original hate and offensive classes → 1, neither → 0. |
| DravidianCodeMix Malayalam-English | 7,506 | Latin-script/romanized portion; same offensive mapping. |
| DravidianCodeMix Kannada-English | 3,629 | Latin-script/romanized portion; same offensive mapping. |
| TextDetox Hinglish (`hin`) | 2,859 | Latin-script usable portion; binary `toxic` 0/1. |
| **Total** | **80,173** | **46,507 non-toxic / 33,666 toxic** in the prepared V2/V3 CSV. |

### Speaker context

The DravidianCodeMix archive is a published, manually annotated code-mixed social-media dataset. The project verifies the archive checksum before preparation. The Davidson dataset contains three English tweet categories; the project explicitly merges hate and offensive into its broad positive class. The TextDetox Hinglish subset provides romanized Hindi binary labels, but its dataset card does not identify the original source of that subset. These corpora have **different annotation rules** and domains. Corpus-level language tags are only proxies: some English sentences appear inside a Tamil-English dataset. No independent human relabeling of the full merged dataset has been completed. The table reports **usable project rows**, not the total sizes advertised by the source publishers.

## 6. Data preparation and split policy

### Slide content

```text
Download/verify → parse source labels → retain intended scripts → normalize IDs
→ remove empty/conflicting duplicates → group spelling/punctuation variants
→ preserve source train/dev/test assignments → remove diagnostic overlap
→ train / calibration / validation / held-out test
```

### Speaker context

The V3 model uses the same prepared dataset fingerprint as V2, allowing a controlled V2-versus-V3 comparison. Source-provided splits are retained where available. Hinglish rows receive deterministic group-based splits. Normalized duplicate and spelling-variant groups are prevented from crossing into an easier training/evaluation overlap. The development partition is divided into calibration and validation. The saved counts are **58,294 train; 5,433 calibration; 5,434 validation; 11,012 test**. The test contains 5,537 source-labeled positives and 5,475 negatives. The small hand-authored diagnostic examples are excluded before training. Test labels are used only for final measurement, not threshold selection.

## 7. Baseline model and why the transformer was added

### Slide content

- Baseline: word unigrams/bigrams + character 3–5-gram TF-IDF → class-weighted Logistic Regression → sigmoid probability calibration.
- Variants compared: word-only; word + character; word + character with training-only spelling augmentation.
- Baseline selected on development data, then measured on its held-out Tamil research split.
- Motivation for MuRIL: subword representations and Indic/transliteration pretraining can better handle multilingual spelling and code mixing.

### Speaker context

The baseline is fast, runs on CPU, and offers inspectable feature associations. It also exposed a weakness: a narrow dataset can give similar scores to ordinary English and a direct insult. The saved baseline test macro F1 was **74.08%**, offensive precision **78.47%**, recall **46.38%**, with **4.15%** false-positive rate on its **4,217-row Tamil research test**. Do **not** compare this directly with the V3 mixed-source headline metric: they use different evaluation populations. MuRIL was selected because its published pretraining covers Indian languages and transliterated counterparts. The project fine-tunes it for this binary label; it does not use a generative LLM.

## 8. Latest completed transformer: V3 candidate

### Slide content

- Base checkpoint: `google/muril-base-cased`; binary sequence-classification head.
- Fine-tuned for **3 epochs on CUDA**; batch size 8 with gradient accumulation 4 (effective batch 32), learning rate `2e-5`, maximum 128 tokens.
- V3 change: inverse-square-root source/class sampling, reducing repeated exposure to tiny groups.
- Model chosen by validation macro F1 across sources; temperature calibrated on a separate calibration partition.
- Validation chooses a precision-focused flag threshold; exported threshold **0.68** and temperature **1.478**.
- Export: tokenizer, `model.safetensors`, `report.json`, and `mozhi.json`; V3 stays isolated under `runtime/v3_candidate` until explicitly selected.

### Speaker context

V2 sampled each source-and-label group equally. That gave tiny groups, such as 240 toxic Malayalam training comments, the same batch share as much larger groups. V3 tempers the weights with an inverse square root while still helping minority groups. The precision-focused rule selects a threshold from validation data; the test set is evaluated after checkpoint, calibration, and threshold decisions. A 0.68 threshold means scores of about **6.8/10 or above** are flagged, but the score represents estimated likelihood, **not harm severity**. Only the first 128 model tokens affect the prediction; longer inputs are marked truncated.

## 9. Evaluation metrics and what they mean

| Metric | Meaning | Why report it |
| --- | --- | --- |
| Precision | TP / (TP + FP) | Of flagged comments, how many match the positive source label? |
| Recall | TP / (TP + FN) | How many source-labeled offensive comments were found? |
| False-positive rate | FP / (FP + TN) | How often non-offensive comments are incorrectly flagged. |
| Macro F1 | Average of the two class F1 scores | Gives both classes a voice when class balance differs. |
| ROC-AUC | Ranking across thresholds | Measures separation but not the deployed threshold's errors. |
| Brier score | Mean squared probability error | Checks probability calibration on source data. Lower is better. |

**Evaluation discipline:** model selection on validation; temperature on calibration; final numbers on untouched test; source/language slices; a separate small diagnostic set for known edge cases. The 44-case diagnostic set is hand-authored and is **not** an independent population estimate.

## 10. Main results: V2 versus V3 on the same held-out 11,012 rows

| Held-out metric | V2 candidate | Latest V3 candidate |
| --- | ---: | ---: |
| Macro F1 | 88.01% | **89.27%** |
| Offensive precision | 90.20% | **92.19%** |
| Offensive recall | 85.44% | **85.95%** |
| False-positive rate | 9.39% | **7.36%** |
| ROC-AUC | 0.9408 | **0.9592** |
| Brier score | 0.0964 | **0.0819** |

**V3 confusion matrix:** TN = 5,072; FP = 403; FN = 778; TP = 4,759. Thus 403 source-labeled non-offensive examples were flagged and 778 source-labeled offensive examples were missed. V2 and V3 share the same dataset fingerprint and test partition, so this is the comparison to chart. The V3 precision-focused threshold and changed sampling both contribute to the changed operating point; this report does not isolate their individual effects.

## 11. Slice results: the headline score hides weaknesses

| V3 test slice (corpus proxy) | Test rows | Macro F1 | Offensive recall | False-positive rate |
| --- | ---: | ---: | ---: | ---: |
| English / Davidson | 4,910 | 92.55% | 95.99% | 6.20% |
| Romanized Tamil / Tamil corpus | 3,410 | 76.70% | 57.19% | 7.36% |
| Tamil script / Tamil corpus | 809 | 72.16% | 47.65% | 7.20% |
| Romanized Hindi / Hinglish corpus | 432 | 70.74% | 69.04% | **26.94%** |
| Romanized Malayalam / Malayalam corpus | 1,071 | 71.95% | 52.54% | 4.05% |
| Romanized Kannada / Kannada corpus | 380 | 73.79% | 54.46% | 9.70% |

**Reading the table:** the English source strongly lifts the overall result. Hinglish false positives remain high. Malayalam has only **59 source-labeled positives** in the test slice, so its recall and precision are unstable estimates. The Romanized Tamil and Kannada slices still miss many positive examples. These “language” slices are based on **source metadata**, not separately verified language ID for each comment. V3 improved overall macro F1 and false-positive rate versus V2, but the Kannada slice worsened slightly and English false positives increased. Do not claim every language improved.

## 12. Observations and challenges

### What worked

- A local extension can put model estimates beside visible YouTube comments without sending text to an external prediction API.
- The latest V3 test macro F1 and overall false-positive rate improved over V2 on the same source-data split.
- Four benign sports comments from actual usage were added to the small development diagnostic set and passed that diagnostic run.
- Source-specific reporting exposed failures that a single aggregate number would hide.

### What remains difficult

- **Source mismatch:** English offensive tweets, Dravidian YouTube comments, and TextDetox Hinglish use different conventions and domains.
- **Label policy:** binary offensive-language mapping is broader than identity-based hate speech; annoyance, profanity, and abuse are not interchangeable.
- **Context:** quoted insults, counterspeech, irony, replies, and missing conversation history confuse a single-comment classifier.
- **Romanization:** the same word can have many spellings; language metadata can be wrong for code-mixed sentences.
- **Class imbalance:** tiny offensive subsets can be oversampled or give unstable evaluation numbers.
- **Live deployment gap:** held-out source-data performance is not the same as independent performance on new YouTube channels.

### Concrete V3 diagnostic failures

The latest 44-case diagnostic check scored 43 cases and missed **8**. It incorrectly flagged “Please do not insult other people” and “He told me to fuck off and I reported him”; it missed obfuscated direct abuse such as “f.u.c.k off” and “fuuuck you.” Tamil and Thanglish quoted abuse also caused errors. Use these examples as a candid error-analysis slide, **not** as a measured error rate for all social media.

## 13. Language display: supporting feature, not the main toxicity result

The language label is a **separate** estimate. Unicode ranges identify clear native scripts. For Latin text, a character-ngram TF-IDF/Logistic Regression model predicts English or a romanized language; general English word-frequency evidence corrects clear English sentences that source-level corpus tags mislabeled. Short or mixed text may show “Language unclear.” The language model's saved raw test accuracy is about **98% against corpus proxy tags**, but that is **not** a trustworthy claim of 98% real-world language identification: the tags are noisy, and the live hybrid detector differs from the raw model report. Do not use language identification to change the toxicity score.

## 14. Responsible use, next steps, and conclusion

**Responsible-use statement:** Scores are learned predictions. A flag is a request for human attention, not proof of abuse; no automatic comment removal occurs. The local backend limits exposure to an external service, but users should still handle exported comments appropriately. Identity-term mentions, negative criticism, and reported abuse should not be presumed toxic.

**Next work:** create a human-labeled, consent-appropriate test set of new comments by language and topic; audit Hinglish false positives and Tamil/Kannada false negatives; add reviewed ordinary conversation and quoted-abuse examples to **training only**; keep a frozen independent test set; compare candidate checkpoints before activation; evaluate thresholds for the desired precision/recall trade-off; improve mixed-language ID separately.

**Closing line:** “Mozhi demonstrates an end-to-end multilingual comment-analysis workflow. The latest candidate improves on its previous model under a controlled held-out comparison, while the slice results and diagnostic failures show why human review and better evaluation data remain essential.”

## 15. References and evidence for slide footnotes

1. Chakravarthi et al., *DravidianCodeMix: Sentiment Analysis and Offensive Language Identification Dataset for Dravidian Languages in Code-Mixed Text*. [Dataset and DOI](https://zenodo.org/records/4750858); [paper](https://arxiv.org/abs/2106.09460).
2. Davidson et al., *Automated Hate Speech Detection and the Problem of Offensive Language* (ICWSM 2017). [Paper](https://ojs.aaai.org/index.php/ICWSM/article/download/14955/14805); [authors' dataset repository](https://github.com/t-davidson/hate-speech-and-offensive-language).
3. TextDetox, [Multilingual Toxicity Dataset](https://huggingface.co/datasets/textdetox/multilingual_toxicity_dataset), `hin` Hinglish split. State the provenance limitation noted above.
4. Khanuja et al., *MuRIL: Multilingual Representations for Indian Languages*. [Paper](https://arxiv.org/abs/2103.10730); [Google model card](https://huggingface.co/google/muril-base-cased).
5. [wordfreq package documentation](https://pypi.org/project/wordfreq/) for the supporting English frequency feature.
6. **Project's primary evidence:** `runtime/v3_candidate/models/transformer/report.json`, `runtime/v3_candidate/models/transformer/mozhi.json`, `runtime/v3_candidate/reports/diagnostics.json`, `runtime/v2_candidate/models/transformer/report.json`, `runtime/data/v2/training.info.json`, and `evaluation/smoke_cases.json`. These local reports, not the external papers, support the project's numerical results.

## 16. Suggested final deck layout for Claude

1. Title and one-line value proposition.
2. Social problem and multilingual examples.
3. Objective, exact binary task, and responsible-use boundary.
4. End-to-end architecture diagram.
5. Dataset sources and label mapping table.
6. Data cleaning, deduplication, and split diagram.
7. Traditional-ML baseline and why it was insufficient.
8. V3 MuRIL training/calibration/threshold pipeline.
9. Web app and YouTube extension workflow with screenshots supplied by the user.
10. Evaluation design and metric definitions.
11. V2 vs V3 comparison chart on the same test set.
12. Per-language/source F1 and false-positive chart.
13. Real diagnostic successes and failures.
14. Challenges, ethics, and next steps.
15. References and closing.

Use **V3 candidate** consistently. Show the date of the V3 saved report as **7 November 2026 (UTC)**. Preserve the caveats beside the numbers. Do not present the V2/V3 source test as independent live YouTube accuracy, and do not compare baseline and V3 headline F1 as a controlled experiment.
