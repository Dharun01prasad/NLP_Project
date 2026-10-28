# Making Mozhi reliable across English, Tamil and Thanglish

## What was wrong

The packaged model is a TF-IDF + logistic-regression research baseline, not a
general-purpose multilingual hate-speech model. Directly running its saved
checkpoint reproduced these results:

| Comment | Offensive-language probability | Flagged at 0.60? |
| --- | ---: | --- |
| Hi how are you | 0.42272 | No |
| why the hell did you come here | 0.11825 | No |
| fuck you | 0.49244 | No |
| fuck off | 0.52822 | No |
| You are an idiot | 0.57342 | No |

The model assigns positive weights to `how`, `are` and `are you`. These are learned
correlations in a narrow corpus, not evidence that those words are abusive. The
original 33,218-row training split contained only 17 comments with `fuck`, three
with `idiot`, and none with `hell`. The loader excludes `not-Tamil` rows because
those rows have no usable offensive/non-offensive annotation; they must not be
silently relabeled as safe or toxic.

The original held-out report records 78.47% offensive precision but only **46.38%
recall**: 556 of 1,037 offensive examples are missed. Lowering the threshold changes
which errors occur; it does not correct the scores or teach sentence context.

There was also a Unicode bug: Python regex `\w` excludes Tamil combining marks.
The original word tokenizer split `வணக்கம் நன்றி` into `வணக`, `கம`, `நன`, `ற`.
New TF-IDF training now uses a Unicode-aware tokenizer. The grouping logic also
preserves marks so `கல்` and `கால்` are not merged. The existing saved model remains
unchanged and requires retraining to benefit from this fix.

## Define what the output means

This upgrade continues to detect **offensive/abusive language**, including direct
personal insults. `fuck off` is abusive; it is not automatically identity-based hate.
`Hi how are you` should be non-offensive. Mild profanity such as `why the hell...`
depends on your annotation policy and is recorded as an unscored context case.

If you need separate labels for profanity, personal attack, threat and identity-based
hate, collect those labels and train a multiclass or multilabel head. The existing
binary data cannot establish those categories. Sentiment is a different task.

## Ready-to-run Colab path

Open `colab/Train_Mozhi.ipynb` in Google Colab, select a GPU runtime, and execute the
cells. When prompted, upload `colab/mozhi_colab.zip`. This upload contains the
prepared corpus and code. The notebook downloads the base model, fine-tunes it,
evaluates it, and downloads `mozhi_transformer.zip`.

The default encoder is **google/muril-base-cased**, whose pretraining includes
transliterated Indian languages. You can compare **FacebookAI/xlm-roberta-base**
using a new output directory. These are candidate architectures, not a guarantee of
accuracy. The base models are not already-trained hate classifiers.

The training script:

1. Re-cleans the input and keeps normalized comment groups out of multiple splits.
2. Uses training-only sampling to balance each source/class combination. This keeps
   the heavily offensive English tweet source from overwhelming the other data.
3. Selects the epoch by mean validation macro-F1 across sources.
4. Fits temperature calibration on a separate calibration partition.
5. Chooses a threshold on validation; the default maximizes binary macro-F1 without
   the original precision-first constraint. `--threshold-policy precision_first`
   is available for comparison.
6. Evaluates the held-out test partition after all those choices. Reports include
   source, supplied language, and script slices, recall, false-positive rate, Brier
   score and confusion matrices.
7. Exports model weights in safetensors format, tokenizer, calibration, threshold
   and report. The activation manifest is written only when export completes.

The tests exercise a tiny real transformer through training, checkpoint selection,
export, reload, prediction and the HTTP API. **The full MuRIL training run has not
been performed as part of this change. No improved accuracy is claimed yet.**

## Run the exported model on Windows

Stop the existing backend. Extract `mozhi_transformer.zip` into
`E:\Coding\NLP_Project\mozhi\runtime\models`, so these files exist:

```text
runtime/models/transformer/mozhi.json
runtime/models/transformer/report.json
runtime/models/transformer/config.json
runtime/models/transformer/model.safetensors
runtime/models/transformer/...tokenizer files...
```

From PowerShell:

```powershell
cd "E:\Coding\NLP_Project\mozhi"
.\.venv\Scripts\python.exe -m pip install -r requirements-transformer.txt
.\.venv\Scripts\python.exe run.py
```

The UI will say **Multilingual transformer ready**. Inference uses local CPU batches
and does not download weights or send comments to any API. The original
`runtime/models/model.joblib` remains available. To compare it:

```powershell
$env:MOZHI_MODEL_BACKEND='baseline'
.\.venv\Scripts\python.exe run.py
```

Use `$env:MOZHI_MODEL_BACKEND='auto'` before restarting to restore transformer
detection. The Experiments web button trains TF-IDF; use the notebook to retrain the
transformer. With a transformer active, the button's API rejects baseline retraining
to avoid an unexpected model switch.

Inputs beyond the configured token limit (128 by default) are explicitly marked
truncated and needing review. Transformer outputs have no invented linear feature
explanations. Scores still mean model likelihood, not severity or proof of safety.

## Data preparation and what still needs improvement

The preparation command reads the local official Tamil offensive split files and
`../DravidianCodeMix/labeled_data.csv`. The latter has Davidson class IDs:
**0 = hate, 1 = offensive, 2 = neither**. For this broad binary task, map 0/1 to toxic
and 2 to non-toxic. Reversing these IDs is a serious label bug.

The prepared corpus has 66,180 rows after checks: 41,630 DravidianCodeMix rows and
24,550 Davidson rows. Davidson contributes 20,439 offensive and only 4,111
non-offensive rows. It is a biased sample of tweets, not representative ordinary
English conversation. The preparation metadata records file hashes, exclusions and
source counts. Original Tamil official holdouts are respected, with lower-priority
overlapping groups removed. Davidson gets a seeded group split.

`TamHateX_V4_final.csv` and the sentiment CSVs are not blindly merged: their labels
need a separate policy review. Automatically translating or transliterating data
can introduce label changes; manually review those transformations first.

To rebuild after adding reviewed examples (use a new output filename if preserving
an older corpus), run:

```powershell
.\.venv\Scripts\python.exe -m toxicity.prepare --extra reviewed_comments.csv --output training_data_v2.csv
```

The extra CSV uses `comment,label,split,language,source`, with explicit
`train/dev/test` assignments and 0/1 labels. Use language values such as `english`,
`tamil`, `thanglish` for reviewed examples. To build the default upload bundle, use
the default `training_data.csv` output, then run `python tools/build_colab.py`.

Add ordinary greetings, questions, disagreements, negative reviews, benign identity
mentions, quotes, counterspeech, sarcasm and spelling variants in **both classes**
and all three language forms. Have fluent Tamil/Thanglish reviewers resolve
ambiguous examples using a written labeling policy. Review training labels through
the existing audit flow, but keep an independently reviewed evaluation set separate.

Do not infer language solely from script: English and Thanglish are both Latin.
DravidianCodeMix receives a corpus-level language tag; its English and romanized
Tamil rows are not falsely assigned to distinct languages. Script metrics expose
some gaps, while real language-specific claims require reviewed language labels.

Split by author/video/thread when those identifiers are available. Current comment
grouping only prevents obvious duplicate/variant leakage, not shared-author or
shared-topic leakage. Calibration data should eventually match real usage rather
than the mixture proportions of these research datasets.

## Repeatable diagnostics

```powershell
.\.venv\Scripts\python.exe -m toxicity.evaluation --backend baseline --output runtime/reports/baseline_diagnostics.json
.\.venv\Scripts\python.exe -m toxicity.evaluation --backend transformer --output runtime/reports/transformer_diagnostics.json
```

`evaluation/smoke_cases.json` contains 40 authored English/Tamil/Thanglish probes:
39 scored examples and one policy-dependent example. The packaged baseline fails
11 of the 39 scored examples. These are **development diagnostics**, not a
statistically representative, independently annotated benchmark. Their expected
labels should be reviewed, particularly quotation/counterspeech cases.

Preparation and transformer training remove these diagnostic groups from the corpus.
Do not copy them into training to make the check pass. Use a fresh, larger reviewed
test set before claiming generalization; track precision, recall, false positives
on harmless conversation and errors separately for each language and category.
Set acceptance criteria before looking at that test set. Do not select models by
repeatedly optimizing against it. `--fail-on-errors` enables diagnostic CI failures.

## References

- [DravidianCodeMix dataset](https://zenodo.org/records/4750858)
- [Davidson data and label-bias research](https://github.com/t-davidson/hate-speech-and-offensive-language)
- [Google MuRIL model card](https://huggingface.co/google/muril-base-cased)
- [XLM-R model card](https://huggingface.co/FacebookAI/xlm-roberta-base)
- [Scikit-learn threshold tuning](https://scikit-learn.org/stable/modules/classification_threshold.html)
