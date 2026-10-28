# Mozhi

**A local research app and Chrome extension for English, Tamil, and Thanglish comment analysis.** Mozhi predicts a broad **offensive-language** label for text and displays a probability-derived score. It includes a web workspace for dataset inspection and experiments, plus on-page indicators for comments currently loaded on YouTube watch pages.

Mozhi is a research prototype. Its `toxic` / `non_toxic` output is **not** a separate hate-speech category, a severity score, or a decision that should be used for automatic moderation. Predictions can be wrong, especially for code-mixed language and unfamiliar contexts.

## What you can do

- Paste up to 500 comments into the web app, review per-comment predictions, filter results, and export a CSV.
- Turn on the Chrome extension to see emoji and likelihood percentages beside rendered YouTube comments and replies. The popup also offers manual batch analysis and CSV export.
- Download the original research corpus or import a labeled CSV/TSV, compare baseline models, inspect evaluation reports, and review suspected label errors with Cleanlab.
- Train an optional multilingual transformer on a GPU and run it locally **where PyTorch is permitted**. The default app does not require PyTorch.

## Tech stack

| Part | Technology |
| --- | --- |
| Web frontend | HTML, CSS, vanilla JavaScript; no frontend build step |
| Chrome extension | Manifest V3, popup, content script, service worker, `chrome.storage.local` |
| Backend/API | Python 3.12, FastAPI, Pydantic, Uvicorn; served on `127.0.0.1:8000` |
| Default model | scikit-learn word/character TF-IDF, class-weighted Logistic Regression, sigmoid calibration |
| Optional GPU model | Fine-tuned MuRIL using PyTorch and Hugging Face Transformers; safetensors export |
| Data and evaluation | pandas, NumPy, SciPy, scikit-learn, Cleanlab |
| Database | **None.** Local CSV, JSON, and model files under `runtime/` |

Inference runs on the local backend. There is no translation API, generative LLM, API key, or cloud inference call.

## Quick start on Windows (PowerShell)

Run these commands from the project folder. A batch file is **not** required.

### 1. Install the baseline dependencies

Install [Python 3.12](https://www.python.org/downloads/) if needed. Then:

```powershell
cd "E:\Coding\NLP_Project\mozhi"
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If `.venv` already exists and works, you can skip the `py -3.12 -m venv .venv` line. If the `py` launcher is unavailable, use `python -m venv .venv` with a Python 3.12 installation.

### 2. Make sure a model is available

- **Existing local project folder:** If `runtime\models\model.joblib` exists, the default baseline is ready. Skip to step 3.
- **Fresh Git clone:** The trained weights are excluded from Git because model files can be large. Train the baseline once:

```powershell
.\.venv\Scripts\python.exe -m toxicity fetch
.\.venv\Scripts\python.exe -m toxicity train
```

The fetch step downloads the original DravidianCodeMix archive and checks its published checksum. Training saves `runtime\models\model.joblib`; it may take several minutes and needs no GPU. Alternatively, place a **trusted, compatible** baseline checkpoint at that path. Do not load untrusted `.joblib` files.

### 3. Start the app

```powershell
.\.venv\Scripts\python.exe run.py
```

Leave the terminal open. Open [the web app](http://127.0.0.1:8000) in Chrome. Open [the status endpoint](http://127.0.0.1:8000/api/status) to confirm `"model_ready": true` and check the active `"backend"`. Select **Load examples** and **Analyze comments** to try it. Press **Ctrl+C** in the terminal to stop the server.

The optional `setup_windows.bat` and `start_windows.bat` scripts perform the equivalent setup/startup. PowerShell commands above are useful when `.bat` files cannot be run.

### macOS/Linux

From the repository root, use Python 3.12 to create a virtual environment, then run:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m toxicity fetch   # fresh clone only
.venv/bin/python -m toxicity train   # fresh clone only
.venv/bin/python run.py
```

Open `http://127.0.0.1:8000`. The `fetch` and `train` steps are unnecessary when a compatible checkpoint is already present.

## Install and use the Chrome extension

1. Start the Python backend and confirm that `/api/status` reports a ready model.
2. In Chrome, open `chrome://extensions` and enable **Developer mode**.
3. Click **Load unpacked** and select the repository's **`extension`** folder (the folder containing `manifest.json`). Pin Mozhi from Chrome's Extensions menu if desired.
4. Open a YouTube video with a `/watch?v=...` URL. Scroll until comments load; expand replies if you want those analyzed.
5. Open the Mozhi popup and turn **Page indicators** **ON**. The saved toggle defaults to OFF.
6. Wait about 1–2 seconds for inline emoji/percent indicators and the page summary. Click an indicator for more detail. Use **Analyze loaded comments** in the popup for its manual result list and CSV export.

The extension analyzes **only comments rendered on the current page**, not every comment on a video. It batches at most 100 comments per API request; popup analysis can cover up to 500 loaded comments. Switching OFF removes annotations and stops scanning. If indicators do not appear after navigating to another video, open the popup on that video once to grant active-tab access. Reload the unpacked extension in `chrome://extensions` after editing extension files.

The extension has `activeTab`, `scripting`, `storage`, and fixed loopback host access to `http://127.0.0.1:8000/*`. It does not request broad YouTube host access.

## Models, scores, and current evidence

### Default baseline

The default runtime loads `runtime/models/model.joblib`. The saved baseline combines word unigrams/bigrams and character 3–5-grams with Logistic Regression and separate sigmoid calibration. Its selected probability threshold is **0.60**. On the original, unverified DravidianCodeMix test labels, its macro-F1 is **74.08%**, toxic precision **78.47%**, and toxic recall **46.38%**. It missed **556 of 1,037** labeled offensive test comments. See [MODEL_CARD.md](MODEL_CARD.md) for the split, comparisons, and confusion matrix.

### Optional transformer experiment

An exported fine-tuned `google/muril-base-cased` model and its report are stored separately under `runtime/gpu_run/models/transformer/`. Its **906 MB weights file is ignored by Git**, so a fresh clone does **not** contain a runnable transformer. The export's mixed-source test macro-F1 is **89.62%**, but its DravidianCodeMix source slice is **77.57%**; the overall score is strongly influenced by a different English tweet dataset. These are not directly comparable to the baseline's overall test score. The small hand-authored Thanglish diagnostic set also shows substantial remaining errors. See [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) and [ROBUSTNESS.md](ROBUSTNESS.md) before using those numbers in a presentation.

To use the optional export on a machine that permits PyTorch, install `requirements-transformer.txt`, provide the **complete** export including `model.safetensors`, and point the app at its runtime, for example:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-transformer.txt
$env:MOZHI_RUNTIME="E:\Coding\NLP_Project\mozhi\runtime\gpu_run"
.\.venv\Scripts\python.exe run.py
```

Check `/api/status` for `"backend": "transformer"`. On this project's previously tested Windows setup, application control blocked `torch_python.dll`; training on a GPU elsewhere does not remove the local PyTorch inference requirement. The baseline works without Torch. The [Colab notebook](colab/Train_Mozhi.ipynb) and [ROBUSTNESS.md](ROBUSTNESS.md) describe training and export.

For either model, `score = 10 × predicted probability`. A flag means the probability meets the saved threshold; **Needs context** means near the threshold or truncated input. It is not a confidence guarantee. The models learn from labeled examples: there is no hardcoded list of offensive phrases. Repeating the same input with a fixed model normally produces the same prediction.

## Dataset, training, and review workflow

The web app has four sections: **Comment analysis**, **Dataset studio**, **Experiments**, and **Label review**.

The baseline research importer reads the official DravidianCodeMix Tamil offensive-language train/dev/test splits. `Not_offensive` maps to `0`; offensive categories map to `1`; `not-Tamil` rows are excluded. For your own data, import UTF-8 CSV/TSV with `comment,label` columns and labels `0`/`1`; optional `split` values are `train`, `dev`, and `test`. Custom column and label mappings are available in the UI. The expanded GPU preparation path also uses the Davidson English hate/offensive tweet dataset, with its original classes mapped explicitly. Sentiment labels are not treated as toxicity labels.

Baseline experiments compare word-only, word+character, and word+character with training-only spelling augmentation. Training, calibration, validation, and test data remain separate. Validation selects the model and threshold; the held-out test measures the selected result. The **Label review** section uses Cleanlab to rank possible training-label issues; only an explicit human decision changes a label, and held-out labels are protected. Original corpus labels have not been independently verified here.

The optional transformer training workflow is in `colab/Train_Mozhi.ipynb`. When prompted, upload `colab/mozhi_colab.zip`; the notebook handles extraction. Training evaluates the export and downloads a model archive. The data source and results are documented in [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md).

## API and project layout

The local API docs are at `http://127.0.0.1:8000/docs`. Key endpoints are `GET /api/status`, `POST /api/analyze`, `POST /api/datasets/research`, `POST /api/datasets/import`, `POST /api/train`, `POST /api/audit`, `GET /api/reviews`, `POST /api/reviews`, and `GET /api/export/{kind}`. An analysis request accepts 1–500 comments, up to 10,000 characters each and 1,000,000 characters total. The backend binds to loopback and is designed for local use.

```text
mozhi/
  run.py                     Local server entry point
  toxicity/                  Data preparation, training, inference, API, audit
  web/                       Research workspace frontend
  extension/                 Chrome Manifest V3 extension
  colab/                     Optional GPU training notebook and local upload bundle
  tests/                     Python, extension, and browser checks
  runtime/models/            Default baseline checkpoint (local, Git-ignored)
  runtime/gpu_run/            Separate transformer export/report (weights Git-ignored)
  PROJECT_CONTEXT.md         Detailed architecture, metrics, limitations, slide plan
  MODEL_CARD.md              Baseline evaluation
  ROBUSTNESS.md              Failure analysis and multilingual training path
```

Generated datasets, checkpoints, and large weights are ignored by Git. A fresh clone needs model preparation before it can analyze comments.

## Tests

Install test dependencies if needed, then run from the project root:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
node --test tests\extension.test.cjs
```

At the last recorded run, **15 Python tests and 16 extension tests passed**. Node.js is needed for the extension tests, not for running the app. The tests cover data and split rules, training/API behavior, review safeguards, extension toggle/observer/batching/navigation, and error handling. Live YouTube behavior should also be checked in Chrome because YouTube can change its page markup.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| `Backend unavailable` in the popup | Start `run.py` and open `/api/status` on `127.0.0.1:8000`. |
| `No trained model` | A fresh clone has no weights. Run the baseline `fetch` and `train` commands above, then restart the server. |
| No YouTube indicators | Use a `/watch?v=...` page, scroll to rendered comments, turn the popup ON, and reopen it after a hard navigation if needed. |
| Port 8000 is already in use | Stop the earlier Mozhi terminal with **Ctrl+C** before starting another server. |
| Torch DLL blocked on Windows | Use the baseline; the optional transformer needs an environment where PyTorch can load under local policy. |

## References and scope

The original YouTube corpus is [DravidianCodeMix on Zenodo](https://zenodo.org/records/4750858); cite its authors for the data. The expanded experiment also uses the Davidson English tweet dataset, described with provenance and hashes in `training_data.info.json` when locally prepared. See [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) for the full project narrative, exact metric scopes, architecture, and presentation outline.

The published labels, source-domain calibration, and small diagnostic examples do not establish real-world accuracy. Evaluate on independently reviewed English, Tamil, and Thanglish comments from new videos before claiming reliable moderation performance.
