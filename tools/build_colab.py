"""Build a small, explicit Colab upload bundle; excludes environments and local state."""
from pathlib import Path
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "colab"


def markdown(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": source.splitlines(keepends=True)}


def main():
    if not (ROOT / "training_data.csv").exists():
        raise SystemExit("Run python -m toxicity.prepare first.")
    OUT.mkdir(exist_ok=True)
    files = list((ROOT / "toxicity").glob("*.py")) + [
        ROOT / "evaluation/smoke_cases.json", ROOT / "training_data.csv",
        ROOT / "training_data.info.json", ROOT / "requirements-transformer.txt",
        ROOT / "ROBUSTNESS.md",
    ]
    with zipfile.ZipFile(OUT / "mozhi_colab.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(ROOT).as_posix())
    cells = [
        markdown("""# Train Mozhi on a Colab GPU, then run it locally

Select **Runtime → Change runtime type → GPU** before running the cells.
Upload the accompanying `mozhi_colab.zip` when prompted. This contains the prepared
local training data and training code. Uploading sends that data to your Colab account.

This is a **binary offensive-language classifier**, including direct insults. It is
not a separate identity-based hate classifier. The dataset combines the local
DravidianCodeMix offensive splits and Davidson English tweets. Source labels are
unverified. Add independently reviewed everyday conversation, negation, quotation,
and Tamil/Thanglish examples before claiming deployment quality.

MuRIL is the default because its pretraining includes transliterated Indian languages.
You can compare `FacebookAI/xlm-roberta-base` in a separate run. A base encoder must
be fine-tuned: downloading it alone does not create an abuse detector.
"""),
        code("""%pip install -q 'torch>=2.6,<3' 'transformers==4.57.6' 'sentencepiece>=0.2,<0.3' 'safetensors>=0.5,<1' 'scikit-learn==1.8.0' 'pandas>=2.2,<3' 'scipy>=1.13' joblib threadpoolctl
import torch
assert torch.cuda.is_available(), 'Select a GPU runtime, then rerun this cell.'
print(torch.cuda.get_device_name(0))
"""),
        code("""from google.colab import files
from pathlib import Path
import zipfile, os
uploaded = files.upload()
assert 'mozhi_colab.zip' in uploaded, 'Upload mozhi_colab.zip from your mozhi/colab folder.'
project = Path('/content/mozhi').resolve()
project.mkdir(exist_ok=True)
with zipfile.ZipFile('mozhi_colab.zip') as archive:
    for member in archive.infolist():
        target = (project / member.filename).resolve()
        if not target.is_relative_to(project):
            raise ValueError('Archive contains a path outside the project.')
    archive.extractall(project)
os.chdir(project)
import pandas as pd
data = pd.read_csv('training_data.csv')
display(data.groupby(['source', 'split', 'label']).size().unstack(fill_value=0))
"""),
        code("""import subprocess, sys
BASE_MODEL = 'google/muril-base-cased'
OUTPUT = 'trained_muril'  # Use a NEW directory for each comparison.
subprocess.run([
    sys.executable, '-m', 'toxicity.train_transformer',
    '--data', 'training_data.csv', '--output', OUTPUT,
    '--base-model', BASE_MODEL, '--epochs', '3', '--batch-size', '8',
    '--accumulation', '4', '--max-length', '128', '--device', 'cuda',
    '--threshold-policy', 'balanced',
], check=True)
# If GPU memory is exhausted, restart with a fresh OUTPUT, batch-size 4,
# accumulation 8. Checkpoints are selected using validation only.
"""),
        code("""import json
report = json.loads(Path(OUTPUT, 'transformer', 'report.json').read_text())
print('Chosen epoch:', report['best_epoch'])
print('Test metrics:', report['test_slices']['overall'])
display(pd.DataFrame(report['test_slices']['source']).T)
display(pd.DataFrame(report['test_slices']['script']).T)
print('Truncated rows:', report['truncated_rows'])
# Source labels are NOT independent, real-world validation.
"""),
        code("""from toxicity.inference import load_bundle
from toxicity.evaluation import evaluate_cases, DEFAULT_CASES
bundle = load_bundle(Path(OUTPUT), backend='transformer')
cases = json.loads(DEFAULT_CASES.read_text(encoding='utf-8'))
diagnostics = evaluate_cases(bundle, cases)
Path(OUTPUT, 'transformer', 'diagnostics.json').write_text(
    json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding='utf-8')
display(pd.DataFrame(diagnostics['results'])[
    ['comment', 'language', 'expected_label', 'probability', 'label', 'correct']])
print(diagnostics['counts'])
# These small development probes are excluded from training, but are not a gold benchmark.
# Inspect failures and collect a separate native-speaker-reviewed test set.
"""),
        code("""import shutil
export = shutil.make_archive('/content/mozhi_transformer', 'zip', OUTPUT)
files.download(export)
"""),
        markdown("""## Run on Windows

Stop the running Mozhi server. Extract `mozhi_transformer.zip` **into
`E:\\Coding\\NLP_Project\\mozhi\\runtime\\models`** so the manifest is at
`runtime/models/transformer/mozhi.json`. Keep `model.joblib` for baseline comparisons.

In PowerShell, from `E:\\Coding\\NLP_Project\\mozhi`:

```powershell
.\\.venv\\Scripts\\python.exe -m pip install -r requirements-transformer.txt
.\\.venv\\Scripts\\python.exe run.py
```

The app should show **Multilingual transformer ready**. Inference is offline on CPU.
To force the old model for comparison, set `$env:MOZHI_MODEL_BACKEND='baseline'`
before starting the server; use `'auto'` to restore automatic checkpoint detection.

Long inputs exceeding the training token limit are marked for review. Do not treat
the likelihood score as severity or deploy solely because a few examples pass.
"""),
    ]
    notebook = {"nbformat": 4, "nbformat_minor": 5, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}, "accelerator": "GPU"}, "cells": cells}
    for index, cell in enumerate(cells):
        cell["id"] = f"mozhi-{index}"
    path = OUT / "Train_Mozhi.ipynb"
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")
    print(path)
    print(OUT / "mozhi_colab.zip")


if __name__ == "__main__":
    main()
