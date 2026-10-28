from __future__ import annotations

import numpy as np
from cleanlab.filter import find_label_issues
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

from .model import SEED, build_pipeline, partition


def audit_labels(df, limit=100, reviewed_ids=(), progress=None):
    """Rank training-label issues with out-of-fold, group-separated predictions."""
    progress = progress or (lambda message: None)
    train = partition(df)[0]["train"].reset_index(drop=True)
    if train.groupby("label").group.nunique().min() < 3:
        raise ValueError("Need at least three distinct groups in each training class for auditing.")
    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
    probabilities = np.zeros((len(train), 2))
    with threadpool_limits(limits=2):
        for fold, (a, b) in enumerate(cv.split(train.comment, train.label, train.group), 1):
            progress(f"Auditing training labels: fold {fold}/3")
            model = build_pipeline("word_char")
            model.fit(train.iloc[a].comment.tolist(), train.iloc[a].label.to_numpy())
            probabilities[b] = model.predict_proba(train.iloc[b].comment.tolist())
        ranked = find_label_issues(labels=train.label.to_numpy(), pred_probs=probabilities,
                                   return_indices_ranked_by="self_confidence", n_jobs=1)
    reviewed = set(reviewed_ids)
    rows = []
    for index in ranked:
        row = train.iloc[int(index)]
        if row.id in reviewed:
            continue
        p = float(probabilities[index, 1])
        rows.append({"id": row.id, "comment": row.comment, "label": int(row.label),
                     "original_label": row.original_label, "suggested_label": int(p >= 0.5),
                     "probability": round(p, 5),
                     "label_quality": round(float(probabilities[index, int(row.label)]), 5)})
        if len(rows) >= limit:
            break
    return {"method": "Cleanlab, three group-separated out-of-fold predictions; training rows only",
            "training_rows": len(train), "flagged_count": int(len(ranked)), "rows": rows,
            "note": "Suggestions may be wrong. Only an explicit reviewer decision changes a label."}
