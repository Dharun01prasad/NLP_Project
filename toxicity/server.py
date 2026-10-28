from __future__ import annotations

import csv
import io
import json
import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from threadpoolctl import threadpool_limits
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .audit import audit_labels
from .data import fetch_research_dataset, read_upload
from .model import partition, predict, train_experiment
from .inference import load_bundle

ROOT = Path(__file__).resolve().parents[1]


class Upload(BaseModel):
    filename: str = Field(default="dataset.csv", max_length=200)
    content: str = Field(min_length=1, max_length=25_000_000)
    text_column: str = "comment"
    label_column: str = "label"
    mapping: dict[str, int | None] | None = None


class Analysis(BaseModel):
    comments: list[str] = Field(min_length=1, max_length=500)

    @field_validator("comments")
    @classmethod
    def valid_comments(cls, items):
        if any(not s.strip() or len(s) > 10000 for s in items):
            raise ValueError("Each comment must contain 1–10,000 characters.")
        if sum(map(len, items)) > 1_000_000:
            raise ValueError("Batch exceeds one million characters.")
        return [s.strip() for s in items]


class TrainOptions(BaseModel):
    augment: bool = True


class Decision(BaseModel):
    id: str = Field(min_length=20, max_length=20)
    label: int = Field(ge=0, le=1)


class Review(BaseModel):
    revision: str
    decisions: list[Decision] = Field(min_length=1, max_length=500)


class Store:
    def __init__(self, path):
        self.path = Path(path)
        for folder in ("data", "models", "reports"):
            (self.path / folder).mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mozhi")
        self.jobs = {}
        self.active = None
        self.bundle = load_bundle(self.path / "models", os.environ.get("MOZHI_MODEL_BACKEND", "auto"))

    def read_json(self, name, fallback=None):
        path = self.path / name
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else fallback

    def write_json(self, name, value):
        path = self.path / name
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temp, path)

    def dataframe(self):
        path = self.path / "data/dataset.csv"
        if not path.exists():
            raise ValueError("Load the research dataset or import a labeled CSV first.")
        return pd.read_csv(path, keep_default_na=False, dtype={"id": str})

    def install_data(self, df, info):
        # Freeze holdout assignments now. Training-label corrections never reshuffle test rows.
        parts, split_info = partition(df)
        parts["train"]["split"] = "train"
        parts["calibration"]["split"] = "dev"
        parts["validation"]["split"] = "dev"
        parts["test"]["split"] = "test"
        frozen = pd.concat(parts.values(), ignore_index=True)
        frozen.to_csv(self.path / "data/dataset.csv", index=False)
        info.update({"revision": uuid.uuid4().hex, "saved_rows": len(frozen),
                     "split": split_info, "reviewed_count": 0, "corrected_count": 0,
                     "class_counts": {"non_toxic": int(frozen.label.eq(0).sum()), "toxic": int(frozen.label.eq(1).sum())},
                     "created_at": datetime.now(timezone.utc).isoformat()})
        self.write_json("data/info.json", info)
        self.write_json("data/reviews.json", [])
        self.write_json("reports/audit.json", None)
        return info

    def launch(self, kind, action):
        with self.lock:
            if self.active:
                raise HTTPException(409, "A dataset job is already running. Wait for it to finish.")
            identity = uuid.uuid4().hex
            job = {"id": identity, "kind": kind, "status": "running", "message": "Starting…"}
            self.jobs[identity] = job
            self.active = identity
        def progress(message):
            with self.lock:
                job["message"] = message
        def run():
            try:
                result = action(progress)
                with self.lock:
                    job.update(status="complete", message="Complete", result=result)
            except Exception as exc:
                with self.lock:
                    job.update(status="failed", message=str(exc))
            finally:
                with self.lock:
                    self.active = None
        self.executor.submit(run)
        return dict(job)


def create_app(runtime=None):
    store = Store(runtime or os.environ.get("MOZHI_RUNTIME", ROOT / "runtime"))
    app = FastAPI(title="Mozhi · Tanglish toxicity lab", version="1.0.0")
    app.state.store = store
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "testserver"])
    app.add_middleware(CORSMiddleware, allow_origin_regex=r"chrome-extension://[a-p]{32}",
                       allow_methods=["GET", "POST"], allow_headers=["Content-Type"])

    @app.middleware("http")
    async def local_requests(request: Request, call_next):
        origin = request.headers.get("origin", "")
        allowed = origin in {"", "http://localhost:8000", "http://127.0.0.1:8000", "http://testserver"}
        extension = origin.startswith("chrome-extension://")
        if request.url.path.startswith("/api/") and request.method == "POST":
            if not allowed and not (extension and request.url.path == "/api/analyze"):
                return Response("Cross-origin writes are not allowed", status_code=403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(ValueError)
    async def value_error(request, exc):
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.get("/api/status")
    def status():
        info = store.read_json("data/info.json")
        report = store.bundle["report"] if store.bundle else None
        return {"model_ready": store.bundle is not None, "dataset": info, "report": report,
                "backend": store.bundle.get("backend", "baseline") if store.bundle else None,
                "active_job": store.jobs.get(store.active), "human_validated": False,
                "model_stale": bool(info and report and info.get("revision") != report["dataset"].get("revision"))}

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str):
        if job_id not in store.jobs:
            raise HTTPException(404, "Job not found")
        return dict(store.jobs[job_id])

    @app.post("/api/datasets/research")
    def research():
        def action(progress):
            progress("Downloading the original 10.8 MB archive and checking its checksum")
            df, info = fetch_research_dataset(store.path / "data/source")
            progress("Checking labels, duplicates, and split overlap")
            return store.install_data(df, info)
        return store.launch("dataset", action)

    @app.post("/api/datasets/import")
    def upload(payload: Upload):
        def action(progress):
            progress("Parsing labeled comments and checking duplicate groups")
            df, info = read_upload(payload.content, payload.text_column, payload.label_column, payload.mapping)
            info.update(name=Path(payload.filename).name, labels_verified=False, source="User import")
            return store.install_data(df, info)
        return store.launch("dataset", action)

    @app.post("/api/train")
    def train(options: TrainOptions):
        if store.bundle and store.bundle.get("backend") == "transformer":
            raise HTTPException(409, "Train the transformer with the Colab notebook. To compare TF-IDF models, restart with MOZHI_MODEL_BACKEND=baseline.")
        def action(progress):
            df = store.dataframe()
            info = store.read_json("data/info.json")
            bundle = train_experiment(df, info, store.path / "models", store.path / "reports", options.augment, progress)
            with store.lock:
                store.bundle = bundle
            return {"selected_model": bundle["report"]["selected_model"]}
        return store.launch("training", action)

    @app.post("/api/audit")
    def audit():
        def action(progress):
            reviewed = store.read_json("data/reviews.json", [])
            result = audit_labels(store.dataframe(), reviewed_ids=[r["id"] for r in reviewed], progress=progress)
            result["revision"] = store.read_json("data/info.json")["revision"]
            store.write_json("reports/audit.json", result)
            return {"flagged_count": result["flagged_count"], "queued": len(result["rows"])}
        return store.launch("audit", action)

    @app.get("/api/reviews")
    def reviews():
        return store.read_json("reports/audit.json") or {"rows": [], "flagged_count": 0}

    @app.post("/api/reviews")
    def apply_reviews(payload: Review):
        with store.lock:
            if store.active:
                raise HTTPException(409, "Wait for the active job before editing labels.")
            info = store.read_json("data/info.json")
            audit = store.read_json("reports/audit.json")
            if not info or payload.revision != info["revision"]:
                raise HTTPException(409, "The dataset changed. Reload the review queue.")
            queue = {row["id"]: row for row in (audit or {}).get("rows", [])}
            if any(d.id not in queue for d in payload.decisions):
                raise ValueError("Only queued training comments can be reviewed.")
            if len({d.id for d in payload.decisions}) != len(payload.decisions):
                raise ValueError("Duplicate review decisions are not allowed.")
            df = store.dataframe()
            history = store.read_json("data/reviews.json", [])
            for decision in payload.decisions:
                mask = df.id.eq(decision.id)
                if not mask.any() or not df.loc[mask, "split"].eq("train").all():
                    raise ValueError("Held-out labels cannot be edited through the training review queue.")
                old = int(df.loc[mask, "label"].iloc[0])
                df.loc[mask, "label"] = decision.label
                history.append({"id": decision.id, "old_label": old, "label": decision.label,
                                "reviewed_at": datetime.now(timezone.utc).isoformat()})
                queue.pop(decision.id)
            df.to_csv(store.path / "data/dataset.csv", index=False)
            info["revision"] = uuid.uuid4().hex
            info["reviewed_count"] = len(history)
            info["corrected_count"] = sum(r["old_label"] != r["label"] for r in history)
            info["class_counts"] = {"non_toxic": int(df.label.eq(0).sum()), "toxic": int(df.label.eq(1).sum())}
            audit.update(rows=list(queue.values()), revision=info["revision"])
            store.write_json("data/info.json", info)
            store.write_json("data/reviews.json", history)
            store.write_json("reports/audit.json", audit)
            return {"saved": len(payload.decisions), "remaining": len(queue)}

    @app.post("/api/analyze")
    def analyze(payload: Analysis):
        bundle = store.bundle
        if bundle is None:
            raise HTTPException(503, "No trained model. Load a labeled dataset and run training in the lab.")
        with store.lock, threadpool_limits(limits=2):
            rows = predict(bundle, payload.comments)
        toxic = sum(row["label"] == "toxic" for row in rows)
        return {"results": rows, "summary": {"analyzed": len(rows), "flagged": toxic,
                "flagged_percent": round(100 * toxic / len(rows), 1),
                "mean_likelihood_score": round(sum(r["score"] for r in rows) / len(rows), 2),
                "uncertain": sum(r["uncertain"] for r in rows)}, "threshold": bundle["threshold"],
                "model": bundle["report"]["selected_model"], "human_validated": False,
                "score_meaning": "Predicted offensive-language likelihood, not abuse severity"}

    @app.get("/api/export/{kind}")
    def export(kind: str):
        if kind == "report" and store.bundle:
            return Response(json.dumps(store.bundle["report"], ensure_ascii=False, indent=2),
                            media_type="application/json", headers={"Content-Disposition": 'attachment; filename="experiment.json"'})
        names = {"report": "reports/experiment.json", "dataset": "data/dataset.csv", "audit": "reports/audit.json"}
        if kind not in names:
            raise HTTPException(404, "Unknown export")
        path = store.path / names[kind]
        if not path.is_file():
            raise HTTPException(404, "Run the corresponding step first.")
        return FileResponse(path, filename=path.name)

    app.mount("/", StaticFiles(directory=ROOT / "web", html=True), name="web")
    return app


app = create_app()
