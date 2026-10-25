import csv
import io
import json
import time
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Response, UploadFile
from pydantic import BaseModel, Field

from app import active_learning, db, drift, imaging, jobqueue, quality, review, store, synth, worker
from app import config
from app.config import API_KEY, EMBEDDED_WORKERS, MAX_UPLOAD_MB
from app.schemas import ALL_FIELDS

db.init()
_pool: worker.WorkerPool | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pool
    if EMBEDDED_WORKERS > 0:
        _pool = worker.WorkerPool(concurrency=EMBEDDED_WORKERS).start()
    yield
    if _pool:
        _pool.stop()


def require_key(x_api_key: str = Header(default="")):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")


app = FastAPI(title="Structured Extraction Pipeline", version="0.1.0", lifespan=lifespan,
              dependencies=[Depends(require_key)],
              description="Batch invoice extraction with confidence scoring, human review, active learning and drift detection.")


# ---------------------------------------------------------------- ingestion ----
def _ingest(namespace: str, filename: str, data: bytes, batch_id: str | None, meta: dict | None = None) -> dict:
    try:
        doc_id, created = store.add_document(namespace, filename, data, batch_id, meta)
    except imaging.UnsupportedDocument as e:
        return {"filename": filename, "status": "rejected", "reason": str(e)}
    if not created:
        return {"filename": filename, "status": "duplicate", "document_id": doc_id}
    job_id, _ = jobqueue.enqueue(doc_id, namespace)
    return {"filename": filename, "status": "queued", "document_id": doc_id, "job_id": job_id}


@app.post("/documents")
async def upload(files: list[UploadFile] = File(...), namespace: str = Form("default"), batch_id: str = Form("")):
    out = []
    for f in files:
        data = await f.read(MAX_UPLOAD_MB * 1024 * 1024 + 1)
        if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
            out.append({"filename": f.filename, "status": "rejected", "reason": f"larger than {MAX_UPLOAD_MB} MB"})
            continue
        out.append(_ingest(namespace, f.filename or "upload", data, batch_id or None))
    return {"results": out, "queued": sum(1 for r in out if r["status"] == "queued")}


class SyntheticRequest(BaseModel):
    namespace: str = "default"
    n: int = Field(100, ge=1, le=2000)
    change_at: int | None = Field(None, ge=0, description="index at which template A switches to template B (the format change)")
    seed0: int | None = None
    noise: float = Field(0.0, ge=0, le=2)
    fail_every: int = Field(0, ge=0, description="script a transient failure on the first attempt of every Nth document")
    poison_every: int = Field(0, ge=0, description="script a permanently failing document every Nth document")


@app.post("/synthetic")
def synthetic(req: SyntheticRequest):
    """Generate a stream of synthetic invoices (with ground truth) and queue them. Pass `change_at` to switch layout
    mid-stream: the stretch-goal drift simulation."""
    seed0 = req.seed0 if req.seed0 is not None else int(time.time() * 1000) % 10_000_000
    batch = f"sim-{seed0}"
    counts = {"queued": 0, "duplicate": 0, "rejected": 0}
    for i, tpl, png, truth in synth.generate_stream(req.n, req.change_at, seed0, req.noise):
        meta = {"sim_truth": truth, "sim_template": tpl, "sim_index": i}
        if req.fail_every and i % req.fail_every == req.fail_every - 1:
            meta["sim_fail"] = {"mode": "retryable", "times": 1}
        if req.poison_every and i % req.poison_every == req.poison_every - 1:
            meta["sim_fail"] = {"mode": "permanent"}
        counts[_ingest(req.namespace, f"inv-{seed0}-{i:05d}-{tpl}.png", png, batch, meta)["status"]] += 1
    return {"batch_id": batch, "seed0": seed0, "change_at": req.change_at, **counts}


@app.get("/documents")
def list_documents(namespace: str = "default", batch_id: str | None = None, limit: int = 500):
    return [{"id": d["id"], "filename": d["filename"], "batch_id": d["batch_id"], "layout_hash": d["layout_hash"],
             "n_pages": d["n_pages"], "created_at": d["created_at"], "template": (d["meta"] or {}).get("sim_template")}
            for d in store.list_documents(namespace, batch_id, limit)]


@app.get("/documents/{document_id}/image")
def document_image(document_id: int, namespace: str = "default"):
    doc = store.get_document(document_id, namespace)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return Response(content=imaging.to_png_bytes(imaging.load_images(store.read_bytes(doc))[0]), media_type="image/png")


# -------------------------------------------------------------------- queue ----
@app.get("/queue/stats")
def queue_stats(namespace: str | None = None):
    return {**jobqueue.stats(namespace), "paused": worker.is_paused(), "pause_reason": store.get_kv("queue_pause_reason", "")}


@app.post("/queue/pause")
def queue_pause():
    worker.pause("manual")
    return {"paused": True}


@app.post("/queue/resume")
def queue_resume():
    worker.resume()
    return {"paused": False}


@app.get("/jobs")
def jobs(namespace: str = "default", status: str | None = None, limit: int = 100):
    return jobqueue.list_jobs(namespace, status, limit)


class RequeueRequest(BaseModel):
    namespace: str = "default"
    job_ids: list[int] | None = None


@app.post("/jobs/requeue-dead")
def requeue_dead(req: RequeueRequest):
    return {"requeued": jobqueue.requeue_dead(req.namespace, req.job_ids)}


class ReprocessRequest(BaseModel):
    namespace: str = "default"
    document_ids: list[int]


@app.post("/reprocess")
def reprocess(req: ReprocessRequest):
    """Run documents through the pipeline again (e.g. after the few-shot pool grew). Creates a NEW extraction version."""
    queued = sum(1 for d in req.document_ids if store.get_document(d, req.namespace) and jobqueue.enqueue(d, req.namespace)[1])
    return {"queued": queued}


# ------------------------------------------------------------------- review ----
@app.get("/reviews/pending")
def pending(namespace: str = "default", limit: int = 25):
    return [{k: e[k] for k in ("id", "document_id", "filename", "priority", "audit_sample", "review_reasons", "record",
                               "field_conf", "pass1", "pass2", "disputed_fields", "issues", "fewshot_doc_ids")}
            for e in store.pending_reviews(namespace, limit)]


class ReviewRequest(BaseModel):
    action: str
    reviewer: str = "reviewer"
    record: dict | None = None
    namespace: str = "default"


@app.post("/reviews/{extraction_id}")
def submit_review(extraction_id: int, req: ReviewRequest):
    try:
        result = review.submit_review(req.namespace, extraction_id, req.reviewer, req.action, req.record)
    except review.ReviewError as e:
        raise HTTPException(status_code=422, detail=str(e))
    if result is None:
        raise HTTPException(status_code=409, detail="Extraction is not pending review (already reviewed, or unknown)")
    return result


class OracleRequest(BaseModel):
    namespace: str = "default"
    limit: int = Field(50, ge=1, le=1000)


@app.post("/demo/oracle-review")
def oracle_review(req: OracleRequest):
    """DEMO ONLY: auto-review SYNTHETIC documents against their ground truth to exercise active learning at volume."""
    return review.oracle_review_pending(req.namespace, req.limit)


# ----------------------------------------------------------- quality / drift ----
@app.get("/quality")
def quality_summary(namespace: str = "default"):
    return quality.summary(namespace)


@app.get("/quality/learning-effect")
def learning_effect(namespace: str = "default"):
    return quality.learning_effect(namespace)


@app.get("/drift")
def drift_report(namespace: str = "default"):
    return drift.evaluate(namespace, persist=False)


@app.get("/drift/timeseries")
def drift_timeseries(namespace: str = "default", step: int = 10):
    return drift.timeseries(namespace, max(1, step))


@app.get("/drift/simulation")
def drift_simulation(namespace: str = "default"):
    return drift.simulation_report(namespace)


@app.post("/drift/baseline/pin")
def pin_baseline(namespace: str = "default"):
    return drift.pin_baseline(namespace)


@app.get("/alerts")
def alerts(namespace: str = "default", status: str | None = None):
    return store.list_alerts(namespace, status)


@app.post("/alerts/{alert_id}/ack")
def ack_alert(alert_id: int, namespace: str = "default"):
    if not store.ack_alert(namespace, alert_id):
        raise HTTPException(status_code=404, detail="Alert not found")
    return {"acknowledged": True}


# ----------------------------------------------------------- learning / export ----
@app.get("/examples")
def examples(namespace: str = "default"):
    return {"summary": active_learning.pool_summary(namespace), "examples": store.list_examples(namespace, active_only=False)}


class ActiveRequest(BaseModel):
    namespace: str = "default"
    active: bool


@app.post("/examples/{example_id}/active")
def set_active(example_id: int, req: ActiveRequest):
    store.set_example_active(req.namespace, example_id, req.active)
    return {"ok": True}


@app.get("/examples/export")
def export_examples(namespace: str = "default"):
    body = "\n".join(json.dumps(r) for r in active_learning.export_finetune_manifest(namespace))
    return Response(content=body, media_type="application/x-ndjson")


@app.get("/export")
def export(namespace: str = "default", format: str = "jsonl"):
    """Only records that are safe to use downstream: auto-approved, human-approved, or human-corrected."""
    recs = store.final_records(namespace)
    if format == "csv":
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["document", "status", *[f for f in ALL_FIELDS if f != "line_items"], "line_items_json"])
        for r in recs:
            w.writerow([r["filename"], r["status"], *[r["record"].get(f) for f in ALL_FIELDS if f != "line_items"],
                        json.dumps(r["record"]["line_items"])])
        return Response(content=buf.getvalue(), media_type="text/csv")
    return Response(content="\n".join(json.dumps(r) for r in recs), media_type="application/x-ndjson")


@app.get("/config")
def get_config():
    """Non-secret settings the UI needs (e.g. to stop a simulation from putting the format change inside the baseline)."""
    return {"baseline_n": config.BASELINE_N, "window_n": config.WINDOW_N, "min_samples": config.MIN_SAMPLES,
            "drift_eval_every": config.DRIFT_EVAL_EVERY, "review_threshold": config.REVIEW_THRESHOLD,
            "cross_check": config.CROSS_CHECK, "extractor": config.EXTRACTOR, "fewshot_k": config.FEWSHOT_K,
            "audit_sample_rate": config.AUDIT_SAMPLE_RATE, "max_attempts": config.MAX_ATTEMPTS}


@app.get("/health")
def health():
    return {"status": "ok", "workers": len(_pool.threads) if _pool else 0}
