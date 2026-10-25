"""
process_job: one claimed job -> one extraction row.

    load images -> pick similar corrected examples -> pass 1 -> (pass 2) -> merge -> validate -> cap -> route -> persist

The result is written in the SAME transaction as the job completion, behind an ownership check (see jobqueue.owns).
A worker whose lease expired and whose job was handed to someone else therefore cannot write a duplicate result.

Failure handling maps the extractor's error taxonomy onto queue behaviour:
  RetryableError / unexpected exception   retry with backoff (bounded by max attempts, then dead-letter)
  PermanentError / unsupported file       dead-letter immediately; retrying cannot help
  ConfigError                             release WITHOUT consuming an attempt; the worker pauses the queue
"""
import time
import traceback

from app import active_learning, db, drift, imaging, jobqueue, scoring, store
from app.config import COST_PER_M_INPUT, COST_PER_M_OUTPUT, CROSS_CHECK, SELECTIVE_CONF
from app.extractors import ConfigError, ExtractionRequest, PermanentError, RetryableError
from app.schemas import ALL_FIELDS, to_canonical


def _cost(tokens_in: int, tokens_out: int) -> float:
    return tokens_in / 1e6 * COST_PER_M_INPUT + tokens_out / 1e6 * COST_PER_M_OUTPUT


def _want_second_pass(mode: str, c1) -> bool:
    if mode == "always":
        return True
    if mode == "never":
        return False
    # selective: spend the second call only where pass 1 shows any doubt (low confidence or a validator hit)
    issues = scoring.validate_record(c1.values, c1.parse_errors)
    return any(c1.conf[f] < SELECTIVE_CONF for f in ALL_FIELDS) or any(i["severity"] == "error" for i in issues)


def extract_document(doc: dict, extractor, attempt: int = 1, cross_check: str | None = None) -> dict:
    """Pure-ish core (no job/queue side effects): everything needed to persist an extraction row."""
    cross_check = cross_check or CROSS_CHECK
    images = imaging.load_images(store.read_bytes(doc))
    examples = active_learning.select_examples(doc["namespace"], doc["layout_hash"], exclude_document_id=doc["id"])

    def request(pass_index):
        return ExtractionRequest(images=images, doc_key=doc["sha256"], layout_hash=doc["layout_hash"], meta=doc["meta"],
                                 examples=examples, pass_index=pass_index, attempt=attempt)

    out1 = extractor.extract(request(1))
    c1 = to_canonical(out1.raw)
    out2 = c2 = None
    if _want_second_pass(cross_check, c1):
        out2 = extractor.extract(request(2))
        c2 = to_canonical(out2.raw)

    merged = scoring.merge_passes(c1, c2)
    issues = scoring.validate_record(merged["values"], merged["parse_errors"])
    conf = scoring.apply_issue_caps(merged["conf"], issues)
    reasons = scoring.route(conf, merged["disputed"], issues)
    audit = (not reasons) and scoring.is_audit_sample(doc["sha256"])

    novel = drift.is_novel(doc)  # a review-PRIORITY hint only; drift evaluation recomputes novelty itself
    tin = out1.tokens_in + (out2.tokens_in if out2 else 0)
    tout = out1.tokens_out + (out2.tokens_out if out2 else 0)
    oracle = None
    if doc["meta"].get("sim_truth"):
        oracle = scoring.oracle_scores(merged["values"], doc["meta"]["sim_truth"])

    return {
        "extractor": extractor.name, "model": out1.model,
        "status": "pending_review" if (reasons or audit) else "auto_approved",
        "needs_review": int(bool(reasons)), "review_reasons": reasons + (["audit_sample"] if audit else []),
        "priority": scoring.review_priority(conf, merged["disputed"], issues, novel) if reasons else (0.05 if audit else 0.0),
        "audit_sample": int(audit), "crosschecked": int(out2 is not None),
        "record": merged["values"], "field_conf": conf, "pass1": c1.values, "pass2": c2.values if c2 else None,
        "disputed_fields": merged["disputed"], "issues": issues,
        "fewshot_doc_ids": [e.document_id for e in examples],
        "mean_conf": sum(conf.values()) / len(conf), "min_conf": min(conf.values()),
        "n_disputed": len(merged["disputed"]), "n_issues": sum(1 for i in issues if i["severity"] == "error"),
        "n_null": sum(1 for f in ALL_FIELDS if not merged["values"][f]),
        "tokens_in": tin, "tokens_out": tout, "cost_usd": _cost(tin, tout),
        "latency_ms": out1.latency_ms + (out2.latency_ms if out2 else 0),
        "oracle_field_acc": oracle[0] if oracle else None, "oracle_exact": oracle[1] if oracle else None,
    }


def process_job(job: dict, extractor, worker_id: str, now: float | None = None) -> str:
    """Returns: done | retry | dead | released | lost. Never raises: one bad document must not kill a worker."""
    now = now or time.time()
    try:
        doc = store.get_document(job["document_id"])
        if doc is None:
            raise PermanentError("document row is missing")
        fields = extract_document(doc, extractor, attempt=job["attempts"])
    except imaging.UnsupportedDocument as e:
        return _fail(job, worker_id, f"unsupported document: {e}", "unsupported", retryable=False)
    except PermanentError as e:
        return _fail(job, worker_id, str(e), "permanent", retryable=False)
    except ConfigError as e:
        return "released" if jobqueue.release(job["id"], worker_id, f"config: {e}") else "lost"
    except RetryableError as e:
        return _fail(job, worker_id, str(e), e.kind, retryable=True, retry_after=e.retry_after)
    except Exception as e:  # unexpected: retry (bounded), keep the traceback for the operator
        return _fail(job, worker_id, f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}", "unexpected", retryable=True)

    with db.tx() as c:
        if not jobqueue.owns(c, job["id"], worker_id):
            return "lost"  # lease expired and another worker took over: discard, don't write a duplicate
        store.insert_extraction(c, document_id=doc["id"], namespace=doc["namespace"], job_id=job["id"], created_at=now, **fields)
        jobqueue.mark_done(c, job["id"], now)
    drift.maybe_evaluate(doc["namespace"])
    return "done"


def _fail(job, worker_id, error, kind, retryable, retry_after=None) -> str:
    status = jobqueue.fail(job["id"], worker_id, error, kind, retryable, retry_after)
    return {"queued": "retry", "dead": "dead", None: "lost"}[status]
