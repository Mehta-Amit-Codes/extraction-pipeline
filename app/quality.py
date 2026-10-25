"""
Quality metrics that stay honest about their own biases.

Headline numbers (STP rate, review rate) come from the system itself. ACCURACY comes only from humans, so:
  * audit accuracy is estimated from the random AUDIT SAMPLE of auto-approved documents, an unbiased estimate of how
    right the "confident" path is, with a Wilson confidence interval so a small sample does not look precise
  * calibration (does 0.9 confidence mean ~90% right?) uses reviewed fields; most reviewed documents were flagged
    BECAUSE they were uncertain, so that curve is biased toward the hard cases. It is shown with and without the audit slice
"""
from app import store
from app.drift import wilson
from app.schemas import ALL_FIELDS

BINS = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 0.95), (0.95, 1.01)]


def _field_outcomes(reviews: list[dict]):
    """(confidence, was_correct, was_audit) for every reviewed field. Correct = the human did not change it."""
    for r in reviews:
        if r["action"] == "reject" or not r.get("field_conf"):
            continue
        changed = set(r["changed_fields"] or [])
        for f in ALL_FIELDS:
            yield r["field_conf"][f], f not in changed, bool(r["audit_sample"])


def calibration(namespace: str, audit_only: bool = False) -> dict:
    rows = [(c, ok) for c, ok, audit in _field_outcomes(store.review_rows(namespace)) if audit or not audit_only]
    out, ece, n = [], 0.0, len(rows)
    for lo, hi in BINS:
        sel = [(c, ok) for c, ok in rows if lo <= c < hi]
        if not sel:
            continue
        acc, mean_c = sum(ok for _, ok in sel) / len(sel), sum(c for c, _ in sel) / len(sel)
        out.append({"bin": f"{lo:.2f}-{min(hi, 1.0):.2f}", "n": len(sel), "mean_confidence": round(mean_c, 3), "accuracy": round(acc, 3)})
        ece += len(sel) / n * abs(acc - mean_c)
    return {"bins": out, "ece": round(ece, 4) if n else None, "n_fields": n}


def summary(namespace: str) -> dict:
    rows = store.extraction_rows(namespace)
    counts = store.status_counts(namespace)
    n = len(rows)
    reviews = store.review_rows(namespace)
    audits = [r for r in reviews if r["audit_sample"]]
    audit_wrong = sum(1 for r in audits if r["changed_fields"])
    lo, hi = wilson(len(audits) - audit_wrong, len(audits))
    corrected: dict[str, int] = {}
    for r in reviews:
        for f in r["changed_fields"] or []:
            corrected[f] = corrected.get(f, 0) + 1
    return {
        "processed": n, "status_counts": counts,
        "stp_rate": round(counts.get("auto_approved", 0) / n, 4) if n else None,
        "review_rate": round(sum(1 for r in rows if r["needs_review"]) / n, 4) if n else None,
        "reviews_done": len(reviews),
        "audit": {"n": len(audits), "doc_accuracy": round(1 - audit_wrong / len(audits), 4) if audits else None,
                  "ci95": [round(lo, 4), round(hi, 4)] if audits else None,
                  "note": "estimated on a random sample of auto-approved documents (unbiased); wide interval = small sample"},
        "corrections_by_field": dict(sorted(corrected.items(), key=lambda kv: -kv[1])),
        "calibration": calibration(namespace), "calibration_audit_only": calibration(namespace, audit_only=True),
        "cost": store.cost_summary(namespace),
    }


def learning_effect(namespace: str) -> dict:
    """SIMULATION: same documents, first extraction vs latest. Only documents extracted more than once count, so this
    is a like-for-like before/after. Remember the simulator's response to examples is scripted: this validates the
    PLUMBING (examples selected and passed), not real model learning."""
    by_doc: dict[int, list[dict]] = {}
    for r in store.reviews_per_document_versions(namespace):
        by_doc.setdefault(r["document_id"], []).append(r)
    pairs = [(v[0], v[-1]) for v in by_doc.values() if len(v) > 1 and v[0]["oracle_field_acc"] is not None]
    if not pairs:
        return {"documents": 0}

    def agg(items):
        return {"oracle_field_accuracy": round(sum(i["oracle_field_acc"] for i in items) / len(items), 4),
                "review_rate": round(sum(i["needs_review"] for i in items) / len(items), 4),
                "exact_rate": round(sum(i["oracle_exact"] for i in items) / len(items), 4)}
    return {"documents": len(pairs), "before": agg([a for a, _ in pairs]), "after": agg([b for _, b in pairs]),
            "after_used_examples": sum(1 for _, b in pairs if b["fewshot_doc_ids"] not in (None, "[]"))}
