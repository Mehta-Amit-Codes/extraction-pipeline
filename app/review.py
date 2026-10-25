"""
Human review: approve / edit / reject a pending extraction.

Human-in-the-loop here is a quality mechanism with three jobs, not a fallback:
  1. it gates what reaches downstream systems
  2. every correction is a labelled example that feeds the few-shot pool (active_learning)
  3. the AUDIT-SAMPLE slice gives unbiased accuracy estimates for the auto-approved stream (quality.py)
"""
from app import active_learning, scoring, store
from app.schemas import ALL_FIELDS, canonicalize_record, fields_equal


class ReviewError(ValueError):
    pass


def submit_review(namespace: str, extraction_id: int, reviewer: str, action: str, corrected: dict | None = None) -> dict | None:
    """Returns a summary dict, or None if the extraction is not pending (already reviewed / unknown)."""
    if action not in ("approve", "edit", "reject"):
        raise ReviewError("action must be approve, edit or reject")
    ext = store.get_extraction(extraction_id, namespace)
    if ext is None or ext["status"] != "pending_review":
        return None
    before = ext["record"]
    final, changed, warnings = None, [], []
    if action == "approve":
        final = before
    elif action == "edit":
        if not corrected:
            raise ReviewError("edit requires a corrected record")
        final, errors = canonicalize_record(corrected)
        if errors:
            raise ReviewError("; ".join(errors))
        changed = [f for f in ALL_FIELDS if not fields_equal(f, before.get(f), final.get(f))]
        if not changed:
            action = "approve"  # an "edit" that changed nothing is an approval; keep the statistics honest
        warnings = [i["message"] for i in scoring.validate_record(final) if i["severity"] == "error"]
    done = store.apply_review(extraction_id, namespace, reviewer, action, final, changed)
    if done is None:  # lost a race with another reviewer
        return None
    doc = store.get_document(ext["document_id"], namespace)
    example_id = active_learning.maybe_add_example(namespace, doc, final, action) if final else None
    return {"extraction_id": extraction_id, "status": done["status"], "changed_fields": changed,
            "added_to_example_pool": example_id is not None, "warnings": warnings}


def oracle_review_pending(namespace: str, limit: int = 50, reviewer: str = "oracle-demo") -> dict:
    """DEMO ONLY: a stand-in reviewer for SYNTHETIC documents that corrects against the known ground truth, so the
    active-learning loop can be exercised on hundreds of documents without hand-clicking. It does nothing for real
    documents (they have no ground truth)."""
    approved = edited = skipped = 0
    for ext in store.pending_reviews(namespace, limit):
        doc = store.get_document(ext["document_id"], namespace)
        truth = (doc["meta"] or {}).get("sim_truth")
        if not truth:
            skipped += 1
            continue
        same = all(fields_equal(f, ext["record"].get(f), truth.get(f)) for f in ALL_FIELDS)
        submit_review(namespace, ext["id"], reviewer, "approve" if same else "edit", None if same else truth)
        approved += int(same)
        edited += int(not same)
    return {"approved": approved, "edited": edited, "skipped_no_ground_truth": skipped}
