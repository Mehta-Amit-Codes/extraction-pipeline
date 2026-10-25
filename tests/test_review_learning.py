import json

import pytest

from app import active_learning, jobqueue, quality, review, scoring, store
from app.schemas import ALL_FIELDS
from tests.helpers import add_stream, add_synth, drain


def pending(ns, n=500):
    return store.pending_reviews(ns, n)


def truth_of(doc_id):
    return store.get_document(doc_id)["meta"]["sim_truth"]


@pytest.fixture()
def reviewed_ns(ns):
    """30 template-B documents processed, so plenty are pending review."""
    ids = add_stream(ns, 30, change_at=0, seed0=3000)
    drain()
    return ns, ids


def test_approve_finalises_the_record_and_logs_the_review(reviewed_ns):
    ns, _ = reviewed_ns
    ext = pending(ns)[0]
    out = review.submit_review(ns, ext["id"], "ann", "approve")
    got = store.get_extraction(ext["id"])
    assert out["status"] == "approved" and got["final_record"] == got["record"] and got["reviewed_at"]
    [r] = [r for r in store.review_rows(ns) if r["extraction_id"] == ext["id"]]
    assert r["reviewer"] == "ann" and r["action"] == "approve" and r["changed_fields"] == []


def test_edit_records_exactly_which_fields_changed_and_teaches_the_pool(reviewed_ns):
    ns, _ = reviewed_ns
    ext = next(e for e in pending(ns) if e["record"]["total"] is not None)
    fixed = dict(ext["record"], total="123.45", vendor_name="Corrected Vendor Ltd")
    out = review.submit_review(ns, ext["id"], "bo", "edit", fixed)
    assert out["status"] == "corrected" and set(out["changed_fields"]) >= {"vendor_name", "total"}
    assert out["added_to_example_pool"]
    [ex] = [e for e in store.list_examples(ns) if e["document_id"] == ext["document_id"]]
    assert ex["source"] == "correction" and ex["record"]["total"] == "123.45"
    assert store.get_extraction(ext["id"])["final_record"]["vendor_name"] == "Corrected Vendor Ltd"


def test_an_edit_that_changes_nothing_is_recorded_as_an_approval(reviewed_ns):
    ns, _ = reviewed_ns
    ext = pending(ns)[0]
    out = review.submit_review(ns, ext["id"], "x", "edit", dict(ext["record"]))
    assert out["status"] == "approved" and out["changed_fields"] == []


def test_invalid_edits_and_actions_are_rejected_without_changing_state(reviewed_ns):
    ns, _ = reviewed_ns
    ext = pending(ns)[0]
    with pytest.raises(review.ReviewError):
        review.submit_review(ns, ext["id"], "x", "edit", dict(ext["record"], total="not money"))
    with pytest.raises(review.ReviewError):
        review.submit_review(ns, ext["id"], "x", "edit", None)
    with pytest.raises(review.ReviewError):
        review.submit_review(ns, ext["id"], "x", "shrug")
    assert store.get_extraction(ext["id"])["status"] == "pending_review"


def test_edits_that_still_fail_validation_are_allowed_but_warn(reviewed_ns):
    ns, _ = reviewed_ns
    ext = pending(ns)[0]
    out = review.submit_review(ns, ext["id"], "x", "edit", dict(ext["record"], total="1.00", subtotal="50.00", tax="0.00"))
    assert out["warnings"] and out["status"] == "corrected"        # the reviewer has authority; the UI shows the warning


def test_reject_keeps_the_record_out_of_exports_and_the_example_pool(reviewed_ns):
    ns, _ = reviewed_ns
    ext = pending(ns)[0]
    out = review.submit_review(ns, ext["id"], "x", "reject")
    assert out["status"] == "rejected" and not out["added_to_example_pool"]
    assert ext["document_id"] not in {r["document_id"] for r in store.final_records(ns)}
    assert not [e for e in store.list_examples(ns) if e["document_id"] == ext["document_id"]]


def test_a_review_can_only_be_applied_once_and_only_within_its_namespace(reviewed_ns):
    ns, _ = reviewed_ns
    ext = pending(ns)[0]
    assert review.submit_review(ns + "-other", ext["id"], "x", "approve") is None
    assert review.submit_review(ns, ext["id"], "x", "approve")
    assert review.submit_review(ns, ext["id"], "y", "reject") is None      # second reviewer lost the race
    assert store.get_extraction(ext["id"])["status"] == "approved"


def test_approving_a_covered_layout_does_not_bloat_the_pool(reviewed_ns):
    ns, _ = reviewed_ns
    first, second = pending(ns)[0], pending(ns)[1]
    assert review.submit_review(ns, first["id"], "x", "approve")["added_to_example_pool"]    # new layout: coverage
    assert not review.submit_review(ns, second["id"], "x", "approve")["added_to_example_pool"]  # already covered


def test_the_review_queue_is_ordered_most_uncertain_first(reviewed_ns):
    ns, _ = reviewed_ns
    prio = [e["priority"] for e in pending(ns)]
    assert prio == sorted(prio, reverse=True) and prio[0] > prio[-1]


def test_oracle_reviewer_corrects_synthetic_documents_to_ground_truth(reviewed_ns):
    ns, ids = reviewed_ns
    result = review.oracle_review_pending(ns, limit=500)
    assert result["edited"] > 0 and result["skipped_no_ground_truth"] == 0 and not pending(ns)
    for rec in store.final_records(ns, ("corrected",)):
        t = truth_of(rec["document_id"])
        assert all(rec["record"][f] == t[f] for f in ALL_FIELDS if f != "line_items")


def test_the_active_learning_loop_improves_the_same_documents_on_reprocessing(ns):
    ids = add_stream(ns, 60, change_at=0, seed0=4000)
    drain()
    review.oracle_review_pending(ns, limit=25)                     # humans correct the 25 most uncertain documents
    example_docs = {e["document_id"] for e in store.list_examples(ns)}
    rest = [d for d in ids if d not in example_docs]
    assert len(example_docs) >= 5 and rest
    for d in rest:
        jobqueue.enqueue(d, ns)
    drain()
    eff = quality.learning_effect(ns)
    assert eff["documents"] == len(rest) and eff["after_used_examples"] > 0
    assert eff["after"]["oracle_field_accuracy"] > eff["before"]["oracle_field_accuracy"] + 0.03


def test_example_pool_summary_and_deactivation(reviewed_ns):
    ns, _ = reviewed_ns
    review.oracle_review_pending(ns, limit=10)
    s = active_learning.pool_summary(ns)
    assert s["examples"] >= 1 and s["layouts_covered"] == 1            # every document is template B: one layout
    eid = store.list_examples(ns)[0]["id"]
    store.set_example_active(ns, eid, False)
    assert eid not in {e["id"] for e in store.list_examples(ns)} and eid in {e["id"] for e in store.list_examples(ns, active_only=False)}


def test_finetune_manifest_contains_verified_answers_only(reviewed_ns):
    ns, _ = reviewed_ns
    review.oracle_review_pending(ns, limit=10)
    rows = active_learning.export_finetune_manifest(ns)
    assert rows and all(r["messages"][2]["role"] == "assistant" for r in rows)
    answer = json.loads(rows[0]["messages"][2]["content"])
    assert set(answer["fields"]) == set(f for f in ALL_FIELDS if f != "line_items") and answer["line_items_confidence"] == 1.0


def test_quality_summary_estimates_accuracy_from_the_audit_sample_with_an_interval(ns, monkeypatch):
    monkeypatch.setattr(scoring, "is_audit_sample", lambda key, rate=1.0: True)       # audit every confident document
    add_stream(ns, 40, seed0=5000)
    drain()
    review.oracle_review_pending(ns, limit=500)
    q = quality.summary(ns)
    audit = q["audit"]
    assert audit["n"] > 0 and 0 <= audit["ci95"][0] <= audit["doc_accuracy"] <= audit["ci95"][1] <= 1
    assert q["processed"] == 40 and q["reviews_done"] >= audit["n"] and q["stp_rate"] is not None
    assert q["cost"]["cost_usd"] > 0 and q["cost"]["extractions"] == 40


def test_calibration_compares_stated_confidence_with_reviewed_correctness(reviewed_ns):
    ns, _ = reviewed_ns
    review.oracle_review_pending(ns, limit=500)
    cal = quality.calibration(ns)
    assert cal["n_fields"] > 0 and cal["ece"] is not None and cal["bins"]
    assert all(0 <= b["accuracy"] <= 1 and b["n"] > 0 for b in cal["bins"])
    assert quality.calibration(ns, audit_only=True)["n_fields"] <= cal["n_fields"]


def test_summary_of_an_empty_namespace_is_safe(ns):
    q = quality.summary(ns)
    assert q["processed"] == 0 and q["stp_rate"] is None and q["audit"]["n"] == 0 and quality.learning_effect(ns) == {"documents": 0}
