import time

import pytest

from app import db, jobqueue, pipeline, scoring, store, worker
from app.extractors import SimulatedExtractor
from tests.helpers import add_stream, add_synth, drain


def all_jobs(ns):
    return jobqueue.list_jobs(ns, limit=1000)


def extraction(doc_id):
    return store.current_extraction(doc_id)


def test_full_pipeline_produces_a_routed_extraction_for_every_document(ns):
    ids = add_stream(ns, 30)
    counts = drain()
    assert counts["done"] == 30 and counts["dead"] == 0
    rows = [extraction(d) for d in ids]
    assert all(r and set(r["record"]) >= {"vendor_name", "total", "line_items"} for r in rows)
    assert {r["status"] for r in rows} <= {"auto_approved", "pending_review"}
    assert any(r["status"] == "auto_approved" for r in rows) and all(r["oracle_field_acc"] is not None for r in rows)
    assert all(r["cost_usd"] > 0 and r["tokens_in"] > 0 for r in rows)


def test_uploading_the_same_bytes_twice_creates_one_document_and_one_job(ns):
    a = add_synth(ns, 1)
    b = add_synth(ns, 1)
    assert a == b and len(all_jobs(ns)) == 1


def test_cross_check_modes(ns, monkeypatch):
    monkeypatch.setattr(pipeline, "SELECTIVE_CONF", 0.0)     # selective: only validator hits trigger pass 2
    docs = {m: [add_synth(ns, i, seed0=s, enqueue=False) for i in range(25)] for m, s in
            (("never", 0), ("always", 1000), ("selective", 2000))}
    ext = SimulatedExtractor()
    for mode, ids in docs.items():
        for d in ids:
            doc = store.get_document(d)
            fields = pipeline.extract_document(doc, ext, cross_check=mode)
            assert fields["crosschecked"] == {"never": 0, "always": 1}.get(mode, fields["crosschecked"])
            if mode == "never":
                assert fields["pass2"] is None
    sel = [pipeline.extract_document(store.get_document(d), ext, cross_check="selective")["crosschecked"] for d in docs["selective"]]
    assert 0 < sum(sel) < len(sel)                            # spent a second call only where there was doubt


def test_disagreement_between_passes_routes_the_document_to_review(ns):
    ids = add_stream(ns, 40, change_at=0)                      # all template B: noisy, so passes disagree sometimes
    drain()
    disputed = [extraction(d) for d in ids if extraction(d)["n_disputed"]]
    assert disputed and all(e["status"] == "pending_review" and any(r.startswith("disputed:") for r in e["review_reasons"]) for e in disputed)
    assert all(e["record"] is not None and e["pass2"] is not None for e in disputed)


def test_a_transient_failure_is_retried_with_backoff_and_then_succeeds(ns):
    d = add_synth(ns, 1, extra_meta={"sim_fail": {"mode": "retryable", "times": 2}})
    counts = drain()
    assert counts["retry"] == 2 and counts["done"] == 1 and counts["dead"] == 0
    assert all_jobs(ns)[0]["attempts"] == 3 and extraction(d)


def test_a_document_that_always_fails_is_dead_lettered_after_max_attempts(ns):
    d = add_synth(ns, 1, extra_meta={"sim_fail": {"mode": "retryable", "times": 99}})
    counts = drain()
    j = all_jobs(ns)[0]
    assert counts["dead"] == 1 and counts["retry"] == 2 and j["status"] == "dead" and j["attempts"] == 3
    assert j["last_error_kind"] == "rate_limit" and extraction(d) is None


def test_a_permanent_failure_is_dead_lettered_without_retrying(ns):
    add_synth(ns, 1, extra_meta={"sim_fail": {"mode": "permanent"}})
    counts = drain()
    j = all_jobs(ns)[0]
    assert counts == {"done": 0, "retry": 0, "dead": 1, "released": 0, "lost": 0}
    assert j["attempts"] == 1 and j["last_error_kind"] == "permanent"


def test_good_documents_are_unaffected_by_poison_documents_in_the_same_batch(ns):
    ids = [add_synth(ns, i, extra_meta={"sim_fail": {"mode": "permanent"}} if i % 5 == 0 else None) for i in range(20)]
    counts = drain()
    assert counts["dead"] == 4 and counts["done"] == 16
    assert sum(1 for d in ids if extraction(d)) == 16


def test_config_errors_do_not_burn_attempts_and_the_circuit_breaker_pauses_the_queue(ns):
    """An expired API key must not turn the whole backlog into dead letters."""
    for i in range(6):
        add_synth(ns, i, extra_meta={"sim_fail": {"mode": "config"}})
    counts = drain(concurrency=1)
    assert worker.is_paused() and "circuit breaker" in store.get_kv("queue_pause_reason")
    assert counts["released"] == 3 and counts["dead"] == 0                 # tripped after CONFIG_ERROR_PAUSE_AFTER
    jobs = all_jobs(ns)
    assert all(j["status"] == "queued" and j["attempts"] == 0 for j in jobs)   # nothing lost, nothing charged


def test_a_paused_queue_does_nothing_and_drain_mode_still_terminates(ns):
    add_stream(ns, 5)
    worker.pause("test")
    assert drain()["done"] == 0
    worker.resume()
    assert drain()["done"] == 5


def test_a_worker_whose_lease_was_taken_over_cannot_write_a_duplicate_result(ns):
    d = add_synth(ns, 1)
    now = time.time()
    slow = jobqueue.claim("slow", now=now, lease_s=10)
    jobqueue.reclaim_expired(now=now + 100)
    fast = jobqueue.claim("fast", now=now + 100)
    assert pipeline.process_job(slow, SimulatedExtractor(), "slow") == "lost"
    assert extraction(d) is None                                           # the stale worker wrote nothing
    assert pipeline.process_job(fast, SimulatedExtractor(), "fast") == "done"
    with db.conn() as c:
        assert c.execute("SELECT COUNT(*) AS n FROM extractions WHERE document_id = ?", (d,)).fetchone()["n"] == 1


def test_many_threads_process_every_document_exactly_once(ns):
    ids = add_stream(ns, 60)
    assert drain(concurrency=8)["done"] == 60
    with db.conn() as c:
        rows = c.execute("SELECT document_id, COUNT(*) AS n FROM extractions WHERE namespace = ? GROUP BY document_id", (ns,)).fetchall()
    assert len(rows) == 60 and all(r["n"] == 1 for r in rows)


def test_reprocessing_creates_a_new_version_and_only_the_latest_is_current(ns):
    d = add_synth(ns, 1)
    drain()
    jobqueue.enqueue(d, ns)
    drain()
    with db.conn() as c:
        rows = c.execute("SELECT version, is_current FROM extractions WHERE document_id = ? ORDER BY version", (d,)).fetchall()
    assert [(r["version"], r["is_current"]) for r in rows] == [(1, 0), (2, 1)]


def test_audit_sampling_sends_confident_documents_to_review_without_counting_them_as_flagged(ns, monkeypatch):
    monkeypatch.setattr(scoring, "is_audit_sample", lambda key, rate=1.0: True)
    ids = add_stream(ns, 20)
    drain()
    audited = [extraction(d) for d in ids if extraction(d)["audit_sample"]]
    assert audited and all(e["status"] == "pending_review" and e["needs_review"] == 0 and "audit_sample" in e["review_reasons"] for e in audited)


def test_few_shot_examples_are_chosen_by_layout_and_never_cross_namespaces(ns):
    from app import active_learning
    b = [add_synth(ns, i, "B", seed0=500) for i in range(3)]
    a = add_synth(ns, 9, "A", seed0=500)
    for d in (*b, a):
        doc = store.get_document(d)
        store.upsert_example(ns, d, doc["meta"]["sim_truth"], "correction", doc["layout_hash"])
    target = store.get_document(add_synth(ns, 50, "B", seed0=500, enqueue=False))
    chosen = active_learning.select_examples(ns, target["layout_hash"], k=2, exclude_document_id=target["id"])
    assert len(chosen) == 2 and {e.document_id for e in chosen} <= set(b)      # template-A example never offered
    assert active_learning.select_examples(ns + "-other", target["layout_hash"]) == []


def test_extractor_receives_the_selected_examples(ns):
    d = [add_synth(ns, i, "B", seed0=700) for i in range(3)]
    for x in d[:2]:
        doc = store.get_document(x)
        store.upsert_example(ns, x, doc["meta"]["sim_truth"], "correction", doc["layout_hash"])

    seen = []

    class Spy(SimulatedExtractor):
        def extract(self, req):
            seen.append([e.document_id for e in req.examples])
            return super().extract(req)

    pipeline.extract_document(store.get_document(d[2]), Spy())
    assert seen and sorted(seen[0]) == sorted(d[:2])
