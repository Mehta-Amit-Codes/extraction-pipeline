import csv
import io
import json
import threading
import time

import pytest
from fastapi.testclient import TestClient

from app import db, main, store, synth, worker
from tests.helpers import drain

client = TestClient(main.app)


def png(i=0, template="A"):
    return synth.generate_document(i, template)[0]


def upload(ns, files):
    return client.post("/documents", data={"namespace": ns}, files=[("files", f) for f in files])


def synthetic(ns, **kw):
    r = client.post("/synthetic", json={"namespace": ns, "seed0": kw.pop("seed0", 100), **kw})
    assert r.status_code == 200, r.text
    return r.json()


# ------------------------------------------------------------------- uploads ----
def test_upload_queues_new_files_flags_duplicates_and_rejects_non_documents(ns):
    good = png(1)
    r = upload(ns, [("a.png", good, "image/png"), ("b.png", good, "image/png"), ("evil.png", b"MZ\x90\x00 not an image", "image/png"),
                    ("notes.txt", b"hello", "text/plain")]).json()
    assert [x["status"] for x in r["results"]] == ["queued", "duplicate", "rejected", "rejected"]
    assert r["queued"] == 1 and "unsupported" in r["results"][2]["reason"]       # the .png extension does not fool the sniffer
    assert client.get("/queue/stats", params={"namespace": ns}).json()["queued"] == 1


def test_oversized_uploads_are_refused(ns, monkeypatch):
    monkeypatch.setattr(main, "MAX_UPLOAD_MB", 0)
    r = upload(ns, [("a.png", png(2), "image/png")]).json()
    assert r["results"][0]["status"] == "rejected" and "larger than" in r["results"][0]["reason"]


def test_same_document_in_two_namespaces_is_independent(ns):
    data = png(3)
    a, b = upload(ns, [("x.png", data, "image/png")]).json(), upload(ns + "-b", [("x.png", data, "image/png")]).json()
    assert a["queued"] == b["queued"] == 1


# ----------------------------------------------------------------- end to end ----
def test_batch_flows_from_synthetic_upload_to_review_to_export(ns):
    info = synthetic(ns, n=24, seed0=200)
    assert info["queued"] == 24
    drain()
    stats = client.get("/queue/stats", params={"namespace": ns}).json()
    assert stats["done"] == 24 and stats["queued"] == stats["running"] == 0

    pend = client.get("/reviews/pending", params={"namespace": ns, "limit": 100}).json()
    assert pend and {"id", "record", "field_conf", "issues", "review_reasons", "pass1"} <= set(pend[0])
    first, second = pend[0], pend[1]
    ok = client.post(f"/reviews/{first['id']}", json={"action": "approve", "namespace": ns, "reviewer": "ann"})
    assert ok.status_code == 200 and ok.json()["status"] == "approved"
    assert client.post(f"/reviews/{first['id']}", json={"action": "approve", "namespace": ns}).status_code == 409      # already reviewed
    bad = client.post(f"/reviews/{second['id']}", json={"action": "edit", "namespace": ns, "record": {**second["record"], "total": "ten"}})
    assert bad.status_code == 422
    assert client.post(f"/reviews/{second['id']}", json={"action": "edit", "namespace": ns}).status_code == 422
    rej = client.post(f"/reviews/{second['id']}", json={"action": "reject", "namespace": ns})
    assert rej.json()["status"] == "rejected"

    lines = [json.loads(l) for l in client.get("/export", params={"namespace": ns}).text.splitlines()]
    assert {l["status"] for l in lines} <= {"auto_approved", "approved", "corrected"}
    assert first["document_id"] in {l["document_id"] for l in lines} and second["document_id"] not in {l["document_id"] for l in lines}

    rows = list(csv.reader(io.StringIO(client.get("/export", params={"namespace": ns, "format": "csv"}).text)))
    assert rows[0][:3] == ["document", "status", "vendor_name"] and rows[0][-1] == "line_items_json" and len(rows) - 1 == len(lines)
    assert json.loads(rows[1][-1]) is not None


def test_quality_drift_alert_and_example_endpoints_respond(ns):
    synthetic(ns, n=20, seed0=300)
    drain()
    client.post("/demo/oracle-review", json={"namespace": ns, "limit": 50})
    q = client.get("/quality", params={"namespace": ns}).json()
    assert q["processed"] == 20 and "calibration" in q and q["cost"]["cost_usd"] > 0
    assert client.get("/drift", params={"namespace": ns}).json()["status"] in ("ok", "insufficient_data")
    assert isinstance(client.get("/drift/timeseries", params={"namespace": ns, "step": 5}).json(), list)
    assert client.get("/drift/simulation", params={"namespace": ns}).json()["total_processed"] == 20
    assert client.get("/alerts", params={"namespace": ns}).json() == []
    ex = client.get("/examples", params={"namespace": ns}).json()
    assert ex["summary"]["examples"] >= 1
    eid = ex["examples"][0]["id"]
    assert client.post(f"/examples/{eid}/active", json={"namespace": ns, "active": False}).status_code == 200
    manifest = client.get("/examples/export", params={"namespace": ns}).text
    assert all(json.loads(l)["messages"] for l in manifest.splitlines() if l)
    assert client.post("/alerts/999999/ack", params={"namespace": ns}).status_code == 404
    assert client.post("/drift/baseline/pin", params={"namespace": ns}).json()["baseline_end"] == 20


def test_document_images_are_served_per_namespace(ns):
    synthetic(ns, n=2, seed0=400)
    doc = store.list_documents(ns)[0]
    r = client.get(f"/documents/{doc['id']}/image", params={"namespace": ns})
    assert r.status_code == 200 and r.content.startswith(b"\x89PNG")
    assert client.get(f"/documents/{doc['id']}/image", params={"namespace": ns + "-other"}).status_code == 404


def test_namespaces_do_not_leak_through_any_read_endpoint(ns):
    synthetic(ns, n=15, seed0=500)
    drain()
    other = ns + "-intruder"
    assert client.get("/reviews/pending", params={"namespace": other}).json() == []
    assert client.get("/export", params={"namespace": other}).text == ""
    assert client.get("/jobs", params={"namespace": other}).json() == []
    assert client.get("/examples", params={"namespace": other}).json()["examples"] == []
    assert client.get("/quality", params={"namespace": other}).json()["processed"] == 0
    pend = client.get("/reviews/pending", params={"namespace": ns}).json()
    assert pend and client.post(f"/reviews/{pend[0]['id']}", json={"action": "approve", "namespace": other}).status_code == 409


def test_an_api_key_is_enforced_when_configured(monkeypatch):
    monkeypatch.setattr(main, "API_KEY", "s3cret")
    assert client.get("/queue/stats").status_code == 401
    assert client.get("/queue/stats", headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get("/queue/stats", headers={"X-API-Key": "s3cret"}).status_code == 200


# ------------------------------------------------------------- operator controls ----
def test_pause_and_resume_are_visible_to_the_whole_system(ns):
    synthetic(ns, n=4, seed0=600)
    assert client.post("/queue/pause").json()["paused"]
    assert client.get("/queue/stats").json()["paused"] is True and drain()["done"] == 0
    client.post("/queue/resume")
    assert client.get("/queue/stats").json()["paused"] is False and drain()["done"] == 4


def test_dead_letters_can_be_inspected_and_requeued_after_the_cause_is_fixed(ns):
    synthetic(ns, n=6, seed0=700, poison_every=3)
    drain()
    dead = client.get("/jobs", params={"namespace": ns, "status": "dead"}).json()
    assert len(dead) == 2 and all(j["last_error_kind"] == "permanent" and j["filename"] for j in dead)
    assert client.post("/jobs/requeue-dead", json={"namespace": ns, "job_ids": [dead[0]["id"]]}).json()["requeued"] == 1
    assert client.get("/queue/stats", params={"namespace": ns}).json()["queued"] == 1
    assert client.post("/jobs/requeue-dead", json={"namespace": ns}).json()["requeued"] == 1


def test_transient_failures_in_a_batch_recover_by_themselves(ns):
    synthetic(ns, n=10, seed0=800, fail_every=2)
    counts = drain()
    assert counts["retry"] == 5 and counts["done"] == 10 and counts["dead"] == 0


def test_reprocess_creates_a_new_extraction_version(ns):
    synthetic(ns, n=3, seed0=900)
    drain()
    docs = [d["id"] for d in store.list_documents(ns)]
    assert client.post("/reprocess", json={"namespace": ns, "document_ids": docs + [10 ** 9]}).json()["queued"] == 3
    drain()
    with db.conn() as c:
        assert c.execute("SELECT MAX(version) AS v FROM extractions WHERE namespace = ?", (ns,)).fetchone()["v"] == 2


# -------------------------------------------------------------------- workers ----
def test_token_bucket_throttles_calls_across_threads():
    bucket = worker.TokenBucket(20)
    t0 = time.monotonic()
    ts = [threading.Thread(target=lambda: [bucket.acquire() for _ in range(10)]) for _ in range(4)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert time.monotonic() - t0 >= 0.8        # 40 calls: 20 from the full bucket, 20 more at 20/s
    start = time.monotonic()
    worker.TokenBucket(0).acquire()
    assert time.monotonic() - start < 0.05      # rate 0 = unlimited


def test_pool_recovers_jobs_from_a_dead_worker_via_lease_reclaim(ns):
    from app import jobqueue
    synthetic(ns, n=3, seed0=1000)
    crashed = jobqueue.claim("crashed-worker", lease_s=0)       # claimed, then the process "died"
    assert crashed and client.get("/queue/stats", params={"namespace": ns}).json()["running"] == 1
    time.sleep(0.05)
    counts = drain(reclaim_every_s=0.0)
    assert counts["done"] == 3 and client.get("/queue/stats", params={"namespace": ns}).json()["done"] == 3


def test_health():
    assert client.get("/health").json()["status"] == "ok"


def test_document_listing_is_scoped_and_exposes_the_simulation_template(ns):
    synthetic(ns, n=4, change_at=2, seed0=1100)
    docs = client.get("/documents", params={"namespace": ns}).json()
    assert len(docs) == 4 and {d["template"] for d in docs} == {"A", "B"} and all(d["layout_hash"] for d in docs)
    assert client.get("/documents", params={"namespace": ns + "-other"}).json() == []


def test_config_endpoint_exposes_only_non_secret_settings():
    cfg = client.get("/config").json()
    assert {"baseline_n", "window_n", "min_samples", "extractor", "review_threshold"} <= set(cfg)
    assert not any("key" in k or "secret" in k or "webhook" in k for k in cfg)
