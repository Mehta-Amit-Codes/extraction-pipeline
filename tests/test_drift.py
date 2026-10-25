import random
import threading
import uuid

import pytest
import requests

from app import drift, imaging, store, synth
from tests.helpers import add_stream, drain


# ------------------------------------------------------------------ statistics ----
def test_fisher_matches_a_hand_computed_value():
    # [[1, 9], [9, 1]] : P(>= 9 of the 10 successes land in window 2) = (C(10,9)C(10,1) + C(10,10)C(10,0)) / C(20,10)
    assert drift.fisher_one_sided(1, 10, 9, 10) == pytest.approx(101 / 184756)


def test_fisher_edge_cases_and_monotonicity():
    assert drift.fisher_one_sided(0, 10, 0, 10) == 1.0 and drift.fisher_one_sided(3, 0, 3, 10) == 1.0
    assert drift.fisher_one_sided(5, 10, 5, 10) >= 0.5                       # no difference: nothing to see
    assert drift.fisher_one_sided(0, 30, 3, 30) > drift.fisher_one_sided(0, 30, 8, 30) > drift.fisher_one_sided(0, 30, 20, 30)
    assert drift.fisher_one_sided(8, 30, 2, 30) > 0.9                         # an improvement is not drift


def test_fisher_is_exact_where_a_z_test_would_not_be_reliable():
    # baseline rate 0/60, current 4/60: z-approximations are shaky at these counts; the exact tail is well-defined
    p = drift.fisher_one_sided(0, 60, 4, 60)
    assert 0.03 < p < 0.08


def test_welch_detects_a_shift_only_in_the_bad_direction():
    assert drift.welch_one_sided(0.90, 0.05, 40, 0.80, 0.05, 40, "down") < 1e-6
    assert drift.welch_one_sided(0.90, 0.05, 40, 0.95, 0.05, 40, "down") > 0.99
    assert drift.welch_one_sided(0.05, 0.02, 40, 0.15, 0.02, 40, "up") < 1e-6
    assert drift.welch_one_sided(0.9, 0.0, 40, 0.8, 0.0, 40, "down") == 0.0      # zero variance, real shift
    assert 0.3 < drift.welch_one_sided(0.9, 0.05, 40, 0.9, 0.05, 40, "down") < 0.7


def test_wilson_interval():
    assert drift.wilson(0, 0) == (0.0, 0.0)
    lo, hi = drift.wilson(50, 100)
    assert lo < 0.5 < hi and hi - lo < 0.21 and lo == pytest.approx(1 - hi, abs=1e-9)
    assert drift.wilson(0, 10)[0] == 0.0 and drift.wilson(10, 10)[1] == 1.0
    assert drift.wilson(5, 10)[1] - drift.wilson(5, 10)[0] > drift.wilson(500, 1000)[1] - drift.wilson(500, 1000)[0]


def test_incr_kv_is_atomic_across_threads():
    key = f"counter-{uuid.uuid4().hex}"
    ts = [threading.Thread(target=lambda: [store.incr_kv(key) for _ in range(50)]) for _ in range(8)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert store.get_kv(key) == "400"


# ---------------------------------------------------------------- layout novelty ----
def h(template, i, noise=0.0):
    return imaging.layout_hash(imaging.load_images(synth.generate_document(i, template, noise)[0])[0])


def test_novelty_threshold_is_derived_from_the_baseline_itself():
    assert drift.novelty_threshold(["0" * 16] * 10) == drift.LAYOUT_MIN_T                  # identical docs: the floor
    rng = random.Random(1)
    scattered = [f"{rng.getrandbits(64):016x}" for _ in range(12)]                         # layouts genuinely far apart
    assert drift.novelty_threshold(scattered) > 3 * drift.LAYOUT_MIN_T
    assert drift.novelty_threshold(["0" * 16] * 3) == drift.LAYOUT_MIN_T                   # too few to estimate


def test_new_documents_of_a_known_template_are_not_novel_but_a_new_template_is():
    base = [h("A", i) for i in range(30)]
    thr = drift.novelty_threshold(base)
    same = [h("A", 200 + i) for i in range(20)] + [h("A", 300 + i, noise=1.0) for i in range(5)]
    other = [h("B", 400 + i) for i in range(20)]
    assert sum(drift.novel_flags(same, base, thr)) == 0
    assert all(drift.novel_flags(other, base, thr))
    assert drift.novel_flags(["f" * 16], [], 6) == [False]                                  # no baseline: nothing is "novel"


# ------------------------------------------------------------- end-to-end streams ----
@pytest.fixture(scope="module")
def stable():
    n = f"stable-{uuid.uuid4().hex[:6]}"
    add_stream(n, 70, seed0=6000)
    drain()
    return n


@pytest.fixture(scope="module")
def drifted():
    n = f"drift-{uuid.uuid4().hex[:6]}"
    add_stream(n, 90, change_at=45, seed0=7000)
    drain()
    return n


def test_a_stable_stream_raises_no_alerts(stable):
    assert store.list_alerts(stable) == []
    r = drift.evaluate(stable, persist=False)
    assert r["status"] == "ok" and not any(x["significant"] for x in r["results"])


def test_a_format_change_raises_a_label_free_layout_alert_and_proxy_alerts(drifted):
    metrics = {a["metric"] for a in store.list_alerts(drifted)}
    assert "novel_layout_rate" in metrics and len(metrics) >= 2
    novel = next(a for a in store.list_alerts(drifted) if a["metric"] == "novel_layout_rate")
    assert novel["baseline"] < 0.05 and novel["current"] > 0.5 and novel["p_value"] < 0.01 and novel["status"] == "open"


def test_alerts_fire_after_the_change_and_never_before_it(drifted):
    sim = drift.simulation_report(drifted)
    # processing order jitters a few places under concurrency, so "the change" is the first template-B document
    # to FINISH, which lands near (not exactly at) its submission index of 46
    assert abs(sim["change_at_index"] - 46) <= 5 and sim["false_alarms_before_change"] == 0
    assert sim["first_alert_overall"] > sim["change_at_index"]
    lag = sim["detection_lag_docs"]["novel_layout_rate"]
    assert 0 < lag <= 25                                   # prompt: the layout signal needs no model output
    assert min(sim["detection_lag_docs"].values()) == lag or lag <= min(sim["detection_lag_docs"].values()) + 10


def test_evaluation_without_persist_changes_nothing_and_repeats_dedupe(drifted):
    before = len(store.list_alerts(drifted))
    drift.evaluate(drifted, persist=False)
    assert len(store.list_alerts(drifted)) == before
    again = drift.evaluate(drifted)                       # persisting again: open alerts are updated, not duplicated
    assert again["alerts"] == [] and len(store.list_alerts(drifted)) == before


def test_timeseries_shows_the_accuracy_collapse_and_the_novelty_spike(drifted):
    ts = drift.timeseries(drifted, step=10)
    assert [p["index"] for p in ts] == sorted(p["index"] for p in ts) and ts[-1]["index"] == 90
    assert ts[0]["novel_layout_rate"] == 0.0 and ts[-1]["novel_layout_rate"] > 0.9
    assert ts[-1]["review_rate"] > ts[0]["review_rate"] and ts[-1]["mean_confidence"] < ts[0]["mean_confidence"]
    assert ts[0]["oracle_accuracy"] > ts[-1]["oracle_accuracy"] + 0.05      # simulation-only ground truth


def test_insufficient_data_is_reported_not_guessed(ns):
    add_stream(ns, 10, seed0=8000)
    drain()
    r = drift.evaluate(ns)
    assert r["status"] == "insufficient_data" and r["alerts"] == []


def test_acknowledging_an_alert_closes_it_and_a_still_drifting_stream_re_alerts(drifted):
    ns = drifted
    for a in store.list_alerts(ns, "open"):
        assert store.ack_alert(ns, a["id"])
    assert store.list_alerts(ns, "open") == [] and not store.ack_alert(ns, 10 ** 9)
    r = drift.evaluate(ns)                                 # the stream is still drifting: a fresh alert is the right outcome
    assert r["alerts"] and store.list_alerts(ns, "open")


def test_pinning_the_new_normal_as_the_baseline_silences_the_alarm():
    n = f"pin-{uuid.uuid4().hex[:6]}"
    add_stream(n, 90, change_at=45, seed0=9000)
    drain()
    assert store.list_alerts(n, "open")
    pin = drift.pin_baseline(n)
    assert pin["alerts_acknowledged"] > 0 and pin["baseline_end"] == 90 and store.list_alerts(n, "open") == []
    add_stream(n, 30, change_at=0, seed0=9500)             # keep processing template B
    drain()
    r = drift.evaluate(n)
    assert r["status"] == "ok" and not any(x["significant"] for x in r["results"])     # B is the baseline now


def test_per_document_novelty_hint_uses_a_bounded_query(drifted):
    docs = store.list_documents(drifted, limit=2)
    assert len(drift.baseline_hashes(drifted)) == drift.BASELINE_N
    assert drift.is_novel(docs[0]) is True                 # the most recent document is template B


def test_alert_webhook_is_called_and_a_failing_webhook_never_breaks_extraction(monkeypatch):
    sent = []
    monkeypatch.setattr(drift, "ALERT_WEBHOOK_URL", "https://hooks.example/test")
    monkeypatch.setattr(drift.requests, "post", lambda url, json=None, timeout=None: sent.append((url, json)))
    drift.notify("acme", "critical", "review_rate moved from 5.0% to 60.0%")
    assert sent[0][0] == "https://hooks.example/test"
    assert "CRITICAL" in sent[0][1]["text"] and "acme" in sent[0][1]["text"]

    def boom(*a, **k):
        raise requests.ConnectionError("slack is down")
    monkeypatch.setattr(drift.requests, "post", boom)
    drift.notify("acme", "warning", "x")                    # must not raise


def test_notify_is_a_no_op_without_a_webhook(monkeypatch):
    monkeypatch.setattr(drift, "ALERT_WEBHOOK_URL", "")
    monkeypatch.setattr(drift.requests, "post", lambda *a, **k: pytest.fail("should not be called"))
    drift.notify("acme", "warning", "x")


def test_a_format_change_inside_the_baseline_window_is_flagged_instead_of_silently_blinding_detection():
    """Regression from a live run: the UI simulation put the change at #40 with a 60-document baseline, so a third of the
    baseline was already the new layout. Detection quietly went blind (no layout alert, 59-document lag)."""
    n = f"contaminated-{uuid.uuid4().hex[:6]}"
    add_stream(n, 60, change_at=20, seed0=12000)         # test baseline is 30 docs: 20 x A then 10 x B
    drain()
    r = drift.evaluate(n, persist=False)
    assert r["status"] == "ok" and r["baseline_stable"] is False
    assert any("novel_layout_rate" in w or "review_rate" in w for w in r["baseline_warnings"])
    assert not any(x["metric"] == "novel_layout_rate" and x["significant"] for x in r["results"])   # the blindness the warning explains


def test_a_clean_baseline_is_reported_stable(stable):
    r = drift.evaluate(stable, persist=False)
    assert r["baseline_stable"] is True and r["baseline_warnings"] == []


def test_stability_is_unknown_when_the_baseline_is_too_small_to_judge():
    assert drift.baseline_stability([{"layout_hash": "0" * 16, "needs_review": 0, "n_issues": 0, "crosschecked": 1, "n_disputed": 0,
                                      "mean_conf": 0.9, "n_null": 0}] * 10) == (None, [])
