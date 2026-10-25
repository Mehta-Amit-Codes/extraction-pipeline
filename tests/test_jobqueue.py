import random
import threading

from app import db, jobqueue as q


T0 = 100.0   # tests drive the queue with an explicit clock: enqueue() and claim() must agree on "now"


def enq(document_id, ns, **kw):
    return q.enqueue(document_id, ns, now=kw.pop("now", T0), **kw)


def enqueue_n(ns, n, **kw):
    return [enq(1000 + i, ns, **kw)[0] for i in range(n)]


def job(job_id):
    with db.conn() as c:
        return dict(c.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone())


def test_enqueue_is_idempotent_while_a_job_is_active(ns):
    j1, created1 = enq(1, ns)
    j2, created2 = enq(1, ns)
    assert j1 == j2 and created1 and not created2
    c = q.claim("w", now=1e9)
    with db.tx() as con:
        q.mark_done(con, c["id"], 1e9)
    _, created3 = enq(1, ns)                 # after completion a fresh job is allowed (reprocessing)
    assert created3


def test_claim_orders_by_priority_then_fifo_and_respects_next_attempt_time(ns):
    low = enq(1, ns, priority=0, now=100)[0]
    high = enq(2, ns, priority=5, now=101)[0]
    later = enq(3, ns, priority=9, now=500)[0]       # not due yet
    first, second = q.claim("w", now=200), q.claim("w", now=200)
    assert [first["id"], second["id"]] == [high, low]
    assert q.claim("w", now=200) is None                   # `later` is not due
    assert q.claim("w", now=600)["id"] == later


def test_attempts_are_counted_at_claim_time(ns):
    jid = enq(1, ns)[0]
    claimed = q.claim("w", now=1e9)
    assert claimed["attempts"] == 1 and claimed["status"] == "running" and claimed["lease_until"] > 1e9
    assert job(jid)["attempts"] == 1


def test_concurrent_workers_never_claim_the_same_job(ns):
    ids = set(enqueue_n(ns, 200))
    claimed, lock = [], threading.Lock()

    def run(name):
        while (j := q.claim(name)) is not None:
            with lock:
                claimed.append(j["id"])

    threads = [threading.Thread(target=run, args=(f"w{i}",)) for i in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(claimed) == 200 and set(claimed) == ids     # every job exactly once


def test_expired_lease_is_reclaimed_and_keeps_its_attempt_count(ns):
    jid = enq(1, ns)[0]
    q.claim("crashy", now=100, lease_s=10)
    assert q.reclaim_expired(now=105) == {"requeued": 0, "dead": 0}      # lease still valid
    assert q.reclaim_expired(now=200) == {"requeued": 1, "dead": 0}
    j = job(jid)
    assert j["status"] == "queued" and j["attempts"] == 1 and j["last_error_kind"] == "crash"


def test_a_poison_document_that_crashes_every_worker_still_ends_up_dead(ns):
    """No failure handler ever runs for a crash, so attempts must be charged at claim time."""
    jid = enq(1, ns, max_attempts=3)[0]
    t = 1000.0
    for _ in range(3):
        assert q.claim("w", now=t, lease_s=10)["id"] == jid
        t += 100
        q.reclaim_expired(now=t)
    assert job(jid)["status"] == "dead" and job(jid)["last_error_kind"] == "crash"
    assert q.claim("w", now=t + 1) is None


def test_retryable_failure_backs_off_then_dead_letters_at_max_attempts(ns):
    jid = enq(1, ns, max_attempts=2)[0]
    q.claim("w", now=100)
    assert q.fail(jid, "w", "boom", "transient", now=100) == "queued"
    assert job(jid)["next_attempt_at"] > 100 and job(jid)["last_error"] == "boom"
    q.claim("w", now=1e6)
    assert q.fail(jid, "w", "boom again", "transient", now=1e6) == "dead"
    assert job(jid)["status"] == "dead" and job(jid)["finished_at"] == 1e6


def test_permanent_failure_dead_letters_immediately(ns):
    jid = enq(1, ns, max_attempts=5)[0]
    q.claim("w", now=100)
    assert q.fail(jid, "w", "unsupported", "unsupported", retryable=False, now=100) == "dead"


def test_retry_after_is_a_floor_for_the_next_attempt(ns):
    jid = enq(1, ns, max_attempts=5)[0]
    q.claim("w", now=100)
    q.fail(jid, "w", "429", "rate_limit", retry_after=120, now=100)
    assert job(jid)["next_attempt_at"] >= 220


def test_backoff_grows_exponentially_with_bounded_jitter_and_a_cap():
    rng = random.Random(1)
    d = [q.backoff_delay(a, rng=rng) for a in (1, 2, 3, 4)]
    assert 0.01 * 0.75 <= d[0] <= 0.01 * 1.25 and d[2] > d[0]
    assert max(q.backoff_delay(30, rng=rng) for _ in range(50)) <= 0.05 * 1.25   # capped (test env: 0.05s)


def test_a_worker_that_lost_its_lease_cannot_complete_or_fail_the_job(ns):
    jid = enq(1, ns, max_attempts=5)[0]
    q.claim("slow", now=100, lease_s=10)
    q.reclaim_expired(now=200)
    q.claim("fast", now=200)
    assert q.fail(jid, "slow", "late failure", "transient", now=210) is None        # fenced out
    with db.tx() as c:
        assert not q.owns(c, jid, "slow") and q.owns(c, jid, "fast")


def test_release_does_not_consume_an_attempt(ns):
    jid = enq(1, ns)[0]
    q.claim("w", now=100)
    assert q.release(jid, "w", "bad api key", delay_s=5, now=100)
    j = job(jid)
    assert j["status"] == "queued" and j["attempts"] == 0 and j["last_error_kind"] == "config" and j["next_attempt_at"] == 105


def test_requeue_dead_resets_the_attempt_budget(ns):
    jid = enq(1, ns, max_attempts=1)[0]
    q.claim("w", now=100)
    q.fail(jid, "w", "x", "transient", now=100)
    assert job(jid)["status"] == "dead"
    assert q.requeue_dead(ns, now=200) == 1
    j = job(jid)
    assert j["status"] == "queued" and j["attempts"] == 0


def test_requeue_dead_can_target_specific_jobs_and_namespaces(ns):
    a, b = enq(1, ns, max_attempts=1)[0], enq(2, ns, max_attempts=1)[0]
    for _ in range(2):
        j = q.claim("w", now=100)
        q.fail(j["id"], "w", "x", "transient", retryable=False, now=100)
    assert q.requeue_dead("another-namespace") == 0
    assert q.requeue_dead(ns, job_ids=[a]) == 1 and job(b)["status"] == "dead"


def test_stats(ns):
    ids = enqueue_n(ns, 4)
    c1 = q.claim("w", now=1e9)
    c2 = q.claim("w", now=1e9)
    q.fail(c1["id"], "w", "x", "transient", now=1e9)       # -> queued, retrying
    with db.tx() as c:
        q.mark_done(c, c2["id"], 1e9)
    s = q.stats(ns, now=1e9 + 30)
    assert s["queued"] == 3 and s["running"] == 0 and s["done"] == 1 and s["retrying"] == 1 and s["dead"] == 0
    assert q.stats("nobody")["queued"] == 0
