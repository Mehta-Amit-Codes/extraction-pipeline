"""
Durable job queue with retries and a dead-letter state.

Semantics worth stating precisely, because they are where batch pipelines usually go wrong:

* AT-LEAST-ONCE delivery, effectively-once results. A claim takes a LEASE. If the worker dies, the lease
  expires and the job is reclaimed. A slow worker that finishes AFTER its lease was reclaimed must not
  write a second result, so completion is fenced: it only succeeds if this worker still owns the job,
  and the extraction row is written in the SAME transaction as the completion (see pipeline.py).
* Attempts are counted at CLAIM time, not on failure. A "poison" document that crashes the worker
  process never reaches a failure handler, but it still burns an attempt each time its lease expires,
  so it ends up dead-lettered instead of crash-looping forever.
* Retryable failures (rate limit, timeout, 5xx, malformed model output) back off exponentially with jitter
  and honour Retry-After. Permanent failures (unsupported file) go straight to dead.
* Config failures (bad API key, unknown model) do NOT consume an attempt: otherwise one expired key would
  dump the entire backlog into the dead-letter queue. The job is released and the worker pauses
  (see worker.py).
"""
import random
import time

from app import db
from app.config import BACKOFF_BASE_S, BACKOFF_CAP_S, LEASE_S, MAX_ATTEMPTS

ACTIVE = ("queued", "running")


def enqueue(document_id: int, namespace: str, priority: int = 0, max_attempts: int | None = None,
            now: float | None = None) -> tuple[int, bool]:
    """Idempotent per document: at most one ACTIVE job. Returns (job_id, created)."""
    now = now or time.time()
    with db.tx() as c:
        row = c.execute("SELECT id FROM jobs WHERE document_id = ? AND status IN ('queued','running')",
                        (document_id,)).fetchone()
        if row:
            return row["id"], False
        cur = c.execute(
            """INSERT INTO jobs (document_id, namespace, status, attempts, max_attempts, priority,
               next_attempt_at, created_at, updated_at) VALUES (?, ?, 'queued', 0, ?, ?, ?, ?, ?)""",
            (document_id, namespace, max_attempts or MAX_ATTEMPTS, priority, now, now, now))
        return cur.lastrowid, True


def claim(worker_id: str, now: float | None = None, lease_s: int = LEASE_S) -> dict | None:
    now = now or time.time()
    with db.tx() as c:
        row = c.execute(
            """SELECT id FROM jobs WHERE status = 'queued' AND next_attempt_at <= ?
               ORDER BY priority DESC, next_attempt_at, id LIMIT 1""", (now,)).fetchone()
        if row is None:
            return None
        c.execute("""UPDATE jobs SET status = 'running', attempts = attempts + 1, lease_until = ?,
                     worker_id = ?, updated_at = ? WHERE id = ?""", (now + lease_s, worker_id, now, row["id"]))
        return dict(c.execute("SELECT * FROM jobs WHERE id = ?", (row["id"],)).fetchone())


def owns(c, job_id: int, worker_id: str) -> bool:
    """Fencing check, run inside the caller's transaction."""
    row = c.execute("SELECT status, worker_id FROM jobs WHERE id = ?", (job_id,)).fetchone()
    return bool(row) and row["status"] == "running" and row["worker_id"] == worker_id


def mark_done(c, job_id: int, now: float) -> None:
    c.execute("""UPDATE jobs SET status = 'done', finished_at = ?, updated_at = ?, lease_until = NULL,
                 last_error = NULL WHERE id = ?""", (now, now, job_id))


def backoff_delay(attempts: int, retry_after: float | None = None, rng: random.Random | None = None) -> float:
    """base * 2^(attempts-1), capped, with +-25% jitter (so a burst of failures doesn't retry in lockstep).
    A server-supplied Retry-After is a floor: we never retry sooner than the API asked."""
    rng = rng or random
    delay = min(BACKOFF_CAP_S, BACKOFF_BASE_S * (2 ** max(0, attempts - 1))) * rng.uniform(0.75, 1.25)
    return max(delay, retry_after or 0.0)


def fail(job_id: int, worker_id: str, error: str, kind: str, retryable: bool = True,
         retry_after: float | None = None, now: float | None = None) -> str | None:
    """Record a failure. Returns the new status ('queued' or 'dead'), or None if this worker lost the job."""
    now = now or time.time()
    with db.tx() as c:
        if not owns(c, job_id, worker_id):
            return None
        row = c.execute("SELECT attempts, max_attempts FROM jobs WHERE id = ?", (job_id,)).fetchone()
        dead = (not retryable) or row["attempts"] >= row["max_attempts"]
        if dead:
            c.execute("""UPDATE jobs SET status = 'dead', finished_at = ?, updated_at = ?, lease_until = NULL,
                         last_error = ?, last_error_kind = ? WHERE id = ?""", (now, now, error[:1000], kind, job_id))
            return "dead"
        c.execute("""UPDATE jobs SET status = 'queued', next_attempt_at = ?, updated_at = ?, lease_until = NULL,
                     last_error = ?, last_error_kind = ? WHERE id = ?""",
                  (now + backoff_delay(row["attempts"], retry_after), now, error[:1000], kind, job_id))
        return "queued"


def release(job_id: int, worker_id: str, error: str, delay_s: float = 5.0, now: float | None = None) -> bool:
    """Give the job back WITHOUT consuming an attempt (config problem, not the document's fault)."""
    now = now or time.time()
    with db.tx() as c:
        if not owns(c, job_id, worker_id):
            return False
        c.execute("""UPDATE jobs SET status = 'queued', attempts = MAX(0, attempts - 1), next_attempt_at = ?,
                     updated_at = ?, lease_until = NULL, last_error = ?, last_error_kind = 'config' WHERE id = ?""",
                  (now + delay_s, now, error[:1000], job_id))
        return True


def reclaim_expired(now: float | None = None) -> dict:
    """Recover jobs whose worker died. Attempts were already counted at claim, so a crash loop terminates."""
    now = now or time.time()
    with db.tx() as c:
        rows = c.execute("SELECT id, attempts, max_attempts FROM jobs WHERE status = 'running' AND lease_until < ?",
                         (now,)).fetchall()
        requeued = dead = 0
        for r in rows:
            if r["attempts"] >= r["max_attempts"]:
                c.execute("""UPDATE jobs SET status = 'dead', finished_at = ?, updated_at = ?, lease_until = NULL,
                             last_error = 'lease expired repeatedly (worker crash or timeout)',
                             last_error_kind = 'crash' WHERE id = ?""", (now, now, r["id"]))
                dead += 1
            else:
                c.execute("""UPDATE jobs SET status = 'queued', next_attempt_at = ?, updated_at = ?, lease_until = NULL,
                             last_error = 'lease expired (worker crash or timeout)', last_error_kind = 'crash'
                             WHERE id = ?""", (now, now, r["id"]))
                requeued += 1
    return {"requeued": requeued, "dead": dead}


def requeue_dead(namespace: str | None = None, job_ids: list[int] | None = None, now: float | None = None) -> int:
    """Operator action after fixing the cause: dead -> queued with a fresh attempt budget."""
    now = now or time.time()
    sql, params = "UPDATE jobs SET status = 'queued', attempts = 0, next_attempt_at = ?, updated_at = ?, finished_at = NULL WHERE status = 'dead'", [now, now]
    if namespace:
        sql += " AND namespace = ?"
        params.append(namespace)
    if job_ids:
        sql += f" AND id IN ({','.join('?' * len(job_ids))})"
        params += job_ids
    with db.tx() as c:
        return c.execute(sql, params).rowcount


def stats(namespace: str | None = None, now: float | None = None) -> dict:
    now = now or time.time()
    where, params = ("WHERE namespace = ?", [namespace]) if namespace else ("", [])
    with db.conn() as c:
        counts = {r["status"]: r["n"] for r in c.execute(
            f"SELECT status, COUNT(*) AS n FROM jobs {where} GROUP BY status", params)}
        retrying = c.execute(
            f"SELECT COUNT(*) AS n FROM jobs {where + (' AND' if where else 'WHERE')} status = 'queued' AND attempts > 0",
            params).fetchone()["n"]
        oldest = c.execute(
            f"SELECT MIN(next_attempt_at) AS t FROM jobs {where + (' AND' if where else 'WHERE')} status = 'queued'",
            params).fetchone()["t"]
        recent = c.execute(
            f"SELECT COUNT(*) AS n FROM jobs {where + (' AND' if where else 'WHERE')} status = 'done' AND finished_at > ?",
            params + [now - 60]).fetchone()["n"]
    return {"queued": counts.get("queued", 0), "running": counts.get("running", 0), "done": counts.get("done", 0),
            "dead": counts.get("dead", 0), "retrying": retrying,
            "oldest_queued_age_s": max(0.0, now - oldest) if oldest else 0.0, "done_last_minute": recent}


def list_jobs(namespace: str, status: str | None = None, limit: int = 100) -> list[dict]:
    sql, params = "SELECT j.*, d.filename FROM jobs j JOIN documents d ON d.id = j.document_id WHERE j.namespace = ?", [namespace]
    if status:
        sql += " AND j.status = ?"
        params.append(status)
    with db.conn() as c:
        return [dict(r) for r in c.execute(sql + " ORDER BY j.updated_at DESC LIMIT ?", params + [limit])]
