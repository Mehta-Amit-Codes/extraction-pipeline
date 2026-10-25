"""
Worker pool. Threads suit this workload (it is I/O-bound on the model API); for more throughput run several
`python -m app.cli worker` processes: claims are atomic across processes, so they never double-process a job.

Operational safeguards:
  * RATE LIMIT      one shared token bucket caps model calls/sec across all threads (RATE_LIMIT_RPS), so a large
                    batch does not hammer the API into a wall of 429s.
  * LEASE RECLAIM   periodically recovers jobs from workers that died mid-flight.
  * CIRCUIT BREAKER N consecutive config errors (bad key, unknown model) pause the queue. Without it, an expired API
                    key turns thousands of healthy documents into failures in minutes.
  * PAUSE/RESUME    operator switch, stored in the DB so the API, CLI and every worker process agree.
"""
import logging
import threading
import time
import uuid

from app import jobqueue, pipeline, store
from app.config import CONFIG_ERROR_PAUSE_AFTER, RATE_LIMIT_RPS, RECLAIM_EVERY_S
from app.extractors import get_extractor

log = logging.getLogger("worker")


class TokenBucket:
    def __init__(self, rate_per_s: float):
        self.rate, self.tokens, self.t, self.lock = rate_per_s, rate_per_s, time.monotonic(), threading.Lock()

    def acquire(self) -> None:
        if self.rate <= 0:
            return
        while True:
            with self.lock:
                now = time.monotonic()
                self.tokens = min(self.rate, self.tokens + (now - self.t) * self.rate)
                self.t = now
                if self.tokens >= 1:
                    self.tokens -= 1
                    return
                wait = (1 - self.tokens) / self.rate
            time.sleep(wait)


def is_paused() -> bool:
    return store.get_kv("queue_paused", "0") == "1"


def pause(reason: str = "manual") -> None:
    store.set_kv("queue_paused", "1")
    store.set_kv("queue_pause_reason", reason)


def resume() -> None:
    store.set_kv("queue_paused", "0")
    store.set_kv("queue_pause_reason", "")


class WorkerPool:
    def __init__(self, concurrency: int = 2, extractor=None, poll_interval: float = 0.25, drain: bool = False,
                 rate_limit_rps: float = RATE_LIMIT_RPS, reclaim_every_s: float = RECLAIM_EVERY_S):
        self.bucket = TokenBucket(rate_limit_rps)
        self.extractor = extractor or get_extractor(before_call=self.bucket.acquire)
        self.concurrency, self.poll, self.drain, self.reclaim_every = concurrency, poll_interval, drain, reclaim_every_s
        self.stop_event = threading.Event()
        self.threads: list[threading.Thread] = []
        self.counts = {"done": 0, "retry": 0, "dead": 0, "released": 0, "lost": 0}
        self._lock = threading.Lock()
        self._config_streak = 0
        self._last_reclaim = 0.0

    def start(self) -> "WorkerPool":
        for i in range(self.concurrency):
            t = threading.Thread(target=self._loop, args=(f"w-{uuid.uuid4().hex[:6]}-{i}",), daemon=True, name=f"extract-{i}")
            t.start()
            self.threads.append(t)
        return self

    def stop(self) -> None:
        self.stop_event.set()

    def join(self, timeout: float | None = None) -> None:
        for t in self.threads:
            t.join(timeout)

    def _maybe_reclaim(self) -> None:
        with self._lock:
            if time.monotonic() - self._last_reclaim < self.reclaim_every:
                return
            self._last_reclaim = time.monotonic()
        r = jobqueue.reclaim_expired()
        if r["requeued"] or r["dead"]:
            log.warning("reclaimed expired leases: %s", r)

    def _record(self, outcome: str) -> None:
        with self._lock:
            self.counts[outcome] += 1
            self._config_streak = self._config_streak + 1 if outcome == "released" else 0
            trip = self._config_streak >= CONFIG_ERROR_PAUSE_AFTER
        if trip:
            pause("circuit breaker: repeated configuration errors (check the API key / model name)")
            log.error("queue paused by circuit breaker after %d consecutive config errors", CONFIG_ERROR_PAUSE_AFTER)

    def _loop(self, worker_id: str) -> None:
        while not self.stop_event.is_set():
            self._maybe_reclaim()
            if is_paused():
                if self.drain:
                    return  # batch mode must terminate even if the circuit breaker tripped mid-run
                time.sleep(self.poll)
                continue
            job = jobqueue.claim(worker_id)
            if job is None:
                if self.drain:
                    s = jobqueue.stats()
                    if s["queued"] + s["running"] == 0:
                        return
                time.sleep(self.poll)
                continue
            self._record(pipeline.process_job(job, self.extractor, worker_id))


def run_until_drained(concurrency: int = 4, **kw) -> dict:
    """Process everything queued, then return the outcome counts (CLI batch mode and tests)."""
    pool = WorkerPool(concurrency=concurrency, drain=True, **kw).start()
    pool.join()
    return pool.counts
