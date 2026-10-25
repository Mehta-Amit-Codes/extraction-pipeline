"""
SQLite: one durable store for documents, the job queue, extractions, reviews, the few-shot pool, alerts.

Why SQLite for the queue (honest trade-off): it is durable, transactional, zero-infra, and safe across
threads AND processes (claims use BEGIN IMMEDIATE, so two workers can never claim one job). It is a
single-node design. The queue is isolated behind app/jobqueue.py, so moving to Postgres
(SELECT ... FOR UPDATE SKIP LOCKED) or a broker (Celery + Redis/RabbitMQ) means re-implementing that
one module, not the pipeline.
"""
import sqlite3
from contextlib import contextmanager

from app.config import DATA_DIR, DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT, namespace TEXT NOT NULL, batch_id TEXT, filename TEXT,
    sha256 TEXT NOT NULL, mime TEXT, path TEXT NOT NULL, layout_hash TEXT, n_pages INTEGER,
    meta TEXT DEFAULT '{}', created_at REAL, UNIQUE (namespace, sha256));
CREATE INDEX IF NOT EXISTS idx_docs_batch ON documents(namespace, batch_id);

CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT, document_id INTEGER NOT NULL, namespace TEXT NOT NULL,
    status TEXT NOT NULL, attempts INTEGER DEFAULT 0, max_attempts INTEGER, priority INTEGER DEFAULT 0,
    next_attempt_at REAL, lease_until REAL, worker_id TEXT, last_error TEXT, last_error_kind TEXT,
    created_at REAL, updated_at REAL, finished_at REAL);
CREATE INDEX IF NOT EXISTS idx_jobs_claim ON jobs(status, next_attempt_at, priority);
CREATE INDEX IF NOT EXISTS idx_jobs_doc ON jobs(document_id);
CREATE INDEX IF NOT EXISTS idx_jobs_ns ON jobs(namespace, status);

CREATE TABLE IF NOT EXISTS extractions (
    id INTEGER PRIMARY KEY AUTOINCREMENT, document_id INTEGER NOT NULL, namespace TEXT NOT NULL,
    job_id INTEGER, version INTEGER, is_current INTEGER DEFAULT 1, created_at REAL,
    extractor TEXT, model TEXT, status TEXT, needs_review INTEGER, review_reasons TEXT, priority REAL DEFAULT 0,
    audit_sample INTEGER DEFAULT 0, crosschecked INTEGER DEFAULT 0,
    record TEXT, field_conf TEXT, pass1 TEXT, pass2 TEXT, disputed_fields TEXT, issues TEXT, fewshot_doc_ids TEXT,
    mean_conf REAL, min_conf REAL, n_disputed INTEGER, n_issues INTEGER, n_null INTEGER,
    tokens_in INTEGER, tokens_out INTEGER, cost_usd REAL, latency_ms REAL,
    oracle_field_acc REAL, oracle_exact INTEGER, final_record TEXT, reviewed_at REAL);
CREATE INDEX IF NOT EXISTS idx_ext_ns ON extractions(namespace, is_current, id);
CREATE INDEX IF NOT EXISTS idx_ext_review ON extractions(namespace, status, priority);
CREATE INDEX IF NOT EXISTS idx_ext_doc ON extractions(document_id);

CREATE TABLE IF NOT EXISTS reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT, extraction_id INTEGER NOT NULL, namespace TEXT NOT NULL,
    reviewer TEXT, action TEXT, before TEXT, after TEXT, changed_fields TEXT, audit_sample INTEGER DEFAULT 0,
    created_at REAL);
CREATE INDEX IF NOT EXISTS idx_reviews_ns ON reviews(namespace, created_at);

CREATE TABLE IF NOT EXISTS examples (
    id INTEGER PRIMARY KEY AUTOINCREMENT, namespace TEXT NOT NULL, document_id INTEGER NOT NULL,
    record TEXT NOT NULL, source TEXT, layout_hash TEXT, created_at REAL, active INTEGER DEFAULT 1,
    UNIQUE (namespace, document_id));
CREATE INDEX IF NOT EXISTS idx_examples_ns ON examples(namespace, active);

CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT, namespace TEXT NOT NULL, created_at REAL, last_seen_at REAL,
    metric TEXT, severity TEXT, message TEXT, baseline REAL, current REAL, p_value REAL,
    at_extraction_index INTEGER, status TEXT DEFAULT 'open');
CREATE INDEX IF NOT EXISTS idx_alerts_ns ON alerts(namespace, status, metric);

CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT);
"""


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=60, isolation_level=None)  # autocommit; explicit BEGIN where needed
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=60000")
    return conn


@contextmanager
def conn():
    c = _connect()
    try:
        yield c
    finally:
        c.close()


@contextmanager
def tx():
    """Write transaction. BEGIN IMMEDIATE takes the write lock up front, so read-then-write sequences
    (like claiming a job) are atomic across threads and processes."""
    c = _connect()
    try:
        c.execute("BEGIN IMMEDIATE")
        yield c
        c.execute("COMMIT")
    except BaseException:
        if c.in_transaction:
            c.execute("ROLLBACK")
        raise
    finally:
        c.close()


def init() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "docs").mkdir(parents=True, exist_ok=True)
    with conn() as c:
        c.executescript(SCHEMA)
