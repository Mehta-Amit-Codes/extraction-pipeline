"""Repository functions over the SQLite store. Namespaces isolate tenants/projects: every query that
returns data is scoped by namespace (and the few-shot pool is per namespace, so one tenant's corrected
documents are never shown to the model while processing another tenant's)."""
import hashlib
import json
import os
import sqlite3
import time

from app import db, imaging
from app.config import DATA_DIR

_JSON_COLS = ("record", "field_conf", "pass1", "pass2", "disputed_fields", "issues", "fewshot_doc_ids",
              "review_reasons", "final_record")
_EXT = {"image/png": "png", "image/jpeg": "jpg", "application/pdf": "pdf"}


def _dumps(v) -> str | None:
    return None if v is None else json.dumps(v)


def _decode(row: sqlite3.Row | None, cols=_JSON_COLS) -> dict | None:
    if row is None:
        return None
    d = dict(row)
    for k in cols:
        if k in d and isinstance(d[k], str):
            d[k] = json.loads(d[k])
    return d


# ------------------------------------------------------------------- documents ----
def add_document(namespace: str, filename: str, data: bytes, batch_id: str | None = None,
                 meta: dict | None = None, now: float | None = None) -> tuple[int, bool]:
    """Validate, fingerprint and store a document. Content-addressed: the same bytes in the same
    namespace return the existing document (created=False), so re-uploading a folder is harmless.
    Raises imaging.UnsupportedDocument at INGEST time so bad files fail fast instead of dead-lettering later."""
    mime = imaging.sniff_mime(data)
    sha = hashlib.sha256(data).hexdigest()
    with db.conn() as c:
        row = c.execute("SELECT id FROM documents WHERE namespace = ? AND sha256 = ?", (namespace, sha)).fetchone()
    if row:
        return row["id"], False
    pages = imaging.load_images(data)
    rel = f"docs/{sha}.{_EXT[mime]}"
    target = DATA_DIR / rel
    if not target.exists():
        tmp = target.with_suffix(target.suffix + f".{os.getpid()}.tmp")
        tmp.write_bytes(data)
        os.replace(tmp, target)  # atomic: readers never see a half-written file
    try:
        with db.tx() as c:
            cur = c.execute(
                """INSERT INTO documents (namespace, batch_id, filename, sha256, mime, path, layout_hash, n_pages, meta, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (namespace, batch_id, filename, sha, mime, rel, imaging.layout_hash(pages[0]), len(pages),
                 json.dumps(meta or {}), now or time.time()))
            return cur.lastrowid, True
    except sqlite3.IntegrityError:  # lost a race with a concurrent identical upload
        with db.conn() as c:
            return c.execute("SELECT id FROM documents WHERE namespace = ? AND sha256 = ?",
                             (namespace, sha)).fetchone()["id"], False


def get_document(document_id: int, namespace: str | None = None) -> dict | None:
    sql, params = "SELECT * FROM documents WHERE id = ?", [document_id]
    if namespace:
        sql += " AND namespace = ?"
        params.append(namespace)
    with db.conn() as c:
        return _decode(c.execute(sql, params).fetchone(), ("meta",))


def read_bytes(doc: dict) -> bytes:
    return (DATA_DIR / doc["path"]).read_bytes()


def list_documents(namespace: str, batch_id: str | None = None, limit: int = 200) -> list[dict]:
    sql, params = "SELECT * FROM documents WHERE namespace = ?", [namespace]
    if batch_id:
        sql += " AND batch_id = ?"
        params.append(batch_id)
    with db.conn() as c:
        return [_decode(r, ("meta",)) for r in c.execute(sql + " ORDER BY id DESC LIMIT ?", params + [limit])]


# ----------------------------------------------------------------- extractions ----
def insert_extraction(c, **f) -> int:
    """Insert inside the caller's transaction (so it commits atomically with the job completion)."""
    c.execute("UPDATE extractions SET is_current = 0 WHERE document_id = ?", (f["document_id"],))
    version = (c.execute("SELECT COALESCE(MAX(version), 0) AS v FROM extractions WHERE document_id = ?",
                         (f["document_id"],)).fetchone()["v"]) + 1
    cols = ["document_id", "namespace", "job_id", "created_at", "extractor", "model", "status", "needs_review",
            "review_reasons", "priority", "audit_sample", "crosschecked", "record", "field_conf", "pass1", "pass2",
            "disputed_fields", "issues", "fewshot_doc_ids", "mean_conf", "min_conf", "n_disputed", "n_issues",
            "n_null", "tokens_in", "tokens_out", "cost_usd", "latency_ms", "oracle_field_acc", "oracle_exact",
            "final_record"]
    vals = [(_dumps(f.get(k)) if k in _JSON_COLS else f.get(k)) for k in cols]
    cur = c.execute(f"INSERT INTO extractions ({','.join(cols)}, version, is_current) "
                    f"VALUES ({','.join('?' * len(cols))}, ?, 1)", [*vals, version])
    return cur.lastrowid


def get_extraction(extraction_id: int, namespace: str | None = None) -> dict | None:
    sql, params = "SELECT * FROM extractions WHERE id = ?", [extraction_id]
    if namespace:
        sql += " AND namespace = ?"
        params.append(namespace)
    with db.conn() as c:
        return _decode(c.execute(sql, params).fetchone())


def current_extraction(document_id: int) -> dict | None:
    with db.conn() as c:
        return _decode(c.execute("SELECT * FROM extractions WHERE document_id = ? AND is_current = 1",
                                 (document_id,)).fetchone())


def pending_reviews(namespace: str, limit: int = 50) -> list[dict]:
    """Review queue, most informative first (uncertainty sampling): lowest-confidence / novel-layout /
    disputed documents carry the highest priority."""
    with db.conn() as c:
        rows = c.execute(
            """SELECT e.*, d.filename, d.layout_hash FROM extractions e JOIN documents d ON d.id = e.document_id
               WHERE e.namespace = ? AND e.status = 'pending_review' AND e.is_current = 1
               ORDER BY e.priority DESC, e.id LIMIT ?""", (namespace, limit)).fetchall()
    return [_decode(r) for r in rows]


def apply_review(extraction_id: int, namespace: str, reviewer: str, action: str, final_record: dict | None,
                 changed_fields: list[str], now: float | None = None) -> dict | None:
    """Finalise a pending extraction. Returns None if it is not pending (e.g. two reviewers raced):
    a review can only be applied once."""
    now = now or time.time()
    status = {"approve": "approved", "edit": "corrected", "reject": "rejected"}[action]
    with db.tx() as c:
        row = c.execute("SELECT * FROM extractions WHERE id = ? AND namespace = ? AND status = 'pending_review'",
                        (extraction_id, namespace)).fetchone()
        if row is None:
            return None
        before = json.loads(row["record"])
        c.execute("UPDATE extractions SET status = ?, final_record = ?, reviewed_at = ? WHERE id = ?",
                  (status, _dumps(final_record), now, extraction_id))
        c.execute("""INSERT INTO reviews (extraction_id, namespace, reviewer, action, before, after, changed_fields,
                     audit_sample, created_at) VALUES (?,?,?,?,?,?,?,?,?)""",
                  (extraction_id, namespace, reviewer, action, _dumps(before), _dumps(final_record),
                   _dumps(changed_fields), row["audit_sample"], now))
        return _decode(c.execute("SELECT * FROM extractions WHERE id = ?", (extraction_id,)).fetchone())


def extraction_rows(namespace: str) -> list[dict]:
    """Light per-extraction rows in processing order (current extractions only), for drift/quality metrics."""
    with db.conn() as c:
        return [dict(r) for r in c.execute(
            """SELECT e.id, e.document_id, e.status, e.needs_review, e.audit_sample, e.crosschecked, e.mean_conf,
                      e.min_conf, e.n_disputed, e.n_issues, e.n_null, e.oracle_field_acc, e.oracle_exact,
                      e.cost_usd, e.latency_ms, d.layout_hash
               FROM extractions e JOIN documents d ON d.id = e.document_id
               WHERE e.namespace = ? AND e.is_current = 1 ORDER BY e.id""", (namespace,))]


def review_rows(namespace: str) -> list[dict]:
    with db.conn() as c:
        rows = c.execute(
            """SELECT r.*, e.field_conf, e.document_id FROM reviews r JOIN extractions e ON e.id = r.extraction_id
               WHERE r.namespace = ? ORDER BY r.id""", (namespace,)).fetchall()
    return [_decode(r, ("before", "after", "changed_fields", "field_conf")) for r in rows]


def status_counts(namespace: str) -> dict:
    with db.conn() as c:
        return {r["status"]: r["n"] for r in c.execute(
            "SELECT status, COUNT(*) AS n FROM extractions WHERE namespace = ? AND is_current = 1 GROUP BY status",
            (namespace,))}


def final_records(namespace: str, statuses=("auto_approved", "approved", "corrected")) -> list[dict]:
    q = ",".join("?" * len(statuses))
    with db.conn() as c:
        rows = c.execute(
            f"""SELECT e.id, e.document_id, e.status, e.record, e.final_record, d.filename, d.sha256
                FROM extractions e JOIN documents d ON d.id = e.document_id
                WHERE e.namespace = ? AND e.is_current = 1 AND e.status IN ({q}) ORDER BY e.id""",
            [namespace, *statuses]).fetchall()
    out = []
    for r in rows:
        rec = json.loads(r["final_record"]) if r["final_record"] else json.loads(r["record"])
        out.append({"document_id": r["document_id"], "filename": r["filename"], "sha256": r["sha256"],
                    "status": r["status"], "record": rec})
    return out


def cost_summary(namespace: str) -> dict:
    with db.conn() as c:
        r = c.execute("""SELECT COUNT(*) AS n, COALESCE(SUM(cost_usd),0) AS cost, COALESCE(SUM(tokens_in),0) AS tin,
                         COALESCE(SUM(tokens_out),0) AS tout, COALESCE(AVG(latency_ms),0) AS lat
                         FROM extractions WHERE namespace = ?""", (namespace,)).fetchone()
    return {"extractions": r["n"], "cost_usd": round(r["cost"], 4), "tokens_in": r["tin"], "tokens_out": r["tout"],
            "avg_latency_ms": round(r["lat"], 1), "cost_per_doc_usd": round(r["cost"] / r["n"], 5) if r["n"] else 0.0}


# -------------------------------------------------------------------- examples ----
def upsert_example(namespace: str, document_id: int, record: dict, source: str, layout_hash: str,
                   now: float | None = None) -> int:
    with db.tx() as c:
        c.execute("""INSERT INTO examples (namespace, document_id, record, source, layout_hash, created_at, active)
                     VALUES (?,?,?,?,?,?,1) ON CONFLICT(namespace, document_id)
                     DO UPDATE SET record = excluded.record, source = excluded.source, active = 1""",
                  (namespace, document_id, json.dumps(record), source, layout_hash, now or time.time()))
        return c.execute("SELECT id FROM examples WHERE namespace = ? AND document_id = ?",
                         (namespace, document_id)).fetchone()["id"]


def list_examples(namespace: str, active_only: bool = True) -> list[dict]:
    sql = "SELECT * FROM examples WHERE namespace = ?" + (" AND active = 1" if active_only else "")
    with db.conn() as c:
        return [_decode(r, ("record",)) for r in c.execute(sql + " ORDER BY id", (namespace,))]


def set_example_active(namespace: str, example_id: int, active: bool) -> None:
    with db.tx() as c:
        c.execute("UPDATE examples SET active = ? WHERE id = ? AND namespace = ?", (1 if active else 0, example_id, namespace))


# ---------------------------------------------------------------------- alerts ----
def open_alert(namespace: str, metric: str) -> dict | None:
    with db.conn() as c:
        return _decode(c.execute("SELECT * FROM alerts WHERE namespace = ? AND metric = ? AND status = 'open'",
                                 (namespace, metric)).fetchone(), ())


def insert_alert(namespace: str, metric: str, severity: str, message: str, baseline: float, current: float,
                 p_value: float | None, at_index: int, now: float | None = None) -> int:
    now = now or time.time()
    with db.tx() as c:
        return c.execute(
            """INSERT INTO alerts (namespace, created_at, last_seen_at, metric, severity, message, baseline, current,
               p_value, at_extraction_index, status) VALUES (?,?,?,?,?,?,?,?,?,?, 'open')""",
            (namespace, now, now, metric, severity, message, baseline, current, p_value, at_index)).lastrowid


def touch_alert(alert_id: int, current: float, p_value: float | None, now: float | None = None) -> None:
    with db.tx() as c:
        c.execute("UPDATE alerts SET last_seen_at = ?, current = ?, p_value = ? WHERE id = ?",
                  (now or time.time(), current, p_value, alert_id))


def list_alerts(namespace: str, status: str | None = None, limit: int = 100) -> list[dict]:
    sql, params = "SELECT * FROM alerts WHERE namespace = ?", [namespace]
    if status:
        sql += " AND status = ?"
        params.append(status)
    with db.conn() as c:
        return [dict(r) for r in c.execute(sql + " ORDER BY id DESC LIMIT ?", params + [limit])]


def ack_alert(namespace: str, alert_id: int) -> bool:
    with db.tx() as c:
        return c.execute("UPDATE alerts SET status = 'acknowledged' WHERE id = ? AND namespace = ?",
                         (alert_id, namespace)).rowcount > 0


# -------------------------------------------------------------------------- kv ----
def get_kv(key: str, default: str | None = None) -> str | None:
    with db.conn() as c:
        row = c.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else default


def set_kv(key: str, value: str) -> None:
    with db.tx() as c:
        c.execute("INSERT INTO kv (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                  (key, str(value)))


def incr_kv(key: str, by: int = 1) -> int:
    """Atomic counter (used to schedule drift evaluations across worker threads/processes)."""
    with db.tx() as c:
        row = c.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
        new = (int(row["value"]) if row else 0) + by
        c.execute("INSERT INTO kv (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                  (key, str(new)))
        return new


# ---------------------------------------------------------- simulation helpers ----
def reviews_per_document_versions(namespace: str) -> list[dict]:
    """Every extraction version per document, oldest first: lets us compare the SAME documents before and after the
    few-shot pool grew (a fair before/after, unlike comparing different documents)."""
    with db.conn() as c:
        return [dict(r) for r in c.execute(
            """SELECT e.id, e.document_id, e.version, e.needs_review, e.oracle_field_acc, e.oracle_exact, e.fewshot_doc_ids,
                      json_extract(d.meta, '$.sim_template') AS template
               FROM extractions e JOIN documents d ON d.id = e.document_id
               WHERE e.namespace = ? ORDER BY e.document_id, e.version""", (namespace,))]
