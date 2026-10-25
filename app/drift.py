"""
Drift detection from LABEL-FREE signals.

In production you do not have ground truth for every document, so you cannot watch accuracy directly. You watch
proxies that move when the documents change shape:

  review_rate        share of extractions the system itself flagged for a human
  issue_rate         share failing deterministic validators (arithmetic, dates, required fields)
  dispute_rate       share where the two passes disagreed (among cross-checked documents)
  mean_confidence    average field confidence                                  (should not drop)
  null_rate          average share of fields left empty                        (should not rise)
  novel_layout_rate  share of documents whose page layout is unlike anything in the baseline.
                     This one needs no model output at all, so it can lead the others.

A window of the most recent documents is compared with a BASELINE window (first BASELINE_N extractions, or a pinned
known-good window). An alert needs BOTH:
  * statistical significance:   Fisher's exact test for rates (exact even at tiny counts: baseline rates sit near
                                zero, where normal approximations are unreliable), Welch for means; alpha is strict
                                because many metrics are tested repeatedly as the stream advances
  * practical significance:     a minimum effect size, so a statistically real but trivial wobble does not page anyone
and a minimum sample size per window. Limits to know: if the format changes INSIDE the baseline window the baseline is
contaminated (pin a known-good baseline); gradual drift moves the baseline-vs-recent gap slowly; and the thresholds were
tuned on synthetic data, so measure the false-alarm rate on your own stable traffic before trusting them.
"""
import math
import statistics

import requests

from app import store
from app.config import (ALERT_WEBHOOK_URL, BASELINE_N, DRIFT_ALPHA, DRIFT_EVAL_EVERY, LAYOUT_MIN_T, MIN_EFFECT_CONF,
                        MIN_EFFECT_PROP, MIN_SAMPLES, WINDOW_N)
from app.imaging import hamming

PROP_METRICS = ("review_rate", "issue_rate", "dispute_rate", "novel_layout_rate")
CONT_METRICS = {"mean_confidence": "down", "null_rate": "up"}   # metric -> the direction that is BAD
N_FIELDS = 9


# ------------------------------------------------------------------ statistics ----
def fisher_one_sided(x1: int, n1: int, x2: int, n2: int) -> float:
    """P(window-2 rate is at least this much higher than window-1 by chance), exact hypergeometric tail."""
    total, k = n1 + n2, x1 + x2
    if n1 == 0 or n2 == 0 or k == 0:
        return 1.0
    denom = math.comb(total, k)
    return min(1.0, sum(math.comb(n2, i) * math.comb(n1, k - i) for i in range(x2, min(n2, k) + 1)) / denom)


def _norm_sf(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2))


def welch_one_sided(m1: float, s1: float, n1: int, m2: float, s2: float, n2: int, direction: str) -> float:
    """One-sided p that window 2's mean moved in the BAD direction (normal approximation: fine for n >= 30)."""
    se = math.sqrt(s1 ** 2 / n1 + s2 ** 2 / n2)
    if se == 0:
        return 0.0 if (m2 < m1 if direction == "down" else m2 > m1) else 1.0
    z = (m1 - m2) / se if direction == "down" else (m2 - m1) / se
    return _norm_sf(z)


def wilson(x: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = x / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


# ---------------------------------------------------------------- layout novelty ----
def novelty_threshold(baseline_hashes: list[str]) -> int:
    """Adaptive: how far apart do baseline documents of the SAME families normally sit? Take the 99th percentile of
    leave-one-out nearest-neighbour distances plus a margin, with a floor. Derived from the baseline itself, so it
    adapts to whatever layout variety your real traffic has."""
    if len(baseline_hashes) < 5:
        return LAYOUT_MIN_T
    nn = sorted(min(hamming(h, o) for j, o in enumerate(baseline_hashes) if j != i) for i, h in enumerate(baseline_hashes))
    return max(LAYOUT_MIN_T, nn[min(len(nn) - 1, int(0.99 * len(nn)))] + 2)


def novel_flags(hashes: list[str], baseline_hashes: list[str], threshold: int, leave_one_out: bool = False) -> list[bool]:
    flags = []
    for i, h in enumerate(hashes):
        others = [o for j, o in enumerate(baseline_hashes) if not (leave_one_out and j == i)]
        flags.append(bool(others) and min(hamming(h, o) for o in others) > threshold)
    return flags


# ---------------------------------------------------------------------- metrics ----
def _window_metrics(rows: list[dict], novel: list[bool]) -> dict:
    n = len(rows)
    cc = [r for r in rows if r["crosschecked"]]
    confs = [r["mean_conf"] for r in rows if r["mean_conf"] is not None]
    nulls = [r["n_null"] / N_FIELDS for r in rows if r["n_null"] is not None]

    def cont(vals):
        return {"kind": "cont", "mean": statistics.fmean(vals) if vals else 0.0,
                "std": statistics.stdev(vals) if len(vals) > 1 else 0.0, "n": len(vals)}

    return {
        "review_rate": {"kind": "prop", "x": sum(1 for r in rows if r["needs_review"]), "n": n},
        "issue_rate": {"kind": "prop", "x": sum(1 for r in rows if r["n_issues"]), "n": n},
        "dispute_rate": {"kind": "prop", "x": sum(1 for r in cc if r["n_disputed"]), "n": len(cc)},
        "novel_layout_rate": {"kind": "prop", "x": sum(novel), "n": n},
        "mean_confidence": cont(confs), "null_rate": cont(nulls),
    }


def _value(m: dict) -> float:
    return m["x"] / m["n"] if m["kind"] == "prop" and m["n"] else (m.get("mean", 0.0))


def baseline_slice(namespace: str, total: int) -> tuple[int, int]:
    pinned = store.get_kv(f"baseline:{namespace}")
    if pinned:
        a, b = (int(x) for x in pinned.split(","))
        return min(a, total), min(b, total)
    return 0, min(BASELINE_N, total)


def pin_baseline(namespace: str) -> dict:
    """Declare the most recent BASELINE_N extractions the new normal (e.g. after fixing prompts or accepting a new
    format). Acknowledges open alerts: the operator has decided this is the baseline now."""
    n = len(store.extraction_rows(namespace))
    start = max(0, n - BASELINE_N)
    store.set_kv(f"baseline:{namespace}", f"{start},{n}")
    acked = 0
    for a in store.list_alerts(namespace, "open"):
        acked += store.ack_alert(namespace, a["id"])
    return {"baseline_start": start, "baseline_end": n, "alerts_acknowledged": acked}


def baseline_hashes(namespace: str) -> list[str]:
    """Just the baseline window's layout hashes: one small indexed query, NOT a scan of every extraction row, so
    per-document novelty checks stay cheap at thousands of documents."""
    from app import db
    pinned = store.get_kv(f"baseline:{namespace}")
    start, limit = (int(pinned.split(",")[0]), int(pinned.split(",")[1]) - int(pinned.split(",")[0])) if pinned else (0, BASELINE_N)
    with db.conn() as c:
        return [r["layout_hash"] for r in c.execute(
            """SELECT d.layout_hash FROM extractions e JOIN documents d ON d.id = e.document_id
               WHERE e.namespace = ? AND e.is_current = 1 ORDER BY e.id LIMIT ? OFFSET ?""", (namespace, limit, start))]


def is_novel(doc: dict) -> bool:
    """Is this document's layout unlike the baseline? False until a baseline of MIN_SAMPLES exists."""
    base = baseline_hashes(doc["namespace"])
    if len(base) < MIN_SAMPLES:
        return False
    return novel_flags([doc["layout_hash"]], base, novelty_threshold(base))[0]


def _compare(name: str, b: dict, c: dict) -> dict | None:
    """Test one metric: window c against window b. Significant = statistically AND practically significant, in the BAD direction."""
    if b["n"] < MIN_SAMPLES or c["n"] < MIN_SAMPLES:
        return None
    if b["kind"] == "prop":
        p = fisher_one_sided(b["x"], b["n"], c["x"], c["n"])
        delta, bad_dir, min_eff = _value(c) - _value(b), "up", MIN_EFFECT_PROP
    else:
        bad_dir, min_eff = CONT_METRICS[name], MIN_EFFECT_CONF
        p = welch_one_sided(b["mean"], b["std"], b["n"], c["mean"], c["std"], c["n"], bad_dir)
        delta = c["mean"] - b["mean"]
    bad_move = delta > 0 if bad_dir == "up" else delta < 0
    return {"metric": name, "baseline": _value(b), "current": _value(c), "delta": delta, "p_value": p,
            "significant": bool(p < DRIFT_ALPHA and abs(delta) >= min_eff and bad_move), "min_effect": min_eff,
            "n_baseline": b["n"], "n_current": c["n"]}


def baseline_stability(base: list[dict]) -> tuple[bool | None, list[str]]:
    """Is the BASELINE itself stationary? Compare its first half with its second half using the same tests.
    If a format change happened inside the baseline window, the baseline is contaminated: the new layout looks
    "normal", the layout signal goes blind and every proxy under-reports. That fails silently unless checked.
    Returns (stable, warnings); stable is None when the baseline is too small to judge."""
    half = len(base) // 2
    if half < MIN_SAMPLES:
        return None, []
    first, second = base[:half], base[half:]
    h1 = [r["layout_hash"] for r in first]
    thr = novelty_threshold(h1)
    m1 = _window_metrics(first, novel_flags(h1, h1, thr, leave_one_out=True))
    m2 = _window_metrics(second, novel_flags([r["layout_hash"] for r in second], h1, thr))
    warnings = []
    for name in (*PROP_METRICS, *CONT_METRICS):
        r = _compare(name, m1[name], m2[name])
        if r and r["significant"]:
            warnings.append(f"{name} differs between the first and second half of the baseline "
                            f"({r['baseline']:.3f} to {r['current']:.3f}, p={r['p_value']:.1e})")
    return (not warnings), warnings


def evaluate(namespace: str, persist: bool = True, now: float | None = None) -> dict:
    rows = store.extraction_rows(namespace)
    b0, b1 = baseline_slice(namespace, len(rows))
    base = rows[b0:b1]
    cur = rows[max(b1, len(rows) - WINDOW_N):]
    if len(base) < MIN_SAMPLES or len(cur) < MIN_SAMPLES:
        return {"status": "insufficient_data", "n_baseline": len(base), "n_current": len(cur),
                "need": MIN_SAMPLES, "results": [], "alerts": []}

    base_hashes = [r["layout_hash"] for r in base]
    thr = novelty_threshold(base_hashes)
    bm = _window_metrics(base, novel_flags(base_hashes, base_hashes, thr, leave_one_out=True))
    cm = _window_metrics(cur, novel_flags([r["layout_hash"] for r in cur], base_hashes, thr))

    results, fired = [], []
    for name in (*PROP_METRICS, *CONT_METRICS):
        r = _compare(name, bm[name], cm[name])
        if r is None:
            continue
        results.append(r)
        if r["significant"] and persist:
            delta, p, b_, c_ = r["delta"], r["p_value"], r["baseline"], r["current"]
            severity = "critical" if abs(delta) >= 2 * r["min_effect"] and p < 1e-6 else "warning"
            fmt = (lambda v: f"{v:.1%}") if name.endswith("_rate") else (lambda v: f"{v:.3f}")
            msg = (f"{name} moved from {fmt(b_)} to {fmt(c_)} over the last {r['n_current']} documents "
                   f"(baseline of {r['n_baseline']}, p={p:.1e})")
            existing = store.open_alert(namespace, name)
            if existing:
                store.touch_alert(existing["id"], c_, p, now)
            else:
                aid = store.insert_alert(namespace, name, severity, msg, b_, c_, p, len(rows), now)
                fired.append({"id": aid, "metric": name, "severity": severity, "message": msg, "at_extraction_index": len(rows)})
                notify(namespace, severity, msg)
    stable, warnings = baseline_stability(base)
    return {"status": "ok", "n_baseline": len(base), "n_current": len(cur), "novelty_threshold_bits": thr,
            "baseline_stable": stable, "baseline_warnings": warnings, "results": results, "alerts": fired}


def notify(namespace: str, severity: str, message: str) -> None:
    if not ALERT_WEBHOOK_URL:
        return
    try:  # alerting must never break extraction
        requests.post(ALERT_WEBHOOK_URL, json={"text": f"[{severity.upper()}] extraction drift in '{namespace}': {message}"}, timeout=5)
    except requests.RequestException:
        pass


def maybe_evaluate(namespace: str) -> dict | None:
    """Called after every extraction. An atomic counter makes exactly one caller per interval run the evaluation,
    even with many worker threads or processes."""
    if store.incr_kv(f"since_eval:{namespace}") % DRIFT_EVAL_EVERY == 0:
        return evaluate(namespace)
    return None


# ------------------------------------------------------------- timeseries / sim ----
def timeseries(namespace: str, step: int = 10) -> list[dict]:
    rows = store.extraction_rows(namespace)
    if not rows:
        return []
    b0, b1 = baseline_slice(namespace, len(rows))
    base_hashes = [r["layout_hash"] for r in rows[b0:b1]]
    thr = novelty_threshold(base_hashes) if len(base_hashes) >= 5 else None
    out = []
    for i in range(min(step, len(rows)), len(rows) + 1, step):
        win = rows[max(0, i - WINDOW_N):i]
        nov = novel_flags([r["layout_hash"] for r in win], base_hashes, thr) if thr is not None else [False] * len(win)
        m = _window_metrics(win, nov)
        oracle = [r["oracle_field_acc"] for r in win if r["oracle_field_acc"] is not None]
        out.append({"index": i, **{k: _value(v) for k, v in m.items()},
                    "oracle_accuracy": statistics.fmean(oracle) if oracle else None})
    return out


def simulation_report(namespace: str) -> dict:
    """For synthetic streams: when did the format change, and how many documents later did each signal alert?"""
    from app import db
    with db.conn() as c:
        rows = c.execute(
            """SELECT e.id, json_extract(d.meta, '$.sim_template') AS tpl FROM extractions e
               JOIN documents d ON d.id = e.document_id WHERE e.namespace = ? AND e.is_current = 1 ORDER BY e.id""",
            (namespace,)).fetchall()
    change = next((i + 1 for i, r in enumerate(rows) if r["tpl"] == "B"), None)
    first = {}
    for a in sorted(store.list_alerts(namespace), key=lambda a: a["id"]):
        first.setdefault(a["metric"], a["at_extraction_index"])
    return {"change_at_index": change, "total_processed": len(rows),
            "first_alert_by_metric": first,
            "detection_lag_docs": {m: (idx - change if change else None) for m, idx in first.items()},
            "first_alert_overall": min(first.values()) if first else None,
            "false_alarms_before_change": sum(1 for m, idx in first.items() if change and idx <= change)}
