"""
Turning "the model said so" into "we can trust it enough to skip a human".

Three independent evidence sources, because each has a blind spot:

  1. SELF-REPORTED confidence  cheap, but LLMs are poorly calibrated and often confidently wrong.
  2. CROSS-CHECK agreement     a second pass; disagreement is strong evidence of trouble. Blind spot: the same
                               model repeats its own mistakes, so agreement does not prove correctness.
  3. VALIDATORS                deterministic business rules (subtotal + tax = total, quantity x price = amount,
                               due date after invoice date). These catch overconfident errors that both passes
                               agree on, and need no model at all.

Plus a fourth, statistical safeguard: AUDIT SAMPLING. A random slice of auto-approved documents goes to a human
anyway. Without it, humans only ever see the documents the system already doubted, so you never learn how wrong
the "confident" ones are, and every accuracy estimate is biased toward the easy cases.
"""
import hashlib
from decimal import Decimal

from app.config import ARITH_TOLERANCE, AUDIT_SAMPLE_RATE, DISPUTE_PENALTY, REVIEW_THRESHOLD
from app.schemas import ALL_FIELDS, REQUIRED_FIELDS, Canonical, fields_equal

# error = forces human review and caps the implicated fields' confidence; warning = recorded and shown only.
# Domain-specific: an invoice with shipping or discount lines can legitimately fail arith_lines, so a real
# deployment should tune this table (or downgrade arith_lines to a warning).
SEVERITY = {"missing_required": "error", "parse_error": "error", "arith_total": "error", "arith_lines": "error",
            "arith_line_item": "error", "date_order": "error", "date_range": "error", "currency_missing": "warning"}
ISSUE_CONF_CAP = 0.4
TOL = Decimal(str(ARITH_TOLERANCE))


def merge_passes(c1: Canonical, c2: Canonical | None) -> dict:
    """Combine the primary pass with the optional cross-check pass.
    Agree  -> mean of the two confidences (two looks beat one).
    Differ -> the field is DISPUTED: keep the more confident value for display, but confidence collapses to
              min(c1, c2) * DISPUTE_PENALTY so it is certain to be reviewed."""
    values, conf, disputed = {}, {}, []
    for f in ALL_FIELDS:
        v1, k1 = c1.values[f], c1.conf[f]
        if c2 is None:
            values[f], conf[f] = v1, k1
            continue
        v2, k2 = c2.values[f], c2.conf[f]
        if fields_equal(f, v1, v2):
            values[f], conf[f] = v1, (k1 + k2) / 2
        else:
            disputed.append(f)
            values[f] = v1 if k1 >= k2 else v2
            conf[f] = min(k1, k2) * DISPUTE_PENALTY
    return {"values": values, "conf": conf, "disputed": disputed,
            "parse_errors": c1.parse_errors + (c2.parse_errors if c2 else [])}


def _d(v) -> Decimal | None:
    return Decimal(v) if v is not None else None


def validate_record(values: dict, parse_errors: list[str] | None = None) -> list[dict]:
    issues = []

    def add(code, fields, message):
        issues.append({"code": code, "severity": SEVERITY[code], "fields": fields, "message": message})

    for err in parse_errors or []:
        add("parse_error", [err.split(":")[0].split("[")[0]], err)
    for f in REQUIRED_FIELDS:
        if values.get(f) is None:
            add("missing_required", [f], f"{f} is missing")

    sub, tax, tot = _d(values.get("subtotal")), _d(values.get("tax")), _d(values.get("total"))
    if None not in (sub, tax, tot) and abs(sub + tax - tot) > TOL:
        add("arith_total", ["subtotal", "tax", "total"], f"subtotal {sub} + tax {tax} = {sub + tax}, but total is {tot}")

    items = values.get("line_items") or []
    if items:
        line_sum = sum(_d(i["amount"]) for i in items)
        if sub is not None and abs(line_sum - sub) > TOL:
            add("arith_lines", ["line_items", "subtotal"], f"line items sum to {line_sum}, but subtotal is {sub}")
        bad = [i["description"] for i in items
               if abs(_d(i["quantity"]) * _d(i["unit_price"]) - _d(i["amount"])) > TOL * max(Decimal(1), _d(i["quantity"]))]
        if bad:
            add("arith_line_item", ["line_items"], f"quantity x unit price != amount for: {', '.join(bad[:3])}")

    inv, due = values.get("invoice_date"), values.get("due_date")
    if inv and due and due < inv:
        add("date_order", ["invoice_date", "due_date"], f"due date {due} is before invoice date {inv}")
    for f in ("invoice_date", "due_date"):
        if values.get(f) and not 2000 <= int(values[f][:4]) <= 2100:
            add("date_range", [f], f"{f} year {values[f][:4]} is implausible")
    if tot is not None and not values.get("currency"):
        add("currency_missing", ["currency"], "a total was extracted but no currency")
    return issues


def apply_issue_caps(conf: dict, issues: list[dict]) -> dict:
    """An error-level validator hit means those fields cannot be trusted, whatever the model claimed."""
    capped = dict(conf)
    for issue in issues:
        if issue["severity"] == "error":
            for f in issue["fields"]:
                if f in capped:
                    capped[f] = min(capped[f], ISSUE_CONF_CAP)
    return capped


def is_audit_sample(doc_key: str, rate: float = AUDIT_SAMPLE_RATE) -> bool:
    """Deterministic, so reprocessing a document makes the same audit decision."""
    return int(hashlib.sha256(f"audit:{doc_key}".encode()).hexdigest()[:8], 16) / 2 ** 32 < rate


def route(conf: dict, disputed: list[str], issues: list[dict], threshold: float = REVIEW_THRESHOLD) -> list[str]:
    """Reasons this extraction needs a human. Empty list = eligible for straight-through processing."""
    reasons = [f"low_confidence:{f}" for f in ALL_FIELDS if conf[f] < threshold]
    reasons += [f"disputed:{f}" for f in disputed]
    reasons += [f"issue:{i['code']}" for i in issues if i["severity"] == "error"]
    return reasons


def review_priority(conf: dict, disputed: list[str], issues: list[dict], novel_layout: bool) -> float:
    """Uncertainty sampling: review the most informative documents first. Lowest confidence, disputes,
    validator failures and unfamiliar layouts rise to the top; those corrections teach the system the most."""
    errors = sum(1 for i in issues if i["severity"] == "error")
    return round((1 - min(conf.values())) + 0.3 * len(disputed) + 0.2 * errors + (0.5 if novel_layout else 0.0), 4)


def oracle_scores(values: dict, truth: dict) -> tuple[float, int]:
    """SIMULATION ONLY: compare against known ground truth. Not available in production, which is exactly
    why drift detection must work from label-free proxy signals."""
    correct = sum(1 for f in ALL_FIELDS if fields_equal(f, values.get(f), truth.get(f)))
    return correct / len(ALL_FIELDS), int(correct == len(ALL_FIELDS))
