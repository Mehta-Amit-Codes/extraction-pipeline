"""
Invoice schema + canonicalisation.

Everything downstream (cross-check agreement, validators, accuracy, drift, review diffs) compares
*canonical* values, so "14 Nov 2023" vs "2023-11-14" or "$1,234.50" vs "1234.5" are not false
disagreements. Canonical forms (also what gets stored):

  strings      stripped, whitespace-collapsed (comparison is additionally case-insensitive)
  dates        ISO yyyy-mm-dd
  money        plain decimal string with 2 places, e.g. "1234.50"
  currency     3 uppercase letters
  quantity     plain decimal, no trailing zeros
  line_items   list of {description, quantity, unit_price, amount}

Parsing is deliberately lenient only where it is UNAMBIGUOUS. "03/04/2024" could be 3 April or 4 March,
so it is rejected (value -> None + a parse error -> human review) instead of guessed. A confidently
wrong date in a financial record is worse than a flagged blank.
"""
import re
from datetime import date, datetime
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any

from pydantic import BaseModel, Field, field_validator

SCALAR_FIELDS = ["vendor_name", "invoice_number", "invoice_date", "due_date", "currency",
                 "subtotal", "tax", "total"]
ALL_FIELDS = SCALAR_FIELDS + ["line_items"]
MONEY_FIELDS = {"subtotal", "tax", "total"}
DATE_FIELDS = {"invoice_date", "due_date"}
REQUIRED_FIELDS = ["invoice_number", "total"]

_DATE_FORMATS = ("%Y-%m-%d", "%d %b %Y", "%d %B %Y", "%b %d, %Y", "%B %d, %Y", "%Y/%m/%d")
CENT = Decimal("0.01")


# ----------------------------------------------------------------- primitives ----
def parse_money(value: Any) -> Decimal | None:
    """Accepts numbers and strings like '$1,234.50', '1.234,50 EUR', '1234.5'. None if unparseable."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float, Decimal)):
        try:
            return Decimal(str(value)).quantize(CENT, ROUND_HALF_UP)
        except InvalidOperation:
            return None
    s = re.sub(r"[^\d.,\-]", "", str(value))
    if not s or not re.search(r"\d", s):
        return None
    neg = s.startswith("-") or str(value).strip().startswith("(")
    s = s.replace("-", "")
    if "," in s and "." in s:                       # last separator is the decimal one
        dec = "," if s.rfind(",") > s.rfind(".") else "."
        thou = "." if dec == "," else ","
        s = s.replace(thou, "").replace(dec, ".")
    elif "," in s:                                   # only commas: decimal comma iff exactly 2 trailing digits
        head, _, tail = s.rpartition(",")
        s = f"{head.replace(',', '')}.{tail}" if len(tail) == 2 and "," not in head else s.replace(",", "")
    try:
        d = Decimal(s).quantize(CENT, ROUND_HALF_UP)
    except InvalidOperation:
        return None
    return -d if neg else d


def parse_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    s = str(value).strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None  # includes ambiguous dd/mm vs mm/dd forms: never guess


def parse_quantity(value: Any) -> Decimal | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return Decimal(str(value).replace(",", "").strip()).normalize()
    except InvalidOperation:
        return None


def _money_str(d: Decimal) -> str:
    return f"{d:.2f}"


def _qty_str(d: Decimal) -> str:
    return format(d.normalize(), "f")


def _clean(s: Any) -> str | None:
    if s is None:
        return None
    out = re.sub(r"\s+", " ", str(s)).strip()
    return out or None


# ------------------------------------------------------------ canonicalisation ----
def canonical_scalar(field: str, value: Any) -> tuple[str | None, str | None]:
    """Returns (canonical value or None, parse_error or None)."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None, None
    if field in MONEY_FIELDS:
        d = parse_money(value)
        return (_money_str(d), None) if d is not None else (None, f"{field}: cannot parse amount {value!r}")
    if field in DATE_FIELDS:
        d = parse_date(value)
        return (d.isoformat(), None) if d else (None, f"{field}: cannot parse unambiguous date {value!r}")
    if field == "currency":
        c = _clean(value)
        c = c.upper() if c else None
        return (c, None) if c and re.fullmatch(r"[A-Z]{3}", c) else (None, f"currency: {value!r} is not a 3-letter code")
    return _clean(value), None


def canonical_line_items(items: Any) -> tuple[list[dict], list[str]]:
    errors, out = [], []
    if not isinstance(items, list):
        return [], ["line_items: expected a list"]
    for i, it in enumerate(items):
        if not isinstance(it, dict):
            errors.append(f"line_items[{i}]: expected an object")
            continue
        qty, unit, amt = parse_quantity(it.get("quantity")), parse_money(it.get("unit_price")), parse_money(it.get("amount"))
        if qty is None or unit is None or amt is None or not _clean(it.get("description")):
            errors.append(f"line_items[{i}]: missing or unparseable description/quantity/unit_price/amount")
            continue
        out.append({"description": _clean(it["description"]), "quantity": _qty_str(qty),
                    "unit_price": _money_str(unit), "amount": _money_str(amt)})
    return out, errors


def _cmp_text(s: str | None) -> str | None:
    return None if s is None else re.sub(r"[\s.,;:]+$", "", re.sub(r"\s+", " ", s.casefold()).strip())


def line_items_key(items: list[dict] | None) -> tuple:
    return tuple(sorted((_cmp_text(i["description"]), i["quantity"], i["unit_price"], i["amount"]) for i in (items or [])))


def fields_equal(field: str, a: Any, b: Any) -> bool:
    """Compare two CANONICAL values for one field."""
    if field == "line_items":
        return line_items_key(a) == line_items_key(b)
    if a is None or b is None:
        return a is None and b is None
    return _cmp_text(a) == _cmp_text(b) if field not in MONEY_FIELDS | DATE_FIELDS else a == b


# ---------------------------------------------------------------- model output ----
class FieldOut(BaseModel):
    value: Any = None
    confidence: float = 0.0

    @field_validator("confidence", mode="before")
    @classmethod
    def _clamp(cls, v):
        try:
            return min(1.0, max(0.0, float(v)))  # models sometimes say 95 or "0.9": clamp/coerce instead of rejecting
        except (TypeError, ValueError):
            return 0.0


class RawExtraction(BaseModel):
    """What the extractor returns: per-field value + self-reported confidence."""
    fields: dict[str, FieldOut] = Field(default_factory=dict)
    line_items: list[dict] = Field(default_factory=list)
    line_items_confidence: float = 0.0

    @field_validator("line_items_confidence", mode="before")
    @classmethod
    def _clamp_li(cls, v):
        try:
            return min(1.0, max(0.0, float(v)))
        except (TypeError, ValueError):
            return 0.0


class Canonical(BaseModel):
    values: dict[str, Any]          # canonical values for ALL_FIELDS (None where absent/unparseable)
    conf: dict[str, float]          # self-reported confidence per field
    parse_errors: list[str] = []


def to_canonical(raw: RawExtraction) -> Canonical:
    values, conf, errors = {}, {}, []
    for f in SCALAR_FIELDS:
        out = raw.fields.get(f)
        v, err = canonical_scalar(f, out.value if out else None)
        values[f] = v
        # An unparseable value must not look trustworthy, whatever the model claimed.
        conf[f] = min(out.confidence if out else 0.0, 0.2) if err else (out.confidence if out else 0.0)
        if err:
            errors.append(err)
    items, li_errors = canonical_line_items(raw.line_items)
    values["line_items"] = items
    conf["line_items"] = min(raw.line_items_confidence, 0.2) if li_errors else raw.line_items_confidence
    return Canonical(values=values, conf=conf, parse_errors=errors + li_errors)


def canonicalize_record(data: dict) -> tuple[dict, list[str]]:
    """Canonicalise a whole record (e.g. one edited by a human reviewer). Returns (record, errors)."""
    record, errors = {}, []
    for f in SCALAR_FIELDS:
        v, err = canonical_scalar(f, data.get(f))
        record[f] = v
        if err:
            errors.append(err)
    items, li_errors = canonical_line_items(data.get("line_items") or [])
    record["line_items"] = items
    return record, errors + li_errors
