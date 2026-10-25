from decimal import Decimal

import pytest

from app import scoring
from app.schemas import (RawExtraction, canonical_line_items, canonical_scalar, canonicalize_record, fields_equal,
                         parse_date, parse_money, to_canonical)


# ------------------------------------------------------------------ parsing ----
@pytest.mark.parametrize("raw,expected", [
    ("$1,234.50", "1234.50"), ("1.234,50 €", "1234.50"), ("1,234", "1234.00"), ("12,50", "12.50"),
    ("(45.00)", "-45.00"), (1234.5, "1234.50"), ("USD 99", "99.00"), ("1 234,56", "1234.56"), (7, "7.00")])
def test_parse_money(raw, expected):
    assert f"{parse_money(raw):.2f}" == expected


@pytest.mark.parametrize("raw", [None, "abc", "", True, "--"])
def test_parse_money_rejects(raw):
    assert parse_money(raw) is None


def test_parse_date_accepts_unambiguous_forms_and_refuses_to_guess():
    assert parse_date("2023-11-14").isoformat() == "2023-11-14"
    assert parse_date("14 Nov 2023").isoformat() == "2023-11-14"
    assert parse_date("November 14, 2023").isoformat() == "2023-11-14"
    assert parse_date("03/04/2024") is None          # day-first or month-first? never guess
    assert parse_date("2023-02-30") is None and parse_date("garbage") is None and parse_date(None) is None


def test_canonical_scalar_normalises_and_reports_errors():
    assert canonical_scalar("currency", "usd") == ("USD", None)
    assert canonical_scalar("currency", "dollars")[0] is None and canonical_scalar("currency", "dollars")[1]
    assert canonical_scalar("total", "$1,000") == ("1000.00", None)
    assert canonical_scalar("invoice_date", "14 Nov 2023") == ("2023-11-14", None)
    assert canonical_scalar("invoice_number", "  INV-1  ") == ("INV-1", None)
    assert canonical_scalar("vendor_name", "   ") == (None, None)
    assert canonical_scalar("tax", "n/a")[1]


def test_canonical_line_items():
    items, errs = canonical_line_items([{"description": " Bolt  ", "quantity": "2.0", "unit_price": "$5", "amount": "10"}])
    assert items == [{"description": "Bolt", "quantity": "2", "unit_price": "5.00", "amount": "10.00"}] and not errs
    items, errs = canonical_line_items([{"description": "x", "quantity": "two", "unit_price": 1, "amount": 2}, "oops"])
    assert items == [] and len(errs) == 2


def test_fields_equal_is_format_and_order_insensitive():
    assert fields_equal("vendor_name", "Acme  Corp", "acme corp.")
    assert fields_equal("total", "10.00", "10.00") and not fields_equal("total", "10.00", "10.01")
    assert fields_equal("invoice_date", None, None) and not fields_equal("invoice_date", None, "2023-01-01")
    a = [{"description": "A", "quantity": "1", "unit_price": "1.00", "amount": "1.00"},
         {"description": "B", "quantity": "2", "unit_price": "1.00", "amount": "2.00"}]
    assert fields_equal("line_items", a, list(reversed(a)))
    assert not fields_equal("line_items", a, a[:1])


def test_to_canonical_clamps_confidence_and_distrusts_unparseable_values():
    raw = RawExtraction.model_validate({"fields": {
        "total": {"value": "garbage", "confidence": 0.99},       # unparseable: must not look trustworthy
        "vendor_name": {"value": "Acme", "confidence": 95},       # models sometimes answer 95 instead of 0.95
        "invoice_number": {"value": "X1", "confidence": "0.9"}}})
    c = to_canonical(raw)
    assert c.values["total"] is None and c.conf["total"] <= 0.2 and any("total" in e for e in c.parse_errors)
    assert c.conf["vendor_name"] == 1.0 and c.conf["invoice_number"] == 0.9
    assert c.conf["tax"] == 0.0                                  # a field the model omitted: zero confidence


def test_canonicalize_record_for_reviewer_edits():
    rec, errs = canonicalize_record({"vendor_name": "Acme", "total": "$10", "invoice_date": "14 Nov 2023", "line_items": []})
    assert rec["total"] == "10.00" and rec["invoice_date"] == "2023-11-14" and not errs
    _, errs = canonicalize_record({"total": "ten"})
    assert errs


# ------------------------------------------------------------------- scoring ----
def canon(values=None, conf=0.95):
    base = {"vendor_name": "Acme", "invoice_number": "INV-1", "invoice_date": "2024-01-01", "due_date": "2024-01-31",
            "currency": "USD", "subtotal": "100.00", "tax": "10.00", "total": "110.00",
            "line_items": [{"description": "A", "quantity": "2", "unit_price": "50.00", "amount": "100.00"}]}
    base.update(values or {})
    from app.schemas import Canonical, ALL_FIELDS
    return Canonical(values=base, conf={f: conf for f in ALL_FIELDS})


def test_merge_without_second_pass_is_identity():
    m = scoring.merge_passes(canon(), None)
    assert m["disputed"] == [] and m["conf"]["total"] == 0.95


def test_merge_agreement_averages_confidence_and_disagreement_collapses_it():
    c1, c2 = canon(conf=0.9), canon({"total": "111.00"}, conf=0.8)
    m = scoring.merge_passes(c1, c2)
    assert m["disputed"] == ["total"]
    assert m["conf"]["total"] == pytest.approx(0.8 * 0.5)       # min(0.9, 0.8) * penalty
    assert m["conf"]["vendor_name"] == pytest.approx(0.85)      # agreed: mean of the two
    assert m["values"]["total"] == "110.00"                      # the more confident pass wins the displayed value


def test_validators_pass_a_consistent_record():
    assert scoring.validate_record(canon().values) == []


@pytest.mark.parametrize("override,code", [
    ({"total": "999.00"}, "arith_total"),
    ({"subtotal": "90.00", "total": "100.00"}, "arith_lines"),
    ({"line_items": [{"description": "A", "quantity": "2", "unit_price": "50.00", "amount": "70.00"}], "subtotal": "70.00", "total": "80.00"}, "arith_line_item"),
    ({"due_date": "2023-12-01"}, "date_order"),
    ({"invoice_number": None}, "missing_required"),
    ({"invoice_date": "1850-01-01"}, "date_range")])
def test_validators_catch_each_error_class(override, code):
    issues = scoring.validate_record(canon(override).values)
    assert code in [i["code"] for i in issues] and all(i["severity"] == "error" for i in issues if i["code"] == code)


def test_missing_currency_is_only_a_warning():
    issues = scoring.validate_record(canon({"currency": None}).values)
    assert [(i["code"], i["severity"]) for i in issues] == [("currency_missing", "warning")]


def test_validator_hits_cap_confidence_of_implicated_fields_only():
    issues = scoring.validate_record(canon({"total": "999.00"}).values)
    capped = scoring.apply_issue_caps({f: 0.99 for f in canon().conf}, issues)
    assert capped["total"] == capped["subtotal"] == capped["tax"] == 0.4 and capped["vendor_name"] == 0.99


def test_confidently_wrong_arithmetic_is_still_routed_to_a_human():
    """The point of validators: the model was 99% sure and both passes agreed, but the numbers don't add up."""
    c = canon({"total": "999.00"}, conf=0.99)
    merged = scoring.merge_passes(c, c)                          # both passes agree, both very confident
    issues = scoring.validate_record(merged["values"])
    reasons = scoring.route(scoring.apply_issue_caps(merged["conf"], issues), merged["disputed"], issues)
    assert "issue:arith_total" in reasons and any(r.startswith("low_confidence:total") for r in reasons)


def test_route_is_empty_for_a_clean_confident_record():
    assert scoring.route({f: 0.97 for f in canon().conf}, [], []) == []


def test_audit_sampling_is_deterministic_and_close_to_the_rate():
    assert scoring.is_audit_sample("doc-1", 0.5) == scoring.is_audit_sample("doc-1", 0.5)
    hits = sum(scoring.is_audit_sample(f"k{i}", 0.1) for i in range(20000))
    assert 0.09 < hits / 20000 < 0.11
    assert not any(scoring.is_audit_sample(f"k{i}", 0.0) for i in range(100))


def test_review_priority_puts_the_most_uncertain_documents_first():
    sure = scoring.review_priority({"a": 0.9}, [], [], False)
    unsure = scoring.review_priority({"a": 0.3}, ["total"], [{"severity": "error"}], True)
    assert unsure > sure


def test_oracle_scores():
    truth = canon().values
    acc, exact = scoring.oracle_scores(dict(truth), truth)
    assert (acc, exact) == (1.0, 1)
    acc, exact = scoring.oracle_scores({**truth, "total": "1.00"}, truth)
    assert exact == 0 and acc == pytest.approx(8 / 9)
