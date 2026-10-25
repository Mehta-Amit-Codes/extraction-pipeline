import io
import itertools
import random

import pytest
from PIL import Image

from app import imaging, synth
from app.schemas import fields_equal, ALL_FIELDS


def png(size=(50, 50), color="white"):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "PNG")
    return buf.getvalue()


def test_mime_sniffing_ignores_extensions_and_rejects_everything_else():
    assert imaging.sniff_mime(png()) == "image/png"
    assert imaging.sniff_mime(b"\xff\xd8\xff\xe0rest") == "image/jpeg" and imaging.sniff_mime(b"%PDF-1.7 ...") == "application/pdf"
    for bad in (b"MZ\x90\x00 an exe", b"<html></html>", b"", b"GIF89a"):
        with pytest.raises(imaging.UnsupportedDocument):
            imaging.sniff_mime(bad)


def test_oversized_images_are_rejected_before_decoding(monkeypatch):
    monkeypatch.setattr(imaging, "MAX_PIXELS", 1000)
    with pytest.raises(imaging.UnsupportedDocument, match="too large"):
        imaging.load_images(png((100, 100)))


def test_corrupt_images_are_unsupported_not_crashes():
    with pytest.raises(imaging.UnsupportedDocument):
        imaging.load_images(b"\x89PNG\r\n\x1a\n" + b"garbage" * 10)


def test_large_images_are_downscaled_to_bound_cost(monkeypatch):
    monkeypatch.setattr(imaging, "MAX_SIDE_PX", 400)
    [page] = imaging.load_images(png((1600, 800)))
    assert max(page.size) == 400 and page.size == (400, 200)


def test_pdfs_are_rendered_page_by_page_up_to_the_page_cap(monkeypatch):
    import pymupdf
    doc = pymupdf.open()
    for i in range(5):
        doc.new_page().insert_text((72, 72), f"page {i}")
    data = doc.tobytes()
    monkeypatch.setattr(imaging, "MAX_PAGES", 3)
    assert len(imaging.load_images(data)) == 3
    with pytest.raises(imaging.UnsupportedDocument):
        imaging.load_images(b"%PDF-1.4 truncated nonsense")


def test_layout_hash_is_stable_and_hamming_works():
    a, b = synth.generate_document(1, "A")[0], synth.generate_document(1, "A")[0]
    ha, hb = (imaging.layout_hash(imaging.load_images(x)[0]) for x in (a, b))
    assert ha == hb and len(ha) == 16 and imaging.hamming(ha, hb) == 0
    assert imaging.hamming("0" * 16, "f" * 16) == 64 and imaging.hamming("0" * 16, "1" + "0" * 15) == 1


def test_the_chosen_layout_hash_separates_templates_with_a_margin():
    """The measurement behind the hash choice: same-template spread must stay well below cross-template distance."""
    h = {t: [imaging.layout_hash(imaging.load_images(synth.generate_document(i + (0 if t == "A" else 900), t)[0])[0])
             for i in range(25)] for t in "AB"}
    intra = max(imaging.hamming(a, b) for t in "AB" for a, b in itertools.combinations(h[t], 2))
    inter = min(imaging.hamming(a, b) for a in h["A"] for b in h["B"])
    assert intra <= 12 and inter >= 14 and inter - intra >= 4
    noisy = imaging.layout_hash(imaging.load_images(synth.generate_document(7, "A", noise=1.0)[0])[0])
    assert min(imaging.hamming(noisy, a) for a in h["A"]) <= 8      # scan noise does not look like a new template


def test_synthetic_truth_is_internally_consistent_and_canonical():
    for seed in range(40):
        t = synth.make_truth(random.Random(seed))
        from decimal import Decimal
        assert sum(Decimal(i["amount"]) for i in t["line_items"]) == Decimal(t["subtotal"])
        assert Decimal(t["subtotal"]) + Decimal(t["tax"]) == Decimal(t["total"])
        for i in t["line_items"]:
            assert Decimal(i["quantity"]) * Decimal(i["unit_price"]) == Decimal(i["amount"])
        from app.schemas import canonicalize_record
        rec, errs = canonicalize_record(t)
        assert not errs and all(fields_equal(f, rec[f], t[f]) for f in ALL_FIELDS)   # truth is already canonical


def test_generation_is_deterministic_and_templates_differ():
    assert synth.generate_document(3, "A") == synth.generate_document(3, "A")
    assert synth.generate_document(3, "A")[0] != synth.generate_document(3, "B")[0]


def test_stream_switches_template_at_the_change_point():
    tpls = [t for _, t, _, _ in synth.generate_stream(10, change_at=6)]
    assert tpls == list("AAAAAABBBB")
    assert {t for _, t, _, _ in synth.generate_stream(5)} == {"A"}
