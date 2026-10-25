"""
Synthetic invoices with ground truth, in two visually different templates.

  A  "classic":  vendor top-left, meta block top-right, table [Description | Qty | Unit Price | Amount],
                 totals bottom-right, ISO dates, "$1,234.50" money.
  B  "modern":   dark header band, left sidebar with the meta, table columns REORDERED
                 [Qty | Item | Amount | Rate], "AMOUNT DUE" box up top, "14 Nov 2023" dates, and
                 European number formatting for EUR ("1.234,50 EUR"). This is the format change.

Why synthetic: (1) a stream with known ground truth is the only way to MEASURE how fast drift detection
reacts, (2) the images are real PNGs that a real vision model can read, so the same files exercise the
Grok extractor, and (3) it needs no private documents.

Every function is deterministic in its seed.
"""
import io
import random
from datetime import date, timedelta
from decimal import Decimal, ROUND_HALF_UP

from PIL import Image, ImageDraw, ImageFont

VENDORS = ["Northwind Traders", "Acme Industrial Supply", "Globex Logistics", "Initech Software",
           "Umbrella Medical Group", "Stark Fabrication", "Wayne Freight Co", "Hooli Cloud Services",
           "Vandelay Imports", "Pied Piper Hosting", "Soylent Foods Ltd", "Cyberdyne Systems",
           "Oscorp Chemicals", "Tyrell Robotics", "Gringotts Treasury", "Monarch Office Supplies"]
ITEMS = ["Steel bracket 40mm", "Consulting hours", "Cloud storage 1TB", "Safety gloves (box)", "Freight surcharge",
         "Server rack rail kit", "Annual support plan", "Thermal paste 30g", "Pallet wrap roll", "Calibration service",
         "Network switch 24-port", "Label printer ribbon", "Maintenance visit", "USB-C docking station"]
CURRENCIES = {"USD": "$", "EUR": "€", "GBP": "£"}
CENT = Decimal("0.01")
SIZE = (850, 1100)


def _font(size: int, bold: bool = False):
    names = (["DejaVuSans-Bold.ttf", "arialbd.ttf"] if bold else []) + ["DejaVuSans.ttf", "arial.ttf", "LiberationSans-Regular.ttf"]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _q(d) -> Decimal:
    return Decimal(d).quantize(CENT, ROUND_HALF_UP)


def make_truth(rng: random.Random) -> dict:
    """Ground truth in canonical form (see app.schemas). Arithmetic is internally consistent."""
    currency = rng.choice(list(CURRENCIES))
    issued = date(2023, 1, 1) + timedelta(days=rng.randint(0, 700))
    items = []
    for desc in rng.sample(ITEMS, rng.randint(1, 7)):
        qty = rng.randint(1, 20)
        unit = _q(Decimal(rng.randint(150, 90000)) / 100)
        items.append({"description": desc, "quantity": str(qty), "unit_price": f"{unit:.2f}", "amount": f"{_q(unit * qty):.2f}"})
    subtotal = sum(Decimal(i["amount"]) for i in items)
    tax = _q(subtotal * Decimal(rng.choice(["0", "0.05", "0.08", "0.10", "0.20"])))
    return {
        "vendor_name": rng.choice(VENDORS),
        "invoice_number": f"{rng.choice(['INV', 'BL', 'SR', 'FA'])}-{rng.randint(10000, 99999)}",
        "invoice_date": issued.isoformat(),
        "due_date": (issued + timedelta(days=rng.choice([14, 30, 45, 60]))).isoformat(),
        "currency": currency,
        "subtotal": f"{subtotal:.2f}", "tax": f"{tax:.2f}", "total": f"{subtotal + tax:.2f}",
        "line_items": items,
    }


def _money(amount: str, currency: str, style: str) -> str:
    d = Decimal(amount)
    if style == "eu":   # 1.234,50 EUR
        s = f"{d:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        return f"{s} {CURRENCIES[currency]}"
    return f"{CURRENCIES[currency]}{d:,.2f}"


def _render_a(t: dict, rng: random.Random) -> Image.Image:
    img = Image.new("RGB", SIZE, "white")
    d = ImageDraw.Draw(img)
    d.text((40, 40), t["vendor_name"], fill="black", font=_font(30, True))
    d.text((40, 82), "100 Market Street, Springfield", fill="#555", font=_font(15))
    d.text((560, 40), "INVOICE", fill="black", font=_font(30, True))
    meta = [("Invoice #:", t["invoice_number"]), ("Date:", t["invoice_date"]), ("Due:", t["due_date"]), ("Currency:", t["currency"])]
    for i, (k, v) in enumerate(meta):
        d.text((560, 95 + i * 26), k, fill="#333", font=_font(16, True))
        d.text((680, 95 + i * 26), v, fill="black", font=_font(16))
    y = 260
    d.rectangle([40, y, 810, y + 32], fill="#e8e8e8")
    for x, h in [(50, "Description"), (430, "Qty"), (520, "Unit Price"), (680, "Amount")]:
        d.text((x, y + 7), h, fill="black", font=_font(16, True))
    y += 40
    for it in t["line_items"]:
        d.text((50, y), it["description"], fill="black", font=_font(16))
        d.text((430, y), it["quantity"], fill="black", font=_font(16))
        d.text((520, y), _money(it["unit_price"], t["currency"], "us"), fill="black", font=_font(16))
        d.text((680, y), _money(it["amount"], t["currency"], "us"), fill="black", font=_font(16))
        y += 30
    y = max(y + 30, 640)
    for label, key, bold in [("Subtotal", "subtotal", False), ("Tax", "tax", False), ("TOTAL", "total", True)]:
        d.text((560, y), label, fill="black", font=_font(18, bold))
        d.text((680, y), _money(t[key], t["currency"], "us"), fill="black", font=_font(18, bold))
        y += 32
    return img


def _fmt_date_b(iso: str) -> str:
    return date.fromisoformat(iso).strftime("%d %b %Y")


def _render_b(t: dict, rng: random.Random) -> Image.Image:
    img = Image.new("RGB", SIZE, "white")
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, SIZE[0], 110], fill="#1f3a5f")
    d.text((SIZE[0] // 2, 55), t["vendor_name"].upper(), fill="white", font=_font(28, True), anchor="mm")
    style = "eu" if t["currency"] == "EUR" else "us"
    d.rectangle([560, 130, 810, 215], outline="#1f3a5f", width=3)
    d.text((575, 140), "AMOUNT DUE", fill="#1f3a5f", font=_font(15, True))
    d.text((575, 168), _money(t["total"], t["currency"], style), fill="black", font=_font(26, True))
    side = [("Invoice No", t["invoice_number"]), ("Issued", _fmt_date_b(t["invoice_date"])), ("Payable by", _fmt_date_b(t["due_date"]))]
    for i, (k, v) in enumerate(side):
        d.text((40, 140 + i * 58), k, fill="#777", font=_font(14))
        d.text((40, 160 + i * 58), v, fill="black", font=_font(18, True))
    y = 330
    d.line([40, y, 810, y], fill="#1f3a5f", width=3)
    for x, h in [(50, "Qty"), (120, "Item"), (520, "Amount"), (680, "Rate")]:
        d.text((x, y + 8), h, fill="#1f3a5f", font=_font(15, True))
    y += 44
    for it in t["line_items"]:
        d.text((50, y), it["quantity"], fill="black", font=_font(16))
        d.text((120, y), it["description"], fill="black", font=_font(16))
        d.text((520, y), _money(it["amount"], t["currency"], style), fill="black", font=_font(16))
        d.text((680, y), _money(it["unit_price"], t["currency"], style), fill="black", font=_font(16))
        y += 30
    y = max(y + 30, 700)
    for label, key in [("Net", "subtotal"), ("VAT / Tax", "tax"), ("Gross", "total")]:
        d.text((40, y), label, fill="#777", font=_font(15))
        d.text((200, y), _money(t[key], t["currency"], style), fill="black", font=_font(17, label == "Gross"))
        y += 30
    return img


def render(truth: dict, template: str, rng: random.Random, noise: float = 0.0) -> Image.Image:
    img = (_render_a if template == "A" else _render_b)(truth, rng)
    if noise > 0:
        n = Image.effect_noise(img.size, 40 * noise).convert("RGB")
        img = Image.blend(img, n, 0.12 * noise)
        img = img.rotate(rng.uniform(-1.2, 1.2) * noise, fillcolor="white")
    return img


def generate_document(seed: int, template: str = "A", noise: float = 0.0) -> tuple[bytes, dict]:
    rng = random.Random(f"{seed}:{template}")
    truth = make_truth(rng)
    buf = io.BytesIO()
    render(truth, template, rng, noise).save(buf, "PNG")
    return buf.getvalue(), truth


def generate_stream(n: int, change_at: int | None = None, seed0: int = 0, noise: float = 0.0):
    """Yields (index, template, png_bytes, truth). Switches from template A to B at index `change_at`."""
    for i in range(n):
        tpl = "B" if change_at is not None and i >= change_at else "A"
        png, truth = generate_document(seed0 + i, tpl, noise)
        yield i, tpl, png, truth
