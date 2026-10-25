"""
Document imaging.

Inputs are UNTRUSTED files, so loading is defensive: magic-byte sniffing (never trust the extension or
the client's content-type), a pixel cap against decompression bombs, and a page cap on PDFs.

Layout fingerprint: an 8x8 average hash (64 bits, 16 hex chars) of the first page. It is a cheap proxy for
"what template is this", used for two things:
  * few-shot selection: show the model corrected examples that LOOK like the new document
  * drift: a label-free signal that documents stopped looking like the baseline
Chosen by measurement on the synthetic templates (see README): best relative separation between
"same template, different content" (<= 9 bits) and "different template" (>= 17 bits), and robust to
scan noise (<= 3 bits). Real document families will differ; recalibrate from your own baseline.
"""
import base64
import io

from PIL import Image

from app.config import MAX_PAGES, MAX_PIXELS, MAX_SIDE_PX


class UnsupportedDocument(Exception):
    """Not an image/PDF we can process. Permanent: retrying will not help."""


def sniff_mime(data: bytes) -> str:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"%PDF"):
        return "application/pdf"
    raise UnsupportedDocument("unsupported file type (expected PNG, JPEG or PDF)")


def _check_size(img: Image.Image) -> None:
    if img.width * img.height > MAX_PIXELS:
        raise UnsupportedDocument(f"image too large ({img.width}x{img.height}); limit is {MAX_PIXELS:,} pixels")


def _pdf_pages(data: bytes) -> list[Image.Image]:
    try:
        import pymupdf
    except ImportError as e:
        raise UnsupportedDocument("PDF support needs the 'pymupdf' package") from e
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as e:
        raise UnsupportedDocument(f"unreadable PDF: {e}") from e
    pages = []
    try:
        for page in list(doc)[:MAX_PAGES]:
            pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2))  # ~144 dpi
            pages.append(Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB"))
    finally:
        doc.close()
    if not pages:
        raise UnsupportedDocument("PDF has no pages")
    return pages


def load_images(data: bytes) -> list[Image.Image]:
    """Decode a document into RGB page images (first MAX_PAGES), downscaled to MAX_SIDE_PX."""
    mime = sniff_mime(data)
    if mime == "application/pdf":
        pages = _pdf_pages(data)
    else:
        try:
            img = Image.open(io.BytesIO(data))
            _check_size(img)  # before decoding pixels
            pages = [img.convert("RGB")]
        except UnsupportedDocument:
            raise
        except Exception as e:
            raise UnsupportedDocument(f"unreadable image: {e}") from e
    out = []
    for p in pages:
        if max(p.size) > MAX_SIDE_PX:  # bound token cost and latency
            scale = MAX_SIDE_PX / max(p.size)
            p = p.resize((int(p.width * scale), int(p.height * scale)), Image.LANCZOS)
        out.append(p)
    return out


def to_data_uri(img: Image.Image, quality: int = 85) -> str:
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def layout_hash(img: Image.Image) -> str:
    """64-bit average hash as 16 hex chars. Bit = 1 where the cell is darker than the page average."""
    px = img.convert("L").resize((8, 8), Image.LANCZOS).tobytes()
    mean = sum(px) / 64
    bits = 0
    for p in px:
        bits = (bits << 1) | (1 if p < mean else 0)
    return f"{bits:016x}"


def hamming(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")
