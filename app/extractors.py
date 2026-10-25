"""
Extractors: image(s) -> RawExtraction (per-field value + self-reported confidence).

  GrokExtractor       the real thing: Grok vision through its OpenAI-compatible API.
  SimulatedExtractor  a SCRIPTED STAND-IN FOR A MODEL, not a model. It exists so the pipeline, drift
                      detection and active-learning plumbing can be run, tested and demoed offline with a
                      known ground truth. Its accuracy numbers say nothing about real model quality.

The simulator is deliberately UNFLATTERING, so the safeguards are not tested against an easy opponent:
  * errors are often CONFIDENTLY WRONG (40% of wrong values still report high confidence)
  * the second pass repeats the first pass's mistake half the time (same model, correlated failures),
    so cross-checking cannot catch everything
  * after a layout change error rates jump, and corrected few-shot examples bring them down only partway

Error taxonomy (drives retry policy, see jobqueue.py):
  RetryableError  rate limit (honours Retry-After), timeout, 5xx, connection, malformed model output
  PermanentError  this document can never succeed (unsupported / rejected as invalid input)
  ConfigError     the deployment is misconfigured (bad key, unknown model): not the document's fault
"""
import json
import random
import re
import time
from dataclasses import dataclass, field

from app import imaging
from app.config import CROSSCHECK_MODEL, EXTRACTION_MODEL, FEWSHOT_MAX_DIST, GROK_BASE_URL, SIM_LATENCY_MS
from app.schemas import ALL_FIELDS, SCALAR_FIELDS, FieldOut, RawExtraction


class ExtractorError(Exception):
    pass


class RetryableError(ExtractorError):
    def __init__(self, msg: str, retry_after: float | None = None, kind: str = "transient"):
        super().__init__(msg)
        self.retry_after, self.kind = retry_after, kind


class PermanentError(ExtractorError):
    kind = "permanent"


class ConfigError(ExtractorError):
    kind = "config"


@dataclass
class Example:
    document_id: int
    image: object            # PIL.Image (first page)
    record: dict             # human-verified canonical record
    layout_hash: str


@dataclass
class ExtractionRequest:
    images: list
    doc_key: str                                   # stable id (content hash): seeds the simulator
    layout_hash: str
    meta: dict = field(default_factory=dict)
    examples: list[Example] = field(default_factory=list)
    pass_index: int = 1                            # 1 = primary pass, 2 = cross-check pass
    attempt: int = 1                               # job attempt number (lets the simulator script transient failures)


@dataclass
class ExtractionOutput:
    raw: RawExtraction
    tokens_in: int
    tokens_out: int
    latency_ms: float
    model: str


# =================================================================== prompts ====
_SCHEMA = """Return ONLY one JSON object, no prose and no code fences:
{
  "fields": {
    "vendor_name":    {"value": <string|null>, "confidence": <0..1>},
    "invoice_number": {"value": <string|null>, "confidence": <0..1>},
    "invoice_date":   {"value": <"YYYY-MM-DD"|null>, "confidence": <0..1>},
    "due_date":       {"value": <"YYYY-MM-DD"|null>, "confidence": <0..1>},
    "currency":       {"value": <3-letter ISO code|null>, "confidence": <0..1>},
    "subtotal":       {"value": <number|null>, "confidence": <0..1>},
    "tax":            {"value": <number|null>, "confidence": <0..1>},
    "total":          {"value": <number|null>, "confidence": <0..1>}
  },
  "line_items": [{"description": <string>, "quantity": <number>, "unit_price": <number>, "amount": <number>}],
  "line_items_confidence": <0..1>
}"""

_RULES = """Rules:
- Copy values exactly as printed; never compute or infer a number that is not on the page. Use null if absent.
- Dates as YYYY-MM-DD. If a date is ambiguous (e.g. 03/04/2024 could be day-first or month-first), output null with confidence <= 0.3.
- Numbers as plain decimals: no currency symbols and no thousands separators (1.234,50 -> 1234.50). Currency as an ISO code inferred from the symbol or text.
- Confidence is your honest probability the value is exactly right. Use >= 0.95 only when the text is clearly legible and unambiguous; ~0.5 when guessing; <= 0.3 when you can barely read it."""

PROMPT_PASS1 = "You extract structured data from invoice images.\n" + _RULES + "\n" + _SCHEMA
PROMPT_PASS2 = ("You are an independent second reviewer re-reading an invoice image from scratch. First locate each value "
                "on the page, read it character by character, then report it. Work bottom-up: totals first, then line items, "
                "then header fields.\n" + _RULES + "\n" + _SCHEMA)


def example_answer(record: dict, confidence: float = 0.97) -> str:
    """Render a human-verified record in the model's output format (used as the assistant turn of a few-shot example)."""
    fields = {f: {"value": record.get(f), "confidence": confidence} for f in SCALAR_FIELDS}
    items = [{"description": i["description"], "quantity": float(i["quantity"]),
              "unit_price": float(i["unit_price"]), "amount": float(i["amount"])} for i in record.get("line_items", [])]
    return json.dumps({"fields": fields, "line_items": items, "line_items_confidence": confidence})


def parse_model_json(text: str) -> RawExtraction:
    """Tolerant parse of the model's reply. Raises ValueError with a message suitable for a repair prompt."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found in the reply")
    try:
        obj = json.loads(cleaned[start:end + 1])
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid JSON: {e}")
    if not isinstance(obj, dict) or not isinstance(obj.get("fields"), dict):
        raise ValueError('missing the top-level "fields" object')
    fields = {}
    for k, v in obj["fields"].items():
        fields[k] = v if isinstance(v, dict) else {"value": v, "confidence": 0.5}  # bare value: assume unsure
    items = obj.get("line_items") or []
    if not isinstance(items, list):
        raise ValueError('"line_items" must be a list')
    return RawExtraction(fields=fields, line_items=items, line_items_confidence=obj.get("line_items_confidence", 0.0))


# ======================================================================== Grok ====
class GrokExtractor:
    name = "grok"

    def __init__(self, client=None, model: str | None = None, crosscheck_model: str | None = None,
                 before_call=None):
        self.model = model or EXTRACTION_MODEL
        self.crosscheck_model = crosscheck_model or CROSSCHECK_MODEL or self.model
        self.before_call = before_call or (lambda: None)  # worker injects the shared rate limiter
        self._client = client

    @property
    def client(self):
        if self._client is None:
            import os
            from openai import OpenAI
            key = os.environ.get("GROK_API_KEY")
            if not key:
                raise ConfigError("GROK_API_KEY is not set")
            self._client = OpenAI(api_key=key, base_url=GROK_BASE_URL, timeout=90, max_retries=0)  # we own retries
        return self._client

    def build_messages(self, req: ExtractionRequest) -> list[dict]:
        system = PROMPT_PASS2 if req.pass_index == 2 else PROMPT_PASS1
        msgs = [{"role": "system", "content": system}]
        for ex in req.examples:  # corrected, human-verified examples that LOOK like this document
            msgs.append({"role": "user", "content": [
                {"type": "text", "text": "Extract this invoice."},
                {"type": "image_url", "image_url": {"url": imaging.to_data_uri(ex.image), "detail": "high"}}]})
            msgs.append({"role": "assistant", "content": example_answer(ex.record)})
        msgs.append({"role": "user", "content": [{"type": "text", "text": "Extract this invoice."}] + [
            {"type": "image_url", "image_url": {"url": imaging.to_data_uri(img), "detail": "high"}} for img in req.images]})
        return msgs

    def _call(self, model: str, messages: list[dict], temperature: float):
        self.before_call()
        try:
            resp = self.client.chat.completions.create(model=model, messages=messages, temperature=temperature, max_tokens=1500)
        except ExtractorError:
            raise
        except Exception as e:
            raise self._classify(e)
        return (resp.choices[0].message.content or ""), resp.usage.prompt_tokens, resp.usage.completion_tokens

    @staticmethod
    def _classify(e: Exception) -> ExtractorError:
        import openai
        if isinstance(e, openai.RateLimitError):
            ra = None
            try:
                ra = float(e.response.headers.get("retry-after"))
            except (AttributeError, TypeError, ValueError):
                pass
            return RetryableError(f"rate limited: {e}", retry_after=ra, kind="rate_limit")
        if isinstance(e, (openai.APITimeoutError, openai.APIConnectionError)):
            return RetryableError(f"network/timeout: {e}", kind="network")
        if isinstance(e, (openai.AuthenticationError, openai.PermissionDeniedError, openai.NotFoundError)):
            return ConfigError(f"{type(e).__name__}: {e}")  # bad key / no access / unknown model
        if isinstance(e, openai.BadRequestError):
            return PermanentError(f"rejected by the API as invalid input: {e}")
        if isinstance(e, openai.APIStatusError) and e.status_code >= 500:
            return RetryableError(f"server error {e.status_code}", kind="server")
        return RetryableError(f"{type(e).__name__}: {e}", kind="unknown")  # unknown: bounded by max attempts

    def extract(self, req: ExtractionRequest) -> ExtractionOutput:
        model = self.crosscheck_model if req.pass_index == 2 else self.model
        temperature = 0.2 if req.pass_index == 2 else 0.0
        messages = self.build_messages(req)
        t0 = time.perf_counter()
        text, tin, tout = self._call(model, messages, temperature)
        try:
            raw = parse_model_json(text)
        except Exception as first_error:
            # One repair round: show the model its own output and the exact problem. Cheaper than a full retry.
            messages += [{"role": "assistant", "content": text},
                         {"role": "user", "content": f"That reply was not usable ({first_error}). "
                                                     "Return ONLY the corrected JSON object."}]
            text2, tin2, tout2 = self._call(model, messages, 0.0)
            tin, tout = tin + tin2, tout + tout2
            try:
                raw = parse_model_json(text2)
            except Exception as e:
                raise RetryableError(f"model returned unusable output twice: {e}", kind="malformed")
        return ExtractionOutput(raw, tin, tout, (time.perf_counter() - t0) * 1000, model)


# ================================================================== simulator ====
# P(error) per field. Template A is the baseline the "model" was built for; template B is the format change.
_P_ERR = {
    "A": {"vendor_name": .01, "invoice_number": .02, "invoice_date": .02, "due_date": .02, "currency": .01,
          "subtotal": .02, "tax": .02, "total": .02, "line_items": .04},
    "B": {"vendor_name": .04, "invoice_number": .10, "invoice_date": .35, "due_date": .35, "currency": .20,
          "subtotal": .25, "tax": .25, "total": .30, "line_items": .55},
}
_FEWSHOT_RELIEF = 0.3        # with matching corrected examples in the prompt, error rates drop to 30% of baseline
_OVERCONFIDENT = 0.4         # share of wrong values that still report high confidence
_PASS2_CORRELATION = 0.5     # share of pass-1 errors that pass 2 repeats


def _rng(*parts) -> random.Random:
    return random.Random(":".join(str(p) for p in parts))


def _corrupt(field: str, truth_value, rng: random.Random):
    """A plausible wrong value for the field."""
    if field == "line_items":
        items = [dict(i) for i in truth_value]
        if len(items) > 1 and rng.random() < 0.4:
            items.pop(rng.randrange(len(items)))                            # dropped a row
        else:
            for i in items:                                                  # swapped Unit Price/Amount columns
                i["unit_price"], i["amount"] = i["amount"], i["unit_price"]
        return items
    if truth_value is None:
        return "N/A"
    s = str(truth_value)
    if field in ("invoice_date", "due_date"):
        y, m, d = s.split("-")
        return f"{y}-{d}-{m}" if int(d) <= 12 else f"{y}-{m}-{int(d) - 1 or 1:02d}"   # day/month swap or off by one
    if field in ("subtotal", "tax", "total"):
        digits = list(s.replace(".", ""))
        if len(digits) > 2:
            i = rng.randrange(len(digits) - 1)
            digits[i], digits[i + 1] = digits[i + 1], digits[i]            # transposed digits
        return f"{''.join(digits[:-2]) or '0'}.{''.join(digits[-2:])}"
    if field == "currency":
        return {"USD": "EUR", "EUR": "USD", "GBP": "USD"}.get(s, "USD")
    if field == "invoice_number":
        return s.replace("0", "O", 1) if "0" in s else s[:-1]
    return s[: max(3, len(s) - 4)]                                          # vendor truncated


def _fmt_noise(field: str, value, rng: random.Random):
    """Occasionally emit the value the way it is printed, to exercise canonicalisation."""
    if value is None or isinstance(value, list):
        return value
    if field in ("subtotal", "tax", "total") and rng.random() < 0.3:
        return f"${float(value):,.2f}"
    if field in ("invoice_date", "due_date") and rng.random() < 0.3:
        from datetime import date
        return date.fromisoformat(value).strftime("%d %b %Y")
    return value


class SimulatedExtractor:
    name = "simulated"

    def __init__(self, latency_ms: int | None = None):
        self.latency_ms = SIM_LATENCY_MS if latency_ms is None else latency_ms

    def _outcome(self, req: ExtractionRequest, f: str, pass_index: int, p_err: float, truth):
        rng = _rng("sim", req.doc_key, f, pass_index)
        if rng.random() < p_err:
            wrong = _corrupt(f, truth, rng)
            conf = rng.uniform(0.86, 0.97) if rng.random() < _OVERCONFIDENT else rng.uniform(0.35, 0.80)
            return wrong, conf, True
        return truth, rng.uniform(0.88, 0.99), False

    def extract(self, req: ExtractionRequest) -> ExtractionOutput:
        script = req.meta.get("sim_fail")  # scripted failures for tests/demos
        if script:
            if script["mode"] == "permanent":
                raise PermanentError("scripted permanent failure")
            if script["mode"] == "config":
                raise ConfigError("scripted config failure (e.g. invalid API key)")
            if script["mode"] == "retryable" and req.attempt <= script.get("times", 1):
                raise RetryableError("scripted transient failure", retry_after=script.get("retry_after"), kind="rate_limit")
        truth = req.meta.get("sim_truth")
        if truth is None:
            raise PermanentError("the simulated extractor only handles synthetic documents (no sim_truth in meta)")
        t0 = time.perf_counter()
        if self.latency_ms:
            time.sleep(self.latency_ms / 1000)

        template = req.meta.get("sim_template", "A")
        close = [e for e in req.examples if imaging.hamming(e.layout_hash, req.layout_hash) <= FEWSHOT_MAX_DIST]
        relief = _FEWSHOT_RELIEF if close else 1.0
        probs = {f: min(0.95, p * relief) if template == "B" else p for f, p in _P_ERR[template].items()}

        fields, li_value, li_conf = {}, None, 0.0
        for f in ALL_FIELDS:
            value, conf, errored = self._outcome(req, f, req.pass_index, probs[f], truth[f])
            if req.pass_index == 2:  # correlated failure: the same "model" often repeats its own mistake
                v1, c1, e1 = self._outcome(req, f, 1, probs[f], truth[f])
                if e1 and _rng("corr", req.doc_key, f).random() < _PASS2_CORRELATION:
                    value, conf = v1, c1
            if f == "line_items":
                li_value, li_conf = value, conf
            else:
                fields[f] = FieldOut(value=_fmt_noise(f, value, _rng("fmt", req.doc_key, f, req.pass_index)), confidence=conf)

        raw = RawExtraction(fields=fields, line_items=[dict(i) for i in li_value], line_items_confidence=li_conf)
        return ExtractionOutput(raw, 1200 + 350 * len(req.examples), 380, (time.perf_counter() - t0) * 1000, "simulated")


def get_extractor(name: str | None = None, before_call=None):
    from app.config import EXTRACTOR
    name = name or EXTRACTOR
    if name == "grok":
        return GrokExtractor(before_call=before_call)
    if name == "simulated":
        return SimulatedExtractor()
    raise ValueError(f"unknown extractor '{name}' (expected 'grok' or 'simulated')")
