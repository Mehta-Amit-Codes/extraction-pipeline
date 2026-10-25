import json
import random
from types import SimpleNamespace

import httpx
import openai
import pytest

from app import extractors as ex
from app import scoring, synth
from app.schemas import to_canonical
from PIL import Image

REQ = httpx.Request("POST", "https://api.x.ai/v1/chat/completions")


def http(status, headers=None):
    return httpx.Response(status, request=REQ, headers=headers or {})


class FakeClient:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=r))],
                               usage=SimpleNamespace(prompt_tokens=100, completion_tokens=50))


TRUTH = synth.make_truth(random.Random(3))
GOOD = ex.example_answer(TRUTH, confidence=0.9)
IMG = Image.new("RGB", (40, 40), "white")


def req(**kw):
    return ex.ExtractionRequest(images=[IMG], doc_key="k", layout_hash="0" * 16, **kw)


# ------------------------------------------------------------------ parsing ----
def test_parse_handles_code_fences_and_bare_values():
    raw = ex.parse_model_json("```json\n" + json.dumps({"fields": {"total": "10.00", "tax": {"value": "1", "confidence": 0.9}}}) + "\n```")
    assert raw.fields["total"].confidence == 0.5 and raw.fields["tax"].confidence == 0.9   # bare value: assume unsure


@pytest.mark.parametrize("bad", ["no json here", "{not json}", '{"nofields": 1}', '{"fields": {}, "line_items": "x"}'])
def test_parse_rejects_unusable_replies(bad):
    with pytest.raises(ValueError):
        ex.parse_model_json(bad)


def test_example_answer_round_trips_through_the_parser():
    out = to_canonical(ex.parse_model_json(GOOD))
    assert out.values["total"] == TRUTH["total"] and out.values["line_items"] == TRUTH["line_items"] and not out.parse_errors


# ----------------------------------------------------------------- Grok calls ----
def test_messages_contain_prompt_examples_with_images_then_the_target():
    g = ex.GrokExtractor(client=FakeClient([]))
    example = ex.Example(1, IMG, TRUTH, "0" * 16)
    msgs = g.build_messages(req(examples=[example, example]))
    assert msgs[0]["role"] == "system" and "invoice" in msgs[0]["content"]
    assert [m["role"] for m in msgs[1:]] == ["user", "assistant", "user", "assistant", "user"]
    assert msgs[1]["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert json.loads(msgs[2]["content"])["fields"]["total"]["value"] == TRUTH["total"]       # the corrected answer
    assert msgs[-1]["content"][1]["image_url"]["detail"] == "high"


def test_second_pass_uses_the_independent_reviewer_prompt_and_optional_other_model():
    client = FakeClient([GOOD, GOOD])
    g = ex.GrokExtractor(client=client, model="m1", crosscheck_model="m2")
    g.extract(req(pass_index=1))
    g.extract(req(pass_index=2))
    assert [c["model"] for c in client.calls] == ["m1", "m2"]
    assert client.calls[0]["messages"][0]["content"] == ex.PROMPT_PASS1
    assert client.calls[1]["messages"][0]["content"] == ex.PROMPT_PASS2


def test_successful_extraction_reports_tokens_and_the_rate_limit_hook_runs_per_call():
    hook = []
    g = ex.GrokExtractor(client=FakeClient([GOOD]), before_call=lambda: hook.append(1))
    out = g.extract(req())
    assert (out.tokens_in, out.tokens_out, len(hook)) == (100, 50, 1) and out.raw.fields["total"].confidence == 0.9


def test_malformed_output_gets_one_repair_round_with_the_error_shown_to_the_model():
    client = FakeClient(["Sure! The invoice total is $10.", GOOD])
    out = ex.GrokExtractor(client=client).extract(req())
    assert len(client.calls) == 2 and out.tokens_in == 200 and out.tokens_out == 100
    repair = client.calls[1]["messages"]
    assert repair[-2] == {"role": "assistant", "content": "Sure! The invoice total is $10."}
    assert "not usable" in repair[-1]["content"]


def test_output_that_is_still_malformed_after_repair_is_retryable():
    with pytest.raises(ex.RetryableError) as e:
        ex.GrokExtractor(client=FakeClient(["nope", "still nope"])).extract(req())
    assert e.value.kind == "malformed"


@pytest.mark.parametrize("raised,expected,attrs", [
    (openai.RateLimitError("slow down", response=http(429, {"retry-after": "7"}), body=None), ex.RetryableError, {"retry_after": 7.0, "kind": "rate_limit"}),
    (openai.RateLimitError("slow down", response=http(429), body=None), ex.RetryableError, {"retry_after": None}),
    (openai.APITimeoutError(request=REQ), ex.RetryableError, {"kind": "network"}),
    (openai.APIConnectionError(request=REQ), ex.RetryableError, {"kind": "network"}),
    (openai.InternalServerError("oops", response=http(500), body=None), ex.RetryableError, {"kind": "server"}),
    (openai.AuthenticationError("bad key", response=http(401), body=None), ex.ConfigError, {}),
    (openai.PermissionDeniedError("no access", response=http(403), body=None), ex.ConfigError, {}),
    (openai.NotFoundError("no such model", response=http(404), body=None), ex.ConfigError, {}),
    (openai.BadRequestError("bad image", response=http(400), body=None), ex.PermanentError, {}),
    (RuntimeError("???"), ex.RetryableError, {"kind": "unknown"})])
def test_api_errors_map_to_the_right_retry_policy(raised, expected, attrs):
    with pytest.raises(expected) as e:
        ex.GrokExtractor(client=FakeClient([raised])).extract(req())
    for k, v in attrs.items():
        assert getattr(e.value, k) == v


def test_missing_api_key_is_a_config_error_not_a_document_failure(monkeypatch):
    monkeypatch.delenv("GROK_API_KEY", raising=False)
    with pytest.raises(ex.ConfigError):
        ex.GrokExtractor().extract(req())


def test_get_extractor_factory():
    assert ex.get_extractor("simulated").name == "simulated" and ex.get_extractor("grok").name == "grok"
    with pytest.raises(ValueError):
        ex.get_extractor("nope")


# ------------------------------------------------------------------ simulator ----
def sim_req(i, template, pass_index=1, examples=(), attempt=1, **meta):
    truth = synth.make_truth(random.Random(i))
    return ex.ExtractionRequest(images=[], doc_key=f"doc-{template}-{i}", layout_hash="0" * 16, pass_index=pass_index,
                                examples=list(examples), attempt=attempt,
                                meta={"sim_truth": truth, "sim_template": template, **meta}), truth


def accuracy(template, n=250, examples=(), pass_index=1):
    sim, accs = ex.SimulatedExtractor(latency_ms=0), []
    for i in range(n):
        r, truth = sim_req(i, template, pass_index, examples)
        accs.append(scoring.oracle_scores(to_canonical(sim.extract(r).raw).values, truth)[0])
    return sum(accs) / n


def test_simulator_is_deterministic():
    sim = ex.SimulatedExtractor(latency_ms=0)
    r, _ = sim_req(5, "B")
    assert sim.extract(r).raw == sim.extract(r).raw


def test_simulated_format_change_hurts_accuracy_and_matching_examples_recover_some_of_it():
    a, b = accuracy("A"), accuracy("B")
    shots = [ex.Example(1, IMG, TRUTH, "0" * 16)]
    b_learned = accuracy("B", examples=shots)
    assert a > 0.95 and b < a - 0.10                      # the format change really degrades the "model"
    assert b_learned > b + 0.05 and b_learned < a         # examples help, but only partway


def test_simulator_is_deliberately_overconfident_and_second_pass_is_deliberately_correlated():
    sim = ex.SimulatedExtractor(latency_ms=0)
    confident_wrong = agree_on_wrong = disagree = 0
    for i in range(120):
        r1, truth = sim_req(i, "B")
        r2, _ = sim_req(i, "B", pass_index=2)
        c1, c2 = to_canonical(sim.extract(r1).raw), to_canonical(sim.extract(r2).raw)
        for f in ("total", "subtotal", "tax", "invoice_date"):
            wrong1 = c1.values[f] != truth[f]
            confident_wrong += wrong1 and c1.conf[f] >= 0.85
            agree_on_wrong += wrong1 and c1.values[f] == c2.values[f]
            disagree += c1.values[f] != c2.values[f]
    assert confident_wrong > 0 and agree_on_wrong > 0 and disagree > 0   # cross-checking can NOT catch everything


def test_simulator_scripted_failures():
    sim = ex.SimulatedExtractor(latency_ms=0)
    r, _ = sim_req(1, "A", sim_fail={"mode": "retryable", "times": 2, "retry_after": 3}, )
    for attempt in (1, 2):
        r.attempt = attempt
        with pytest.raises(ex.RetryableError) as e:
            sim.extract(r)
        assert e.value.retry_after == 3
    r.attempt = 3
    assert sim.extract(r).raw.fields
    for mode, cls in (("permanent", ex.PermanentError), ("config", ex.ConfigError)):
        r, _ = sim_req(1, "A", sim_fail={"mode": mode})
        with pytest.raises(cls):
            sim.extract(r)


def test_simulator_refuses_real_documents():
    with pytest.raises(ex.PermanentError):
        ex.SimulatedExtractor().extract(ex.ExtractionRequest(images=[], doc_key="x", layout_hash="0" * 16, meta={}))
