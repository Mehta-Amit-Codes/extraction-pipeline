# Structured Extraction Pipeline at Scale

[![Diagram](https://img.shields.io/badge/gitdiagram-view%20architecture-blue)](https://gitdiagram.com/mehta-amit-codes/extraction-pipeline)

Batch-extract structured records (invoices here) from thousands of document images/PDFs with a vision model, and
make the output **reliable enough to use**: retries and a dead-letter queue, layered confidence scoring, a human review
loop, an active-learning loop that feeds corrections back into the prompt, and drift monitoring that notices when the
documents change shape. Includes a Streamlit UI and a one-click format-change simulation.

The extractor is **Grok vision** (OpenAI-compatible API). A **scripted simulator** stands in for it offline so the whole
system can run, be tested and be demoed with known ground truth. It is not a model; see "What was verified".

## Pipeline

```
 upload / folder watch / synthetic ──► validate (magic bytes, size, pixels) ──► content-hash dedup ──► durable queue
                                                                                                          │ atomic claim + lease
                                              ┌── few-shot: corrected examples that LOOK like this document ◄┤
                                              ▼                                                            worker pool
                                         pass 1 (+ pass 2 cross-check) ──► merge ──► validators ──► route
                                                                                          │
                       confident + valid ───────────────────────────────────────────────►├──► auto-approved ─┐ (3% randomly
                       uncertain / disputed / invalid ──► REVIEW QUEUE (most uncertain first) ◄───────────────┘  audited too)
                                                          approve / edit / reject
                                                                │ edits (and new layouts) ──► few-shot pool ──► next documents
 label-free signals over rolling windows ──► drift tests ──► alerts (Slack-style webhook)
```

## Design decisions

**Three independent confidence signals, because each has a blind spot**

| Signal | Strength | Blind spot |
|---|---|---|
| Self-reported confidence | cheap | LLMs are poorly calibrated and often confidently wrong |
| Cross-check pass (second, differently-phrased read) | disagreement is strong evidence of trouble | same model repeats its own mistakes; agreement ≠ correct |
| Validators (subtotal + tax = total, qty × price = amount, due ≥ invoice date, required fields) | deterministic; catches errors both passes agree on | only checks what arithmetic and logic can check |

Plus **audit sampling**: a random slice of auto-approved documents goes to a human anyway. Without it humans only see what the
system already doubted, so you never learn how wrong the "confident" path is and every accuracy estimate is biased toward
easy cases. The audit sample gives an unbiased estimate with a Wilson interval.

**Queue semantics** (`app/jobqueue.py`)
- Durable SQLite queue; claims are atomic (`BEGIN IMMEDIATE`), safe across threads *and* processes.
- Claims take a **lease**; a dead worker's jobs are reclaimed. Completion is **fenced**: a slow worker that lost its lease cannot
  write a second result, because the extraction row is committed in the same transaction as the completion behind an ownership check.
- Attempts are counted **at claim time**, so a poison document that crashes the worker still ends up dead-lettered.

**Retry policy follows an error taxonomy**

| Failure | Handling |
|---|---|
| rate limit (429), timeout, 5xx, network, malformed output after one repair round | retry with exponential backoff + jitter; `Retry-After` is a floor |
| unsupported/corrupt file, API rejects input | dead-letter immediately |
| bad API key, unknown model | **release without consuming an attempt**; after 3 in a row the queue pauses itself |

The last row exists because otherwise one expired key turns the entire backlog into dead letters within minutes.

**Active learning** (`app/active_learning.py`): examples are chosen by **visual layout similarity** (64-bit average hash), so a
corrected layout-A invoice is never shown on a layout-B document. An *edit* always joins the pool; an *approve* joins only if it
covers a layout the pool lacks. The pool is per namespace. Review order is uncertainty sampling.

**Drift detection from label-free signals** (`app/drift.py`): in production you cannot watch accuracy, so you watch proxies that
move when documents change: review rate, validator failure rate, cross-check dispute rate, mean confidence, null-field rate,
and **novel-layout rate** (needs no model output at all). A recent window is compared with a baseline using Fisher's exact test
(exact at tiny counts, where baseline rates sit near zero) and Welch for means. An alert needs statistical **and** practical
significance plus a minimum sample size. The novelty threshold is derived from the baseline's own nearest-neighbour distances.
A **baseline-stability check** warns when the baseline window itself contains a shift (see "A bug the live run found").

## Measured results (read the caveats)

All on **synthetic** invoices with a **scripted** extractor, so these measure the *detector's mechanics*, not your model or documents.

| Experiment | Result |
|---|---|
| 5 drift streams (300 docs, layout changes at ~150), `scripts/measure_detection.py` | layout signal detected 5/5, **median lag 10 docs**; model-output proxies 5/5, **median lag 30**; 0 alerts before the change |
| 5 stable streams (no change) | 0 streams with an alert |
| 1,500 simulated stable streams, `scripts/false_alarm_montecarlo.py` | per-metric false-alarm rate **0.07%–3.8% per 300-doc stream**, rising with the metric's baseline noise |
| `kill -9` on a worker mid-batch (80 docs) | 4 jobs left on dead leases; a second worker reclaimed them; **80 done, 0 dead, 80 extraction rows over 80 distinct documents** |
| Layout hash on rendered invoices | same template ≤ 9 bits apart, different template ≥ 17 bits, scan noise ≤ 3 bits |

Honest reading of these numbers:
- Lag is **quantised** to the evaluation interval (every 20 documents).
- The false-alarm rate is **not** the 1% that `alpha=0.01` suggests: checking the stream repeatedly gives the test many chances to fire.
  With six metrics tested together, expect a few percent of *stable* 300-document streams to raise a spurious alert. Raise
  `MIN_EFFECT_*` or lower `DRIFT_ALPHA` if that is too noisy, and **measure it on your own stable traffic**.
- "0/5 stable streams" alone is weak evidence (it is compatible with a rate up to ~40%); the Monte Carlo is the better estimate.
- Throughput (~11 docs/s here) is limited by synthetic image rendering on one CPU, not by the pipeline. Real throughput will be
  bound by model-API latency; use `RATE_LIMIT_RPS` and more workers. It was not load-tested at real API latencies.

### A bug the live run found
Driving the UI against a live server, the simulation put the format change at document #40 with a 60-document baseline, so a third
of the baseline was already the new layout. Detection **silently went blind**: no layout alert, 59-document lag. The limitation was
documented but failed silently. Now: `evaluate()` compares the baseline's own halves and reports `baseline_stable: false` with the
reason, the UI shows a warning banner, and the simulation form refuses a change point inside the baseline window.

## What was verified, and what wasn't

| Part | Status |
|---|---|
| Parsing/canonicalisation, validators, routing, audit sampling | 36 tests |
| Queue: concurrent claims (8 threads × 200 jobs, once each), leases, crash loops, backoff, `Retry-After`, fencing, release, requeue | 15 tests |
| Pipeline end to end: retries, DLQ, permanent failures, config circuit breaker, stale-lease takeover, exactly-once under threads, audit routing, few-shot selection | 16 tests |
| Grok extractor against a fake client incl. **real `openai` exception classes** (error → retry policy mapping), repair round, prompt/example construction | tested |
| Review, active-learning loop (before/after on the *same* documents), calibration, audit CI | 16 tests |
| Drift statistics vs hand-computed values, novelty, stable vs drifting streams, dedup, ack, pin, baseline contamination, webhook | 22 tests |
| HTTP API: uploads (disguised executables rejected), auth, namespace isolation, pause/requeue/reprocess, export | 17 tests |
| Real processes: `kill -9` crash recovery; live API with embedded workers; UI driven headlessly through generate → review → simulate → alerts | run manually |
| **Real Grok vision calls** | **not exercised** (no network/key here). The request construction, error mapping and parsing are tested against fakes; expect to tune prompts and thresholds on first real run |
| **Real documents / real PDFs from the wild** | **not exercised**; PDF rendering was tested on generated PDFs |
| **The simulator's response to examples** | scripted: the "learning" before/after validates the *plumbing* (examples selected and passed), **not real model learning** |

## Setup

```bash
cp .env.example .env              # EXTRACTOR=simulated works offline; set EXTRACTOR=grok + GROK_API_KEY for real extraction
pip install -r requirements.txt
uvicorn app.main:app --port 8006  # API + embedded workers
streamlit run streamlit_app.py    # UI on :8501
pip install -r requirements-dev.txt && pytest          # 161 tests, no network, no API key
```

Scale out: set `EMBEDDED_WORKERS=0` and run several `python -m app.cli worker --concurrency 8` processes.
CLI: `app.cli ingest <folder> [--watch]` (idempotent folder ingestion), `app.cli simulate --n 300 --change-at 150`.
Reproduce the measurements: `python scripts/measure_detection.py` and `python scripts/false_alarm_montecarlo.py`.

## Demo script (≈5 minutes)

1. **Pipeline** → generate 120 synthetic invoices; watch the queue drain. Add "transient failure every 7th" to see retries recover and "unprocessable every 25th" to fill the dead-letter queue.
2. **Review** → the queue is ordered most-uncertain-first. Open one: image on the left, editable fields with confidence badges on the right, ⚔️ where the two passes disagreed, validator errors above. Correct something and save.
3. **Quality & Drift** → straight-through rate, flagged rate, audit-estimated accuracy with its interval, calibration table (does 0.9 mean ~90%?).
4. **Simulation** → 300 documents, change at 150. The label-free *novel-layout* signal fires first (~10 docs), the confidence-based proxies later (~30). Try 100 / change at 40 to see the guard that refuses to contaminate the baseline.
5. **Learning** → run the demo reviewer, then reprocess documents that are not in the pool and compare before/after on the same documents.

## Known limitations

- **SQLite queue is single-node** (one writer). The queue sits behind `app/jobqueue.py`; moving to Postgres (`SELECT … FOR UPDATE SKIP LOCKED`) or Celery + Redis/RabbitMQ means re-implementing that one module. Celery was *not* used; this is a deliberate trade-off for zero infrastructure, not an oversight.
- **No lease heartbeat**: `LEASE_S` (default 600) must exceed the worst case (2 passes × (call + repair) × 90 s API timeout = 360 s) or a live worker's result is discarded.
- **Drift windows are count-based**, not time-based; the baseline is the first `BASELINE_N` extractions unless pinned. Gradual drift moves the baseline-vs-recent gap slowly.
- **Thresholds and the layout hash were tuned on two synthetic templates.** Real document families will need recalibration from your own stable traffic.
- Only the **first page** feeds the layout fingerprint and few-shot examples; `line_items` correctness is all-or-nothing per document.
- The `arith_lines` validator assumes subtotal = sum of line amounts; invoices with shipping/discount lines can legitimately fail it. Tune `SEVERITY` in `app/scoring.py`.
- Reviewer identity is a free-text name, and the API has one shared key; there is no per-user auth.

## Layout

```
app/
  config.py          settings (all env-overridable)      schemas.py   canonical forms, parsing, comparison
  imaging.py         safe loading, layout fingerprint    synth.py     synthetic invoices + ground truth
  db.py / store.py   SQLite schema and repository        jobqueue.py  leases, retries, dead letters, fencing
  extractors.py      Grok vision + scripted simulator    scoring.py   merge, validators, routing, audit sampling
  pipeline.py        one job end to end                  worker.py    pool, rate limit, circuit breaker
  review.py          approve/edit/reject + demo oracle   active_learning.py   example selection and admission
  quality.py         audit accuracy, calibration         drift.py     proxy metrics, tests, alerts, baseline check
  main.py            FastAPI                             cli.py       worker / ingest / simulate
streamlit_app.py     UI
scripts/             measure_detection.py, false_alarm_montecarlo.py
tests/               161 tests
```
