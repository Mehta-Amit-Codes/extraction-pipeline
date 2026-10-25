"""
How often does the drift RULE raise an alert on pure noise?

Repeatedly testing a stream as it grows inflates false alarms (every evaluation is another chance). This simulates
streams whose true rates NEVER change and applies the exact production decision rule (Fisher / Welch + alpha + minimum
effect size + minimum sample, evaluated every DRIFT_EVAL_EVERY documents), counting the fraction of streams that alert
at least once. Cheap and fast, so it can use thousands of trials, unlike full pipeline runs.

    python scripts/false_alarm_montecarlo.py
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import BASELINE_N, DRIFT_ALPHA, DRIFT_EVAL_EVERY, MIN_EFFECT_CONF, MIN_EFFECT_PROP, MIN_SAMPLES, WINDOW_N
from app.drift import fisher_one_sided, welch_one_sided

TRIALS, N = 1500, 300
rng = random.Random(7)


def mean_sd(v):
    m = sum(v) / len(v)
    return m, (sum((x - m) ** 2 for x in v) / (len(v) - 1)) ** 0.5


def stream_alerts_prop(p):
    xs = [rng.random() < p for _ in range(N)]
    base = xs[:BASELINE_N]
    for i in range(BASELINE_N + MIN_SAMPLES, N + 1, DRIFT_EVAL_EVERY):
        cur = xs[max(BASELINE_N, i - WINDOW_N):i]
        x1, x2 = sum(base), sum(cur)
        if (x2 / len(cur) - x1 / len(base)) >= MIN_EFFECT_PROP and fisher_one_sided(x1, len(base), x2, len(cur)) < DRIFT_ALPHA:
            return True
    return False


def stream_alerts_cont(mu, sd):
    xs = [rng.gauss(mu, sd) for _ in range(N)]
    base = xs[:BASELINE_N]
    m1, s1 = mean_sd(base)
    for i in range(BASELINE_N + MIN_SAMPLES, N + 1, DRIFT_EVAL_EVERY):
        cur = xs[max(BASELINE_N, i - WINDOW_N):i]
        m2, s2 = mean_sd(cur)
        if (m1 - m2) >= MIN_EFFECT_CONF and welch_one_sided(m1, s1, len(base), m2, s2, len(cur), "down") < DRIFT_ALPHA:
            return True
    return False


print(f"{TRIALS} stable streams of {N} docs; evaluated every {DRIFT_EVAL_EVERY}; alpha={DRIFT_ALPHA}, "
      f"min effect {MIN_EFFECT_PROP:.0%} (rates) / {MIN_EFFECT_CONF} (confidence)\n")
print("per-metric false-alarm rate (share of stable streams that ever alert):")
for p in (0.02, 0.10, 0.20, 0.35):
    r = sum(stream_alerts_prop(p) for _ in range(TRIALS)) / TRIALS
    print(f"  rate metric, true rate {p:>4.0%}:            {r:6.2%}")
for sd in (0.05, 0.10):
    r = sum(stream_alerts_cont(0.9, sd) for _ in range(TRIALS)) / TRIALS
    print(f"  mean confidence, per-doc sd {sd:.2f}:       {r:6.2%}")
print("\nSix metrics are tested together in production, so the per-stream rate is roughly the sum of the above for one\n"
      "stream's true rates; compare it with your tolerance for pages, and raise MIN_EFFECT_* or lower alpha if too high.")
