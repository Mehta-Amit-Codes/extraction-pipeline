"""
Reproduce the drift-detection numbers instead of trusting a single run.

Runs K STABLE streams (no format change) to count false alarms, and K DRIFT streams (A -> B at --change-at) to get
the detection-lag distribution per signal. Each run uses its own throwaway database, in parallel processes.

    python scripts/measure_detection.py --runs 5 --n 300 --change-at 150

Caveat to keep in mind: lag is quantised to the evaluation interval (DRIFT_EVAL_EVERY, default 20 documents), and
everything here is on SYNTHETIC documents with a SCRIPTED extractor. It measures the detector's mechanics, not how
your real model/documents will behave. Re-measure on your own stable traffic before trusting the thresholds.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(kind: str, seed0: int, n: int, change_at: int, concurrency: int) -> dict:
    tmp = tempfile.mkdtemp(prefix=f"measure-{kind}-")
    env = {**os.environ, "PIPELINE_DB": f"{tmp}/p.db", "DATA_DIR": f"{tmp}/data"}
    cmd = [sys.executable, "-m", "app.cli", "simulate", "--n", str(n), "--seed0", str(seed0), "--json",
           "--concurrency", str(concurrency), "--change-at", str(change_at)] + (["--no-change"] if kind == "stable" else [])
    out = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, check=True).stdout.strip().splitlines()[-1]
    return json.loads(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--change-at", type=int, default=150)
    ap.add_argument("--concurrency", type=int, default=3)
    a = ap.parse_args()
    jobs = [("stable", 1000 + i) for i in range(a.runs)] + [("drift", 2000 + i) for i in range(a.runs)]
    with ThreadPoolExecutor(max_workers=min(len(jobs), os.cpu_count() or 2)) as pool:
        results = list(pool.map(lambda j: (j[0], run(j[0], j[1], a.n, a.change_at, a.concurrency)), jobs))

    stable = [r for k, r in results if k == "stable"]
    drift = [r for k, r in results if k == "drift"]
    print(f"STABLE streams ({len(stable)} x {a.n} docs, no change):")
    print(f"  streams with at least one alert (false alarm): {sum(1 for r in stable if r['first_alert_by_metric'])}/{len(stable)}")
    for r in stable:
        if r["first_alert_by_metric"]:
            print(f"    seed {r['seed0']}: {r['first_alert_by_metric']}")
    print(f"\nDRIFT streams ({len(drift)} x {a.n} docs, A -> B at ~{a.change_at}):")
    metrics = sorted({m for r in drift for m in r["detection_lag_docs"]})
    for m in metrics:
        lags = [r["detection_lag_docs"][m] for r in drift if m in r["detection_lag_docs"]]
        missed = len(drift) - len(lags)
        print(f"  {m:20s} detected {len(lags)}/{len(drift)}  lag (docs after change): "
              f"median {statistics.median(lags):.0f}, range {min(lags)}-{max(lags)}" if lags else f"  {m:20s} never detected")
    overall = [r["first_alert_overall"] - r["change_at_index"] for r in drift if r["first_alert_overall"]]
    print(f"\n  first alert of ANY signal: detected {len(overall)}/{len(drift)}, median lag {statistics.median(overall):.0f} docs" if overall else "\n  no detection")
    print(f"  alerts BEFORE the change (should be 0): {sum(r['false_alarms_before_change'] for r in drift)}")
    print(f"  all jobs completed: {all(r['workers']['dead'] == 0 and r['workers']['done'] == a.n for _, r in results)}")


if __name__ == "__main__":
    main()
