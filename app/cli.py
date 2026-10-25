"""CLI: `python -m app.cli worker | ingest <folder> | simulate`"""
import argparse
import json
import logging
import time
from pathlib import Path

from app import db, drift, jobqueue, store, synth, worker


def _ingest_folder(folder: Path, namespace: str, batch: str) -> dict:
    counts = {"queued": 0, "duplicate": 0, "rejected": 0}
    for p in sorted(folder.iterdir()):
        if not p.is_file():
            continue
        try:
            doc_id, created = store.add_document(namespace, p.name, p.read_bytes(), batch)
        except Exception as e:
            counts["rejected"] += 1
            print(f"rejected {p.name}: {e}")
            continue
        if created:
            jobqueue.enqueue(doc_id, namespace)
            counts["queued"] += 1
        else:
            counts["duplicate"] += 1
    return counts


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ap = argparse.ArgumentParser(prog="app.cli")
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("worker", help="run extraction workers")
    w.add_argument("--concurrency", type=int, default=4)
    w.add_argument("--drain", action="store_true", help="exit when the queue is empty")
    i = sub.add_parser("ingest", help="queue every file in a folder (idempotent; --watch keeps polling for new files)")
    i.add_argument("folder")
    i.add_argument("--namespace", default="default")
    i.add_argument("--watch", action="store_true")
    s = sub.add_parser("simulate", help="synthetic stream with a mid-run format change; prints detection lag")
    s.add_argument("--namespace", default="sim")
    s.add_argument("--n", type=int, default=300)
    s.add_argument("--change-at", type=int, default=150)
    s.add_argument("--no-change", action="store_true", help="stable stream (no format change): measures false alarms")
    s.add_argument("--seed0", type=int, default=None)
    s.add_argument("--json", action="store_true", help="print one JSON line (for scripts)")
    s.add_argument("--concurrency", type=int, default=4)
    a = ap.parse_args()
    db.init()

    if a.cmd == "worker":
        pool = worker.WorkerPool(concurrency=a.concurrency, drain=a.drain).start()
        try:
            pool.join()
        except KeyboardInterrupt:
            pool.stop()
        print(json.dumps(pool.counts))
    elif a.cmd == "ingest":
        folder = Path(a.folder)
        while True:
            print(json.dumps(_ingest_folder(folder, a.namespace, f"folder-{folder.name}")))
            if not a.watch:
                break
            time.sleep(5)
    elif a.cmd == "simulate":
        seed0 = a.seed0 if a.seed0 is not None else int(time.time()) % 100000
        change_at = None if a.no_change else a.change_at
        for idx, tpl, png, truth in synth.generate_stream(a.n, change_at, seed0):
            doc_id, created = store.add_document(a.namespace, f"sim-{seed0}-{idx}.png", png, f"sim-{seed0}",
                                                 {"sim_truth": truth, "sim_template": tpl, "sim_index": idx})
            if created:
                jobqueue.enqueue(doc_id, a.namespace)
        counts = worker.run_until_drained(a.concurrency)
        report = drift.simulation_report(a.namespace)
        if a.json:
            print(json.dumps({"workers": counts, "seed0": seed0, "change_at": change_at, **report}))
        else:
            print("workers:", json.dumps(counts))
            print("simulation:", json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
