from app import jobqueue, store, synth, worker


def add_synth(namespace, i, template="A", seed0=0, extra_meta=None, enqueue=True, noise=0.0):
    png, truth = synth.generate_document(seed0 + i, template, noise)
    meta = {"sim_truth": truth, "sim_template": template, "sim_index": i, **(extra_meta or {})}
    doc_id, created = store.add_document(namespace, f"{template}-{seed0 + i}.png", png, "batch", meta)
    if enqueue and created:
        jobqueue.enqueue(doc_id, namespace)
    return doc_id


def add_stream(namespace, n, change_at=None, seed0=0):
    ids = []
    for i in range(n):
        ids.append(add_synth(namespace, i, "B" if change_at is not None and i >= change_at else "A", seed0))
    return ids


def drain(concurrency=4, **kw):
    return worker.run_until_drained(concurrency, **kw)
