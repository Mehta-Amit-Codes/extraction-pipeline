"""
Active learning: human corrections become few-shot examples for the next documents.

Selection is by VISUAL layout similarity (Hamming distance between layout fingerprints): show the model corrected
examples of documents that LOOK like the one it is about to read. Examples further than FEWSHOT_MAX_DIST are never
shown (a corrected template-A invoice would only mislead on a template-B document), so an unfamiliar layout gets
zero-shot until a human corrects one.

Which reviews join the pool:
  * an EDIT (the model was wrong, a human fixed it)   -> always: the most informative signal there is
  * an APPROVE of a layout the pool does not cover    -> yes: gives that layout coverage
  * an APPROVE of an already-covered layout           -> no: it adds cost (images are tokens) and no information
  * a REJECT                                          -> no
The pool is PER NAMESPACE: one tenant's documents are never shown to the model while processing another's.
"""
import json

from app import imaging, store
from app.config import FEWSHOT_K, FEWSHOT_MAX_DIST
from app.extractors import Example, example_answer


def select_examples(namespace: str, layout_hash: str, k: int = FEWSHOT_K, exclude_document_id: int | None = None) -> list[Example]:
    """The k nearest corrected examples by layout, within FEWSHOT_MAX_DIST (ties: newest first)."""
    pool = [e for e in store.list_examples(namespace) if e["document_id"] != exclude_document_id]
    near = sorted(((imaging.hamming(e["layout_hash"], layout_hash), -e["id"], e) for e in pool), key=lambda t: t[:2])
    chosen = [e for d, _, e in near if d <= FEWSHOT_MAX_DIST][:k]
    out = []
    for e in chosen:
        doc = store.get_document(e["document_id"], namespace)
        if doc is None:
            continue
        try:
            img = imaging.load_images(store.read_bytes(doc))[0]
        except Exception:
            continue  # a missing/corrupt example file must not fail the document being processed
        out.append(Example(document_id=e["document_id"], image=img, record=e["record"], layout_hash=e["layout_hash"]))
    return out


def covered(namespace: str, layout_hash: str, exclude_document_id: int | None = None) -> bool:
    return any(imaging.hamming(e["layout_hash"], layout_hash) <= FEWSHOT_MAX_DIST
               for e in store.list_examples(namespace) if e["document_id"] != exclude_document_id)


def maybe_add_example(namespace: str, document: dict, final_record: dict, action: str) -> int | None:
    if action == "edit":
        return store.upsert_example(namespace, document["id"], final_record, "correction", document["layout_hash"])
    if action == "approve" and not covered(namespace, document["layout_hash"], document["id"]):
        return store.upsert_example(namespace, document["id"], final_record, "approved_new_layout", document["layout_hash"])
    return None


def pool_summary(namespace: str) -> dict:
    ex = store.list_examples(namespace)
    clusters: list[str] = []
    for e in ex:  # greedy clustering: how many distinct layouts does the pool cover?
        if not any(imaging.hamming(e["layout_hash"], c) <= FEWSHOT_MAX_DIST for c in clusters):
            clusters.append(e["layout_hash"])
    by_source: dict[str, int] = {}
    for e in ex:
        by_source[e["source"]] = by_source.get(e["source"], 0) + 1
    return {"examples": len(ex), "layouts_covered": len(clusters), "by_source": by_source}


def export_finetune_manifest(namespace: str) -> list[dict]:
    """Fine-tuning set from human-verified records: one chat-format row per example, with the image referenced by its
    content-addressed path. Convert to your provider's vision fine-tuning format; kept provider-neutral on purpose."""
    from app.extractors import PROMPT_PASS1
    rows = []
    for e in store.list_examples(namespace):
        doc = store.get_document(e["document_id"], namespace)
        if doc:
            rows.append({"image_path": doc["path"], "sha256": doc["sha256"], "layout_hash": e["layout_hash"],
                         "messages": [{"role": "system", "content": PROMPT_PASS1},
                                      {"role": "user", "content": "Extract this invoice. [image]"},
                                      {"role": "assistant", "content": example_answer(e["record"], confidence=1.0)}]})
    return rows
