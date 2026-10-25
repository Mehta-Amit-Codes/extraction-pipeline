"""
Streamlit UI for the Structured Extraction Pipeline.

Tabs
  1. Pipeline        upload / generate documents, watch the queue, pause and resume
  2. Review          approve / edit / reject flagged extractions (most uncertain first)
  3. Quality & Drift KPIs, alerts, drift charts, calibration, and the format-change SIMULATION (stretch goal)
  4. Learning        the few-shot pool built from corrections, and a before/after on the same documents
  5. Dead letters    failed documents, requeue, export

Run:  streamlit run streamlit_app.py      (the API must be running; see README)
"""
import os
import time

import altair as alt
import pandas as pd
import requests
import streamlit as st

API_URL = os.environ.get("EXTRACTION_API_URL", "http://localhost:8006")
SCALAR = ["vendor_name", "invoice_number", "invoice_date", "due_date", "currency", "subtotal", "tax", "total"]
ITEM_COLS = ["description", "quantity", "unit_price", "amount"]

st.set_page_config(page_title="Extraction Pipeline", page_icon="🧾", layout="wide")
st.title("Structured Extraction Pipeline")
st.caption(f"Batch extraction · confidence scoring · human review · active learning · drift detection · API: {API_URL}")


def api(method: str, path: str, quiet: bool = False, raw: bool = False, **kw):
    headers = {"X-API-Key": st.session_state.get("api_key", "")} if st.session_state.get("api_key") else {}
    try:
        r = requests.request(method, f"{API_URL}{path}", headers=headers, timeout=120, **kw)
    except requests.RequestException as e:
        if not quiet:
            st.error(f"Cannot reach the API at {API_URL}: {e}")
        return None
    if r.status_code >= 400:
        if not quiet:
            try:
                detail = r.json().get("detail", r.text)
            except ValueError:
                detail = r.text
            st.error(f"{path}: {r.status_code} {detail}")
        return None
    return r.content if raw else (r.json() if r.content and "json" in r.headers.get("content-type", "") else r.text)


def show_df(df, container=st):
    """Streamlit renamed use_container_width -> width='stretch'; support both."""
    try:
        container.dataframe(df, width="stretch")
    except Exception:
        container.dataframe(df, use_container_width=True)


def show_chart(chart):
    try:
        st.altair_chart(chart, width="stretch")
    except Exception:
        st.altair_chart(chart, use_container_width=True)


def conf_badge(c: float) -> str:
    return "🟢" if c >= 0.85 else "🟡" if c >= 0.6 else "🔴"


# ------------------------------------------------------------------- sidebar ----
with st.sidebar:
    st.subheader("Workspace")
    st.session_state.setdefault("namespace", "default")
    st.session_state["namespace"] = st.text_input("Namespace", value=st.session_state["namespace"],
                                                  help="Isolates documents, reviews, the example pool and drift baselines.")
    st.session_state["api_key"] = st.text_input("API key (only if the server requires one)", type="password",
                                                value=st.session_state.get("api_key", ""))
    NS = st.session_state["namespace"]
    health = api("GET", "/health", quiet=True)
    st.success(f"API up · {health['workers']} embedded worker(s)") if health else st.error("API unreachable")

tab_pipe, tab_review, tab_quality, tab_learn, tab_dead = st.tabs(
    ["Pipeline", "Review", "Quality & Drift", "Learning", "Dead letters & Export"])

# ------------------------------------------------------------------- pipeline ----
with tab_pipe:
    st.subheader("Queue")
    qs = api("GET", "/queue/stats", params={"namespace": NS}, quiet=True)
    if qs:
        if qs["paused"]:
            st.error(f"Queue paused: {qs['pause_reason'] or 'manual'}")
        cols = st.columns(7)
        for col, (label, key) in zip(cols, [("Queued", "queued"), ("Running", "running"), ("Done", "done"), ("Retrying", "retrying"),
                                            ("Dead-lettered", "dead"), ("Done / min", "done_last_minute"), ("Oldest wait (s)", "oldest_queued_age_s")]):
            col.metric(label, round(qs[key]) if key == "oldest_queued_age_s" else qs[key])
        b1, b2, b3, _ = st.columns([1, 1, 1, 5])
        if b1.button("Pause"):
            api("POST", "/queue/pause")
            st.rerun()
        if b2.button("Resume"):
            api("POST", "/queue/resume")
            st.rerun()
        b3.button("Refresh")

    st.divider()
    left, right = st.columns(2)
    with left:
        st.subheader("Upload documents")
        st.caption("PNG, JPEG or PDF (first pages). Identical files are de-duplicated; unsupported files are rejected up front.")
        files = st.file_uploader("Invoices", type=["png", "jpg", "jpeg", "pdf"], accept_multiple_files=True)
        if st.button("Queue uploads", type="primary") and files:
            res = api("POST", "/documents", data={"namespace": NS}, files=[("files", (f.name, f.getvalue(), f.type)) for f in files])
            if res:
                st.success(f"{res['queued']} queued")
                bad = [r for r in res["results"] if r["status"] != "queued"]
                if bad:
                    show_df(pd.DataFrame(bad))
    with right:
        st.subheader("Generate synthetic invoices")
        st.caption("Real PNGs with known ground truth, in two layouts. With the default simulated extractor this runs fully offline.")
        n = st.slider("Documents", 10, 600, 120, step=10)
        change = st.checkbox("Switch layout mid-stream (format change)", value=False)
        change_at = st.slider("…at document #", 5, max(10, n - 5), n // 2) if change else None
        if change_at is not None and change_at < ((api("GET", "/config", quiet=True) or {}).get("baseline_n", 60)):
            st.caption("⚠️ This change falls inside the drift baseline window; the Quality tab will warn that the baseline is unstable.")
        c1, c2 = st.columns(2)
        fail_every = c1.number_input("Transient failure every Nth doc (0 = none)", 0, 50, 0)
        poison_every = c2.number_input("Unprocessable doc every Nth (0 = none)", 0, 50, 0)
        if st.button("Generate and queue"):
            with st.spinner("Rendering invoices…"):
                res = api("POST", "/synthetic", json={"namespace": NS, "n": n, "change_at": change_at,
                                                      "fail_every": int(fail_every), "poison_every": int(poison_every)})
            if res:
                st.success(f"Queued {res['queued']} documents (batch {res['batch_id']})")

    st.subheader("Recent jobs")
    jobs = api("GET", "/jobs", params={"namespace": NS, "limit": 30}, quiet=True) or []
    if jobs:
        df = pd.DataFrame(jobs)[["id", "filename", "status", "attempts", "last_error_kind", "last_error"]]
        show_df(df)

# --------------------------------------------------------------------- review ----
with tab_review:
    pending = api("GET", "/reviews/pending", params={"namespace": NS, "limit": 50}, quiet=True) or []
    st.subheader(f"Review queue: {len(pending)} pending")
    st.caption("Most uncertain first (lowest confidence, disagreements, validator failures, unfamiliar layouts). "
               "Each correction becomes a labelled example that improves later extractions.")
    if not pending:
        st.info("Nothing to review. Process some documents first.")
    else:
        labels = [f"#{p['id']} · {p['filename']} · priority {p['priority']:.2f}" + (" · AUDIT" if p["audit_sample"] else "") for p in pending]
        idx = st.selectbox("Document", range(len(pending)), format_func=lambda i: labels[i])
        item = pending[idx]
        img_col, form_col = st.columns([1, 1])
        with img_col:
            img = api("GET", f"/documents/{item['document_id']}/image", params={"namespace": NS}, raw=True, quiet=True)
            if img:
                st.image(img, caption=item["filename"])
            if item["fewshot_doc_ids"]:
                st.caption(f"extracted with {len(item['fewshot_doc_ids'])} corrected example(s) of similar layout in the prompt")
        with form_col:
            if item["audit_sample"] and not item["review_reasons"][:-1]:
                st.info("Random audit: the system was confident about this one. Check it carefully; this keeps the accuracy estimate honest.")
            st.markdown("**Why it was flagged:** " + (", ".join(f"`{r}`" for r in item["review_reasons"]) or "n/a"))
            for issue in item["issues"]:
                (st.error if issue["severity"] == "error" else st.warning)(f"{issue['code']}: {issue['message']}")

            with st.form(f"review-{item['id']}"):
                values = {}
                for f in SCALAR:
                    c = item["field_conf"][f]
                    disputed = f in item["disputed_fields"]
                    values[f] = st.text_input(f"{conf_badge(c)} {f} · confidence {c:.2f}" + (" · ⚔️ passes disagree" if disputed else ""),
                                              value=item["record"].get(f) or "", key=f"{item['id']}-{f}")
                    if disputed and item["pass2"]:
                        st.caption(f"pass 1: `{item['pass1'].get(f)}` · pass 2: `{item['pass2'].get(f)}`")
                st.markdown(f"{conf_badge(item['field_conf']['line_items'])} **line items** · confidence {item['field_conf']['line_items']:.2f}"
                            + (" · ⚔️ passes disagree" if "line_items" in item["disputed_fields"] else ""))
                items_df = st.data_editor(pd.DataFrame(item["record"].get("line_items") or [], columns=ITEM_COLS), num_rows="dynamic",
                                          key=f"{item['id']}-items", hide_index=True)
                reviewer = st.text_input("Reviewer", value="reviewer", key="reviewer-name")
                a, b, c = st.columns(3)
                approve = a.form_submit_button("Approve as is", type="primary")
                save = b.form_submit_button("✏️ Save corrections")
                reject = c.form_submit_button("Reject (not an invoice / unreadable)")

            if approve or save or reject:
                record = {**{f: (v.strip() or None) for f, v in values.items()},
                          "line_items": [r for r in items_df.to_dict("records") if any(str(v).strip() not in ("", "nan", "None") for v in r.values())]}
                action = "approve" if approve else "edit" if save else "reject"
                res = api("POST", f"/reviews/{item['id']}", json={"action": action, "namespace": NS, "reviewer": reviewer,
                                                                   "record": record if action == "edit" else None})
                if res:
                    for w in res["warnings"]:
                        st.warning(f"Saved, but still inconsistent: {w}")
                    st.toast(f"{res['status']}" + (" · added to the example pool" if res["added_to_example_pool"] else ""))
                    if not res["warnings"]:
                        st.rerun()

# -------------------------------------------------------------- quality / drift ----
with tab_quality:
    q = api("GET", "/quality", params={"namespace": NS}, quiet=True)
    if q and q["processed"]:
        cols = st.columns(6)
        cols[0].metric("Processed", q["processed"])
        cols[1].metric("Straight-through", f"{q['stp_rate']:.0%}", help="Auto-approved without a human")
        cols[2].metric("Flagged for review", f"{q['review_rate']:.0%}")
        cols[3].metric("Reviews done", q["reviews_done"])
        cols[4].metric("Cost / doc", f"${q['cost']['cost_per_doc_usd']:.4f}", help="Rates are illustrative; set COST_PER_M_* in .env")
        aud = q["audit"]
        cols[5].metric("Auto-approved accuracy", f"{aud['doc_accuracy']:.0%}" if aud["n"] else "n/a",
                       help="Estimated from a random sample of auto-approved documents (unbiased). " + aud["note"])
        if aud["n"]:
            st.caption(f"Audit sample n={aud['n']} · 95% interval {aud['ci95'][0]:.0%} to {aud['ci95'][1]:.0%} "
                       "(a small sample gives a wide interval)")
    else:
        st.info("No extractions yet in this namespace.")

    st.divider()
    st.subheader("Drift alerts")
    alerts = api("GET", "/alerts", params={"namespace": NS}, quiet=True) or []
    open_alerts = [a for a in alerts if a["status"] == "open"]
    if open_alerts:
        for a in open_alerts:
            box = st.error if a["severity"] == "critical" else st.warning
            c1, c2 = st.columns([8, 1])
            box(f"**{a['metric']}** · {a['message']} · fired at document #{a['at_extraction_index']}", icon="🚨")
            if c2.button("Acknowledge", key=f"ack-{a['id']}"):
                api("POST", f"/alerts/{a['id']}/ack", params={"namespace": NS})
                st.rerun()
    else:
        st.success("No open drift alerts.")

    sim = api("GET", "/drift/simulation", params={"namespace": NS}, quiet=True)
    ts = api("GET", "/drift/timeseries", params={"namespace": NS, "step": 5}, quiet=True) or []
    if ts:
        df = pd.DataFrame(ts)
        rates = df.melt("index", ["review_rate", "issue_rate", "dispute_rate", "novel_layout_rate"], "signal", "value")
        marks = pd.DataFrame({"index": [a["at_extraction_index"] for a in alerts], "what": "alert"})
        layers = [alt.Chart(rates).mark_line().encode(x=alt.X("index:Q", title="documents processed"),
                                                      y=alt.Y("value:Q", title="rate in rolling window", scale=alt.Scale(domain=[0, 1])), color="signal:N")]
        if len(marks):
            layers.append(alt.Chart(marks).mark_rule(color="red", strokeDash=[4, 4]).encode(x="index:Q"))
        if sim and sim.get("change_at_index"):
            layers.append(alt.Chart(pd.DataFrame({"index": [sim["change_at_index"]]})).mark_rule(color="black").encode(x="index:Q"))
        st.markdown("**Label-free signals** (red dashed = alert fired" + (", black = the format change" if sim and sim.get("change_at_index") else "") + ")")
        show_chart(alt.layer(*layers))

        conf = df.melt("index", [c for c in ("mean_confidence", "oracle_accuracy") if df[c].notna().any()], "signal", "value").dropna()
        if len(conf):
            st.markdown("**Confidence vs. true accuracy** (the oracle line exists only for synthetic documents with known ground truth)")
            show_chart(alt.Chart(conf).mark_line().encode(x=alt.X("index:Q", title="documents processed"),
                                                          y=alt.Y("value:Q", scale=alt.Scale(domain=[0.4, 1])), color="signal:N"))
    drift = api("GET", "/drift", params={"namespace": NS}, quiet=True)
    if drift and drift["status"] == "ok" and drift.get("baseline_stable") is False:
        st.warning("⚠️ **The baseline window is not stable**: it appears to contain a shift of its own, so detection will under-report "
                   "(a new layout that is already 'normal' in the baseline cannot look novel). Pin a known-good baseline once the "
                   "pipeline is healthy. Details: " + "; ".join(drift["baseline_warnings"]))
    if drift and drift["status"] == "ok":
        rows = [{"metric": r["metric"], "baseline": round(r["baseline"], 3), "current": round(r["current"], 3), "change": round(r["delta"], 3),
                 "p-value": f"{r['p_value']:.1e}", "alert-worthy": "yes" if r["significant"] else "no"} for r in drift["results"]]
        with st.expander("Statistical comparison: recent window vs baseline", expanded=False):
            show_df(pd.DataFrame(rows))
            st.caption(f"Baseline {drift['n_baseline']} docs · current {drift['n_current']} docs · novel-layout distance threshold "
                       f"{drift['novelty_threshold_bits']} bits (derived from the baseline). An alert needs statistical AND practical significance.")
    elif drift:
        st.caption(f"Drift evaluation needs at least {drift['need']} documents in both the baseline and the recent window "
                   f"(have {drift['n_baseline']} / {drift['n_current']}).")
    if st.button("Treat the most recent documents as the new baseline", help="Use after fixing prompts or accepting a new format. Acknowledges open alerts."):
        st.success(api("POST", "/drift/baseline/pin", params={"namespace": NS}))

    st.divider()
    st.subheader("Simulate a format change (stretch goal)")
    st.caption("Generates a fresh synthetic stream in its own namespace: layout A, then a different layout B partway through. "
               "Watch the alerts fire, and see which signal notices first.")
    cfg = api("GET", "/config", quiet=True) or {"baseline_n": 60, "window_n": 60, "min_samples": 30}
    s1, s2, s3 = st.columns(3)
    sim_n = s1.number_input("Documents", 100, 600, 300, step=50)
    sim_change = s2.number_input("Format changes at #", 20, 590, 150, step=10)
    min_change = cfg["baseline_n"] + 10
    sim_problem = None
    if sim_change < min_change:
        sim_problem = (f"The change must come AFTER the baseline window (the first {cfg['baseline_n']} documents), otherwise the baseline "
                       f"already contains the new layout and detection goes blind. Use #{min_change} or later.")
    elif sim_n - sim_change < cfg["min_samples"] + 10:
        sim_problem = f"Leave at least {cfg['min_samples'] + 10} documents after the change so there is something to detect."
    if sim_problem:
        st.error(sim_problem)
    if s3.button("▶ Run simulation", type="primary", disabled=bool(sim_problem)):
        sim_ns = f"sim-{int(time.time())}"
        res = api("POST", "/synthetic", json={"namespace": sim_ns, "n": int(sim_n), "change_at": int(sim_change)})
        if res:
            bar, status = st.progress(0.0), st.empty()
            deadline = time.time() + 20 + sim_n * 1.5
            while time.time() < deadline:
                s = api("GET", "/queue/stats", params={"namespace": sim_ns}, quiet=True)
                if s is None:
                    break
                bar.progress(min(1.0, s["done"] / max(1, res["queued"])))
                status.caption(f"{s['done']}/{res['queued']} processed · {s['dead']} dead-lettered")
                if s["done"] + s["dead"] >= res["queued"] or s["paused"]:
                    break
                time.sleep(1.0)
            st.session_state["namespace"] = sim_ns
            st.success(f"Done. Switched the workspace to namespace '{sim_ns}': scroll up to see the charts and alerts.")
            st.rerun()
    if sim and sim.get("change_at_index"):
        st.markdown(f"**Simulation result** · format changed at document #{sim['change_at_index']} of {sim['total_processed']}")
        lag = pd.DataFrame([{"signal": m, "first alert at doc": sim["first_alert_by_metric"][m], "documents after the change": l}
                            for m, l in sim["detection_lag_docs"].items()]).sort_values("documents after the change")
        show_df(lag)
        st.caption("Lag is quantised to the evaluation interval. The layout signal needs no model output at all, so it can lead the "
                   "proxies that depend on the model's own confidence.")

    st.divider()
    st.subheader("Is the model's confidence trustworthy?")
    if q and q["calibration"]["bins"]:
        cal = pd.DataFrame(q["calibration"]["bins"])
        show_df(cal.rename(columns={"bin": "stated confidence", "n": "fields reviewed", "mean_confidence": "mean stated", "accuracy": "actually correct"}))
        st.caption(f"Expected calibration error {q['calibration']['ece']:.3f} (0 = perfectly calibrated). Biased toward hard cases: most reviewed "
                   "documents were flagged because they were uncertain. Audit-only version: "
                   f"{q['calibration_audit_only']['ece']} over {q['calibration_audit_only']['n_fields']} fields.")
        if q["corrections_by_field"]:
            st.markdown("**Fields humans correct most**")
            st.bar_chart(pd.Series(q["corrections_by_field"]))
    else:
        st.caption("Review some documents to see calibration.")

# ------------------------------------------------------------------- learning ----
with tab_learn:
    ex = api("GET", "/examples", params={"namespace": NS}, quiet=True)
    st.subheader("Few-shot example pool")
    st.caption("Human-verified documents shown to the model as worked examples. Chosen by visual similarity: a corrected layout-A "
               "invoice is never shown on a layout-B document.")
    if ex:
        s = ex["summary"]
        cols = st.columns(3)
        cols[0].metric("Examples", s["examples"])
        cols[1].metric("Layouts covered", s["layouts_covered"])
        cols[2].metric("From corrections", s["by_source"].get("correction", 0))
        if ex["examples"]:
            show_df(pd.DataFrame([{"id": e["id"], "document": e["document_id"], "source": e["source"], "active": bool(e["active"]),
                                   "layout": e["layout_hash"]} for e in ex["examples"]]))
            eid = st.selectbox("Toggle an example", [e["id"] for e in ex["examples"]])
            tcol1, tcol2 = st.columns(2)
            if tcol1.button("Disable"):
                api("POST", f"/examples/{eid}/active", json={"namespace": NS, "active": False})
                st.rerun()
            if tcol2.button("Enable"):
                api("POST", f"/examples/{eid}/active", json={"namespace": NS, "active": True})
                st.rerun()
    st.divider()
    st.subheader("Demo: close the loop at volume")
    st.caption("A stand-in reviewer corrects SYNTHETIC documents against their ground truth, so the loop runs on hundreds of documents "
               "without hand-clicking. It does nothing for real documents.")
    o1, o2 = st.columns(2)
    limit = o1.number_input("Documents to auto-review", 5, 500, 25)
    if o1.button("🤖 Run demo reviewer"):
        st.success(api("POST", "/demo/oracle-review", json={"namespace": NS, "limit": int(limit)}))
    if o2.button("🔁 Reprocess documents that are NOT in the pool"):
        docs = api("GET", "/documents", params={"namespace": NS, "limit": 1000}, quiet=True) or []
        in_pool = {e["document_id"] for e in (ex["examples"] if ex else [])}
        ids = [d["id"] for d in docs if d["id"] not in in_pool]
        st.success(api("POST", "/reprocess", json={"namespace": NS, "document_ids": ids}))
    eff = api("GET", "/quality/learning-effect", params={"namespace": NS}, quiet=True)
    if eff and eff["documents"]:
        st.markdown(f"**Before vs. after, on the same {eff['documents']} documents** ({eff['after_used_examples']} re-extracted with examples in the prompt)")
        show_df(pd.DataFrame({"before": eff["before"], "after": eff["after"]}))
        st.warning("The simulator's response to examples is SCRIPTED, so this validates the plumbing (examples selected and passed to the "
                   "extractor), not real model learning. With EXTRACTOR=grok the same loop measures the real effect.")
    st.divider()
    manifest = api("GET", "/examples/export", params={"namespace": NS}, raw=True, quiet=True)
    if manifest:
        st.download_button("⬇ Fine-tuning manifest (JSONL)", manifest, file_name=f"finetune_{NS}.jsonl")

# ---------------------------------------------------------------- dead letters ----
with tab_dead:
    st.subheader("Dead-lettered documents")
    st.caption("Documents that failed permanently or exhausted their retries. Fix the cause, then requeue with a fresh attempt budget.")
    dead = api("GET", "/jobs", params={"namespace": NS, "status": "dead", "limit": 200}, quiet=True) or []
    if dead:
        show_df(pd.DataFrame(dead)[["id", "filename", "attempts", "last_error_kind", "last_error"]])
        d1, d2 = st.columns(2)
        pick = d1.multiselect("Requeue selected", [j["id"] for j in dead])
        if d1.button("Requeue selected") and pick:
            st.success(api("POST", "/jobs/requeue-dead", json={"namespace": NS, "job_ids": pick}))
        if d2.button("Requeue ALL dead letters"):
            st.success(api("POST", "/jobs/requeue-dead", json={"namespace": NS}))
    else:
        st.success("Dead-letter queue is empty.")
    st.divider()
    st.subheader("Export")
    st.caption("Only records that are safe downstream: auto-approved, human-approved or human-corrected. Pending and rejected are excluded.")
    e1, e2 = st.columns(2)
    jl = api("GET", "/export", params={"namespace": NS, "format": "jsonl"}, raw=True, quiet=True)
    cs = api("GET", "/export", params={"namespace": NS, "format": "csv"}, raw=True, quiet=True)
    if jl:
        e1.download_button("⬇ JSONL", jl, file_name=f"extractions_{NS}.jsonl")
    if cs:
        e2.download_button("⬇ CSV", cs, file_name=f"extractions_{NS}.csv")
