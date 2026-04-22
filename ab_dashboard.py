"""
Connie A/B Test Dashboard
Reads live from Langfuse — compares prod-a vs prod-b eval runs
and shows live traces with automatic scoring.
"""

import os
import json
import requests
import streamlit as st
import pandas as pd
from datetime import datetime, timezone, timedelta
from requests.auth import HTTPBasicAuth
from collections import defaultdict

# ── Config ─────────────────────────────────────────────────────────────
LANGFUSE_HOST = "https://langfuse.forgehg.cloud"
DATASET_ID    = "cmn60zfsi00inzb078j4a21ak"
DATASET_NAME  = "sykes-connie-functional-evals-v2"
PROD_A_RUN    = "eval-prod-a-20260421"
PROD_B_RUN    = "eval-prod-b-20260421"

THRESHOLDS = {
    "connie-task-completion-ds": 80,
    "connie-task-adherence":     80,
    "cs-handoff-ds":             80,
    "brand-tone-ds":             70,
    "brand_tone-ds":             70,
    "length-ok-ds":              85,
    "connie-hallucination-ds":   80,
    "task-completion":           80,
}

DISPLAY_NAMES = {
    "connie-task-completion-ds": "Task Completion",
    "connie-task-adherence":     "Task Adherence",
    "cs-handoff-ds":             "CS Handoff",
    "brand-tone-ds":             "Brand Tone",
    "brand_tone-ds":             "Brand Tone",
    "length-ok-ds":              "Length OK",
    "connie-hallucination-ds":   "Hallucination Resistance",
    "task-completion":           "Task Completion (legacy)",
}

# Only show dataset-run evaluators on A/B dashboard
DATASET_EVALUATORS = {
    "connie-task-completion-ds", "connie-task-adherence",
    "cs-handoff-ds", "brand-tone-ds", "brand_tone-ds",
    "length-ok-ds", "connie-hallucination-ds", "task-completion",
}

IMPROVEMENT_THRESHOLD = 5.0  # pp

# ── Auth ────────────────────────────────────────────────────────────────
def get_auth():
    try:
        pk = st.secrets["LANGFUSE_PUBLIC_KEY"]
        sk = st.secrets["LANGFUSE_SECRET_KEY"]
    except Exception:
        pk = os.getenv("LANGFUSE_PUBLIC_KEY", "")
        sk = os.getenv("LANGFUSE_SECRET_KEY", "")
    return HTTPBasicAuth(pk, sk)

# ── Data fetching ───────────────────────────────────────────────────────
@st.cache_data(ttl=120)
def fetch_run_scores(run_name):
    auth = get_auth()
    items = []
    page = 1
    while True:
        resp = requests.get(
            f"{LANGFUSE_HOST}/api/public/datasets/{DATASET_NAME}/runs/{run_name}/items",
            auth=auth,
            params={"page": page, "limit": 50},
            verify=False
        )
        if resp.status_code != 200:
            # fallback: try query param style
            resp = requests.get(
                f"{LANGFUSE_HOST}/api/public/dataset-run-items",
                auth=auth,
                params={"runName": run_name, "datasetId": DATASET_ID, "page": page, "limit": 50},
                verify=False
            )
        if resp.status_code != 200:
            break
        data = resp.json().get("data", [])
        items.extend(data)
        if len(data) < 50:
            break
        page += 1

    scores_by_name = defaultdict(list)
    trace_details  = []

    for item in items:
        trace_id = item.get("traceId")
        if not trace_id:
            continue
        try:
            score_resp = requests.get(
                f"{LANGFUSE_HOST}/api/public/scores",
                auth=auth,
                params={"traceId": trace_id},
                verify=False
            )
            if score_resp.status_code == 200:
                trace_scores = score_resp.json().get("data", [])
                item_scores  = {}
                for s in trace_scores:
                    name = s.get("name", "")
                    val  = s.get("value")
                    if val is not None:
                        scores_by_name[name].append(val)
                        item_scores[name] = val

                trace_resp = requests.get(
                    f"{LANGFUSE_HOST}/api/public/traces/{trace_id}",
                    auth=auth, verify=False
                )
                if trace_resp.status_code == 200:
                    t   = trace_resp.json()
                    inp = t.get("input") or {}
                    out = t.get("output") or {}
                    if isinstance(inp, str):
                        try: inp = json.loads(inp)
                        except: inp = {}
                    if isinstance(out, str):
                        try: out = json.loads(out)
                        except: out = {}
                    trace_details.append({
                        "trace_id":    trace_id,
                        "user_message": inp.get("user_message", inp.get("query", "")),
                        "response":    out.get("response", out.get("text", "")),
                        "scores":      item_scores,
                        "timestamp":   t.get("timestamp", "")
                    })
        except Exception:
            continue

    return scores_by_name, trace_details, len(items)


@st.cache_data(ttl=60)
def fetch_live_traces(hours=24):
    auth  = get_auth()
    since = (datetime.now(timezone.utc) - timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")
    traces = []
    page   = 1
    while len(traces) < 60:
        resp = requests.get(
            f"{LANGFUSE_HOST}/api/public/traces",
            auth=auth,
            params={"fromTimestamp": since, "page": page, "limit": 20},
            verify=False
        )
        if resp.status_code != 200:
            break
        data = resp.json().get("data", [])
        if not data:
            break
        for t in data:
            scores = {}
            for s in (t.get("scores") or []):
                if s.get("value") is not None:
                    scores[s["name"]] = s["value"]
            inp = t.get("input") or {}
            out = t.get("output") or {}
            if isinstance(inp, str):
                try: inp = json.loads(inp)
                except: inp = {}
            if isinstance(out, str):
                try: out = json.loads(out)
                except: out = {}
            traces.append({
                "trace_id":     t.get("id", ""),
                "timestamp":    t.get("timestamp", ""),
                "user_message": inp.get("user_message", str(inp)[:80]),
                "response":     out.get("response", str(out)[:200]),
                "scores":       scores,
                "prompt_label": (t.get("metadata") or {}).get("prompt_label", "—")
            })
        page += 1
        if len(data) < 20:
            break
    return traces


def summarise_scores(scores_by_name):
    seen   = set()
    result = {}
    for raw_name, values in scores_by_name.items():
        # only show dataset-run evaluators
        if raw_name not in DATASET_EVALUATORS:
            continue
        display = DISPLAY_NAMES.get(raw_name, raw_name)
        if display in seen or not values:
            continue
        seen.add(display)
        passing = sum(1 for v in values if v > 0.5)
        result[display] = {
            "pct":      round(passing / len(values) * 100, 1),
            "n":        len(values),
            "raw_name": raw_name
        }
    return result


def score_colour(pct, threshold):
    if pct >= threshold:        return "#27AE60"
    elif pct >= threshold * 0.85: return "#F39C12"
    else:                       return "#E74C3C"


def score_card(label, pct, threshold, n, delta=None):
    colour = score_colour(pct, threshold)
    delta_html = ""
    if delta is not None:
        arrow   = "▲" if delta > 0 else ("▼" if delta < 0 else "→")
        d_color = "#27AE60" if delta > 0 else ("#E74C3C" if delta < 0 else "#888")
        delta_html = f'<span style="color:{d_color};font-size:0.82rem;margin-left:8px">{arrow} {abs(delta):.1f}pp</span>'
    return f"""
<div style="background:#13151f;border-radius:10px;padding:16px 18px;margin-bottom:10px;
            border-left:4px solid {colour};">
  <div style="color:#666;font-size:0.7rem;text-transform:uppercase;letter-spacing:1.2px;margin-bottom:4px">{label}</div>
  <div style="display:flex;align-items:baseline">
    <span style="color:{colour};font-size:1.9rem;font-weight:700;font-variant-numeric:tabular-nums">{pct}%</span>
    {delta_html}
  </div>
  <div style="color:#444;font-size:0.68rem;margin-top:3px">threshold ≥{threshold}% &nbsp;·&nbsp; n={n}</div>
</div>"""


# ── Page setup ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Connie A/B Dashboard",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family:'Inter',sans-serif; background:#0d0f1a; color:#e0e0e0; }
.block-container { padding:2rem 2.5rem 4rem; max-width:1440px; }
h1 { font-size:1.5rem!important; font-weight:700!important; color:#fff!important; }
.section-hdr { font-size:0.68rem; text-transform:uppercase; letter-spacing:2px; color:#444;
               margin:20px 0 12px; padding-bottom:6px; border-bottom:1px solid #1c1f2e; }
.run-pill { display:inline-block; padding:3px 10px; border-radius:20px;
            font-size:0.72rem; font-weight:600; font-family:'JetBrains Mono',monospace; }
.pill-a { background:#0d1a2e; color:#5b9fff; border:1px solid #1e3a6a; }
.pill-b { background:#1a0d2e; color:#b05bff; border:1px solid #3a1e6a; }
.trace-card { background:#13151f; border-radius:8px; padding:14px 16px;
              margin-bottom:8px; border:1px solid #1c1f2e; font-size:0.83rem; }
.score-pill { display:inline-block; padding:2px 8px; border-radius:10px;
              font-size:0.68rem; font-weight:600; margin-right:4px;
              font-family:'JetBrains Mono',monospace; }
.s-pass { background:#0d2a1a; color:#27AE60; }
.s-fail { background:#2a0d0d; color:#E74C3C; }
</style>
""", unsafe_allow_html=True)

# ── Header ──────────────────────────────────────────────────────────────
hc1, hc2 = st.columns([5, 1])
with hc1:
    st.markdown("""
    <h1>🏡 Connie — A/B Test Dashboard</h1>
    <p style="color:#555;font-size:0.82rem;margin-top:-6px">
        Live · langfuse.forgehg.cloud · sykes-connie-functional-evals-v2 · 173 cases
    </p>""", unsafe_allow_html=True)
with hc2:
    st.markdown("<div style='margin-top:18px'>", unsafe_allow_html=True)
    if st.button("↻ Refresh data"):
        st.cache_data.clear()
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📊  A/B Comparison", "📡  Live Traces", "ℹ️  About"])

# ══════════════════════════════════════════════════════════════════════
# TAB 1 — A/B COMPARISON
# ══════════════════════════════════════════════════════════════════════
with tab1:
    with st.spinner("Loading prod-a…"):
        a_raw, a_traces, a_n = fetch_run_scores(PROD_A_RUN)
    with st.spinner("Loading prod-b…"):
        b_raw, b_traces, b_n = fetch_run_scores(PROD_B_RUN)

    a_sum = summarise_scores(a_raw)
    b_sum = summarise_scores(b_raw)

    if not a_sum and not b_sum:
        st.warning("No scores found yet. Langfuse evaluators may still be scoring — wait a few minutes and refresh.")
    else:
        # Verdict
        wins_b, wins_a, ties = 0, 0, 0
        all_metrics = sorted(set(list(a_sum) + list(b_sum)))
        for m in all_metrics:
            ap = a_sum.get(m, {}).get("pct")
            bp = b_sum.get(m, {}).get("pct")
            if ap is None or bp is None: continue
            d = bp - ap
            if d >= IMPROVEMENT_THRESHOLD:    wins_b += 1
            elif d <= -IMPROVEMENT_THRESHOLD: wins_a += 1
            else:                             ties   += 1

        if wins_b > wins_a:
            vc, vi = "border:1px solid #27AE60;background:linear-gradient(135deg,#0d2a1a,#0f2e1c)", "✅"
            vt = "Recommendation: Promote prod-b to production"
            vb = f"prod-b outperforms prod-a on <strong>{wins_b}</strong> metric(s) by ≥5pp. Move the <strong>production</strong> label to prod-b in Langfuse Prompt Management."
        elif wins_a > wins_b:
            vc, vi = "border:1px solid #E74C3C;background:linear-gradient(135deg,#2a0d0d,#2e0f0f)", "❌"
            vt = "Recommendation: Keep prod-a — prod-b does not improve"
            vb = f"prod-a outperforms prod-b on <strong>{wins_a}</strong> metric(s). Investigate prod-b before promoting."
        else:
            vc, vi = "border:1px solid #F39C12;background:linear-gradient(135deg,#2a1f0d,#2e220f)", "⏳"
            vt = "No clear winner yet"
            vb = "Scores are within the ±5pp significance threshold. Evaluators may still be running — refresh in a few minutes."

        st.markdown(f"""
        <div style="border-radius:12px;padding:20px 24px;margin-bottom:24px;{vc}">
            <div style="font-size:1.05rem;font-weight:700;color:#fff;margin-bottom:6px">{vi}&nbsp; {vt}</div>
            <div style="color:#bbb;font-size:0.88rem">{vb}</div>
            <div style="margin-top:10px;color:#555;font-size:0.72rem">
                prod-b wins: {wins_b} &nbsp;·&nbsp; prod-a wins: {wins_a} &nbsp;·&nbsp; ties: {ties} &nbsp;·&nbsp; significance: ±{IMPROVEMENT_THRESHOLD}pp
            </div>
        </div>""", unsafe_allow_html=True)

        # Run info cards
        st.markdown('<div class="section-hdr">Run details</div>', unsafe_allow_html=True)
        r1, r2 = st.columns(2)
        with r1:
            st.markdown(f"""
            <div style="background:#13151f;border-radius:8px;padding:14px 16px;border-left:4px solid #5b9fff">
                <span class="run-pill pill-a">prod-a</span>
                <div style="margin-top:8px;color:#aaa;font-size:0.78rem;font-family:'JetBrains Mono',monospace">{PROD_A_RUN}</div>
                <div style="color:#555;font-size:0.72rem;margin-top:4px">{a_n} cases &nbsp;·&nbsp; current production &nbsp;·&nbsp; baseline prompt</div>
            </div>""", unsafe_allow_html=True)
        with r2:
            st.markdown(f"""
            <div style="background:#13151f;border-radius:8px;padding:14px 16px;border-left:4px solid #b05bff">
                <span class="run-pill pill-b">prod-b</span>
                <div style="margin-top:8px;color:#aaa;font-size:0.78rem;font-family:'JetBrains Mono',monospace">{PROD_B_RUN}</div>
                <div style="color:#555;font-size:0.72rem;margin-top:4px">{b_n} cases &nbsp;·&nbsp; challenger &nbsp;·&nbsp; CS handoff URL fix</div>
            </div>""", unsafe_allow_html=True)

        # Score cards side by side
        st.markdown('<div class="section-hdr">Evaluator scores</div>', unsafe_allow_html=True)
        ca, cb = st.columns(2)

        with ca:
            st.markdown('<div style="color:#5b9fff;font-size:0.78rem;font-weight:600;margin-bottom:10px">prod-a — current production</div>', unsafe_allow_html=True)
            for m in all_metrics:
                ai = a_sum.get(m)
                bi = b_sum.get(m)
                raw = (ai or bi or {}).get("raw_name", m)
                thr = THRESHOLDS.get(raw, THRESHOLDS.get(m, 80))
                if ai:
                    delta = -(bi["pct"] - ai["pct"]) if bi else None
                    st.markdown(score_card(m, ai["pct"], thr, ai["n"], delta), unsafe_allow_html=True)
                else:
                    st.markdown(f'<div style="color:#333;font-size:0.78rem;padding:8px 0">No data — {m}</div>', unsafe_allow_html=True)

        with cb:
            st.markdown('<div style="color:#b05bff;font-size:0.78rem;font-weight:600;margin-bottom:10px">prod-b — challenger</div>', unsafe_allow_html=True)
            for m in all_metrics:
                ai = a_sum.get(m)
                bi = b_sum.get(m)
                raw = (bi or ai or {}).get("raw_name", m)
                thr = THRESHOLDS.get(raw, THRESHOLDS.get(m, 80))
                if bi:
                    delta = (bi["pct"] - ai["pct"]) if ai else None
                    st.markdown(score_card(m, bi["pct"], thr, bi["n"], delta), unsafe_allow_html=True)
                else:
                    st.markdown(f'<div style="color:#333;font-size:0.78rem;padding:8px 0">No data — {m}</div>', unsafe_allow_html=True)

        # Drilldown
        if b_traces:
            st.markdown('<div class="section-hdr">Case drilldown — prod-b</div>', unsafe_allow_html=True)
            show_fails = st.checkbox("Show failing cases only", value=True)
            shown = 0
            for t in b_traces:
                tc  = t["scores"].get("connie-task-completion", t["scores"].get("task-completion"))
                passing = tc is not None and tc >= 0.5
                if show_fails and passing:
                    continue
                border = "#27AE60" if passing else "#E74C3C"
                pills  = "".join(
                    f'<span class="score-pill {"s-pass" if v>=0.5 else "s-fail"}">{DISPLAY_NAMES.get(k,k)}: {v:.2f}</span>'
                    for k, v in t["scores"].items()
                )
                q = (t.get("user_message") or "")[:130]
                r = (t.get("response")     or "")[:220]
                st.markdown(f"""
                <div class="trace-card" style="border-left:3px solid {border}">
                    <div style="color:#666;font-size:0.72rem;margin-bottom:6px">💬 {q}</div>
                    <div style="color:#ccc;font-style:italic;margin-bottom:8px;font-size:0.82rem">"{r}{"…" if len(t.get("response",""))>220 else ""}"</div>
                    <div>{pills or '<span style="color:#333;font-size:0.7rem">No scores yet</span>'}</div>
                </div>""", unsafe_allow_html=True)
                shown += 1
                if shown >= 20:
                    break
            if shown == 0:
                st.success("No failing cases found in prod-b — all cases passed.")

# ══════════════════════════════════════════════════════════════════════
# TAB 2 — LIVE TRACES
# ══════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-hdr">Live Connie traces</div>', unsafe_allow_html=True)
    hours = st.slider("Hours back", 1, 72, 24)

    with st.spinner("Fetching live traces…"):
        live = fetch_live_traces(hours=hours)

    if not live:
        st.info("No traces found in this time window.")
    else:
        st.markdown(f'<div style="color:#444;font-size:0.78rem;margin-bottom:14px">{len(live)} traces</div>', unsafe_allow_html=True)
        for t in live[:30]:
            tc      = t["scores"].get("connie-task-completion", t["scores"].get("task-completion"))
            passing = tc is not None and tc >= 0.5
            border  = "#27AE60" if passing else ("#E74C3C" if tc is not None else "#1c1f2e")
            pills   = "".join(
                f'<span class="score-pill {"s-pass" if v>=0.5 else "s-fail"}">{DISPLAY_NAMES.get(k,k)}: {v:.2f}</span>'
                for k, v in t["scores"].items()
            )
            ts    = t.get("timestamp", "")[:16].replace("T", " ")
            label = t.get("prompt_label", "—")
            q     = (t.get("user_message") or "")[:120]
            r     = (t.get("response")     or "")[:200]
            st.markdown(f"""
            <div class="trace-card" style="border-left:3px solid {border}">
                <div style="display:flex;justify-content:space-between;margin-bottom:5px">
                    <span style="color:#333;font-size:0.68rem;font-family:'JetBrains Mono',monospace">{ts}</span>
                    <span style="color:#444;font-size:0.68rem">label: {label}</span>
                </div>
                <div style="color:#666;font-size:0.78rem;margin-bottom:5px">💬 {q}</div>
                <div style="color:#bbb;font-style:italic;margin-bottom:7px;font-size:0.8rem">"{r}{"…" if len(t.get("response",""))>200 else ""}"</div>
                <div>{pills or '<span style="color:#333;font-size:0.68rem">Evaluators scoring…</span>'}</div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# TAB 3 — ABOUT
# ══════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("""
    ### What this dashboard shows
    Live A/B test results comparing two Connie prompt versions, read directly from Langfuse.
    All scores are generated automatically by Langfuse evaluators — no manual scoring.

    **prod-a** — current production prompt, no changes (baseline)  
    **prod-b** — challenger: CS handoff message updated to include `sykes.co.uk/contact` explicitly

    The pipeline calls Connie via Mastra with `promptLabel: prod-a` or `promptLabel: prod-b`,
    logs the response as a Langfuse trace, and the evaluators score it automatically within minutes.

    ### How to promote a winner
    If prod-b wins → **Langfuse → Prompt Management → connie-agent** → move the `production`
    label from prod-a's version to prod-b's version.

    ### Dataset
    `sykes-connie-functional-evals-v2` — 173 hand-crafted cases:
    search (93) · property questions (15) · CS handoff (20) · objection handling (15) · adversarial/safety (15) · end-to-end (15)

    ### Thresholds
    | Evaluator | Threshold | Priority |
    |---|---|---|
    | Task Completion | ≥ 80% | Critical |
    | Task Adherence | ≥ 80% | Critical |
    | CS Handoff | ≥ 80% | Critical |
    | Safety / Adversarial | 100% | Hard floor |
    | Hallucination Resistance | ≥ 80% | Critical |
    | Brand Tone | ≥ 70% | Important |
    | Length OK | ≥ 85% | Informational |

    ---
    *Data: `langfuse.forgehg.cloud` · refreshes every 2 min · built by Calbert Graham, Data Science*
    """)
