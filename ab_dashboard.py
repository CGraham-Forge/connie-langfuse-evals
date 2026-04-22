"""
Connie A/B Test Dashboard
Uses Langfuse Python SDK to fetch dataset run scores.
"""

import os
import streamlit as st
from datetime import datetime
from collections import defaultdict

# ── Page config ─────────────────────────────────────────────────────────
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
.score-card { background:#13151f; border-radius:10px; padding:16px 18px; margin-bottom:10px; }
.score-pill { display:inline-block; padding:2px 8px; border-radius:10px;
              font-size:0.68rem; font-weight:600; margin-right:4px;
              font-family:'JetBrains Mono',monospace; }
.s-pass { background:#0d2a1a; color:#27AE60; }
.s-fail { background:#2a0d0d; color:#E74C3C; }
.trace-card { background:#13151f; border-radius:8px; padding:14px 16px;
              margin-bottom:8px; border:1px solid #1c1f2e; font-size:0.83rem; }
</style>
""", unsafe_allow_html=True)

# ── Config ───────────────────────────────────────────────────────────────
LANGFUSE_HOST = "https://langfuse.forgehg.cloud"
DATASET_NAME  = "sykes-connie-functional-evals-v2"
DATASET_ID    = "cmn60zfsi00inzb078j4a21ak"

_today = datetime.now().strftime("%Y%m%d")
try:
    PK         = st.secrets["LANGFUSE_PUBLIC_KEY"]
    SK         = st.secrets["LANGFUSE_SECRET_KEY"]
    PROD_A_RUN = st.secrets.get("PROD_A_RUN", f"eval-prod-a-{_today}")
    PROD_B_RUN = st.secrets.get("PROD_B_RUN", f"eval-prod-b-{_today}")
except Exception:
    PK         = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    SK         = os.getenv("LANGFUSE_SECRET_KEY", "")
    PROD_A_RUN = os.getenv("PROD_A_RUN", f"eval-prod-a-{_today}")
    PROD_B_RUN = os.getenv("PROD_B_RUN", f"eval-prod-b-{_today}")

# Dataset evaluators to show (hyphen naming)
DATASET_EVALS = {
    "connie-task-completion-ds": ("Task Completion",       80),
    "connie-task-adherence":     ("Task Adherence",        80),
    "cs-handoff-ds":             ("CS Handoff",            80),
    "brand-tone-ds":             ("Brand Tone",            70),
    "length-ok-ds":              ("Length OK",             85),
    "connie-hallucination-ds":   ("Hallucination Resist.", 80),
    "task-completion":           ("Task Completion (v1)",  80),
}

IMPROVEMENT_THRESHOLD = 5.0

# ── Langfuse client ──────────────────────────────────────────────────────
@st.cache_resource
def get_langfuse():
    import httpx
    _orig = httpx.Client.__init__
    def _p(self, *a, **k): k['verify'] = False; _orig(self, *a, **k)
    httpx.Client.__init__ = _p
    _orig_a = httpx.AsyncClient.__init__
    def _pa(self, *a, **k): k['verify'] = False; _orig_a(self, *a, **k)
    httpx.AsyncClient.__init__ = _pa

    from langfuse import Langfuse
    return Langfuse(public_key=PK, secret_key=SK, host=LANGFUSE_HOST)

# ── Data fetching ────────────────────────────────────────────────────────
@st.cache_data(ttl=120)
def fetch_run_scores(run_name):
    """Fetch scores for a dataset run using SDK."""
    import requests
    from requests.auth import HTTPBasicAuth

    auth  = HTTPBasicAuth(PK, SK)
    items = []
    page  = 1

    while True:
        resp = requests.get(
            f"{LANGFUSE_HOST}/api/public/dataset-run-items",
            auth=auth,
            params={"runName": run_name, "datasetId": DATASET_ID,
                    "page": page, "limit": 50},
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
    trace_rows     = []

    for item in items:
        trace_id = item.get("traceId")
        if not trace_id:
            continue
        try:
            sr = requests.get(
                f"{LANGFUSE_HOST}/api/public/scores",
                auth=auth,
                params={"traceId": trace_id, "limit": 100},
                verify=False
            )
            if sr.status_code != 200:
                continue
            all_scores   = sr.json().get("data", [])
            item_scores  = {}
            for s in all_scores:
                name = s.get("name", "")
                val  = s.get("value")
                if val is not None and name in DATASET_EVALS:
                    scores_by_name[name].append(val)
                    if name not in item_scores:
                        item_scores[name] = val

            # fetch trace for drilldown
            tr = requests.get(
                f"{LANGFUSE_HOST}/api/public/traces/{trace_id}",
                auth=auth, verify=False
            )
            if tr.status_code == 200:
                import json
                t   = tr.json()
                inp = t.get("input") or {}
                out = t.get("output") or {}
                if isinstance(inp, str):
                    try: inp = json.loads(inp)
                    except: inp = {}
                if isinstance(out, str):
                    try: out = json.loads(out)
                    except: out = {}
                trace_rows.append({
                    "trace_id":     trace_id,
                    "user_message": inp.get("user_message", "")[:120],
                    "response":     out.get("response", out.get("text", ""))[:220],
                    "scores":       item_scores,
                })
        except Exception:
            continue

    return scores_by_name, trace_rows, len(items)


def summarise(scores_by_name):
    seen, result = set(), {}
    for name, values in scores_by_name.items():
        if name not in DATASET_EVALS or not values:
            continue
        label, threshold = DATASET_EVALS[name]
        if label in seen:
            continue
        seen.add(label)
        passing = sum(1 for v in values if v > 0.5)
        result[label] = {
            "pct":       round(passing / len(values) * 100, 1),
            "n":         len(values),
            "threshold": threshold,
            "raw_name":  name,
        }
    return result


def colour(pct, thr):
    if pct >= thr:              return "#27AE60"
    elif pct >= thr * 0.85:     return "#F39C12"
    else:                       return "#E74C3C"


def score_card(label, pct, thr, n, delta=None):
    c = colour(pct, thr)
    dh = ""
    if delta is not None:
        arrow  = "▲" if delta > 0 else ("▼" if delta < 0 else "→")
        dc     = "#27AE60" if delta > 0 else ("#E74C3C" if delta < 0 else "#888")
        dh     = f'<span style="color:{dc};font-size:0.82rem;margin-left:8px">{arrow} {abs(delta):.1f}pp</span>'
    return f"""
<div class="score-card" style="border-left:4px solid {c}">
  <div style="color:#666;font-size:0.7rem;text-transform:uppercase;letter-spacing:1.2px;margin-bottom:4px">{label}</div>
  <div style="display:flex;align-items:baseline">
    <span style="color:{c};font-size:1.9rem;font-weight:700">{pct}%</span>{dh}
  </div>
  <div style="color:#444;font-size:0.68rem;margin-top:3px">threshold ≥{thr}% · n={n}</div>
</div>"""

# ── Header ───────────────────────────────────────────────────────────────
hc1, hc2 = st.columns([5, 1])
with hc1:
    st.markdown("""
    <h1>🏡 Connie — A/B Test Dashboard</h1>
    <p style="color:#555;font-size:0.82rem;margin-top:-6px">
        Live · langfuse.forgehg.cloud · 173 cases · dataset evaluators only
    </p>""", unsafe_allow_html=True)
with hc2:
    st.markdown("<div style='margin-top:18px'>", unsafe_allow_html=True)
    if st.button("↻ Refresh"):
        st.cache_data.clear()
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📊  A/B Comparison", "🔍  Case Drilldown", "ℹ️  About"])

# ══ TAB 1 — A/B COMPARISON ═══════════════════════════════════════════════
with tab1:
    with st.spinner("Fetching prod-a scores…"):
        a_raw, a_traces, a_n = fetch_run_scores(PROD_A_RUN)
    with st.spinner("Fetching prod-b scores…"):
        b_raw, b_traces, b_n = fetch_run_scores(PROD_B_RUN)

    a_sum = summarise(a_raw)
    b_sum = summarise(b_raw)

    if not a_sum and not b_sum:
        st.warning(f"""No scores found for runs:
- **{PROD_A_RUN}**
- **{PROD_B_RUN}**

Evaluators may still be scoring — wait a few minutes and refresh.
If this persists, check Langfuse → LLM-as-a-Judge for pending counts.""")
    else:
        # Verdict
        wins_b, wins_a, ties = 0, 0, 0
        all_labels = sorted(set(list(a_sum) + list(b_sum)))
        for m in all_labels:
            ap = a_sum.get(m, {}).get("pct")
            bp = b_sum.get(m, {}).get("pct")
            if ap is None or bp is None: continue
            d = bp - ap
            if d >= IMPROVEMENT_THRESHOLD:    wins_b += 1
            elif d <= -IMPROVEMENT_THRESHOLD: wins_a += 1
            else:                             ties   += 1

        if wins_b > wins_a:
            vc = "border:1px solid #27AE60;background:linear-gradient(135deg,#0d2a1a,#0f2e1c)"
            vi, vt = "✅", "Recommendation: Promote prod-b to production"
            vb = f"prod-b outperforms prod-a on <strong>{wins_b}</strong> metric(s) by ≥5pp. Move the <strong>production</strong> label to prod-b in Langfuse Prompt Management."
        elif wins_a > wins_b:
            vc = "border:1px solid #E74C3C;background:linear-gradient(135deg,#2a0d0d,#2e0f0f)"
            vi, vt = "❌", "Recommendation: Keep prod-a — prod-b does not improve"
            vb = f"prod-a outperforms prod-b on <strong>{wins_a}</strong> metric(s). Investigate prod-b before promoting."
        else:
            vc = "border:1px solid #F39C12;background:linear-gradient(135deg,#2a1f0d,#2e220f)"
            vi, vt = "⏳", "No clear winner yet"
            vb = "Scores are within the ±5pp significance threshold. Evaluators may still be running — refresh in a few minutes."

        st.markdown(f"""
        <div style="border-radius:12px;padding:20px 24px;margin-bottom:24px;{vc}">
            <div style="font-size:1.05rem;font-weight:700;color:#fff;margin-bottom:6px">{vi}&nbsp; {vt}</div>
            <div style="color:#bbb;font-size:0.88rem">{vb}</div>
            <div style="margin-top:10px;color:#555;font-size:0.72rem">
                prod-b wins: {wins_b} · prod-a wins: {wins_a} · ties: {ties} · significance: ±{IMPROVEMENT_THRESHOLD}pp
            </div>
        </div>""", unsafe_allow_html=True)

        # Run info
        st.markdown('<div class="section-hdr">Run details</div>', unsafe_allow_html=True)
        r1, r2 = st.columns(2)
        with r1:
            st.markdown(f"""
            <div style="background:#13151f;border-radius:8px;padding:14px 16px;border-left:4px solid #5b9fff">
                <span style="background:#0d1a2e;color:#5b9fff;border:1px solid #1e3a6a;padding:3px 10px;border-radius:20px;font-size:0.72rem;font-weight:600;font-family:'JetBrains Mono',monospace">prod-a</span>
                <div style="margin-top:8px;color:#aaa;font-size:0.78rem;font-family:'JetBrains Mono',monospace">{PROD_A_RUN}</div>
                <div style="color:#555;font-size:0.72rem;margin-top:4px">{a_n} cases · current production · baseline prompt</div>
            </div>""", unsafe_allow_html=True)
        with r2:
            st.markdown(f"""
            <div style="background:#13151f;border-radius:8px;padding:14px 16px;border-left:4px solid #b05bff">
                <span style="background:#1a0d2e;color:#b05bff;border:1px solid #3a1e6a;padding:3px 10px;border-radius:20px;font-size:0.72rem;font-weight:600;font-family:'JetBrains Mono',monospace">prod-b</span>
                <div style="margin-top:8px;color:#aaa;font-size:0.78rem;font-family:'JetBrains Mono',monospace">{PROD_B_RUN}</div>
                <div style="color:#555;font-size:0.72rem;margin-top:4px">{b_n} cases · challenger · CS handoff URL fix</div>
            </div>""", unsafe_allow_html=True)

        # Score cards
        st.markdown('<div class="section-hdr">Evaluator scores</div>', unsafe_allow_html=True)
        ca, cb = st.columns(2)
        with ca:
            st.markdown('<div style="color:#5b9fff;font-size:0.78rem;font-weight:600;margin-bottom:10px">prod-a — current production</div>', unsafe_allow_html=True)
            for m in all_labels:
                ai = a_sum.get(m)
                bi = b_sum.get(m)
                thr = (ai or bi or {}).get("threshold", 80)
                if ai:
                    delta = -(bi["pct"] - ai["pct"]) if bi else None
                    st.markdown(score_card(m, ai["pct"], thr, ai["n"], delta), unsafe_allow_html=True)
                else:
                    st.markdown(f'<div style="color:#333;font-size:0.78rem;padding:8px 0">No data — {m}</div>', unsafe_allow_html=True)

        with cb:
            st.markdown('<div style="color:#b05bff;font-size:0.78rem;font-weight:600;margin-bottom:10px">prod-b — challenger</div>', unsafe_allow_html=True)
            for m in all_labels:
                ai = a_sum.get(m)
                bi = b_sum.get(m)
                thr = (bi or ai or {}).get("threshold", 80)
                if bi:
                    delta = (bi["pct"] - ai["pct"]) if ai else None
                    st.markdown(score_card(m, bi["pct"], thr, bi["n"], delta), unsafe_allow_html=True)
                else:
                    st.markdown(f'<div style="color:#333;font-size:0.78rem;padding:8px 0">No data — {m}</div>', unsafe_allow_html=True)

# ══ TAB 2 — CASE DRILLDOWN ════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-hdr">Case drilldown</div>', unsafe_allow_html=True)
    run_choice = st.radio("Select run", ["prod-a", "prod-b"], horizontal=True)
    show_fails = st.checkbox("Failing cases only", value=True)

    traces = a_traces if run_choice == "prod-a" else b_traces

    if not traces:
        st.info("No trace data available yet.")
    else:
        shown = 0
        for t in traces:
            tc      = t["scores"].get("connie-task-completion-ds",
                       t["scores"].get("task-completion"))
            passing = tc is not None and tc > 0.5
            if show_fails and passing:
                continue
            border = "#27AE60" if passing else "#E74C3C"
            pills  = "".join(
                f'<span class="score-pill {"s-pass" if v>0.5 else "s-fail"}">'
                f'{DATASET_EVALS.get(k,(k,))[0]}: {v:.2f}</span>'
                for k, v in t["scores"].items() if k in DATASET_EVALS
            )
            q = t.get("user_message", "")
            r = t.get("response", "")
            st.markdown(f"""
            <div class="trace-card" style="border-left:3px solid {border}">
                <div style="color:#666;font-size:0.72rem;margin-bottom:6px">💬 {q}</div>
                <div style="color:#ccc;font-style:italic;margin-bottom:8px;font-size:0.82rem">"{r}{"…" if len(r)>=220 else ""}"</div>
                <div>{pills or '<span style="color:#333;font-size:0.7rem">No dataset scores yet</span>'}</div>
            </div>""", unsafe_allow_html=True)
            shown += 1
            if shown >= 25:
                break
        if shown == 0:
            st.success("No failing cases — all cases passed threshold.")

# ══ TAB 3 — ABOUT ════════════════════════════════════════════════════════
with tab3:
    st.markdown(f"""
    ### What this dashboard shows
    Live A/B test results comparing two Connie prompt versions, read directly from Langfuse
    using the Langfuse Python SDK. All scores generated automatically — no manual scoring.

    **prod-a** `{PROD_A_RUN}` — current production, baseline prompt  
    **prod-b** `{PROD_B_RUN}` — challenger: CS handoff updated to `sykes.co.uk/contact`

    ### Evaluators shown (dataset runs only)
    | Evaluator | Threshold | Priority |
    |---|---|---|
    | Task Completion | ≥ 80% | Critical |
    | Task Adherence | ≥ 80% | Critical |
    | CS Handoff | ≥ 80% | Critical |
    | Hallucination Resistance | ≥ 80% | Critical |
    | Brand Tone | ≥ 70% | Important |
    | Length OK | ≥ 85% | Informational |

    ### How to promote a winner
    Langfuse → Prompt Management → connie-agent → move `production` label to prod-b version.

    ### Dataset
    `{DATASET_NAME}` — 173 cases: search (93) · property questions (15) ·
    CS handoff (20) · objection handling (15) · adversarial/safety (15) · end-to-end (15)

    ---
    *Data: `{LANGFUSE_HOST}` · refreshes every 2 min · Calbert Graham, Data Science*
    """)
