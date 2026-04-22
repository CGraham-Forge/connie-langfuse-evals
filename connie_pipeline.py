"""
connie_pipeline.py
──────────────────────────────────────────────────────────────────────
Connie MLOps pipeline — uses Langfuse's built-in LLM evaluator.
No external LLM keys needed — all scoring done by Langfuse internally.

Usage:
    python connie_pipeline.py --step pull_traces
    python connie_pipeline.py --step run_eval
    python connie_pipeline.py --step monitor_ab

Environment variables needed (just Langfuse):
    LANGFUSE_PUBLIC_KEY
    LANGFUSE_SECRET_KEY
    LANGFUSE_HOST
    MASTRA_HOST  (optional, defaults to https://mastra.staging.forgehg.cloud)
"""

import os
import json
import time
import argparse
from datetime import datetime, timedelta, timezone

from langfuse import Langfuse

# ── Config ────────────────────────────────────────────────────────────
DATASET_NAME      = "sykes-connie-functional-evals-v2"
PROMPT_NAME       = "connie-agent"
SCORE_NAME        = "connie-task-completion"
FAILURE_THRESHOLD = 0.6
AB_THRESHOLD      = 0.05
AB_MONITOR_HOURS  = 48
MIN_AB_TRACES     = 50

# ── Client ────────────────────────────────────────────────────────────
langfuse = Langfuse(
    public_key = os.environ["LANGFUSE_PUBLIC_KEY"],
    secret_key = os.environ["LANGFUSE_SECRET_KEY"],
    host       = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
)


# ══════════════════════════════════════════════════════════════════════
# STEP 1+2+3 — Pull failing traces, deduplicate, add to dataset
# ══════════════════════════════════════════════════════════════════════

def pull_and_add_failures():
    """
    Pull traces from last 24hrs scoring below threshold.
    Add new ones to golden dataset as pending review.
    """
    print("\n── STEP 1: Pulling failing traces from last 24hrs ──")

    import requests
    from requests.auth import HTTPBasicAuth

    since     = datetime.now(timezone.utc) - timedelta(hours=24)
    since_str = since.strftime("%Y-%m-%dT%H:%M:%SZ")
    host      = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    auth      = HTTPBasicAuth(os.environ["LANGFUSE_PUBLIC_KEY"], os.environ["LANGFUSE_SECRET_KEY"])

    failing_trace_ids = {}
    page = 1

    while True:
        resp = requests.get(
            f"{host}/api/public/scores",
            auth   = auth,
            params = {
                "fromTimestamp": since_str,
                "name":          SCORE_NAME,
                "page":          page,
                "limit":         100
            }
        )
        resp.raise_for_status()
        data = resp.json().get("data", [])
        if not data:
            break
        for score in data:
            val      = score.get("value", 1.0)
            trace_id = score.get("traceId")
            if val < FAILURE_THRESHOLD and trace_id:
                failing_trace_ids[trace_id] = {
                    "id":    trace_id,
                    "score": val,
                    "input": score.get("comment", "")
                }
        if len(data) < 100:
            break
        page += 1

    failing_traces = []
    for trace_id, score_info in failing_trace_ids.items():
        resp = requests.get(
            f"{host}/api/public/traces/{trace_id}",
            auth = auth
        )
        if resp.status_code == 200:
            trace = resp.json()
            trace["_score_value"] = score_info["score"]
            failing_traces.append(trace)

    print(f"  Found {len(failing_traces)} failing traces (score < {FAILURE_THRESHOLD})")

    if not failing_traces:
        print("  Nothing to add today.")
        return 0

    print("\n── STEP 2: Deduplicating against existing dataset ──")
    dataset = langfuse.get_dataset(DATASET_NAME)
    existing_ids = set(
        item.source_trace_id
        for item in dataset.items
        if item.source_trace_id
    )
    new_failures = [t for t in failing_traces if t.get("id") not in existing_ids]
    print(f"  {len(failing_traces) - len(new_failures)} already in dataset")
    print(f"  {len(new_failures)} new items to add")

    if not new_failures:
        print("  Nothing new to add.")
        return 0

    print(f"\n── STEP 3: Adding {len(new_failures)} items to dataset ──")
    for trace in new_failures:
        raw_input  = trace.get("input") or {}
        user_input = raw_input if isinstance(raw_input, str) else raw_input.get("input", json.dumps(raw_input))
        score_val  = trace.get("_score_value")

        langfuse.create_dataset_item(
            dataset_name    = DATASET_NAME,
            input           = {
                "messages":     [],
                "user_message": user_input
            },
            expected_output = {
                "expected_action": "pending_review",
                "pass_criteria":   "To be filled in during manual review",
                "fail_criteria":   f"Auto-added — scored {score_val} on {SCORE_NAME}",
                "scoring_method":  "llm_judge"
            },
            source_trace_id = trace.get("id"),
            metadata        = {
                "status":     "pending_review",
                "auto_added": True,
                "score":      score_val,
                "added_at":   datetime.now(timezone.utc).isoformat()
            }
        )

    langfuse.flush()
    print(f"  ✓ Done — {len(new_failures)} items added")
    print(f"\n  ⏸  ACTION REQUIRED:")
    print(f"  Go to Langfuse → Datasets → {DATASET_NAME}")
    print(f"  Review new items → archive ones that aren't useful")
    print(f"  Then run: python connie_pipeline.py --step run_eval")
    return len(new_failures)


# ══════════════════════════════════════════════════════════════════════
# STEP 5+8 — Run eval and compare prompts
# ══════════════════════════════════════════════════════════════════════

def run_eval():
    """
    Run golden dataset against prod-a and prod-b prompt labels.
    Langfuse evaluators score responses automatically.
    Compare scores and report winner.
    """
    import requests as req

    today  = datetime.now().strftime('%Y%m%d')
    mastra = os.getenv("MASTRA_HOST", "https://mastra.staging.forgehg.cloud")

    def call_connie(item, prompt_label=None):
        """Call Connie via Mastra with full conversation history."""
        inp = item.input
        if isinstance(inp, str):
            try:
                inp = json.loads(inp)
            except Exception:
                inp = {"messages": [], "user_message": inp}

        messages = []
        for msg in (inp.get("messages") or []):
            role = "assistant" if msg.get("role") == "ai" else "user"
            messages.append({"role": role, "content": msg.get("content", "")})
        messages.append({"role": "user", "content": inp.get("user_message", "")})

        payload = {
            "messages":   messages,
            "threadId":   f"eval-{item.id}-{prompt_label or 'prod'}",
            "resourceId": "eval-runner"
        }
        if prompt_label:
            payload["promptLabel"] = prompt_label

        resp = req.post(
            f"{mastra}/api/agents/connie-agent/generate",
            json    = payload,
            timeout = 30
        )
        resp.raise_for_status()
        return resp.json().get("text", "")

    def run_and_score(run_name, description, prompt_label=None):
        """Run all dataset items through Connie and log traces to Langfuse."""
        dataset = langfuse.get_dataset(DATASET_NAME)
        scores  = []
        total   = len(dataset.items)

        for i, item in enumerate(dataset.items):
            try:
                response = call_connie(item, prompt_label=prompt_label)

                inp2 = item.input if isinstance(item.input, dict) else json.loads(item.input or "{}")
                eo   = item.expected_output
                if isinstance(eo, str):
                    try:
                        eo = json.loads(eo)
                    except Exception:
                        eo = {}

                trace_input = {
                    "user_message":    inp2.get("user_message", ""),
                    "conversation":    inp2.get("messages", []),
                    "pass_criteria":   (eo or {}).get("pass_criteria", ""),
                    "fail_criteria":   (eo or {}).get("fail_criteria", ""),
                    "expected_action": (eo or {}).get("expected_action", ""),
                    "scoring_method":  (eo or {}).get("scoring_method", ""),
                    "dataset_item_id": item.id,
                    "run":             run_name,
                    "prompt_label":    prompt_label or "default"
                }
                trace_output = {
                    "response":       response,
                    "ideal_response": (eo or {}).get("ideal_response", "")
                }

                with langfuse.start_as_current_observation(
                    name   = run_name,
                    input  = trace_input,
                    output = trace_output
                ):
                    trace_id = langfuse.get_current_trace_id()

                langfuse.api.dataset_run_items.create(
                    run_name        = run_name,
                    dataset_item_id = item.id,
                    trace_id        = trace_id,
                    run_description = description
                )
                scores.append(1)

                if (i + 1) % 10 == 0:
                    print(f"  {i+1}/{total} done")

            except Exception as e:
                print(f"  Error on item {item.id}: {e}")

        langfuse.flush()
        n = len(scores)
        print(f"  ✓ {n} traces logged — Langfuse evaluator will score automatically")
        return 0.0, n

    # ── Run prod-a eval ───────────────────────────────────────────────
    print("\n── STEP 5: Running eval on prod-a (current production) ──")
    prod_a_run  = f"eval-prod-a-{today}"
    _, prod_a_n = run_and_score(prod_a_run, f"prod-a eval — {today}", prompt_label="prod-a")
    print(f"  prod-a: {prod_a_n} traces logged")

    # ── Run prod-b eval ───────────────────────────────────────────────
    print("\n── STEP 8: Running eval on prod-b (challenger) ──")
    prod_b_run  = f"eval-prod-b-{today}"
    _, prod_b_n = run_and_score(prod_b_run, f"prod-b eval — {today}", prompt_label="prod-b")
    print(f"  prod-b: {prod_b_n} traces logged")

    # ── Step 9: Report ────────────────────────────────────────────────
    print(f"\n── STEP 9: Runs complete ──")
    print(f"  prod-a run : {prod_a_run}")
    print(f"  prod-b run : {prod_b_run}")
    print(f"\n  Langfuse evaluators are now scoring both runs automatically.")
    print(f"  Once scoring completes, compare runs in Langfuse:")
    print(f"  Datasets → {DATASET_NAME} → Runs → select both and compare")
    print(f"\n  Or use aggregate_scores.py to pull the numbers:")
    print(f"  python aggregate_scores.py --run-name {prod_a_run}")
    print(f"  python aggregate_scores.py --run-name {prod_b_run}")


# ══════════════════════════════════════════════════════════════════════
# STEP 11+12 — Monitor A/B and declare winner
# ══════════════════════════════════════════════════════════════════════

def monitor_ab():
    """
    Compare live scores between prod-a and prod-b.
    Report winner based on connie-task-completion scores.
    """
    print(f"\n── STEP 11: Monitoring A/B test (last {AB_MONITOR_HOURS}hrs) ──")

    import requests
    from requests.auth import HTTPBasicAuth

    since     = datetime.now(timezone.utc) - timedelta(hours=AB_MONITOR_HOURS)
    since_str = since.strftime("%Y-%m-%dT%H:%M:%SZ")
    host      = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    auth      = HTTPBasicAuth(os.environ["LANGFUSE_PUBLIC_KEY"], os.environ["LANGFUSE_SECRET_KEY"])

    prod_a_scores = []
    prod_b_scores = []
    page = 1

    while True:
        resp = requests.get(
            f"{host}/api/public/traces",
            auth   = auth,
            params = {"fromTimestamp": since_str, "page": page, "limit": 100}
        )
        resp.raise_for_status()
        data = resp.json().get("data", [])
        if not data:
            break
        for trace in data:
            label = (trace.get("metadata") or {}).get("prompt_label")
            score = next(
                (s.get("value") for s in (trace.get("scores") or [])
                 if s.get("name") == SCORE_NAME),
                None
            )
            if score is None:
                continue
            if label == "prod-a":
                prod_a_scores.append(score)
            elif label == "prod-b":
                prod_b_scores.append(score)
        if len(data) < 100:
            break
        page += 1

    if prod_a_scores:
        print(f"  prod-a: {len(prod_a_scores)} traces, mean={sum(prod_a_scores)/len(prod_a_scores):.3f}")
    else:
        print(f"  prod-a: no data yet")

    if prod_b_scores:
        print(f"  prod-b: {len(prod_b_scores)} traces, mean={sum(prod_b_scores)/len(prod_b_scores):.3f}")
    else:
        print(f"  prod-b: no data yet")

    if len(prod_a_scores) < MIN_AB_TRACES or len(prod_b_scores) < MIN_AB_TRACES:
        print(f"\n  Not enough data yet — need {MIN_AB_TRACES} traces per variant")
        return

    prod_a_mean = sum(prod_a_scores) / len(prod_a_scores)
    prod_b_mean = sum(prod_b_scores) / len(prod_b_scores)
    improvement = prod_b_mean - prod_a_mean

    print(f"\n── STEP 12: Decision ──")
    if improvement >= AB_THRESHOLD:
        print(f"  ✓ prod-b wins! (improvement={improvement:+.3f})")
        print(f"  → Go to Langfuse → Prompt Management → connie-agent")
        print(f"  → Move the 'production' label from prod-a version to prod-b version")
    else:
        print(f"  ✗ prod-b did not win (improvement={improvement:+.3f})")
        print(f"  → Keep prod-a as production")


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Connie MLOps pipeline")
    parser.add_argument("--step", choices=[
        "pull_traces", "run_eval", "monitor_ab"
    ], required=True)
    args = parser.parse_args()

    print("=" * 60)
    print(f"CONNIE PIPELINE — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Step: {args.step}")
    print("=" * 60)

    if args.step == "pull_traces":
        pull_and_add_failures()
    elif args.step == "run_eval":
        run_eval()
    elif args.step == "monitor_ab":
        monitor_ab()

if __name__ == "__main__":
    main()