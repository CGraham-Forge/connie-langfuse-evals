

```markdown
# Connie Langfuse Evals

Automated evaluation pipeline for Connie, the Sykes Cottages AI holiday booking assistant. Built and maintained by the Data Science team at Forge Holiday Group.

## What this repo contains

- `connie_pipeline.py` — main eval pipeline that calls Connie via Mastra and logs traces to Langfuse
- `run_evals.py` — runs individual eval cases against the dataset
- `aggregate_scores.py` — pulls scores from Langfuse and prints a pass/fail summary
- `write_scores.py` — fetches A/B scores from Langfuse and writes scores.json for the dashboard
- `ab_dashboard.py` — Streamlit A/B test dashboard reading from scores.json
- `scores.json` — latest A/B test scores (updated after each pipeline run)

## Dataset

173 hand-crafted test cases stored in Langfuse as `sykes-connie-functional-evals-v2`, covering:

- Search (93 cases) — full conversation flow from greeting to property matching
- Property questions (15) — factual questions about specific properties
- CS handoff (20) — directing support issues to customer care
- Objection handling (15) — price objections, bad reviews, hesitation
- Adversarial / safety (15) — jailbreaks, prompt injection, out-of-scope requests
- End-to-end (15) — full multi-turn booking journeys

## Evaluators

There are two types of evaluator configured in Langfuse. Both use the same scoring logic but target different data sources.

### Dataset run evaluators (-ds suffix)

These score the synthetic test cases run through the eval pipeline. They are the basis for all A/B comparison and go-live decisions. Seven evaluators are active and score automatically after each pipeline run.

| Evaluator | Threshold | Priority |
|---|---|---|
| connie-task-completion-ds | ≥ 80% | Critical |
| connie-task-adherence | ≥ 80% | Critical |
| cs-handoff-ds | ≥ 80% | Critical |
| connie-hallucination-ds | ≥ 80% | Critical |
| brand-tone-ds | ≥ 70% | Important |
| length-ok-ds | ≥ 85% | Informational |

### Live trace evaluators (no suffix)

These are designed to score real customer conversations in production. They are currently set to **Inactive** and must remain so until launch.

Before launch the following actions are required — **dev team support needed**:

- Live trace evaluators must be scoped correctly so they only target real Connie customer conversations, not evaluator-generated traces or pipeline traces
- Appropriate filters must be applied in Langfuse (environment = production, trace name = invoke_agent Connie Agent, excluding all Execute evaluator traces)
- Without correct scoping, evaluators will pick up their own execution traces and create an infinite scoring loop — this has been identified and the evaluators have been deactivated pending proper configuration
- Once scoped correctly, live trace evaluators will provide continuous production monitoring alongside the dataset eval pipeline

## Running the pipeline

Set environment variables each session:

```powershell
$env:LANGFUSE_PUBLIC_KEY="pk-lf-..."
$env:LANGFUSE_SECRET_KEY="sk-lf-..."
$env:LANGFUSE_HOST="https://langfuse.forgehg.cloud"
```

Run a full A/B eval:

```powershell
python connie_pipeline.py --step run_eval
```

Fetch scores and update the dashboard:

```powershell
python write_scores.py --prod-a eval-prod-a-YYYYMMDD --prod-b eval-prod-b-YYYYMMDD
git add scores.json
git commit -m "Update A/B scores YYYY-MM-DD"
git push origin master
```

## A/B test workflow

1. Make a prompt change in Langfuse Prompt Management, label it `prod-b`
2. Run the pipeline — it calls Connie with both `prod-a` and `prod-b` labels simultaneously
3. Wait 5-10 minutes for Langfuse evaluators to score automatically
4. Run `write_scores.py` and push `scores.json`
5. Review the Streamlit dashboard for the verdict
6. If prod-b wins by ≥5pp on the target metric, move the `production` label to prod-b in Langfuse

## Dashboard

Deployed at Streamlit Cloud. Reads from `scores.json` — no direct Langfuse connection required.

## Contacts

| Area | Owner |
|---|---|
| Eval pipeline and datasets | Calbert Graham, Data Science |
| Langfuse instance and evaluator configuration | Dev team |
| Prompt changes | PO + DS jointly |
| Go-live decision | Joe Donoghue, Product |
```
