# Weekly Checklist

Use this checklist to run, monitor, and improve the regulatory QA system safely.

## 1) Start Services (Monday or after restart)

```powershell
python -m src.web_api --host 127.0.0.1 --port 8000
```

Verify:

- `http://127.0.0.1:8000/`
- `http://127.0.0.1:8000/health`

## 2) Daily Monitoring (Mon-Fri)

Generate 24h report:

```powershell
python -m src.monitor_report --hours 24 --out logs/reports/day_<YYYY-MM-DD>.json
```

Review in report:

- `insufficient_rate`
- `low_confidence_count`
- `top_low_conf_tokens`
- `recent_failures`

## 3) Add Real Failures (Daily)

From `recent_failures`, add 3-10 meaningful cases to:

- `tests/real_failures_set.jsonl`

Use one JSON object per line:

```json
{"id":"rf001","question":"how to qualify a testing device?","expected_mode":"answer","must_include":["qualification"],"must_not_include":["insufficient evidence"],"require_source_tags":true,"min_citations":1}
```

## 4) Validate Real Failures (Daily)

```powershell
pytest -q tests/test_real_failures.py -vv
```

If tests fail, fix one failure pattern at a time.

## 5) Regression Guard (After Every Fix)

```powershell
pytest -q tests/test_real_failures.py tests/test_golden.py tests/test_stress_eval.py
```

Only keep fixes that pass all three.

## 6) Retrieval Quality Check (2-3x/week)

Baseline:

```powershell
python -m src.retrieval_metrics --json > metrics_base.json
```

Semantic compare:

```powershell
python -m src.retrieval_metrics --json --compare-semantic --semantic-weight 0.18 --semantic-top-n 30 > metrics_semantic_compare.json
```

Review:

- `summary.mrr`
- `summary.recall_at_k`
- `delta`
- `misses`

## 7) Confidence Calibration (Mid-week)

```powershell
python -m src.stress_eval > stress_sweep.json
```

Adjust `QA_CONF_THRESHOLD` only when it improves safety and answer coverage together.

## 8) Cache/Performance Sanity (Weekly)

```powershell
$env:DEBUG_INDEX_CACHE="1"
python -m src.cli "How should computerized systems be validated?"
```

Ensure repeated runs show cache hits.

## 9) Commit Discipline (Daily or Per Feature)

```powershell
git add <files>
git commit -m "<focused change>"
git push origin main
```

Push only after green tests.

## 10) End-of-Week Summary

Capture:

- insufficient-rate trend
- top 5 failure themes
- fixes shipped
- retrieval metric changes
- priority items for next week

---

## How To Use This File

1. Open this file at start of day.
2. Run sections `2 -> 5` each day.
3. Run sections `6 -> 8` mid-week and before major merges.
4. Run section `10` every Friday and create next week’s priorities from it.
