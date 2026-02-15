# Regulatory Chatbot

Regulatory Q&A system for PDF corpora (ICH/FDA/EMA/SOPs) with retrieval, rerank, citations, confidence gating, and optional web API/UI.

## Run CLI

```bash
python -m src.cli "What documentation is required for training and qualification?"
```

Force index rebuild:

```bash
python -m src.cli --rebuild-index "How should computerized systems be validated?"
```

## Run Web API + UI

```bash
python -m src.web_api --host 127.0.0.1 --port 8000
```

Open:

- `http://127.0.0.1:8000/` UI
- `http://127.0.0.1:8000/health`
- `http://127.0.0.1:8000/metrics`
- `http://127.0.0.1:8000/admin/report?hours=24`

## Logging + Monitoring

- Request telemetry log (JSONL): `logs/qa_requests.jsonl`
- Each API request stores:
  - timestamp, question hash/text, intent/scope
  - latency, insufficient flag
  - confidence metrics
  - top citations
  - suggestions shown to user

Generate rolling report:

```bash
python -m src.monitor_report --hours 24
python -m src.monitor_report --hours 24 --out logs/reports/latest.json
```

## Retrieval Metrics

```bash
python -m src.retrieval_metrics --json
python -m src.retrieval_metrics --json --compare-semantic --semantic-weight 0.18 --semantic-top-n 30
```

## Tests

```bash
pytest -q
```
