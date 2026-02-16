# Metadata Workflow

Use this flow every time new documents are added.

## 1) Place Documents In Correct Authority Folder

- `data/fda` -> FDA
- `data/ich` -> ICH
- `data/ema` -> EMA
- `data/eu_gmp` -> EU GMP
- `data/pic_s` -> PIC/S
- `data/who` -> WHO
- `data/sop` or `data/sops` -> SOP
- `data/others` -> unknown/mixed

## 2) Restart API And Wait For Index Readiness

```powershell
python -m src.web_api --host 127.0.0.1 --port 8000
```

Check:

- `http://127.0.0.1:8000/health`
- Wait until:
  - `index_ready: true`
  - `index_warming: false`

## 3) Run Daily Quality Gate

```powershell
python -m src.daily_ops
```

Review:

- `logs/reports/daily_ops_<YYYY-MM-DD>.md`
- `logs/reports/daily_ops_<YYYY-MM-DD>.json`

## 4) Find Metadata Gaps (`domain=Other`)

```powershell
python -m src.metadata_report --top 25
```

Optional save:

```powershell
python -m src.metadata_report --top 25 --out logs/reports/metadata_other_<YYYY-MM-DD>.json
```

## 5) Close Gaps

- For high-value docs showing as `domain=Other`, update `src/domain_map.py`
- Keep uncertain docs as `Other` (do not force mislabeling)
- Re-run step 4 until top `Other` docs are acceptable

## 6) Lock With Tests

- Add representative real-failure/golden cases for new topics
- Run:

```powershell
pytest -q tests/test_real_failures.py tests/test_golden.py tests/test_stress_eval.py -p no:cacheprovider
```
