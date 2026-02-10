# src/eval_runner.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from .qa_engine import answer




@dataclass(frozen=True)
class EvalCase:
    id: str
    question: str
    scope: str
    intent: str
    anchor_terms: list[str]
    must_include: list[str]
    must_not_include: list[str]
    min_citations: int
    max_citations: int
    format_rules: dict[str, Any]


def load_golden(path: str | Path) -> list[EvalCase]:
    p = Path(path)
    cases: list[EvalCase] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        cases.append(
            EvalCase(
                id=obj["id"],
                question=obj["question"],
                scope=obj.get("scope", "MIXED"),
                intent=obj.get("intent", "unknown"),
                anchor_terms=obj.get("anchor_terms", []),
                must_include=obj.get("must_include", []),
                must_not_include=obj.get("must_not_include", []),
                min_citations=int(obj.get("min_citations", 1)),
                max_citations=int(obj.get("max_citations", 6)),
                format_rules=obj.get("format_rules", {}),
            )
        )
    return cases


def run_eval(golden_path: str | Path = None) -> dict[str, Any]:
    if golden_path is None:
        golden_path = Path(__file__).resolve().parents[1] / "golden_set.jsonl"

    cases = load_golden(golden_path)
    results: list[dict[str, Any]] = []

    for c in cases:
        res = answer(c.question)
        results.append(
            {
                "id": c.id,
                "question": c.question,
                "intent_expected": c.intent,
                "intent_got": res.intent,
                "scope_expected": c.scope,
                "scope_got": res.scope,
                "citations": [vars(x) for x in (res.citations or [])],
                "text": res.text,
            }
        )

    return {"count": len(results), "results": results}


if __name__ == "__main__":
    out = run_eval()
    print(json.dumps(out, indent=2, ensure_ascii=False))
