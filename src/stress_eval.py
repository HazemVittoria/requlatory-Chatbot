from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .qa_engine import answer


@dataclass(frozen=True)
class StressCase:
    id: str
    question: str
    expected_mode: str


def load_stress(path: str | Path) -> list[StressCase]:
    p = Path(path)
    cases: list[StressCase] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        cases.append(
            StressCase(
                id=obj["id"],
                question=obj["question"],
                expected_mode=obj.get("expected_mode", "answer"),
            )
        )
    return cases


def _predict_mode(text: str) -> str:
    if "insufficient evidence" in (text or "").lower():
        return "insufficient"
    return "answer"


def evaluate(path: str | Path, threshold: float) -> dict[str, Any]:
    os.environ["QA_CONF_THRESHOLD"] = str(threshold)
    cases = load_stress(path)

    total = len(cases)
    correct = 0
    fp = 0  # predicted answer but should be insufficient
    fn = 0  # predicted insufficient but should be answer
    failures: list[dict[str, Any]] = []

    for c in cases:
        res = answer(c.question)
        pred = _predict_mode(res.text or "")
        ok = pred == c.expected_mode
        if ok:
            correct += 1
        else:
            if pred == "answer" and c.expected_mode == "insufficient":
                fp += 1
            if pred == "insufficient" and c.expected_mode == "answer":
                fn += 1

        conf = None
        for item in (res.used_chunks or []):
            if isinstance(item, dict) and item.get("kind") == "confidence":
                conf = item
                break

        if not ok:
            failures.append(
                {
                    "id": c.id,
                    "question": c.question,
                    "expected_mode": c.expected_mode,
                    "predicted_mode": pred,
                    "confidence": conf,
                    "answer_preview": (res.text or "")[:240],
                    "citations": [f"{x.doc_id}|p{x.page}|{x.chunk_id}" for x in (res.citations or [])],
                }
            )

    return {
        "threshold": threshold,
        "total": total,
        "correct": correct,
        "accuracy": round((correct / total) if total else 0.0, 4),
        "false_positive_answer": fp,
        "false_negative_insufficient": fn,
        "failures": failures,
    }


def sweep(path: str | Path, thresholds: list[float]) -> dict[str, Any]:
    runs = [evaluate(path, t) for t in thresholds]
    best = max(runs, key=lambda x: (x["accuracy"], -x["false_positive_answer"]))
    return {"runs": runs, "best": best}


if __name__ == "__main__":
    stress_path = Path(__file__).resolve().parents[1] / "tests" / "stress_set.jsonl"
    thresholds = [0.18, 0.22, 0.26, 0.30, 0.34]
    out = sweep(stress_path, thresholds)
    print(json.dumps(out, indent=2, ensure_ascii=False))
