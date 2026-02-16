from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import pytest

from src.qa_engine import answer

STRESS_PATH = Path(__file__).resolve().parent / "real_failures_set.jsonl"
_SOURCE_RE = re.compile(r"\[Source:\s*(.+?),\s*p\.(\d+)\]")


@dataclass(frozen=True)
class StressCase:
    id: str
    question: str
    expected_mode: str
    must_include: list[str]
    must_not_include: list[str]
    require_source_tags: bool
    min_citations: int


def _load_stress(path: Path) -> list[StressCase]:
    cases: list[StressCase] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        cases.append(
            StressCase(
                id=obj["id"],
                question=obj["question"],
                expected_mode=obj.get("expected_mode", "answer"),
                must_include=obj.get("must_include", []),
                must_not_include=obj.get("must_not_include", []),
                require_source_tags=bool(obj.get("require_source_tags", False)),
                min_citations=int(obj.get("min_citations", 1)),
            )
        )
    return cases


def _tokenize(text: str) -> set[str]:
    stop = {
        "what",
        "how",
        "when",
        "where",
        "which",
        "the",
        "and",
        "for",
        "with",
        "this",
        "that",
        "from",
        "into",
        "under",
    }
    words = {w.lower() for w in re.findall(r"[a-zA-Z]{3,}", text or "")}

    def norm(w: str) -> str:
        if w.startswith("qualif"):
            return "qualif"
        if w.startswith("test") or w in {"analytical", "analysis"}:
            return "test"
        if w in {"device", "equipment", "instrument", "instruments"}:
            return "equipment"
        return w

    return {norm(w) for w in words if w not in stop}


def _source_tags(text: str) -> list[tuple[str, int]]:
    tags: list[tuple[str, int]] = []
    for m in _SOURCE_RE.finditer(text or ""):
        tags.append((m.group(1).strip(), int(m.group(2))))
    return tags


@pytest.mark.parametrize("case", _load_stress(STRESS_PATH))
def test_stress(case: StressCase):
    res = answer(case.question)
    text = (res.text or "").strip()
    low = text.lower()

    assert text, f"{case.id}: empty answer text"

    for s in case.must_include:
        assert s.lower() in low, f"{case.id}: missing must-include '{s}'"
    for s in case.must_not_include:
        assert s.lower() not in low, f"{case.id}: contains must-not '{s}'"

    cits = res.citations or []
    if case.expected_mode == "insufficient":
        assert "insufficient evidence" in low, f"{case.id}: should be insufficient-evidence response"
    else:
        assert "insufficient evidence" not in low, f"{case.id}: should not be insufficient-evidence response"
        assert len(cits) >= case.min_citations, f"{case.id}: too few citations"

        # Off-topic drift guard: answer should overlap question vocabulary.
        q_tokens = _tokenize(case.question)
        a_tokens = _tokenize(text)
        assert q_tokens & a_tokens, f"{case.id}: likely off-topic answer (no token overlap)"

    # Citation-grounding checks
    if case.require_source_tags and case.expected_mode == "answer":
        tags = _source_tags(text)
        bullet_count = sum(1 for ln in text.splitlines() if ln.strip().startswith("- "))
        assert len(tags) >= max(1, min(3, bullet_count)), f"{case.id}: missing source tags in bullets"

        cited_docs = {c.doc_id for c in cits}
        for doc, page in tags:
            assert doc in cited_docs, f"{case.id}: source tag doc not in citations ({doc})"
            assert page > 0, f"{case.id}: invalid source tag page"

    # Confidence metadata should always be present for calibration.
    confidence = None
    for item in (res.used_chunks or []):
        if isinstance(item, dict) and item.get("kind") == "confidence":
            confidence = item
            break

    assert confidence is not None, f"{case.id}: missing confidence metadata"
    for k in ("retrieval_confidence", "sentence_confidence", "overall_confidence", "threshold"):
        assert k in confidence, f"{case.id}: confidence missing '{k}'"
        assert 0.0 <= float(confidence[k]) <= 1.0, f"{case.id}: confidence '{k}' out of range"
