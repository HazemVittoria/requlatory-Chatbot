# tests/test_golden.py
from __future__ import annotations

import re
from pathlib import Path

import pytest

from src.eval_runner import load_golden
from src.qa_engine import answer

GOLDEN_PATH = Path(__file__).resolve().parents[1] / "golden_set.jsonl"


def _is_truncated(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    # obvious truncation markers
    if t.endswith(("(", "e.g.", "e.", "i.e.", ",")):
        return True
    # if not bullets, usually should end with punctuation
    if "\n-" not in t and t[-1] not in ".!?":
        if not re.search(r"[\d\]\)]$", t):
            return True
    return False


def _broken_start(text: str) -> bool:
    t = (text or "").lstrip()
    if not t:
        return True
    # chopped sentence artifact: starts with lowercase word
    return bool(re.match(r"^[a-z]{2,}\b", t))


def _count_bullets(text: str) -> int:
    return sum(1 for ln in (text or "").splitlines() if ln.strip().startswith("- "))


def _has_verb_like(text: str) -> bool:
    verbs = (
        "ensure",
        "document",
        "review",
        "investigate",
        "verify",
        "approve",
        "assess",
        "implement",
        "record",
        "evaluate",
    )
    tl = (text or "").lower()
    return any(v in tl for v in verbs)


@pytest.mark.parametrize("case", load_golden(GOLDEN_PATH))
def test_golden(case):
    res = answer(case.question)

    # Basic
    assert res.text and res.text.strip(), f"{case.id}: empty answer"

    # Quality checks
    assert not _broken_start(res.text), f"{case.id}: broken-start sentence"
    assert not _is_truncated(res.text), f"{case.id}: truncated ending"

    # Must include / must not include
    low = res.text.lower()
    for s in case.must_include:
        assert s.lower() in low, f"{case.id}: missing must-include '{s}'"
    for s in case.must_not_include:
        assert s.lower() not in low, f"{case.id}: contains must-not '{s}'"

    # Intent check (enforce if provided)
    if case.intent and case.intent != "unknown":
        assert res.intent == case.intent, f"{case.id}: intent mismatch got={res.intent} expected={case.intent}"

    # Citations
    cits = res.citations or []
    assert len(cits) >= case.min_citations, f"{case.id}: too few citations ({len(cits)})"
    assert len(cits) <= case.max_citations, f"{case.id}: too many citations ({len(cits)})"

    tup = [(c.doc_id, c.page, c.chunk_id) for c in cits]
    assert len(tup) == len(set(tup)), f"{case.id}: duplicate citations"

    # Formatting rules
    fr = case.format_rules or {}
    min_b = int(fr.get("min_bullets", 0))
    max_b = int(fr.get("max_bullets", 10))
    bcount = _count_bullets(res.text)
    assert bcount >= min_b, f"{case.id}: too few bullets ({bcount})"
    assert bcount <= max_b, f"{case.id}: too many bullets ({bcount})"

    # Procedure: at least one verb-like token
    if case.intent in ("procedure", "procedure_requirements"):
        assert _has_verb_like(res.text), f"{case.id}: procedure lacks verb-like tokens"
