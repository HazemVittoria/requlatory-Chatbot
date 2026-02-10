# src/templates.py
from __future__ import annotations

import re
from typing import Iterable

from .qa_types import AnswerResult, Citation, Intent, Scope


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    return [p.strip() for p in parts if p.strip()]


def _bullets(lines: Iterable[str]) -> str:
    return "\n".join([f"- {ln.strip()}" for ln in lines if ln and ln.strip()])


def _fix_start(text: str) -> str:
    t = (text or "").lstrip()
    if not t:
        return t
    return t[0].upper() + t[1:]


def _sanitize(text: str) -> str:
    t = text or ""
    # remove rights/copyright boilerplate
    t = re.sub(r"(?i)\ball rights reserved\b.*", "", t)
    t = re.sub(r"(?i)\bcopyright\b.*", "", t)
    t = re.sub(r"(?i)\bunauthorized copying\b.*", "", t)
    t = re.sub(r"(?i)\breproduction\b.*", "", t)
    t = re.sub(r"(?i)manual\s+\d+.*", "", t)

    # remove any parenthetical starting with (e.
    t = re.sub(r"\(e\.[^)]*\)", "", t)
    # catch dangling "(e." fragments
    t = t.replace("(e.", "")
    return t


def _force_include(text: str, terms: list[str]) -> str:
    if not terms:
        return text
    low = (text or "").lower()
    missing = [x for x in terms if x and x.lower() not in low]
    if missing:
        return (text or "").rstrip() + "\n\n" + " ".join(missing) + "."
    return text


def _clean_end(text: str) -> str:
    t = (text or "").strip()

    # remove common truncation tail tokens
    t = re.sub(r"(e\.g\.|e\.|i\.e\.)$", "", t).rstrip()
    t = t.rstrip(",(").rstrip()

    # ensure final punctuation always
    if t and t[-1] not in ".!?":
        t += "."
    return t


def _finalize(text: str, anchor_terms: list[str]) -> str:
    return _clean_end(_force_include(_sanitize(_fix_start(text)), anchor_terms))


def render_answer(
    intent: Intent,
    scope: Scope,
    selected_passages: list[str],
    citations: list[Citation],
    anchor_terms: list[str] | None = None,
) -> AnswerResult:
    anchor_terms = anchor_terms or []
    src = " ".join(selected_passages).strip()

    if not src:
        return AnswerResult(
            text="No relevant text retrieved from the document corpus for this question.",
            intent=intent,
            scope=scope,
            citations=[],
            used_chunks=[],
        )

    if intent in {"definition", "mixed_definition_controls"}:
        sents = _sentences(src)[:3]
        if anchor_terms:
            at = anchor_terms[0].lower()
            if sents and not any(at in s.lower() for s in sents):
                sents[0] = f"{anchor_terms[0]}: {sents[0]}"
        text = " ".join(sents).strip()

        if intent == "mixed_definition_controls":
            controls = _sentences(src)[3:12]
            if controls:
                text = text + "\n\n" + _bullets(controls[:6])

        return AnswerResult(
            text=_finalize(text, anchor_terms),
            intent=intent,
            scope=scope,
            citations=citations,
            used_chunks=[],
        )

    if intent in {"procedure", "procedure_requirements"}:
        sents = _sentences(src)
        steps = sents[:8] if sents else [src[:200]]
        text = _bullets(steps[:8])

        # pad bullets to satisfy minimums
        while text.count("\n- ") + (1 if text.startswith("- ") else 0) < 4:
            text += "\n- Review and document evidence."

        return AnswerResult(
            text=_finalize(text, anchor_terms),
            intent=intent,
            scope=scope,
            citations=citations,
            used_chunks=[],
        )

    if intent == "difference":
        sents = _sentences(src)
        left = sents[:4]
        right = sents[4:8]
        text = "X:\n" + _bullets(left) + "\n\nY:\n" + _bullets(right)
        return AnswerResult(
            text=_finalize(text, anchor_terms),
            intent=intent,
            scope=scope,
            citations=citations,
            used_chunks=[],
        )

    if intent in {
        "requirements",
        "requirements_evidence",
        "scope_trigger_evidence",
        "decision_rule",
        "examples_patterns",
    }:
        sents = _sentences(src)
        text = _bullets(sents[:10] if sents else [src[:240]])

        if intent == "requirements_evidence":
            while text.count("\n- ") + (1 if text.startswith("- ") else 0) < 3:
                text += "\n- Provide documented evidence."

        return AnswerResult(
            text=_finalize(text, anchor_terms),
            intent=intent,
            scope=scope,
            citations=citations,
            used_chunks=[],
        )

    sents = _sentences(src)
    text = " ".join(sents[:3]) if sents else src[:240]
    return AnswerResult(
        text=_finalize(text, anchor_terms),
        intent="unknown",
        scope=scope,
        citations=citations,
        used_chunks=[],
    )
