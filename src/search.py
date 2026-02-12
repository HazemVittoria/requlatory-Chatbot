# src/search.py
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .ingestion import build_corpus

_CORPUS: list[dict[str, Any]] | None = None
_VECTORIZER: TfidfVectorizer | None = None
_X = None

_WS_RE = re.compile(r"\s+")


def _norm(s: str) -> str:
    return _WS_RE.sub(" ", (s or "")).strip().lower()


def _anchor_hit_count(text_l: str, anchor_terms: list[str]) -> int:
    if not anchor_terms:
        return 0
    terms = {a.strip().lower() for a in anchor_terms if a and a.strip()}
    return sum(1 for t in terms if t in text_l)


def _count_hits(text_l: str, phrases: list[str]) -> int:
    return sum(1 for p in phrases if p in text_l)


def _rebuild_index() -> None:
    global _CORPUS, _VECTORIZER, _X
    _CORPUS = build_corpus(Path("data"))
    texts = [(c.get("text") or "") for c in _CORPUS]
    _VECTORIZER = TfidfVectorizer(stop_words="english")
    _X = _VECTORIZER.fit_transform(texts)


def _ensure_index() -> None:
    if _CORPUS is None or _VECTORIZER is None or _X is None:
        _rebuild_index()


def expand_query(query: str) -> str:
    q = (query or "").strip()
    ql = q.lower()
    extras: list[str] = []
    if "batch" in ql and ("analyses" in ql or "analysis" in ql):
        extras += [
            "batch analysis",
            "analysis of batches",
            "sampling and testing",
            "test results",
            "COA",
            "certificate of analysis",
        ]
    return (q + " " + " ".join(extras)).strip()


def search_chunks(
    query: str,
    scope: str = "MIXED",
    top_k: int = 5,
    anchor_terms: list[str] | None = None,
    intent: str | None = None,
) -> list[dict]:
    _ensure_index()
    assert _CORPUS is not None and _VECTORIZER is not None and _X is not None

    anchor_terms = anchor_terms or []

    # Query-specific guard: "computerized systems" questions must retrieve computerized-related chunks
    ql = (query or "").lower()
    need_computerized = any(
        k in ql for k in ("computerized", "computerised", "computer", "software", "csv", "electronic system")
    )

    q_aug = query + (" " + " ".join(anchor_terms) if anchor_terms else "")
    q2 = expand_query(q_aug)
    qv = _VECTORIZER.transform([q2])
    sims = cosine_similarity(qv, _X).flatten()

    # -------------------------
    # Deterministic rerank boost
    # -------------------------
    boosted = sims.copy()
    intent_l = (intent or "").strip().lower()
    is_definition = "definition" in intent_l
    is_procedure = "procedure" in intent_l

    def_phrases = [" is ", " means ", " defined as ", " refers to ", " definition"]
    proc_phrases = [
        "validation",
        " validated",
        " shall ",
        " should ",
        " procedure",
        " step",
        " record",
        " document",
        " investigation",
    ]

    if is_definition or is_procedure or anchor_terms or need_computerized:
        for i, c in enumerate(_CORPUS):
            t_l = _norm(c.get("text") or "")
            if not t_l:
                continue

            # anchor-term boost (strong signal)
            a_hits = _anchor_hit_count(t_l, anchor_terms)
            if a_hits:
                boosted[i] += 0.12 * a_hits

            # intent-based phrase boost
            if is_definition:
                boosted[i] += 0.06 * _count_hits(t_l, def_phrases)

            if is_procedure:
                boosted[i] += 0.05 * _count_hits(t_l, proc_phrases)

            # Extra boost for CSV / computerized systems validation questions
            if need_computerized:
                csv_terms = (
                    "computerised validation",
                    "computerized validation",
                    "csv",
                    "validation of computerized",
                    "computerised system",
                    "computerized system",
                    "electronic record",
                    "electronic signature",
                    "audit trail",
                    "part 11",
                    "annex 11",
                    "gamp",
                    "data integrity",
                    "access control",
                    "user access",
                    "backup",
                    "restore",
                )
                hits = sum(1 for t in csv_terms if t in t_l)
                if hits:
                    boosted[i] += 0.20 * hits

            # light penalty for very short/header-ish chunks
            if len(t_l) < 180:
                boosted[i] -= 0.03

    idx_all = boosted.argsort()[::-1]

    out: list[dict] = []
    for i in idx_all:
        c = _CORPUS[int(i)]
        src = c.get("source") or c.get("scope")  # tolerate either key
        if scope != "MIXED" and src != scope:
            continue

        if need_computerized:
            tl = _norm(c.get("text") or "")
            if not any(
                k in tl
                for k in (
                    "computer",
                    "computerized",
                    "computerised",
                    "software",
                    "csv",
                    "electronic",
                    "system",
                    "part 11",
                    "audit trail",
                    "annex 11",
                    "gamp",
                    "data integrity",
                )
            ):
                continue

        out.append(c)
        if len(out) >= top_k:
            break

    return out
