# src/search.py
from __future__ import annotations

from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .ingestion import build_corpus

_CORPUS: list[dict[str, Any]] | None = None
_VECTORIZER: TfidfVectorizer | None = None
_X = None


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


def search_chunks(query: str, scope: str = "MIXED", top_k: int = 5) -> list[dict]:
    _ensure_index()
    assert _CORPUS is not None and _VECTORIZER is not None and _X is not None

    q2 = expand_query(query)
    qv = _VECTORIZER.transform([q2])
    sims = cosine_similarity(qv, _X).flatten()

    idx_all = sims.argsort()[::-1]

    out: list[dict] = []
    for i in idx_all:
        c = _CORPUS[int(i)]
        src = c.get("source") or c.get("scope")  # tolerate either key
        if scope != "MIXED" and src != scope:
            continue
        out.append(c)
        if len(out) >= top_k:
            break

    return out
