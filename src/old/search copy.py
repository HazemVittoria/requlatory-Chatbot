from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def expand_query(query: str) -> str:
    q = query.strip()
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
    return q + " " + " ".join(extras)


def search(corpus: list[dict], query: str, top_k: int = 30) -> list[tuple[float, dict]]:
    texts = [c["text"] for c in corpus]
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(texts)

    q2 = expand_query(query)
    qv = vectorizer.transform([q2])
    sims = cosine_similarity(qv, X).flatten()

    idx = sims.argsort()[::-1][:top_k]
    return [(float(sims[i]), corpus[i]) for i in idx]
