from __future__ import annotations

import numpy as np

from src import search as search_module


class _FakeVectorizer:
    def transform(self, items: list[str]) -> np.ndarray:
        # Deterministic non-zero vector so cosine similarity works.
        return np.asarray([[1.0, 0.5]], dtype=float)


def _install_fake_index(monkeypatch, corpus: list[dict]) -> None:
    monkeypatch.setattr(search_module, "_CORPUS", corpus)
    monkeypatch.setattr(search_module, "_VECTORIZER", _FakeVectorizer())
    monkeypatch.setattr(search_module, "_VECTORIZER_CHAR", _FakeVectorizer())
    # Two-dimensional vectors for all chunks.
    x = np.asarray([[1.0, 0.5] for _ in corpus], dtype=float)
    monkeypatch.setattr(search_module, "_X", x)
    monkeypatch.setattr(search_module, "_X_CHAR", x)
    monkeypatch.setattr(search_module, "_SEMANTIC_RERANKER", None)
    monkeypatch.setattr(search_module, "_ENABLE_SEMANTIC_RERANK", False)
    monkeypatch.setattr(search_module, "_ensure_index", lambda: None)


def test_scope_fda_excludes_non_fda_authorities(monkeypatch):
    corpus = [
        {"authority": "FDA", "source": "FDA", "domain": "Validation", "text": "FDA validation guidance", "file": "a.pdf", "page": 1, "chunk_id": "p1_c1"},
        {"authority": "EMA", "source": "EMA", "domain": "Validation", "text": "EMA validation guidance", "file": "b.pdf", "page": 1, "chunk_id": "p1_c1"},
        {"authority": "ICH", "source": "ICH", "domain": "PQS", "text": "ICH Q10 guidance", "file": "c.pdf", "page": 1, "chunk_id": "p1_c1"},
    ]
    _install_fake_index(monkeypatch, corpus)

    out = search_module.search_chunks("validation", scope="FDA", top_k=10)
    assert out
    assert all(str(c.get("authority")) == "FDA" for c in out)


def test_scope_ich_excludes_fda_and_ema(monkeypatch):
    corpus = [
        {"authority": "FDA", "source": "FDA", "domain": "Validation", "text": "FDA content", "file": "a.pdf", "page": 1, "chunk_id": "p1_c1"},
        {"authority": "EMA", "source": "EMA", "domain": "Validation", "text": "EMA content", "file": "b.pdf", "page": 1, "chunk_id": "p1_c1"},
        {"authority": "ICH", "source": "ICH", "domain": "PQS", "text": "ICH content", "file": "c.pdf", "page": 1, "chunk_id": "p1_c1"},
    ]
    _install_fake_index(monkeypatch, corpus)

    out = search_module.search_chunks("quality system", scope="ICH", top_k=10)
    assert out
    assert all(str(c.get("authority")) == "ICH" for c in out)


def test_scope_ema_includes_eu_gmp_and_excludes_fda(monkeypatch):
    corpus = [
        {"authority": "FDA", "source": "FDA", "domain": "Inspection", "text": "FDA inspection", "file": "a.pdf", "page": 1, "chunk_id": "p1_c1"},
        {"authority": "EMA", "source": "EMA", "domain": "Inspection", "text": "EMA inspection", "file": "b.pdf", "page": 1, "chunk_id": "p1_c1"},
        {"authority": "EU_GMP", "source": "EU_GMP", "domain": "Inspection", "text": "EU GMP inspection", "file": "c.pdf", "page": 1, "chunk_id": "p1_c1"},
    ]
    _install_fake_index(monkeypatch, corpus)

    out = search_module.search_chunks("inspection", scope="EMA", top_k=10)
    assert out
    authorities = {str(c.get("authority")) for c in out}
    assert "FDA" not in authorities
    assert authorities <= {"EMA", "EU_GMP", "PIC_S"}
    assert {"EMA", "EU_GMP"} <= authorities
