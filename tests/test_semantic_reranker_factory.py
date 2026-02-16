from __future__ import annotations

import numpy as np

from src import semantic_reranker as sr


def test_build_semantic_reranker_falls_back_to_lsa_when_embedding_unavailable(monkeypatch):
    monkeypatch.setattr(sr.EmbeddingSemanticReranker, "is_available", staticmethod(lambda: False))
    cfg = sr.SemanticRerankConfig(backend="embedding")
    rr = sr.build_semantic_reranker(cfg)
    assert isinstance(rr, sr.LSASemanticReranker)


def test_build_semantic_reranker_auto_prefers_lsa_when_embedding_missing(monkeypatch):
    monkeypatch.setattr(sr.EmbeddingSemanticReranker, "is_available", staticmethod(lambda: False))
    cfg = sr.SemanticRerankConfig(backend="auto")
    rr = sr.build_semantic_reranker(cfg)
    assert isinstance(rr, sr.LSASemanticReranker)


def test_build_semantic_reranker_auto_prefers_embedding_when_available(monkeypatch):
    monkeypatch.setattr(sr.EmbeddingSemanticReranker, "is_available", staticmethod(lambda: True))
    cfg = sr.SemanticRerankConfig(backend="auto")
    rr = sr.build_semantic_reranker(cfg)
    assert isinstance(rr, sr.EmbeddingSemanticReranker)


def test_embedding_reranker_reads_cached_embeddings_without_model(monkeypatch, tmp_path):
    monkeypatch.setattr(sr.EmbeddingSemanticReranker, "_cache_dir", staticmethod(lambda: tmp_path))
    rr = sr.EmbeddingSemanticReranker(sr.SemanticRerankConfig(backend="embedding", embedding_model="mini"))
    docs = ["doc one", "doc two"]
    key = rr._cache_key(docs)
    np.save(tmp_path / f"{key}.npy", np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))

    monkeypatch.setattr(rr, "_ensure_model", lambda: (_ for _ in ()).throw(AssertionError("model should not load")))
    rr.fit(docs)
    assert rr._doc_emb is not None
    assert rr._doc_emb.shape == (2, 2)
