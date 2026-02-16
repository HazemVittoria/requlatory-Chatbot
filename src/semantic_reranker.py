from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer


@dataclass(frozen=True)
class SemanticRerankConfig:
    backend: str = "auto"  # auto | lsa | embedding
    n_components: int = 128
    min_df: int = 1
    max_features: int = 50000
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_batch_size: int = 64


class LSASemanticReranker:
    """
    Lightweight semantic reranker using LSA embeddings over corpus text.
    Works offline and is deterministic for reproducible retrieval behavior.
    """

    def __init__(self, config: SemanticRerankConfig | None = None):
        self.config = config or SemanticRerankConfig()
        self.backend_name = "lsa"
        self._vectorizer: TfidfVectorizer | None = None
        self._svd: TruncatedSVD | None = None
        self._normalizer: Normalizer | None = None
        self._doc_emb: np.ndarray | None = None

    def fit(self, texts: Sequence[str]) -> None:
        docs = [t or "" for t in texts]
        if not docs:
            self._vectorizer = None
            self._svd = None
            self._normalizer = None
            self._doc_emb = None
            return

        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=self.config.min_df,
            max_features=self.config.max_features,
        )
        x = vectorizer.fit_transform(docs)
        n_docs, n_feats = x.shape
        if n_docs < 2 or n_feats < 2:
            self._vectorizer = vectorizer
            self._svd = None
            self._normalizer = None
            self._doc_emb = None
            return

        n_comp = min(self.config.n_components, n_docs - 1, n_feats - 1)
        if n_comp < 2:
            self._vectorizer = vectorizer
            self._svd = None
            self._normalizer = None
            self._doc_emb = None
            return

        svd = TruncatedSVD(n_components=n_comp, random_state=0)
        normalizer = Normalizer(copy=False)
        doc_emb = svd.fit_transform(x)
        doc_emb = normalizer.fit_transform(doc_emb)

        self._vectorizer = vectorizer
        self._svd = svd
        self._normalizer = normalizer
        self._doc_emb = doc_emb

    def score_query(self, query: str, doc_indices: Sequence[int] | None = None) -> np.ndarray:
        if (
            self._vectorizer is None
            or self._svd is None
            or self._normalizer is None
            or self._doc_emb is None
        ):
            if doc_indices is None:
                return np.asarray([])
            return np.zeros(len(doc_indices), dtype=float)

        qv = self._vectorizer.transform([query or ""])
        q_emb = self._svd.transform(qv)
        q_emb = self._normalizer.transform(q_emb)
        q = q_emb[0]

        if doc_indices is None:
            docs = self._doc_emb
        else:
            docs = self._doc_emb[np.asarray(list(doc_indices), dtype=int)]

        sims = docs @ q
        # convert cosine-ish range [-1,1] to [0,1]
        sims = (sims + 1.0) / 2.0
        return np.asarray(sims, dtype=float)


def _import_sentence_transformers() -> Any:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer


class EmbeddingSemanticReranker:
    """
    Optional model-based semantic reranker.
    Uses sentence-transformers when available.
    """

    def __init__(self, config: SemanticRerankConfig | None = None):
        self.config = config or SemanticRerankConfig()
        self.backend_name = "embedding"
        self._model: Any | None = None
        self._doc_emb: np.ndarray | None = None

    @staticmethod
    def _cache_dir() -> Path:
        return Path(".cache") / "semantic_embeddings"

    def _cache_key(self, docs: Sequence[str]) -> str:
        h = hashlib.sha256()
        h.update(self.config.embedding_model.encode("utf-8", errors="ignore"))
        h.update(b"|")
        h.update(str(len(docs)).encode("ascii", errors="ignore"))
        for d in docs:
            b = (d or "").encode("utf-8", errors="ignore")
            h.update(b"|")
            h.update(str(len(b)).encode("ascii", errors="ignore"))
            h.update(b[:2048])
        return h.hexdigest()[:24]

    def _cache_path(self, key: str) -> Path:
        return self._cache_dir() / f"{key}.npy"

    def _load_cache(self, key: str, expected_rows: int) -> np.ndarray | None:
        path = self._cache_path(key)
        if not path.exists():
            return None
        try:
            arr = np.load(path)
        except Exception:
            return None
        if not isinstance(arr, np.ndarray) or arr.ndim != 2 or arr.shape[0] != expected_rows:
            return None
        return np.asarray(arr, dtype=np.float32)

    def _save_cache(self, key: str, emb: np.ndarray) -> None:
        path = self._cache_path(key)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, np.asarray(emb, dtype=np.float32))
        except Exception:
            # Cache is opportunistic; retrieval should still work without it.
            return

    @staticmethod
    def is_available() -> bool:
        try:
            _import_sentence_transformers()
            return True
        except Exception:
            return False

    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model
        SentenceTransformer = _import_sentence_transformers()
        self._model = SentenceTransformer(self.config.embedding_model)
        return self._model

    @staticmethod
    def _l2_normalize(x: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        return x / norms

    def fit(self, texts: Sequence[str]) -> None:
        docs = [t or "" for t in texts]
        if not docs:
            self._doc_emb = None
            return
        cache_key = self._cache_key(docs)
        cached = self._load_cache(cache_key, expected_rows=len(docs))
        if cached is not None:
            self._doc_emb = self._l2_normalize(cached)
            return
        model = self._ensure_model()
        emb = model.encode(
            docs,
            batch_size=max(8, int(self.config.embedding_batch_size)),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        arr = np.asarray(emb, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        self._doc_emb = self._l2_normalize(arr)
        self._save_cache(cache_key, self._doc_emb)

    def score_query(self, query: str, doc_indices: Sequence[int] | None = None) -> np.ndarray:
        if self._doc_emb is None:
            if doc_indices is None:
                return np.asarray([])
            return np.zeros(len(doc_indices), dtype=float)

        model = self._ensure_model()
        q_emb = model.encode(
            [query or ""],
            batch_size=1,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        q = np.asarray(q_emb, dtype=np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        q = self._l2_normalize(q)[0]

        if doc_indices is None:
            docs = self._doc_emb
        else:
            docs = self._doc_emb[np.asarray(list(doc_indices), dtype=int)]

        sims = docs @ q
        sims = (sims + 1.0) / 2.0
        return np.asarray(sims, dtype=float)


def build_semantic_reranker(config: SemanticRerankConfig | None = None) -> LSASemanticReranker | EmbeddingSemanticReranker:
    cfg = config or SemanticRerankConfig()
    backend = (cfg.backend or "auto").strip().lower()

    if backend == "embedding":
        if EmbeddingSemanticReranker.is_available():
            return EmbeddingSemanticReranker(cfg)
        return LSASemanticReranker(cfg)

    if backend == "lsa":
        return LSASemanticReranker(cfg)

    # auto mode: prefer embedding backend when available, else deterministic LSA.
    if EmbeddingSemanticReranker.is_available():
        return EmbeddingSemanticReranker(cfg)
    return LSASemanticReranker(cfg)
