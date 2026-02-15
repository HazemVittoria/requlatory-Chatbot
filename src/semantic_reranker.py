from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer


@dataclass(frozen=True)
class SemanticRerankConfig:
    n_components: int = 128
    min_df: int = 1
    max_features: int = 50000


class LSASemanticReranker:
    """
    Lightweight semantic reranker using LSA embeddings over corpus text.
    Works offline and is deterministic for reproducible retrieval behavior.
    """

    def __init__(self, config: SemanticRerankConfig | None = None):
        self.config = config or SemanticRerankConfig()
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
