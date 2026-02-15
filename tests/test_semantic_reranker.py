from __future__ import annotations

import numpy as np

from src.semantic_reranker import LSASemanticReranker


def test_lsa_semantic_reranker_prefers_related_text():
    texts = [
        "Process validation requires documented evidence of control strategy.",
        "Training records should be retained for personnel qualification.",
        "Computerized systems require audit trails and access controls.",
    ]
    reranker = LSASemanticReranker()
    reranker.fit(texts)

    sims = reranker.score_query("When is process validation required?")
    assert isinstance(sims, np.ndarray)
    assert len(sims) == 3
    assert sims[0] >= sims[1]
    assert sims[0] >= sims[2]


def test_lsa_semantic_reranker_subset_indices():
    texts = [
        "Risk management includes risk assessment and risk review.",
        "Supplier qualification requires oversight.",
        "Deviations should be investigated and documented.",
    ]
    reranker = LSASemanticReranker()
    reranker.fit(texts)

    sims = reranker.score_query("How should deviations be investigated?", [1, 2])
    assert len(sims) == 2
    assert sims[1] >= sims[0]
