from __future__ import annotations

import numpy as np

from src.rerank_features import RerankContext, apply_rerank_features
from src.rerank_weights import RerankWeights


def test_anchor_overlap_boost_prefers_chunk_with_anchor_terms():
    corpus = [
        {"text": "General process statement with limited signal."},
        {"text": "CAPA records and deviation investigation outcomes are documented and approved."},
    ]
    base = np.asarray([0.0, 0.0])
    ctx = RerankContext.from_query(
        query="What is expected for CAPA documentation?",
        intent="requirements",
        anchor_terms=["capa", "deviation"],
    )
    scores = apply_rerank_features(base, corpus, ctx, RerankWeights())
    assert scores[1] > scores[0]


def test_structural_penalties_downgrade_noisy_short_chunks():
    corpus = [
        {"text": "2024 TABLE CONTENT"},
        {"text": "Document control procedures should be maintained and reviewed at defined intervals."},
    ]
    base = np.asarray([0.0, 0.0])
    ctx = RerankContext.from_query(query="Document control procedure", intent="procedure", anchor_terms=[])
    scores = apply_rerank_features(base, corpus, ctx, RerankWeights())
    assert scores[1] > scores[0]


def test_csv_keywords_in_query_do_not_change_feature_scoring():
    corpus = [
        {"text": "Validation procedures should be documented and approved."},
        {"text": "Operational controls are reviewed on schedule."},
    ]
    base = np.asarray([0.2, 0.2])
    csv_ctx = RerankContext.from_query(
        query="How to validate computerized systems under csv and annex 11?",
        intent="procedure_requirements",
        anchor_terms=[],
    )
    generic_ctx = RerankContext.from_query(
        query="How to validate systems?",
        intent="procedure_requirements",
        anchor_terms=[],
    )
    csv_scores = apply_rerank_features(base, corpus, csv_ctx, RerankWeights())
    generic_scores = apply_rerank_features(base, corpus, generic_ctx, RerankWeights())
    assert np.allclose(csv_scores, generic_scores)
