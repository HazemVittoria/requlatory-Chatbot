from __future__ import annotations

import numpy as np

from src.rerank_features import RerankContext, apply_rerank_features
from src.rerank_weights import RerankWeights


def test_definition_feature_boost_prefers_definition_like_chunk():
    corpus = [
        {"text": "Data integrity is defined as completeness, consistency, and accuracy of data."},
        {"text": "General background content without explicit definitional language."},
    ]
    base = np.asarray([0.0, 0.0])
    ctx = RerankContext.from_query(
        query="What is data integrity?",
        intent="definition",
        anchor_terms=["data integrity"],
    )
    scores = apply_rerank_features(base, corpus, ctx, RerankWeights())
    assert scores[0] > scores[1]


def test_risk_feature_boost_penalizes_csv_for_non_csv_risk_questions():
    corpus = [
        {"text": "Risk management includes risk assessment, risk control, and risk review."},
        {"text": "Annex 11 and Part 11 require computerized system audit trail controls."},
    ]
    base = np.asarray([0.0, 0.0])
    ctx = RerankContext.from_query(
        query="How should risks be managed?",
        intent="procedure",
        anchor_terms=["risk management"],
    )
    scores = apply_rerank_features(base, corpus, ctx, RerankWeights())
    assert scores[0] > scores[1]


def test_training_requirements_evidence_boost_prefers_training_docs():
    corpus = [
        {"text": "Training records and qualification records should be documented and retained for personnel."},
        {"text": "Changes are reviewed and alternative routines are put in place for system failure."},
        {"text": "Equipment qualification includes IQ OQ and PQ protocols."},
    ]
    base = np.asarray([0.0, 0.0, 0.0])
    ctx = RerankContext.from_query(
        query="What documentation is required for training and qualification?",
        intent="requirements_evidence",
        anchor_terms=["training", "qualification"],
    )
    scores = apply_rerank_features(base, corpus, ctx, RerankWeights())
    assert scores[0] > scores[1]
    assert scores[0] > scores[2]
