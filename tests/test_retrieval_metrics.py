from __future__ import annotations

from src.retrieval_metrics import _first_relevant_rank, _required_hits


def test_required_hits_adaptive_strict_mode():
    terms = ["process validation", "evidence", "required"]
    assert _required_hits(terms, min_term_hits=0, strict_multi_term=True) == 2
    assert _required_hits(terms, min_term_hits=0, strict_multi_term=False) == 1
    assert _required_hits(terms, min_term_hits=3, strict_multi_term=True) == 3


def test_first_relevant_rank_uses_hit_threshold():
    chunks = [
        {"text": "This chunk mentions process validation only."},
        {"text": "This chunk mentions process validation and evidence together."},
    ]
    terms = ["process validation", "evidence"]

    rank_strict, hits_strict = _first_relevant_rank(
        chunks,
        terms,
        min_term_hits=0,
        strict_multi_term=True,
    )
    assert rank_strict == 2
    assert hits_strict >= 2

    rank_loose, hits_loose = _first_relevant_rank(
        chunks,
        terms,
        min_term_hits=0,
        strict_multi_term=False,
    )
    assert rank_loose == 1
    assert hits_loose >= 1
