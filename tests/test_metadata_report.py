from __future__ import annotations

from src.metadata_report import _build_summary


def test_build_summary_counts_other_docs():
    corpus = [
        {"file": "a.pdf", "authority": "FDA", "domain": "Validation"},
        {"file": "a.pdf", "authority": "FDA", "domain": "Validation"},
        {"file": "b.pdf", "authority": "EMA", "domain": "Other"},
        {"file": "c.pdf", "authority": "WHO", "domain": "Other"},
    ]
    out = _build_summary(corpus, top=10)
    assert out["chunks_total"] == 4
    assert out["docs_total"] == 3
    assert out["domain_counts"]["Validation"] == 2
    assert out["domain_counts"]["Other"] == 2
    assert out["other_docs_total"] == 2
    assert len(out["top_other_docs"]) == 2
