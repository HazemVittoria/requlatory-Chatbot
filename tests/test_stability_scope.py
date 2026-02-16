from __future__ import annotations

from src.qa_engine import answer


def test_fda_stability_query_excludes_who_citations():
    res = answer("What are FDA stability testing requirements?")
    assert res.scope == "FDA"
    cits = res.citations or []
    assert cits, "expected at least one citation for scope-filter assertion"
    assert all("who" not in (c.doc_id or "").lower() for c in cits), "FDA-scope query should not cite WHO docs"


def test_mixed_stability_query_can_retrieve_who_stability_doc():
    res = answer("What are stability testing requirements?")
    cits = res.citations or []
    assert cits, "expected citations for stability requirements query"
    assert any(
        "stability-testing" in (c.doc_id or "").lower() or "who" in (c.doc_id or "").lower()
        for c in cits
    ), "MIXED-scope stability query should include WHO stability guidance when relevant"
