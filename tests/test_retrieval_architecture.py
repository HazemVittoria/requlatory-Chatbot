from __future__ import annotations

from src.templates import Citation, render_answer


def test_templates_use_specialization_policy_entrypoint(monkeypatch):
    called = {"value": False}

    class _Plan:
        intro = "policy intro"
        bullets = ["Step A", "Step B"]
        sentence_confidence = 0.77

    def _fake_policy(**kwargs):
        called["value"] = True
        return _Plan()

    monkeypatch.setattr("src.templates.select_procedure_specialization", _fake_policy)
    res = render_answer(
        intent="procedure",
        scope="MIXED",
        selected_passages=["Procedure text from source."],
        citations=[Citation(doc_id="doc.pdf", page=1, chunk_id="p1_c1")],
        anchor_terms=["validation"],
        question="How should validation be performed?",
        retrieval_scores=[0.8],
        presentation_intent="procedure",
    )
    assert called["value"] is True
    assert "Step A" in res.text
