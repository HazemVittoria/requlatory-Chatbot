from __future__ import annotations

from src.web_api import _question_hash, build_answer_payload


def test_question_hash_is_stable():
    q = "How should OOS/OOT results be investigated?"
    assert _question_hash(q) == _question_hash(q)
    assert len(_question_hash(q)) == 12


def test_build_answer_payload_shape():
    payload = build_answer_payload("How should computerized systems be validated?")
    assert payload["question"]
    assert isinstance(payload["question_hash"], str)
    assert isinstance(payload["answer"], str)
    assert isinstance(payload["insufficient_evidence"], bool)
    assert isinstance(payload["intent"], str)
    assert isinstance(payload["scope"], str)
    assert isinstance(payload["citations"], list)
    assert isinstance(payload["suggestions"], list)
    assert isinstance(payload["confidence"], dict)
    assert payload["latency_ms"] >= 0
    for key in ("retrieval_confidence", "sentence_confidence", "domain_relevance", "overall_confidence", "threshold"):
        assert key in payload["confidence"]


def test_build_answer_payload_insufficient_for_offtopic():
    payload = build_answer_payload("How many vacation days do employees get?")
    assert payload["insufficient_evidence"] is True
    assert "insufficient evidence" in payload["answer"].lower()
    assert len(payload["suggestions"]) >= 1
