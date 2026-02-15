from __future__ import annotations

from pathlib import Path

from src.telemetry import QueryLogger, build_report, event_from_payload


def test_query_logger_and_report(tmp_path: Path):
    log_path = tmp_path / "qa_requests.jsonl"
    logger = QueryLogger(log_path)

    payload_ok = {
        "question": "How should computerized systems be validated?",
        "question_hash": "abc123",
        "intent": "procedure_requirements",
        "scope": "MIXED",
        "latency_ms": 123.4,
        "insufficient_evidence": False,
        "confidence": {"overall_confidence": 0.62, "threshold": 0.22},
        "citations": [{"doc_id": "x.pdf", "page": 2, "chunk_id": "p2_c1"}],
        "suggestions": [],
    }
    payload_fail = {
        "question": "How many vacation days do employees get?",
        "question_hash": "def456",
        "intent": "unknown",
        "scope": "MIXED",
        "latency_ms": 77.0,
        "insufficient_evidence": True,
        "confidence": {"overall_confidence": 0.12, "threshold": 0.22},
        "citations": [{"doc_id": "y.pdf", "page": 3, "chunk_id": "p3_c1"}],
        "suggestions": ["What GMP/GxP requirement applies to ...?"],
    }

    logger.write(event_from_payload(payload_ok, source="test"))
    logger.write(event_from_payload(payload_fail, source="test"))

    report = build_report(log_path, hours=24)
    assert report["events_total_window"] == 2
    assert report["insufficient_count"] == 1
    assert report["insufficient_rate"] == 0.5
    assert report["intent_counts"]["procedure_requirements"] == 1
    assert report["intent_counts"]["unknown"] == 1
    assert len(report["recent_failures"]) == 1
    assert "vacation" in str(report["top_low_conf_tokens"]).lower()
