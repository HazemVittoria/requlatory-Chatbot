from __future__ import annotations

from src.intent_router import route, to_presentation_intent


def test_presentation_intent_mapping_core_intents():
    assert to_presentation_intent("definition", question="what is OOS", anchor_terms=["out of specification"]) == "definition"
    assert to_presentation_intent("procedure_requirements", question="How should CSV be validated?", anchor_terms=["validation"]) == "procedure"
    assert to_presentation_intent("requirements_evidence", question="What evidence is required?", anchor_terms=["evidence"]) == "evidence"
    assert to_presentation_intent("requirements", question="What requirements apply?", anchor_terms=["requirements"]) == "requirements"


def test_presentation_intent_maps_examples_to_inspection_when_needed():
    p = to_presentation_intent(
        "examples_patterns",
        question="What are common inspection findings?",
        anchor_terms=["inspection"],
    )
    assert p == "inspection"


def test_route_exposes_presentation_intent():
    r = route("How should computerized systems be validated?")
    assert r.presentation_intent in {"definition", "requirements", "procedure", "evidence", "inspection"}
