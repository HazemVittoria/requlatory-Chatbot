from __future__ import annotations

from src.specialized_templates import select_specialized_procedure


def test_specialized_risk_management_playbook_selected():
    plan = select_specialized_procedure(
        intent="procedure",
        anchor_terms=["risk management", "ICH Q9"],
        source_text="This section describes risk assessment and controls.",
    )
    assert plan is not None
    assert any("risk management plan" in b.lower() for b in plan.bullets)


def test_specialized_csv_playbook_selected_for_csv_validation():
    plan = select_specialized_procedure(
        intent="procedure_requirements",
        anchor_terms=["computerized systems", "validation", "annex 11"],
        source_text="Audit trail and electronic signatures are expected under Part 11.",
    )
    assert plan is not None
    assert "risk-based lifecycle approach" in plan.intro.lower()


def test_specialized_playbook_not_selected_for_generic_procedure():
    plan = select_specialized_procedure(
        intent="procedure",
        anchor_terms=["deviation", "investigation"],
        source_text="Investigate deviation root cause and document CAPA.",
    )
    assert plan is None
