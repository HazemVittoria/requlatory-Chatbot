from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SpecializedProcedurePlan:
    intro: str
    bullets: list[str]
    sentence_confidence: float = 0.78


def select_specialized_procedure(
    *,
    intent: str,
    anchor_terms: list[str],
    source_text: str,
) -> SpecializedProcedurePlan | None:
    intent_l = (intent or "").strip().lower()
    anchors_l = " ".join(anchor_terms or []).lower()
    src_l = (source_text or "").lower()

    if intent_l == "procedure" and "risk management" in anchors_l:
        return SpecializedProcedurePlan(
            intro="",
            bullets=[
                "Define a quality risk management plan, including the risk question, scope, and quality objective before starting the assessment.",
                "Identify hazards and potential harms using relevant process, product, and patient-safety information.",
                "Analyze each risk by assessing severity, probability of occurrence, and detectability where applicable.",
                "Evaluate risks against predefined acceptance criteria to prioritize action.",
                "Implement risk controls to reduce risk to an acceptable level, proportional to risk significance.",
                "Document rationale, decisions, and residual risk acceptance with responsible approvers.",
                "Communicate outcomes to stakeholders and integrate controls into procedures and training.",
                "Perform periodic risk review and update the assessment when changes, deviations, or new knowledge is identified.",
            ],
            sentence_confidence=0.75,
        )

    csv_signals = (
        "computerized",
        "computerised",
        "csv",
        "part 11",
        "annex 11",
        "electronic record",
        "electronic signature",
        "audit trail",
        "gamp",
    )
    csv_question = (
        intent_l == "procedure_requirements"
        and (
            "computerized" in anchors_l
            or "annex 11" in anchors_l
            or "part 11" in anchors_l
            or "csv" in anchors_l
        )
        and "validation" in anchors_l
    )
    if csv_question and any(s in src_l for s in csv_signals):
        return SpecializedProcedurePlan(
            intro=(
                "Computerized systems should be validated using a risk-based lifecycle approach, "
                "supported by documented evidence."
            ),
            bullets=[
                "Define intended use and GxP impact (scope the computerized system).",
                "Perform and document a risk assessment to determine validation depth and controls.",
                "Define requirements (URS) and ensure traceability to tests.",
                "Assess/supervise suppliers and clarify responsibilities.",
                "Execute documented testing (as applicable: IQ/OQ/PQ or verification against requirements).",
                "Ensure data integrity controls (access control, audit trails, security, backup/restore, record retention).",
                "Control changes (impact assessment, change control, regression testing/revalidation as needed).",
                "Review/approve results and maintain evidence (reports, deviations, approvals).",
                "Perform periodic review to confirm continued validated state and security posture.",
                "Manage retirement/decommissioning with data retention and migration controls.",
            ],
            sentence_confidence=0.78,
        )

    return None
