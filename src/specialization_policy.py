from __future__ import annotations

from .specialized_templates import SpecializedProcedurePlan, select_specialized_procedure


def select_procedure_specialization(
    *,
    intent: str,
    anchor_terms: list[str],
    source_text: str,
) -> SpecializedProcedurePlan | None:
    """
    Single entry point for procedure-template specialization.
    Retrieval remains generic; domain-specific playbooks are applied only at render time.
    """
    return select_specialized_procedure(
        intent=intent,
        anchor_terms=anchor_terms,
        source_text=source_text,
    )
