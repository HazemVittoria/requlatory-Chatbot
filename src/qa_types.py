from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Any

Intent = Literal[
    "definition","procedure","difference","requirements","requirements_evidence",
    "scope_trigger_evidence","decision_rule","examples_patterns",
    "mixed_definition_controls","procedure_requirements","unknown",
]
Scope = Literal["ICH","FDA","EMA","SOPS","MIXED"]

@dataclass(frozen=True)
class Citation:
    doc_id: str
    page: int
    chunk_id: str | None = None

@dataclass(frozen=True)
class AnswerResult:
    text: str
    intent: Intent
    scope: Scope
    citations: list[Citation]
    used_chunks: list[dict[str, Any]] | None = None
