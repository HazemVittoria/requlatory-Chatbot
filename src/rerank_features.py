from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from .rerank_weights import RerankWeights

_WS_RE = re.compile(r"\s+")

_DEF_PHRASES = [" is ", " means ", " defined as ", " refers to ", " definition"]
_PROC_PHRASES = [
    " shall ",
    " should ",
    " procedure",
    " step",
    " record",
    " document",
    " investigation",
    " validation",
    " qualified",
]
_RISK_TERMS = (
    "risk management",
    "risk assessment",
    "risk control",
    "risk review",
    "identify hazards",
    "evaluate risks",
    "reduce risks",
    "accept risks",
)
_CSV_TERMS = (
    "computerised",
    "computerized",
    "part 11",
    "annex 11",
    "electronic signature",
    "audit trail",
    "gamp",
)
_TRAINING_COMP_TERMS = (
    "computerised",
    "computerized",
    "edp",
    "software",
    "audit trail",
    "data integrity",
    "part 11",
    "annex 11",
    "electronic record",
    "electronic signature",
)
_TRAINING_DOC_TERMS = (
    "record",
    "records",
    "document",
    "documentation",
    "evidence",
    "maintained",
    "retained",
    "stored",
    "training record",
    "training records",
    "qualification record",
    "qualification records",
    "personnel",
    "staff",
    "competent",
    "competence",
    "education",
    "experience",
)
_TRAINING_EVIDENCE_TERMS = (
    "training records",
    "training record",
    "qualification records",
    "qualification record",
    "training measures",
    "qualifications should be documented",
    "documented and stored",
    "stored as part of the life cycle documentation",
    "education and experience",
    "staff qualifications",
    "competence",
    "competent",
)
_TRAINING_OFF_TOPIC_TERMS = (
    "changes are reviewed",
    "alternative routines",
    "system failure",
    "source code",
    "business failure of the supplier",
    "inspection considerations",
    "define is the system defined",
)
_TRAINING_EQ_TERMS = (
    "iq",
    "oq",
    "pq",
    "installation qualification",
    "operational qualification",
    "performance qualification",
    "equipment qualification",
    "equipment",
)


def _norm(s: str) -> str:
    return _WS_RE.sub(" ", (s or "")).strip().lower()


def _anchor_hit_count(text_l: str, anchor_terms: list[str]) -> int:
    if not anchor_terms:
        return 0
    terms = {a.strip().lower() for a in anchor_terms if a and a.strip()}
    return sum(1 for t in terms if t in text_l)


def _count_hits(text_l: str, phrases: list[str] | tuple[str, ...]) -> int:
    return sum(1 for p in phrases if p in text_l)


@dataclass(frozen=True)
class RerankContext:
    query_l: str
    intent_l: str
    anchor_terms: list[str]
    need_computerized: bool
    need_doc_evidence: bool
    need_qualification: bool
    need_risk_management: bool

    @classmethod
    def from_query(cls, query: str, intent: str | None, anchor_terms: list[str] | None) -> "RerankContext":
        q_l = (query or "").lower()
        intent_l = (intent or "").strip().lower()
        anchors = anchor_terms or []
        return cls(
            query_l=q_l,
            intent_l=intent_l,
            anchor_terms=anchors,
            need_computerized=any(
                k in q_l for k in ("computerized", "computerised", "computer", "software", "csv", "part 11", "annex 11")
            ),
            need_doc_evidence=any(k in q_l for k in ("documentation", "document", "record", "records", "evidence")),
            need_qualification="qualification" in q_l,
            need_risk_management=any(k in q_l for k in ("risk", "risks", "risk management")),
        )


def combine_hybrid_scores(
    sims_word: np.ndarray,
    sims_char: np.ndarray,
    overlap_scores: np.ndarray,
    weights: RerankWeights,
) -> np.ndarray:
    return (
        (weights.hybrid_word * sims_word)
        + (weights.hybrid_char * sims_char)
        + (weights.hybrid_overlap * overlap_scores)
    )


def apply_rerank_features(
    base_scores: np.ndarray,
    corpus: list[dict[str, Any]],
    ctx: RerankContext,
    weights: RerankWeights,
) -> np.ndarray:
    boosted = base_scores.copy()

    for i, c in enumerate(corpus):
        t_l = _norm(c.get("text") or "")
        if not t_l:
            continue
        raw_text = (c.get("text") or "").strip()

        if re.match(r"^\d{2,4}\s+[A-Z]", raw_text):
            boosted[i] -= weights.structure_num_prefix_penalty
        if re.match(r"^[a-z].*\)\.", raw_text[:30]):
            boosted[i] -= weights.structure_mid_sentence_penalty
        if re.search(r"\bpage\s+\d+\s+of\s+\d+\b", t_l):
            boosted[i] -= weights.structure_page_marker_penalty
        if len(t_l) < 180:
            boosted[i] -= weights.structure_short_chunk_penalty

        a_hits = _anchor_hit_count(t_l, ctx.anchor_terms)
        if a_hits:
            boosted[i] += weights.anchor_hit_boost * a_hits

        if "definition" in ctx.intent_l:
            boosted[i] += weights.definition_phrase_boost * _count_hits(t_l, _DEF_PHRASES)
        if "procedure" in ctx.intent_l:
            boosted[i] += weights.procedure_phrase_boost * _count_hits(t_l, _PROC_PHRASES)

        if ctx.need_risk_management:
            risk_hits = _count_hits(t_l, _RISK_TERMS)
            if risk_hits:
                boosted[i] += weights.risk_phrase_boost * risk_hits
            if not ctx.need_computerized:
                csv_hits = _count_hits(t_l, _CSV_TERMS)
                if csv_hits:
                    boosted[i] -= weights.risk_csv_penalty * csv_hits

        if ctx.intent_l == "requirements_evidence" and "training" in ctx.query_l:
            if not ctx.need_computerized:
                comp_hits = _count_hits(t_l, _TRAINING_COMP_TERMS)
                if comp_hits:
                    boosted[i] -= weights.training_computerized_penalty * comp_hits

            doc_hits = _count_hits(t_l, _TRAINING_DOC_TERMS)
            if doc_hits:
                boosted[i] += weights.training_doc_term_boost * doc_hits
            else:
                boosted[i] -= weights.training_missing_doc_penalty

            evidence_hits = _count_hits(t_l, _TRAINING_EVIDENCE_TERMS)
            if evidence_hits:
                boosted[i] += weights.training_evidence_term_boost * evidence_hits

            has_training = any(term in t_l for term in ("training", "trained"))
            has_qual = any(term in t_l for term in ("qualification", "qualified", "competence", "personnel", "staff"))
            has_docs = any(term in t_l for term in ("record", "records", "document", "documentation", "stored", "retained"))
            coverage = int(has_training) + int(has_qual) + int(has_docs)
            if coverage >= 2:
                boosted[i] += weights.training_coverage_boost * coverage
            if coverage == 3:
                boosted[i] += weights.training_full_coverage_bonus

            if ctx.need_doc_evidence or ctx.need_qualification:
                off_hits = _count_hits(t_l, _TRAINING_OFF_TOPIC_TERMS)
                if off_hits:
                    boosted[i] -= weights.training_off_topic_penalty * off_hits

            eq_hits = _count_hits(t_l, _TRAINING_EQ_TERMS)
            if eq_hits:
                boosted[i] -= weights.training_equipment_penalty * eq_hits

            if "histogram" in t_l or "pareto" in t_l or "process capability" in t_l:
                boosted[i] -= weights.training_q9_noise_penalty
            if "contains nonbinding recommendations" in t_l:
                boosted[i] -= weights.training_nonbinding_penalty

    return boosted
