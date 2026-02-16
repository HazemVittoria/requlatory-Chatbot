from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from .rerank_weights import RerankWeights

_WS_RE = re.compile(r"\s+")

_DEF_PHRASES = [" is ", " means ", " defined as ", " refers to ", " definition"]
_PROC_PHRASES = [
    " procedure",
    " step",
    " review",
    " control",
    " investigation",
    " validation",
    " qualification",
]

_ACTION_PHRASES = [
    " shall ",
    " should ",
    " must ",
    " required ",
    " requires ",
    " procedure",
    " record",
    " document",
]
_NOISE_TERMS = (
    "all rights reserved",
    "table of contents",
    "copyright",
    "contains nonbinding recommendations",
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

    @classmethod
    def from_query(cls, query: str, intent: str | None, anchor_terms: list[str] | None) -> "RerankContext":
        anchors = anchor_terms or []
        return cls(
            query_l=(query or "").lower(),
            intent_l=(intent or "").strip().lower(),
            anchor_terms=anchors,
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
        if "procedure" in ctx.intent_l or "requirements" in ctx.intent_l:
            boosted[i] += weights.procedure_phrase_boost * _count_hits(t_l, _PROC_PHRASES)

        action_hits = _count_hits(t_l, _ACTION_PHRASES)
        if action_hits:
            boosted[i] += weights.lexical_action_phrase_boost * action_hits

        noise_hits = _count_hits(t_l, _NOISE_TERMS)
        if noise_hits:
            boosted[i] -= weights.lexical_noise_penalty * noise_hits

    return boosted
