# src/qa_engine.py
from __future__ import annotations

from .intent_router import route
from .search import search_chunks
from .templates import render_answer
from .qa_types import AnswerResult, Citation


def answer(question: str) -> AnswerResult:
    # 1) Intent + scope
    r = route(question)

    # 2) Retrieval
    chunks = search_chunks(question, scope=r.scope, top_k=12)

    # Anchor-aware reordering (definition/mixed-definition)
    if r.anchor_terms:
        at = r.anchor_terms[0].lower()
        anchored = [c for c in chunks if at in c["text"].lower()]
        non_anchored = [c for c in chunks if at not in c["text"].lower()]
        chunks = anchored + non_anchored

    # Keep only the top 5 after reordering
    chunks = chunks[:5]


    # 3) Extract text + citations
    passages: list[str] = []
    citations: list[Citation] = []

    for c in chunks:
        passages.append(c["text"])
        citations.append(
            Citation(
                doc_id=c["file"],
                page=c["page"],
                chunk_id=c.get("chunk_id"),
            )
        )



    # 4) Render answer
    return render_answer(
        intent=r.intent,
        scope=r.scope,
        selected_passages=passages,
        citations=citations,
        anchor_terms=r.anchor_terms,
    )
