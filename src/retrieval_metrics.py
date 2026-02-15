from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .intent_router import route
from .search import search_chunks
from . import search as search_module

@dataclass(frozen=True)
class RetrievalCase:
    id: str
    dataset: str
    question: str
    scope: str
    intent: str
    anchor_terms: list[str]
    relevance_terms: list[str]
    enabled: bool = True


def _norm(s: str) -> str:
    return " ".join((s or "").lower().split())


def _clean_term(term: str) -> str:
    t = _norm(term)
    t = t.replace("_", " ").strip(" ,.;:!?")
    return t


def _extract_terms(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        t = _clean_term(raw)
        if not t or t in {"insufficient evidence"}:
            continue
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def load_golden_cases(path: Path) -> list[RetrievalCase]:
    rows = _load_jsonl(path)
    cases: list[RetrievalCase] = []
    for obj in rows:
        q = str(obj["question"])
        anchor_terms = list(obj.get("anchor_terms", []))
        must_include = list(obj.get("must_include", []))
        relevance_terms = _extract_terms(must_include + anchor_terms)
        if not relevance_terms:
            # fallback to route-derived anchors
            r = route(q)
            relevance_terms = _extract_terms(list(r.anchor_terms))

        cases.append(
            RetrievalCase(
                id=str(obj["id"]),
                dataset="golden",
                question=q,
                scope=str(obj.get("scope", "MIXED")),
                intent=str(obj.get("intent", "unknown")),
                anchor_terms=anchor_terms,
                relevance_terms=relevance_terms,
                enabled=bool(relevance_terms),
            )
        )
    return cases


def load_stress_cases(path: Path) -> list[RetrievalCase]:
    rows = _load_jsonl(path)
    cases: list[RetrievalCase] = []
    for obj in rows:
        if str(obj.get("expected_mode", "answer")) != "answer":
            continue

        q = str(obj["question"])
        r = route(q)
        anchor_terms = list(r.anchor_terms)
        must_include = list(obj.get("must_include", []))
        relevance_terms = _extract_terms(must_include + anchor_terms)

        cases.append(
            RetrievalCase(
                id=str(obj["id"]),
                dataset="stress",
                question=q,
                scope="MIXED",
                intent=str(r.intent),
                anchor_terms=anchor_terms,
                relevance_terms=relevance_terms,
                enabled=bool(relevance_terms),
            )
        )
    return cases


def _matched_term_count(text: str, terms: list[str]) -> int:
    t = _norm(text)
    if not t or not terms:
        return 0
    return sum(1 for term in terms if term in t)


def _required_hits(terms: list[str], min_term_hits: int, strict_multi_term: bool) -> int:
    if min_term_hits > 0:
        return min(max(1, min_term_hits), max(1, len(terms)))
    if strict_multi_term and len(terms) >= 2:
        return 2
    return 1


def _first_relevant_rank(
    chunks: list[dict],
    terms: list[str],
    *,
    min_term_hits: int,
    strict_multi_term: bool,
) -> tuple[int | None, int]:
    req = _required_hits(terms, min_term_hits=min_term_hits, strict_multi_term=strict_multi_term)
    best_hits = 0
    for i, c in enumerate(chunks, start=1):
        hits = _matched_term_count(str(c.get("text", "")), terms)
        if hits > best_hits:
            best_hits = hits
        if hits >= req:
            return i, hits
    return None, best_hits


def evaluate_cases(
    cases: list[RetrievalCase],
    k_values: list[int],
    max_k: int,
    *,
    min_term_hits: int = 0,
    strict_multi_term: bool = True,
) -> dict[str, Any]:
    enabled_cases = [c for c in cases if c.enabled]
    per_case: list[dict[str, Any]] = []

    hits_at_k = {k: 0 for k in k_values}
    mrr_total = 0.0

    by_dataset: dict[str, dict[str, Any]] = {}
    by_intent: dict[str, dict[str, Any]] = {}

    for c in enabled_cases:
        chunks = search_chunks(
            query=c.question,
            scope=c.scope,
            top_k=max_k,
            anchor_terms=c.anchor_terms,
            intent=c.intent,
        )
        rank, matched_hits = _first_relevant_rank(
            chunks,
            c.relevance_terms,
            min_term_hits=min_term_hits,
            strict_multi_term=strict_multi_term,
        )
        rr = 0.0 if rank is None else 1.0 / rank
        mrr_total += rr
        for k in k_values:
            if rank is not None and rank <= k:
                hits_at_k[k] += 1

        row = {
            "id": c.id,
            "dataset": c.dataset,
            "intent": c.intent,
            "rank_first_relevant": rank,
            "matched_term_hits": matched_hits,
            "required_term_hits": _required_hits(
                c.relevance_terms,
                min_term_hits=min_term_hits,
                strict_multi_term=strict_multi_term,
            ),
            "reciprocal_rank": round(rr, 6),
        }
        per_case.append(row)

        if c.dataset not in by_dataset:
            by_dataset[c.dataset] = {"count": 0, "mrr_sum": 0.0, "hits": {k: 0 for k in k_values}}
        d = by_dataset[c.dataset]
        d["count"] += 1
        d["mrr_sum"] += rr
        for k in k_values:
            if rank is not None and rank <= k:
                d["hits"][k] += 1

        if c.intent not in by_intent:
            by_intent[c.intent] = {"count": 0, "mrr_sum": 0.0, "hits": {k: 0 for k in k_values}}
        it = by_intent[c.intent]
        it["count"] += 1
        it["mrr_sum"] += rr
        for k in k_values:
            if rank is not None and rank <= k:
                it["hits"][k] += 1

    n = len(enabled_cases)
    summary = {
        "count": n,
        "mrr": round((mrr_total / n) if n else 0.0, 6),
        "recall_at_k": {
            str(k): round((hits_at_k[k] / n) if n else 0.0, 6)
            for k in k_values
        },
    }

    for d in by_dataset.values():
        count = d["count"]
        d["mrr"] = round((d["mrr_sum"] / count) if count else 0.0, 6)
        d["recall_at_k"] = {
            str(k): round((d["hits"][k] / count) if count else 0.0, 6)
            for k in k_values
        }
        del d["mrr_sum"]
        del d["hits"]

    for it in by_intent.values():
        count = it["count"]
        it["mrr"] = round((it["mrr_sum"] / count) if count else 0.0, 6)
        it["recall_at_k"] = {
            str(k): round((it["hits"][k] / count) if count else 0.0, 6)
            for k in k_values
        }
        del it["mrr_sum"]
        del it["hits"]

    misses = [r for r in per_case if r["rank_first_relevant"] is None]
    return {
        "matching": {
            "strict_multi_term": strict_multi_term,
            "min_term_hits": min_term_hits,
            "adaptive_rule": "requires 2 hits for multi-term cases when strict_multi_term=true and min_term_hits=0",
        },
        "summary": summary,
        "by_dataset": by_dataset,
        "by_intent": by_intent,
        "misses": misses[:50],
        "cases_evaluated": per_case,
    }


@contextmanager
def _semantic_mode(enabled: bool, weight: float, top_n: int):
    prev_enabled = getattr(search_module, "_ENABLE_SEMANTIC_RERANK", False)
    prev_weight = getattr(search_module, "_SEMANTIC_RERANK_WEIGHT", 0.18)
    prev_top_n = getattr(search_module, "_SEMANTIC_RERANK_TOP_N", 30)
    search_module._ENABLE_SEMANTIC_RERANK = bool(enabled)
    search_module._SEMANTIC_RERANK_WEIGHT = float(weight)
    search_module._SEMANTIC_RERANK_TOP_N = int(top_n)
    try:
        yield
    finally:
        search_module._ENABLE_SEMANTIC_RERANK = prev_enabled
        search_module._SEMANTIC_RERANK_WEIGHT = prev_weight
        search_module._SEMANTIC_RERANK_TOP_N = prev_top_n


def _to_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def _diff_summary(base: dict[str, Any], variant: dict[str, Any]) -> dict[str, Any]:
    bsum = base.get("summary", {})
    vsum = variant.get("summary", {})
    out: dict[str, Any] = {
        "mrr_delta": round(_to_float(vsum.get("mrr")) - _to_float(bsum.get("mrr")), 6),
        "recall_at_k_delta": {},
        "rank_changes": {"improved": 0, "worsened": 0, "unchanged": 0},
    }

    b_recall = bsum.get("recall_at_k", {}) or {}
    v_recall = vsum.get("recall_at_k", {}) or {}
    for k in sorted(set(b_recall.keys()) | set(v_recall.keys()), key=lambda s: int(s)):
        out["recall_at_k_delta"][k] = round(_to_float(v_recall.get(k)) - _to_float(b_recall.get(k)), 6)

    base_cases = {r["id"]: r for r in (base.get("cases_evaluated") or []) if isinstance(r, dict)}
    var_cases = {r["id"]: r for r in (variant.get("cases_evaluated") or []) if isinstance(r, dict)}
    for cid in sorted(set(base_cases.keys()) & set(var_cases.keys())):
        br = base_cases[cid].get("rank_first_relevant")
        vr = var_cases[cid].get("rank_first_relevant")
        if br is None and vr is None:
            out["rank_changes"]["unchanged"] += 1
            continue
        if br is None and vr is not None:
            out["rank_changes"]["improved"] += 1
            continue
        if br is not None and vr is None:
            out["rank_changes"]["worsened"] += 1
            continue
        if int(vr) < int(br):
            out["rank_changes"]["improved"] += 1
        elif int(vr) > int(br):
            out["rank_changes"]["worsened"] += 1
        else:
            out["rank_changes"]["unchanged"] += 1
    return out


def main() -> int:
    p = argparse.ArgumentParser(prog="python -m src.retrieval_metrics")
    p.add_argument("--golden", default="golden_set.jsonl", help="Path to golden_set jsonl.")
    p.add_argument("--stress", default="tests/stress_set.jsonl", help="Path to stress_set jsonl.")
    p.add_argument("--k", default="1,3,5", help="Comma-separated k values (e.g. 1,3,5).")
    p.add_argument("--max-k", type=int, default=10, help="Max retrieved chunks per query for scoring.")
    p.add_argument(
        "--min-term-hits",
        type=int,
        default=0,
        help="Minimum relevance-term hits required in one chunk (0 = adaptive strict mode).",
    )
    p.add_argument(
        "--no-strict-multi-term",
        action="store_true",
        help="Disable strict multi-term rule (default strict: require 2 hits when a case has >=2 terms).",
    )
    p.add_argument(
        "--compare-semantic",
        action="store_true",
        help="Run baseline (semantic OFF) and variant (semantic ON) in one command and report deltas.",
    )
    p.add_argument(
        "--semantic-weight",
        type=float,
        default=0.18,
        help="Semantic rerank weight for --compare-semantic variant run.",
    )
    p.add_argument(
        "--semantic-top-n",
        type=int,
        default=30,
        help="Semantic rerank top-N candidates for --compare-semantic variant run.",
    )
    p.add_argument("--json", action="store_true", help="Emit full JSON (default).")
    args = p.parse_args()

    k_values = sorted({int(x.strip()) for x in args.k.split(",") if x.strip()})
    if not k_values:
        k_values = [1, 3, 5]
    max_k = max(args.max_k, max(k_values))

    golden_cases = load_golden_cases(Path(args.golden))
    stress_cases = load_stress_cases(Path(args.stress))
    all_cases = golden_cases + stress_cases

    eval_kwargs = {
        "k_values": k_values,
        "max_k": max_k,
        "min_term_hits": max(0, args.min_term_hits),
        "strict_multi_term": not args.no_strict_multi_term,
    }

    if args.compare_semantic:
        with _semantic_mode(enabled=False, weight=args.semantic_weight, top_n=args.semantic_top_n):
            baseline = evaluate_cases(all_cases, **eval_kwargs)
        with _semantic_mode(enabled=True, weight=args.semantic_weight, top_n=args.semantic_top_n):
            semantic = evaluate_cases(all_cases, **eval_kwargs)
        out = {
            "mode": "semantic_compare",
            "settings": {
                "semantic_weight": args.semantic_weight,
                "semantic_top_n": args.semantic_top_n,
            },
            "baseline_semantic_off": baseline,
            "variant_semantic_on": semantic,
            "delta": _diff_summary(baseline, semantic),
        }
    else:
        out = evaluate_cases(all_cases, **eval_kwargs)

    if args.json:
        print(json.dumps(out, indent=2, ensure_ascii=False))
    else:
        # kept for future compact mode; current default remains JSON for easy parsing
        print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
