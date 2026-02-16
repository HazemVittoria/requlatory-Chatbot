# src/search.py
from __future__ import annotations

import os
import pickle
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .ingestion import build_corpus
from .metadata_schema import authority_from_source, authorities_for_scope
from .rerank_features import RerankContext, apply_rerank_features, combine_hybrid_scores
from .rerank_weights import get_rerank_weights
from .semantic_reranker import build_semantic_reranker, SemanticRerankConfig

_DEBUG = os.getenv("DEBUG_RETRIEVAL") == "1"
_DEBUG_INDEX_CACHE = _DEBUG or (os.getenv("DEBUG_INDEX_CACHE", "0").strip().lower() in {"1", "true", "yes"})


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except Exception:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except Exception:
        return default


def _env_str(name: str, default: str) -> str:
    raw = os.getenv(name, "").strip()
    return raw if raw else default


_ENABLE_SEMANTIC_RERANK = os.getenv("ENABLE_SEMANTIC_RERANK", "0").strip().lower() not in {"0", "false", "no"}
_SEMANTIC_RERANK_WEIGHT = _env_float("SEMANTIC_RERANK_WEIGHT", 0.18)
_SEMANTIC_RERANK_TOP_N = _env_int("SEMANTIC_RERANK_TOP_N", 30)

_WORD_VECT_KW = {"stop_words": "english", "ngram_range": (1, 2), "min_df": 1}
_CHAR_VECT_KW = {"analyzer": "char_wb", "ngram_range": (3, 5), "min_df": 1}
_SEMANTIC_CONFIG = SemanticRerankConfig(
    backend=_env_str("SEMANTIC_RERANK_BACKEND", "auto"),
    embedding_model=_env_str("SEMANTIC_EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
    embedding_batch_size=_env_int("SEMANTIC_EMBEDDING_BATCH_SIZE", 64),
)

_INDEX_CACHE_VERSION = "search-index-v3-architecture"
_INDEX_CACHE_FILE = Path(".cache") / "retrieval_index.pkl"

_CORPUS: list[dict[str, Any]] | None = None
_VECTORIZER: TfidfVectorizer | None = None
_VECTORIZER_CHAR: TfidfVectorizer | None = None
_SEMANTIC_RERANKER = None
_X = None
_X_CHAR = None

_WS_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[a-zA-Z]{3,}")


def _is_truthy_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes"}


def _data_fingerprint(data_dir: Path) -> list[tuple[str, int, int]]:
    out: list[tuple[str, int, int]] = []
    if not data_dir.exists():
        return out
    for p in sorted(data_dir.rglob("*.pdf"), key=lambda x: str(x).lower()):
        try:
            st = p.stat()
        except OSError:
            continue
        rel = str(p.relative_to(data_dir)).replace("\\", "/")
        out.append((rel, int(st.st_size), int(st.st_mtime_ns)))
    return out


def _index_meta(data_dir: Path) -> dict[str, Any]:
    return {
        "version": _INDEX_CACHE_VERSION,
        "data_dir": str(data_dir.resolve()),
        "data_fingerprint": _data_fingerprint(data_dir),
        "word_vectorizer": dict(_WORD_VECT_KW),
        "char_vectorizer": dict(_CHAR_VECT_KW),
        "semantic_config": asdict(_SEMANTIC_CONFIG),
    }


def _load_index_cache(meta: dict[str, Any]) -> bool:
    global _CORPUS, _VECTORIZER, _VECTORIZER_CHAR, _SEMANTIC_RERANKER, _X, _X_CHAR
    if not _INDEX_CACHE_FILE.exists():
        return False
    try:
        with _INDEX_CACHE_FILE.open("rb") as f:
            payload = pickle.load(f)
    except Exception:
        return False

    if not isinstance(payload, dict):
        return False
    if payload.get("meta") != meta:
        return False

    corpus = payload.get("corpus")
    vect = payload.get("vectorizer")
    vect_char = payload.get("vectorizer_char")
    x = payload.get("x")
    x_char = payload.get("x_char")
    sem = payload.get("semantic_reranker")
    if corpus is None or vect is None or vect_char is None or x is None or x_char is None:
        return False

    _CORPUS = corpus
    _VECTORIZER = vect
    _VECTORIZER_CHAR = vect_char
    _X = x
    _X_CHAR = x_char
    _SEMANTIC_RERANKER = sem
    return True


def _save_index_cache(meta: dict[str, Any]) -> None:
    payload = {
        "meta": meta,
        "corpus": _CORPUS,
        "vectorizer": _VECTORIZER,
        "vectorizer_char": _VECTORIZER_CHAR,
        "x": _X,
        "x_char": _X_CHAR,
        "semantic_reranker": _SEMANTIC_RERANKER,
    }
    try:
        _INDEX_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with _INDEX_CACHE_FILE.open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        if _DEBUG_INDEX_CACHE:
            print(f"[index-cache] failed to save cache: {e}")


def _rebuild_index(force: bool = False) -> None:
    global _CORPUS, _VECTORIZER, _VECTORIZER_CHAR, _SEMANTIC_RERANKER, _X, _X_CHAR
    data_dir = Path("data")
    meta = _index_meta(data_dir)
    t0 = time.perf_counter()

    if not force and _load_index_cache(meta):
        if _DEBUG_INDEX_CACHE:
            dt = time.perf_counter() - t0
            print(f"[index-cache] hit: loaded index in {dt:.2f}s from {_INDEX_CACHE_FILE}")
        return

    if _DEBUG_INDEX_CACHE:
        reason = "force rebuild" if force else "cache miss/invalid"
        print(f"[index-cache] rebuild: {reason}")

    _CORPUS = build_corpus(data_dir)
    texts = [(c.get("text") or "") for c in _CORPUS]
    _VECTORIZER = TfidfVectorizer(**_WORD_VECT_KW)
    _VECTORIZER_CHAR = TfidfVectorizer(**_CHAR_VECT_KW)
    _X = _VECTORIZER.fit_transform(texts)
    _X_CHAR = _VECTORIZER_CHAR.fit_transform(texts)
    _SEMANTIC_RERANKER = build_semantic_reranker(_SEMANTIC_CONFIG)
    _SEMANTIC_RERANKER.fit(texts)
    _save_index_cache(meta)
    if _DEBUG_INDEX_CACHE:
        dt = time.perf_counter() - t0
        print(f"[index-cache] built and saved in {dt:.2f}s ({len(_CORPUS)} chunks)")


def _ensure_index() -> None:
    if _is_truthy_env("REBUILD_INDEX"):
        _rebuild_index(force=True)
        return
    if _CORPUS is None or _VECTORIZER is None or _VECTORIZER_CHAR is None or _X is None or _X_CHAR is None:
        _rebuild_index(force=False)


def resolve_authority_filter(scope: str, authority_filter: list[str] | None = None) -> set[str]:
    if authority_filter:
        out: set[str] = set()
        for a in authority_filter:
            out.add(authority_from_source(a))
        return out
    return authorities_for_scope(scope)


def infer_domain_boost(query: str, intent: str | None = None, anchor_terms: list[str] | None = None) -> dict[str, float]:
    del intent  # keep function signature stable for callers
    q = f"{query or ''} {' '.join(anchor_terms or [])}".lower()
    out: dict[str, float] = {}

    domain_signals: dict[str, tuple[str, ...]] = {
        "DataIntegrity": ("data integrity", "part 11", "annex 11", "audit trail", "alcoa", "electronic signature"),
        "Validation": ("validation", "validated", "iq", "oq", "pq"),
        "QC_Lab": ("laboratory", "analytical", "oos", "oot", "out of specification", "out of trend"),
        "Suppliers": ("supplier", "vendor", "contractor"),
        "Inspection": ("inspection", "483", "inspectional", "audit"),
        "Distribution": ("distribution", "transport", "storage", "cold chain"),
        "PQS": ("deviation", "capa", "quality system", "risk management", "change control"),
    }

    for domain, signals in domain_signals.items():
        hits = sum(1 for sig in signals if sig in q)
        if hits <= 0:
            continue
        # Generic, monotonic scoring by signal density.
        out[domain] = min(0.16, 0.06 + (0.02 * hits))

    return out


def expand_query(query: str) -> str:
    # Phase B: keep retrieval deterministic and avoid question-specific query inflation.
    return _WS_RE.sub(" ", (query or "")).strip()


def semantic_backend_name() -> str:
    if _SEMANTIC_RERANKER is None:
        return "none"
    name = getattr(_SEMANTIC_RERANKER, "backend_name", "")
    if isinstance(name, str) and name.strip():
        return name.strip().lower()
    return type(_SEMANTIC_RERANKER).__name__


def search_chunks(
    query: str,
    scope: str = "MIXED",
    top_k: int = 5,
    anchor_terms: list[str] | None = None,
    intent: str | None = None,
    authority_filter: list[str] | None = None,
    domain_boost: dict[str, float] | None = None,
) -> list[dict]:
    _ensure_index()
    assert _CORPUS is not None and _VECTORIZER is not None and _VECTORIZER_CHAR is not None and _X is not None and _X_CHAR is not None

    anchor_terms = anchor_terms or []
    weights = get_rerank_weights()
    ctx = RerankContext.from_query(query=query, intent=intent, anchor_terms=anchor_terms)
    eff_domain_boost = domain_boost or infer_domain_boost(query, intent=intent, anchor_terms=anchor_terms)
    allowed_authorities = resolve_authority_filter(scope, authority_filter=authority_filter)

    q_aug = query + (" " + " ".join(anchor_terms) if anchor_terms else "")
    q2 = expand_query(q_aug)
    qv_word = _VECTORIZER.transform([q2])
    qv_char = _VECTORIZER_CHAR.transform([q2])
    sims_word = cosine_similarity(qv_word, _X).flatten()
    sims_char = cosine_similarity(qv_char, _X_CHAR).flatten()

    query_tokens = {t for t in _TOKEN_RE.findall(q2.lower()) if t not in {"what", "when", "where", "which", "should"}}
    overlap_scores = []
    for c in _CORPUS:
        c_tokens = set(_TOKEN_RE.findall((c.get("text") or "").lower()))
        if not query_tokens:
            overlap_scores.append(0.0)
            continue
        overlap_scores.append(len(query_tokens & c_tokens) / max(1, len(query_tokens)))

    # Hybrid lexical + approximate semantic score
    base_scores = combine_hybrid_scores(
        sims_word=sims_word,
        sims_char=sims_char,
        overlap_scores=np.asarray(overlap_scores),
        weights=weights,
    )
    boosted = apply_rerank_features(base_scores=base_scores, corpus=_CORPUS, ctx=ctx, weights=weights)
    if eff_domain_boost:
        for i, c in enumerate(_CORPUS):
            d = str(c.get("domain") or "Other")
            boosted[i] += float(eff_domain_boost.get(d, 0.0))

    # Optional second-stage semantic rerank (A/B via env flag).
    if _ENABLE_SEMANTIC_RERANK and _SEMANTIC_RERANKER is not None and len(boosted) > 0:
        top_n = max(1, min(len(boosted), _SEMANTIC_RERANK_TOP_N))
        candidate_idx = boosted.argsort()[::-1][:top_n]
        sem_scores = _SEMANTIC_RERANKER.score_query(q2, candidate_idx)
        for j, ix in enumerate(candidate_idx):
            sem_delta = float(sem_scores[j]) - 0.5
            boosted[int(ix)] += _SEMANTIC_RERANK_WEIGHT * sem_delta

    if _DEBUG:
        print("\n=== RETRIEVAL DEBUG ===")
        print(f"Query: {query}")
        print(f"Intent: {intent}")
        print(f"Scope: {scope}")
        print(f"Authority filter: {sorted(allowed_authorities)}")
        print(f"Domain boost: {eff_domain_boost}")
        print(f"Anchor terms: {anchor_terms}")
        print(
            "Semantic rerank: "
            f"{'on' if _ENABLE_SEMANTIC_RERANK else 'off'} "
            f"(backend={semantic_backend_name()}, "
            f"top_n={_SEMANTIC_RERANK_TOP_N}, weight={_SEMANTIC_RERANK_WEIGHT})"
        )

    idx_all = boosted.argsort()[::-1]

    if _DEBUG:
        print("\nTop 5 chunks (after rerank):")
        for rank, ix in enumerate(idx_all[:5], start=1):
            c = _CORPUS[int(ix)]
            snippet = (c.get("text") or "")[:120].replace("\n", " ")
            print(f"{rank}. Score={boosted[int(ix)]:.4f} | {c.get('file')} | p{c.get('page')} | {snippet}")
        print("=== END DEBUG ===\n")

    out: list[dict] = []
    for ix in idx_all:
        c = dict(_CORPUS[int(ix)])
        authority = str(c.get("authority") or c.get("source") or "OTHER")
        authority = authority_from_source(authority)
        if allowed_authorities and authority not in allowed_authorities:
            continue

        c["_score"] = float(boosted[int(ix)])
        out.append(c)
        if len(out) >= top_k:
            break

    return out
