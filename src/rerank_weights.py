from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path


@dataclass(frozen=True)
class RerankWeights:
    hybrid_word: float = 0.62
    hybrid_char: float = 0.28
    hybrid_overlap: float = 0.10

    structure_num_prefix_penalty: float = 0.35
    structure_mid_sentence_penalty: float = 0.25
    structure_page_marker_penalty: float = 0.25
    structure_short_chunk_penalty: float = 0.03

    anchor_hit_boost: float = 0.12

    definition_phrase_boost: float = 0.06
    procedure_phrase_boost: float = 0.05

    risk_phrase_boost: float = 0.18
    risk_csv_penalty: float = 0.20

    training_computerized_penalty: float = 0.60
    training_doc_term_boost: float = 0.15
    training_missing_doc_penalty: float = 0.10
    training_evidence_term_boost: float = 0.20
    training_coverage_boost: float = 0.12
    training_full_coverage_bonus: float = 0.25
    training_off_topic_penalty: float = 0.35
    training_equipment_penalty: float = 0.20
    training_q9_noise_penalty: float = 0.60
    training_nonbinding_penalty: float = 0.60


_CACHE: RerankWeights | None = None
_CACHE_KEY: str | None = None


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "config" / "rerank_weights.json"


def _load_override(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


def _apply_override(base: RerankWeights, override: dict) -> RerankWeights:
    allowed = {f.name for f in fields(RerankWeights)}
    patch = {}
    for k, v in override.items():
        if k in allowed:
            try:
                patch[k] = float(v)
            except Exception:
                continue
    return replace(base, **patch)


def get_rerank_weights() -> RerankWeights:
    global _CACHE, _CACHE_KEY

    env_path = os.getenv("RERANK_WEIGHTS_FILE", "").strip()
    config_path = Path(env_path) if env_path else _default_config_path()
    cache_key = str(config_path.resolve()) if config_path else ""

    if _CACHE is not None and _CACHE_KEY == cache_key:
        return _CACHE

    base = RerankWeights()
    override = _load_override(config_path)
    merged = _apply_override(base, override)

    _CACHE = merged
    _CACHE_KEY = cache_key
    return merged


def weights_as_dict() -> dict:
    return asdict(get_rerank_weights())
