from __future__ import annotations

import json
from pathlib import Path

from src.rerank_weights import get_rerank_weights


def test_rerank_weights_can_be_overridden_by_file(monkeypatch, tmp_path: Path):
    path = tmp_path / "weights.json"
    path.write_text(json.dumps({"hybrid_word": 0.5, "risk_phrase_boost": 0.33}), encoding="utf-8")
    monkeypatch.setenv("RERANK_WEIGHTS_FILE", str(path))

    w = get_rerank_weights()
    assert w.hybrid_word == 0.5
    assert w.risk_phrase_boost == 0.33
