from __future__ import annotations

import json
import re
import threading
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

_STOP = {
    "what",
    "how",
    "when",
    "where",
    "which",
    "with",
    "this",
    "that",
    "from",
    "into",
    "about",
    "should",
    "would",
    "could",
    "please",
    "under",
}
_TOKEN_RE = re.compile(r"[a-zA-Z]{4,}")


def default_log_path() -> Path:
    return Path("logs") / "qa_requests.jsonl"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_ts(ts: str) -> datetime | None:
    t = (ts or "").strip()
    if not t:
        return None
    try:
        if t.endswith("Z"):
            t = t[:-1] + "+00:00"
        return datetime.fromisoformat(t)
    except Exception:
        return None


class QueryLogger:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or default_log_path()
        self._lock = threading.Lock()

    def write(self, event: dict[str, Any]) -> None:
        row = dict(event)
        row.setdefault("ts_utc", _utc_now_iso())
        raw = json.dumps(row, ensure_ascii=False)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(raw + "\n")


def event_from_payload(payload: dict[str, Any], *, source: str = "web_api") -> dict[str, Any]:
    cits = payload.get("citations") or []
    suggestions = payload.get("suggestions") or []
    retrieval_profile = payload.get("retrieval_profile") or {}
    return {
        "ts_utc": _utc_now_iso(),
        "source": source,
        "question_hash": payload.get("question_hash", ""),
        "question_text": payload.get("question", ""),
        "intent": payload.get("intent", "unknown"),
        "presentation_intent": payload.get("presentation_intent", "requirements"),
        "scope": payload.get("scope", "MIXED"),
        "latency_ms": float(payload.get("latency_ms", 0.0)),
        "insufficient_evidence": bool(payload.get("insufficient_evidence", False)),
        "confidence": payload.get("confidence", {}),
        "retrieval_profile": retrieval_profile,
        "citations": cits[:5],
        "suggestions": suggestions[:5],
    }


def load_events(path: Path, *, limit: int = 100000) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    if len(rows) > limit:
        rows = rows[-limit:]
    return rows


def _rolling_events(events: list[dict[str, Any]], hours: int) -> list[dict[str, Any]]:
    if hours <= 0:
        return events
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=hours)
    out: list[dict[str, Any]] = []
    for e in events:
        dt = _parse_ts(str(e.get("ts_utc", "")))
        if dt is None:
            continue
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        if dt >= cutoff:
            out.append(e)
    return out


def _extract_patterns(events: list[dict[str, Any]], top_n: int = 10) -> list[dict[str, Any]]:
    cnt: Counter[str] = Counter()
    for e in events:
        q = str(e.get("question_text", "")).lower()
        for tok in _TOKEN_RE.findall(q):
            if tok in _STOP:
                continue
            cnt[tok] += 1
    out = []
    for token, n in cnt.most_common(top_n):
        out.append({"token": token, "count": n})
    return out


def build_report(path: Path, *, hours: int = 24, limit: int = 100000) -> dict[str, Any]:
    events_all = load_events(path, limit=limit)
    events = _rolling_events(events_all, hours=hours)

    total = len(events)
    insuff = [e for e in events if bool(e.get("insufficient_evidence", False))]
    low_conf = []
    intents: Counter[str] = Counter()
    latency_sum = 0.0

    for e in events:
        intents[str(e.get("intent", "unknown"))] += 1
        latency_sum += float(e.get("latency_ms", 0.0))
        conf = e.get("confidence") or {}
        overall = float(conf.get("overall_confidence", 0.0) or 0.0)
        threshold = float(conf.get("threshold", 0.0) or 0.0)
        if overall < max(threshold, 0.22):
            low_conf.append(e)

    avg_latency = (latency_sum / total) if total else 0.0
    insuff_rate = (len(insuff) / total) if total else 0.0

    failures = []
    for e in insuff[-20:]:
        failures.append(
            {
                "ts_utc": e.get("ts_utc"),
                "question_hash": e.get("question_hash"),
                "question_text": e.get("question_text"),
                "intent": e.get("intent"),
                "confidence": e.get("confidence", {}),
                "citations": e.get("citations", []),
                "suggestions": e.get("suggestions", []),
            }
        )

    return {
        "window_hours": hours,
        "events_total_window": total,
        "events_total_all": len(events_all),
        "insufficient_count": len(insuff),
        "insufficient_rate": round(insuff_rate, 4),
        "avg_latency_ms": round(avg_latency, 2),
        "intent_counts": dict(intents),
        "low_confidence_count": len(low_conf),
        "top_low_conf_tokens": _extract_patterns(low_conf, top_n=12),
        "recent_failures": failures,
    }
