from __future__ import annotations

import argparse
import hashlib
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from .qa_engine import answer
from .telemetry import QueryLogger, build_report, default_log_path, event_from_payload


def _question_hash(question: str) -> str:
    return hashlib.sha256((question or "").encode("utf-8")).hexdigest()[:12]


def _extract_confidence(res: Any) -> dict[str, float]:
    for item in (getattr(res, "used_chunks", None) or []):
        if isinstance(item, dict) and item.get("kind") == "confidence":
            return {
                "retrieval_confidence": float(item.get("retrieval_confidence", 0.0)),
                "sentence_confidence": float(item.get("sentence_confidence", 0.0)),
                "domain_relevance": float(item.get("domain_relevance", 0.0)),
                "overall_confidence": float(item.get("overall_confidence", 0.0)),
                "threshold": float(item.get("threshold", 0.0)),
            }
    return {
        "retrieval_confidence": 0.0,
        "sentence_confidence": 0.0,
        "domain_relevance": 0.0,
        "overall_confidence": 0.0,
        "threshold": 0.0,
    }


def _extract_suggestions(res: Any) -> list[str]:
    for item in (getattr(res, "used_chunks", None) or []):
        if isinstance(item, dict) and item.get("kind") == "suggestions":
            vals = item.get("items", [])
            if isinstance(vals, list):
                return [str(v).strip() for v in vals if str(v).strip()]
    return []


def build_answer_payload(question: str) -> dict[str, Any]:
    q = (question or "").strip()
    if not q:
        raise ValueError("Question cannot be empty.")

    t0 = time.perf_counter()
    res = answer(q)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    text = getattr(res, "text", "") or ""
    insufficient = "insufficient evidence" in text.lower()
    cits = []
    for c in (getattr(res, "citations", None) or []):
        cits.append(
            {
                "doc_id": getattr(c, "doc_id", ""),
                "page": int(getattr(c, "page", 0) or 0),
                "chunk_id": getattr(c, "chunk_id", None),
            }
        )

    payload = {
        "question": q,
        "question_hash": _question_hash(q),
        "answer": text,
        "insufficient_evidence": insufficient,
        "intent": getattr(res, "intent", "unknown"),
        "scope": getattr(res, "scope", "MIXED"),
        "citations": cits,
        "suggestions": _extract_suggestions(res),
        "confidence": _extract_confidence(res),
        "latency_ms": round(latency_ms, 2),
    }
    return payload


class ApiMetrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.total_requests = 0
        self.insufficient_count = 0
        self.total_latency_ms = 0.0
        self.intent_counts: dict[str, int] = {}
        self.last_question_hash = ""

    def record(self, payload: dict[str, Any]) -> None:
        with self._lock:
            self.total_requests += 1
            if bool(payload.get("insufficient_evidence")):
                self.insufficient_count += 1
            self.total_latency_ms += float(payload.get("latency_ms", 0.0))
            intent = str(payload.get("intent", "unknown"))
            self.intent_counts[intent] = self.intent_counts.get(intent, 0) + 1
            self.last_question_hash = str(payload.get("question_hash", ""))

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            total = self.total_requests
            avg_latency = (self.total_latency_ms / total) if total else 0.0
            insuff_rate = (self.insufficient_count / total) if total else 0.0
            return {
                "total_requests": total,
                "insufficient_count": self.insufficient_count,
                "insufficient_rate": round(insuff_rate, 4),
                "avg_latency_ms": round(avg_latency, 2),
                "intent_counts": dict(self.intent_counts),
                "last_question_hash": self.last_question_hash,
            }


_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Regulatory QA Console</title>
  <style>
    :root {
      --ink: #1f2937;
      --muted: #5b6472;
      --bg: #f5f7fb;
      --card: #ffffff;
      --accent: #0b6bcb;
      --ok: #0f9d58;
      --warn: #b45309;
      --border: #dbe3ee;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Segoe UI", "Trebuchet MS", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(1200px 600px at 120% -10%, #d7ecff 0%, transparent 70%),
        radial-gradient(900px 500px at -20% 120%, #dff8ea 0%, transparent 70%),
        var(--bg);
      min-height: 100vh;
      padding: 20px;
    }
    .shell {
      max-width: 1100px;
      margin: 0 auto;
      display: grid;
      grid-template-columns: 1fr;
      gap: 14px;
    }
    .panel {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 16px;
      box-shadow: 0 8px 24px rgba(13, 38, 76, 0.06);
    }
    .title {
      margin: 0 0 6px 0;
      font-size: 1.4rem;
      letter-spacing: 0.3px;
    }
    .subtitle {
      margin: 0;
      color: var(--muted);
      font-size: 0.95rem;
    }
    textarea {
      width: 100%;
      min-height: 90px;
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 12px;
      font: inherit;
      resize: vertical;
      background: #fbfdff;
    }
    .row {
      display: flex;
      gap: 10px;
      align-items: center;
      margin-top: 10px;
      flex-wrap: wrap;
    }
    button {
      border: 0;
      background: var(--accent);
      color: white;
      padding: 10px 16px;
      border-radius: 10px;
      font-weight: 600;
      cursor: pointer;
    }
    button.secondary {
      background: #edf4ff;
      color: #15539f;
      border: 1px solid #c7defd;
    }
    .chips {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-top: 10px;
    }
    .chip {
      border: 1px solid var(--border);
      border-radius: 999px;
      padding: 4px 10px;
      font-size: 0.84rem;
      color: var(--muted);
      background: #f9fbff;
    }
    .chip.ok { border-color: #b8e9ce; color: var(--ok); background: #effcf4; }
    .chip.warn { border-color: #f6d2a4; color: var(--warn); background: #fff8ef; }
    #answer {
      white-space: pre-wrap;
      line-height: 1.45;
      margin-top: 10px;
    }
    .warnbox {
      margin-top: 10px;
      padding: 10px;
      border-radius: 10px;
      background: #fff4e8;
      border: 1px solid #f9d6a7;
      color: #8c4a06;
      display: none;
    }
    #suggestionsWrap { display: none; margin-top: 10px; }
    ul { margin: 8px 0 0 0; padding-left: 20px; }
    .muted { color: var(--muted); font-size: 0.9rem; }
    @media (max-width: 740px) {
      body { padding: 12px; }
      .panel { padding: 12px; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <section class="panel">
      <h1 class="title">Regulatory QA Console</h1>
      <p class="subtitle">Ask a regulatory question from the loaded document corpus.</p>
      <textarea id="q" placeholder="Example: What documentation is required for training and qualification?"></textarea>
      <div class="row">
        <button id="askBtn">Ask</button>
        <button class="secondary" id="metricsBtn">Refresh Metrics</button>
        <span class="muted" id="status"></span>
      </div>
    </section>

    <section class="panel">
      <div class="chips" id="chips"></div>
      <div class="warnbox" id="warnBox">Insufficient evidence in provided regulatory files.</div>
      <div id="answer"></div>
      <div id="suggestionsWrap">
        <h3>Suggested Rephrasings</h3>
        <ul id="suggestions"></ul>
      </div>
      <h3>Citations</h3>
      <ul id="citations"></ul>
    </section>

    <section class="panel">
      <h3>Runtime Metrics</h3>
      <pre id="metrics" class="muted">{}</pre>
    </section>
  </div>

  <script>
    const qEl = document.getElementById("q");
    const askBtn = document.getElementById("askBtn");
    const metricsBtn = document.getElementById("metricsBtn");
    const statusEl = document.getElementById("status");
    const chipsEl = document.getElementById("chips");
    const answerEl = document.getElementById("answer");
    const citationsEl = document.getElementById("citations");
    const suggestionsWrapEl = document.getElementById("suggestionsWrap");
    const suggestionsEl = document.getElementById("suggestions");
    const metricsEl = document.getElementById("metrics");
    const warnBox = document.getElementById("warnBox");

    async function fetchMetrics() {
      const r = await fetch("/metrics");
      const j = await r.json();
      metricsEl.textContent = JSON.stringify(j, null, 2);
    }

    function chip(label, cls = "") {
      const s = document.createElement("span");
      s.className = "chip " + cls;
      s.textContent = label;
      return s;
    }

    async function ask() {
      const q = qEl.value.trim();
      if (!q) return;
      statusEl.textContent = "Querying...";
      askBtn.disabled = true;
      try {
        const r = await fetch("/answer", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({question: q}),
        });
        const data = await r.json();
        if (!r.ok) throw new Error(data.error || "Request failed");

        chipsEl.innerHTML = "";
        chipsEl.appendChild(chip("intent: " + data.intent));
        chipsEl.appendChild(chip("scope: " + data.scope));
        chipsEl.appendChild(chip("latency: " + data.latency_ms + "ms"));
        const conf = data.confidence || {};
        const cls = data.insufficient_evidence ? "warn" : "ok";
        chipsEl.appendChild(chip("confidence: " + Number(conf.overall_confidence || 0).toFixed(3), cls));

        warnBox.style.display = data.insufficient_evidence ? "block" : "none";
        answerEl.textContent = data.answer || "";

        suggestionsEl.innerHTML = "";
        const suggestions = data.suggestions || [];
        if (suggestions.length > 0) {
          suggestionsWrapEl.style.display = "block";
          for (const s of suggestions) {
            const li = document.createElement("li");
            li.textContent = s;
            suggestionsEl.appendChild(li);
          }
        } else {
          suggestionsWrapEl.style.display = "none";
        }

        citationsEl.innerHTML = "";
        for (const c of (data.citations || [])) {
          const li = document.createElement("li");
          li.textContent = `${c.doc_id} | page ${c.page} | ${c.chunk_id || ""}`.trim();
          citationsEl.appendChild(li);
        }
        statusEl.textContent = "Done.";
        await fetchMetrics();
      } catch (e) {
        statusEl.textContent = String(e);
      } finally {
        askBtn.disabled = false;
      }
    }

    askBtn.addEventListener("click", ask);
    metricsBtn.addEventListener("click", fetchMetrics);
    qEl.addEventListener("keydown", (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "Enter") ask();
    });
    fetchMetrics();
  </script>
</body>
</html>
"""


class RegulatoryApiHandler(BaseHTTPRequestHandler):
    metrics = ApiMetrics()
    logger = QueryLogger(default_log_path())

    def _send_json(self, status: int, body: dict[str, Any]) -> None:
        raw = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _send_html(self, status: int, html: str) -> None:
        raw = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query or "")
        if path == "/":
            self._send_html(200, _HTML)
            return
        if path == "/health":
            self._send_json(200, {"status": "ok"})
            return
        if path == "/metrics":
            self._send_json(200, self.metrics.snapshot())
            return
        if path == "/admin/report":
            try:
                hours = int((qs.get("hours") or ["24"])[0])
            except Exception:
                hours = 24
            report = build_report(default_log_path(), hours=max(0, hours))
            self._send_json(200, report)
            return
        self._send_json(404, {"error": "Not Found"})

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path != "/answer":
            self._send_json(404, {"error": "Not Found"})
            return

        try:
            clen = int(self.headers.get("Content-Length", "0") or "0")
        except ValueError:
            clen = 0
        raw = self.rfile.read(clen) if clen > 0 else b""
        try:
            obj = json.loads(raw.decode("utf-8") if raw else "{}")
        except json.JSONDecodeError:
            self._send_json(400, {"error": "Invalid JSON body."})
            return

        question = str((obj or {}).get("question", "")).strip()
        if not question:
            self._send_json(400, {"error": "Field 'question' is required."})
            return

        try:
            payload = build_answer_payload(question)
        except Exception as e:
            self._send_json(500, {"error": f"Failed to answer question: {e}"})
            return

        self.metrics.record(payload)
        self.logger.write(event_from_payload(payload, source="web_api"))
        self._send_json(200, payload)

    def log_message(self, fmt: str, *args: Any) -> None:  # keep server output concise
        return


def run_server(host: str = "127.0.0.1", port: int = 8000) -> None:
    server = ThreadingHTTPServer((host, port), RegulatoryApiHandler)
    print(f"Regulatory QA API running on http://{host}:{port}")
    server.serve_forever()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m src.web_api")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args(argv)
    run_server(host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
