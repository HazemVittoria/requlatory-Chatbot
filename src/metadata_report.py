from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from . import search as search_module


def _build_summary(corpus: list[dict[str, Any]], *, top: int) -> dict[str, Any]:
    by_file: dict[str, dict[str, Any]] = {}
    domain_counts: Counter[str] = Counter()
    authority_counts: Counter[str] = Counter()

    for c in corpus:
        file_name = str(c.get("file") or "")
        authority = str(c.get("authority") or c.get("source") or "OTHER")
        domain = str(c.get("domain") or "Other")

        domain_counts[domain] += 1
        authority_counts[authority] += 1

        if file_name not in by_file:
            by_file[file_name] = {
                "file": file_name,
                "authority": authority,
                "domain": domain,
                "chunk_count": 0,
            }
        by_file[file_name]["chunk_count"] += 1

    docs = list(by_file.values())
    other_docs = [d for d in docs if str(d.get("domain")) == "Other"]
    other_docs.sort(key=lambda x: int(x.get("chunk_count", 0)), reverse=True)

    return {
        "chunks_total": len(corpus),
        "docs_total": len(docs),
        "domain_counts": dict(domain_counts),
        "authority_counts": dict(authority_counts),
        "other_docs_total": len(other_docs),
        "top_other_docs": other_docs[: max(0, top)],
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m src.metadata_report")
    p.add_argument("--top", type=int, default=25, help="Top N docs to show for domain=Other.")
    p.add_argument("--json", action="store_true", help="Emit JSON (default).")
    p.add_argument("--out", default="", help="Optional output JSON path.")
    args = p.parse_args(argv)

    search_module._ensure_index()
    corpus = list(search_module._CORPUS or [])
    report = _build_summary(corpus, top=max(0, args.top))

    raw = json.dumps(report, indent=2, ensure_ascii=False)
    print(raw)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(raw, encoding="utf-8")
        print(f"\nSaved report: {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
