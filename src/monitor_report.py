from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .telemetry import build_report, default_log_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m src.monitor_report")
    p.add_argument("--log", default=str(default_log_path()), help="Path to JSONL telemetry log.")
    p.add_argument("--hours", type=int, default=24, help="Rolling window (hours).")
    p.add_argument("--limit", type=int, default=100000, help="Max log rows to inspect.")
    p.add_argument("--out", default="", help="Optional output report path (json).")
    args = p.parse_args(argv)

    report = build_report(Path(args.log), hours=max(0, args.hours), limit=max(1000, args.limit))
    raw = json.dumps(report, indent=2, ensure_ascii=False)
    print(raw)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "report": report,
        }
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nSaved report: {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
