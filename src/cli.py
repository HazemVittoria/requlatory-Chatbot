from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, is_dataclass
from typing import Any, Callable


def _resolve_engine_callable() -> Callable[..., Any]:
    try:
        from . import qa_engine  # type: ignore
    except Exception as e:
        raise RuntimeError(f"Failed to import src.qa_engine: {e}") from e

    for name in ("answer", "answer_question", "ask", "run", "qa"):
        fn = getattr(qa_engine, name, None)
        if callable(fn):
            return fn

    cls = getattr(qa_engine, "QAEngine", None)
    if cls is not None:
        inst = cls()
        fn = getattr(inst, "answer", None)
        if callable(fn):
            return fn

    raise RuntimeError(
        "Could not find callable entrypoint in src.qa_engine. "
        "Expected answer/answer_question/ask/run/qa or QAEngine().answer()."
    )


def _to_jsonable(x: Any) -> Any:
    if is_dataclass(x):
        return asdict(x)
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    if x is None or isinstance(x, (str, int, float, bool)):
        return x
    return str(x)


def _normalize_result(res: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"answer": "", "citations": [], "raw": res}

    if res is None:
        return out

    # Handle AnswerResult-like objects (text may be under different attributes)
    for attr in ("text", "answer", "final", "response", "content"):
        val = getattr(res, attr, None)
        if isinstance(val, str) and val.strip():
            out["answer"] = val
            break

    # Handle citations if present as attribute
    cits = getattr(res, "citations", None)
    if isinstance(cits, list):
        rendered: list[str] = []
        for c in cits:
            doc = getattr(c, "doc_id", None) or getattr(c, "doc", None) or ""
            page = getattr(c, "page", None)
            chunk = getattr(c, "chunk_id", None) or getattr(c, "chunk", None) or ""
            if doc and page is not None:
                rendered.append(f"{doc} | page {page} | {chunk}".strip())
            else:
                rendered.append(str(c))
        out["citations"] = rendered

    # If already extracted answer or citations, return
    if out["answer"] or out["citations"]:
        return out

    # Plain string response
    if isinstance(res, str):
        out["answer"] = res
        return out

    # Dict response
    if isinstance(res, dict):
        for k in ("answer", "final", "text", "response"):
            if k in res and isinstance(res[k], str):
                out["answer"] = res[k]
                break
        for k in ("citations", "sources", "refs", "reference"):
            if k in res and isinstance(res[k], list):
                out["citations"] = res[k]
                break
        return out

    # Tuple/list response
    if isinstance(res, (tuple, list)) and len(res) >= 1:
        if isinstance(res[0], str):
            out["answer"] = res[0]
        if len(res) >= 2 and isinstance(res[1], list):
            out["citations"] = res[1]
        return out

    # Fallback
    out["answer"] = str(res)
    return out



def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m src.cli", add_help=True)
    p.add_argument("question", help="Question text (wrap in quotes).")
    p.add_argument("--json", action="store_true", help="Emit JSON to stdout.")
    p.add_argument("--topk", type=int, default=None, help="Optional retrieval top-k if supported.")
    p.add_argument("--rebuild-index", action="store_true", help="Force rebuild corpus/vector index before answering.")
    args = p.parse_args(argv)

    if args.rebuild_index:
        os.environ["REBUILD_INDEX"] = "1"

    engine_fn = _resolve_engine_callable()

    kwargs: dict[str, Any] = {}
    if args.topk is not None:
        kwargs["topk"] = args.topk

    try:
        res = engine_fn(args.question, **kwargs)
    except TypeError:
        res = engine_fn(args.question)

    norm = _normalize_result(res)

    if args.json:
        print(json.dumps(_to_jsonable(norm), ensure_ascii=False, indent=2))
        return 0

    ans = (norm.get("answer") or "").rstrip()
    print(ans)

    cits = norm.get("citations") or []
    if cits:
        print("\nCitations:")
        for c in cits:
            print(f"- {c}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
