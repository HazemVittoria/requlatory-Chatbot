from __future__ import annotations

import re
import logging

logging.getLogger("pypdf").setLevel(logging.ERROR)
from pathlib import Path
from collections import Counter

from pypdf import PdfReader

from .metadata_rules import build_document_metadata
from .metadata_schema import authority_from_folder, authority_from_source


_WS_RE = re.compile(r"\s+")
_PAGE_RE = re.compile(r"\bpage\s+\d+\s+of\s+\d+\b", flags=re.IGNORECASE)
_PAGE_NUM_RE = re.compile(r"^\s*(?:page\s+)?\d+\s*(?:/\s*\d+)?\s*$", flags=re.IGNORECASE)
_HEADING_RE = re.compile(r"^\d+(?:\.\d+)*\s+[A-Z][A-Za-z0-9 \-(),/&]{2,}$")
_LIST_ROW_RE = re.compile(r"^\s*(?:[-*]|[a-zA-Z]\)|\d+[.)])\s+")
_TABLE_RE = re.compile(r"^\s*table\s+\d+[A-Za-z]?(?:[:.\-]\s*|\s+).*$", flags=re.IGNORECASE)


def _normalize_line(line: str) -> str:
    s = (line or "").replace("\u00a0", " ").replace("\uf0b7", " ")
    s = re.sub(r"[\t\r]", " ", s)
    return _WS_RE.sub(" ", s).strip()


def _normalize_for_count(line: str) -> str:
    s = _normalize_line(line).lower()
    if not s:
        return ""
    s = _PAGE_RE.sub(" ", s)
    s = re.sub(r"\b\d+\b", "#", s)
    return _WS_RE.sub(" ", s).strip()


def _looks_like_artifact_line(line: str) -> bool:
    s = line.strip()
    sl = s.lower()
    if not s:
        return True
    if _PAGE_NUM_RE.match(s):
        return True
    if _PAGE_RE.search(sl):
        return True
    if "table of contents" in sl:
        return True
    if set(s) <= {".", "-", "_"} and len(s) >= 6:
        return True
    if re.match(r"^\d+(\.\d+)*\s*$", s):
        return True
    return False


def _is_table_or_list_paragraph(text: str) -> bool:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return True
    joined = " ".join(lines).lower()
    if "inspector" in joined and "aide memoir" in joined:
        return True
    if any(_TABLE_RE.match(ln) for ln in lines):
        return True
    if re.search(r"\b(histogram|pareto|process capability)\b", joined):
        return True
    if len(lines) >= 3:
        list_like = sum(1 for ln in lines if _LIST_ROW_RE.match(ln) or re.search(r"\s{3,}", ln))
        if list_like / len(lines) >= 0.6:
            return True
    return False


def _paragraphs_from_lines(lines: list[str]) -> list[str]:
    paragraphs: list[str] = []
    current: list[str] = []

    def flush() -> None:
        if not current:
            return
        text = " ".join(current).strip()
        text = _WS_RE.sub(" ", text)
        if text:
            paragraphs.append(text)
        current.clear()

    for raw in lines:
        line = _normalize_line(raw)
        if not line or _looks_like_artifact_line(line):
            flush()
            continue

        # Keep section headings as their own paragraph boundary.
        if _HEADING_RE.match(line):
            flush()
            paragraphs.append(line)
            continue

        if _LIST_ROW_RE.match(line):
            flush()
            paragraphs.append(line)
            continue

        if current:
            prev = current[-1]
            if prev.endswith((".", "!", "?", ":", ";")):
                flush()
        current.append(line)

    flush()
    return paragraphs


def _clean_chunk_text(text: str) -> str:
    if not text:
        return ""
    s = _WS_RE.sub(" ", text.replace("\u00a0", " ")).strip()
    return s



def chunk_text(text: str, chunk_size: int = 900) -> list[str]:
    if not text:
        return []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""

    for p in paragraphs:
        if len(current) + len(p) < chunk_size:
            current += " " + p if current else p
        else:
            if current:
                chunks.append(current.strip())
            current = p

    if current:
        chunks.append(current.strip())

    return chunks




def is_low_value(text: str) -> bool:
    if not text:
        return True
    t = text.lower()

    # Drop table-heavy fragments
    if re.search(r"\btable\s+\d+\b", t):
        return True

# Drop "Inspector’s Aide Memoir" table content
    if "aide memoir" in t or ("inspector" in t and "memoir" in t):
        return True

# Drop Q9 annex/tool-list noise
    if "histogram" in t or "pareto" in t or "process capability" in t:
        return True

    if "table of contents" in t:
        return True
    if "................................" in text:
        return True

    if len(text) < 60:
        return not bool(re.search(r"\b(is|are|means|defined as|refers to)\b", t))

    return False


def _resolve_authority(folder: Path, source: str | None = None, authority: str | None = None) -> str:
    if authority:
        return authority_from_source(authority)
    if source:
        return authority_from_source(source)
    return authority_from_folder(folder.name)


def ingest_folder(folder: Path, source: str | None = None, authority: str | None = None) -> list[dict]:
    chunks: list[dict] = []
    if not folder.exists():
        return chunks

    authority_value = _resolve_authority(folder, source=source, authority=authority)

    for pdf_path in folder.glob("*.pdf"):
        reader = PdfReader(str(pdf_path))
        page_lines: list[tuple[int, list[str]]] = []
        line_df: Counter[str] = Counter()

        for page_idx, page in enumerate(reader.pages, start=1):
            raw = page.extract_text() or ""
            lines = [ln for ln in raw.splitlines() if _normalize_line(ln)]
            page_lines.append((page_idx, lines))

            seen_on_page = {_normalize_for_count(ln) for ln in lines}
            seen_on_page.discard("")
            line_df.update(seen_on_page)

        sample_text = ""
        if page_lines:
            sample_text = "\n".join(page_lines[0][1][:80])
        doc_meta = build_document_metadata(pdf_path.name, authority=authority_value, sample_text=sample_text)

        min_df = max(2, int(len(page_lines) * 0.2 + 0.999))
        repeated = {ln for ln, freq in line_df.items() if freq >= min_df}

        for page_idx, lines in page_lines:
            filtered_lines = [
                ln
                for ln in lines
                if _normalize_for_count(ln) not in repeated and not _looks_like_artifact_line(ln)
            ]
            paragraphs = _paragraphs_from_lines(filtered_lines)
            paragraphs = [p for p in paragraphs if not _is_table_or_list_paragraph(p)]
            page_text = "\n\n".join(paragraphs)
            if not page_text:
                continue

            for i, c in enumerate(chunk_text(page_text), start=1):
                c = _clean_chunk_text(c)
                if is_low_value(c):
                    continue
                chunks.append(
                    {
                        "source": source or authority_value,
                        "file": pdf_path.name,
                        "page": page_idx,
                        "chunk_id": f"p{page_idx}_c{i}",
                        "text": c,
                        **doc_meta,
                    }
                )
    return chunks


def build_corpus(data_dir: Path) -> list[dict]:
    corpus: list[dict] = []

    folder_specs: list[tuple[str, str]] = [
        ("ich", "ICH"),
        ("fda", "FDA"),
        ("ema", "EMA"),
        ("eu_gmp", "EU_GMP"),
        ("pic_s", "PIC_S"),
        ("who", "WHO"),
        ("sop", "SOP"),
        ("sops", "SOP"),
        ("other", "OTHER"),
        ("others", "OTHER"),
    ]
    seen: set[Path] = set()
    for name, authority in folder_specs:
        folder = data_dir / name
        if not folder.exists():
            continue
        seen.add(folder.resolve())
        corpus += ingest_folder(folder, source=authority, authority=authority)

    # Fallback: ingest additional subfolders not explicitly listed.
    if data_dir.exists():
        for child in data_dir.iterdir():
            if not child.is_dir():
                continue
            cpath = child.resolve()
            if cpath in seen:
                continue
            corpus += ingest_folder(child, authority=authority_from_folder(child.name))
    return corpus
