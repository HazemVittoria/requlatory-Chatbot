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
_SECTION_PREFIX_RE = re.compile(
    r"^\s*(?:annex|chapter|section|part|appendix|module)\b",
    flags=re.IGNORECASE,
)
_NUMBERED_HEADING_RE = re.compile(
    r"^\s*(?:\d+(?:\.\d+){0,4}|[ivxlcdm]{1,8})[.)]?\s+[A-Z][A-Za-z0-9 \-(),/&]{2,}$"
)
_ALL_CAPS_HEADING_RE = re.compile(r"^[A-Z0-9][A-Z0-9 \-(),/&]{4,}$")
_LIST_ROW_RE = re.compile(
    r"^\s*(?:[-*•]|[a-zA-Z]\)|\(?[ivxlcdm]{1,8}\)|\d+(?:\.\d+){0,3}[.)])\s+",
    flags=re.IGNORECASE,
)
_TABLE_RE = re.compile(r"^\s*table\s+\d+[A-Za-z]?(?:[:.\-]\s*|\s+).*$", flags=re.IGNORECASE)
_URL_LINE_RE = re.compile(r"^\s*(?:https?://|www\.)", flags=re.IGNORECASE)
_FIGURE_RE = re.compile(r"^\s*(?:figure|fig\.)\s*\d+[A-Za-z]?\b", flags=re.IGNORECASE)


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


def _is_heading_line(line: str) -> bool:
    s = (line or "").strip()
    if not s:
        return False
    if len(s) > 180:
        return False
    if _TABLE_RE.match(s) or _FIGURE_RE.match(s):
        return False
    if _SECTION_PREFIX_RE.match(s):
        return True
    if _NUMBERED_HEADING_RE.match(s):
        return True
    # Pure uppercase section titles are common in scanned/regulatory PDFs.
    if _ALL_CAPS_HEADING_RE.match(s):
        letters = sum(1 for ch in s if ch.isalpha())
        return letters >= 6
    return False


def _is_list_item_line(line: str) -> bool:
    return bool(_LIST_ROW_RE.match(line or ""))


def _looks_like_table_row(line: str) -> bool:
    s = (line or "").strip()
    if not s:
        return False
    sl = s.lower()
    if _TABLE_RE.match(s):
        return True
    if "|" in s or "\t" in s:
        return True
    if re.search(r"\brow\b|\bcolumn\b", sl):
        return True

    # Multi-column rows often have repeated wide spacing.
    cols = [c for c in re.split(r"\s{2,}", s) if c.strip()]
    if len(cols) >= 3:
        numericish = sum(bool(re.search(r"\d", c)) for c in cols)
        shortish = sum(len(c.strip()) <= 12 for c in cols)
        headerish = sum(bool(re.match(r"^[A-Za-z][A-Za-z/\-]{1,20}$", c.strip())) for c in cols)
        if numericish >= 2 or (numericish >= 1 and shortish >= 2) or headerish >= 3:
            return True

    # Dense alpha-numeric cells are often table leftovers.
    letters = sum(ch.isalpha() for ch in s)
    digits = sum(ch.isdigit() for ch in s)
    punct = sum((not ch.isalnum() and not ch.isspace()) for ch in s)
    total = max(1, len(s))
    non_alpha_ratio = (digits + punct) / total
    if non_alpha_ratio >= 0.42 and letters <= 24 and digits >= 3:
        return True

    return False


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
    if _URL_LINE_RE.match(s):
        return True
    if _FIGURE_RE.match(s):
        return True
    if _looks_like_table_row(s):
        return True
    if set(s) <= {".", "-", "_"} and len(s) >= 6:
        return True
    if re.match(r"^[\W_]{4,}$", s):
        return True
    if re.match(r"^\d+(\.\d+)*\s*$", s):
        return True
    if re.match(r"^\(?[ivxlcdm]{1,8}\)?$", s, flags=re.IGNORECASE):
        return True
    return False


def _is_table_or_list_paragraph(text: str) -> bool:
    lines = [_normalize_line(ln) for ln in text.splitlines() if _normalize_line(ln)]
    if not lines:
        return True
    joined = " ".join(lines).lower()
    if "inspector" in joined and "aide memoir" in joined:
        return True
    if any(_looks_like_table_row(ln) for ln in lines):
        return True
    if re.search(r"\b(histogram|pareto|process capability)\b", joined):
        return True
    if all(_is_list_item_line(ln) for ln in lines) and len(lines) >= 10:
        # Very long list-only fragments are usually low retrieval value.
        return True
    if len(lines) >= 6:
        list_like = sum(1 for ln in lines if _is_list_item_line(ln) or re.search(r"\s{3,}", ln))
        if list_like / len(lines) >= 0.6:
            return True
    return False


def _is_list_continuation(line: str) -> bool:
    s = (line or "").strip()
    if not s:
        return False
    if _is_heading_line(s) or _is_list_item_line(s):
        return False
    # Continuation lines usually start with lowercase or opening punctuation.
    return s[0].islower() or s[0] in {"(", "[", "/", "&"}


def _build_page_blocks(lines: list[str]) -> list[str]:
    blocks: list[str] = []
    section = ""
    para: list[str] = []
    list_items: list[str] = []

    def with_section(text: str) -> str:
        t = (text or "").strip()
        if not t:
            return ""
        if not section:
            return t
        if t.lower().startswith(section.lower()):
            return t
        return f"{section}\n{t}"

    def flush_para() -> None:
        if not para:
            return
        t = _WS_RE.sub(" ", " ".join(para)).strip()
        para.clear()
        if not t:
            return
        t = with_section(t)
        if t and not _is_table_or_list_paragraph(t):
            blocks.append(t)

    def flush_list() -> None:
        if not list_items:
            return
        t = "\n".join(x.strip() for x in list_items if x.strip())
        list_items.clear()
        if not t:
            return
        t = with_section(t)
        if t and not _is_table_or_list_paragraph(t):
            blocks.append(t)

    for raw in lines:
        line = _normalize_line(raw)
        if not line or _looks_like_artifact_line(line):
            flush_para()
            flush_list()
            continue

        if _is_heading_line(line):
            flush_para()
            flush_list()
            section = line
            blocks.append(line)
            continue

        if _is_list_item_line(line):
            flush_para()
            list_items.append(line)
            continue

        if list_items and _is_list_continuation(line):
            list_items[-1] = f"{list_items[-1]} {line}".strip()
            continue

        if list_items:
            flush_list()

        # Prefer short coherent paragraphs over page-wide merged text.
        if para and para[-1].endswith((".", "!", "?")):
            flush_para()
        para.append(line)

    flush_para()
    flush_list()
    return blocks


def _clean_chunk_text(text: str) -> str:
    if not text:
        return ""
    s = _WS_RE.sub(" ", text.replace("\u00a0", " ")).strip()
    return s


def _split_long_block(block: str, chunk_size: int) -> list[str]:
    text = (block or "").strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    out: list[str] = []
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    cur = ""
    for s in sentences:
        if not cur:
            cur = s
            continue
        if len(cur) + 1 + len(s) <= chunk_size:
            cur += " " + s
        else:
            out.append(cur.strip())
            cur = s
    if cur:
        out.append(cur.strip())
    if out:
        return out

    # Fallback hard split if sentence splitting fails.
    hard: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        part = text[start:end].strip()
        if part:
            hard.append(part)
        if end == len(text):
            break
        start = end
    return hard


def _chunk_blocks(blocks: list[str], chunk_size: int = 900, min_chunk_size: int = 220) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    def flush() -> None:
        nonlocal current_len
        if not current:
            return
        chunks.append("\n".join(current).strip())
        current.clear()
        current_len = 0

    for block in blocks:
        if not block:
            continue
        sub_blocks = _split_long_block(block, chunk_size=max(320, chunk_size))
        for sub in sub_blocks:
            add_len = len(sub) if not current else (1 + len(sub))
            if current and current_len + add_len > chunk_size:
                flush()
            current.append(sub)
            current_len += add_len
    flush()

    # Merge tiny tails to avoid brittle short chunks.
    merged: list[str] = []
    for c in chunks:
        if merged and len(c) < min_chunk_size:
            merged[-1] = f"{merged[-1]}\n{c}".strip()
        else:
            merged.append(c)
    return merged



def chunk_text(text: str, chunk_size: int = 900) -> list[str]:
    if not text:
        return []
    blocks = [b.strip() for b in re.split(r"\n\s*\n+", text) if b.strip()]
    return _chunk_blocks(blocks, chunk_size=chunk_size)




def is_low_value(text: str) -> bool:
    if not text:
        return True
    t = text.lower()

    # Drop table-heavy fragments
    if re.search(r"\btable\s+\d+\b", t):
        return True

    # Drop "Inspector's Aide Memoir" style table content.
    if "aide memoir" in t or ("inspector" in t and "memoir" in t):
        return True

    # Drop Q9 annex/tool-list noise.
    if "histogram" in t or "pareto" in t or "process capability" in t:
        return True

    if "table of contents" in t:
        return True
    if "................................" in text:
        return True
    if _URL_LINE_RE.search(text):
        return True
    if _looks_like_table_row(text):
        return True
    if re.search(r"\b(?:row|column)\b", t) and re.search(r"\b\d+\b", t):
        return True
    if sum(ch.isdigit() for ch in text) >= 12 and len(re.findall(r"[A-Za-z]{3,}", text)) <= 4:
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
            blocks = _build_page_blocks(filtered_lines)
            if not blocks:
                continue

            for i, c in enumerate(_chunk_blocks(blocks), start=1):
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
