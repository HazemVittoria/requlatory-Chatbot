from __future__ import annotations

import re
from pathlib import Path
from typing import List, Dict

from pypdf import PdfReader


def clean_text(s: str) -> str:
    s = (s or "").replace("\u00a0", " ")
    return " ".join(s.split()).strip()


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> list[str]:
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks


def is_low_value(text: str) -> bool:
    if not text:
        return True
    t = text.lower()

    if "table of contents" in t:
        return True
    if "................................" in text:
        return True

    if len(text) < 60:
        return not bool(re.search(r"\b(is|are|means|defined as|refers to)\b", t))

    return False


def ingest_folder(folder: Path, source: str) -> list[dict]:
    chunks: list[dict] = []
    if not folder.exists():
        return chunks

    for pdf_path in folder.glob("*.pdf"):
        reader = PdfReader(str(pdf_path))
        for page_idx, page in enumerate(reader.pages, start=1):
            page_text = clean_text(page.extract_text())
            if not page_text:
                continue

            for i, c in enumerate(chunk_text(page_text), start=1):
                c = clean_text(c)
                if is_low_value(c):
                    continue
                chunks.append(
                    {
                        "source": source,
                        "file": pdf_path.name,
                        "page": page_idx,
                        "chunk_id": f"p{page_idx}_c{i}",
                        "text": c,
                    }
                )
    return chunks


def build_corpus(data_dir: Path) -> list[dict]:
    corpus: list[dict] = []
    corpus += ingest_folder(data_dir / "ich", "ICH")
    corpus += ingest_folder(data_dir / "fda", "FDA")
    corpus += ingest_folder(data_dir / "ema", "EMA")
    if (data_dir / "sops").exists():
        corpus += ingest_folder(data_dir / "sops", "SOPS")
    return corpus
