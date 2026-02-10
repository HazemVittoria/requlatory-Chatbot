# src/step5_answer_with_citations.py
from __future__ import annotations

import re
from pathlib import Path

from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def clean_text(s: str) -> str:
    s = (s or "").replace("\u00a0", " ")
    return " ".join(s.split()).strip()


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> list[str]:
    if not text:
        return []
    chunks = []
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
    if len(text) < 250:
        return True
    return False


def build_chunks(pdf_path: Path, source_type: str = "ICH") -> list[dict]:
    reader = PdfReader(str(pdf_path))
    chunks = []
    for page_idx, page in enumerate(reader.pages, start=1):
        page_text = clean_text(page.extract_text())
        for i, c in enumerate(chunk_text(page_text), start=1):
            c = clean_text(c)
            if is_low_value(c):
                continue
            chunks.append(
                {
                    "source": source_type,
                    "file": pdf_path.name,
                    "page": page_idx,
                    "chunk_id": f"p{page_idx}_c{i}",
                    "text": c,  # keep original chunk text for traceability
                }
            )
    return chunks


def search(chunks: list[dict], query: str, top_k: int = 5) -> list[tuple[float, dict]]:
    texts = [c["text"] for c in chunks]
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(texts)
    q = vectorizer.transform([query])
    sims = cosine_similarity(q, X).flatten()
    ranked_idx = sims.argsort()[::-1][:top_k]
    return [(float(sims[i]), chunks[i]) for i in ranked_idx]


_line_number_re = re.compile(r"(?:(?<=\s)|^)\d{1,4}(?=\s)")


def prettify_for_answer(text: str) -> str:
    """
    Clean text only for display in the final answer (does not modify stored chunks):
    - remove standalone line numbers like '75 76 77'
    - normalize whitespace
    """
    t = text
    t = _line_number_re.sub("", t)
    t = clean_text(t)
    return t


def build_answer(query: str, results: list[tuple[float, dict]], use_top_n: int = 3) -> tuple[str, list[str]]:
    """
    Simple extractive synthesis:
    - take top N chunks
    - clean for display
    - present as a concise combined answer
    - return citations list
    """
    picked = results[:use_top_n]
    if not picked:
        return "No relevant text found in the indexed document.", []

    # Combine top chunks (lightly de-duplicated)
    seen = set()
    parts = []
    citations = []
    for score, item in picked:
        pretty = prettify_for_answer(item["text"])
        # crude de-dup: first 120 chars fingerprint
        fp = pretty[:120].lower()
        if fp in seen:
            continue
        seen.add(fp)
        parts.append(pretty)
        citations.append(f"{item['source']} | {item['file']} | page {item['page']} | {item['chunk_id']}")

    answer = (
        f"Question: {query}\n\n"
        f"Answer (from ICH Q9 text):\n"
        + "\n\n".join(parts[:use_top_n])
    )
    return answer, citations


def main():
    pdf_path = Path(r"data\ich\Q9.pdf")  # change if needed
    if not pdf_path.exists():
        raise FileNotFoundError(f"Not found: {pdf_path.resolve()}")

    chunks = build_chunks(pdf_path, source_type="ICH")
    print(f"Indexed {len(chunks)} chunks from {pdf_path.name}")

    while True:
        query = input("\nAsk a question (or type 'exit'): ").strip()
        if query.lower() == "exit":
            break
        if not query:
            continue

        results = search(chunks, query, top_k=6)
        answer, citations = build_answer(query, results, use_top_n=3)

        print("\n" + "=" * 80)
        print(answer)
        print("\nCitations:")
        for c in citations:
            print(f"- {c}")
        print("=" * 80)


if __name__ == "__main__":
    main()
