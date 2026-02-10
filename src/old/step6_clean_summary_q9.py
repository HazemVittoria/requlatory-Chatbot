from __future__ import annotations

import re
from pathlib import Path
from typing import List, Dict, Tuple

from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------- Extraction + chunking (same foundation) ----------

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
                    "text": c,  # keep original for auditability
                }
            )
    return chunks


def search(chunks: list[dict], query: str, top_k: int = 8) -> list[tuple[float, dict]]:
    texts = [c["text"] for c in chunks]
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(texts)
    q = vectorizer.transform([query])
    sims = cosine_similarity(q, X).flatten()
    ranked_idx = sims.argsort()[::-1][:top_k]
    return [(float(sims[i]), chunks[i]) for i in ranked_idx]


# ---------- Step 6: Controlled synthesis (clean summary + citations) ----------

_line_number_re = re.compile(r"(?:(?<=\s)|^)\d{1,4}(?=\s)")


def prettify_for_display(text: str) -> str:
    """Clean only for output (does NOT change stored chunks)."""
    t = _line_number_re.sub("", text)      # remove standalone line numbers
    t = clean_text(t)
    return t


def split_sentences(text: str) -> list[str]:
    """
    Simple sentence splitter (good enough for guidelines).
    Avoids extra dependencies.
    """
    text = text.replace("e.g.,", "e_g_")  # prevent split on e.g.
    parts = re.split(r"(?<=[.!?])\s+", text)
    parts = [p.replace("e_g_", "e.g.,").strip() for p in parts]
    return [p for p in parts if len(p) >= 40]


def sentence_score(sentence: str, query: str) -> float:
    """
    Heuristic: prefer definition-like sentences and those matching the query terms.
    """
    s = sentence.lower()
    q = query.lower()

    score = 0.0

    # Prefer definitional patterns
    if "is a systematic process" in s:
        score += 4.0
    if s.startswith("quality risk management is") or " quality risk management is " in s:
        score += 3.0
    if "definition" in q and (" is " in s):
        score += 1.0

    # Prefer sentences mentioning key Q9 terms
    for kw in ["assessment", "control", "communication", "review", "risk", "quality", "lifecycle"]:
        if kw in s:
            score += 0.3

    # Prefer shorter, cleaner sentences (not lists/diagrams)
    if sentence.count("•") > 0:
        score -= 0.8
    if sentence.count(":") > 2:
        score -= 0.5
    if len(sentence) > 320:
        score -= 0.7

    # Light query term overlap
    for term in re.findall(r"[a-z]{3,}", q):
        if term in s:
            score += 0.05

    return score


def build_clean_answer(
    query: str,
    results: list[tuple[float, dict]],
    max_sentences: int = 3,
) -> tuple[str, list[str]]:
    """
    Build a short, readable answer using high-value sentences from top chunks.
    Return (answer_text, citations_used).
    """
    candidates: List[Tuple[float, str, str]] = []  # (score, sentence, citation)

    for sim, item in results:
        text = prettify_for_display(item["text"])

        # Drop obvious diagram-like lines (common in Q9 figures)
        if "risk assessment initiate" in text.lower():
            # still keep other sentences if any
            pass

        for sent in split_sentences(text):
            # Remove stray diagram fragments
            if "risk assessment initiate quality risk management process" in sent.lower():
                continue

            cit = f"{item['source']} | {item['file']} | page {item['page']} | {item['chunk_id']}"
            sc = sentence_score(sent, query)

            # Keep only meaningful candidates
            if sc >= 1.0:
                candidates.append((sc, sent, cit))

    if not candidates:
        # fallback: show the best chunk (cleaned), still citeable
        top = results[0][1]
        fallback = prettify_for_display(top["text"])
        citation = f"{top['source']} | {top['file']} | page {top['page']} | {top['chunk_id']}"
        answer = f"{fallback}"
        return answer, [citation]

    # Pick top sentences with de-duplication
    candidates.sort(key=lambda x: x[0], reverse=True)

    chosen: List[Tuple[str, str]] = []
    seen = set()
    for sc, sent, cit in candidates:
        key = sent[:120].lower()
        if key in seen:
            continue
        seen.add(key)
        chosen.append((sent, cit))
        if len(chosen) >= max_sentences:
            break

    answer_text = " ".join([s for s, _ in chosen]).strip()

    citations = []
    for _, cit in chosen:
        if cit not in citations:
            citations.append(cit)

    return answer_text, citations


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

        results = search(chunks, query, top_k=10)
        answer, citations = build_clean_answer(query, results, max_sentences=3)

        print("\n" + "=" * 80)
        print(f"Question: {query}\n")
        print("Answer:")
        print(answer)
        print("\nCitations:")
        for c in citations:
            print(f"- {c}")
        print("=" * 80)


if __name__ == "__main__":
    main()
