from __future__ import annotations

import re
from pathlib import Path
from typing import List, Dict, Tuple

from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------- shared utilities ----------

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


# ---------- ingestion ----------

def ingest_folder(folder: Path, source_type: str) -> list[dict]:
    chunks = []
    for pdf_path in folder.glob("*.pdf"):
        reader = PdfReader(str(pdf_path))
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
                        "text": c,
                    }
                )
    return chunks


def build_corpus(base_data: Path) -> list[dict]:
    corpus = []
    corpus += ingest_folder(base_data / "ich", "ICH")
    corpus += ingest_folder(base_data / "fda", "FDA")
    corpus += ingest_folder(base_data / "ema", "EMA")
    return corpus


# ---------- search ----------

def search(corpus: list[dict], query: str, top_k: int = 7) -> list[tuple[float, dict]]:
    texts = [c["text"] for c in corpus]
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(texts)
    q = vectorizer.transform([query])
    sims = cosine_similarity(q, X).flatten()
    ranked_idx = sims.argsort()[::-1][:top_k]
    return [(float(sims[i]), corpus[i]) for i in ranked_idx]


# ---------- output ----------

_line_number_re = re.compile(r"(?:(?<=\s)|^)\d{1,4}(?=\s)")


def prettify(text: str) -> str:
    text = _line_number_re.sub("", text)
    return clean_text(text)


def main():
    base_data = Path("data")
    if not base_data.exists():
        raise FileNotFoundError("data folder not found")

    corpus = build_corpus(base_data)

    print(f"Indexed {len(corpus)} chunks total")
    print("Sources included:")
    print(f"  ICH: {len([c for c in corpus if c['source']=='ICH'])}")
    print(f"  FDA: {len([c for c in corpus if c['source']=='FDA'])}")
    print(f"  EMA: {len([c for c in corpus if c['source']=='EMA'])}")

    while True:
        query = input("\nAsk a question (or type 'exit'): ").strip()
        if query.lower() == "exit":
            break

        results = search(corpus, query, top_k=6)

        print("\n" + "=" * 80)
        print(f"Question: {query}\n")

        for score, item in results:
            print(f"[{score:.3f}] {item['source']} | {item['file']} | page {item['page']} | {item['chunk_id']}")
            print(prettify(item["text"])[:600])
            print("-" * 80)


if __name__ == "__main__":
    main()
