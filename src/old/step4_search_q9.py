# src/step4_search_q9.py
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
    """Filter out Table-of-Contents / headings-only / low-information chunks."""
    if not text:
        return True

    t = text.lower()

    # Common TOC patterns
    if "table of contents" in t:
        return True
    if "................................" in text:
        return True

    # If it contains lots of dotted leaders, often TOC-like
    if text.count(".") > 80 and text.count("..") > 10:
        return True

    # Too short tends to be headings only
    if len(text) < 250:
        return True

    return False


def build_chunks(pdf_path: Path) -> list[dict]:
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
                    "source": "ICH",
                    "file": pdf_path.name,
                    "page": page_idx,
                    "chunk_id": f"p{page_idx}_c{i}",
                    "text": c,
                }
            )
    return chunks


def search(chunks: list[dict], query: str, top_k: int = 5) -> list[tuple[float, dict]]:
    texts = [c["text"] for c in chunks]

    # TF-IDF is a simple baseline keyword search with weighting
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(texts)

    q = vectorizer.transform([query])
    sims = cosine_similarity(q, X).flatten()

    ranked_idx = sims.argsort()[::-1][:top_k]
    return [(float(sims[i]), chunks[i]) for i in ranked_idx]


def main():
    pdf_path = Path(r"data\ich\Q9.pdf")  # change if needed
    if not pdf_path.exists():
        raise FileNotFoundError(f"Not found: {pdf_path.resolve()}")

    chunks = build_chunks(pdf_path)
    print(f"Loaded {len(chunks)} filtered chunks from {pdf_path.name}")

    while True:
        query = input("\nAsk a question (or type 'exit'): ").strip()
        if query.lower() == "exit":
            break

        results = search(chunks, query, top_k=5)
        for score, item in results:
            print("\n---")
            print(f"Score: {score:.3f}")
            print(f"Citation: {item['source']} | {item['file']} | page {item['page']} | {item['chunk_id']}")
            print(item["text"][:700])


if __name__ == "__main__":
    main()
