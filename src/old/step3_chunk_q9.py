from pathlib import Path
from pypdf import PdfReader


def clean_text(s: str) -> str:
    s = (s or "").replace("\u00a0", " ")
    # collapse whitespace
    return " ".join(s.split()).strip()


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> list[str]:
    """Character-based chunking with overlap."""
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


def main():
    pdf_path = Path(r"data\ich\Q9.pdf")  # change to your exact filename

    if not pdf_path.exists():
        raise FileNotFoundError(f"Not found: {pdf_path.resolve()}")

    reader = PdfReader(str(pdf_path))
    total_pages = len(reader.pages)

    chunks = []
    for page_idx, page in enumerate(reader.pages, start=1):
        page_text = clean_text(page.extract_text())
        for i, c in enumerate(chunk_text(page_text), start=1):
            chunks.append(
                {
                    "source": "ICH",
                    "file": pdf_path.name,
                    "page": page_idx,
                    "chunk_id": f"p{page_idx}_c{i}",
                    "text": c,
                }
            )

    print(f"File: {pdf_path.name}")
    print(f"Pages: {total_pages}")
    print(f"Chunks: {len(chunks)}")

    # Preview
    for item in chunks[:3]:
        print("\n---")
        print(f"{item['source']} | {item['file']} | page {item['page']} | {item['chunk_id']}")
        print(item["text"][:600])


if __name__ == "__main__":
    main()