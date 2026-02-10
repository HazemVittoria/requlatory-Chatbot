from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================
# Core utilities
# =========================

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
    """
    Keep this conservative, but not too aggressive.
    FDA/EMA procedural documents often have shorter meaningful paragraphs.
    """
    if not text:
        return True
    t = text.lower()
    if "table of contents" in t:
        return True
    if "................................" in text:
        return True
    # lowered from 250 -> 140 to avoid dropping shorter-but-meaningful content
    if len(text) < 140:
        return True
    return False


def is_procedural_query(query: str) -> bool:
    q = query.lower()
    return any(
        kw in q
        for kw in [
            "how to",
            "how should",
            "carry out",
            "perform",
            "conduct",
            "procedure",
            "sampling",
            "testing",
            "analysis",
            "batch",
        ]
    )


# =========================
# Ingestion
# =========================

def ingest_folder(folder: Path, source: str) -> list[dict]:
    chunks = []
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
                        "text": c,  # preserved for audit
                    }
                )
    return chunks


def build_corpus(data_dir: Path) -> list[dict]:
    corpus = []
    corpus += ingest_folder(data_dir / "ich", "ICH")
    corpus += ingest_folder(data_dir / "fda", "FDA")
    corpus += ingest_folder(data_dir / "ema", "EMA")
    return corpus


# =========================
# Search
# =========================

def expand_query(query: str) -> str:
    """
    Lightweight query expansion for better matching:
    'batch analyses' often appears as 'batch analysis', 'analysis of batches', etc.
    """
    q = query.strip()
    ql = q.lower()
    extras = []
    if "batch" in ql and ("analyses" in ql or "analysis" in ql):
        extras += ["batch analysis", "analysis of batches", "sampling and testing", "test results", "COA", "certificate of analysis"]
    return q + " " + " ".join(extras)


def search(corpus: list[dict], query: str, top_k: int = 30) -> list[tuple[float, dict]]:
    texts = [c["text"] for c in corpus]
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(texts)

    q2 = expand_query(query)
    qv = vectorizer.transform([q2])
    sims = cosine_similarity(qv, X).flatten()

    # return more candidates; Step 8 will re-rank for answer-building
    idx = sims.argsort()[::-1][:top_k]
    return [(float(sims[i]), corpus[i]) for i in idx]


# =========================
# Step 8: Clean synthesis
# =========================

_line_number_re = re.compile(r"(?:(?<=\s)|^)\d{1,4}(?=\s)")


def prettify(text: str) -> str:
    # remove standalone line numbers
    text = _line_number_re.sub("", text)

    # remove common header/title lines (ICH/FDA/EMA)
    text = re.sub(
        r"(Guideline\s*on\s*the\s*requirements.*?Rev\.?|EMA/CHMP/QWP/\d+/\d+\s*Rev\.?|Page\s+\d+/\d+)",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # fix broken hyphenation: "risk - based" -> "risk-based"
    text = re.sub(r"(\w+)\s*-\s*(\w+)", r"\1-\2", text)

    # add spaces after punctuation if missing (common in extracted PDFs)
    text = re.sub(r"([.,;:])([A-Za-z])", r"\1 \2", text)

    # fix very common missing-space merges: "onthe" -> "on the", "andthe" -> "and the", "itis" -> "it is"
    # (keep list small and safe)
    fixes = {
        "onthe": "on the",
        "andthe": "and the",
        "itis": "it is",
        "itisnecessary": "it is necessary",
        "inthe": "in the",
        "tobe": "to be",
        "forthe": "for the",
        "bythe": "by the",
        "ofthe": "of the",
        "witha": "with a",
        "withthe": "with the",
        "shouldbe": "should be",
        "havebeen": "have been",
    }
    for bad, good in fixes.items():
        text = re.sub(rf"\b{bad}\b", good, text, flags=re.IGNORECASE)

    # normalize whitespace
    return clean_text(text)




def split_sentences(text: str) -> list[str]:
    text = text.replace("e.g.,", "e_g_")
    parts = re.split(r"(?<=[.!?])\s+", text)
    parts = [p.replace("e_g_", "e.g.,").strip() for p in parts]
    return [p for p in parts if len(p) >= 40]


def sentence_score(sentence: str, query: str, source: str) -> float:
    s = sentence.lower()
    q = query.lower()
    score = 0.0

    procedural = is_procedural_query(query)

    # Prefer operational terms for procedural questions
    if procedural:
        for kw in ["batch", "analysis", "analytical", "testing", "test", "sample", "sampling", "specification", "results", "certificate", "coa"]:
            if kw in s:
                score += 0.6

        # Strong source bias for procedural questions
        if source == "FDA":
            score += 2.2
        elif source == "EMA":
            score += 1.8
        elif source == "ICH":
            score += 0.3
    else:
        # Conceptual questions: mild ICH preference
        if source == "ICH":
            score += 0.6
        if "systematic process" in s:
            score += 2.0

    # Query overlap
    for term in re.findall(r"[a-z]{4,}", q):
        if term in s:
            score += 0.15

    # Penalize diagram-like fragments
    if "risk assessment initiate" in s:
        score -= 2.0

    return score


def build_answer(
    query: str,
    results: list[tuple[float, dict]],
    max_actions: int = 6,
) -> tuple[str, list[str]]:
    procedural = is_procedural_query(query)

    # Re-rank: boost FDA/EMA for procedural questions
    rescored = []
    for sim, item in results:
        boost = 0.0
        if procedural:
            if item["source"] == "FDA":
                boost = 0.20
            elif item["source"] == "EMA":
                boost = 0.15
        rescored.append((sim + boost, item))
    rescored.sort(key=lambda x: x[0], reverse=True)

    # Collect candidate action sentences
    actions = []
    citations = []
    seen = set()

    action_verbs = (
        "collect",
        "provide",
        "verify",
        "test",
        "perform",
        "review",
        "retain",
        "document",
        "investigate",
        "report",
        "obtain",
    )

    for sim, item in rescored:
        text = prettify(item["text"])
        for sent in split_sentences(text):
            s = sent.strip()
            s_l = s.lower()

            # DROP header/title lines (EMA/ICH artifacts)
            if s_l.startswith("guideline on the requirements") or "ema/chmp/qwp" in s_l:
                continue
            if s_l.startswith("page ") and "batch" not in s_l:
                continue

            # Action filter
            if not any(v in s_l for v in action_verbs):
                continue

            # De-duplication
            key = s[:140].lower()
            if key in seen:
                continue
            seen.add(key)

            actions.append(s)
            cit = f"{item['source']} | {item['file']} | page {item['page']} | {item['chunk_id']}"
            if cit not in citations:
                citations.append(cit)

            if len(actions) >= max_actions:
                break
        if len(actions) >= max_actions:
            break

    # Fallback if no actions matched
    if not actions and rescored:
        top = rescored[0][1]
        return (
            prettify(top["text"]),
            [f"{top['source']} | {top['file']} | page {top['page']} | {top['chunk_id']}"],
        )

    answer = (
        "Key actions for batch analyses (regulatory guidance):\n"
        + "\n".join(f"- {a}" for a in actions)
    )
    return answer, citations[:4]





# =========================
# Main
# =========================

def main():
    data_dir = Path("data")
    corpus = build_corpus(data_dir)

    # Debug counts by agency (critical for your FDA/EMA issue)
    ich_n = sum(1 for c in corpus if c["source"] == "ICH")
    fda_n = sum(1 for c in corpus if c["source"] == "FDA")
    ema_n = sum(1 for c in corpus if c["source"] == "EMA")

    print(f"Indexed {len(corpus)} chunks across ICH, FDA, EMA")
    print(f"  ICH chunks: {ich_n}")
    print(f"  FDA chunks: {fda_n}")
    print(f"  EMA chunks: {ema_n}")


    if fda_n == 0 or ema_n == 0:
        print("\nNOTE: If FDA or EMA chunks are 0, that PDF likely has no extractable text (scan/OCR).")
        print("      In that case, Step 9 will add OCR fallback for those documents.\n")

    while True:
        query = input("\nAsk a question (or type 'exit'): ").strip()
        if query.lower() == "exit":
            break

        results = search(corpus, query, top_k=40)
        answer, citations = build_answer(query, results)

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
