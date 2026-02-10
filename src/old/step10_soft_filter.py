from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional

from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# Configuration
# ============================================================

DATA_DIR = Path("data")

FOLDERS = {
    "ICH": DATA_DIR / "ich",
    "FDA": DATA_DIR / "fda",
    "EMA": DATA_DIR / "ema",
    "SOP": DATA_DIR / "sops",
    "OTHER": DATA_DIR / "others",
}

# Chunking
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150

# Retrieval
TOP_K = 40
POOL_MULTIPLIER = 6  # retrieve TOP_K * POOL_MULTIPLIER before reranking

# Synthesis
MAX_PROCEDURAL_ACTIONS = 7
MAX_CONCEPT_SENTENCES = 4


# ============================================================
# Data model
# ============================================================

@dataclass(frozen=True)
class Chunk:
    source: str
    file: str
    page: int
    chunk_id: str
    text: str


# ============================================================
# Text utilities
# ============================================================

_line_number_re = re.compile(r"(?:(?<=\s)|^)\d{1,4}(?=\s)")

def clean_text(s: str) -> str:
    s = (s or "").replace("\u00a0", " ")
    return " ".join(s.split()).strip()

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    if not text:
        return []
    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        c = text[start:end].strip()
        if c:
            chunks.append(c)
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
    # keep conservative but not too aggressive
    if len(text) < 120:
        return True
    return False

def prettify(text: str) -> str:
    text = _line_number_re.sub("", text)

    # remove common header/title lines
    text = re.sub(
        r"(Guideline\s*on\s*the\s*requirements.*?Rev\.?|EMA/CHMP/QWP/\d+/\d+\s*Rev\.?|Page\s+\d+/\d+)",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # fix broken hyphenation
    text = re.sub(r"(\w+)\s*-\s*(\w+)", r"\1-\2", text)

    # add spaces after punctuation if missing
    text = re.sub(r"([.,;:])([A-Za-z])", r"\1 \2", text)

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

    return clean_text(text)

def split_sentences(text: str) -> List[str]:
    # tiny heuristic sentence splitter
    text = text.replace("e.g.,", "e_g_")
    parts = re.split(r"(?<=[.!?])\s+", text)
    parts = [p.replace("e_g_", "e.g.,").strip() for p in parts]
    return [p for p in parts if len(p) >= 35]


# ============================================================
# Query understanding (SOFT bias only)
# ============================================================

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
            "inspect",
            "investigate",
        ]
    )

def detect_intent_soft(query: str) -> str:
    """
    Returns a *hint* (ICH/FDA/EMA/SOP/OTHER). This must NOT be used to hard filter.
    """
    q = query.lower()

    if any(k in q for k in ["sop", "work instruction", "our procedure", "internal procedure", "internal"]):
        return "SOP"

    # explicit regulator mention
    if "fda" in q or "dqst" in q or "inspection" in q:
        return "FDA"
    if "ema" in q or "impd" in q or "clinical trial" in q or "investigational" in q:
        return "EMA"
    if "ich" in q or "q8" in q or "q9" in q or "q10" in q:
        return "ICH"

    # procedural default hint
    if is_procedural_query(query):
        return "SOP"

    # everything else: unknown topic, likely "OTHER" (e.g., chemistry lecture)
    return "OTHER"

def expand_query(query: str) -> str:
    q = query.strip()
    ql = q.lower()
    extras: List[str] = []
    if "batch" in ql and ("analyses" in ql or "analysis" in ql):
        extras += [
            "batch analysis",
            "analysis of batches",
            "sampling and testing",
            "test results",
            "COA",
            "certificate of analysis",
        ]
    return q + (" " + " ".join(extras) if extras else "")


# ============================================================
# Ingestion
# ============================================================

def extract_pdf_text(pdf_path: Path) -> Tuple[List[Tuple[int, str]], bool]:
    """
    Returns list of (page_number, text) and a flag indicating likely scanned/no-text PDF.
    """
    reader = PdfReader(str(pdf_path))
    pages: List[Tuple[int, str]] = []
    total_chars = 0

    for page_idx, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        txt = clean_text(raw)
        if txt:
            total_chars += len(txt)
        pages.append((page_idx, txt))

    likely_scanned = total_chars < 300  # heuristic
    return pages, likely_scanned

def ingest_folder(folder: Path, source: str) -> List[Chunk]:
    chunks: List[Chunk] = []
    if not folder.exists():
        return chunks

    for pdf_path in sorted(folder.glob("*.pdf")):
        pages, likely_scanned = extract_pdf_text(pdf_path)
        if likely_scanned:
            # You can add OCR here later; for now, warn clearly.
            print(f"[WARN] Likely scanned/no-text PDF (needs OCR): {source}/{pdf_path.name}")

        for page_idx, page_text in pages:
            if not page_text:
                continue
            for i, c in enumerate(chunk_text(page_text), start=1):
                c = clean_text(c)
                
                if is_low_value(c):
                    continue
                if looks_like_outline(c):
                    continue

                chunks.append(
                    Chunk(
                        source=source,
                        file=pdf_path.name,
                        page=page_idx,
                        chunk_id=f"p{page_idx}_c{i}",
                        text=c,
                    )
                )
    return chunks

def build_corpus(folders: Dict[str, Path]) -> List[Chunk]:
    corpus: List[Chunk] = []
    for source, folder in folders.items():
        corpus.extend(ingest_folder(folder, source))
    return corpus


# ============================================================
# Indexing + Retrieval (global search + soft biases)
# ============================================================

def build_index(corpus: List[Chunk]) -> Tuple[TfidfVectorizer, any]:
    texts = [c.text for c in corpus]
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(texts)
    return vectorizer, X

def source_priors(query: str, hint: str) -> Dict[str, float]:
    """
    Small score adjustments. No hard filtering.
    """
    priors = {"ICH": 0.0, "FDA": 0.0, "EMA": 0.0, "SOP": 0.0, "OTHER": 0.0}

    # soft hint
    if hint in priors:
        priors[hint] += 0.10

    # procedural boost
    if is_procedural_query(query):
        priors["SOP"] += 0.16
        priors["FDA"] += 0.11
        priors["EMA"] += 0.07

    # common conceptual bias (mild) toward ICH only when query is about quality systems/risk/etc.
    q = query.lower()
    if any(k in q for k in ["quality risk", "risk management", "qrm", "pharmaceutical quality", "control strategy"]):
        priors["ICH"] += 0.08

    return priors

def retrieve(
    corpus: List[Chunk],
    vectorizer: TfidfVectorizer,
    X,
    query: str,
    hint: str,
    top_k: int = TOP_K,
) -> List[Tuple[float, Chunk]]:
    q2 = expand_query(query)
    qv = vectorizer.transform([q2])
    sims = cosine_similarity(qv, X).flatten()

    pool_k = min(len(corpus), max(top_k * POOL_MULTIPLIER, 200))
    idx = sims.argsort()[::-1][:pool_k]

    priors = source_priors(query, hint)

    rescored: List[Tuple[float, Chunk]] = []
    for i in idx:
        item = corpus[i]
        score = float(sims[i]) + priors.get(item.source, 0.0)
        rescored.append((score, item))

    rescored.sort(key=lambda x: x[0], reverse=True)
    return rescored[:top_k]


# ============================================================
# Answer synthesis (procedural vs conceptual)
# ============================================================

ACTION_VERBS = (
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
    "ensure",
    "confirm",
    "record",
    "evaluate",
    "assess",
    "establish",
)

def looks_like_outline(text: str) -> bool:
    t = text.strip()
    if len(t) < 200:
        return False
    # many section-number patterns like "4.1.2." in short span
    hits = len(re.findall(r"\b\d+\.\d+(\.\d+)?\b", t))
    if hits >= 8:
        return True
    # many semicolon-like separators / very short “sentences”
    if sum(1 for s in re.split(r"[.!?]", t) if 0 < len(s.strip()) < 35) >= 8:
        return True
    return False


def sentence_score(sentence: str, query: str, source: str) -> float:
    s = sentence.lower()
    q = query.lower()

    score = 0.0
    procedural = is_procedural_query(query)

    if procedural:
        for kw in ["batch", "analysis", "analytical", "testing", "test", "sample", "sampling",
                   "specification", "results", "certificate", "coa", "deviation", "oos", "oot"]:
            if kw in s:
                score += 0.6

        if source == "SOP":
            score += 2.8
        elif source == "FDA":
            score += 2.2
        elif source == "EMA":
            score += 1.6
        elif source == "ICH":
            score += 0.4
        else:
            score += 0.8
    else:
        # conceptual: no regulator bias; keep it small
        if source == "ICH" and any(k in q for k in ["risk", "quality", "control strategy", "validation"]):
            score += 0.4

    # query overlap
    for term in re.findall(r"[a-z]{4,}", q):
        if term in s:
            score += 0.12

    # penalty for obvious artifacts
    if "ema/chmp/qwp" in s or s.startswith("page "):
        score -= 1.0

    return score

def format_citation(c: Chunk) -> str:
    return f"{c.source} | {c.file} | page {c.page} | {c.chunk_id}"

def build_answer(query: str, retrieved: List[Tuple[float, Chunk]]) -> Tuple[str, List[str]]:
    procedural = is_procedural_query(query)

    # gather candidate sentences from top chunks
    sent_candidates: List[Tuple[float, str, Chunk]] = []
    for base_score, chunk in retrieved[: min(len(retrieved), 25)]:
        text = prettify(chunk.text)
        for sent in split_sentences(text):
            s = sent.strip()
            if not s:
                continue

            if procedural:
                # procedural mode: require action-ish language
                sl = s.lower()
                if not any(v in sl for v in ACTION_VERBS):
                    continue

            sc = sentence_score(s, query, chunk.source)
            if sc <= (1.0 if procedural else 0.4):
                continue

            sent_candidates.append((sc + (0.15 * base_score), s, chunk))

    # fallback: if nothing matched, use top chunk text (prettified)
    if not sent_candidates:
        if not retrieved:
            return ("No indexed content matched the question. Check PDF extraction/OCR.", [])
        top_chunk = retrieved[0][1]
        return (prettify(top_chunk.text), [format_citation(top_chunk)])

    # sort and pick best, dedupe by prefix
    sent_candidates.sort(key=lambda x: x[0], reverse=True)
    seen = set()
    picked: List[Tuple[str, Chunk]] = []
    for _, s, ch in sent_candidates:
        key = s[:140].lower()
        if key in seen:
            continue
        seen.add(key)
        picked.append((s, ch))
        if procedural and len(picked) >= MAX_PROCEDURAL_ACTIONS:
            break
        if (not procedural) and len(picked) >= MAX_CONCEPT_SENTENCES:
            break

    citations: List[str] = []
    for _, ch in picked:
        cit = format_citation(ch)
        if cit not in citations:
            citations.append(cit)
        if len(citations) >= 5:
            break

    if procedural:
        answer = "Key actions (from indexed documents):\n" + "\n".join(f"- {s}" for s, _ in picked)
    else:
        # conceptual/explanatory summary bullets
        answer = "Key points (from indexed documents):\n" + "\n".join(f"- {s}" for s, _ in picked)

    return answer, citations


# ============================================================
# Main
# ============================================================

def print_counts(corpus: List[Chunk]) -> None:
    total = len(corpus)
    by_source: Dict[str, int] = {}
    for c in corpus:
        by_source[c.source] = by_source.get(c.source, 0) + 1

    print(f"Indexed {total} chunks")
    for k in ["ICH", "FDA", "EMA", "SOP", "OTHER"]:
        print(f"  {k} chunks: {by_source.get(k, 0)}")

    if by_source.get("OTHER", 0) == 0 and FOLDERS.get("OTHER", Path()).exists():
        print("[NOTE] OTHER chunks are 0. If you added PDFs there, they likely need OCR (scanned).")

def main() -> None:
    corpus = build_corpus(FOLDERS)
    print_counts(corpus)

    if not corpus:
        print("No chunks indexed. Check your data folders and PDFs.")
        return

    vectorizer, X = build_index(corpus)

    while True:
        query = input("\nAsk a question (or type 'exit'): ").strip()
        if not query:
            continue
        if query.lower() == "exit":
            break

        hint = detect_intent_soft(query)
        retrieved = retrieve(corpus, vectorizer, X, query, hint=hint, top_k=TOP_K)
        answer, citations = build_answer(query, retrieved)

        print("\n" + "=" * 90)
        print(f"Question: {query}")
        print(f"Intent hint (soft): {hint}")
        print("\nAnswer:")
        print(answer)
        print("\nCitations:")
        for c in citations:
            print(f"- {c}")
        print("=" * 90)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(0)
