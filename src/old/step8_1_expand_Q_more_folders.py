from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =============================================================================
# Text utilities
# =============================================================================

_WORD_RE = re.compile(r"[a-zA-Z]{2,}")
_LINE_NUMBER_RE = re.compile(r"(?:(?<=\s)|^)\d{1,4}(?=\s)")
_MULTI_DOTS_RE = re.compile(r"\.{10,}")

# Common header/title noise patterns (extend as you discover new ones)
_HEADER_NOISE_RE = re.compile(
    r"(Guideline\s*on\s*the\s*requirements.*?Rev\.?|EMA/CHMP/QWP/\d+/\d+\s*Rev\.?|Page\s+\d+/\d+)",
    flags=re.IGNORECASE,
)


def clean_text(s: str) -> str:
    s = (s or "").replace("\u00a0", " ")
    return " ".join(s.split()).strip()


def prettify(text: str) -> str:
    text = _LINE_NUMBER_RE.sub("", text)
    text = _HEADER_NOISE_RE.sub("", text)
    # fix broken hyphenation: "risk - based" -> "risk-based"
    text = re.sub(r"(\w+)\s*-\s*(\w+)", r"\1-\2", text)
    # add spaces after punctuation if missing
    text = re.sub(r"([.,;:])([A-Za-z])", r"\1 \2", text)
    # small safe merge fixes
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


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> List[str]:
    if not text:
        return []
    chunks: List[str] = []
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

    if _MULTI_DOTS_RE.search(text):
        return True

    # Always drop extremely short text
    if len(text) < 60:
        return True

    # Keep short but meaningful qualification/validation content
    if len(text) < 120:
        if any(k in t for k in ["dq", "iq", "oq", "pq", "qualification", "validation"]):
            return False
        return True

    # Otherwise keep it
    return False



def split_sentences(text: str) -> List[str]:
    # very lightweight splitter; avoid splitting on e.g.
    text = text.replace("e.g.,", "e_g_").replace("i.e.,", "i_e_")
    parts = re.split(r"(?<=[.!?])\s+", text)
    parts = [p.replace("e_g_", "e.g.,").replace("i_e_", "i.e.,").strip() for p in parts]
    # keep moderately informative sentences
    return [p for p in parts if len(p) >= 35]


def normalize_source(folder_name: str) -> str:
    # Map folder names to a stable label
    # "sops" -> "SOPS", "others" -> "OTHERS", etc.
    return folder_name.strip().upper()


# =============================================================================
# Query routing (answer modes)
# =============================================================================

class Mode:
    RESPONSIBILITY = "RESPONSIBILITY"
    PARAMETERS = "PARAMETERS"
    DEFINITION = "DEFINITION"
    PROCEDURE = "PROCEDURE"
    REQUIREMENTS = "REQUIREMENTS"
    GENERAL = "GENERAL"


def detect_mode(query: str) -> str:
    q = query.lower().strip()

    # responsibility / role
    if any(k in q for k in ["who is responsible", "responsible for", "accountable", "ownership", "owner of"]):
        return Mode.RESPONSIBILITY

    # parameters / list / criteria
    if any(k in q for k in ["parameter", "parameters", "validation parameters", "criteria", "acceptance criteria", "attributes", "what are the"]):
        # "what are the" is broad; keep it here only if typical list query terms also appear
        if any(k in q for k in ["parameter", "criteria", "acceptance", "attribute", "validation"]):
            return Mode.PARAMETERS

    # definition
    if any(k in q for k in ["what is", "define", "definition of", "meaning of"]):
        return Mode.DEFINITION

    # procedure / how-to
    if any(k in q for k in ["how to", "how should", "procedure", "steps", "process", "perform", "conduct", "carry out"]):
        return Mode.PROCEDURE

    # requirements
    if any(k in q for k in ["shall", "must", "should", "required", "requirement", "requirements", "need to"]):
        return Mode.REQUIREMENTS

    return Mode.GENERAL


def expand_query(query: str, mode: str) -> str:
    q = query.strip()
    ql = q.lower()
    extras: List[str] = []

    # Validation parameters synonyms
    if mode == Mode.PARAMETERS or ("validation" in ql and "parameter" in ql):
        extras += [
            "specificity", "selectivity", "linearity", "range", "accuracy", "precision",
            "repeatability", "intermediate precision", "reproducibility",
            "detection limit", "quantitation limit", "robustness", "system suitability",
            "acceptance criteria", "acceptance limits"
        ]

    # Responsibilities
    if mode == Mode.RESPONSIBILITY:
        extras += ["responsibilities", "roles", "accountable", "approved by", "review and approval", "owner", "responsible"]

    # Procedures
    if mode == Mode.PROCEDURE:
        extras += ["steps", "workflow", "records", "documentation", "review", "approval", "verification"]

    # Requirements
    if mode == Mode.REQUIREMENTS:
        extras += ["shall", "must", "should", "required", "ensure", "documented"]

    # Common analytical method validation terms
    if "analytical" in ql and "valid" in ql:
        extras += ["validation protocol", "validation report", "method validation", "validation study"]

    return q + " " + " ".join(extras)


# =============================================================================
# Ingestion (all subfolders under data/)
# =============================================================================

@dataclass(frozen=True)
class Chunk:
    source: str   # e.g. ICH, FDA, EMA, SOPS, OTHERS
    folder: str   # original folder name (for audit)
    file: str
    page: int
    chunk_id: str
    text: str     # preserved for audit


def ingest_pdf(pdf_path: Path, source_label: str, folder_name: str) -> List[Chunk]:
    chunks: List[Chunk] = []
    reader = PdfReader(str(pdf_path))

    for page_idx, page in enumerate(reader.pages, start=1):
        page_text = clean_text(page.extract_text() or "")
        if not page_text:
            continue

        for i, c in enumerate(chunk_text(page_text), start=1):
            c = prettify(c)
            if is_low_value(c):
                continue
            chunks.append(
                Chunk(
                    source=source_label,
                    folder=folder_name,
                    file=pdf_path.name,
                    page=page_idx,
                    chunk_id=f"p{page_idx}_c{i}",
                    text=c,
                )
            )
    return chunks


def build_corpus(data_dir: Path) -> List[Chunk]:
    if not data_dir.exists():
        raise FileNotFoundError(f"Data folder not found: {data_dir.resolve()}")

    corpus: List[Chunk] = []
    # Include ema, fda, ich, sops, others, and any additional subfolder the user adds.
    for folder in sorted([p for p in data_dir.iterdir() if p.is_dir()]):
        folder_name = folder.name
        source_label = normalize_source(folder_name)
        for pdf_path in sorted(folder.glob("*.pdf")):
            corpus.extend(ingest_pdf(pdf_path, source_label=source_label, folder_name=folder_name))

    return corpus


# =============================================================================
# Search index (build TF-IDF once; reuse for all queries)
# =============================================================================

class SearchIndex:
    def __init__(self, corpus: List[Chunk]):
        self.corpus = corpus
        self.texts = [c.text for c in corpus]

        # Fit once. NOTE: for mixed regulatory docs + internal SOPs, English stop-words still helps.
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.X = self.vectorizer.fit_transform(self.texts)

    def search(self, query: str, top_k: int = 40) -> List[Tuple[float, Chunk]]:
        mode = detect_mode(query)
        q2 = expand_query(query, mode)
        qv = self.vectorizer.transform([q2])
        sims = cosine_similarity(qv, self.X).flatten()
        idx = sims.argsort()[::-1][:top_k]
        return [(float(sims[i]), self.corpus[i]) for i in idx]


# =============================================================================
# Sentence scoring + answer building
# =============================================================================

def tokenize_query_terms(query: str) -> List[str]:
    q = query.lower()
    # keep terms length >= 4 to reduce noise
    return [t for t in re.findall(r"[a-z]{4,}", q)]


def mode_keywords(mode: str) -> List[str]:
    if mode == Mode.PARAMETERS:
        return [
            "specificity", "selectivity", "linearity", "range", "accuracy", "precision",
            "repeatability", "intermediate precision", "reproducibility",
            "detection limit", "quantitation limit", "robustness", "system suitability",
            "acceptance", "criteria", "limits", "lod", "loq"
        ]
    if mode == Mode.RESPONSIBILITY:
        return ["responsible", "responsibility", "accountable", "shall", "must", "approve", "review", "owner", "role"]
    if mode == Mode.DEFINITION:
        return ["definition", "means", "is defined as", "refers to", "is the", "shall mean"]
    if mode == Mode.PROCEDURE:
        return ["procedure", "step", "perform", "conduct", "document", "record", "verify", "review", "approve"]
    if mode == Mode.REQUIREMENTS:
        return ["shall", "must", "should", "required", "ensure", "documented", "need to"]
    return []


def sentence_score(sentence: str, query: str, mode: str, source: str, chunk_sim: float) -> float:
    s = sentence.lower()
    score = 0.0

    # Base relevance to query terms
    q_terms = tokenize_query_terms(query)
    for term in q_terms:
        if term in s:
            score += 0.18

    # Mode keywords boost
    for kw in mode_keywords(mode):
        if kw in s:
            score += 0.55

    # Prefer normative language for requirements
    if mode in (Mode.REQUIREMENTS, Mode.PROCEDURE, Mode.PARAMETERS, Mode.RESPONSIBILITY):
        if "shall" in s:
            score += 0.45
        if "must" in s:
            score += 0.40
        if "should" in s:
            score += 0.20

    # Penalize sections that often pollute answers when not requested
    if mode != Mode.RESPONSIBILITY and ("responsibilities" in s or s.startswith("responsibilities")):
        score -= 1.0
    if mode != Mode.PROCEDURE and (s.startswith("scope") or "this sop applies" in s):
        score -= 0.9
    if mode != Mode.DEFINITION and (s.startswith("definitions") or s.startswith("definition")):
        score -= 0.4

    # Light source preferences (tune as needed)
    # Keep mild: internal SOPs may be the best answer for operational questions.
    if mode in (Mode.PROCEDURE, Mode.RESPONSIBILITY) and source in {"SOPS", "SOP"}:
        score += 0.25

    # Include chunk similarity as a weak signal
    score += chunk_sim * 0.45

    return score


def build_answer(
    query: str,
    results: List[Tuple[float, Chunk]],
    max_bullets: int = 7,
    min_best_score: float = 1.25,   # answerability threshold (tune)
) -> Tuple[str, List[str]]:
    mode = detect_mode(query)

    # Collect candidate sentences from the top chunks, then score/sort.
    candidates: List[Tuple[float, str, Chunk]] = []
    top_chunks = results[:20]  # limit for speed + precision

    for sim, item in top_chunks:
        text = prettify(item.text)
        for sent in split_sentences(text):
            s = sent.strip()
            s_l = s.lower()

            # Remove obvious header fragments
            if s_l.startswith("guideline on the requirements") or "ema/chmp/qwp" in s_l:
                continue
            if s_l.startswith("page ") and "batch" not in s_l:
                continue

            sc = sentence_score(s, query, mode, item.source, sim)
            candidates.append((sc, s, item))

    if not candidates:
        # fallback: show top chunk if any
        if results:
            top = results[0][1]
            return (
                prettify(top.text),
                [f"{top.source} | {top.file} | page {top.page} | {top.chunk_id}"],
            )
        return ("No documents indexed.", [])

    candidates.sort(key=lambda x: x[0], reverse=True)

    best_score = candidates[0][0]

    # Answerability: if nothing scores well, say "not found" and show closest excerpts.
    if best_score < min_best_score:
        # Provide top 3 closest chunks as excerpts
        excerpts: List[str] = []
        citations: List[str] = []
        for sim, item in results[:3]:
            excerpts.append(f"- {prettify(item.text)[:350]}...")
            citations.append(f"{item.source} | {item.file} | page {item.page} | {item.chunk_id}")

        msg = (
            "Not found as a direct answer in the indexed documents. Closest excerpts:\n"
            + "\n".join(excerpts)
        )
        return msg, citations

    # Build bullets from the best-scoring sentences with de-dup
    bullets: List[str] = []
    citations: List[str] = []
    seen: set[str] = set()

    for sc, s, item in candidates:
        key = s[:160].lower()
        if key in seen:
            continue
        seen.add(key)

        bullets.append(s)
        cit = f"{item.source} | {item.file} | page {item.page} | {item.chunk_id}"
        if cit not in citations:
            citations.append(cit)

        if len(bullets) >= max_bullets:
            break

    # Mode-specific header (neutral, not “batch analyses”)
    if mode == Mode.PARAMETERS:
        header = "Likely validation parameters / criteria mentioned in the documents:"
    elif mode == Mode.RESPONSIBILITY:
        header = "Responsibilities / roles mentioned in the documents:"
    elif mode == Mode.DEFINITION:
        header = "Relevant definitions / meaning in the documents:"
    elif mode == Mode.PROCEDURE:
        header = "Procedure-related guidance excerpts:"
    elif mode == Mode.REQUIREMENTS:
        header = "Requirement-related guidance excerpts:"
    else:
        header = "Key guidance excerpts:"

    answer = header + "\n" + "\n".join(f"- {b}" for b in bullets)
    return answer, citations[:6]


# =============================================================================
# Main
# =============================================================================

def print_counts(corpus: List[Chunk]) -> None:
    counts: Dict[str, int] = {}
    for c in corpus:
        counts[c.source] = counts.get(c.source, 0) + 1

    total = len(corpus)
    print(f"Indexed {total} chunks across folders:")
    for k in sorted(counts.keys()):
        print(f"  {k}: {counts[k]}")

    # OCR hint
    if total == 0:
        print("\nNOTE: No text was extracted. PDFs might be scanned (needs OCR).\n")


def main() -> None:
    data_dir = Path("data")
    corpus = build_corpus(data_dir)
    print_counts(corpus)

    index = SearchIndex(corpus)

    while True:
        query = input("\nAsk a question (or type 'exit'): ").strip()
        if not query:
            continue
        if query.lower() == "exit":
            break

        results = index.search(query, top_k=45)
        answer, citations = build_answer(query, results)

        print("\n" + "=" * 90)
        print(f"Question: {query}\n")
        print("Answer:")
        print(answer)
        print("\nCitations:")
        for c in citations:
            print(f"- {c}")
        print("=" * 90)


if __name__ == "__main__":
    main()
