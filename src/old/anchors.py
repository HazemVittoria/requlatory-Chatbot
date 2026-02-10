# src/anchors.py
from __future__ import annotations
import re
from typing import List

_ACRONYM = re.compile(r"\b[A-Z]{2,6}(?:\s*/\s*[A-Z]{2,6})*\b")  # OOS / OOT

def extract_anchor_terms(question: str) -> List[str]:
    q = (question or "").strip()
    low = q.lower()

    anchors: List[str] = []

    # Hard-coded high-value anchors from your golden set themes
    # (keeps it deterministic and avoids NLP dependencies)
    for term in [
        "data integrity",
        "deviation",
        "deviations",
        "investigation",
        "capa",
        "effectiveness",
        "process validation",
        "computerized system",
        "computerized systems",
        "supplier",
        "supplier qualification",
        "training",
        "qualification",
        "regulatory notification",
        "approval",
        "oos",
        "oot",
    ]:
        if term in low:
            anchors.append(term)

    # Capture acronyms like OOS / OOT
    for m in _ACRONYM.findall(q):
        anchors.append(m.lower().replace(" ", ""))  # "OOS/OOT" -> "oos/oot"

    # De-dup, preserve order
    out: List[str] = []
    for a in anchors:
        a = a.strip()
        if a and a not in out:
            out.append(a)
    return out
