from __future__ import annotations

import re

from .domain_map import mapped_domain_for_file
from .metadata_schema import (
    AUTHORITY_VALUES,
    DOC_TYPE_VALUES,
    DOMAIN_VALUES,
    JURISDICTION_VALUES,
    METADATA_VERSION,
    validate_chunk_metadata,
)


def infer_doc_type(file_name: str) -> str:
    n = (file_name or "").strip().lower()
    if "annex" in n:
        return "Annex"
    if re.search(r"\bq\d+\b", n):
        return "Q_Guideline"
    if "21 cfr" in n or "cfr" in n or "regulation" in n:
        return "Regulation"
    if "sop" in n or "standard operating procedure" in n:
        return "SOP"
    if "policy" in n:
        return "Policy"
    return "Guideline"


def infer_jurisdiction(authority: str) -> str:
    a = (authority or "").strip().upper()
    if a == "FDA":
        return "US"
    if a in {"EMA", "EU_GMP"}:
        return "EU"
    if a in {"ICH", "PIC_S", "WHO"}:
        return "Global"
    return "Mixed"


def extract_effective_date(file_name: str) -> str | None:
    n = (file_name or "").strip()
    if not n:
        return None

    m = re.search(r"\b(20\d{2})[-_](0[1-9]|1[0-2])[-_](0[1-9]|[12]\d|3[01])\b", n)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"

    m = re.search(r"\b(0[1-9]|[12]\d|3[01])[-_](0[1-9]|1[0-2])[-_](20\d{2})\b", n)
    if m:
        return f"{m.group(3)}-{m.group(2)}-{m.group(1)}"
    return None


_DOMAIN_KEYWORDS: dict[str, tuple[str, ...]] = {
    "PQS": ("quality system", "quality management", "q10", "deviation", "capa"),
    "DataIntegrity": ("data integrity", "audit trail", "electronic", "alcoa", "part 11", "annex 11"),
    "Validation": ("validation", "validated", "qualification", "iq", "oq", "pq", "annex 15"),
    "Production": ("manufacturing", "production", "batch", "sterile", "annex 1"),
    "QC_Lab": ("laboratory", "analytical", "testing", "oos", "oot", "coa"),
    "Suppliers": ("supplier", "vendor", "contractor", "outsourced"),
    "Inspection": ("inspection", "483", "inspectional", "findings", "bimo"),
    "Distribution": ("distribution", "transport", "storage", "cold chain"),
    "GCP": ("clinical", "gcp", "investigator", "trial"),
    "GLP": ("glp", "nonclinical", "toxicology"),
    "PV": ("pharmacovigilance", "adverse event", "signal detection"),
}


def infer_domain(file_name: str, sample_text: str = "") -> str:
    mapped = mapped_domain_for_file(file_name)
    if mapped:
        return mapped

    hay = f"{file_name or ''}\n{sample_text or ''}".lower()
    best_domain = "Other"
    best_score = 0
    for domain, kws in _DOMAIN_KEYWORDS.items():
        score = sum(1 for kw in kws if kw in hay)
        if score > best_score:
            best_score = score
            best_domain = domain
    return best_domain if best_score > 0 else "Other"


def infer_tags(file_name: str, sample_text: str = "") -> list[str]:
    hay = f"{file_name or ''}\n{sample_text or ''}".lower()
    out: list[str] = []
    candidates = (
        "annex 11",
        "annex 15",
        "part 11",
        "data integrity",
        "process validation",
        "computerized systems",
        "risk management",
        "inspection",
        "supplier qualification",
        "oos",
        "oot",
        "capa",
    )
    for tag in candidates:
        if tag in hay:
            out.append(tag)
    return out[:8]


def build_document_metadata(file_name: str, authority: str, sample_text: str = "") -> dict[str, object]:
    authority_norm = (authority or "").strip().upper()
    if authority_norm not in AUTHORITY_VALUES:
        authority_norm = "OTHER"

    doc_type = infer_doc_type(file_name)
    if doc_type not in DOC_TYPE_VALUES:
        doc_type = "Guideline"

    domain = infer_domain(file_name, sample_text=sample_text)
    if domain not in DOMAIN_VALUES:
        domain = "Other"

    jurisdiction = infer_jurisdiction(authority_norm)
    if jurisdiction not in JURISDICTION_VALUES:
        jurisdiction = "Mixed"

    meta: dict[str, object] = {
        "metadata_version": METADATA_VERSION,
        "authority": authority_norm,
        "domain": domain,
        "doc_type": doc_type,
        "jurisdiction": jurisdiction,
        "effective_date": extract_effective_date(file_name),
        "tags": infer_tags(file_name, sample_text=sample_text),
    }
    validate_chunk_metadata(meta)
    return meta
