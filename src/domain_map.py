from __future__ import annotations

# Filename-pattern hard map used before keyword fallback rules.
# Keep this list short and high-confidence.
_FILE_DOMAIN_PATTERNS: list[tuple[str, str]] = [
    ("annex11", "DataIntegrity"),
    ("annex 11", "DataIntegrity"),
    ("part 11", "DataIntegrity"),
    ("electronic records", "DataIntegrity"),
    ("q9", "PQS"),
    ("q10", "PQS"),
    ("q12", "PQS"),
    ("q13", "PQS"),
    ("q7", "Production"),
    ("q8", "Validation"),
    ("q11", "Validation"),
    ("annex15", "Validation"),
    ("annex 15", "Validation"),
    ("process validation", "Validation"),
    ("210", "Production"),
    ("211", "Production"),
    ("820", "PQS"),
    ("supplier", "Suppliers"),
    ("inspection", "Inspection"),
    ("bimo", "Inspection"),
    ("gcp", "GCP"),
    ("glp", "GLP"),
    ("pharmacovigilance", "PV"),
]


def mapped_domain_for_file(file_name: str) -> str | None:
    n = (file_name or "").strip().lower().replace("_", " ")
    if not n:
        return None
    for pattern, domain in _FILE_DOMAIN_PATTERNS:
        if pattern in n:
            return domain
    return None
