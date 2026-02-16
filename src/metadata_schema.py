from __future__ import annotations

from typing import Iterable

AUTHORITY_VALUES = (
    "FDA",
    "EU_GMP",
    "EMA",
    "ICH",
    "PIC_S",
    "WHO",
    "SOP",
    "OTHER",
)

DOMAIN_VALUES = (
    "PQS",
    "DataIntegrity",
    "Validation",
    "Production",
    "QC_Lab",
    "Suppliers",
    "Inspection",
    "Distribution",
    "GCP",
    "GLP",
    "PV",
    "Other",
)

DOC_TYPE_VALUES = (
    "Regulation",
    "Guideline",
    "Annex",
    "Q_Guideline",
    "SOP",
    "Policy",
)

JURISDICTION_VALUES = (
    "US",
    "EU",
    "Global",
    "Mixed",
)

METADATA_VERSION = "v1"

_FOLDER_AUTHORITY_MAP = {
    "fda": "FDA",
    "eu_gmp": "EU_GMP",
    "eugmp": "EU_GMP",
    "ema": "EMA",
    "ich": "ICH",
    "pic_s": "PIC_S",
    "pics": "PIC_S",
    "who": "WHO",
    "sop": "SOP",
    "sops": "SOP",
    "other": "OTHER",
    "others": "OTHER",
}

_SOURCE_AUTHORITY_MAP = {
    "FDA": "FDA",
    "EU_GMP": "EU_GMP",
    "EMA": "EMA",
    "ICH": "ICH",
    "PIC_S": "PIC_S",
    "WHO": "WHO",
    "SOP": "SOP",
    "SOPS": "SOP",
    "OTHER": "OTHER",
}

_SCOPE_AUTHORITY_MAP = {
    "FDA": {"FDA"},
    "ICH": {"ICH"},
    # Keep EMA scope inclusive of EU GMP and PIC/S-style guidance.
    "EMA": {"EMA", "EU_GMP", "PIC_S"},
    "SOPS": {"SOP"},
    "MIXED": set(AUTHORITY_VALUES),
}


def _norm(x: str) -> str:
    return (x or "").strip().upper()


def authority_from_folder(folder_name: str) -> str:
    key = (folder_name or "").strip().lower().replace("-", "_")
    return _FOLDER_AUTHORITY_MAP.get(key, "OTHER")


def authority_from_source(source: str) -> str:
    return _SOURCE_AUTHORITY_MAP.get(_norm(source), "OTHER")


def authorities_for_scope(scope: str) -> set[str]:
    return set(_SCOPE_AUTHORITY_MAP.get(_norm(scope), set(AUTHORITY_VALUES)))


def _validate_enum(name: str, value: str, allowed: Iterable[str]) -> None:
    if value not in set(allowed):
        raise ValueError(f"Invalid {name}: {value}")


def validate_chunk_metadata(meta: dict[str, object]) -> None:
    _validate_enum("authority", str(meta.get("authority", "")), AUTHORITY_VALUES)
    _validate_enum("domain", str(meta.get("domain", "")), DOMAIN_VALUES)
    _validate_enum("doc_type", str(meta.get("doc_type", "")), DOC_TYPE_VALUES)
    _validate_enum("jurisdiction", str(meta.get("jurisdiction", "")), JURISDICTION_VALUES)
