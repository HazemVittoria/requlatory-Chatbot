from __future__ import annotations

from src.metadata_rules import build_document_metadata, infer_doc_type, infer_domain


def test_doc_type_detection_annex_and_q_guideline():
    assert infer_doc_type("EudraLex_Vol4_Annex15.pdf") == "Annex"
    assert infer_doc_type("Q10.pdf") == "Q_Guideline"


def test_domain_detection_from_filename_and_keywords():
    assert infer_domain("EU_GMP_Annex11.pdf", sample_text="") == "DataIntegrity"
    assert infer_domain(
        "Unmapped Guidance.pdf",
        sample_text="This section covers supplier qualification and vendor oversight.",
    ) == "Suppliers"


def test_build_document_metadata_has_required_fields():
    meta = build_document_metadata(
        "21 CFR Part 11 (up to date as of 2-12-2026).pdf",
        authority="FDA",
        sample_text="Electronic records and audit trails are required.",
    )
    assert meta["authority"] == "FDA"
    assert meta["doc_type"] in {"Regulation", "Guideline"}
    assert meta["domain"] in {"DataIntegrity", "Other"}
    assert meta["jurisdiction"] == "US"
    assert "metadata_version" in meta
