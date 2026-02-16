from __future__ import annotations

from src.ingestion import _build_page_blocks, _chunk_blocks, _looks_like_table_row


def test_table_row_detector_flags_common_layouts():
    assert _looks_like_table_row("Parameter   Min   Max")
    assert _looks_like_table_row("pH   6.8   7.2")
    assert _looks_like_table_row("Column A | Column B | Column C")
    assert not _looks_like_table_row("This paragraph describes how deviations are investigated.")


def test_build_page_blocks_keeps_section_context_and_list_items():
    lines = [
        "5.2 Deviation Handling",
        "- Open investigation record",
        "- Identify root cause and immediate containment",
        "CAPA actions shall be documented and approved by QA.",
    ]
    blocks = _build_page_blocks(lines)
    joined = "\n".join(blocks)

    assert "5.2 Deviation Handling" in joined
    assert "- Open investigation record" in joined
    assert "CAPA actions shall be documented and approved by QA." in joined


def test_build_page_blocks_drops_table_fragments_but_keeps_sections():
    lines = [
        "Table 1: Limits",
        "Parameter   Min   Max",
        "pH   6.8   7.2",
        "3.1 Scope",
        "This section defines handling of process deviations and investigations.",
    ]
    blocks = _build_page_blocks(lines)
    joined = "\n".join(blocks).lower()

    assert "parameter   min   max" not in joined
    assert "pH   6.8   7.2".lower() not in joined
    assert "3.1 scope" in joined
    assert "deviations and investigations" in joined


def test_chunk_blocks_respects_structured_boundaries():
    blocks = [
        "1. Scope\nThis section defines application range and responsibilities.",
        "2. Responsibilities\n- QA reviews records.\n- QC performs testing.",
        "3. Procedure\nStep 1: Review data.\nStep 2: Approve outcomes.",
    ]
    chunks = _chunk_blocks(blocks, chunk_size=120, min_chunk_size=30)

    assert len(chunks) >= 2
    assert any("1. Scope" in c for c in chunks)
    assert any("2. Responsibilities" in c for c in chunks)
    assert any("3. Procedure" in c for c in chunks)
