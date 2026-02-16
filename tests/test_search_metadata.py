from __future__ import annotations

from src.search import infer_domain_boost, resolve_authority_filter


def test_scope_authority_filter_for_ema_includes_eu_gmp():
    allowed = resolve_authority_filter("EMA", authority_filter=None)
    assert "EMA" in allowed
    assert "EU_GMP" in allowed


def test_explicit_authority_filter_overrides_scope():
    allowed = resolve_authority_filter("MIXED", authority_filter=["fda"])
    assert allowed == {"FDA"}


def test_domain_boost_detects_oos_and_validation_signals():
    boosts = infer_domain_boost("How should OOS be investigated during process validation?")
    assert boosts.get("QC_Lab", 0.0) > 0.0
    assert boosts.get("Validation", 0.0) > 0.0
