import sys
from pathlib import Path

"""
Smoke test: hedge lexicons load and detect known terms.
"""
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_clinical_extractor_finds_known_hedges():
    from src.hedge_screener_contextual import find_contextual_hedges

    text = (
        "CT chest shows findings suggestive of pneumonia in the right lower lobe. "
        "Mild stenosis is present in the LAD. "
        "Cannot rule out pulmonary embolism."
    )
    out = find_contextual_hedges(text, domain="clinical", min_claim_signals=0)
    # HedgeMatch is a dataclass with .text and .category
    found = " ".join(h.text.lower() for h in out)
    assert "suggestive of" in found, found
    assert "cannot rule out" in found, found
    assert "mild" in found, found


def test_legal_lexicon_has_5_categories():
    """Matches Appendix A."""
    from src.hedge_screener_legal import LEGAL_HEDGE_PATTERNS

    assert set(LEGAL_HEDGE_PATTERNS.keys()) == {
        "legal_hedge", "evidential_hedge", "standard_hedge",
        "conditional_hedge", "scope_hedge",
    }


def test_scientific_lexicon_has_5_categories():
    """Matches Appendix A."""
    from src.hedge_screener_scientific import SCIENTIFIC_HEDGE_PATTERNS

    assert set(SCIENTIFIC_HEDGE_PATTERNS.keys()) == {
        "results_hedge", "evidential_hedge", "scope_hedge",
        "limitation_hedge", "quantitative_hedge",
    }


def test_epistemic_strength_scale():
    """4-point scale: possible=1, suggestive of=2, likely=3, confirmed=4."""
    from src.distortion_analyzer import get_epistemic_strength

    assert get_epistemic_strength("possible pneumonia") == 1
    assert get_epistemic_strength("suggestive of pneumonia") == 2
    assert get_epistemic_strength("likely pneumonia") == 3
    assert get_epistemic_strength("confirmed pneumonia") == 4
