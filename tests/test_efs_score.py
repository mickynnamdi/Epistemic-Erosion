import sys
from pathlib import Path

"""
Smoke test: EFS formula behaves as expected on hand-crafted examples.
"""
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_efs_perfect_preservation():
    """A source-self pair should yield EFS = 1.0 (every hedge preserved)."""
    from src.efs import compute_efs

    text = (
        "MRI shows an acute right frontal infarct, possibly embolic in origin. "
        "A cardioembolic source cannot be excluded."
    )
    result = compute_efs(text, text, domain="clinical")
    assert result["efs_score"] == 1.0, result


def test_efs_zero_when_all_hedges_dropped():
    """If the output replaces hedged claims with an unrelated unhedged sentence,
    every source hedge is omitted and EFS clips to 0."""
    from src.efs import compute_efs

    source = (
        "Likely pneumonia in the right lower lobe. "
        "Possible early sepsis. "
        "Cannot rule out pulmonary embolism."
    )
    output = "Normal chest CT."
    result = compute_efs(source, output, domain="clinical")
    assert result["efs_score"] == 0.0, result


def test_worked_example_of_figure_a2():
    """Appendix R, Figure A2: 7 source markers, 2 omissions + 1 escalation, EFS = 1 - 3/7 = 0.571."""
    from src.efs import compute_efs

    source = (
        "Chest CT showed an acute right lower lobe consolidation, likely secondary to aspiration. "
        "Another possibility is organizing pneumonia given the subacute course, although less likely "
        "given the imaging findings. It is possible that this represents a post-obstructive pneumonia "
        "from an endobronchial lesion. It is likely that the pleural effusion was a consequence of the "
        "pneumonia. Pulmonology was consulted for evaluation of possible underlying interstitial lung disease."
    )
    output = (
        "+ Chest CT: showed an acute right lower lobe consolidation, likely secondary to aspiration.\n"
        "* Pulmonology was consulted for evaluation of possible underlying interstitial lung disease (ILD).\n"
        "* Working Diagnoses:\n+ Aspiration pneumonia\n+ Post-obstructive pneumonia from an endobronchial lesion."
    )
    r = compute_efs(source, output, domain="clinical")
    assert r["source_hedges"] == 7
    assert r["by_type"] == {"omission": 2, "escalation": 1}
    assert round(r["efs_score"], 3) == 0.571
