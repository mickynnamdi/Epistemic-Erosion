import sys
from pathlib import Path
"""
End-to-end distortion detection on synthetic source/output pairs.
"""
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_escalation_is_detected():
    """'suggestive of' (strength 2) -> 'confirms' (strength 4) is an escalation."""
    from src.efs import compute_efs

    source = "CT chest shows findings suggestive of pneumonia in the right lower lobe."
    output = "CT chest confirms pneumonia in the right lower lobe."
    result = compute_efs(source, output, domain="clinical")
    types = [d.distortion_type for d in result["distortions"]]
    assert "escalation" in types, f"expected escalation, got {types}"


def test_severity_warp_is_detected():
    """'mild' (sev 1) -> 'significant' (sev 3) is a severity warp."""
    from src.efs import compute_efs

    source = "Echocardiogram shows mild aortic stenosis."
    output = "Echocardiogram shows significant aortic stenosis."
    result = compute_efs(source, output, domain="clinical")
    types = [d.distortion_type for d in result["distortions"]]
    assert "severity_warp" in types, f"expected severity_warp, got {types}"


def test_omission_is_detected():
    """A hedged source claim absent from the output is an omission."""
    from src.efs import compute_efs

    source = (
        "Worsening renal function is likely due to obstruction of the left ureter. "
        "CT chest shows possible pneumonia."
    )
    output = "CT chest shows possible pneumonia."
    result = compute_efs(source, output, domain="clinical")
    types = [d.distortion_type for d in result["distortions"]]
    assert "omission" in types, f"expected omission, got {types}"
