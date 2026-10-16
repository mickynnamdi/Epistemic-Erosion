"""Tests of the one-to-one aligner, the scoring entry point, and (when present) your scored runs.

Always run: aligner behaviour, threshold option, domain validation.
Run once the public samples are rebuilt (scripts/data/): identity EFS(s, s) = 1.0 on every
legal, scientific and PubMedQA document, and the greedy-aligner identity defect of Appendix J.
Run once scripts/score_generations.py has produced results/scores/: the paper's headline values.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from configs.paths import DATA_DIR, RESULTS_DIR  # noqa: E402

TRUNC = 12000
TYPES = ["omission", "escalation", "attenuation", "fabrication", "scope_shift", "severity_warp"]


def _need(path):
    if not Path(path).exists():
        pytest.skip(f"{Path(path).name} not present (see docs/REPRODUCE.md)")
    return path


def _scores(study):
    df = pd.read_csv(_need(RESULTS_DIR / "scores" / f"{study}.csv"), low_memory=False)
    return df[df.status == "ok"]


@pytest.mark.parametrize("parquet,domain", [
    ("legal_sample.parquet", "legal"),
    ("scientific_sample.parquet", "scientific"),
    ("medical_sample.parquet", "clinical"),   # PubMedQA abstracts through the clinical pipeline
])
def test_identity_holds_on_real_public_documents(parquet, domain):
    """EFS(s, s) == 1.0 and zero distortions for every public source document."""
    from src.efs import compute_efs
    docs = pd.read_parquet(_need(DATA_DIR / parquet)).source_text.astype(str).tolist()
    assert len(docs) >= 192
    bad = [(i, r["efs_score"]) for i, t in enumerate(docs)
           for r in [compute_efs(t[:TRUNC], t[:TRUNC], domain=domain)]
           if r["efs_score"] != 1.0 or r["distortion_count"] != 0]
    assert not bad, f"{domain}: {len(bad)}/{len(docs)} documents violate identity: {bad[:5]}"


def test_aligner_omission_below_threshold_near_duplicate_matched_one_to_one():
    from src.claim_aligner import MATCH_THRESHOLD, align_claims, score_pair

    assert MATCH_THRESHOLD == 0.30
    src = [
        {"text": "Worsening renal function is likely due to obstruction of the left ureter.",
         "clinical_terms": ["renal function", "ureter"]},
        {"text": "CT chest shows findings suggestive of pneumonia in the right lower lobe.",
         "clinical_terms": ["pneumonia", "right lower lobe"]},
        {"text": "Findings are suggestive of pneumonia in the right lower lobe on CT chest.",
         "clinical_terms": ["pneumonia", "right lower lobe"]},
    ]
    out = [
        {"text": "CT chest shows findings suggestive of pneumonia in the right lower lobe",   # near-duplicate
         "clinical_terms": ["pneumonia", "right lower lobe"]},
        {"text": "The patient was discharged home in stable condition with follow-up.",
         "clinical_terms": []},
    ]
    a = align_claims(src, out)
    assert score_pair(src[0], out[1]) < MATCH_THRESHOLD and score_pair(src[0], out[0]) < MATCH_THRESHOLD
    assert a[0] is None                                   # omission: no counterpart above tau
    assert score_pair(src[1], out[0]) == 1.0 and a[1] == 0  # exact-match anchor
    assert a[2] is None                                   # one-to-one: output 0 is already taken
    matched = [j for j in a.values() if j is not None]
    assert len(matched) == len(set(matched))


def test_threshold_option_and_domain_validation():
    import src.claim_aligner as CA
    from src.efs import compute_efs
    s = "CT chest shows findings suggestive of pneumonia in the right lower lobe."
    o = "CT of the chest was performed; the lower lobe shows pneumonia."
    default = compute_efs(s, o, domain="clinical")
    assert compute_efs(s, o, domain="clinical", threshold=0.30)["efs_score"] == default["efs_score"]
    compute_efs(s, o, domain="clinical", threshold=0.95)
    assert CA._THRESHOLD_OVERRIDE is None                 # override is restored after the call
    with pytest.raises(ValueError):
        compute_efs(s, o, domain="science")


def test_greedy_baseline_aligner_identity_false_positives():
    """Appendix J / Table A11: the greedy baseline flags ~43% of identical legal documents; ours none."""
    from src.efs import compute_efs, compute_efs_greedy
    docs = pd.read_parquet(_need(DATA_DIR / "legal_sample.parquet")).source_text.astype(str).tolist()
    greedy_fp = sum(compute_efs_greedy(t[:TRUNC], t[:TRUNC], domain="legal")["distortion_count"] > 0 for t in docs)
    hungarian_fp = sum(compute_efs(t[:TRUNC], t[:TRUNC], domain="legal")["distortion_count"] > 0 for t in docs)
    assert hungarian_fp == 0
    assert round(100 * greedy_fp / len(docs), 1) == 43.2


def test_scored_depth_study_matches_paper_headline():
    df = _scores("depth")
    assert len(df) == 18480
    assert df.groupby("seed").size().to_dict() == {0: 6072, 1: 6072, 123: 6336}
    assert (df[TYPES].sum(axis=1) == df.n_distortions).all()
    H = df.source_hedges.clip(lower=1)
    assert np.allclose(np.maximum(0, 1 - df.n_distortions / H), df.efs, atol=1e-9)
    assert round(float(df.efs.mean()), 3) == 0.450
    assert round(float(df[df.seed == 123].efs.mean()), 3) == 0.443
    shares = {t: round(100 * df[t].sum() / df.n_distortions.sum(), 1) for t in TYPES}
    assert shares == {"omission": 66.3, "escalation": 11.7, "attenuation": 1.8,
                      "fabrication": 8.4, "scope_shift": 7.0, "severity_warp": 4.7}


@pytest.mark.parametrize("domain,paper_mean", [("legal", 0.359), ("scientific", 0.427)])
def test_scored_crossdomain_matches_paper(domain, paper_mean):
    df = _scores("crossdomain")
    g = df[df.domain == domain]
    assert len(g) == 1152 and g.doc_id.nunique() == 192
    assert round(float(g.efs.mean()), 3) == paper_mean
