"""
EFS pipeline with the one-to-one claim aligner (the pipeline used for the paper).

Claim extraction (extract_clinical_claims), hedge finding (find_hedges), epistemic
strength, severity levels and the distortion-typing rules come from
src.distortion_analyzer; this module supplies the alignment:
  * escalation / attenuation / scope shift and severity warp are evaluated on the same
    one-to-one assignment (src.claim_aligner);
  * omission fires when a source claim has no counterpart above the threshold;
  * fabrication uses the assignment's matched set.

EFS(s, s) == 1.0 with zero distortions for any s.
"""
from __future__ import annotations

from typing import List, Optional

from src.claim_aligner import align_claims
from src.distortion_analyzer import (
    DistortionInstance,
    _detect_domain,
    _estimate_severity,
    _get_severity_level,
    extract_clinical_claims,
    find_hedges,
)


def detect_distortions_hungarian(source_text: str, output_text: str) -> List[DistortionInstance]:
    distortions: List[DistortionInstance] = []

    source_claims = extract_clinical_claims(source_text, require_hedge=True)
    output_claims = extract_clinical_claims(output_text, require_hedge=False)
    # require_hedge=True already guarantees this.
    assert all(c["hedges"] for c in source_claims), "source claim without a hedge"

    if not source_claims:
        assignment = {}
    else:
        assignment = align_claims(source_claims, output_claims)

    matched_output_indices = {j for j in assignment.values() if j is not None}

    for i, s_claim in enumerate(source_claims):
        j = assignment.get(i)

        if j is None:
            distortions.append(DistortionInstance(
                distortion_type="omission",
                source_span=s_claim["text"][:200], output_span="",
                source_hedge=s_claim["hedges"][0].text if s_claim["hedges"] else None,
                output_hedge=None,
                source_context=s_claim["text"][:300],
                output_context="[NOT FOUND IN OUTPUT]",
                severity_estimate=_estimate_severity(s_claim),
            ))
            continue

        o_claim = output_claims[j]
        s_strength = s_claim["epistemic_strength"]
        o_strength = o_claim["epistemic_strength"]

        if s_strength < o_strength or (s_claim["hedges"] and not o_claim["hedges"]):
            distortions.append(DistortionInstance(
                distortion_type="escalation",
                source_span=s_claim["text"][:200], output_span=o_claim["text"][:200],
                source_hedge=s_claim["hedges"][0].text if s_claim["hedges"] else None,
                output_hedge=o_claim["hedges"][0].text if o_claim["hedges"] else None,
                source_context=s_claim["text"][:300], output_context=o_claim["text"][:300],
                severity_estimate=_estimate_severity(s_claim),
            ))
        elif s_strength > o_strength and o_claim["hedges"]:
            distortions.append(DistortionInstance(
                distortion_type="attenuation",
                source_span=s_claim["text"][:200], output_span=o_claim["text"][:200],
                source_hedge=s_claim["hedges"][0].text if s_claim["hedges"] else None,
                output_hedge=o_claim["hedges"][0].text if o_claim["hedges"] else None,
                source_context=s_claim["text"][:300], output_context=o_claim["text"][:300],
                severity_estimate="low",
            ))
        elif s_claim["hedges"] and o_claim["hedges"]:
            s_terms = {t.lower() for t in s_claim["clinical_terms"]}
            o_terms = {t.lower() for t in o_claim["clinical_terms"]}
            if o_terms - s_terms:
                distortions.append(DistortionInstance(
                    distortion_type="scope_shift",
                    source_span=s_claim["text"][:200], output_span=o_claim["text"][:200],
                    source_hedge=s_claim["hedges"][0].text,
                    output_hedge=o_claim["hedges"][0].text if o_claim["hedges"] else None,
                    source_context=s_claim["text"][:300], output_context=o_claim["text"][:300],
                    severity_estimate=_estimate_severity(s_claim),
                ))

        # SEVERITY_WARP on the same aligned pair.
        s_sev = _get_severity_level(s_claim["text"])
        if s_sev is not None:
            o_sev = _get_severity_level(o_claim["text"])
            if o_sev is None:
                distortions.append(DistortionInstance(
                    distortion_type="severity_warp",
                    source_span=s_claim["text"][:200], output_span=o_claim["text"][:200],
                    source_hedge=None, output_hedge=None,
                    source_context=s_claim["text"][:300], output_context=o_claim["text"][:300],
                    severity_estimate="moderate" if s_sev >= 2 else "low",
                ))
            elif abs(o_sev - s_sev) >= 1:
                distortions.append(DistortionInstance(
                    distortion_type="severity_warp",
                    source_span=s_claim["text"][:200], output_span=o_claim["text"][:200],
                    source_hedge=None, output_hedge=None,
                    source_context=s_claim["text"][:300], output_context=o_claim["text"][:300],
                    severity_estimate="high" if abs(o_sev - s_sev) >= 2 else "moderate",
                ))

    # FABRICATION: hedged output claims with no source basis.
    source_all_terms = set()
    for sc in source_claims:
        source_all_terms.update(t.lower() for t in sc["clinical_terms"])
    for idx, o_claim in enumerate(output_claims):
        if idx in matched_output_indices or not o_claim["hedges"]:
            continue
        o_terms = {t.lower() for t in o_claim["clinical_terms"]}
        if o_terms and not (o_terms & source_all_terms):
            distortions.append(DistortionInstance(
                distortion_type="fabrication",
                source_span="", output_span=o_claim["text"][:200],
                source_hedge=None,
                output_hedge=o_claim["hedges"][0].text if o_claim["hedges"] else None,
                source_context="[NOT FOUND IN SOURCE]", output_context=o_claim["text"][:300],
                severity_estimate=_estimate_severity(o_claim),
            ))
    return distortions


def compute_efs_hungarian(source_text: str, output_text: str,
                          domain: Optional[str] = None) -> dict:
    """
    EFS = max(0, 1 - |D| / max(|H(S)|, 1)) with the one-to-one aligner.

      * legal/scientific text is scored by the cross-domain pipeline (clinical claim
        regexes on legal prose would produce boilerplate false positives);
      * the denominator is max(total_hedges, 1).
    """
    if domain is None:
        domain = _detect_domain(source_text)

    if domain in ("legal", "scientific"):
        # the cross-domain module uses the one-to-one aligner when EFS_ALIGNER == "hungarian"
        import os
        from src.distortion_analyzer_crossdomain import compute_efs_crossdomain
        prev = os.environ.get("EFS_ALIGNER")
        os.environ["EFS_ALIGNER"] = "hungarian"
        try:
            return compute_efs_crossdomain(source_text, output_text, domain)
        finally:
            if prev is None:
                os.environ.pop("EFS_ALIGNER", None)
            else:
                os.environ["EFS_ALIGNER"] = prev

    distortions = detect_distortions_hungarian(source_text, output_text)
    source_hedges = find_hedges(source_text)
    n_h = max(len(source_hedges), 1)

    efs = max(0.0, 1.0 - len(distortions) / n_h)

    by_type: dict = {}
    for d in distortions:
        by_type[d.distortion_type] = by_type.get(d.distortion_type, 0) + 1

    return {
        "efs_score": efs, "efs": efs,
        "distortion_count": len(distortions),
        "distortion_rate": (len(distortions) / n_h) if n_h else 0.0,
        "source_hedges": n_h,
        "by_type": by_type,
        "distortions": distortions,
        "domain": domain,
    }
