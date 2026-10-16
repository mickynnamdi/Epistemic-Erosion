"""
Cross-domain distortion analyzer.

Extends the clinical distortion analyzer to compute full EFS
(not just hedge retention) for legal and scientific text.

Each domain gets its own claim extractor and term patterns.
"""

import re
from typing import List, Optional
from dataclasses import dataclass

from src.hedge_screener import HedgeMatch
from src.hedge_screener_legal import find_legal_hedges, legal_hedge_density
from src.hedge_screener_scientific import find_scientific_hedges, scientific_hedge_density
from src.hedge_screener import find_hedges, hedge_density
from src.distortion_analyzer import (
    DistortionInstance, EPISTEMIC_STRENGTH, get_epistemic_strength,
    _estimate_severity, _get_severity_level, _SEVERITY_LEVELS,
)

import os as _os
from src.claim_aligner import align_claims


def _use_hungarian() -> bool:
    """True for the one-to-one aligner (EFS_ALIGNER="hungarian", set by the package);
    "greedy" selects the greedy baseline aligner of the Appendix J ablation."""
    return _os.environ.get("EFS_ALIGNER", "greedy").lower() == "hungarian"


# -----------------------------------------------------------------------
# Domain-specific claim term patterns
# -----------------------------------------------------------------------

LEGAL_CLAIM_PATTERNS = re.compile(
    r'\b(?:liability|liable|negligent|negligence|discriminat\w+|violat\w+|'
    r'breach|damages?|injunction|relief|judgment|ruling|holding|'
    r'statute|constitutional|amendment|rights?|claim|cause of action|'
    r'plaintiff|defendant|evidence|testimony|witness|'
    r'guilty|innocent|convicted|acquitted|sentenced|'
    r'alleged|complaint|motion|summary judgment)\b',
    re.IGNORECASE
)

SCIENTIFIC_CLAIM_PATTERNS = re.compile(
    r'\b(?:result|finding|observation|experiment|analysis|'
    r'performance|accuracy|precision|recall|f1|baseline|'
    r'improvement|degradation|correlation|significant|'
    r'hypothesis|effect|impact|contribution|limitation|'
    r'dataset|benchmark|metric|score|loss|error|'
    r'outperform|comparable|superior|inferior|'
    r'method|approach|model|framework|algorithm|'
    r'training|evaluation|ablation)\b',
    re.IGNORECASE
)

CLINICAL_CLAIM_PATTERNS = re.compile(
    r'\b(?:diagnosis|diagnos\w+|pneumonia|infection|disease|syndrome|'
    r'failure|infarct\w*|embolism|malignancy|cancer|tumor|mass|'
    r'effusion|hemorrhage|fracture|obstruction|stenosis|'
    r'elevated|abnormal|positive|negative)\b',
    re.IGNORECASE
)

DOMAIN_CONFIG = {
    "clinical": {
        "hedge_fn": find_hedges,
        "density_fn": hedge_density,
        "claim_pattern": CLINICAL_CLAIM_PATTERNS,
    },
    "legal": {
        "hedge_fn": find_legal_hedges,
        "density_fn": legal_hedge_density,
        "claim_pattern": LEGAL_CLAIM_PATTERNS,
    },
    "scientific": {
        "hedge_fn": find_scientific_hedges,
        "density_fn": scientific_hedge_density,
        "claim_pattern": SCIENTIFIC_CLAIM_PATTERNS,
    },
}


def extract_domain_claims(text: str, domain: str, require_hedge: bool = True) -> List[dict]:
    """
    Extract claims from text using domain-specific hedge finder and term patterns.

    Args:
      require_hedge: If True (source-side default; Section 4.2, Stage 2), admit only
        sentences with both a hedge marker and a domain-relevant claim term, so
        boilerplate without epistemic content does not become a "claim." If False
        (output-side default), admit any sentence with a domain term so that a
        dehedged output paraphrase can still be matched against a source hedged
        claim (necessary for escalation / attenuation detection).
    """
    config = DOMAIN_CONFIG[domain]
    hedges = config["hedge_fn"](text)
    claim_pattern = config["claim_pattern"]

    claims = []
    sentences = re.split(r'(?<=[.!?])\s+|\n(?=[A-Z#\d])', text)

    for sent in sentences:
        sent = sent.strip()
        if len(sent) < 10:
            continue

        sent_start = text.find(sent)
        sent_hedges = [h for h in hedges if h.start >= sent_start and
                       h.end <= sent_start + len(sent)]

        domain_terms = claim_pattern.findall(sent)

        # Section 4.2, Stage 2: a hedged claim needs both a hedge marker and a
        # domain-relevant claim term. This applies on the source side
        # (require_hedge=True) so omission only fires on dropped claims, not on
        # dropped boilerplate. The output side relaxes the hedge requirement so
        # that a dehedged paraphrase still aligns to a source claim (so that
        # escalation/attenuation are detectable).
        keep = (domain_terms and sent_hedges) if require_hedge else bool(domain_terms)
        if keep:
            # Only consider hedges with real epistemic strength (>0); temporal
            # markers like "preliminary" or "new" register as hedges for span
            # counting but carry strength=0 and would otherwise distort the
            # sentence's overall epistemic level.
            scored_strengths = [
                get_epistemic_strength(h.text) for h in sent_hedges
                if get_epistemic_strength(h.text) > 0
            ]
            claims.append({
                "text": sent,
                "hedges": sent_hedges,
                "epistemic_strength": max(scored_strengths, default=4),
                "domain_terms": domain_terms,
            })

    return claims


def detect_distortions_crossdomain(source_text: str, output_text: str, domain: str) -> List[DistortionInstance]:
    """
    Compare source and output to detect uncertainty distortions,
    using domain-specific hedge detection and claim extraction.
    """
    distortions = []

    # Asymmetric extraction: source uses AND (Section 4.2, Stage 2), output uses OR
    # so dehedged paraphrases can still align to source claims.
    source_claims = extract_domain_claims(source_text, domain, require_hedge=True)
    output_claims = extract_domain_claims(output_text, domain, require_hedge=False)

    matched_output_indices = set()

    # One-to-one alignment (exact-anchored, length-normalized, Hungarian), or the greedy
    # raw-count baseline of the Appendix J ablation when EFS_ALIGNER == "greedy".
    _hungarian = _use_hungarian()
    if _hungarian:
        _hedged_idx = [i for i, c in enumerate(source_claims) if c["hedges"]]
        _assign = align_claims([source_claims[i] for i in _hedged_idx], output_claims)
        _pos = {gi: k for k, gi in enumerate(_hedged_idx)}

    # Check each source hedged claim
    for _si, s_claim in enumerate(source_claims):
        if not s_claim["hedges"]:
            continue

        # If the source claim has no detected domain terms, an empty s_terms
        # produces overlap=0 against every output claim, which correctly falls
        # through to the OMISSION branch below — do not early-skip.
        s_terms = set(t.lower() for t in s_claim["domain_terms"])

        best_match = None
        best_match_idx = None
        best_overlap = 0
        if _hungarian:
            j = _assign.get(_pos[_si])
            if j is not None:
                best_match, best_match_idx, best_overlap = output_claims[j], j, 1
        else:
            for idx, o_claim in enumerate(output_claims):
                o_terms = set(t.lower() for t in o_claim["domain_terms"])
                overlap = len(s_terms & o_terms)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_match = o_claim
                    best_match_idx = idx

        if best_match is None or best_overlap == 0:
            distortions.append(DistortionInstance(
                distortion_type="omission",
                source_span=s_claim["text"][:200],
                output_span="",
                source_hedge=s_claim["hedges"][0].text if s_claim["hedges"] else None,
                output_hedge=None,
                source_context=s_claim["text"][:300],
                output_context="[NOT FOUND IN OUTPUT]",
                severity_estimate=_estimate_severity_domain(s_claim),
            ))
            continue

        matched_output_indices.add(best_match_idx)

        s_strength = s_claim["epistemic_strength"]
        o_strength = best_match["epistemic_strength"]

        if s_strength < o_strength or (s_claim["hedges"] and not best_match["hedges"]):
            distortions.append(DistortionInstance(
                distortion_type="escalation",
                source_span=s_claim["text"][:200],
                output_span=best_match["text"][:200],
                source_hedge=s_claim["hedges"][0].text if s_claim["hedges"] else None,
                output_hedge=best_match["hedges"][0].text if best_match["hedges"] else None,
                source_context=s_claim["text"][:300],
                output_context=best_match["text"][:300],
                severity_estimate=_estimate_severity_domain(s_claim),
            ))
        elif s_strength > o_strength and best_match["hedges"]:
            distortions.append(DistortionInstance(
                distortion_type="attenuation",
                source_span=s_claim["text"][:200],
                output_span=best_match["text"][:200],
                source_hedge=s_claim["hedges"][0].text if s_claim["hedges"] else None,
                output_hedge=best_match["hedges"][0].text if best_match["hedges"] else None,
                source_context=s_claim["text"][:300],
                output_context=best_match["text"][:300],
                severity_estimate="low",
            ))
        elif s_claim["hedges"] and best_match["hedges"]:
            # Check scope shift
            s_terms_set = set(t.lower() for t in s_claim["domain_terms"])
            o_terms_set = set(t.lower() for t in best_match["domain_terms"])
            extra_terms = o_terms_set - s_terms_set
            if extra_terms and len(extra_terms) >= 1:
                distortions.append(DistortionInstance(
                    distortion_type="scope_shift",
                    source_span=s_claim["text"][:200],
                    output_span=best_match["text"][:200],
                    source_hedge=s_claim["hedges"][0].text,
                    output_hedge=best_match["hedges"][0].text if best_match["hedges"] else None,
                    source_context=s_claim["text"][:300],
                    output_context=best_match["text"][:300],
                    severity_estimate=_estimate_severity_domain(s_claim),
                ))

    # Check severity warps
    _detect_severity_warps_domain(source_claims, output_claims, distortions)

    # Primary (strict) fabrication: hedged output claim with no domain term
    # shared with the source.
    source_all_terms = set()
    for sc in source_claims:
        source_all_terms.update(t.lower() for t in sc["domain_terms"])

    flagged_output_spans = set()  # dedup across strict + loose
    for idx, o_claim in enumerate(output_claims):
        if idx in matched_output_indices:
            continue
        if not o_claim["hedges"]:
            continue
        o_terms = set(t.lower() for t in o_claim["domain_terms"])
        if o_terms and not (o_terms & source_all_terms):
            distortions.append(DistortionInstance(
                distortion_type="fabrication",
                source_span="",
                output_span=o_claim["text"][:200],
                source_hedge=None,
                output_hedge=o_claim["hedges"][0].text if o_claim["hedges"] else None,
                source_context="[NOT FOUND IN SOURCE]",
                output_context=o_claim["text"][:300],
                severity_estimate=_estimate_severity_domain(o_claim),
            ))
            flagged_output_spans.add(o_claim["text"][:200])

    # Secondary (loose) fabrication: content-bigram overlap < 0.30. Catches
    # padding fabrications the strict detector misses (e.g. "the case may
    # have implications..."). Spot check on 44 pairs: 11 of 16 flagged spans were
    # fabrications (precision 0.69; Appendix A).
    from src.fabrication_loose import detect_loose_fabrications
    claim_pattern = DOMAIN_CONFIG[domain]["claim_pattern"]
    for sent, hedge, overlap in detect_loose_fabrications(source_text, output_text, claim_pattern):
        key = sent[:200]
        if key in flagged_output_spans:
            continue  # primary already caught it
        distortions.append(DistortionInstance(
            distortion_type="fabrication",
            source_span="",
            output_span=key,
            source_hedge=None,
            output_hedge=hedge,
            source_context=f"[NOT FOUND IN SOURCE - loose filter, bigram_overlap={overlap}]",
            output_context=sent[:300],
            severity_estimate="moderate",
        ))
        flagged_output_spans.add(key)

    return distortions


def _estimate_severity_domain(claim: dict) -> str:
    """Generic severity estimate."""
    terms = set(t.lower() for t in claim.get("domain_terms", []))
    if len(terms) >= 3:
        return "high"
    elif len(terms) >= 2:
        return "moderate"
    return "low"


def _detect_severity_warps_domain(source_claims, output_claims, distortions):
    """Detect severity modifier changes across domains."""
    # The one-to-one aligner uses the same assignment as escalation; the greedy
    # baseline derives its own match.
    _hungarian = _use_hungarian()
    _assign = align_claims(source_claims, output_claims) if _hungarian else None

    for _si, s_claim in enumerate(source_claims):
        s_sev = _get_severity_level(s_claim["text"])
        if s_sev is None:
            continue
        # Allow empty s_terms: severity warps with no anchor will simply
        # fail to find a best_match and be skipped naturally.
        s_terms = set(t.lower() for t in s_claim["domain_terms"])

        best_match = None
        best_overlap = 0
        if _hungarian:
            j = _assign.get(_si)
            if j is not None:
                best_match, best_overlap = output_claims[j], 1
        else:
            for o_claim in output_claims:
                o_terms = set(t.lower() for t in o_claim["domain_terms"])
                overlap = len(s_terms & o_terms)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_match = o_claim

        if best_match is None or best_overlap == 0:
            continue

        o_sev = _get_severity_level(best_match["text"])
        if o_sev is None or abs(o_sev - s_sev) >= 1:
            distortions.append(DistortionInstance(
                distortion_type="severity_warp",
                source_span=s_claim["text"][:200],
                output_span=best_match["text"][:200] if best_match else "",
                source_hedge=None,
                output_hedge=None,
                source_context=s_claim["text"][:300],
                output_context=best_match["text"][:300] if best_match else "",
                severity_estimate="moderate",
            ))


def compute_efs_crossdomain(source_text: str, output_text: str, domain: str) -> dict:
    """
    Compute full EFS for any domain (clinical, legal, scientific).
    """
    config = DOMAIN_CONFIG[domain]
    source_stats = config["density_fn"](source_text)
    output_stats = config["density_fn"](output_text)
    distortions = detect_distortions_crossdomain(source_text, output_text, domain)

    n_source_hedges = max(source_stats["total_hedges"], 1)
    n_distortions = len(distortions)

    efs = max(0.0, 1.0 - n_distortions / n_source_hedges)

    by_type = {}
    fab_strict = 0
    fab_loose = 0
    for d in distortions:
        by_type[d.distortion_type] = by_type.get(d.distortion_type, 0) + 1
        if d.distortion_type == "fabrication":
            if "loose filter" in (d.source_context or ""):
                fab_loose += 1
            else:
                fab_strict += 1

    return {
        "efs_score": round(efs, 3),
        "distortion_count": n_distortions,
        "distortion_rate": round(n_distortions / n_source_hedges, 3),
        "source_hedges": source_stats["total_hedges"],
        "output_hedges": output_stats["total_hedges"],
        "hedge_retention_rate": output_stats["total_hedges"] / n_source_hedges,
        "by_type": by_type,
        "fabrication_strict": fab_strict,
        "fabrication_loose": fab_loose,
        "distortions": distortions,
    }
