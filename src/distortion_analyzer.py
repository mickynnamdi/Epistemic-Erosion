"""
Distortion analysis: compare hedge markers between source and LLM output.

This is the core analysis module that:
  1. Extracts uncertainty spans from source and output
  2. Aligns them via semantic overlap
  3. Classifies distortion type (escalation, attenuation, omission, etc.)
  4. Computes the Epistemic Faithfulness Score (EFS)
"""

import re
from dataclasses import dataclass
from typing import List, Tuple, Optional

from src.hedge_screener import find_hedges, HedgeMatch, hedge_density


@dataclass
class DistortionInstance:
    """A single detected uncertainty distortion."""
    distortion_type: str       # escalation, attenuation, omission, fabrication, scope_shift, severity_warp
    source_span: str           # the source text span
    output_span: str           # the output text span (empty for omission)
    source_hedge: Optional[str]  # the hedge marker in source
    output_hedge: Optional[str]  # the hedge marker in output (if any)
    source_context: str
    output_context: str
    severity_estimate: str     # low, moderate, high (heuristic, not clinical)


# ---------------------------------------------------------------------------
# Core hedge terms ranked by epistemic strength (weaker → stronger)
# ---------------------------------------------------------------------------

EPISTEMIC_STRENGTH = {
    # Weak (high uncertainty)
    "possible": 1, "possibly": 1, "may": 1, "might": 1, "could": 1,
    "cannot rule out": 1, "cannot exclude": 1, "rule out": 1,
    "questionable": 1, "equivocal": 1, "indeterminate": 1,
    # Negation hedges (hedge_screener.py "negation_hedge") assert a claim at low
    # certainty. They are listed explicitly so that the longest-match lookup in
    # get_epistemic_strength() does not resolve "unlikely" to "likely" or
    # "no definite evidence" to "definite".
    "unlikely": 1, "less likely": 1, "low probability": 1, "low suspicion": 1,
    "without definite": 1, "no definite evidence": 1,
    # Moderate
    "suspected": 2, "suggestive of": 2, "suspicious": 2,
    "concerning for": 2, "consistent with": 2, "compatible with": 2,
    "appears to": 2, "seems to": 2,
    # Moderately strong
    "likely": 3, "probable": 3, "most likely": 3, "presumed": 3,
    # Strong (low/no uncertainty)
    "confirmed": 4, "definite": 4, "diagnostic of": 4, "proven": 4,
}


def get_epistemic_strength(text: str) -> int:
    """Get the epistemic strength of a text span (0=no hedge found, 1-4 scale)."""
    text_lower = text.lower()
    for term, strength in sorted(EPISTEMIC_STRENGTH.items(), key=lambda x: -len(x[0])):
        if term in text_lower:
            return strength
    return 0  # no hedge detected → treated as definitive


def extract_clinical_claims(text: str, require_hedge: bool = True) -> List[dict]:
    """
    Extract key clinical claims (diagnoses, findings, recommendations).

    Args:
      require_hedge: If True (source-side default; Section 4.2, Stage 2), admit
        only sentences with both a hedge marker and a domain-relevant claim
        term. If False (output-side default), admit any sentence with a
        clinical term so dehedged paraphrases align for escalation /
        attenuation detection.
    """
    hedges = find_hedges(text)
    claims = []

    # Also extract sentences containing key clinical patterns
    # Split into sentences (rough clinical sentence splitting)
    sentences = re.split(r'(?<=[.!?])\s+|\n(?=[A-Z#\d])', text)

    for sent in sentences:
        sent = sent.strip()
        if len(sent) < 10:
            continue

        # Find hedges within this sentence
        sent_hedges = [h for h in hedges if h.start >= text.find(sent) and
                       h.end <= text.find(sent) + len(sent)]

        # Check if sentence contains clinical content
        clinical_patterns = re.findall(
            r'\b(?:'
            # Diagnoses & conditions
            r'diagnosis|diagnos\w+|disease|syndrome|disorder|condition|pathology|'
            r'pneumonia|infection|sepsis|inflammation|inflammatory|abscess|'
            # Cardiovascular
            r'failure|infarct\w*|infarction|ischemia|ischemic|embolism|embolus|'
            r'thrombus|thrombosis|aneurysm|stenosis|occlusion|atherosclero\w+|'
            r'arrhythmia|fibrillation|cardiomyopathy|hypertension|hypotension|'
            # Neuro
            r'stroke|TIA|vasospasm|vasculitis|encephalopathy|seizure|hemorrhage|'
            r'subarachnoid|intracranial|cerebral|cortical|parietal|temporal|'
            r'frontal|occipital|cerebellar|brainstem|hydrocephalus|edema|'
            # Pulmonary
            r'pulmonary|respiratory|asthma|COPD|atelectasis|consolidation|'
            r'effusion|pneumothorax|pleural|bronch\w+|alveolar|emphysema|'
            # Oncologic
            r'malignancy|cancer|carcinoma|sarcoma|lymphoma|leukemia|tumor|tumour|'
            r'mass|nodule|lesion|metasta\w+|neoplas\w+|adenopathy|'
            # GI/GU
            r'fracture|obstruction|perforation|hepatic|renal|pancreati\w+|'
            r'cirrhosis|cholecystitis|gastritis|colitis|appendicitis|'
            # Anatomy / artery / vessel terms
            r'artery|arterial|vein|venous|ventricle|ventricular|atrial|atrium|'
            r'aorta|aortic|coronary|carotid|hepatic|renal|cerebral|spinal|'
            r'lobe|lobar|cortex|cortical|parenchyma|'
            # Abnormality / generic finding
            r'elevated|abnormal\w*|positive|negative|finding\w*|deficit\w*|'
            r'opacity|opacities|infiltrate|enhancement|hypodens\w+|hyperdens\w+|'
            r'hypointens\w+|hyperintens\w+|hyperechoic|hypoechoic|'
            # Common medical abbreviations
            r'MI|CHF|DVT|PE|COPD|AKI|CKD|MRI|CT|CTA|EKG|ECG|UTI|'
            r'CAA|ICA|MCA|ACA|PCA|RCVS|SAH|ICH|IVH|TIA|GBM|GIB'
            r')\b',
            sent, re.IGNORECASE
        )

        # Fallback: capture capitalized medical-style abbreviations (2-5 letters)
        # that appear inside hedged clauses, e.g., "possible XYZ" — this rescues
        # rare abbreviations not in our explicit list.
        if sent_hedges and not clinical_patterns:
            abbrev_matches = re.findall(r'\b[A-Z]{2,5}\b', sent)
            # Filter out obviously non-clinical abbreviations
            blacklist = {'MRI', 'CT', 'IV', 'PO', 'PRN', 'BID', 'TID', 'QD', 'QID',
                         'AM', 'PM', 'ICU', 'ED', 'OR', 'PT', 'OT', 'NPO', 'DNR'}
            clinical_patterns = [a for a in abbrev_matches if a not in blacklist]

        # Section 4.2, Stage 2: a hedged claim needs both a hedge marker and a
        # domain-relevant claim term. This applies on the source side
        # (require_hedge=True) so omission only fires on dropped claims, not on
        # dropped boilerplate. Output-side extraction relaxes the hedge rule so
        # a dehedged paraphrase can still align (escalation/attenuation).
        keep = (clinical_patterns and sent_hedges) if require_hedge else bool(clinical_patterns)
        if keep:
            # Only consider hedges with real epistemic strength (>0) when
            # computing the sentence's overall epistemic_strength. Temporal
            # markers like "acute" register as hedges for span-counting but
            # carry strength=0; treating them as the sentence's epistemic
            # signal causes spurious attenuation/escalation classifications.
            scored_strengths = [
                get_epistemic_strength(h.text) for h in sent_hedges
                if get_epistemic_strength(h.text) > 0
            ]
            claims.append({
                "text": sent,
                "hedges": sent_hedges,
                "epistemic_strength": max(scored_strengths, default=4),
                "clinical_terms": clinical_patterns,
            })

    return claims


def detect_distortions(source_text: str, output_text: str) -> List[DistortionInstance]:
    """
    Compare source and output to detect uncertainty distortions.

    Strategy:
      1. Extract hedged claims from source
      2. For each, search for corresponding content in output
      3. Compare epistemic strength to detect escalation/attenuation
      4. Check for omissions (source claims missing from output)
      5. Check for fabrications (output hedges not in source)
    """
    distortions = []

    # Asymmetric extraction: source uses AND (Section 4.2, Stage 2), output uses OR
    # so dehedged paraphrases can still align to source claims.
    source_claims = extract_clinical_claims(source_text, require_hedge=True)
    output_claims = extract_clinical_claims(output_text, require_hedge=False)

    source_hedges = find_hedges(source_text)
    output_hedges = find_hedges(output_text)

    # Track matched output claims to detect fabrication later
    matched_output_indices = set()

    # --- Check each source hedged claim ---
    for s_claim in source_claims:
        if not s_claim["hedges"]:
            continue  # only check hedged source claims

        # Find matching output content (by shared clinical terms).
        # If the source claim has no detected clinical terms, an empty s_terms
        # produces overlap=0 against every output claim, which correctly falls
        # through to the OMISSION branch below — do not early-skip.
        s_terms = set(t.lower() for t in s_claim["clinical_terms"])

        best_match = None
        best_match_idx = None
        best_overlap = 0
        for idx, o_claim in enumerate(output_claims):
            o_terms = set(t.lower() for t in o_claim["clinical_terms"])
            overlap = len(s_terms & o_terms)
            if overlap > best_overlap:
                best_overlap = overlap
                best_match = o_claim
                best_match_idx = idx

        if best_match is None or best_overlap == 0:
            # OMISSION: hedged source claim not found in output
            distortions.append(DistortionInstance(
                distortion_type="omission",
                source_span=s_claim["text"][:200],
                output_span="",
                source_hedge=s_claim["hedges"][0].text if s_claim["hedges"] else None,
                output_hedge=None,
                source_context=s_claim["text"][:300],
                output_context="[NOT FOUND IN OUTPUT]",
                severity_estimate=_estimate_severity(s_claim),
            ))
            continue

        matched_output_indices.add(best_match_idx)

        # Compare epistemic strength
        s_strength = s_claim["epistemic_strength"]
        o_strength = best_match["epistemic_strength"]

        if s_strength < o_strength or (s_claim["hedges"] and not best_match["hedges"]):
            # ESCALATION: source was uncertain, output is more certain
            distortions.append(DistortionInstance(
                distortion_type="escalation",
                source_span=s_claim["text"][:200],
                output_span=best_match["text"][:200],
                source_hedge=s_claim["hedges"][0].text if s_claim["hedges"] else None,
                output_hedge=best_match["hedges"][0].text if best_match["hedges"] else None,
                source_context=s_claim["text"][:300],
                output_context=best_match["text"][:300],
                severity_estimate=_estimate_severity(s_claim),
            ))
        elif s_strength > o_strength and best_match["hedges"]:
            # ATTENUATION: source was certain, output added uncertainty
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
            # Same strength but check for SCOPE_SHIFT: hedge present in both
            # but applied to different clinical scope
            s_terms_set = set(t.lower() for t in s_claim["clinical_terms"])
            o_terms_set = set(t.lower() for t in best_match["clinical_terms"])
            # If they share some terms but output has extra clinical terms
            # under the same hedge, the scope has expanded
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
                    severity_estimate=_estimate_severity(s_claim),
                ))

    # --- Check for SEVERITY_WARP: severity modifiers changed ---
    _detect_severity_warps(source_claims, output_claims, distortions)

    # --- Check for FABRICATION: hedged claims in output with no source basis ---
    source_all_terms = set()
    for sc in source_claims:
        source_all_terms.update(t.lower() for t in sc["clinical_terms"])

    flagged_output_text = set()
    for idx, o_claim in enumerate(output_claims):
        if idx in matched_output_indices:
            continue
        if not o_claim["hedges"]:
            continue
        # Output has a hedged clinical claim with no match in source
        o_terms = set(t.lower() for t in o_claim["clinical_terms"])
        if o_terms and not (o_terms & source_all_terms):
            distortions.append(DistortionInstance(
                distortion_type="fabrication",
                source_span="",
                output_span=o_claim["text"][:200],
                source_hedge=None,
                output_hedge=o_claim["hedges"][0].text if o_claim["hedges"] else None,
                source_context="[NOT FOUND IN SOURCE]",
                output_context=o_claim["text"][:300],
                severity_estimate=_estimate_severity(o_claim),
            ))
            flagged_output_text.add(o_claim["text"][:200])

    # Secondary (loose) fabrication detector (Appendix A). It catches padding
    # fabrications the strict detector above misses. In the paper's clinical pipeline
    # (one-to-one aligner, src/distortion_analyzer_hungarian.py) only the strict detector is
    # used; this greedy-baseline path applies both.
    from src.fabrication_loose import detect_loose_fabrications
    from src.distortion_analyzer_crossdomain import CLINICAL_CLAIM_PATTERNS
    for sent, hedge, overlap in detect_loose_fabrications(source_text, output_text, CLINICAL_CLAIM_PATTERNS):
        key = sent[:200]
        if key in flagged_output_text:
            continue
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
        flagged_output_text.add(key)

    return distortions


# Severity modifier terms for severity_warp detection
_SEVERITY_LEVELS = {
    "mild": 1, "minimal": 1, "slight": 1, "subtle": 1, "trace": 1,
    "borderline": 1, "small": 1, "low-grade": 1,
    "moderate": 2, "intermediate": 2,
    "severe": 3, "significant": 3, "marked": 3, "extensive": 3,
    "profound": 3, "large": 3,
}


def _get_severity_level(text: str) -> Optional[int]:
    """Extract severity level from text (1=mild, 2=moderate, 3=severe)."""
    text_lower = text.lower()
    for term, level in sorted(_SEVERITY_LEVELS.items(), key=lambda x: -len(x[0])):
        if term in text_lower:
            return level
    return None


def _detect_severity_warps(source_claims, output_claims, distortions):
    """Detect cases where severity modifiers changed between source and output."""
    for s_claim in source_claims:
        s_sev = _get_severity_level(s_claim["text"])
        if s_sev is None:
            continue

        s_terms = set(t.lower() for t in s_claim["clinical_terms"])
        if not s_terms:
            continue

        # Find matching output claim
        best_match = None
        best_overlap = 0
        for o_claim in output_claims:
            o_terms = set(t.lower() for t in o_claim["clinical_terms"])
            overlap = len(s_terms & o_terms)
            if overlap > best_overlap:
                best_overlap = overlap
                best_match = o_claim

        if best_match is None or best_overlap == 0:
            continue

        o_sev = _get_severity_level(best_match["text"])
        if o_sev is None:
            # Severity modifier dropped entirely
            distortions.append(DistortionInstance(
                distortion_type="severity_warp",
                source_span=s_claim["text"][:200],
                output_span=best_match["text"][:200],
                source_hedge=None,
                output_hedge=None,
                source_context=s_claim["text"][:300],
                output_context=best_match["text"][:300],
                severity_estimate="moderate" if s_sev >= 2 else "low",
            ))
        elif abs(o_sev - s_sev) >= 1:
            # Severity level changed
            distortions.append(DistortionInstance(
                distortion_type="severity_warp",
                source_span=s_claim["text"][:200],
                output_span=best_match["text"][:200],
                source_hedge=None,
                output_hedge=None,
                source_context=s_claim["text"][:300],
                output_context=best_match["text"][:300],
                severity_estimate="high" if abs(o_sev - s_sev) >= 2 else "moderate",
            ))


def _estimate_severity(claim: dict) -> str:
    """Heuristic severity estimate based on clinical terms."""
    high_risk_terms = {"malignancy", "cancer", "tumor", "mass", "embolism",
                       "infarct", "hemorrhage", "failure"}
    terms = set(t.lower() for t in claim.get("clinical_terms", []))
    if terms & high_risk_terms:
        return "high"
    elif len(terms) >= 2:
        return "moderate"
    return "low"


_CLINICAL_DISCRIMINATOR_RE = re.compile(
    r'\b(?:patient|discharge|admission|admitted|diagnosis|mg|mL|tablet|'
    r'BID|TID|QID|PRN|MRI|CT|EKG|ECG|BNP|LVEF|HCT|WBC|'
    r'allergy|allergies|attending|nursing|ICU|ED|emergency)\b',
    re.IGNORECASE,
)
_LEGAL_DISCRIMINATOR_RE = re.compile(
    r'\b(?:plaintiff|defendant|court|alleges|alleged|complaint|'
    r'pursuant|hereinafter|whereas|civil action|case no)\b',
    re.IGNORECASE,
)
_SCIENTIFIC_DISCRIMINATOR_RE = re.compile(
    r'\b(?:abstract|introduction|methodology|hypothesis|'
    r'we propose|we present|we introduce|we show|we evaluate|'
    r'state-of-the-art|outperform|benchmark dataset|ablation)\b',
    re.IGNORECASE,
)


def _detect_domain(text: str) -> str:
    """Auto-detect domain (clinical/legal/scientific) by counting matches
    against each domain's claim-term lexicon plus a small set of domain
    discriminators that the lexicons share (e.g. "evidence" appears in both
    clinical and legal). The pipeline is domain-specific; calling
    it without a domain hint runs this heuristic on the source text."""
    # Import locally to avoid a circular import at module load.
    from src.distortion_analyzer_crossdomain import (
        CLINICAL_CLAIM_PATTERNS, LEGAL_CLAIM_PATTERNS, SCIENTIFIC_CLAIM_PATTERNS,
    )
    # Discriminators get 3× weight so a few highly-specific matches outweigh
    # accidental shared-vocabulary overlap (e.g. "rights" / "evidence" both
    # appear in clinical and legal corpora).
    DW = 3
    counts = {
        "clinical":
            len(CLINICAL_CLAIM_PATTERNS.findall(text))
            + DW * len(_CLINICAL_DISCRIMINATOR_RE.findall(text)),
        "legal":
            len(LEGAL_CLAIM_PATTERNS.findall(text))
            + DW * len(_LEGAL_DISCRIMINATOR_RE.findall(text)),
        "scientific":
            len(SCIENTIFIC_CLAIM_PATTERNS.findall(text))
            + DW * len(_SCIENTIFIC_DISCRIMINATOR_RE.findall(text)),
    }
    best = max(counts.values())
    if best == 0:
        return "clinical"
    # On ties, prefer clinical, then legal, then scientific.
    for d in ("clinical", "legal", "scientific"):
        if counts[d] == best:
            return d
    return "clinical"


def compute_efs(source_text: str, output_text: str, domain: Optional[str] = None) -> dict:
    """
    Compute the Epistemic Faithfulness Score (EFS).

    Args:
      source_text: original text containing epistemic markers.
      output_text: LLM-generated text to compare against the source.
      domain: one of "clinical", "legal", "scientific". If None, the domain
        is auto-detected from ``source_text`` via lexicon-match counts.

    Returns dict with:
      - efs_score: 0-1 (1 = perfect preservation)
      - distortion_count: total distortions detected
      - distortion_rate: distortions per source hedge
      - by_type: count per distortion type
      - distortions: list of DistortionInstance
      - domain: the domain used for detection (resolved from the argument)
    """
    if domain is None:
        domain = _detect_domain(source_text)

    if domain in ("legal", "scientific"):
        # Dispatch non-clinical text through the domain-aware pipeline so the
        # clinical-only claim regex + abbreviation fallback do not produce
        # boilerplate false positives on legal/scientific corpora.
        from src.distortion_analyzer_crossdomain import compute_efs_crossdomain
        result = compute_efs_crossdomain(source_text, output_text, domain)
        result["domain"] = domain
        return result

    source_stats = hedge_density(source_text)
    output_stats = hedge_density(output_text)
    distortions = detect_distortions(source_text, output_text)

    n_source_hedges = max(source_stats["total_hedges"], 1)
    n_distortions = len(distortions)

    # EFS = 1 - (distortions / source hedges), clamped to [0, 1]
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
        "domain": "clinical",
        "by_type": by_type,
        "fabrication_strict": fab_strict,
        "fabrication_loose": fab_loose,
        "distortions": distortions,
    }


if __name__ == "__main__":
    # Example
    source = """
    CT chest shows findings suggestive of pneumonia in the right lower lobe.
    Cannot rule out underlying malignancy given patient's smoking history.
    Troponin mildly elevated, likely demand ischemia. Echocardiogram shows
    possible mild aortic stenosis. Recommend outpatient follow-up with
    pulmonology for further evaluation.
    """

    output_good = """
    CT chest findings are suggestive of pneumonia in the right lower lobe.
    Underlying malignancy cannot be excluded given smoking history.
    Troponin was mildly elevated, likely representing demand ischemia.
    Echocardiogram demonstrated possible mild aortic stenosis.
    Outpatient pulmonology follow-up recommended.
    """

    output_bad = """
    CT chest confirms pneumonia in the right lower lobe.
    The patient has demand ischemia with elevated troponin.
    Echocardiogram shows aortic stenosis.
    Follow up with pulmonology.
    """

    print("=== GOOD OUTPUT (preserves uncertainty) ===")
    efs_good = compute_efs(source, output_good)
    print(f"EFS: {efs_good['efs_score']}, Distortions: {efs_good['distortion_count']}")
    print(f"By type: {efs_good['by_type']}")

    print("\n=== BAD OUTPUT (distorts uncertainty) ===")
    efs_bad = compute_efs(source, output_bad)
    print(f"EFS: {efs_bad['efs_score']}, Distortions: {efs_bad['distortion_count']}")
    print(f"By type: {efs_bad['by_type']}")
    for d in efs_bad["distortions"]:
        print(f"\n  [{d.distortion_type}] severity={d.severity_estimate}")
        print(f"  Source: {d.source_span[:100]}")
        print(f"  Output: {d.output_span[:100]}")
