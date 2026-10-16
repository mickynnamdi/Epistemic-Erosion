"""
Hedge / uncertainty marker screener for clinical text.

Identifies and counts epistemic uncertainty expressions in clinical notes.
Used for:
  1. Pre-screening notes by uncertainty density for stratified sampling
  2. Extracting uncertainty-bearing spans for annotation and EFS scoring
"""

import re
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class HedgeMatch:
    """A single uncertainty marker found in text."""
    text: str           # the matched string
    category: str       # which hedge category it belongs to
    start: int          # char offset start
    end: int            # char offset end
    context: str        # surrounding text window


# ---------------------------------------------------------------------------
# Hedge pattern taxonomy — organized by clinical uncertainty type
# ---------------------------------------------------------------------------

HEDGE_PATTERNS = {
    # Diagnostic uncertainty: hedged diagnoses and differentials
    "diagnostic_hedge": [
        r"\b(?:likely|most likely|probable|possible|presumed|suspected|"
        r"presumptive|tentative|working diagnosis of)\b",
        r"\bcannot (?:rule out|exclude|be excluded)\b",
        r"\brule out\b",
        r"\bdifferential (?:diagnosis|includes?|considerations?)\b",
        r"\b(?:suggestive|suspicious|concerning|worrisome)\s+(?:of|for)\b",
        r"\b(?:consistent with|compatible with|in keeping with)\b",
        r"\bmay (?:represent|indicate|suggest|reflect|be (?:due to|related to|secondary to|caused by))\b",
        r"\b(?:appears?|seems?) to (?:be|have|show|represent)\b",
        r"\bcould (?:represent|indicate|suggest|be)\b",
    ],

    # Evidential hedges: qualifying the evidence base
    "evidential_hedge": [
        r"\b(?:reportedly|apparently|allegedly|per (?:patient|family|report|history))\b",
        r"\b(?:no (?:definite|definitive|convincing|clear|obvious) (?:evidence|sign))\b",
        r"\b(?:equivocal|indeterminate|nonspecific|non-specific|inconclusive|ambiguous)\b",
        r"\bof (?:uncertain|unknown|unclear|undetermined) (?:significance|etiology|origin|cause)\b",
    ],

    # Degree/severity modifiers
    "severity_modifier": [
        r"\b(?:mild|minimal|slight|subtle|trace|borderline|low-grade|small)\b",
        r"\b(?:moderate|intermediate)\b",
        r"\b(?:severe|significant|marked|extensive|profound|large)\b",
    ],

    # Temporal uncertainty
    "temporal_hedge": [
        r"\b(?:acute|subacute|chronic|new|old|evolving|developing|worsening|improving|stable|unchanged)\b",
        r"\b(?:interval (?:change|improvement|worsening|development))\b",
    ],

    # Recommendation hedges (conditional next steps)
    "recommendation_hedge": [
        r"\b(?:consider|recommend|suggest|advise|may (?:benefit|warrant))\b",
        r"\b(?:if (?:clinically|symptoms?|concerned|indicated|warranted))\b",
        r"\b(?:further (?:evaluation|workup|investigation|imaging|testing) (?:is |may be )?(?:recommended|suggested|warranted|indicated|needed))\b",
        r"\bclinical correlation (?:is )?(?:recommended|advised|suggested)\b",
    ],

    # Negation with residual uncertainty
    "negation_hedge": [
        r"\bno (?:evidence|sign|finding|indication) (?:of|for|to suggest)\b",
        r"\b(?:unlikely|less likely|low probability|low suspicion)\b",
        r"\bwithout (?:definite|clear|obvious)\b",
    ],
}

# Compile all patterns
_COMPILED_PATTERNS = {}
for category, patterns in HEDGE_PATTERNS.items():
    _COMPILED_PATTERNS[category] = [
        re.compile(p, re.IGNORECASE) for p in patterns
    ]


def find_hedges(text: str, context_window: int = 80) -> List[HedgeMatch]:
    """
    Find all uncertainty/hedge markers in clinical text.

    Args:
        text: Clinical note text
        context_window: Characters of context to include around each match

    Returns:
        List of HedgeMatch objects, sorted by position
    """
    matches = []
    for category, compiled in _COMPILED_PATTERNS.items():
        for pattern in compiled:
            for m in pattern.finditer(text):
                start = m.start()
                end = m.end()
                ctx_start = max(0, start - context_window)
                ctx_end = min(len(text), end + context_window)
                matches.append(HedgeMatch(
                    text=m.group(),
                    category=category,
                    start=start,
                    end=end,
                    context=text[ctx_start:ctx_end],
                ))
    # Sort by position and deduplicate overlapping spans
    matches.sort(key=lambda m: (m.start, -m.end))
    return _deduplicate_overlaps(matches)


def _deduplicate_overlaps(matches: List[HedgeMatch]) -> List[HedgeMatch]:
    """Remove overlapping matches, keeping the longer one."""
    if not matches:
        return matches
    result = [matches[0]]
    for m in matches[1:]:
        prev = result[-1]
        if m.start < prev.end:
            # Overlapping — keep the longer match
            if (m.end - m.start) > (prev.end - prev.start):
                result[-1] = m
        else:
            result.append(m)
    return result


def hedge_density(text: str) -> dict:
    """
    Compute hedge density metrics for a clinical note.

    Returns:
        Dict with total count, per-category counts, and density per 1000 chars.
    """
    matches = find_hedges(text)
    total = len(matches)
    by_category = {}
    for m in matches:
        by_category[m.category] = by_category.get(m.category, 0) + 1

    text_len = max(len(text), 1)
    return {
        "total_hedges": total,
        "hedges_per_1k_chars": round(total / text_len * 1000, 2),
        "by_category": by_category,
        "diagnostic_hedges": by_category.get("diagnostic_hedge", 0),
        "evidential_hedges": by_category.get("evidential_hedge", 0),
        "negation_hedges": by_category.get("negation_hedge", 0),
    }


def classify_uncertainty_density(text: str) -> str:
    """Classify a note as low/medium/high uncertainty density."""
    stats = hedge_density(text)
    diag = stats["diagnostic_hedges"]
    total = stats["total_hedges"]

    if diag >= 5 or total >= 15:
        return "high"
    elif diag >= 2 or total >= 7:
        return "medium"
    else:
        return "low"


if __name__ == "__main__":
    # Synthetic example
    test_text = """
    The patient presented with chest pain. CT angiography suggestive of
    pulmonary embolism, cannot rule out aortic dissection. Likely viral
    pneumonia given clinical presentation. Troponin mildly elevated,
    of uncertain significance. Recommend further evaluation with
    echocardiogram if clinically indicated. No definitive evidence of
    acute coronary syndrome. Per patient, symptoms started 3 days ago.
    """
    matches = find_hedges(test_text)
    print(f"Found {len(matches)} hedge markers:\n")
    for m in matches:
        print(f"  [{m.category}] '{m.text}' (pos {m.start}-{m.end})")

    print(f"\nDensity stats: {hedge_density(test_text)}")
    print(f"Classification: {classify_uncertainty_density(test_text)}")
