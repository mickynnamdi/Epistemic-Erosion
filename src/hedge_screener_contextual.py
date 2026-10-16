"""
Context-dependent uncertainty detector for multi-domain hedge screening.

Wraps the regex-based hedge screeners (clinical, legal, scientific) with a
contextual filtering layer that distinguishes genuine epistemic uncertainty
from surface-level pattern matches.

Problem solved:
  - "may represent pneumonia" → genuine uncertainty (kept)
  - "may I help you"          → politeness formula  (filtered out)
  - "stable vitals"           → clinical state      (kept)
  - "stable relationship"     → non-clinical usage  (filtered out)

Approach:
  1. Find candidate hedges via regex (existing domain screeners)
  2. Extract the sentence containing each hedge
  3. Check whether the sentence contains domain-relevant claim patterns
  4. Only retain hedges that co-occur with domain-relevant content

Exports:
  find_contextual_hedges(text, domain) — same interface as find_hedges()
  contextual_hedge_density(text, domain) — unified density function
"""

import re
from typing import Dict, List, Optional, Set, Tuple

from src.hedge_screener import (
    HedgeMatch,
    _deduplicate_overlaps,
    find_hedges,
    hedge_density,
)
from src.hedge_screener_legal import find_legal_hedges, legal_hedge_density
from src.hedge_screener_scientific import (
    find_scientific_hedges,
    scientific_hedge_density,
)


# ============================================================================
# Sentence boundary detection
# ============================================================================

# Sentence-ending punctuation, handling abbreviations loosely.
# Clinical text often lacks clean sentence boundaries, so we also split on
# newlines and semicolons that appear between clauses.
_SENT_SPLIT = re.compile(
    r"(?<=[.!?;])\s+|(?<=\n)\s*",
    re.MULTILINE,
)


def _extract_sentence(text: str, start: int, end: int) -> Tuple[str, int, int]:
    """
    Extract the sentence containing the span [start, end) from text.

    Returns (sentence_text, sentence_start, sentence_end).
    """
    # Find sentence boundaries around the hedge match
    # Look backwards for sentence start
    sent_start = 0
    for m in _SENT_SPLIT.finditer(text):
        if m.end() <= start:
            sent_start = m.end()
        else:
            break

    # Look forwards for sentence end
    sent_end = len(text)
    for m in _SENT_SPLIT.finditer(text):
        if m.start() >= end:
            sent_end = m.start()
            break

    return text[sent_start:sent_end].strip(), sent_start, sent_end


# ============================================================================
# Domain-specific context validators
# ============================================================================

# --- Clinical domain ---

# Patterns that indicate the sentence is making a clinical/medical claim
_CLINICAL_CLAIM_PATTERNS = [
    # Diagnoses and conditions
    re.compile(
        r"\b(?:diagnosis|diagnos(?:ed|is|tic)|finding|impression|assessment|"
        r"condition|disease|disorder|syndrome|infection|lesion|mass|nodule|"
        r"fracture|effusion|pneumonia|embolism|infarct|tumor|carcinoma|"
        r"neoplasm|pathology|malignancy|hemorrhage|edema|inflammation|"
        r"obstruction|stenosis|thrombosis|ischemia)\b",
        re.IGNORECASE,
    ),
    # Clinical observations / results
    re.compile(
        r"\b(?:exam(?:ination)?|imaging|x-?ray|ct|mri|ultrasound|echo|"
        r"lab(?:oratory)?|result|report|scan|biopsy|specimen|culture|"
        r"vitals?|blood\s+pressure|heart\s+rate|temperature|saturation|"
        r"troponin|hemoglobin|creatinine|glucose|potassium|sodium|"
        r"white\s+(?:blood\s+)?cell|wbc|platelet|ekg|ecg|eeg)\b",
        re.IGNORECASE,
    ),
    # Clinical actions and states
    re.compile(
        r"\b(?:patient|pt|symptoms?|present(?:s|ed|ation|ing)|complain(?:s|t|ing)|"
        r"history|h/o|onset|duration|course|prognosis|treatment|therapy|"
        r"medication|dose|admission|discharge|follow[- ]?up|evaluation|"
        r"workup|intervention|procedure|surgery|operative)\b",
        re.IGNORECASE,
    ),
    # Anatomical references
    re.compile(
        r"\b(?:chest|lung|heart|abdomen|liver|kidney|brain|spine|bone|joint|"
        r"artery|vein|lymph|tissue|organ|lobe|ventricle|atrium|valve|"
        r"pulmonary|cardiac|hepatic|renal|cerebral|cranial|thoracic)\b",
        re.IGNORECASE,
    ),
]

# Patterns that indicate the hedge is NOT expressing genuine clinical uncertainty
_CLINICAL_FALSE_POSITIVE_PATTERNS = [
    # Politeness / conversational "may"
    re.compile(r"\bmay\s+I\b", re.IGNORECASE),
    re.compile(r"\bmay\s+(?:we|you)\s+(?:help|assist|ask|have|take|go)\b", re.IGNORECASE),
    # Non-clinical "stable" usage
    re.compile(r"\bstable\s+(?:relationship|income|employment|job|housing|home|"
               r"family|environment|mood|personality)\b", re.IGNORECASE),
    # Non-clinical "mild" usage
    re.compile(r"\bmild\s+(?:mannered|weather|climate|flavor|taste|interest)\b",
               re.IGNORECASE),
    # Non-clinical "significant" usage
    re.compile(r"\bsignificant\s+(?:other|relationship|achievement|milestone|"
               r"event|moment|contribution)\b", re.IGNORECASE),
    # General "consider" not in recommendation context
    re.compile(r"\bconsider\s+(?:the\s+(?:fact|possibility|option|idea|notion)|"
               r"whether|how|why|that\s+(?:he|she|they|we|it\s+(?:is|was)\s+"
               r"(?:a\s+)?(?:good|bad|nice)))\b", re.IGNORECASE),
    # Non-clinical "chronic" usage
    re.compile(r"\bchronic\s+(?:complaint|complainer|problem\s+(?:with|in)\s+"
               r"(?:management|society|politics))\b", re.IGNORECASE),
    # Idiomatic expressions
    re.compile(r"\b(?:come what may|be that as it may|may\s+(?:God|the\s+Lord))\b",
               re.IGNORECASE),
    # "suggest" in non-epistemic sense (suggesting an activity)
    re.compile(r"\b(?:suggest|recommend)\s+(?:that\s+(?:you|we|he|she|they)\s+"
               r"(?:try|go|visit|eat|drink|watch|read|take\s+a))\b",
               re.IGNORECASE),
]

# --- Legal domain ---

_LEGAL_CLAIM_PATTERNS = [
    re.compile(
        r"\b(?:court|judge|ruling|holding|opinion|decision|order|judgment|"
        r"statute|regulation|law|constitution|amendment|precedent|"
        r"case|matter|cause\s+of\s+action|claim|count|charge|motion|"
        r"appeal|petition|complaint|indictment|verdict|sentence|"
        r"injunction|warrant|subpoena)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:plaintiff|defendant|petitioner|respondent|appellant|appellee|"
        r"prosecution|defense|counsel|attorney|party|parties|witness|"
        r"jury|grand\s+jury|magistrate|tribunal|bench)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:liability|negligence|breach|damages?|remedy|relief|"
        r"standing|jurisdiction|venue|due\s+process|equal\s+protection|"
        r"probable\s+cause|reasonable\s+doubt|burden\s+of\s+proof|"
        r"standard\s+of\s+review|mens\s+rea|actus\s+reus|"
        r"tort|contract|fiduciary|duty|rights?|violation|"
        r"guilty|innocent|convicted|acquitted|sentenced)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:evidence|testimony|exhibit|record|deposition|affidavit|"
        r"stipulation|discovery|interrogator(?:y|ies)|admission|"
        r"hearsay|privilege|objection|sustained|overruled)\b",
        re.IGNORECASE,
    ),
]

_LEGAL_FALSE_POSITIVE_PATTERNS = [
    # Polite / conversational usage
    re.compile(r"\bmay\s+I\b", re.IGNORECASE),
    re.compile(r"\bmay\s+(?:we|you)\s+(?:help|assist)\b", re.IGNORECASE),
    # "generally" in casual / non-legal speech
    re.compile(r"\bgenerally\s+(?:speaking|happy|nice|good|fine|okay)\b",
               re.IGNORECASE),
    # Idiomatic
    re.compile(r"\b(?:come what may|be that as it may)\b", re.IGNORECASE),
]

# --- Scientific domain ---

_SCIENTIFIC_CLAIM_PATTERNS = [
    re.compile(
        r"\b(?:result|finding|observation|experiment|analysis|study|"
        r"research|investigation|hypothesis|theory|model|framework|"
        r"method|approach|technique|algorithm|metric|measure|"
        r"dataset|benchmark|baseline|evaluation|ablation|comparison)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:performance|accuracy|precision|recall|f1|auc|bleu|rouge|"
        r"loss|error|rate|score|improvement|degradation|gain|"
        r"correlation|regression|classification|prediction|"
        r"training|validation|test(?:ing)?|inference|"
        r"convergence|overfit|underfit|generali[sz]ation)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:variable|parameter|coefficient|estimate|confidence|"
        r"significance|p-?value|effect\s+size|variance|standard\s+"
        r"(?:deviation|error)|mean|median|distribution|sample|"
        r"population|statistic(?:al|s)?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:figure|table|section|equation|appendix|"
        r"previous(?:ly)?|prior\s+(?:work|research|studies)|"
        r"we\s+(?:find|found|show|observe|note|report|demonstrate|"
        r"propose|introduce|present|evaluate|compare|achieve))\b",
        re.IGNORECASE,
    ),
]

_SCIENTIFIC_FALSE_POSITIVE_PATTERNS = [
    re.compile(r"\bmay\s+I\b", re.IGNORECASE),
    re.compile(r"\bmay\s+(?:we|you)\s+(?:help|assist)\b", re.IGNORECASE),
    re.compile(r"\b(?:come what may|be that as it may)\b", re.IGNORECASE),
    # "significant other" is not a scientific claim
    re.compile(r"\bsignificant\s+other\b", re.IGNORECASE),
    # "preliminary" in non-research contexts
    re.compile(r"\bpreliminary\s+(?:hearing|round|match|bout|race|"
               r"interview|sketch|drawing|draft)\b", re.IGNORECASE),
]

# Registry mapping domain names to their patterns and functions
_DOMAIN_REGISTRY: Dict[str, dict] = {
    "clinical": {
        "find_fn": find_hedges,
        "density_fn": hedge_density,
        "claim_patterns": _CLINICAL_CLAIM_PATTERNS,
        "false_positive_patterns": _CLINICAL_FALSE_POSITIVE_PATTERNS,
    },
    "legal": {
        "find_fn": find_legal_hedges,
        "density_fn": legal_hedge_density,
        "claim_patterns": _LEGAL_CLAIM_PATTERNS,
        "false_positive_patterns": _LEGAL_FALSE_POSITIVE_PATTERNS,
    },
    "scientific": {
        "find_fn": find_scientific_hedges,
        "density_fn": scientific_hedge_density,
        "claim_patterns": _SCIENTIFIC_CLAIM_PATTERNS,
        "false_positive_patterns": _SCIENTIFIC_FALSE_POSITIVE_PATTERNS,
    },
}


# ============================================================================
# Core context validation logic
# ============================================================================

def _is_contextually_valid(
    text: str,
    match: HedgeMatch,
    claim_patterns: List[re.Pattern],
    false_positive_patterns: List[re.Pattern],
    min_claim_signals: int = 1,
) -> bool:
    """
    Determine whether a regex hedge match represents genuine epistemic
    uncertainty in context, rather than a surface-level false positive.

    Strategy:
      1. If the sentence containing the hedge matches a known false-positive
         pattern that overlaps the hedge span, reject it.
      2. If the sentence contains at least `min_claim_signals` domain-relevant
         claim indicators, accept the hedge as genuine.
      3. Otherwise reject: the hedge word appears in a non-domain context.

    Args:
        text: Full document text.
        match: The HedgeMatch to validate.
        claim_patterns: Compiled regexes for domain-relevant claim content.
        false_positive_patterns: Compiled regexes for known false positives.
        min_claim_signals: Minimum number of distinct claim patterns that must
            match in the sentence for the hedge to be accepted.

    Returns:
        True if the hedge is contextually valid (genuine uncertainty).
    """
    sentence, sent_start, sent_end = _extract_sentence(
        text, match.start, match.end
    )

    if not sentence.strip():
        return False

    # Step 1: Check for explicit false-positive patterns in the sentence
    for fp_pat in false_positive_patterns:
        fp_match = fp_pat.search(sentence)
        if fp_match:
            # The false-positive pattern must overlap with or be near the
            # hedge span to count as a disqualification.  Map fp_match
            # offsets back to document coordinates.
            fp_doc_start = sent_start + fp_match.start()
            fp_doc_end = sent_start + fp_match.end()
            # Check for overlap or adjacency (within 5 chars)
            if (fp_doc_start <= match.end + 5) and (fp_doc_end >= match.start - 5):
                return False

    # Step 2: Count how many claim-pattern families fire in this sentence
    claim_hits = 0
    for cp in claim_patterns:
        if cp.search(sentence):
            claim_hits += 1

    return claim_hits >= min_claim_signals


# ============================================================================
# Public API
# ============================================================================

def find_contextual_hedges(
    text: str,
    domain: str = "clinical",
    context_window: int = 80,
    min_claim_signals: int = 1,
) -> List[HedgeMatch]:
    """
    Find uncertainty/hedge markers that are contextually valid for the domain.

    Same interface as find_hedges() but filters out false positives by
    checking whether each hedge co-occurs with domain-relevant content
    in the same sentence.

    Args:
        text: Document text.
        domain: One of "clinical", "legal", "scientific".
        context_window: Characters of context to include around each match.
        min_claim_signals: Minimum number of domain claim patterns that must
            appear in the hedge's sentence for it to be retained.

    Returns:
        List of HedgeMatch objects, sorted by position, with false positives
        removed.

    Raises:
        ValueError: If domain is not recognized.
    """
    if domain not in _DOMAIN_REGISTRY:
        raise ValueError(
            f"Unknown domain {domain!r}. "
            f"Choose from: {', '.join(sorted(_DOMAIN_REGISTRY))}"
        )

    config = _DOMAIN_REGISTRY[domain]
    # Step 1: Get all candidate hedges from the regex-based screener
    candidates = config["find_fn"](text, context_window=context_window)

    # Step 2: Filter by contextual validity
    valid = [
        m for m in candidates
        if _is_contextually_valid(
            text,
            m,
            claim_patterns=config["claim_patterns"],
            false_positive_patterns=config["false_positive_patterns"],
            min_claim_signals=min_claim_signals,
        )
    ]

    return valid


def contextual_hedge_density(
    text: str,
    domain: str = "clinical",
    min_claim_signals: int = 1,
) -> dict:
    """
    Compute hedge density metrics using context-filtered hedges.

    Works for all three domains: clinical, legal, scientific.

    Args:
        text: Document text.
        domain: One of "clinical", "legal", "scientific".
        min_claim_signals: Minimum claim signals for contextual filtering.

    Returns:
        Dict with total count, per-category counts, density per 1000 chars,
        and the number of raw candidates that were filtered out.
    """
    if domain not in _DOMAIN_REGISTRY:
        raise ValueError(
            f"Unknown domain {domain!r}. "
            f"Choose from: {', '.join(sorted(_DOMAIN_REGISTRY))}"
        )

    config = _DOMAIN_REGISTRY[domain]

    # Raw candidates (before filtering)
    raw_matches = config["find_fn"](text)
    raw_total = len(raw_matches)

    # Contextually filtered matches
    filtered_matches = find_contextual_hedges(
        text, domain=domain, min_claim_signals=min_claim_signals
    )
    total = len(filtered_matches)

    by_category: Dict[str, int] = {}
    for m in filtered_matches:
        by_category[m.category] = by_category.get(m.category, 0) + 1

    text_len = max(len(text), 1)
    return {
        "total_hedges": total,
        "hedges_per_1k_chars": round(total / text_len * 1000, 2),
        "by_category": by_category,
        "raw_candidates": raw_total,
        "filtered_out": raw_total - total,
        "precision_lift": (
            round(1.0 - total / raw_total, 3) if raw_total > 0 else 0.0
        ),
    }


def classify_contextual_uncertainty(
    text: str,
    domain: str = "clinical",
) -> str:
    """
    Classify a document as low/medium/high uncertainty density using
    context-filtered hedges.

    Args:
        text: Document text.
        domain: One of "clinical", "legal", "scientific".

    Returns:
        One of "low", "medium", "high".
    """
    stats = contextual_hedge_density(text, domain=domain)
    total = stats["total_hedges"]

    # Thresholds are calibrated per domain
    if domain == "clinical":
        diag = stats["by_category"].get("diagnostic_hedge", 0)
        if diag >= 5 or total >= 12:
            return "high"
        elif diag >= 2 or total >= 5:
            return "medium"
        else:
            return "low"
    elif domain == "legal":
        legal_h = stats["by_category"].get("legal_hedge", 0)
        if legal_h >= 5 or total >= 12:
            return "high"
        elif legal_h >= 2 or total >= 5:
            return "medium"
        else:
            return "low"
    else:  # scientific
        results_h = stats["by_category"].get("results_hedge", 0)
        if results_h >= 5 or total >= 12:
            return "high"
        elif results_h >= 2 or total >= 5:
            return "medium"
        else:
            return "low"


# ============================================================================
# Self-test
# ============================================================================

if __name__ == "__main__":
    # --- Clinical examples ---
    clinical_text = """
    The patient presented with chest pain. CT angiography suggestive of
    pulmonary embolism, cannot rule out aortic dissection. Likely viral
    pneumonia given clinical presentation. Troponin mildly elevated,
    of uncertain significance. Recommend further evaluation with
    echocardiogram if clinically indicated. No definitive evidence of
    acute coronary syndrome. Per patient, symptoms started 3 days ago.
    """

    non_clinical_text = """
    May I help you with your order? The weather has been mild lately.
    Our stable relationship with the vendor is significant for the
    company. Consider the fact that prices may change. He appears to
    be a generally happy person.
    """

    print("=== Clinical text (should retain most hedges) ===")
    ctx_matches = find_contextual_hedges(clinical_text, domain="clinical")
    raw_matches = find_hedges(clinical_text)
    print(f"  Raw hedges: {len(raw_matches)}, Contextual: {len(ctx_matches)}")
    for m in ctx_matches:
        print(f"    [{m.category}] '{m.text}'")

    print("\n=== Non-clinical text (should filter most hedges) ===")
    ctx_matches = find_contextual_hedges(non_clinical_text, domain="clinical")
    raw_matches = find_hedges(non_clinical_text)
    print(f"  Raw hedges: {len(raw_matches)}, Contextual: {len(ctx_matches)}")
    for m in ctx_matches:
        print(f"    [{m.category}] '{m.text}'")

    print("\n=== Density comparison ===")
    print(f"  Clinical density: {contextual_hedge_density(clinical_text)}")
    print(f"  Non-clinical density: {contextual_hedge_density(non_clinical_text)}")
