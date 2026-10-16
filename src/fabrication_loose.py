"""Secondary fabrication detector (loose / bigram-content-overlap).

Operationalises §3.2 fabrication ("hedged claim with no basis in source") via
content-bigram overlap, complementing the primary detector's strict
zero-domain-term-overlap rule.

Spot check on 44 source-output pairs: 11 of the 16 flagged spans were fabrications
(precision 0.69; Appendix A). It is used as a secondary detector alongside the
strict one in the legal and scientific pipelines. Both paths return DistortionInstance(distortion_type="fabrication"),
but secondary detections carry source_context="[NOT FOUND IN SOURCE - loose
filter (bigram overlap=X)]" so downstream reporting can separate the two.
"""
import re

# Modal / evidential hedges that signal speculative extension. Conservative
# set — severity modifiers are excluded because dropping a severity modifier
# is a different distortion type (severity_warp).
_FABRICATION_HEDGE_RE = re.compile(
    r"\b(?:may|could|might|possible|possibly|potential|potentially|"
    r"likely|unlikely|probable|probably|presumed|suspected|"
    r"suggests?|suggested|suggestive|appears?|seems?|indicates?|"
    r"perhaps|somewhat|arguably|"
    r"should\s+(?:be|have|consider|order|provide|examine)|"
    r"would\s+(?:be|have|require|provide|need|order))\b",
    re.IGNORECASE,
)

# Content-word stopwords. Excluded tokens are not informative for "basis."
_STOP = {
    "this", "that", "these", "those", "with", "from", "have", "been", "were",
    "their", "would", "could", "should", "might", "must", "shall", "will",
    "than", "also", "such", "into", "upon", "over", "under", "more", "less",
    "many", "much", "very", "only", "even", "both", "each", "some", "other",
    "another", "where", "when", "what", "while", "before", "after", "which",
    "they", "them", "there", "here", "does", "done", "make", "made", "case",
    "cases", "study", "studies", "method", "methods", "approach", "model",
    "models", "data", "using", "used", "based", "like", "most",
}


def _content_bigrams(text: str):
    toks = [t for t in re.findall(r"\b[a-z]{3,}\b", text.lower()) if t not in _STOP]
    return set(zip(toks[:-1], toks[1:]))


def _split_sents(text: str):
    text = re.sub(r"\s+", " ", text).strip()
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text)
            if len(s.strip().split()) >= 5]


def detect_loose_fabrications(source: str, output: str, claim_pattern,
                              bigram_overlap_threshold: float = 0.30):
    """Yield (sentence, hedge, overlap) for output sentences that meet all of:
      (a) contain a modal/evidential hedge
      (b) contain a domain-relevant claim term (per `claim_pattern`)
      (c) share less than `bigram_overlap_threshold` content bigrams with source

    `claim_pattern` is a compiled regex; pass the domain's CLAIM_PATTERNS
    (CLINICAL/LEGAL/SCIENTIFIC) so the AND-with-domain-term requirement
    (Section 4.2, Stage 2) is enforced.
    """
    src_bg = _content_bigrams(source)
    for sent in _split_sents(output):
        if not _FABRICATION_HEDGE_RE.search(sent):
            continue
        if not claim_pattern.search(sent):
            continue
        sent_bg = _content_bigrams(sent)
        if not sent_bg:
            continue
        overlap = len(sent_bg & src_bg) / len(sent_bg)
        if overlap < bigram_overlap_threshold:
            m = _FABRICATION_HEDGE_RE.search(sent)
            yield sent, m.group(0), round(overlap, 3)
