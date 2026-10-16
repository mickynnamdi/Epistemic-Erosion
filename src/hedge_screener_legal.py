"""
Hedge / uncertainty marker screener for legal text.

Identifies epistemic uncertainty expressions in court opinions, briefs,
and legal summaries. Adapted from clinical hedge_screener.py.
"""

import re
from dataclasses import dataclass
from typing import List

from src.hedge_screener import HedgeMatch, _deduplicate_overlaps


LEGAL_HEDGE_PATTERNS = {
    # Legal diagnostic uncertainty (holdings, rulings)
    "legal_hedge": [
        r"\b(?:arguably|arguable|potentially|conceivably)\b",
        r"\b(?:appears? to|seems? to)\b",
        r"\b(?:likely|unlikely|probable|improbable|plausible|implausible)\b",
        r"\b(?:may|might|could)\s+(?:be|have|constitute|violate|establish|warrant)\b",
        r"\b(?:suggests?|indicating|implies?|signals?)\s+that\b",
        r"\b(?:on balance|on its face|prima facie)\b",
        r"\b(?:presumed|presumptive|presumption of)\b",
        r"\b(?:alleged|allegedly|purported|purportedly)\b",
    ],

    # Evidential qualifiers
    "evidential_hedge": [
        r"\b(?:the record (?:suggests?|indicates?|reflects?))\b",
        r"\b(?:the evidence (?:suggests?|indicates?|supports?|weighs?))\b",
        r"\b(?:based on|in light of|given)\s+(?:the (?:record|evidence|facts?))\b",
        r"\b(?:insufficient (?:evidence|basis|showing|proof))\b",
        r"\b(?:no (?:clear|definitive|conclusive|sufficient) (?:evidence|proof|showing|basis))\b",
        r"\b(?:fails? to (?:establish|demonstrate|show|prove))\b",
    ],

    # Standard of review / burden qualifiers
    "standard_hedge": [
        r"\b(?:preponderance of|beyond a reasonable doubt|clear and convincing)\b",
        r"\b(?:reasonable (?:person|jury|basis|grounds|suspicion|belief))\b",
        r"\b(?:substantial (?:evidence|basis|likelihood|probability))\b",
        r"\b(?:mere (?:speculation|conjecture|allegation|assertion))\b",
        r"\b(?:genuine (?:dispute|issue|question) of (?:material )?fact)\b",
    ],

    # Conditional / hedged holdings
    "conditional_hedge": [
        r"\b(?:assuming (?:arguendo|without deciding|for purposes of))\b",
        r"\b(?:even (?:assuming|if|were (?:we|the court) to))\b",
        r"\b(?:to the extent that)\b",
        r"\b(?:insofar as)\b",
        r"\b(?:absent (?:a showing|evidence|proof))\b",
    ],

    # Scope qualifiers
    "scope_hedge": [
        r"\b(?:in (?:some|certain|limited|narrow) circumstances)\b",
        r"\b(?:under (?:some|certain|particular|specific) conditions)\b",
        r"\b(?:generally|typically|ordinarily|usually|often|sometimes)\b",
        r"\b(?:in (?:most|many|some) cases)\b",
        r"\b(?:not (?:necessarily|always|inevitably))\b",
    ],
}

_COMPILED_LEGAL = {}
for category, patterns in LEGAL_HEDGE_PATTERNS.items():
    _COMPILED_LEGAL[category] = [re.compile(p, re.IGNORECASE) for p in patterns]


def find_legal_hedges(text: str, context_window: int = 80) -> List[HedgeMatch]:
    """Find all uncertainty/hedge markers in legal text."""
    matches = []
    for category, compiled in _COMPILED_LEGAL.items():
        for pattern in compiled:
            for m in pattern.finditer(text):
                start, end = m.start(), m.end()
                ctx_start = max(0, start - context_window)
                ctx_end = min(len(text), end + context_window)
                matches.append(HedgeMatch(
                    text=m.group(),
                    category=category,
                    start=start,
                    end=end,
                    context=text[ctx_start:ctx_end],
                ))
    matches.sort(key=lambda m: (m.start, -m.end))
    return _deduplicate_overlaps(matches)


def legal_hedge_density(text: str) -> dict:
    """Compute hedge density metrics for legal text."""
    matches = find_legal_hedges(text)
    total = len(matches)
    by_category = {}
    for m in matches:
        by_category[m.category] = by_category.get(m.category, 0) + 1
    text_len = max(len(text), 1)
    return {
        "total_hedges": total,
        "hedges_per_1k_chars": round(total / text_len * 1000, 2),
        "by_category": by_category,
    }
