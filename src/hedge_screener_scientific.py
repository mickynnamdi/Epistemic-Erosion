"""
Hedge / uncertainty marker screener for scientific text.

Identifies epistemic uncertainty expressions in research papers,
abstracts, and TLDRs. Adapted from clinical hedge_screener.py.
"""

import re
from dataclasses import dataclass
from typing import List

from src.hedge_screener import HedgeMatch, _deduplicate_overlaps


SCIENTIFIC_HEDGE_PATTERNS = {
    # Results hedging
    "results_hedge": [
        r"\b(?:suggests?|indicating|implies?|hints? at)\b",
        r"\b(?:appears? to|seems? to)\b",
        r"\b(?:may|might|could)\s+(?:be|have|indicate|suggest|explain|contribute|lead|cause|play)\b",
        r"\b(?:possibly|potentially|conceivably|presumably|ostensibly)\b",
        r"\b(?:likely|unlikely|probable|plausible)\b",
        r"\b(?:tends? to|is apt to)\b",
    ],

    # Evidential hedging
    "evidential_hedge": [
        r"\b(?:our (?:results?|findings?|data|analysis|observations?|experiments?)\s+(?:suggest|indicate|show|demonstrate|reveal|support))\b",
        r"\b(?:the (?:results?|findings?|data|evidence)\s+(?:suggest|indicate|are consistent with|support|point to))\b",
        r"\b(?:based on (?:our|these|the|limited|preliminary) (?:results?|findings?|data|evidence|observations?))\b",
        r"\b(?:consistent with|in (?:line|agreement|accordance) with)\b",
    ],

    # Scope/generalizability hedges
    "scope_hedge": [
        r"\b(?:in (?:our|this|the present) (?:study|experiment|analysis|work|setting|context))\b",
        r"\b(?:under (?:these|certain|specific|our) (?:conditions|settings|assumptions|constraints))\b",
        r"\b(?:for (?:this|the|our) (?:particular|specific) (?:task|dataset|domain|setting))\b",
        r"\b(?:on (?:this|the|our) (?:benchmark|dataset|task))\b",
        r"\b(?:to (?:some|a certain|a limited) (?:extent|degree))\b",
        r"\b(?:largely|partially|somewhat|relatively|comparatively)\b",
    ],

    # Limitation hedges
    "limitation_hedge": [
        r"\b(?:limited (?:to|by|in|sample|data|scope))\b",
        r"\b(?:preliminary|exploratory|initial|pilot)\b",
        r"\b(?:further (?:research|investigation|study|work|experiments?) (?:is|are) (?:needed|required|warranted|necessary))\b",
        r"\b(?:remains? (?:to be|unclear|an open|unknown))\b",
        r"\b(?:beyond the scope of)\b",
        r"\b(?:(?:one|a) (?:limitation|caveat|concern|drawback) (?:of|is))\b",
    ],

    # Quantitative hedging
    "quantitative_hedge": [
        r"\b(?:approximately|roughly|about|around|nearly|close to)\b",
        r"\b(?:marginal|modest|slight|small|negligible|minor)\b",
        r"\b(?:substantial|significant|considerable|notable|marked)\b",
        r"\b(?:outperforms?|underperforms?|comparable to|competitive with|on par with)\b",
    ],
}

_COMPILED_SCI = {}
for category, patterns in SCIENTIFIC_HEDGE_PATTERNS.items():
    _COMPILED_SCI[category] = [re.compile(p, re.IGNORECASE) for p in patterns]


def find_scientific_hedges(text: str, context_window: int = 80) -> List[HedgeMatch]:
    """Find all uncertainty/hedge markers in scientific text."""
    matches = []
    for category, compiled in _COMPILED_SCI.items():
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


def scientific_hedge_density(text: str) -> dict:
    """Compute hedge density metrics for scientific text."""
    matches = find_scientific_hedges(text)
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
