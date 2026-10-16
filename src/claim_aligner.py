"""
Claim alignment for the EFS pipeline (Section 4.2, Stage 2).

Each epistemically marked source claim is aligned to at most one output claim:

  * Exact anchor: a normalized-text match scores 1.0 and therefore always wins,
    which guarantees EFS(s, s) == 1.0.
  * Pair score: 0.65 * token-level F1 over content words + 0.35 * containment of
    the source claim's domain terms. Neither term is biased by candidate length.
  * One-to-one assignment with the Hungarian algorithm, maximizing the total score.
  * A threshold (tau = 0.30) below which a source claim is unmatched, i.e. an omission.

tau is the aligner's only free parameter. Over tau in [0.20, 0.50], mean EFS on 1,000
seed-123 clinical generations stays within 0.441-0.449, while the omission share of
detected distortions moves from 56.3% to 74.3% (Appendix J.5). The greedy baseline
aligner compared in Appendix J scores candidates by the raw count of shared terms
and takes the first maximum (many-to-one); it is in src/distortion_analyzer.py.
"""
from __future__ import annotations

import re
from contextlib import contextmanager
from typing import Dict, Iterator, List, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

# Below this score a source claim is treated as absent from the output.
MATCH_THRESHOLD = 0.30
_THRESHOLD_OVERRIDE: Optional[float] = None


@contextmanager
def match_threshold(tau: float) -> Iterator[None]:
    """Temporarily score with alignment threshold `tau` (threshold sweep, Appendix J.5).

    Not thread-safe: the override is module-global, which keeps every call site
    of `align_claims` unchanged and the default behaviour identical to the paper.
    """
    global _THRESHOLD_OVERRIDE
    previous, _THRESHOLD_OVERRIDE = _THRESHOLD_OVERRIDE, float(tau)
    try:
        yield
    finally:
        _THRESHOLD_OVERRIDE = previous

_WS = re.compile(r"\s+")
_NONWORD = re.compile(r"[^a-z0-9 ]+")

# Function words carry no claim identity; excluding them keeps token F1 from
# being inflated by shared boilerplate.
_STOP = {
    "the", "a", "an", "of", "and", "or", "to", "in", "on", "for", "with", "is",
    "was", "were", "are", "be", "been", "has", "have", "had", "at", "by", "as",
    "that", "this", "it", "its", "from", "he", "she", "his", "her", "they",
    "their", "there", "which", "who", "no", "not", "but", "if", "than", "then",
}


def _norm(text: str) -> str:
    """Normalize for exact-match comparison: case, punctuation, whitespace."""
    return _WS.sub(" ", _NONWORD.sub(" ", text.lower())).strip()


def _content_tokens(text: str) -> set:
    return {t for t in _norm(text).split() if t and t not in _STOP}


def _token_f1(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    p, r = inter / len(b), inter / len(a)
    return 2 * p * r / (p + r)


def _containment(s_terms: set, o_terms: set) -> float:
    """Fraction of the source claim's domain terms present in the candidate."""
    if not s_terms:
        return 0.0
    return len(s_terms & o_terms) / len(s_terms)


def _terms(claim: dict) -> set:
    """Clinical pipeline keys these 'clinical_terms'; cross-domain 'domain_terms'."""
    for k in ("clinical_terms", "domain_terms"):
        if k in claim:
            return {t.lower() for t in claim[k]}
    return set()


def score_pair(s_claim: dict, o_claim: dict,
               emb_sim: Optional[float] = None) -> float:
    """
    Similarity in [0, 1] between a source claim and a candidate output claim.

    An exact normalized-text match short-circuits to 1.0 so that identical text
    can never be mis-paired. Otherwise the score blends lexical overlap
    (length-normalized) with clinical-term containment, and optionally a
    sentence-embedding cosine when one is supplied.
    """
    s_text, o_text = s_claim["text"], o_claim["text"]
    if _norm(s_text) == _norm(o_text):
        return 1.0

    f1 = _token_f1(_content_tokens(s_text), _content_tokens(o_text))
    cont = _containment(_terms(s_claim), _terms(o_claim))
    if emb_sim is None:
        return 0.65 * f1 + 0.35 * cont
    return 0.45 * emb_sim + 0.35 * f1 + 0.20 * cont


def align_claims(source_claims: List[dict], output_claims: List[dict],
                 emb_matrix: Optional[np.ndarray] = None,
                 threshold: Optional[float] = None) -> Dict[int, Optional[int]]:
    """
    One-to-one assignment of source claims to output claims.

    Returns {source_idx: output_idx or None}. None means the source claim has no
    counterpart above `threshold` and should be reported as an omission.

    emb_matrix, if given, is [len(source_claims) x len(output_claims)] of cosine
    similarities in [0, 1].
    """
    if threshold is None:
        threshold = MATCH_THRESHOLD if _THRESHOLD_OVERRIDE is None else _THRESHOLD_OVERRIDE
    if not source_claims or not output_claims:
        return {i: None for i in range(len(source_claims))}

    S, O = len(source_claims), len(output_claims)
    score = np.zeros((S, O), dtype=float)
    for i, s in enumerate(source_claims):
        for j, o in enumerate(output_claims):
            e = float(emb_matrix[i, j]) if emb_matrix is not None else None
            score[i, j] = score_pair(s, o, e)

    # Hungarian maximizes total score; pad so it works for S != O.
    n = max(S, O)
    padded = np.zeros((n, n), dtype=float)
    padded[:S, :O] = score
    rows, cols = linear_sum_assignment(-padded)

    out: Dict[int, Optional[int]] = {}
    for r, c in zip(rows, cols):
        r, c = int(r), int(c)      # numpy ints hash like ints, but keep keys plain
        if r >= S:
            continue
        if c >= O or score[r, c] < threshold:
            out[r] = None          # genuine omission
        else:
            out[r] = c
    for i in range(S):
        out.setdefault(i, None)
    return out
