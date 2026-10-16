"""Entry point for the Epistemic Faithfulness Score (EFS).

    from src.efs import compute_efs
    r = compute_efs(source_text, output_text, domain="clinical")   # or "legal" / "scientific"
    r["efs_score"]     # max(0, 1 - |D| / max(|H(S)|, 1))
    r["by_type"]       # counts of the six distortion types
    r["distortions"]   # one record per detected distortion

`compute_efs` uses the one-to-one (Hungarian) claim aligner described in the
paper (Section 4.2); EFS(s, s) == 1.0 with zero distortions for every s.

Always pass `domain`. If it is omitted the domain is guessed from the source
text, which misroutes some documents (e.g. about a third of PubMedQA abstracts
are routed to the scientific or legal lexicon); every script in this
repository passes it explicitly.

`compute_efs_greedy` is the greedy baseline aligner used only for the ablations of
Appendix J (identity ceilings, greedy-aligner EFS) and to label the learned
detector's training data (Appendix N).
"""
from __future__ import annotations

import os
import warnings
from contextlib import nullcontext
from typing import Optional

from src.claim_aligner import MATCH_THRESHOLD, match_threshold
from src.distortion_analyzer import compute_efs as _compute_efs_greedy_raw
from src.distortion_analyzer_hungarian import compute_efs_hungarian

DOMAINS = ("clinical", "legal", "scientific")
ALIGNER = "hungarian"


def _check_domain(domain: Optional[str]) -> None:
    if domain is None:
        warnings.warn("compute_efs called without `domain`; it will be guessed from the text. "
                      "Pass domain='clinical'|'legal'|'scientific' to match the paper.", stacklevel=3)
    elif domain not in DOMAINS:
        raise ValueError(f"domain must be one of {DOMAINS} (or None), got {domain!r}")


def compute_efs(source_text: str, output_text: str, domain: Optional[str] = None,
                threshold: Optional[float] = None) -> dict:
    """EFS = max(0, 1 - |D| / max(|H(S)|, 1)) with the one-to-one aligner.

    threshold: alignment threshold tau (default 0.30, the paper's operating value).
    """
    _check_domain(domain)
    ctx = match_threshold(threshold) if threshold is not None else nullcontext()
    with ctx:
        return compute_efs_hungarian(source_text, output_text, domain=domain)


def compute_efs_greedy(source_text: str, output_text: str, domain: Optional[str] = None) -> dict:
    """EFS with the greedy baseline aligner (Appendix J ablations only), all domains."""
    _check_domain(domain)
    prev = os.environ.get("EFS_ALIGNER")
    os.environ["EFS_ALIGNER"] = "greedy"
    try:
        return _compute_efs_greedy_raw(source_text, output_text, domain=domain)
    finally:
        if prev is None:
            os.environ.pop("EFS_ALIGNER", None)
        else:
            os.environ["EFS_ALIGNER"] = prev


__all__ = ["compute_efs", "compute_efs_greedy", "compute_efs_hungarian", "DOMAINS", "MATCH_THRESHOLD",
           "ALIGNER"]
