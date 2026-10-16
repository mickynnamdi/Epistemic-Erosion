"""EFS pipeline.

Scoring uses the one-to-one claim aligner of Section 4.2 (exact-match anchor,
0.65 * token-F1 + 0.35 * term containment, Hungarian assignment, tau = 0.30):

  * `src.efs.compute_efs`   -> all domains (the entry point to use)
  * `src.efs.compute_efs_greedy` -> the greedy baseline aligner of the Appendix J ablation

The cross-domain module reads EFS_ALIGNER ("hungarian" = one-to-one, "greedy" = greedy
baseline); the package sets "hungarian" by default.
"""
import os as _os

_os.environ.setdefault("EFS_ALIGNER", "hungarian")
