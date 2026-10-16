"""Rebuild the 20 source-output pairs the human annotation app served to every rater.

Paper: Appendix K (20-pair human pilot) and Table A13 (AI annotators vs human majority). The
annotation app did not show sheet rows 1..20: it drew a stratified random subsample of the
64-row legal/scientific sheet and renumbered it 1..20, so annotation id k must be joined to
sheet row MAP[k]. This module reproduces that draw exactly.
Inputs: the legal/scientific annotation sheet CSV (columns id, domain, model, task, efs_quartile,
source_text, output_text, ...); access: withheld with the rater files (not shipped).
Outputs: none (prints the mapping annotation id -> sheet row id; --json also prints it as JSON).
Usage:
    python scripts/validation/annotation_sampling.py --sheet path/to/annotation_sheet_legal_scientific.csv
    from scripts.validation.annotation_sampling import sample_pairs, annotation_to_sheet_row
"""
import argparse
import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.paths import DATA_DIR  # noqa: E402

N_ANNOTATE = 20          # pairs served to each rater
SEED = 42                # random.Random(42) in the app
DEFAULT_SHEET = DATA_DIR / "annotation_sheets" / "annotation_sheet_legal_scientific.csv"


def read_sheet(path):
    """Sheet rows as dicts, in file order (the order the app read them in)."""
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def sample_pairs(rows, n=N_ANNOTATE, seed=SEED):
    """Return the n sheet rows the app served, in served order, with `id` renumbered 1..n.

    Each returned row is a copy carrying `sheet_row_id` (the original sheet id). Strata are
    (efs_quartile, domain); every stratum gets n // n_strata rows, the first n % n_strata strata
    (in sorted order) one more; the pooled draw is then shuffled with the same generator.
    If the sheet has no more than n rows, all rows are returned unchanged in file order.
    """
    if n >= len(rows):
        return [dict(r, sheet_row_id=int(r["id"])) for r in rows]
    rng = random.Random(seed)
    buckets = {}
    for r in rows:
        buckets.setdefault((r.get("efs_quartile", "unknown"), r.get("domain", "unknown")), []).append(r)
    keys = sorted(buckets)
    per_bucket, remainder = divmod(n, len(keys))
    picked = []
    for i, key in enumerate(keys):
        k = per_bucket + (1 if i < remainder else 0)
        picked.extend(rng.sample(buckets[key], min(k, len(buckets[key]))))
    rng.shuffle(picked)
    return [dict(r, id=i + 1, sheet_row_id=int(r["id"])) for i, r in enumerate(picked)]


def annotation_to_sheet_row(rows, n=N_ANNOTATE, seed=SEED):
    """Mapping {annotation id (1..n): sheet row id}."""
    return {p["id"]: p["sheet_row_id"] for p in sample_pairs(rows, n, seed)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sheet", type=Path, default=DEFAULT_SHEET,
                    help="legal/scientific annotation sheet CSV (default: %(default)s)")
    ap.add_argument("--n", type=int, default=N_ANNOTATE)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--json", action="store_true", help="also print the mapping as JSON")
    args = ap.parse_args()
    if not args.sheet.exists():
        raise SystemExit(f"[missing input] annotation sheet not found at {args.sheet} (pass --sheet)")

    pairs = sample_pairs(read_sheet(args.sheet), args.n, args.seed)
    print("annotation_id\tsheet_row_id\tdomain\tmodel\ttask\tefs_quartile")
    for p in pairs:
        print(f"{p['id']}\t{p['sheet_row_id']}\t{p['domain']}\t{p['model']}\t{p['task']}\t{p['efs_quartile']}")
    if args.json:
        print(json.dumps({p["id"]: p["sheet_row_id"] for p in pairs}))


if __name__ == "__main__":
    main()
