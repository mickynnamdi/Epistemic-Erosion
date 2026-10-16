"""Build every result table from your own runs and check it against the paper.

Inputs
    results/scores/*.csv         from scripts/score_generations.py
    results/analysis/*.json      from the other scripts in scripts/analysis and scripts/validation
Outputs
    results/tables/*.csv         the paper's tables, from your runs
    results/PAPER_CHECK.md       every printed paper value next to the value you reproduced

A check is skipped (not failed) when its inputs have not been produced yet, e.g. when you
have no PhysioNet access. Only files produced by this package's scripts are read.

Usage:  python scripts/make_tables.py
"""
import json
import math
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from configs.paths import RESULTS_DIR  # noqa: E402
from analysis import core_tables as ct  # noqa: E402
from analysis.paper_values import APPROXIMATE, KNOWN_DIFFERENCES, PAPER  # noqa: E402


def _have(*studies):
    return all((RESULTS_DIR / "scores" / f"{s}.csv").exists() for s in studies)


def compute():
    values, tables = {}, {}
    if _have("depth"):
        depth = ct.load("depth")
        t1, a18, v = ct.table1(depth)
        tables["table1_models"], tables["tableA18_distortion_types"] = t1, a18
        values.update(v)
        if _have("crossdomain"):
            xd = ct.load("crossdomain")
            values.update(ct.table_a1(depth, ct.load("breadth") if _have("breadth") else None, xd))
            values.update(ct.appendix_b_clusters(depth, xd))
            values.update(ct.crossdomain(depth, xd))
            tab, v = ct.length(depth, xd)
            tables["tableA4_length_quintiles"] = tab
            values.update(v)
        if _have("mitigation", "temperature"):
            tab, v = ct.mitigation(depth, ct.load("mitigation"), ct.load("temperature"))
            tables["tablesA5_A6_A7_mitigation"] = tab
            values.update(v)
    if _have("hosted"):
        tab, v = ct.hosted(ct.load("hosted"))
        tables["table2_hosted"] = tab
        values.update(v)
    for f in sorted((RESULTS_DIR / "analysis").glob("*.json")) if (RESULTS_DIR / "analysis").exists() else []:
        values.update({k: x for k, x in json.load(open(f)).items() if not k.startswith("_")})
    return values, tables


def rnd(x, k):
    """Round half up (as printed in the paper), not Python's round-half-to-even."""
    return float(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-k), rounding=ROUND_HALF_UP))


def check(value, spec):
    kind, expected = spec[0], spec[1]
    if kind == "dp":
        return math.isclose(rnd(value, spec[2]), expected, abs_tol=10 ** -(spec[2] + 3))
    if kind == "int":
        return int(rnd(value, 0)) == int(expected)
    if kind == "sci":
        if value == 0:
            return expected == 0
        e = math.floor(math.log10(abs(expected)))
        return math.isclose(rnd(float(value) / 10 ** e, spec[2] - 1), round(expected / 10 ** e, spec[2] - 1), abs_tol=1e-9)
    if kind == "lt":
        return float(value) < expected
    if kind == "ge":
        return float(value) >= expected
    raise ValueError(kind)


def fmt(spec):
    kind, x = spec[0], spec[1]
    return {"dp": lambda: f"{x:.{spec[2]}f}", "int": lambda: f"{int(x):,}", "sci": lambda: f"{x:.{spec[2] - 1}e}",
            "lt": lambda: f"< {x:g}", "ge": lambda: f">= {x:g}"}[kind]()


def main():
    values, tables = compute()
    out = RESULTS_DIR / "tables"
    out.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(out / f"{name}.csv", index=False)
    rows, n_ok, n_fail, n_skip, n_approx = [], 0, 0, 0, 0
    for key, spec in PAPER.items():
        where = spec[-1]
        if key not in values or values[key] is None or (isinstance(values[key], float) and math.isnan(values[key])):
            rows.append((where, key, fmt(spec), "not computed", "skipped"))
            n_skip += 1
            continue
        ok = check(values[key], spec)
        approx = next((why for pre, why in APPROXIMATE.items() if key.startswith(pre)), None)
        if ok:
            n_ok += 1
            result = "match"
        elif approx:
            n_approx += 1
            result = f"approximate ({approx})"
        else:
            n_fail += 1
            result = "MISMATCH" + (f" (known difference: {KNOWN_DIFFERENCES[key]})" if key in KNOWN_DIFFERENCES else "")
        rows.append((where, key, fmt(spec), f"{values[key]:.6g}", result))
    lines = ["# Paper check", "",
             "Every value printed in the paper that this package recomputes, compared at the",
             "paper's printed precision with the value produced by your run of the scripts.", "",
             f"**{n_ok} match, {n_fail} mismatch, {n_approx} approximate, {n_skip} not computed** "
             "(not computed: the inputs have not been produced).", "",
             "Exact agreement is expected when the scores are computed from the paper's generations; outputs you",
             "regenerate yourself differ in wording (GPU non-determinism) and give close, not identical, values.", "",
             "| Where | Check | Paper | Your run | Result |", "|---|---|---|---|---|"]
    lines += [f"| {w} | `{k}` | {p} | {y} | {r} |" for w, k, p, y, r in rows]
    (RESULTS_DIR / "PAPER_CHECK.md").write_text("\n".join(lines) + "\n")
    print(f"tables -> {out}\ncheck  -> {RESULTS_DIR / 'PAPER_CHECK.md'}")
    print(f"{n_ok} match, {n_fail} mismatch, {n_approx} approximate, {n_skip} not computed")
    for w, k, p, y, r in rows:
        if r.startswith("MISMATCH"):
            print(f"  MISMATCH {k}: paper {p}, run {y}  ({w})")


if __name__ == "__main__":
    main()
