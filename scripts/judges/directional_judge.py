"""Independent directional LLM judge: is the output more, less or equally certain than its source?

Paper: Section 6.6 and Appendix J.8. Llama-3.3-70B judges 750 source/output pairs (clinical x
    Llama-3.1-8B, Med42-v2-8B, Mistral-7B; scientific x Med42-v2-8B, Mistral-7B; the first 150
    generations of each). Strict parser (plain JSON only): 541 of 750 accepted, 96.3% "more
    certain" (clinical 98.3%, n = 242; scientific 94.6%, n = 299), 2.8% "less certain".
    Recovering the 209 replies wrapped in a Markdown code fence: 97.1% over all 750 (clinical
    98.7%, scientific 94.7%), 2.0% less certain.
Inputs: EE_DATA_DIR/{full_sample,legal_sample,scientific_sample}.parquet, generations under
    EE_OUTPUTS_DIR (full_generations/<model>, cross_domain/<domain>/<model>, else
    fullscale/<domain>/<model>), prompt
    scripts/validation/directional_judge_prompt.txt; access: PhysioNet credentialed (clinical
    shards), public (scientific), GPU (2 x 80 GB, Llama-3.3-70B bf16) for the judge stage.
Outputs: RESULTS_DIR/judges/directional/<domain>_<model>/<id>_<task>.json (one per judged pair),
    RESULTS_DIR/analysis/directional_judge.json.
Usage:
    python scripts/judges/directional_judge.py --stage judge --domain clinical --model llama-3.1-8b
    python scripts/judges/directional_judge.py --stage analyze [--judgements DIR]
"""
import argparse
import glob
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from configs.paths import DATA_DIR, OUTPUTS_DIR, RESULTS_DIR, require  # noqa: E402

PROMPT_FILE = ROOT / "scripts" / "validation" / "directional_judge_prompt.txt"
SYS = "You are a precise JSON-outputting evaluator. Output only valid JSON."
DEFAULT_JUDGE = "llama-3.3-70b"
MAX_PAIRS, EXCERPT_CHARS, MAX_NEW = 150, 3000, 300
DOMAIN_SOURCE = {"clinical": ("full_sample.parquet", "hadm_id", "discharge_text"),
                 "legal": ("legal_sample.parquet", "case_id", "source_text"),
                 "scientific": ("scientific_sample.parquet", "paper_id", "source_text")}
GEN_DIR = {"clinical": "full_generations/{model}", "legal": "cross_domain/legal/{model}",
           "scientific": "cross_domain/scientific/{model}"}
# The shards reported in the paper (each complete at 150 pairs).
PAPER_SHARDS = ["clinical_llama-3.1-8b", "clinical_med42-8b", "clinical_mistral-7b",
                "scientific_med42-8b", "scientific_mistral-7b"]


def render(source, output):
    template = PROMPT_FILE.read_text().rstrip("\n")
    return template.replace("{source_excerpt}", source).replace("{output_excerpt}", output)


def load_pairs(domain, model):
    """Valid (source, output) pairs of one shard, in sorted file order."""
    parq, idcol, srccol = DOMAIN_SOURCE[domain]
    df = pd.read_parquet(require(DATA_DIR / parq, f"{domain} sample", "EE_DATA_DIR"))
    src_map = {str(i): str(t) for i, t in zip(df[idcol], df[srccol])}
    key = "hadm_id" if domain == "clinical" else "doc_id"
    gdir = OUTPUTS_DIR / GEN_DIR[domain].format(model=model)
    if domain != "clinical" and not gdir.exists():
        # The paper judged an earlier run in cross_domain/, which scripts/generate/generate.py does not
        # produce; without it, the scored cross-domain generations in fullscale/ are judged instead.
        gdir = OUTPUTS_DIR / "fullscale" / domain / model
        print(f"  note: cross_domain/{domain}/{model} not found; judging fullscale/{domain}/{model}")
    gdir = require(gdir, f"{domain} generations of {model}", "EE_OUTPUTS_DIR")
    pairs = []
    for jf in sorted(glob.glob(str(gdir / "*.json"))):
        d = json.load(open(jf))
        if d.get("error") and str(d["error"]) != "None":
            continue
        did = str(d.get(key))
        source = src_map.get(did)
        if not source or not d.get("output_text"):
            continue
        pairs.append({"id": did, "task": d.get("task"), "source_text": source, "output_text": d["output_text"]})
    return pairs


def stage_judge(a):
    from src.generator import GenerationRequest, generate
    pairs = load_pairs(a.domain, a.model)[:a.max_pairs]
    shard = f"{a.domain}_{a.model}" if a.judge == DEFAULT_JUDGE else f"{a.judge}__{a.domain}_{a.model}"
    out_dir = RESULTS_DIR / "judges" / "directional" / shard
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{a.domain}/{a.model}: {len(pairs)} pairs | judge={a.judge}", flush=True)
    for i, p in enumerate(pairs):
        rpath = out_dir / f"{p['id']}_{p['task']}.json"
        if rpath.exists():      # resume
            continue
        req = GenerationRequest(hadm_id=0, task="judge", model=a.judge, source_text="", system_prompt=SYS,
                                user_prompt=render(p["source_text"][:EXCERPT_CHARS], p["output_text"][:EXCERPT_CHARS]),
                                temperature=0.0, max_tokens=MAX_NEW)
        res = generate(req, backend="hf")
        json.dump({"judge": a.judge, "domain": a.domain, "model": a.model, "id": p["id"], "task": p["task"],
                   "judge_raw": res.output_text, "error": res.error}, open(rpath, "w"), indent=2)
        if (i + 1) % 20 == 0:
            print(f"  [{i + 1}/{len(pairs)}]", flush=True)


def shift_strict(raw):
    """The parser used for the paper's headline: the reply must be plain JSON."""
    try:
        return json.loads(raw).get("certainty_shift")
    except Exception:  # noqa: BLE001
        return None


def shift_recovered(raw):
    """Also accept a reply that is one JSON object wrapped in a Markdown code fence."""
    s = shift_strict(raw)
    if s is not None:
        return s
    m = re.fullmatch(r"\s*```(?:json)?\s*(.*?)\s*```\s*", str(raw), re.S)
    if m:
        try:
            return json.loads(m.group(1)).get("certainty_shift")
        except Exception:  # noqa: BLE001
            return None
    return None


def rates(df, col, tag):
    ok = df[df[col].notna()]
    v = {f"{tag}.n": len(ok), f"{tag}.more_certain_pct": 100 * float((ok[col] == "more_certain").mean()),
         f"{tag}.less_certain_pct": 100 * float((ok[col] == "less_certain").mean())}
    for dom, g in ok.groupby("domain"):
        v[f"{tag}.{dom}.n"] = len(g)
        v[f"{tag}.{dom}.more_certain_pct"] = 100 * float((g[col] == "more_certain").mean())
    return v


def stage_analyze(a):
    recs = []
    for shard in PAPER_SHARDS:
        files = sorted(f for f in glob.glob(str(a.judgements / shard / "*.json"))
                       if not Path(f).name.startswith("_"))
        if not files:
            raise SystemExit(f"no judgements for shard {shard} under {a.judgements}")
        recs += [json.load(open(f)) for f in files]
    df = pd.DataFrame(recs)
    df["strict"] = df["judge_raw"].map(shift_strict)
    df["recovered"] = df["judge_raw"].map(shift_recovered)
    v = {"S6.6.n_submitted": len(df)}
    v.update(rates(df, "strict", "S6.6.strict"))
    v.update(rates(df, "recovered", "J8.recovered"))
    v["J8.code_fenced_recovered"] = int(df["recovered"].notna().sum() - df["strict"].notna().sum())
    (RESULTS_DIR / "analysis").mkdir(parents=True, exist_ok=True)
    json.dump(v, open(RESULTS_DIR / "analysis" / "directional_judge.json", "w"), indent=1)
    for k, x in v.items():
        print(f"  {k:42s} {x}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=["judge", "analyze"])
    ap.add_argument("--domain", choices=list(DOMAIN_SOURCE))
    ap.add_argument("--model", help="model whose generations are judged (e.g. llama-3.1-8b)")
    ap.add_argument("--judge", default=DEFAULT_JUDGE)
    ap.add_argument("--max-pairs", type=int, default=MAX_PAIRS)
    ap.add_argument("--judgements", type=Path, default=RESULTS_DIR / "judges" / "directional",
                    help="directory with one sub-folder per shard (<domain>_<model>)")
    a = ap.parse_args()
    if a.stage == "judge":
        if not (a.domain and a.model):
            ap.error("--stage judge needs --domain and --model")
        stage_judge(a)
    else:
        stage_analyze(a)


if __name__ == "__main__":
    main()
