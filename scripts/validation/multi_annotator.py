"""Label each annotation-sheet pair with an AI annotator on the six-type human schema.

Paper: Appendix J.9 / Table A13 (Claude Sonnet 4.6, Llama-3.3-70B, gpt-oss-20B and Gemma-4-31B
as annotators; 164 pairs = 64 legal/scientific + 100 clinical). This script only produces the
labels; scripts/validation/multi_annotator_agreement.py computes the agreement.
Inputs: the legal/scientific sheet (--sheet-legal-scientific) and the clinical sheet
(--sheet-clinical, MIMIC-IV text); columns id, source_text, output_text. access: sheets are
withheld (clinical sheet: PhysioNet credentialed); GPU (Gemma-4-31B, Llama-3.3-70B bf16,
gpt-oss-20B) or paid API (Claude, ANTHROPIC_API_KEY from the environment only).
Outputs: RESULTS_DIR/multi_annotator/<model>_<backend>/<sheet>_<id>.json, one file per pair, with
<sheet> the fixed sheet names below whatever the CSV files are called locally.
Usage:
    python scripts/validation/multi_annotator.py --model gemma-4-31b --backend hf
    python scripts/validation/multi_annotator.py --model claude-sonnet-4-6 --backend anthropic \
        --sheet-legal-scientific path/ls.csv --sheet-clinical path/clinical.csv

Study settings, kept as run: the same prompt for every annotator; source cut to 3,000 and output
to 2,500 characters; open models through src.generator (weights resolved with configs.models,
bf16, greedy decoding, max 300 new tokens, or 2,500 for gpt-oss so its reasoning channel can
finish; a JSON-only system prompt, merged into the user turn for Gemma); Claude through forced
tool use (max_tokens 400, API-default temperature, as in the paper's run). Pairs already
labelled are skipped; pairs whose reply could not be parsed are retried on the next run.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.models import MODELS  # noqa: E402
from configs.paths import DATA_DIR, RESULTS_DIR  # noqa: E402

# Output files are keyed by these sheet names, independent of the local file names.
SHEET_LS = "annotation_sheet_legal_scientific"
SHEET_CLINICAL = "annotation_sheet_clinical"
DEFAULT_SHEETS = {SHEET_LS: DATA_DIR / "annotation_sheets" / f"{SHEET_LS}.csv",
                  SHEET_CLINICAL: DATA_DIR / "annotation_sheets" / f"{SHEET_CLINICAL}.csv"}
AI_DIR = RESULTS_DIR / "multi_annotator"

TYPES = ["escalation", "attenuation", "omission", "fabrication",
         "scope_shift", "severity_warp"]

PROMPT = """You are an expert annotator judging whether an AI SUMMARY distorts the CERTAINTY (not the factual content) of its SOURCE. Consider only how confidently claims are asserted.

For each of six distortion types, answer "yes" only if that distortion is present:
- escalation: a hedged/uncertain source claim is stated more confidently in the output (e.g., "possible X" -> "X").
- attenuation: a confident source claim is made more uncertain in the output.
- omission: a hedged source claim is dropped from the output entirely.
- fabrication: the output adds a hedge/qualifier that has no basis in the source.
- scope_shift: a source hedge is applied to a broader/narrower scope in the output.
- severity_warp: a severity/degree modifier (mild/moderate/severe) is changed or dropped.

SOURCE:
{source}

OUTPUT:
{output}

Respond with ONLY a JSON object:
{{"escalation":"yes|no","attenuation":"yes|no","omission":"yes|no","fabrication":"yes|no","scope_shift":"yes|no","severity_warp":"yes|no","overall_severity":"none|low|moderate|high"}}"""


def parse_labels(raw):
    """Robustly extract the 6 yes/no labels: try JSON, then per-key regex."""
    txt = re.sub(r"```(?:json)?", "", raw or "").strip()
    obj = None
    m = re.search(r"\{.*\}", txt, re.S)
    if m:
        cand = re.sub(r",(\s*[}\]])", r"\1", m.group(0))  # drop trailing commas
        try:
            obj = json.loads(cand)
        except Exception:
            obj = None
    if isinstance(obj, dict) and any(t in obj for t in TYPES):
        labels = {t: 1 if str(obj.get(t, "")).strip().lower() == "yes" else 0 for t in TYPES}
        return labels, obj.get("overall_severity")
    # regex fallback (handles minor JSON breakage)
    labels, got = {}, False
    for t in TYPES:
        mm = re.search(rf'"{t}"\s*:\s*"?(yes|no)"?', txt, re.I)
        labels[t] = 1 if (mm and mm.group(1).lower() == "yes") else 0
        got = got or bool(mm)
    if got:
        sev = re.search(r'"overall_severity"\s*:\s*"?(none|low|moderate|high)"?', txt, re.I)
        return labels, (sev.group(1) if sev else None)
    return None, None


def load_pairs(sheets):
    """sheets: {sheet name: csv path}. Rows with empty source or output are skipped."""
    rows = []
    for name, p in sheets.items():
        if not Path(p).exists():
            print(f"  [skip] {name}: {p} not found")
            continue
        df = pd.read_csv(p, dtype=str)
        for _, r in df.iterrows():
            s, o = str(r.get("source_text", "")), str(r.get("output_text", ""))
            if s.strip() and o.strip() and s != "nan" and o != "nan":
                rows.append({"id": str(r.get("id", len(rows))), "sheet": name,
                             "source": s, "output": o})
    return rows


def call_hf(model, prompt):
    from src.generator import GenerationRequest, generate
    # gpt-oss writes a long analysis channel before the final answer, so it gets a
    # larger token budget.
    mt = 2500 if "oss" in model.lower() else 300
    sys_p = ("You are a precise JSON-outputting annotator. After any reasoning, "
             "end your reply with the single JSON object and nothing after it."
             if "oss" in model.lower()
             else "You are a precise JSON-outputting annotator. Output only valid JSON.")
    req = GenerationRequest(hadm_id=0, task="annotate", model=model, source_text="",
                            system_prompt=sys_p,
                            user_prompt=prompt, temperature=0.0, max_tokens=mt)
    return generate(req, backend="hf").output_text


def call_anthropic(model, prompt):
    from anthropic import Anthropic
    # The client reads ANTHROPIC_API_KEY from the environment; no key file is read.
    c = Anthropic()
    # Forced tool use returns the labels in a fixed schema.
    yn = {"type": "string", "enum": ["yes", "no"]}
    tool = {
        "name": "submit_labels",
        "description": "Submit the six distortion-presence annotations.",
        "input_schema": {
            "type": "object",
            "properties": {**{t: yn for t in TYPES},
                           "overall_severity": {"type": "string",
                                                "enum": ["none", "low", "moderate", "high"]}},
            "required": TYPES + ["overall_severity"],
        },
    }
    m = c.messages.create(
        model=model, max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
        tools=[tool], tool_choice={"type": "tool", "name": "submit_labels"})
    for block in m.content:
        if getattr(block, "type", None) == "tool_use":
            return json.dumps(block.input)
    return ""


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True,
                    help="configs.models key for --backend hf (gemma-4-31b, llama-3.3-70b, gpt-oss-20b); "
                         "Anthropic model id for --backend anthropic (claude-sonnet-4-6)")
    ap.add_argument("--backend", default="hf", choices=["hf", "anthropic"])
    ap.add_argument("--sheet-legal-scientific", type=Path, default=DEFAULT_SHEETS[SHEET_LS])
    ap.add_argument("--sheet-clinical", type=Path, default=DEFAULT_SHEETS[SHEET_CLINICAL])
    args = ap.parse_args()
    if args.backend == "hf" and args.model not in MODELS:
        raise SystemExit(f"--model must be a configs.models key for --backend hf, got {args.model!r}")

    pairs = load_pairs({SHEET_LS: args.sheet_legal_scientific, SHEET_CLINICAL: args.sheet_clinical})
    print(f"{args.model} ({args.backend}): {len(pairs)} pairs to annotate")
    out_dir = AI_DIR / f"{args.model.replace('/','_')}_{args.backend}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, p in enumerate(pairs):
        rpath = out_dir / f"{p['sheet']}_{p['id']}.json"
        if rpath.exists():
            prev = json.load(open(rpath))
            if not prev.get("parse_error"):  # keep good results; retry failures
                continue
        prompt = PROMPT.format(source=p["source"][:3000], output=p["output"][:2500])
        try:
            raw = call_hf(args.model, prompt) if args.backend == "hf" else call_anthropic(args.model, prompt)
        except Exception as e:
            raw = ""
            print(f"  [{i}] error: {str(e)[:80]}")
        rec = {"id": p["id"], "sheet": p["sheet"], "raw": raw}
        labels, sev = parse_labels(raw)
        if labels is not None:
            rec.update(labels)
            rec["overall_severity"] = sev
        else:
            rec["parse_error"] = True
        json.dump(rec, open(rpath, "w"), indent=2)
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(pairs)}]")
    print(f"done -> {out_dir}")


if __name__ == "__main__":
    main()
