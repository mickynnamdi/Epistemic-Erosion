"""Preliminary multilingual probe: hedge-density retention when a model summarises an English source in another language.

Paper: Appendix F.7. Lexicons of 17 (English), 19 (Spanish, 18 distinct) and 16 (French)
patterns; hedge-density retention (summary density / English source density, hedges per
1,000 words): Gemma-4-E4B 0.955 / 0.531 / 0.411 and Llama-3.1-8B 1.541 / 0.562 / 0.434 (en / es / fr).
Inputs: the first 400 rows of the train split of GEM/cochrane-simplification (Devaraj et al.,
2021; CC BY 4.0; public), 80 of which are summarised; GPU for the generate stage.
Outputs: data/cochrane_sample.parquet; results/multilingual/<model>_<lang>/<i>.json;
results/analysis/multilingual_probe.json.
Usage:
    python scripts/analysis/multilingual_probe.py --stage data
    python scripts/analysis/multilingual_probe.py --stage generate --model gemma-4-4b --lang es
    python scripts/analysis/multilingual_probe.py --stage analyze [--outputs DIR]
Decoding: greedy, at most 400 new tokens, source truncated to 6,000 characters.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.paths import DATA_DIR, RESULTS_DIR, require  # noqa: E402

DATASET_URL = ("https://huggingface.co/datasets/GEM/cochrane-simplification/resolve/"
               "75a92ae445171fa1b7641a229bfe3c77c0d8723d/train.json")
N_SAMPLE, N_SUMMARISED = 400, 80
LANG_NAME = {"en": "English", "es": "Spanish", "fr": "French"}
MODELS = ["gemma-4-4b", "llama-3.1-8b"]
LEXICON = {   # small parallel lexicons of epistemic markers (verbatim from the paper's run)
    "en": [r"\bpossible\b", r"\bpossibly\b", r"\bprobable\b", r"\bprobably\b", r"\bmay\b",
           r"\bmight\b", r"\bcould\b", r"\bsuggests?\b", r"\bappears?\b", r"\bseems?\b",
           r"\buncertain\b", r"\bunclear\b", r"\blikely\b", r"\bpotential(?:ly)?\b",
           r"\blow[- ]quality evidence\b", r"\bcannot be (?:ruled out|excluded)\b", r"\bperhaps\b"],
    # NOTE: "probablemente" is listed twice, as in the paper's run (19 patterns, 18 distinct)
    "es": [r"\bposible(?:s)?\b", r"\bposiblemente\b", r"\bprobable(?:s)?\b", r"\bprobablemente\b",
           r"\bpuede(?:n)?\b", r"\bpodr[ií]a(?:n)?\b", r"\bsugiere(?:n)?\b", r"\bparece(?:n)?\b",
           r"\bincierto(?:s)?\b", r"\binciert[ao](?:s)?\b", r"\bpoco clara?\b", r"\bprobablemente\b",
           r"\bevidencia de baja calidad\b", r"\bno se puede(?:n)? (?:descartar|excluir)\b",
           r"\bquiz[áa]s?\b", r"\btal vez\b", r"\bpotencial(?:mente)?\b", r"\bpresunto\b", r"\bsospecha\b"],
    "fr": [r"\bpossible(?:s)?\b", r"\bpossiblement\b", r"\bprobable(?:s)?\b", r"\bprobablement\b",
           r"\bpeut\b", r"\bpourrait(?:ent)?\b", r"\bsugg[èe]re(?:nt)?\b", r"\bsemble(?:nt)?\b",
           r"\bincertain(?:e|s|es)?\b", r"\bpeu clair(?:e|s)?\b", r"\bpeut[- ]être\b",
           r"\bpreuve de faible qualit[ée]\b", r"\bne peut(?:vent)? [êe]tre (?:exclu|écart[ée])",
           r"\bpotentiel(?:le|lement)?\b", r"\bpr[ée]sum[ée]\b", r"\bsuspect[ée]\b"],
}


def density(text, lang):
    """Hedges per 1,000 words."""
    hits = sum(len(re.findall(p, text, flags=re.IGNORECASE)) for p in LEXICON[lang])
    return hits / max(len(text.split()), 1) * 1000


def stage_data():
    import urllib.request
    recs = json.loads(urllib.request.urlopen(DATASET_URL).read())
    df = pd.DataFrame(recs)[["source", "target"]].head(N_SAMPLE)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DATA_DIR / "cochrane_sample.parquet")
    print(f"wrote {len(df)} rows to {DATA_DIR / 'cochrane_sample.parquet'}")


def stage_generate(model, lang, backend):
    from src.generator import GenerationRequest, generate
    df = pd.read_parquet(require(DATA_DIR / "cochrane_sample.parquet", "Cochrane sample (--stage data)", "EE_DATA_DIR"))
    out = RESULTS_DIR / "multilingual" / f"{model}_{lang}"
    out.mkdir(parents=True, exist_ok=True)
    name = LANG_NAME[lang]
    system = (f"You are a medical writer. Write a faithful plain-language summary in {name}. "
              f"Preserve the authors' exact level of certainty: keep every hedge/qualifier "
              f"at the same strength as the source. Respond only in {name}.")
    for i, src in enumerate(df["source"].astype(str).head(N_SUMMARISED)):
        path = out / f"{i}.json"
        if path.exists():
            continue
        req = GenerationRequest(hadm_id=0, task="summary", model=model, source_text=src, system_prompt=system,
                                user_prompt=f"Source (English):\n\n{src[:6000]}\n\nSummary in {name}:",
                                temperature=0.0, max_tokens=400)
        res = generate(req, backend=backend)
        text = res.output_text or ""
        json.dump({"idx": i, "error": res.error, "src_density_en": round(density(src, "en"), 3),
                   "out_density": round(density(text, lang), 3), "out_words": len(text.split())},
                  open(path, "w"), indent=2)


def stage_analyze(root):
    v = {"F7.lexicon_patterns.en": len(LEXICON["en"]), "F7.lexicon_patterns.es": len(LEXICON["es"]),
         "F7.lexicon_patterns.fr": len(LEXICON["fr"])}
    for model in MODELS:
        for lang in LANG_NAME:
            d = root / f"{model}_{lang}"
            if not d.exists():
                continue
            rows = [json.load(open(p)) for p in sorted(d.glob("*.json")) if not p.name.startswith("_")]
            good = pd.DataFrame([r for r in rows if not r.get("error") and r["out_words"] > 20])
            v[f"F7.{model}.{lang}.n"] = len(good)
            # retention = mean summary density / mean source density (per-item densities are stored
            # rounded to 3 decimals, as in the paper's run)
            v[f"F7.{model}.{lang}.retention"] = float(good["out_density"].mean() / max(good["src_density_en"].mean(), 1e-6))
            print(f"{model:13s} {lang}: n={len(good)} retention={v[f'F7.{model}.{lang}.retention']:.3f}")
    (RESULTS_DIR / "analysis").mkdir(parents=True, exist_ok=True)
    json.dump(v, open(RESULTS_DIR / "analysis" / "multilingual_probe.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=["data", "generate", "analyze"])
    ap.add_argument("--model", default="gemma-4-4b", choices=MODELS)
    ap.add_argument("--lang", default="es", choices=list(LANG_NAME))
    ap.add_argument("--backend", default="hf", choices=["hf", "vllm"])
    ap.add_argument("--outputs", type=Path, default=None,
                    help="analyze: folder with <model>_<lang>/ sub-folders (default results/multilingual)")
    a = ap.parse_args()
    if a.stage == "data":
        stage_data()
    elif a.stage == "generate":
        stage_generate(a.model, a.lang, a.backend)
    else:
        stage_analyze(a.outputs or RESULTS_DIR / "multilingual")


if __name__ == "__main__":
    main()
