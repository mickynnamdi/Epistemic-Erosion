"""Generate the hosted-model outputs of Table 2 through Azure OpenAI / Azure AI Foundry.

Paper: Section 6.5, Table 2, Table A2, Appendix E (4 models x 3 domains x 3 seeds x
100 documents x 2 tasks = 7,200 requests; 7,005 scored).
Inputs: data/{legal,scientific,medical}_sample.parquet (public; rebuilt by scripts/data/),
data/hosted_samples/seed{0,1,2}.json (the document ids of each seed, as used in the paper),
and a paid Azure deployment: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY in the environment.
Outputs: outputs/frontier_seed<S>/<domain>/<model>/<doc>_<model>_<task>.json
Usage:
    python scripts/hosted/generate.py --deployment gpt-4o --seed 1 --domain legal
    python scripts/hosted/generate.py --deployment Phi-4 --seed 0 --domain all
Then score with: python scripts/score_generations.py --study hosted

Decoding: temperature 0 and max_tokens=1024; GPT-5.x and o-series deployments accept
no temperature, so they run at the API default with max_completion_tokens=1024.
Deployments used in the paper (Azure, API version 2024-10-21): Phi-4, gpt-4o (named
"gpt-4o-2" for seed 0), Mistral-Large-3, and gpt-5.4 (printed "GPT-5" in the paper;
see docs/MODELS.md). Existing output files are skipped, so a run can be resumed.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.paths import DATA_DIR, OUTPUTS_DIR, require  # noqa: E402
from prompts.hosted_prompts import HOSTED_DOMAINS  # noqa: E402

MAX_SOURCE_CHARS = 12000
MAX_OUTPUT_TOKENS = 1024
API_VERSION = "2024-10-21"
SAMPLES = Path(__file__).resolve().parents[2] / "data" / "hosted_samples"


def call(client, deployment, system, user):
    """One chat completion with retries on rate limits; returns (text, prompt_tokens, completion_tokens)."""
    reasoning = any(p in deployment.lower() for p in ("gpt-5", "o1", "o3", "o4"))
    kwargs = dict(model=deployment, messages=[{"role": "system", "content": system},
                                              {"role": "user", "content": user}])
    if reasoning:
        kwargs["max_completion_tokens"] = MAX_OUTPUT_TOKENS
    else:
        kwargs.update(max_tokens=MAX_OUTPUT_TOKENS, temperature=0)
    last = None
    for wait in (0, 5, 12, 30, 60, 120):
        time.sleep(wait)
        try:
            r = client.chat.completions.create(**kwargs)
            u = r.usage
            return r.choices[0].message.content or "", (u.prompt_tokens if u else 0), (u.completion_tokens if u else 0)
        except Exception as e:  # retry only on rate limits
            last = e
            if not ("429" in str(e) or "rate limit" in str(e).lower() or "RateLimitReached" in str(e)):
                raise
    raise last


def run(client, deployment, seed, domain):
    system, tasks, sample_file, id_col = HOSTED_DOMAINS[domain]
    ids = json.load(open(SAMPLES / f"seed{seed}.json"))[domain]
    df = pd.read_parquet(require(DATA_DIR / sample_file, f"{domain} sample", "EE_DATA_DIR"))
    df[id_col] = df[id_col].astype(str)
    df = df[df[id_col].isin(ids)].set_index(id_col)
    missing = sorted(set(ids) - set(df.index))
    if missing:
        print(f"  note: {len(missing)} of {len(ids)} {domain} ids are not in {sample_file}; they are skipped "
              f"(in the paper: 19 seed-0 biomedical abstracts, excluded from Table 2)")
    out_dir = OUTPUTS_DIR / f"frontier_seed{seed}" / domain / deployment.replace("/", "_")
    out_dir.mkdir(parents=True, exist_ok=True)
    done = err = skipped = 0
    for doc_id, row in df.iterrows():
        source = row["source_text"][:MAX_SOURCE_CHARS]
        for task, template in tasks.items():
            path = out_dir / f"{doc_id}_{deployment.replace('/', '_')}_{task}.json"
            if path.exists():
                skipped += 1
                continue
            rec = {"doc_id": str(doc_id), "task": task, "model": deployment, "provider": "azure-openai",
                   "source_text_length": len(source), "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
            try:
                text, p, c = call(client, deployment, system, template.format(source=source))
                rec.update(output_text=text, prompt_tokens=p, completion_tokens=c, error=None)
                done += 1
            except Exception as e:   # e.g. content-filter rejections; kept and excluded at scoring time
                rec.update(output_text="", error=str(e))
                err += 1
            path.write_text(json.dumps(rec, indent=2))
    print(f"  seed {seed} {domain} {deployment}: generated {done}, errors {err}, already present {skipped}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deployment", required=True, help="Azure deployment name, e.g. gpt-4o, Phi-4, gpt-5.4")
    ap.add_argument("--seed", type=int, required=True, choices=[0, 1, 2])
    ap.add_argument("--domain", default="all", choices=["legal", "scientific", "medical", "all"])
    a = ap.parse_args()
    endpoint, key = os.environ.get("AZURE_OPENAI_ENDPOINT"), os.environ.get("AZURE_OPENAI_API_KEY")
    if not endpoint or not key:
        raise SystemExit("Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY (see configs/paths.example.env).")
    from openai import AzureOpenAI
    client = AzureOpenAI(azure_endpoint=endpoint, api_key=key, api_version=API_VERSION)
    for domain in (HOSTED_DOMAINS if a.domain == "all" else [a.domain]):
        run(client, a.deployment, a.seed, domain)


if __name__ == "__main__":
    main()
