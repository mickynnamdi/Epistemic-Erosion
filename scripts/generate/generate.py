"""Generate the open-weight model outputs of every study in the paper.

Paper: Section 5 (setup) and every open-weight result:
  depth        Table 1, A18, Sec. 6.1-6.2   11 models x 3 tasks x 3 seeded MIMIC samples (18,480)
  breadth      Appendix B specialty test    Llama-3.1-8B, Med42-v2-8B, Mistral-7B x 3 tasks x 1,000 (9,000)
  mitigation   Tables A5-A7                 7 prompting strategies (13,824)
  temperature  Table A5                     Llama-3.1-8B, 50 episodes x 3 tasks, T in {0.3, 0.7} (300)
  crossdomain  Section 6.5                  3 models x 2 tasks x 192 legal / scientific documents (2 x 1,152)
Inputs: the samples in EE_DATA_DIR (MIMIC-IV for the clinical studies: PhysioNet credentialed);
model weights from EE_MODELS_ROOT or the Hugging Face Hub at the pinned revision (GPU; the
70B models need ~140 GB of GPU memory).
Outputs: one JSON per generation under EE_OUTPUTS_DIR, in the folders listed in configs/studies.py.
Usage:
    python scripts/generate/generate.py --study depth --seed 123 --model llama-3.1-8b
    python scripts/generate/generate.py --study mitigation --model gemma-4-4b --strategy fewshot
    python scripts/generate/generate.py --study temperature --temperature 0.3
    python scripts/generate/generate.py --study crossdomain --domain legal --model mistral-7b
Then score with scripts/score_generations.py --study <study>.

Decoding: greedy (do_sample=False), at most 2,048 new tokens (1,024 for the temperature
study), sources truncated to 12,000 characters, prompts of Appendix S (prompts/). Files that
already exist are skipped, so a run can be resumed. Greedy decoding on GPUs is not bit-exact
across hardware and library versions: regenerated outputs differ in wording from the paper's,
while the study-level EFS is stable.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.models import STUDY_11  # noqa: E402
from configs.paths import OUTPUTS_DIR, require  # noqa: E402
from configs.studies import CLINICAL_SEEDS, STUDIES, sample_path  # noqa: E402
from prompts.legal_prompts import LEGAL_SYSTEM_PROMPT, LEGAL_TASK_A_PROMPT, LEGAL_TASK_B_PROMPT  # noqa: E402
from prompts.scientific_prompts import (SCIENTIFIC_SYSTEM_PROMPT, SCIENTIFIC_TASK_A_PROMPT,  # noqa: E402
                                        SCIENTIFIC_TASK_B_PROMPT)
from prompts.task_prompts import (COMBO_SUFFIX, COT_SUFFIX, POSTHOC_SUFFIX, SYSTEM_PROMPT,  # noqa: E402
                                  SYSTEM_PROMPT_COMBO, SYSTEM_PROMPT_CONSTRAINED, SYSTEM_PROMPT_COT,
                                  SYSTEM_PROMPT_FEWSHOT, SYSTEM_PROMPT_MITIGATION, SYSTEM_PROMPT_POSTHOC,
                                  SYSTEM_PROMPT_STRICT)
from src.generator import GenerationRequest, build_prompt, generate  # noqa: E402

MAX_SOURCE_CHARS = 12000
CLINICAL_TASKS = ["summary", "referral", "discharge_instructions"]
STRATEGIES = {  # strategy: (system prompt, suffix appended to the task prompt) -- Section 5, Appendix S
    "none": (SYSTEM_PROMPT, ""), "prompt": (SYSTEM_PROMPT_MITIGATION, ""), "fewshot": (SYSTEM_PROMPT_FEWSHOT, ""),
    "posthoc": (SYSTEM_PROMPT_POSTHOC, POSTHOC_SUFFIX), "strict": (SYSTEM_PROMPT_STRICT, ""),
    "cot": (SYSTEM_PROMPT_COT, COT_SUFFIX), "combo": (SYSTEM_PROMPT_COMBO, COMBO_SUFFIX),
    "constrained": (SYSTEM_PROMPT_CONSTRAINED, ""),
}
CROSS = {"legal": (LEGAL_SYSTEM_PROMPT, {"case_summary": LEGAL_TASK_A_PROMPT, "legal_memo": LEGAL_TASK_B_PROMPT}),
         "scientific": (SCIENTIFIC_SYSTEM_PROMPT, {"paper_summary": SCIENTIFIC_TASK_A_PROMPT,
                                                   "plain_language": SCIENTIFIC_TASK_B_PROMPT})}
PAPER_MODELS = {  # models used per study in the paper
    "depth": STUDY_11, "breadth": ["llama-3.1-8b", "med42-8b", "mistral-7b"],
    "mitigation": ["llama-3.1-8b", "med42-8b", "mistral-7b", "gemma-4-4b"],
    "temperature": ["llama-3.1-8b"], "crossdomain": ["llama-3.1-8b", "med42-8b", "mistral-7b"],
}


def jobs(a):
    """Yield (doc_id, source, system, user, task, condition, out_path)."""
    if a.study == "crossdomain":
        part = next(p for p in STUDIES["crossdomain"]["parts"] if p["domain"] == a.domain)
        sample = part["sample"].replace("_sample.parquet", "_sample_full.parquet") if a.pool else part["sample"]
        df = pd.read_parquet(require(sample_path(sample), f"{a.domain} sample", "EE_DATA_DIR"))
        if a.pool:   # screened pool in id order: the detector transfer test uses the first documents
            df = df.sort_values(part["id_col"], key=lambda s: s.astype(str))
        system, tasks = CROSS[a.domain]
        out = OUTPUTS_DIR / part["dir"] / a.model
        for _, r in df.iterrows():
            src = str(r["source_text"])[:MAX_SOURCE_CHARS]
            for task, tpl in tasks.items():
                yield (str(r[part["id_col"]]), src, system, tpl.format(source_text=src), task, "none",
                       out / f"{r[part['id_col']]}_{a.model}_{task}.json")
        return
    if a.study == "breadth":
        sample, out, cond = "full_sample_1000.parquet", OUTPUTS_DIR / "fullscale" / "clinical" / a.model, "none"
    elif a.study == "depth":
        d = {123: "full_generations", 0: "full_generations_seed0", 1: "full_generations_seed1"}[a.seed]
        sample, out, cond = CLINICAL_SEEDS[a.seed], OUTPUTS_DIR / d / a.model, "none"
    elif a.study == "mitigation":
        sample, cond = CLINICAL_SEEDS[123], a.strategy
        out = OUTPUTS_DIR / "mitigation_generations" / f"{a.model}_{a.strategy}"
    else:  # temperature
        sample, cond = CLINICAL_SEEDS[123], f"T={a.temperature}"
        out = OUTPUTS_DIR / "temperature_ablation" / f"temp_{a.temperature}"
    df = pd.read_parquet(require(sample_path(sample), f"MIMIC sample {sample}", "EE_DATA_DIR"))
    if a.study == "temperature":
        df = df.sample(n=50, random_state=42)          # the paper's 50-episode subset
    system, suffix = STRATEGIES[a.strategy if a.study == "mitigation" else "none"]
    for _, r in df.iterrows():
        src, spec = str(r["discharge_text"])[:MAX_SOURCE_CHARS], str(r.get("specialty", ""))
        for task in CLINICAL_TASKS:
            tag = cond if a.study != "temperature" else f"temp{a.temperature}"
            yield (str(int(r["hadm_id"])), src, system, build_prompt(task, src, spec) + suffix, task, cond,
                   out / f"{int(r['hadm_id'])}_{a.model}_{task}_{tag}.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--study", required=True, choices=list(PAPER_MODELS))
    ap.add_argument("--model", default="llama-3.1-8b", help="short name from configs/models.py")
    ap.add_argument("--seed", type=int, default=123, choices=[123, 0, 1], help="depth study sample")
    ap.add_argument("--strategy", default="fewshot", choices=[s for s in STRATEGIES if s != "none"])
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--domain", default="legal", choices=list(CROSS))
    ap.add_argument("--pool", action="store_true",
                    help="crossdomain: generate for the screened pool instead of the 192-document sample "
                         "(only needed for the detector transfer test, Table A14; use with --limit 200)")
    ap.add_argument("--sampling-seed", type=int, default=0,
                    help="torch seed for the temperature study (the paper's runs were unseeded)")
    ap.add_argument("--backend", default="hf", choices=["hf", "vllm"])
    ap.add_argument("--limit", type=int, default=None, help="only the first N generations (smoke test)")
    a = ap.parse_args()
    if a.model not in PAPER_MODELS[a.study]:
        print(f"note: the paper ran {a.study} with {PAPER_MODELS[a.study]}, not {a.model}")
    temperature = a.temperature if a.study == "temperature" else 0.0
    max_tokens = 1024 if a.study == "temperature" else 2048
    if temperature > 0:
        import torch
        torch.manual_seed(a.sampling_seed)
    n_new = n_err = n_skip = 0
    for i, (doc_id, src, system, user, task, cond, path) in enumerate(jobs(a)):
        if a.limit is not None and i >= a.limit:
            break
        if path.exists():
            n_skip += 1
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        req = GenerationRequest(hadm_id=doc_id, task=task, model=a.model, source_text=src, system_prompt=system,
                                user_prompt=user, temperature=temperature, max_tokens=max_tokens,
                                mitigation=cond if a.study == "mitigation" else "none")
        res = generate(req, backend=a.backend)
        rec = {"doc_id": doc_id, "hadm_id": doc_id, "task": task, "model": a.model, "mitigation": req.mitigation,
               "system_prompt": system, "user_prompt_sha256": hashlib.sha256(user.encode()).hexdigest(),
               "source_text_hash": hashlib.md5(src.encode()).hexdigest(), "source_text_length": len(src),
               "output_text": res.output_text, "model_id": res.model_id, "input_tokens": res.input_tokens,
               "output_tokens": res.output_tokens, "latency_seconds": res.latency_seconds,
               "timestamp": res.timestamp, "error": res.error}
        if a.study == "temperature":
            rec["temperature"] = temperature
        path.write_text(json.dumps(rec, indent=2))
        n_new += 1
        n_err += bool(res.error)
    print(f"{a.study} {a.model}: generated {n_new} ({n_err} errors), already present {n_skip}")


if __name__ == "__main__":
    main()
