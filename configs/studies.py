"""Where each study's generations live, and which sample they were generated from.

Folder names (relative to EE_OUTPUTS_DIR) are those used for the paper's runs, so
scripts/generate/*.py write here and scripts/score_generations.py reads here.
One generation = one JSON file with at least: model, task, output_text, and an
id (doc_id, or hadm_id for clinical notes).
"""
from configs.paths import DATA_DIR

CLINICAL_SEEDS = {123: "full_sample.parquet", 0: "full_sample_seed0.parquet", 1: "full_sample_seed1.parquet"}

STUDIES = {
    # Section 6.1-6.2, Table 1, A18: 11 models x 3 tasks x 3 seeded samples (18,480 generations)
    "depth": {"domain": "clinical", "parts": [
        {"seed": 123, "sample": "full_sample.parquet",       "dir": "full_generations"},
        {"seed": 0,   "sample": "full_sample_seed0.parquet", "dir": "full_generations_seed0"},
        {"seed": 1,   "sample": "full_sample_seed1.parquet", "dir": "full_generations_seed1"},
    ]},
    # Section 6.1 / Appendix B specialty test: 3 models x 3 tasks x 1,000 episodes (9,000)
    "breadth": {"domain": "clinical", "parts": [
        {"seed": 1000, "sample": "full_sample_1000.parquet", "dir": "fullscale/clinical"},
    ]},
    # Section 6.4, Tables A5-A7: 7 prompting strategies (13,824)
    "mitigation": {"domain": "clinical", "parts": [
        {"seed": 123, "sample": "full_sample.parquet", "dir": "mitigation_generations"},
    ]},
    # Table A5 temperature rows: Llama-3.1-8B, 50 episodes x 3 tasks, T in {0.3, 0.7} (300)
    "temperature": {"domain": "clinical", "parts": [
        {"seed": 123, "sample": "full_sample.parquet", "dir": "temperature_ablation"},
    ]},
    # Section 6.5: 3 models x 2 tasks x 192 documents per domain (2 x 1,152)
    "crossdomain": {"parts": [
        {"domain": "legal",      "sample": "legal_sample.parquet",      "id_col": "case_id",  "dir": "fullscale/legal"},
        {"domain": "scientific", "sample": "scientific_sample.parquet", "id_col": "paper_id", "dir": "fullscale/scientific"},
    ]},
    # Table 2: 4 hosted models x 3 domains x 3 seeds x 100 documents x 2 tasks (7,200 requested)
    "hosted": {"parts": [
        {"seed": s, "domain": d, "pipeline": p, "sample": f, "id_col": c, "dir": f"frontier_seed{s}/{d}"}
        for s in (0, 1, 2)
        for d, p, f, c in (("legal", "legal", "legal_sample.parquet", "case_id"),
                           ("medical", "clinical", "medical_sample.parquet", "abstract_id"),
                           ("scientific", "scientific", "scientific_sample.parquet", "paper_id"))
    ]},
}


def sample_path(name: str):
    return DATA_DIR / name
