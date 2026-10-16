"""Registry of every model used in the paper.

Open-weight models are used exactly as published (no fine-tuning). `revision`
pins the Hugging Face commit of the checkpoint used for the paper's runs, so a
fresh download reproduces the same weights. `local_dir` is the folder name
looked up under EE_MODELS_ROOT first; if it is absent, the model is loaded from
the Hub at `revision`. See docs/MODELS.md for licences and hosted models.
"""
from pathlib import Path
from typing import Tuple

MODELS = {
    # short name       Hugging Face repo                          pinned commit                               local folder
    "llama-3.1-8b":  {"hf_id": "meta-llama/Llama-3.1-8B-Instruct",   "revision": "0e9e39f249a16976918f6564b8830bc894c89659", "local_dir": "Llama-3.1-8B-Instruct",   "params_b": 8,  "medical": False, "family": "Llama"},
    "llama-3.3-70b": {"hf_id": "meta-llama/Llama-3.3-70B-Instruct",  "revision": "6f6073b423013f6a7d4d9f39144961bfbfbc386b", "local_dir": "Llama-3.3-70B-Instruct",  "params_b": 70, "medical": False, "family": "Llama"},
    "med42-8b":      {"hf_id": "m42-health/Llama3-Med42-8B",         "revision": "ceab7e7ee4b9dbde7ba82867f34274db51487d83", "local_dir": "Llama3-Med42-8B",         "params_b": 8,  "medical": True,  "family": "Llama"},
    "med42-70b":     {"hf_id": "m42-health/Llama3-Med42-70B",        "revision": "867064e18aad7bf3f4795f20dcb25a1108952543", "local_dir": "Llama3-Med42-70B",        "params_b": 70, "medical": True,  "family": "Llama"},
    "openbio-70b":   {"hf_id": "aaditya/Llama3-OpenBioLLM-70B",      "revision": "7ad17ef0d2185811f731f89d20885b2f99b1e994", "local_dir": "OpenBioLLM-70B",          "params_b": 70, "medical": True,  "family": "Llama"},
    "mistral-7b":    {"hf_id": "mistralai/Mistral-7B-Instruct-v0.3", "revision": "c170c708c41dac9275d15a8fff4eca08d52bab71", "local_dir": "Mistral-7B-Instruct-v0.3", "params_b": 7,  "medical": False, "family": "Mistral"},
    "biomistral-7b": {"hf_id": "BioMistral/BioMistral-7B-DARE",      "revision": "588fca722fbffc46eab97cbe610d4203f303add8", "local_dir": "BioMistral-7B-DARE",      "params_b": 7,  "medical": True,  "family": "Mistral"},
    "medgemma-27b":  {"hf_id": "google/medgemma-27b-it",             "revision": "2d3e00ea38b50018bf5dd3aa1009457cd2d5a48f", "local_dir": "medgemma-27b-it",         "params_b": 27, "medical": True,  "family": "Gemma 3"},
    "gemma-4-4b":    {"hf_id": "google/gemma-4-E4B-it",              "revision": "ab4d2025e9fbb0b0dbbf5dbde081d3e546155586", "local_dir": "gemma-4-E4B-it",          "params_b": 4,  "medical": False, "family": "Gemma 4"},
    "gemma-4-31b":   {"hf_id": "google/gemma-4-31B-it",              "revision": "e51e7dcdb6febd74c182fe0cb41c236363ae2ac5", "local_dir": "gemma-4-31B-it",          "params_b": 31, "medical": False, "family": "Gemma 4"},
    "gemma-4-moe":   {"hf_id": "google/gemma-4-26B-A4B-it",          "revision": "1db3cff1840c2ae59759d8e842ff37831cf8cb63", "local_dir": "gemma-4-26B-A4B-it",      "params_b": 26, "medical": False, "family": "Gemma 4"},
    # used only by the token-distribution probe (Appendix M) and the AI-annotator study (Appendix J.9)
    "gemma-3-4b":    {"hf_id": "google/gemma-3-4b-it",               "revision": "093f9f388b31de276ce2de164bdc2081324b9767", "local_dir": "gemma-3-4b-it",           "params_b": 4,  "medical": False, "family": "Gemma 3"},
    "gpt-oss-20b":   {"hf_id": "openai/gpt-oss-20b",                 "revision": "6cee5e81ee83917806bbde320786a8fb61efebee", "local_dir": "gpt-oss-20b",             "params_b": 20, "medical": False, "family": "gpt-oss"},
}

# The 11 open-weight models of the main study, in the order of Table 1.
STUDY_11 = ["biomistral-7b", "llama-3.1-8b", "openbio-70b", "mistral-7b", "med42-8b", "llama-3.3-70b",
            "gemma-4-4b", "med42-70b", "gemma-4-moe", "gemma-4-31b", "medgemma-27b"]

PAPER_NAMES = {
    "biomistral-7b": "BioMistral-7B", "llama-3.1-8b": "Llama-3.1-8B", "openbio-70b": "OpenBioLLM-70B",
    "mistral-7b": "Mistral-7B-Instruct", "med42-8b": "Med42-v2-8B", "llama-3.3-70b": "Llama-3.3-70B",
    "gemma-4-4b": "Gemma-4-E4B", "med42-70b": "Med42-v2-70B", "gemma-4-moe": "Gemma-4-26B-MoE",
    "gemma-4-31b": "Gemma-4-31B", "medgemma-27b": "MedGemma-27B",
}


def resolve(name: str, models_root: Path) -> Tuple[str, str]:
    """Return (path_or_hub_id, revision) for `from_pretrained`.

    A local checkpoint under models_root/<local_dir> wins; otherwise the pinned Hub revision is used.
    """
    spec = MODELS[name]
    local = Path(models_root) / spec["local_dir"]
    if local.is_dir():
        return str(local), None
    return spec["hf_id"], spec["revision"]
