"""Every file-system location used by the package, in one place.

Nothing is hard-coded to a machine. Each location has a default relative to the
repository root and can be overridden with an environment variable, either
exported in the shell or written in a `.env` file at the repository root
(copy `configs/paths.example.env` to `.env` and edit it; `.env` is git-ignored).

Restricted data (PhysioNet credentialed access) is never shipped. Point
EE_MIMIC_ROOT / EE_RADGRAPH_ROOT at your own copy; see docs/DATA.md.

  EE_ROOT           repository root (default: the directory containing configs/)
  EE_DATA_DIR       rebuilt samples, *.parquet            (default: $EE_ROOT/data)
  EE_EXTERNAL_DIR   public corpora downloaded by scripts/data/download_public_data.sh
                                                          (default: $EE_DATA_DIR/external)
  EE_OUTPUTS_DIR    model generations, one JSON per generation
                                                          (default: $EE_ROOT/outputs)
  EE_RESULTS_DIR    scores, analyses, tables, figures     (default: $EE_ROOT/results)
  EE_MIMIC_ROOT     local PhysioNet download containing mimiciv/3.1 and mimic-iv-note/2.2
                                                          (restricted; default: $EE_ROOT/mimic)
  EE_RADGRAPH_ROOT  local PhysioNet download of RadGraph 1.0.0
                                                          (restricted; default: $EE_ROOT/radgraph)
  EE_MODELS_ROOT    optional directory of local Hugging Face checkpoints; if a model
                    is not found there it is loaded from the Hub at the pinned
                    revision in configs/models.py     (default: $EE_ROOT/models)
  EE_BIOSCOPE_DIR   BioScope abstracts.xml and full_papers.xml (default: $EE_EXTERNAL_DIR/bioscope)
  EE_NUBES_DIR      NUBes / IULA+ corpus                  (default: $EE_EXTERNAL_DIR/nubes)
  EE_MODEL_OVERRIDE optional: load this checkpoint path instead of the one resolved from
                    configs/models.py (src/generator.py; e.g. a copy on fast local disk)
"""
import os
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """Minimal .env reader (KEY=VALUE lines); never overrides variables already set."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and value:
            os.environ.setdefault(key, value)


_load_dotenv(_REPO / ".env")


def _path(var: str, default: Path) -> Path:
    return Path(os.environ.get(var, default)).expanduser()


PROJECT_ROOT = _path("EE_ROOT", _REPO)
DATA_DIR = _path("EE_DATA_DIR", PROJECT_ROOT / "data")
EXTERNAL_DIR = _path("EE_EXTERNAL_DIR", DATA_DIR / "external")
OUTPUTS_DIR = _path("EE_OUTPUTS_DIR", PROJECT_ROOT / "outputs")
RESULTS_DIR = _path("EE_RESULTS_DIR", PROJECT_ROOT / "results")
MODELS_ROOT = _path("EE_MODELS_ROOT", PROJECT_ROOT / "models")
PROMPTS_DIR = PROJECT_ROOT / "prompts"

# --- MIMIC-IV (restricted) ---
MIMIC_ROOT = _path("EE_MIMIC_ROOT", PROJECT_ROOT / "mimic")
MIMIC_NOTES = MIMIC_ROOT / "mimic-iv-note" / "2.2" / "note"
MIMIC_HOSP = MIMIC_ROOT / "mimiciv" / "3.1" / "hosp"
MIMIC_ICU = MIMIC_ROOT / "mimiciv" / "3.1" / "icu"
DISCHARGE_NOTES = MIMIC_NOTES / "discharge.csv.gz"
RADIOLOGY_NOTES = MIMIC_NOTES / "radiology.csv.gz"
ADMISSIONS = MIMIC_HOSP / "admissions.csv.gz"
SERVICES = MIMIC_HOSP / "services.csv.gz"
DIAGNOSES_ICD = MIMIC_HOSP / "diagnoses_icd.csv.gz"
D_ICD_DIAGNOSES = MIMIC_HOSP / "d_icd_diagnoses.csv.gz"
PATIENTS = MIMIC_HOSP / "patients.csv.gz"
LABEVENTS = MIMIC_HOSP / "labevents.csv.gz"
PRESCRIPTIONS = MIMIC_HOSP / "prescriptions.csv.gz"
TRANSFERS = MIMIC_HOSP / "transfers.csv.gz"
ICUSTAYS = MIMIC_ICU / "icustays.csv.gz"

# --- RadGraph (restricted) ---
RADGRAPH_ROOT = _path("EE_RADGRAPH_ROOT", PROJECT_ROOT / "radgraph")

# --- Public corpora (downloaded, never redistributed) ---
MULTILEXSUM_DIR = EXTERNAL_DIR / "multilexsum" / "releases" / "v20230518"
SCITLDR_DIR = EXTERNAL_DIR / "scitldr" / "scitldr_data" / "SciTLDR-Data" / "SciTLDR-AIC"
BIOSCOPE_DIR = _path("EE_BIOSCOPE_DIR", EXTERNAL_DIR / "bioscope")
NUBES_DIR = _path("EE_NUBES_DIR", EXTERNAL_DIR / "nubes")

# --- Local checkpoints (optional). Keys are the short model names used everywhere. ---
from configs.models import MODELS  # noqa: E402

LOCAL_MODELS = {name: MODELS_ROOT / spec["local_dir"] for name, spec in MODELS.items()}
STUDY_MODELS = ["llama-3.3-70b", "med42-70b", "llama-3.1-8b", "med42-8b"]
EXTENDED_MODELS = ["mistral-7b", "biomistral-7b", "openbio-70b"]
MULTIMODAL_MODELS = {"medgemma-27b", "gemma-4-31b", "gemma-4-moe", "gemma-4-4b"}


def require(path: Path, what: str, env_var: str) -> Path:
    """Return `path` if it exists, else stop with an actionable message."""
    if not Path(path).exists():
        raise SystemExit(
            f"\n[missing input] {what} not found at:\n    {path}\n"
            f"Set {env_var} (in your shell or in .env) and see docs/DATA.md for how to obtain it.\n")
    return Path(path)
