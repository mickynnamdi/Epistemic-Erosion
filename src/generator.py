"""
LLM generation pipeline for clinical text tasks.

Open-weight models are loaded from EE_MODELS_ROOT if a local copy exists, else from
the Hugging Face Hub at the revision pinned in configs/models.py. Hosted (Azure)
models are driven by scripts/hosted/.

Supports two backends:
  - vLLM server (an OpenAI-compatible endpoint at http://localhost:8000/v1)
  - HuggingFace transformers (no server needed; the default of the released scripts)
"""

import json
import time
import hashlib
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.paths import OUTPUTS_DIR, MODELS_ROOT, MULTIMODAL_MODELS
from configs.models import resolve as resolve_model
from prompts.task_prompts import (
    SYSTEM_PROMPT, SYSTEM_PROMPT_MITIGATION,
    SYSTEM_PROMPT_FEWSHOT, SYSTEM_PROMPT_POSTHOC, POSTHOC_SUFFIX,
    TASK_A_PROMPT, TASK_B_PROMPT, TASK_C_PROMPT,
    REFERRAL_TARGETS,
)

# Models that don't support system role — merge system into user message
NO_SYSTEM_ROLE = {"medgemma-27b", "medgemma-4b", "biomistral-7b",
                  "gemma-4-31b", "gemma-4-moe", "gemma-4-4b"}


def _device_map():
    """EE_FORCE_GPU=1 places the whole model on GPU 0 and raises an error if it does not fit.
    The default "auto" map offloads layers to CPU when it judges GPU memory
    insufficient (for example when a leftover process holds memory on the device), which
    makes generation 20-30x slower without any error."""
    import os
    return {"": 0} if os.environ.get("EE_FORCE_GPU") == "1" else "auto"


@dataclass
class GenerationRequest:
    hadm_id: int
    task: str                # "summary", "referral", "discharge_instructions"
    model: str               # short name from configs/models.py
    source_text: str
    system_prompt: str
    user_prompt: str
    temperature: float = 0.0
    max_tokens: int = 2048
    mitigation: str = "none"
    specialty: str = ""


@dataclass
class GenerationResult:
    request: GenerationRequest
    output_text: str
    model_id: str
    latency_seconds: float
    input_tokens: int
    output_tokens: int
    timestamp: str
    error: Optional[str] = None


def build_prompt(task: str, source_text: str, specialty: str = "") -> str:
    """Build the user prompt for a given task."""
    if task == "summary":
        return TASK_A_PROMPT.format(source_text=source_text)
    elif task == "referral":
        referral_spec = REFERRAL_TARGETS.get(specialty, "Cardiology")
        return TASK_B_PROMPT.format(
            source_text=source_text,
            referral_specialty=referral_spec,
        )
    elif task == "discharge_instructions":
        return TASK_C_PROMPT.format(source_text=source_text)
    else:
        raise ValueError(f"Unknown task: {task}")


def _build_messages(request: GenerationRequest) -> list:
    """Build chat messages, handling models that lack system role support."""
    if request.model in NO_SYSTEM_ROLE:
        # Merge system prompt into user message for Gemma
        combined = f"{request.system_prompt}\n\n{request.user_prompt}"
        return [{"role": "user", "content": combined}]
    else:
        return [
            {"role": "system", "content": request.system_prompt},
            {"role": "user", "content": request.user_prompt},
        ]


# ---------------------------------------------------------------------------
# vLLM backend
# ---------------------------------------------------------------------------

def generate_vllm(request: GenerationRequest, base_url: str = "http://localhost:8000/v1") -> GenerationResult:
    """Generate using a local vLLM server."""
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key="not-needed")
    t0 = time.time()

    try:
        response = client.chat.completions.create(
            model=request.model,
            messages=_build_messages(request),
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        output = response.choices[0].message.content
        return GenerationResult(
            request=request,
            output_text=output,
            model_id=response.model,
            latency_seconds=time.time() - t0,
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )
    except Exception as e:
        return GenerationResult(
            request=request, output_text="", model_id=request.model,
            latency_seconds=time.time() - t0,
            input_tokens=0, output_tokens=0,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"), error=str(e),
        )


# ---------------------------------------------------------------------------
# HuggingFace transformers backend
# ---------------------------------------------------------------------------

class HFModel:
    """Wrapper that loads a HF model once and reuses it for multiple generations."""

    def __init__(self, model_key: str):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        # A local checkpoint under EE_MODELS_ROOT is used if present; otherwise the
        # pinned Hugging Face revision from configs/models.py. EE_MODEL_OVERRIDE
        # forces a specific path (e.g. a copy on fast local disk).
        import os
        override = os.environ.get("EE_MODEL_OVERRIDE")
        model_path, revision = (override, None) if override else resolve_model(model_key, MODELS_ROOT)
        rev = {"revision": revision} if revision else {}
        self.model_key = model_key
        self.is_multimodal = model_key in MULTIMODAL_MODELS
        print(f"Loading model: {model_key} from {model_path}")
        t0 = time.time()

        if self.is_multimodal:
            from transformers import AutoProcessor, AutoModelForImageTextToText
            self.tokenizer = AutoProcessor.from_pretrained(
                model_path, trust_remote_code=True, **rev,
            )
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_path, **rev,
                torch_dtype=torch.bfloat16,
                device_map=_device_map(),
                trust_remote_code=True,
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True, **rev,
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, **rev,
                torch_dtype=torch.bfloat16,
                device_map=_device_map(),
                trust_remote_code=True,
            )
        self.model.eval()
        self.model_path = model_path

        print(f"Model loaded in {time.time() - t0:.1f}s")

    def generate(self, request: GenerationRequest) -> GenerationResult:
        import torch

        t0 = time.time()
        try:
            messages = _build_messages(request)

            if self.is_multimodal:
                # Multimodal processors: use the underlying tokenizer directly
                prompt_text = self.tokenizer.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                )
                encoded = self.tokenizer.tokenizer(prompt_text, return_tensors="pt")
                input_ids = encoded["input_ids"].to(self.model.device)
            else:
                encoded = self.tokenizer.apply_chat_template(
                    messages,
                    return_tensors="pt",
                    add_generation_prompt=True,
                )
                # apply_chat_template returns BatchEncoding in transformers 5.x
                if hasattr(encoded, "input_ids"):
                    input_ids = encoded["input_ids"]
                else:
                    input_ids = encoded
                input_ids = input_ids.to(self.model.device)

            n_input = input_ids.shape[1]

            generate_kwargs = {
                "max_new_tokens": request.max_tokens,
                "do_sample": False,
            }
            # Only set temperature if sampling
            if request.temperature > 0:
                generate_kwargs["do_sample"] = True
                generate_kwargs["temperature"] = request.temperature

            with torch.no_grad():
                output_ids = self.model.generate(input_ids, **generate_kwargs)

            output_text = self.tokenizer.decode(
                output_ids[0][n_input:], skip_special_tokens=True,
            )
            n_output = output_ids.shape[1] - n_input

            return GenerationResult(
                request=request,
                output_text=output_text,
                model_id=self.model_path,
                latency_seconds=time.time() - t0,
                input_tokens=n_input,
                output_tokens=n_output,
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            return GenerationResult(
                request=request, output_text="", model_id=request.model,
                latency_seconds=time.time() - t0,
                input_tokens=0, output_tokens=0,
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"), error=str(e),
            )


# Global model cache — load once, reuse across generations
_HF_MODEL_CACHE = {}


def generate_hf(request: GenerationRequest) -> GenerationResult:
    """Generate using HuggingFace transformers (model loaded once and cached)."""
    if request.model not in _HF_MODEL_CACHE:
        _HF_MODEL_CACHE[request.model] = HFModel(request.model)
    return _HF_MODEL_CACHE[request.model].generate(request)


def generate(request: GenerationRequest, backend: str = "vllm", **kwargs) -> GenerationResult:
    """Route generation to the appropriate backend."""
    if backend == "vllm":
        return generate_vllm(request, **kwargs)
    elif backend == "hf":
        return generate_hf(request)
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'vllm' or 'hf'.")


# ---------------------------------------------------------------------------
# Batch generation
# ---------------------------------------------------------------------------

def run_generation_batch(
    episodes_df,
    models: list,
    tasks: list,
    mitigation: str = "none",
    backend: str = "vllm",
    output_dir: Optional[Path] = None,
    max_source_chars: int = 12000,  # ~3000 tokens, safe for Med42 (8192 ctx) and Llama (131K ctx)
    vllm_base_url: str = "http://localhost:8000/v1",
) -> list:
    """
    Run generation for all episodes x models x tasks.

    Returns list of GenerationResult objects.
    """
    if output_dir is None:
        output_dir = OUTPUTS_DIR / f"generations_{mitigation}"
    output_dir.mkdir(parents=True, exist_ok=True)

    from prompts.task_prompts import (
        SYSTEM_PROMPT_STRICT, SYSTEM_PROMPT_COT, COT_SUFFIX,
        SYSTEM_PROMPT_COMBO, COMBO_SUFFIX, SYSTEM_PROMPT_CONSTRAINED,
    )
    MITIGATION_SYSTEMS = {
        "none": SYSTEM_PROMPT,
        "prompt": SYSTEM_PROMPT_MITIGATION,
        "fewshot": SYSTEM_PROMPT_FEWSHOT,
        "posthoc": SYSTEM_PROMPT_POSTHOC,
        "strict": SYSTEM_PROMPT_STRICT,
        "cot": SYSTEM_PROMPT_COT,
        "combo": SYSTEM_PROMPT_COMBO,
        "constrained": SYSTEM_PROMPT_CONSTRAINED,
    }
    # Suffixes appended to the user prompt for strategies that need one
    MITIGATION_SUFFIXES = {
        "posthoc": POSTHOC_SUFFIX,
        "cot": COT_SUFFIX,
        "combo": COMBO_SUFFIX,
    }
    system = MITIGATION_SYSTEMS.get(mitigation, SYSTEM_PROMPT)

    results = []
    total = len(episodes_df) * len(models) * len(tasks)
    done = 0

    for _, episode in episodes_df.iterrows():
        hadm_id = int(episode["hadm_id"])
        source = episode["discharge_text"][:max_source_chars]
        specialty = episode.get("specialty", "")

        for model in models:
            for task in tasks:
                done += 1
                print(f"[{done}/{total}] hadm_id={hadm_id} model={model} task={task}")

                user_prompt = build_prompt(task, source, specialty)
                if mitigation in MITIGATION_SUFFIXES:
                    user_prompt += MITIGATION_SUFFIXES[mitigation]
                request = GenerationRequest(
                    hadm_id=hadm_id,
                    task=task,
                    model=model,
                    source_text=source,
                    system_prompt=system,
                    user_prompt=user_prompt,
                    specialty=specialty,
                    mitigation=mitigation,
                )

                kwargs_for_gen = {"base_url": vllm_base_url} if backend == "vllm" else {}
                result = generate(request, backend=backend, **kwargs_for_gen)
                results.append(result)

                # Save individual result
                result_id = f"{hadm_id}_{model}_{task}_{mitigation}"
                result_path = output_dir / f"{result_id}.json"
                save_result(result, result_path)

                if result.error:
                    print(f"  ERROR: {result.error}")
                else:
                    print(f"  OK ({result.output_tokens} tokens, {result.latency_seconds:.1f}s)")

    return results


def save_result(result: GenerationResult, path: Path):
    """Save a generation result to JSON."""
    data = {
        "hadm_id": result.request.hadm_id,
        "task": result.request.task,
        "model": result.request.model,
        "mitigation": result.request.mitigation,
        "specialty": result.request.specialty,
        "system_prompt": result.request.system_prompt,
        "user_prompt": result.request.user_prompt[:500] + "...",
        "source_text_hash": hashlib.md5(result.request.source_text.encode()).hexdigest(),
        "source_text_length": len(result.request.source_text),
        "output_text": result.output_text,
        "model_id": result.model_id,
        "latency_seconds": result.latency_seconds,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "timestamp": result.timestamp,
        "error": result.error,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
