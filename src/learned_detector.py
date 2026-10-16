"""
Learned Epistemic Distortion Detector.

Fine-tunes a transformer encoder on sentence pairs labelled by the rule-based
analyzer, to classify each pair as:
  - no_distortion
  - escalation
  - attenuation
  - omission
  - fabrication
  - scope_shift
  - severity_warp

Training data comes from the rule-based analyzer's output on 6,336
clinical generations (self-training approach). The paper's four encoders and
three seeds are run by scripts/detector/train.py (see docs/MODELS.md).

Usage:
  # Generate training data
  python -m src.learned_detector generate_data

  # Train classifier
  python -m src.learned_detector train

  # Predict
  python -m src.learned_detector predict --source "..." --output "..." --model-dir results/learned_detector/model_roberta-base_seed42
"""

import json
import glob
import random
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.paths import DATA_DIR, OUTPUTS_DIR, RESULTS_DIR


LABELS = ["no_distortion", "escalation", "attenuation", "omission",
          "fabrication", "scope_shift", "severity_warp"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}

DATA_PATH = RESULTS_DIR / "learned_detector"   # training pairs and checkpoints (MIMIC-derived, keep local)


def generate_training_data():
    """
    Extract sentence-pair training examples from rule-based analyzer results.

    For each distortion instance, we create a (source_sentence, output_sentence, label) triple.
    For negative examples, we sample aligned sentence pairs that have no distortion.
    """
    from src.distortion_analyzer import compute_efs, extract_clinical_claims

    DATA_PATH.mkdir(parents=True, exist_ok=True)

    episodes = pd.read_parquet(DATA_DIR / "full_sample.parquet")
    src_lookup = {int(r["hadm_id"]): r["discharge_text"][:12000] for _, r in episodes.iterrows()}

    examples = []
    # Use all available models for maximum training data
    models = ["llama-3.1-8b", "mistral-7b", "med42-8b", "llama-3.3-70b",
              "med42-70b", "biomistral-7b", "openbio-70b", "medgemma-27b",
              "gemma-4-4b", "gemma-4-31b", "gemma-4-moe"]

    for model in models:
        # sorted: the train/val split below depends on this order (the paper's run used unsorted glob order)
        files = sorted(glob.glob(str(OUTPUTS_DIR / "full_generations" / model / "*.json")))
        for jf in files:
            with open(jf) as f:
                d = json.load(f)
            if d.get("error") and d["error"] != "None":
                continue
            hadm_id = int(d["hadm_id"])
            source = src_lookup.get(hadm_id)
            if not source:
                continue

            efs = compute_efs(source, d["output_text"], domain="clinical")

            # Positive examples: distortion instances
            for dist in efs["distortions"]:
                examples.append({
                    "source_text": dist.source_span[:512] if dist.source_span else "",
                    "output_text": dist.output_span[:512] if dist.output_span else "",
                    "label": dist.distortion_type,
                    "model": model,
                    "task": d["task"],
                })

            # Negative examples: aligned claims with no distortion
            source_claims = extract_clinical_claims(source)
            output_claims = extract_clinical_claims(d["output_text"])

            for s_claim in source_claims[:3]:  # limit negatives
                if not s_claim["hedges"]:
                    continue
                s_terms = set(t.lower() for t in s_claim["clinical_terms"])
                if not s_terms:
                    continue
                for o_claim in output_claims:
                    o_terms = set(t.lower() for t in o_claim["clinical_terms"])
                    if len(s_terms & o_terms) > 0 and o_claim["hedges"]:
                        # Both have hedges and matching terms — likely preserved
                        s_str = s_claim["epistemic_strength"]
                        o_str = o_claim["epistemic_strength"]
                        if s_str == o_str:
                            examples.append({
                                "source_text": s_claim["text"][:512],
                                "output_text": o_claim["text"][:512],
                                "label": "no_distortion",
                                "model": model,
                                "task": d["task"],
                            })
                            break

        print(f"  {model}: {len(examples)} examples so far")

    df = pd.DataFrame(examples)
    print(f"\nTotal examples: {len(df)}")
    print(f"Label distribution:\n{df['label'].value_counts()}")

    # Use all data with class-weighted loss (no undersampling)
    # Shuffle and split train/val
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    split = int(len(df) * 0.85)
    train_df = df[:split]
    val_df = df[split:]

    train_df.to_csv(DATA_PATH / "train.csv", index=False)
    val_df.to_csv(DATA_PATH / "val.csv", index=False)
    print(f"\nSaved: train={len(train_df)}, val={len(val_df)}")
    print(f"Train label distribution:\n{train_df['label'].value_counts()}")


def train(
    model_name: str = "bert-base-uncased",
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 3e-5,
    max_length: int = 256,
    save_dir: str = None,
    seed: int = 42,
):
    """Fine-tune a sentence-pair classifier for distortion detection."""
    import torch, random, numpy as np
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from transformers import get_linear_schedule_with_warmup

    # Set seeds for reproducibility across runs
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    # Per-model + per-seed save directory
    model_short = model_name.split("/")[-1]
    out_dir = Path(save_dir) if save_dir else DATA_PATH / f"model_{model_short}_seed{seed}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training {model_name} on {device}, saving to {out_dir}")

    # Load data
    train_df = pd.read_csv(DATA_PATH / "train.csv")
    val_df = pd.read_csv(DATA_PATH / "val.csv")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    ).to(device)

    class DistortionDataset(Dataset):
        def __init__(self, df):
            self.df = df.reset_index(drop=True)

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            src = str(row["source_text"]) if pd.notna(row["source_text"]) else ""
            out = str(row["output_text"]) if pd.notna(row["output_text"]) else ""
            label = LABEL2ID[row["label"]]
            encoded = tokenizer(
                src, out,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            return {
                "input_ids": encoded["input_ids"].squeeze(),
                "attention_mask": encoded["attention_mask"].squeeze(),
                "labels": torch.tensor(label, dtype=torch.long),
            }

    train_dataset = DistortionDataset(train_df)
    val_dataset = DistortionDataset(val_df)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Compute class weights for imbalanced data
    from collections import Counter
    label_counts = Counter(train_df["label"])
    total_samples = len(train_df)
    class_weights = torch.tensor(
        [total_samples / (len(LABELS) * label_counts.get(l, 1)) for l in LABELS],
        dtype=torch.float32
    ).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    print(f"Class weights: {dict(zip(LABELS, class_weights.cpu().tolist()))}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * 0.1), total_steps)

    best_val_acc, best_epoch, best_preds = 0, None, None
    for epoch in range(epochs):
        # Train
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = loss_fn(outputs.logits, batch["labels"])
            total_loss += loss.item()
            preds = outputs.logits.argmax(dim=-1)
            correct += (preds == batch["labels"]).sum().item()
            total += len(batch["labels"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        train_acc = correct / total

        # Validate
        model.eval()
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                preds = outputs.logits.argmax(dim=-1)
                val_correct += (preds == batch["labels"]).sum().item()
                val_total += len(batch["labels"])
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(batch["labels"].cpu().tolist())

        val_acc = val_correct / val_total
        print(f"Epoch {epoch+1}/{epochs}: loss={total_loss/len(train_loader):.4f} "
              f"train_acc={train_acc:.3f} val_acc={val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc, best_epoch, best_preds = val_acc, epoch + 1, list(all_preds)
            out_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)
            print(f"  Saved best model (val_acc={val_acc:.3f})")

    # Per-class metrics + JSON save (seed-aware)
    from sklearn.metrics import classification_report
    import json
    print(f"\nBest validation accuracy: {best_val_acc:.3f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=LABELS))
    # "report" is computed from the LAST epoch's predictions, as in the paper's Table A16 (whose
    # accuracy column is the best epoch's); "report_best_epoch" matches the saved checkpoint.
    report = classification_report(all_labels, all_preds, target_names=LABELS, output_dict=True)
    report_best = classification_report(all_labels, best_preds, target_names=LABELS, output_dict=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"model": model_name, "seed": seed, "best_val_acc": best_val_acc, "best_epoch": best_epoch,
                   "last_epoch": epochs, "report": report, "report_best_epoch": report_best}, f, indent=2)
    print(f"  Saved metrics to {out_dir/'metrics.json'}")


def predict(source_text: str, output_text: str, model_dir: str) -> dict:
    """Predict distortion type for a single source-output pair, with a checkpoint written by train()."""
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()

    encoded = tokenizer(
        source_text, output_text,
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model(**encoded)
        probs = torch.softmax(outputs.logits, dim=-1)[0]
        pred_id = probs.argmax().item()

    return {
        "predicted_label": ID2LABEL[pred_id],
        "confidence": probs[pred_id].item(),
        "all_probs": {ID2LABEL[i]: round(p.item(), 4) for i, p in enumerate(probs)},
    }


def compute_efs_learned(source_text: str, output_text: str, model_dir: str) -> dict:
    """
    Compute EFS using the learned detector instead of rule-based analysis.

    Splits source and output into sentences, aligns them, and classifies
    each pair using the trained model.
    """
    import re
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()

    # Split into sentences
    src_sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n', source_text) if len(s.strip()) > 15]
    out_sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n', output_text) if len(s.strip()) > 15]

    distortions = []
    # For each source sentence, find best matching output sentence and classify
    for src_sent in src_sents[:50]:  # limit for speed
        best_match = None
        best_overlap = 0
        src_words = set(src_sent.lower().split())

        for out_sent in out_sents:
            out_words = set(out_sent.lower().split())
            overlap = len(src_words & out_words)
            if overlap > best_overlap:
                best_overlap = overlap
                best_match = out_sent

        if best_match and best_overlap >= 3:
            encoded = tokenizer(
                src_sent[:256], best_match[:256],
                truncation=True, max_length=256, return_tensors="pt",
            )
            with torch.no_grad():
                logits = model(**encoded).logits
                pred_id = logits.argmax(dim=-1).item()
                label = ID2LABEL[pred_id]

            if label != "no_distortion":
                distortions.append({
                    "type": label,
                    "source": src_sent[:200],
                    "output": best_match[:200],
                })

    n_source_sents = max(len(src_sents), 1)
    efs = max(0.0, 1.0 - len(distortions) / n_source_sents)

    by_type = {}
    for d in distortions:
        by_type[d["type"]] = by_type.get(d["type"], 0) + 1

    return {
        "efs_score": round(efs, 3),
        "distortion_count": len(distortions),
        "by_type": by_type,
        "distortions": distortions,
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m src.learned_detector [generate_data|train|predict]")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "generate_data":
        generate_training_data()
    elif cmd == "train":
        train()
    elif cmd == "predict":
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--source", required=True)
        parser.add_argument("--output", required=True)
        parser.add_argument("--model-dir", required=True, help="a model_<encoder>_seed<S> folder written by train()")
        args = parser.parse_args(sys.argv[2:])
        result = predict(args.source, args.output, args.model_dir)
        print(json.dumps(result, indent=2))
    else:
        print(f"Unknown command: {cmd}")
