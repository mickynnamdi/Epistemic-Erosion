"""Train the learned distortion detector (4 encoders x 3 seeds).

Paper: Section 4.3 / 6.9, Appendix N, Table A16 (validation accuracy 88.5-88.8%).
Inputs: results/learned_detector/{train,val}.csv, built by
    python -m src.learned_detector generate_data
from the seed-123 clinical generations (MIMIC-derived: PhysioNet credentialed). GPU.
Outputs: results/learned_detector/model_<encoder>_seed<N>/ (checkpoint + metrics.json).
Checkpoints contain MIMIC-derived information: never publish them outside PhysioNet.
Usage:
    python scripts/detector/train.py --model roberta-base --seed 42
    python scripts/detector/train.py --all-seeds            # the paper's 12 runs
Then summarise with: python scripts/analysis/detector_table.py
Hyperparameters (Table A3): AdamW, lr 3e-5, 5 epochs, max length 256, batch 16,
class-weighted cross-entropy, 10% linear warm-up.
"""
import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.learned_detector import train

ENCODERS = [
    "roberta-base",
    "roberta-large",
    "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
    "allenai/scibert_scivocab_uncased",
]
SEEDS = [42, 123, 2024]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="roberta-base", help="HuggingFace model name")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--all-seeds", action="store_true",
                    help="Train all 4 encoders × 3 seeds (12 runs)")
    a = ap.parse_args()
    if a.all_seeds:
        for s in SEEDS:
            for m in ENCODERS:
                print(f"\n=== Training {m} seed={s} ===")
                train(model_name=m, epochs=a.epochs, batch_size=a.batch_size,
                      lr=a.lr, seed=s)
    else:
        train(model_name=a.model, epochs=a.epochs, batch_size=a.batch_size,
              lr=a.lr, seed=a.seed)


if __name__ == "__main__":
    main()
