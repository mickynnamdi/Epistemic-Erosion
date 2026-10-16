"""Draw EFS per model (a plot of Table 1's EFS column) and Figure A1 (EFS distributions per domain).

Paper: Table 1, EFS column (11 open-weight models, mean over the 3 seeds, error bars = cross-seed std,
dashed line = overall mean 0.450; plotted here, not a figure in the paper) and Figure A1 (clinical seed 123 all 11 models, legal,
scientific; mean +- std and share of generations at EFS = 0). Figure 2 is drawn by
scripts/analysis/metric_blindness.py --figure.
Inputs: results/scores/{depth,crossdomain}.csv (scripts/score_generations.py).
Outputs: results/figures/efs_per_model.{pdf,png}, results/figures/figA1_efs_distributions.{pdf,png}.
Usage:
    python scripts/figures/make_figures.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.models import MODELS  # noqa: E402
from configs.paths import RESULTS_DIR, require  # noqa: E402

PAL = {"blue": "#4477AA", "green": "#228833", "purple": "#AA3377", "orange": "#EE7733", "red": "#EE6677"}
ARCH_COLORS = {"Llama": PAL["blue"], "Mistral": PAL["orange"], "Gemma 3": PAL["green"], "Gemma 4": PAL["purple"]}
ARCH_GROUPS = [("Llama", ["llama-3.1-8b", "med42-8b", "llama-3.3-70b", "med42-70b", "openbio-70b"]),
               ("Mistral", ["mistral-7b", "biomistral-7b"]), ("Gemma 3", ["medgemma-27b"]),
               ("Gemma 4", ["gemma-4-4b", "gemma-4-31b", "gemma-4-moe"])]
LABEL = {"llama-3.1-8b": "Llama-3.1-8B", "med42-8b": "Med42-v2-8B", "llama-3.3-70b": "Llama-3.3-70B",
         "med42-70b": "Med42-v2-70B", "openbio-70b": "OpenBioLLM-70B", "mistral-7b": "Mistral-7B-Inst.",
         "biomistral-7b": "BioMistral-7B", "medgemma-27b": "MedGemma-27B", "gemma-4-4b": "Gemma-4-E4B",
         "gemma-4-31b": "Gemma-4-31B", "gemma-4-moe": "Gemma-4-26B-MoE"}
COL_W = 3.6
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
                     "font.size": 11, "savefig.dpi": 300, "savefig.bbox": "tight", "axes.linewidth": 0.7})


def load(study):
    df = pd.read_csv(require(RESULTS_DIR / "scores" / f"{study}.csv", f"{study} scores", "EE_RESULTS_DIR"),
                     low_memory=False)
    return df[df.status == "ok"]


def figure2(depth, out):
    rows = []
    for arch, models in ARCH_GROUPS:
        for m in models:
            per_seed = depth[depth.model == m].groupby("seed").efs.mean()
            rows.append({"arch": arch, "label": LABEL[m] + (" †" if MODELS[m]["medical"] else ""),
                         "med": MODELS[m]["medical"], "mean": per_seed.mean(), "err": per_seed.std(ddof=1)})
    overall = depth.groupby(["seed", "model"]).efs.mean().mean()
    fig, ax = plt.subplots(figsize=(COL_W * 1.35, 4.4))
    ys, cur, prev, bands, start = [], 0.0, None, [], 0.0
    for r in rows:
        if prev is not None and r["arch"] != prev:
            bands.append((start - 0.5, cur - 0.5))
            cur += 0.7
            start = cur
        prev = r["arch"]
        ys.append(cur)
        cur += 1
    bands.append((start - 0.5, cur - 0.5))
    lo = min(r["mean"] - r["err"] for r in rows)
    hi = max(r["mean"] + r["err"] for r in rows)
    span = hi - lo
    annot_x = hi + span * 0.18
    for i, (y0, y1) in enumerate(bands):
        ax.axhspan(y0, y1, facecolor=["#f5f7fa", "#ffffff"][i % 2], zorder=0)
    for r, y in zip(rows, ys):
        c = ARCH_COLORS[r["arch"]]
        ax.plot([r["mean"] - r["err"], r["mean"] + r["err"]], [y, y], color=c, lw=1.4, alpha=0.55, zorder=2)
        ax.scatter(r["mean"], y, marker="D" if r["med"] else "o", s=42, color=c, edgecolors="white", linewidths=0.5, zorder=4)
        ax.annotate(f"{r['mean']:.3f}", xy=(annot_x, y), fontsize=10, color="#444444", va="center", family="monospace")
    ax.axvline(overall, color="#9aa0a6", linestyle=(0, (3, 2)), lw=0.7, zorder=1)
    ax.annotate(f"Overall mean = {overall:.3f}", xy=(overall, -1.2), fontsize=10, color="#5f6368", ha="center",
                fontstyle="italic")
    ax.set_yticks(ys)
    ax.set_yticklabels([r["label"] for r in rows], fontsize=10.5)
    ax.set_xlabel("Epistemic Faithfulness Score (EFS)")
    ax.set_xlim(max(0, lo - span * 0.15), min(1, hi + span * 0.55))
    ax.set_ylim(cur - 0.5, -0.5)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.02))
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    handles = [mpatches.Patch(color=ARCH_COLORS[a], label=a) for a in ARCH_COLORS] + [
        plt.Line2D([0], [0], marker="D", color="#666", linestyle="", label="† Medical/domain-adapted"),
        plt.Line2D([0], [0], marker="o", color="#666", linestyle="", label="General-purpose")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.13), fontsize=9, ncol=3, frameon=False)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"efs_per_model.{ext}")
    plt.close(fig)


def figure_a1(depth, xd, out):
    panels = [("Clinical", depth[depth.seed == 123].efs, PAL["blue"]),
              ("Legal", xd[xd.domain == "legal"].efs, PAL["orange"]),
              ("Scientific", xd[xd.domain == "scientific"].efs, PAL["green"])]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4))
    for ax, (name, e, c) in zip(axes, panels):
        ax.hist(e, bins=np.linspace(0, 1, 21), density=True, color=c, alpha=0.75, edgecolor="white", linewidth=0.4)
        ax.axvline(e.mean(), color="#333333", linestyle="--", lw=0.8)
        ax.set_title(f"{name}\n(EFS = {e.mean():.3f} $\\pm$ {e.std(ddof=1):.3f})", fontsize=11)
        ax.annotate(f"{100 * (e == 0).mean():.0f}% at 0", xy=(0.62, 0.9), xycoords="axes fraction", fontsize=8.5,
                    color=c, fontweight="bold")
        ax.set_xlabel("EFS")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("Density")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"figA1_efs_distributions.{ext}")
    plt.close(fig)


def main():
    out = RESULTS_DIR / "figures"
    out.mkdir(parents=True, exist_ok=True)
    depth = load("depth")
    figure2(depth, out)
    figure_a1(depth, load("crossdomain"), out)
    print(f"figures -> {out}")


if __name__ == "__main__":
    main()
