"""
Per-model CDF of per-trace count bias.

Score per trace:  s = log((pred + 1) / (gold + 1))
  pred / gold = number of distinct error categories per trace.

Layout: 2 rows (GAIA / SWE) × 3 cols (Baseline / +CG / +GI+SI).
Each panel: x = bias score s, y = fraction of traces with bias ≤ s.
One CDF curve per model.

Reading the plot:
  curve at x=0 → fraction of traces the model under-predicts on
  curve at y=0.5 → median per-trace bias (>0 = exploratory, <0 = conservative)
"""
import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Reuse the data loader from the Likert script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _plot_count_bias_likert import collect, SPLIT_LABEL  # noqa


METHOD_ORDER = ["Baseline", "+CG", "+GI+SI"]
SPLIT_ORDER = ["GAIA_dedup", "SWE_Bench_dedup"]


def cdf_xy(scores):
    s = np.sort(np.array(scores))
    y = np.arange(1, len(s) + 1) / len(s)
    return s, y


def make_plot(runs, out_path, pooled_thresholds=None):
    models = sorted({r["model"] for r in runs})
    cmap = plt.cm.tab10 if len(models) <= 10 else plt.cm.tab20
    color = {m: cmap(i % cmap.N) for i, m in enumerate(models)}

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.2), sharex=True, sharey=True)

    for i, split in enumerate(SPLIT_ORDER):
        for j, method in enumerate(METHOD_ORDER):
            ax = axes[i, j]
            relevant = [r for r in runs if r["split"] == split and r["method"] == method]
            for r in sorted(relevant, key=lambda r: r["model"]):
                scores = [row["score"] for row in r["rows"]]
                if not scores:
                    continue
                xs, ys = cdf_xy(scores)
                # Step plot for clean CDF look
                ax.step(xs, ys, where="post",
                        color=color[r["model"]], linewidth=1.5,
                        label=f"{r['model']} (n={len(scores)})")

            ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
            ax.axhline(0.5, color="gray", linewidth=0.6, linestyle=":", alpha=0.6)

            # Threshold reference lines from the pooled distribution
            if pooled_thresholds is not None:
                over_t, under_t = pooled_thresholds
                for t, ls in [(over_t[0], (0,(2,2))), (over_t[1], (0,(2,2))),
                              (-under_t[0], (0,(2,2))), (-under_t[1], (0,(2,2)))]:
                    ax.axvline(t, color="lightgray", linewidth=0.5, linestyle=ls, alpha=0.7)

            if i == 0:
                ax.set_title(method)
            if j == 0:
                ax.set_ylabel(f"{SPLIT_LABEL[split]}\nfraction of traces ≤ s",
                              fontsize=9)
            if i == 1:
                ax.set_xlabel("count-bias score   s = log((pred+1)/(gold+1))\n"
                              "← under-predict           over-predict →",
                              fontsize=9)
            ax.set_xlim(-2.5, 2.5)
            ax.set_ylim(-0.02, 1.02)
            ax.grid(True, alpha=0.25)
            if relevant:
                ax.legend(loc="lower right", fontsize=7, framealpha=0.85)

    fig.suptitle("Per-model CDF of per-trace count bias (TRAIL, category-presence)",
                 fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[saved] {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="/data/wang/junh/githubs/trail-benchmark/baselines/outputs/_plot_count_bias_cdf.png")
    args = p.parse_args()

    runs = collect()
    print(f"[runs] {len(runs)} (model, split, method) triples loaded")

    # Compute pooled thresholds for reference grid lines
    all_scores = [r["score"] for run in runs for r in run["rows"]]
    over = [s for s in all_scores if s > 0]
    under = [-s for s in all_scores if s < 0]
    over_t = (np.percentile(over, 33), np.percentile(over, 67)) if over else (0, 0)
    under_t = (np.percentile(under, 33), np.percentile(under, 67)) if under else (0, 0)

    make_plot(runs, args.out, pooled_thresholds=(over_t, under_t))


if __name__ == "__main__":
    main()
