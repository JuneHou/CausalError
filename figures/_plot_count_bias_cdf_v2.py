"""
Per-model CDF of per-trace count bias — publication-styled variant.

Differences from `_plot_count_bias_cdf.py`:
  * Wong colorblind-safe palette (Nature Methods 2011, doi:10.1038/nmeth.1618).
  * Median bias marker (filled circle) on each curve at (median_s, 0.5),
    with a small annotation showing the numeric median.
  * Lighter grid, thinner axes, serif title — closer to Nature/Science style.
  * Output → _plot_count_bias_cdf_v2.png (does not overwrite the original).
"""
import sys
import os
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# Reuse the data loader from the Likert script (kept in baselines/outputs/).
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOADER_DIR = "/data/wang/junh/githubs/trail-benchmark/baselines/outputs"
sys.path.insert(0, LOADER_DIR)
from _plot_count_bias_likert import collect, SPLIT_LABEL  # noqa


METHOD_ORDER = ["Baseline", "+CG", "+GI+SI"]
SPLIT_ORDER = ["GAIA_dedup", "SWE_Bench_dedup"]


# ---------------- Palette (Wong, colorblind-safe) ----------------
# https://www.nature.com/articles/nmeth.1618
WONG = {
    "black":      "#000000",
    "orange":     "#E69F00",
    "sky_blue":   "#56B4E9",
    "green":      "#009E73",
    "yellow":     "#F0E442",
    "blue":       "#0072B2",
    "vermillion": "#D55E00",
    "purple":     "#CC79A7",
}

# Map models to Wong colors. Same family → similar hues.
MODEL_COLOR = {
    "Gemini-2.5-Flash":  WONG["orange"],      # warm — Gemini family
    "Gemini-2.5-Pro":    WONG["vermillion"],
    "Mistral-Small-24B": WONG["green"],
    "GPT-oss-120B":      WONG["blue"],        # cool — GPT-oss family
    "GPT-oss-20B":       WONG["sky_blue"],
    "Gemma-3-27B":       WONG["purple"],
    "QwenLong-L1-32B":   WONG["black"],
}


# ---------------- Style ----------------
def set_style():
    mpl.rcParams.update({
        "font.family":         "serif",
        "font.serif":          ["DejaVu Serif", "Times New Roman", "Times", "Liberation Serif"],
        "font.size":           9,
        "axes.titlesize":      10,
        "axes.labelsize":      9,
        "axes.linewidth":      0.6,
        "axes.spines.top":     False,
        "axes.spines.right":   False,
        "xtick.major.width":   0.5,
        "ytick.major.width":   0.5,
        "xtick.labelsize":     8,
        "ytick.labelsize":     8,
        "legend.fontsize":     7,
        "legend.frameon":      False,
        "grid.linewidth":      0.4,
        "grid.alpha":          0.4,
        "lines.linewidth":     1.6,
        "savefig.dpi":         200,
        "savefig.bbox":        "tight",
    })


# ---------------- Plotting ----------------

def cdf_xy(scores):
    s = np.sort(np.array(scores))
    y = np.arange(1, len(s) + 1) / len(s)
    return s, y


def make_plot(runs, out_path):
    set_style()

    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.4),
                             sharex=True, sharey=True)

    for i, split in enumerate(SPLIT_ORDER):
        for j, method in enumerate(METHOD_ORDER):
            ax = axes[i, j]
            relevant = [r for r in runs if r["split"] == split and r["method"] == method]
            relevant.sort(key=lambda r: r["model"])

            for r in relevant:
                scores = [row["score"] for row in r["rows"]]
                if not scores:
                    continue
                xs, ys = cdf_xy(scores)
                col = MODEL_COLOR.get(r["model"], "#777777")
                ax.step(xs, ys, where="post", color=col, linewidth=1.4,
                        label=f"{r['model']}", solid_capstyle="round")

                # Median marker at (median, 0.5)
                med = float(np.median(scores))
                ax.scatter([med], [0.5], s=22, color=col,
                           edgecolor="white", linewidth=0.7, zorder=5)

            # Reference lines
            ax.axvline(0, color="#444444", linewidth=0.6, linestyle="-", alpha=0.6)
            ax.axhline(0.5, color="#888888", linewidth=0.4, linestyle=":", alpha=0.7)

            # Median value labels (placed in a small column at right of panel,
            # one per model in the legend, to avoid label collisions on the curves).
            if relevant:
                # Sort medians for readable column
                med_sorted = sorted(
                    [(np.median([row["score"] for row in r["rows"]]), r["model"])
                     for r in relevant if r["rows"]],
                    key=lambda t: t[0],
                )
                y_anchor = 0.04
                line_h   = 0.05
                ax.text(2.45, y_anchor + line_h * (len(med_sorted) + 0.5),
                        "median  s",
                        fontsize=6.2, color="#333333",
                        ha="right", va="bottom", style="italic")
                for k, (med, name) in enumerate(med_sorted):
                    ax.text(2.45, y_anchor + line_h * (len(med_sorted) - k - 0.3),
                            f"{med:+.2f}",
                            fontsize=6.5, color=MODEL_COLOR.get(name, "#444"),
                            ha="right", va="bottom",
                            family="monospace")

            if i == 0:
                ax.set_title(method, pad=6)
            if j == 0:
                ax.set_ylabel(f"{SPLIT_LABEL[split]}\nfraction of traces $\\leq s$",
                              fontsize=9)
            if i == 1:
                ax.set_xlabel(r"count-bias score   $s = \log\!\left(\frac{\mathrm{pred}+1}{\mathrm{gold}+1}\right)$"
                              "\n← under-predict           over-predict →",
                              fontsize=8.5)
            ax.set_xlim(-2.6, 2.6)
            ax.set_ylim(-0.02, 1.02)
            ax.grid(True, axis="both", linestyle="-", alpha=0.25)

    # One shared legend (top of figure) — model name + color swatch
    handles = [mpl.lines.Line2D([0], [0], color=MODEL_COLOR[m], linewidth=2.0,
                                label=m)
               for m in MODEL_COLOR]
    fig.legend(handles=handles, ncol=len(MODEL_COLOR),
               loc="upper center", bbox_to_anchor=(0.5, 1.02),
               fontsize=8, frameon=False, handlelength=1.6,
               columnspacing=1.2)

    fig.suptitle("Per-model CDF of per-trace count bias — TRAIL",
                 fontsize=11, y=1.06)
    plt.tight_layout(rect=[0, 0, 1, 1.0])
    plt.savefig(out_path)
    print(f"[saved] {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out",
                   default=os.path.join(SCRIPT_DIR, "_plot_count_bias_cdf_v2.png"))
    args = p.parse_args()
    runs = collect()
    print(f"[runs] {len(runs)} (model, split, method) triples")
    make_plot(runs, args.out)


if __name__ == "__main__":
    main()
