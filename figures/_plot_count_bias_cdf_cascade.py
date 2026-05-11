"""
Per-model CDF of per-trace count bias — Baseline vs CASCADE.

Companion to `_plot_count_bias_cdf_v2.py`. Two differences:
  * Only `Baseline` and `+GI+SI` (the full method, named CASCADE) are shown.
    The +CG column is dropped — figure becomes 2 rows × 2 cols.
  * Per-panel legend lists each model with its **median count-bias**
    (e.g., "+0.21" / "-0.31") instead of the trace count `(n=...)`.
    Sign-prefix makes the conservative/aggressive direction readable
    at a glance.

Score per trace:   s = log((pred + 1) / (gold + 1))
  pred / gold = number of distinct error categories per trace.
  s < 0 → under-prediction (conservative).
  s > 0 → over-prediction (aggressive).

Reading the plot:
  * Inside each model (rows of curves) — does Baseline lie left of zero
    (conservative) or right (aggressive)?
  * Across columns — does CASCADE pull the curve toward zero (count
    calibration) or in some specific direction (induced bias)?

Output → _plot_count_bias_cdf_cascade.png
"""
import sys
import os
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# Reuse the data loader from the Likert script.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from _plot_count_bias_likert import collect, SPLIT_LABEL  # noqa


ALL_METHODS   = ["Baseline", "+CG", "+GI+SI"]
DEFAULT_PAIR  = "Baseline,+GI+SI"
METHOD_LABELS = {
    "Baseline": "Baseline",
    "+CG":      "+CG (one-pass)",
    "+GI+SI":   "CASCADE (ours)",
}
HIGHLIGHT_METHOD = "+GI+SI"  # bolded in panel title if present
SPLIT_ORDER   = ["GAIA_dedup", "SWE_Bench_dedup"]


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

MODEL_COLOR = {
    "Gemini-2.5-Flash":  WONG["orange"],
    "Gemini-2.5-Pro":    WONG["vermillion"],
    "Mistral-Small-24B": WONG["green"],
    "GPT-oss-120B":      WONG["blue"],
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


def make_plot(runs, out_path, method_pair):
    set_style()

    fig, axes = plt.subplots(
        len(SPLIT_ORDER), len(method_pair),
        figsize=(4.3 * len(method_pair), 7.0), sharex=True, sharey=True,
    )

    for i, split in enumerate(SPLIT_ORDER):
        for j, method in enumerate(method_pair):
            ax = axes[i, j]
            relevant = [r for r in runs if r["split"] == split and r["method"] == method]
            relevant.sort(key=lambda r: r["model"])

            legend_handles = []
            for r in relevant:
                scores = [row["score"] for row in r["rows"]]
                if not scores:
                    continue
                xs, ys = cdf_xy(scores)
                col = MODEL_COLOR.get(r["model"], "#777777")
                med = float(np.median(scores))

                line, = ax.step(
                    xs, ys, where="post",
                    color=col, linewidth=1.4,
                    solid_capstyle="round",
                    label=f"{r['model']} ({med:+.2f})",
                )
                # Median marker at (median, 0.5)
                ax.scatter([med], [0.5], s=22, color=col,
                           edgecolor="white", linewidth=0.7, zorder=5)
                legend_handles.append(line)

            # Reference lines
            ax.axvline(0, color="#444444", linewidth=0.6, linestyle="-", alpha=0.6)
            ax.axhline(0.5, color="#888888", linewidth=0.4, linestyle=":", alpha=0.7)

            if i == 0:
                ax.set_title(METHOD_LABELS.get(method, method), pad=6,
                             fontweight=("bold" if method == HIGHLIGHT_METHOD else "normal"))
            if j == 0:
                ax.set_ylabel(f"{SPLIT_LABEL[split]}\nfraction of traces $\\leq s$",
                              fontsize=9)
            if i == len(SPLIT_ORDER) - 1:
                ax.set_xlabel(
                    r"count-bias score   $s = \log\!\left(\frac{\mathrm{pred}+1}{\mathrm{gold}+1}\right)$"
                    "\n← under-predict           over-predict →",
                    fontsize=8.5,
                )

            ax.set_xlim(-2.6, 2.6)
            ax.set_ylim(-0.02, 1.02)
            ax.grid(True, axis="both", linestyle="-", alpha=0.25)

            # Per-panel legend — model name + median (signed)
            if legend_handles:
                ax.legend(
                    handles=legend_handles,
                    loc="lower right",
                    fontsize=6.8,
                    handlelength=1.4,
                    handletextpad=0.4,
                    labelspacing=0.25,
                    borderpad=0.3,
                    title="model  (median $s$)",
                    title_fontsize=6.5,
                )

    fig.suptitle("Per-model CDF of per-trace count bias — TRAIL", fontsize=11, y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 1.0])
    plt.savefig(out_path)
    print(f"[saved] {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--method-pair",
        default=DEFAULT_PAIR,
        help=(f"Comma-separated methods to render (any subset of {ALL_METHODS}). "
              f"Default: {DEFAULT_PAIR!r}. Examples: 'Baseline,+CG', 'Baseline,+GI+SI', "
              "'Baseline,+CG,+GI+SI', '+CG,+GI+SI'."),
    )
    p.add_argument(
        "--out", default=None,
        help=("Output path. If omitted, filename encodes the method pair: "
              "e.g., _plot_count_bias_cdf_cascade.png for default, "
              "_plot_count_bias_cdf_cg-cascade.png for '+CG,+GI+SI'."),
    )
    args = p.parse_args()

    methods = [m.strip() for m in args.method_pair.split(",") if m.strip()]
    bad = [m for m in methods if m not in ALL_METHODS]
    if bad:
        p.error(f"Unknown method(s) {bad!r}. Choose from {ALL_METHODS}.")
    if len(methods) < 2:
        p.error("--method-pair must list at least two methods")

    if args.out is None:
        tag_map = {"Baseline": "baseline", "+CG": "cg", "+GI+SI": "cascade"}
        tag = "-".join(tag_map.get(m, m.lower()) for m in methods)
        args.out = os.path.join(SCRIPT_DIR, f"_plot_count_bias_cdf_{tag}.png")

    runs = collect()
    print(f"[runs] {len(runs)} (model, split, method) triples loaded")
    print(f"[methods] {methods}")
    make_plot(runs, args.out, method_pair=methods)


if __name__ == "__main__":
    main()
