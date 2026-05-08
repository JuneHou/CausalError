"""
Cross-benchmark PR scatter with iso-F1 contours, faceted by method.

Three panels: Baseline / +CG / +GI+SI.
Each panel: one point per (model, benchmark) on macro-Precision vs macro-Recall axes.

Macro definition (consistent with MAST/eval/calculate_scores_yesno.py):
  Build per-trace binary category-presence vector (TRAIL: 20-dim, MAST: 13-dim),
  compute precision/recall per category across traces, then average equally.
  TRAIL metrics txt files already contain per-category P/R — we average those.
  MAST metrics json files store macro_precision / macro_recall directly.

Iso-F1 contours = curves of constant F1 = 2PR/(P+R).
"""
import json
import os
import re
import argparse
import math
import glob
import numpy as np
import matplotlib.pyplot as plt


# ---------------- File parsing ----------------

def parse_trail_metrics_txt(path):
    """Parse the per-category P/R/F1 table from a TRAIL metrics.txt file.

    Returns dict with keys: macro_precision, macro_recall, macro_f1, weighted_f1,
    n_categories, n_traces (best effort).
    """
    pcat_p, pcat_r, pcat_f1 = [], [], []
    in_table = False
    weighted_f1 = None
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            m = re.match(r"^\s*Weighted F1:\s*([\d.]+)", line)
            if m:
                weighted_f1 = float(m.group(1))
                continue
            if line.startswith("Per-Category Statistics"):
                in_table = True
                continue
            if not in_table:
                continue
            if line.strip().startswith("---") or line.strip().startswith("Category"):
                continue
            if not line.strip():
                continue
            # Match category rows: name (with spaces) then 4 numeric fields
            m = re.match(r"^\s*(.+?)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s*$", line)
            if not m:
                continue
            cat, p, r, f1, supp = m.groups()
            try:
                p = float(p); r = float(r); f1 = float(f1)
            except ValueError:
                continue
            pcat_p.append(p); pcat_r.append(r); pcat_f1.append(f1)

    if not pcat_p:
        return None
    return {
        "macro_precision": float(np.mean(pcat_p)),
        "macro_recall":    float(np.mean(pcat_r)),
        "macro_f1":        float(np.mean(pcat_f1)),
        "weighted_f1":     weighted_f1,
        "n_categories":    len(pcat_p),
    }


def parse_mast_metrics_json(path):
    with open(path) as f:
        d = json.load(f)
    return {
        "macro_precision": d.get("macro_precision"),
        "macro_recall":    d.get("macro_recall"),
        "macro_f1":        d.get("macro_f1"),
        "weighted_f1":     d.get("weighted_f1"),
        "n_categories":    13,
        "n_traces":        d.get("n_traces"),
    }


# ---------------- Method/model classification ----------------

def classify_trail(stem):
    """stem = filename minus 'outputs_' prefix and '-metrics.txt' suffix."""
    # Order matters — check most specific first.
    if "graph_inject_causal_only_span_index" in stem and "calibrated" not in stem:
        method = "+GI+SI"
        rest = stem.replace("-graph_inject_causal_only_span_index", "")
    elif stem.endswith("-graph_causal_only"):
        method = "+CG"
        rest = stem[:-len("-graph_causal_only")]
    elif stem.endswith("-graph_causal_only_span_index"):
        return None  # not one of the three target methods
    elif "graph_inject_causal_corr0.2_span_index" in stem:
        return None
    elif "graph_t0.2" in stem or "graph_suppes" in stem or "calibrated" in stem:
        return None
    else:
        method = "Baseline"
        rest = stem

    # rest = "<model_id>-<split>"
    # Splits we know about:
    for sp in ("GAIA_dedup", "SWE_Bench_dedup", "SWE Bench"):
        if rest.endswith("-" + sp) or rest.endswith(sp):
            model = rest[:-len(sp)].rstrip("-")
            split_label = "TRAIL-GAIA" if "GAIA" in sp else "TRAIL-SWE"
            return model, split_label, method
    return None


def classify_mast(stem):
    """stem = filename minus '-metrics.json' suffix."""
    if "-baseline" in stem:
        method = "Baseline"
        model = stem.replace("-yesno-baseline", "").replace("-thinking", "")
    elif "-graph-inject-" in stem and "causal_only" in stem:
        method = "+GI+SI"
        model = re.sub(r"-yesno-graph-inject-(codename-)?causal_only(-thinking)?", "", stem)
    elif "-with-graph-" in stem and "causal_only" in stem:
        method = "+CG"
        model = re.sub(r"-yesno-with-graph-(codename-)?causal_only(-thinking)?", "", stem)
    else:
        return None
    return model, "MAST", method


# ---------------- Model name normalization ----------------

MODEL_CANONICAL = [
    (r"gemini.*flash", "Gemini-2.5-Flash"),
    (r"gemini.*pro",   "Gemini-2.5-Pro"),
    (r"gpt-4o",        "GPT-4o"),
    (r"gpt-oss-120b",  "GPT-oss-120B"),
    (r"gpt-oss-20b",   "GPT-oss-20B"),
    (r"gemma-3-27b",   "Gemma-3-27B"),
    (r"mistral.*small.*24b|Mistral-Small-3\.1-24B", "Mistral-Small-24B"),
    (r"qwenlong",      "QwenLong-L1-32B"),
    (r"qwq-32b",       "QwQ-32B"),
    (r"qwen3-30b",     "Qwen3-30B"),
]
def canonical(model_id):
    s = model_id.lower()
    for pat, name in MODEL_CANONICAL:
        if re.search(pat, s):
            return name
    return model_id  # leave as-is if no rule matches


# ---------------- Collection ----------------

def collect_trail(dirs):
    out = []
    for d in dirs:
        for f in glob.glob(os.path.join(d, "*-metrics.txt")):
            stem = os.path.basename(f)[len("outputs_"):-len("-metrics.txt")] if os.path.basename(f).startswith("outputs_") else os.path.basename(f)[:-len("-metrics.txt")]
            cls = classify_trail(stem)
            if cls is None:
                continue
            model_raw, bench, method = cls
            metrics = parse_trail_metrics_txt(f)
            if metrics is None or metrics["macro_precision"] is None:
                continue
            out.append({
                "model": canonical(model_raw),
                "benchmark": bench,
                "method": method,
                "P": metrics["macro_precision"],
                "R": metrics["macro_recall"],
                "F1": metrics["macro_f1"],
                "src": f,
            })
    return out


def collect_mast(dirs):
    out = []
    seen = set()  # (model, method) -> first-found wins (precedence by dirs order)
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, "*-metrics.json"))):
            stem = os.path.basename(f)[:-len("-metrics.json")]
            cls = classify_mast(stem)
            if cls is None:
                continue
            model_raw, bench, method = cls
            key = (canonical(model_raw), method)
            if key in seen:
                continue
            seen.add(key)
            metrics = parse_mast_metrics_json(f)
            if metrics["macro_precision"] is None:
                continue
            out.append({
                "model": canonical(model_raw),
                "benchmark": bench,
                "method": method,
                "P": metrics["macro_precision"],
                "R": metrics["macro_recall"],
                "F1": metrics["macro_f1"],
                "src": f,
            })
    return out


# ---------------- Plotting ----------------

METHOD_ORDER = ["Baseline", "+CG", "+GI+SI"]
BENCH_MARKERS = {"TRAIL-GAIA": "o", "TRAIL-SWE": "s", "MAST": "^"}
BENCH_FILL = {"TRAIL-GAIA": True, "TRAIL-SWE": True, "MAST": True}


def make_plot(rows, out_path, title=None):
    # Stable color per model
    models = sorted({r["model"] for r in rows})
    cmap = plt.cm.tab10 if len(models) <= 10 else plt.cm.tab20
    color = {m: cmap(i % cmap.N) for i, m in enumerate(models)}

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.6), sharex=True, sharey=True)

    # Iso-F1 contour grid
    pp = np.linspace(0.001, 1, 200)
    rr = np.linspace(0.001, 1, 200)
    PP, RR = np.meshgrid(pp, rr)
    F1grid = 2 * PP * RR / (PP + RR)
    levels = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    for ax, method in zip(axes, METHOD_ORDER):
        # iso-F1
        cs = ax.contour(PP, RR, F1grid, levels=levels,
                        colors="lightgray", linewidths=0.7, linestyles="--")
        ax.clabel(cs, fmt="F1=%.2f", fontsize=6, inline=True, inline_spacing=2)
        # P=R diagonal
        ax.plot([0, 1], [0, 1], color="gray", linestyle=":", linewidth=0.8)

        method_rows = [r for r in rows if r["method"] == method]
        for r in method_rows:
            ax.scatter(r["P"], r["R"],
                       s=80,
                       marker=BENCH_MARKERS.get(r["benchmark"], "o"),
                       color=color[r["model"]],
                       edgecolor="black", linewidth=0.6,
                       alpha=0.9, zorder=5)
        ax.set_title(method)
        ax.set_xlabel("macro Precision")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.25)
        ax.set_aspect("equal", adjustable="box")
    axes[0].set_ylabel("macro Recall")

    # Legends — model colors and benchmark markers
    from matplotlib.lines import Line2D
    color_handles = [Line2D([0], [0], marker="o", color="w",
                            markerfacecolor=color[m], markeredgecolor="black",
                            markersize=8, label=m)
                     for m in models]
    bench_handles = [Line2D([0], [0], marker=BENCH_MARKERS[b], color="w",
                            markerfacecolor="lightgray",
                            markeredgecolor="black", markersize=9, label=b)
                     for b in BENCH_MARKERS]
    leg1 = fig.legend(handles=color_handles, title="Model",
                      loc="center right", bbox_to_anchor=(1.13, 0.6),
                      fontsize=8, title_fontsize=9, frameon=True)
    fig.legend(handles=bench_handles, title="Benchmark",
               loc="center right", bbox_to_anchor=(1.13, 0.25),
               fontsize=8, title_fontsize=9, frameon=True)
    fig.add_artist(leg1)

    fig.suptitle(title or "macro-PR by method (each point = one (model, benchmark))",
                 fontsize=11)
    plt.tight_layout(rect=[0, 0, 0.92, 0.96])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[saved] {out_path}")


# ---------------- Main ----------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trail_dirs", nargs="+", default=[
        "/data/wang/junh/githubs/trail-benchmark/benchmarking/outputs/zero_shot2",
    ])
    p.add_argument("--mast_dirs", nargs="+", default=[
        "/data/wang/junh/githubs/MAST/outputs_v2",
        "/data/wang/junh/githubs/MAST/outputs_think",
        "/data/wang/junh/githubs/MAST/outputs_full_api",
        "/data/wang/junh/githubs/MAST/outputs",
        "/data/wang/junh/githubs/MAST/outputs_full",
    ])
    p.add_argument("--out", default="/data/wang/junh/githubs/trail-benchmark/baselines/outputs/_plot_cross_benchmark_pr.png")
    p.add_argument("--csv", default="/data/wang/junh/githubs/trail-benchmark/baselines/outputs/_plot_cross_benchmark_pr.csv")
    p.add_argument("--benchmarks", nargs="+", default=None,
                   help="Filter to a subset of benchmarks (e.g. --benchmarks TRAIL-GAIA TRAIL-SWE)")
    p.add_argument("--title", default=None,
                   help="Override the figure suptitle")
    args = p.parse_args()

    rows = collect_trail(args.trail_dirs) + collect_mast(args.mast_dirs)
    if args.benchmarks:
        wanted = set(args.benchmarks)
        rows = [r for r in rows if r["benchmark"] in wanted]
    if not rows:
        raise SystemExit("no rows collected")

    # Print summary
    print(f"{'model':<22} {'benchmark':<12} {'method':<10} {'P':>6} {'R':>6} {'F1':>6}")
    print("-" * 72)
    for r in sorted(rows, key=lambda x: (x["method"], x["model"], x["benchmark"])):
        print(f"{r['model']:<22} {r['benchmark']:<12} {r['method']:<10} "
              f"{r['P']:>6.3f} {r['R']:>6.3f} {r['F1']:>6.3f}")

    # Save CSV for inspection
    with open(args.csv, "w") as f:
        f.write("model,benchmark,method,P,R,F1,src\n")
        for r in rows:
            f.write(f"{r['model']},{r['benchmark']},{r['method']},{r['P']:.4f},{r['R']:.4f},{r['F1']:.4f},{r['src']}\n")
    print(f"[csv] {args.csv}")

    make_plot(rows, args.out, title=args.title)


if __name__ == "__main__":
    main()
