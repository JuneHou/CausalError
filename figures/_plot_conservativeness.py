"""
Per-trace conservativeness/quality plots for one model.

Plot 1: log(pred/gold) on x-axis  vs  F1 on y-axis
Plot 2: Precision vs Recall scatter with iso-F1 contours

Matching: TP = |set(gen_locations) ∩ set(gt_locations)| (same as calculate_scores.py).
Each point = one trace. Hollow points indicate gold=0 traces (handled separately).
"""
import json
import glob
import os
import argparse
import math
import numpy as np
import matplotlib.pyplot as plt


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def per_trace_stats(gt_dir, pred_dir):
    rows = []
    for gt_path in sorted(glob.glob(os.path.join(gt_dir, "*.json"))):
        name = os.path.basename(gt_path)
        pred_path = os.path.join(pred_dir, name)
        if not os.path.exists(pred_path):
            continue
        try:
            gt = load_json(gt_path)
            pred = load_json(pred_path)
        except Exception:
            continue
        gt_locs = [e.get("location", "") for e in gt.get("errors", [])]
        pred_locs = [e.get("location", "") for e in pred.get("errors", [])]
        gt_set = set(gt_locs)
        pred_set = set(pred_locs)
        tp = len(gt_set & pred_set)
        gold = len(gt_set)
        predn = len(pred_set)
        precision = tp / predn if predn else 0.0
        recall = tp / gold if gold else float("nan")  # undefined when no gold errors
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 and not math.isnan(recall) else 0.0)
        rows.append({
            "trace": name,
            "gold": gold,
            "pred": predn,
            "tp": tp,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })
    return rows


def make_plots(rows, model_label, out_path):
    # Separate gold=0 (no errors to find) from rest
    rest = [r for r in rows if r["gold"] > 0]
    zero_gold = [r for r in rows if r["gold"] == 0]

    # log(pred/gold) with smoothing for gold=0 / pred=0
    def log_ratio(pred, gold):
        return math.log((pred + 1) / (gold + 1))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # ---- Plot 1: log(pred/gold) vs F1 ----
    ax = axes[0]
    xs = [log_ratio(r["pred"], r["gold"]) for r in rest]
    ys = [r["f1"] for r in rest]
    ax.scatter(xs, ys, s=28, alpha=0.55, edgecolor="black", linewidth=0.4,
               label=f"gold>0 (n={len(rest)})")
    if zero_gold:
        xs0 = [log_ratio(r["pred"], 0) for r in zero_gold]
        ys0 = [0.0 for _ in zero_gold]
        ax.scatter(xs0, ys0, s=28, alpha=0.55, marker="x", color="red",
                   label=f"gold=0 (n={len(zero_gold)})")
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("log( (pred+1) / (gold+1) )   ←conservative   |   exploratory→")
    ax.set_ylabel("F1 (per trace)")
    ax.set_title(f"Plot 1: count-bias vs F1\n{model_label}")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Aggregate marker (dataset-level)
    total_pred = sum(r["pred"] for r in rest)
    total_gold = sum(r["gold"] for r in rest)
    total_tp = sum(r["tp"] for r in rest)
    P = total_tp / total_pred if total_pred else 0.0
    R = total_tp / total_gold if total_gold else 0.0
    F = 2 * P * R / (P + R) if (P + R) > 0 else 0.0
    agg_x = math.log((total_pred + 1) / (total_gold + 1))
    ax.scatter([agg_x], [F], s=180, marker="*", color="gold",
               edgecolor="black", linewidth=1.2, zorder=5,
               label="aggregate (micro)")
    ax.legend(loc="lower left", fontsize=8)

    # ---- Plot 2: Precision vs Recall with iso-F1 ----
    ax = axes[1]
    pp = np.linspace(0.001, 1, 200)
    rr = np.linspace(0.001, 1, 200)
    PP, RR = np.meshgrid(pp, rr)
    F1grid = 2 * PP * RR / (PP + RR)
    levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    cs = ax.contour(PP, RR, F1grid, levels=levels, colors="lightgray",
                    linewidths=0.8, linestyles="--")
    ax.clabel(cs, fmt="F1=%.1f", fontsize=7)

    xs = [r["precision"] for r in rest]
    ys = [r["recall"] for r in rest]
    ax.scatter(xs, ys, s=28, alpha=0.55, edgecolor="black", linewidth=0.4,
               label=f"gold>0 (n={len(rest)})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle=":", linewidth=0.8,
            label="P=R (unbiased)")
    ax.scatter([P], [R], s=180, marker="*", color="gold",
               edgecolor="black", linewidth=1.2, zorder=5,
               label=f"aggregate (P={P:.2f}, R={R:.2f}, F1={F:.2f})")
    ax.set_xlabel("Precision   →high = conservative & accurate")
    ax.set_ylabel("Recall   →high = exploratory & catches errors")
    ax.set_title(f"Plot 2: PR scatter with iso-F1\n{model_label}")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[saved] {out_path}")
    print(f"[aggregate] pred={total_pred}  gold={total_gold}  tp={total_tp}  "
          f"P={P:.3f}  R={R:.3f}  F1={F:.3f}  log(pred/gold)={agg_x:.3f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gt", default="/data/wang/junh/githubs/trail-benchmark/benchmarking/processed_annotations_gaia")
    p.add_argument("--pred", default="/data/wang/junh/githubs/trail-benchmark/baselines/outputs/outputs_mistralai-Mistral-Small-3.1-24B-Instruct-2503-GAIA_dedup-who_and_when_w1")
    p.add_argument("--label", default="Mistral-Small-3.1-24B  /  TRAIL-GAIA(dedup)  /  Baseline (w1)")
    p.add_argument("--out", default="/data/wang/junh/githubs/trail-benchmark/baselines/outputs/_plot_conservativeness.png")
    args = p.parse_args()

    rows = per_trace_stats(args.gt, args.pred)
    print(f"[traces matched] {len(rows)}")
    if not rows:
        raise SystemExit("no traces matched")
    make_plots(rows, args.label, args.out)


if __name__ == "__main__":
    main()
