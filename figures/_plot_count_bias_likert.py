"""
Per-trace count bias loader + diverging-stacked-bar (Likert) plot.

Score per trace:  s = log((pred + 1) / (gold + 1))
  pred / gold = number of distinct error categories in the trace.

The collect() / per_trace_scores() functions are also imported by the CDF script.
"""
import json
import glob
import os
import math
import argparse
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# ---------------- Path conventions ----------------
# Search-path priority: first match per (model, split, method) wins. Order matches
# the dataset conventions documented in paper/main_results_table.tex:
#   Gemini-Flash → zero_shot/, original GAIA / SWE Bench
#   QwenLong GAIA → zero_shot/compressed/GAIA_dedup-*
#   everything else → zero_shot2/

TRAIL_DIRS_DEFAULT = [
    "/data/wang/junh/githubs/trail-benchmark/benchmarking/outputs/zero_shot2",
    "/data/wang/junh/githubs/trail-benchmark/benchmarking/outputs/zero_shot/compressed",
    "/data/wang/junh/githubs/trail-benchmark/benchmarking/outputs/zero_shot",
]
GT_GAIA = "/data/wang/junh/githubs/trail-benchmark/benchmarking/processed_annotations_gaia"
GT_SWE  = "/data/wang/junh/githubs/trail-benchmark/benchmarking/processed_annotations_swe_bench"

SPLIT_LABEL = {"GAIA_dedup": "GAIA", "SWE_Bench_dedup": "SWE"}


# ---------------- Method / split classification ----------------

def classify_dir(stem):
    """stem = directory basename (no trailing slash).

    Returns (model_id, split, method, pref) where pref is a tie-break rank for
    methods that have multiple sources (lower = preferred). Used only for
    +GI+SI: causal_only_span_index (pref 0) preferred over corr0.2_span_index
    (pref 1), since the latter is only reported for Gemini-Flash GAIA.
    """
    if not stem.startswith("outputs_"):
        return None
    body = stem[len("outputs_"):]
    if "calibrated" in body:
        return None

    # Method classification — most specific first.
    if body.endswith("graph_inject_causal_only_span_index"):
        method = "+GI+SI"; pref = 0
        rest = body[:-len("-graph_inject_causal_only_span_index")]
    elif body.endswith("graph_inject_causal_corr0.2_span_index"):
        method = "+GI+SI"; pref = 1
        rest = body[:-len("-graph_inject_causal_corr0.2_span_index")]
    elif body.endswith("graph_causal_only"):
        method = "+CG"; pref = 0
        rest = body[:-len("-graph_causal_only")]
    elif (body.endswith("graph_causal_only_span_index") or
          "graph_t0.2" in body or
          "graph_suppes" in body or
          "who_and_when" in body or
          "graph_causal_only_train" in body):
        return None
    else:
        method = "Baseline"; pref = 0
        rest = body

    # Split detection — accept dedup splits, original ("GAIA"/"SWE Bench"),
    # and compressed variants. Normalise everything to GAIA_dedup / SWE_Bench_dedup
    # for downstream grouping (the GT match still skips traces not in the dedup set).
    for sp in ("GAIA_dedup", "SWE_Bench_dedup", "SWE Bench", "GAIA"):
        if rest.endswith("-" + sp) or rest.endswith(sp):
            model_id = rest[:-len(sp)].rstrip("-")
            if sp in ("GAIA_dedup", "GAIA"):
                split = "GAIA_dedup"
            else:
                split = "SWE_Bench_dedup"
            return model_id, split, method, pref
    return None


MODEL_CANONICAL = [
    (r"gemini.*flash", "Gemini-2.5-Flash"),
    (r"gemini.*pro",   "Gemini-2.5-Pro"),
    (r"gpt-oss-120b",  "GPT-oss-120B"),
    (r"gpt-oss-20b",   "GPT-oss-20B"),
    (r"gemma-3-27b",   "Gemma-3-27B"),
    (r"mistral.*small", "Mistral-Small-24B"),
    (r"qwenlong",      "QwenLong-L1-32B"),
]
# Only include models that appear in paper/main_results_table.tex.
MAIN_TABLE_MODELS = {name for _, name in MODEL_CANONICAL}

def canonical(model_id):
    s = model_id.lower()
    for pat, name in MODEL_CANONICAL:
        if re.search(pat, s):
            return name
    return model_id


# ---------------- Per-trace scores ----------------

def count_distinct_categories(errors):
    cats = set()
    for e in errors or []:
        if not isinstance(e, dict):
            continue
        c = (e.get("category") or "").strip()
        if c:
            cats.add(c.lower())
    return len(cats)


def mean_locations_per_category(errors):
    """Average number of distinct step-locations attached to each predicted category.

    Captures "label tiling" — when one category (e.g. Hallucinations) is repeated
    across many step IDs in the same trace.
    """
    cats_to_locs: dict[str, set[str]] = {}
    for e in errors or []:
        if not isinstance(e, dict):
            continue
        c = (e.get("category") or "").strip().lower()
        if not c:
            continue
        loc = (e.get("location") or "").strip()
        cats_to_locs.setdefault(c, set()).add(loc)
    if not cats_to_locs:
        return 0.0
    return sum(len(v) for v in cats_to_locs.values()) / len(cats_to_locs)


def _count_bias_score(gt_errors, pr_errors):
    gold = count_distinct_categories(gt_errors)
    pred = count_distinct_categories(pr_errors)
    return gold, pred, math.log((pred + 1) / (gold + 1))


def _spread_bias_score(gt_errors, pr_errors):
    """Per-trace location-spread bias.

    s = log((L_pred + 1) / (L_gold + 1))
      L = mean #distinct-locations per distinct category (see
      `mean_locations_per_category`).

    s > 0 → model attaches more locations per category than gold (label tiling).
    """
    gold = mean_locations_per_category(gt_errors)
    pred = mean_locations_per_category(pr_errors)
    return gold, pred, math.log((pred + 1) / (gold + 1))


_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)

def _load_json_lenient(path):
    """Strip markdown code fences and extract the first balanced JSON object."""
    with open(path) as f:
        txt = f.read()
    # Fast path: clean JSON
    try:
        return json.loads(txt)
    except Exception:
        pass
    # Strip code fences
    t = txt.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t)
        t = re.sub(r"\s*```\s*$", "", t)
        try:
            return json.loads(t)
        except Exception:
            pass
    # Regex grab of the first {...} block, then chip from the end until it parses
    m = _JSON_OBJ_RE.search(txt)
    if not m:
        return None
    candidate = m.group(0)
    while len(candidate) > 2:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            candidate = candidate[:-1]
    return None


def per_trace_scores(pred_dir, gt_dir, score_fn=None):
    """Walk gt_dir, match against pred_dir, return per-trace rows.

    score_fn(gt_errors, pr_errors) -> (gold, pred, score). Defaults to
    `_count_bias_score` (distinct-category count bias). Pass `_spread_bias_score`
    to compute per-trace location-spread bias instead.
    """
    if score_fn is None:
        score_fn = _count_bias_score
    rows = []
    n_gt = 0
    n_pred_present = 0
    n_pred_unparsable = 0
    for gt_path in sorted(glob.glob(os.path.join(gt_dir, "*.json"))):
        n_gt += 1
        name = os.path.basename(gt_path)
        pred_path = os.path.join(pred_dir, name)
        if not os.path.exists(pred_path):
            continue
        n_pred_present += 1
        gt = _load_json_lenient(gt_path)
        pr = _load_json_lenient(pred_path)
        if gt is None or pr is None:
            n_pred_unparsable += 1
            continue
        gold, pred, s = score_fn(gt.get("errors", []), pr.get("errors", []))
        rows.append({"gold": gold, "pred": pred, "score": s})
    return rows, n_gt, n_pred_present, n_pred_unparsable


def collect(verbose=False, search_dirs=None, score_fn=None):
    """Walk priority-ordered output dirs.

    Two-pass dedup by (model, split, method):
      1. First-found-wins across search dirs.
      2. Within the same dir, lower 'pref' wins (causal_only_span_index over
         corr0.2_span_index for the +GI+SI fallback case).
    """
    if search_dirs is None:
        search_dirs = TRAIL_DIRS_DEFAULT

    # Pass 1: enumerate all candidate dirs with classification.
    candidates = []
    skipped = []
    for sd in search_dirs:
        for d in sorted(glob.glob(os.path.join(sd, "outputs_*"))):
            if not os.path.isdir(d):
                continue
            stem = os.path.basename(d)
            cls = classify_dir(stem)
            if cls is None:
                if verbose:
                    skipped.append((sd, stem, "classify_returned_None"))
                continue
            model_id, split, method, pref = cls
            candidates.append({
                "sd": sd, "stem": stem, "dir": d,
                "key": (canonical(model_id), split, method),
                "model_id": model_id, "split": split, "method": method,
                "pref": pref,
            })

    # Pass 2: keep best (lowest search-dir index, then lowest pref) per key.
    sd_rank = {sd: i for i, sd in enumerate(search_dirs)}
    candidates.sort(key=lambda c: (sd_rank.get(c["sd"], 999), c["pref"], c["stem"]))
    seen = set()
    runs = []
    for c in candidates:
        if c["key"][0] not in MAIN_TABLE_MODELS:
            if verbose:
                skipped.append((c["sd"], c["stem"], f"not_in_main_table ({c['key'][0]})"))
            continue
        if c["key"] in seen:
            if verbose:
                skipped.append((c["sd"], c["stem"], "lower_priority_duplicate"))
            continue
        gt_dir = GT_GAIA if c["split"] == "GAIA_dedup" else GT_SWE
        rows, n_gt, n_present, n_unparsable = per_trace_scores(c["dir"], gt_dir, score_fn=score_fn)
        if not rows:
            if verbose:
                skipped.append((c["sd"], c["stem"],
                                f"no_rows (gt={n_gt} present={n_present} unparsable={n_unparsable})"))
            continue
        seen.add(c["key"])
        runs.append({
            "model": canonical(c["model_id"]),
            "split": c["split"],
            "method": c["method"],
            "rows":  rows,
            "src":   c["dir"],
        })
    if verbose:
        print(f"[skipped] {len(skipped)} dirs:")
        for sd, s, why in skipped:
            short = sd.rsplit("/", 1)[-1] if sd else "?"
            print(f"  - [{short}] {s}    [{why}]")
    return runs


# ---------------- Bucket logic ----------------

def compute_thresholds(all_scores):
    over  = [s for s in all_scores if s > 0]
    under = [-s for s in all_scores if s < 0]
    over_t  = (np.percentile(over, 33),  np.percentile(over, 67))  if over  else (0, 0)
    under_t = (np.percentile(under, 33), np.percentile(under, 67)) if under else (0, 0)
    return over_t, under_t


def bucket(score, over_t, under_t, pred=None, gold=None):
    if pred is not None and gold is not None and pred == gold:
        return "neutral"
    if score == 0:
        return "neutral"
    if score > 0:
        if score < over_t[0]:  return "over_low"
        if score < over_t[1]:  return "over_med"
        return "over_high"
    mag = -score
    if mag < under_t[0]:  return "under_low"
    if mag < under_t[1]:  return "under_med"
    return "under_high"


BUCKET_ORDER_DOWN = ["under_high", "under_med", "under_low"]
BUCKET_ORDER_UP   = ["over_low", "over_med", "over_high"]
BUCKET_COLORS = {
    "over_high":  "#7a0e0e",
    "over_med":   "#d83a3a",
    "over_low":   "#f4a3a3",
    "neutral":    "#cccccc",
    "under_low":  "#9ec5e8",
    "under_med":  "#3a78c2",
    "under_high": "#0e3a7a",
}
BUCKET_LABEL = {
    "over_high":  "Over (high)",
    "over_med":   "Over (med)",
    "over_low":   "Over (low)",
    "neutral":    "Neutral (pred = gold)",
    "under_low":  "Under (low)",
    "under_med":  "Under (med)",
    "under_high": "Under (high)",
}


# ---------------- Plotting ----------------

METHOD_ORDER = ["Baseline", "+CG", "+GI+SI"]


def plot(runs, out_path, thresholds):
    over_t, under_t = thresholds
    rows_by_method = {m: [] for m in METHOD_ORDER}
    for run in runs:
        if run["method"] not in METHOD_ORDER:
            continue
        n = len(run["rows"])
        cnt = {k: 0 for k in BUCKET_COLORS}
        for r in run["rows"]:
            cnt[bucket(r["score"], over_t, under_t, pred=r["pred"], gold=r["gold"])] += 1
        rows_by_method[run["method"]].append({
            "model": run["model"], "split": run["split"], "n": n,
            "frac": {k: cnt[k]/n for k in BUCKET_COLORS},
        })

    fig, axes = plt.subplots(1, 3, figsize=(17, 6.5), sharey=True)
    for ax, method in zip(axes, METHOD_ORDER):
        items = sorted(rows_by_method[method],
                       key=lambda r: (r["model"], 0 if r["split"]=="GAIA_dedup" else 1))
        labels = [f"{r['model']}\n{SPLIT_LABEL[r['split']]} (n={r['n']})" for r in items]
        x = np.arange(len(items))

        bottom = np.zeros(len(items))
        for b in BUCKET_ORDER_UP:
            h = np.array([r["frac"][b] for r in items])
            ax.bar(x, h, bottom=bottom, color=BUCKET_COLORS[b],
                   edgecolor="white", linewidth=0.4)
            bottom += h
        top = np.zeros(len(items))
        for b in BUCKET_ORDER_DOWN[::-1]:
            h = np.array([r["frac"][b] for r in items])
            ax.bar(x, -h, bottom=top, color=BUCKET_COLORS[b],
                   edgecolor="white", linewidth=0.4)
            top -= h
        neutral_h = np.array([r["frac"]["neutral"] for r in items])
        ax.bar(x, neutral_h, bottom=-neutral_h/2, color=BUCKET_COLORS["neutral"],
               edgecolor="white", linewidth=0.4)
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=7.5)
        ax.set_title(method)
        ax.set_ylim(-1.0, 1.0)
        ax.grid(True, axis="y", alpha=0.25)

    axes[0].set_ylabel("← under-predict     fraction of traces     over-predict →",
                       fontsize=9)
    handles = [Patch(facecolor=BUCKET_COLORS[k], edgecolor="white", label=BUCKET_LABEL[k])
               for k in ["over_high","over_med","over_low","neutral",
                         "under_low","under_med","under_high"]]
    fig.legend(handles=handles, loc="center right", bbox_to_anchor=(1.10, 0.5),
               fontsize=8, title="Bucket\n(s = log((pred+1)/(gold+1)))",
               title_fontsize=8, frameon=True)
    fig.suptitle("Per-trace count bias by model & method (TRAIL, category-presence count)",
                 fontsize=11)
    note = (f"thresholds (33rd/67th pct, pooled): "
            f"over s={over_t[0]:.2f}/{over_t[1]:.2f}   |   "
            f"under |s|={under_t[0]:.2f}/{under_t[1]:.2f}")
    fig.text(0.5, 0.005, note, ha="center", fontsize=8, color="dimgray")
    plt.tight_layout(rect=[0, 0.03, 0.92, 0.96])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[saved] {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="/data/wang/junh/githubs/trail-benchmark/baselines/outputs/_plot_count_bias_likert.png")
    p.add_argument("--csv", default="/data/wang/junh/githubs/trail-benchmark/baselines/outputs/_plot_count_bias_likert.csv")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    runs = collect(verbose=True)
    print(f"\n[runs] {len(runs)} (model, split, method) triples loaded:")
    for r in sorted(runs, key=lambda x: (x["method"], x["model"], x["split"])):
        print(f"  {r['method']:<10} {r['model']:<22} {r['split']:<18} n={len(r['rows'])}")

    all_scores = [r["score"] for run in runs for r in run["rows"]]
    over_t, under_t = compute_thresholds(all_scores)
    print(f"\n[thresholds] over={over_t}  under(|.|)={under_t}")

    with open(args.csv, "w") as f:
        f.write("model,split,method,n_traces,frac_over_high,frac_over_med,frac_over_low,frac_neutral,frac_under_low,frac_under_med,frac_under_high\n")
        for run in runs:
            n = len(run["rows"])
            cnt = {k: 0 for k in BUCKET_COLORS}
            for r in run["rows"]:
                cnt[bucket(r["score"], over_t, under_t, pred=r["pred"], gold=r["gold"])] += 1
            f.write(f"{run['model']},{run['split']},{run['method']},{n},"
                    f"{cnt['over_high']/n:.3f},{cnt['over_med']/n:.3f},{cnt['over_low']/n:.3f},"
                    f"{cnt['neutral']/n:.3f},"
                    f"{cnt['under_low']/n:.3f},{cnt['under_med']/n:.3f},{cnt['under_high']/n:.3f}\n")
    print(f"[csv] {args.csv}")
    plot(runs, args.out, (over_t, under_t))


if __name__ == "__main__":
    main()
