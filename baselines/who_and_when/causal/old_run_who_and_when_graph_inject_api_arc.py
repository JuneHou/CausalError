"""
baselines/who_and_when/causal/run_who_and_when_graph_inject_api_arc.py

ARC LLM API variant of +GI+SI (two-pass dynamic graph injection) for Who&When.
Uses Virginia Tech's OpenAI-compatible ARC endpoint for fairshare inference.

Identical logic to run_who_and_when_graph_inject_api_deepinfra.py; only the
endpoint URL, auth env var, and rate limits differ.

ARC rate limits (fairshare, as of 2025-05):
  30 req/min, 1000 req/hr, 3000 req/3hr

Intended model: gpt-oss-120b (no openai/ prefix on ARC).

Auth:
    set -a; source /data/wang/junh/.cache/keys/arc_llm_api.sh; set +a
    export ARC_LLM_API_KEY="$API_KEY"

Usage (from baselines/who_and_when/causal/):
    python run_who_and_when_graph_inject_api_arc.py \\
        --variant w1 --causal_only --span_index \\
        --split GAIA_dedup --model gpt-oss-120b

    python run_who_and_when_graph_inject_api_arc.py \\
        --variant w1 --causal_only --span_index \\
        --split SWE_Bench_dedup --model gpt-oss-120b
"""

import os
import re
import sys
import glob
import json
import time
import argparse
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from openai import OpenAI
from tqdm import tqdm

WAW_DIR   = Path(__file__).resolve().parent.parent
REPO      = WAW_DIR.parent.parent
BENCH_DIR = REPO / "benchmarking"
sys.path.insert(0, str(WAW_DIR))
sys.path.insert(0, str(BENCH_DIR))
sys.path.insert(0, str(BENCH_DIR / "eval"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_who_and_when_vllm import (           # noqa: E402
    TAXONOMY_BLOCK as WW_TAXONOMY_BLOCK,
    SCORES_PROMPT_TEMPLATE,
    _validate_scores_block,
    extract_span_ids,
    extract_task_description,
    format_trace_for_prompt,
    get_ordered_step_spans,
    parse_json_output,
)
from run_eval_graph_inject_vllm import (      # noqa: E402
    DEFAULT_CAUSAL_GRAPH,
    DEFAULT_SUPPES_GRAPH,
    build_span_index,
    format_graph_guidance,
    load_graph_edges,
    propagate_confidence,
    validate_locations,
)
from run_who_and_when_graph_inject_vllm import (  # noqa: E402
    W1_PASS1_TEMPLATE,
    W2_PASS1_STEP_TEMPLATE,
    _build_pass2_prompt,
)


ARC_BASE_URL  = "https://llm-api.arc.vt.edu/api/v1/"
DEFAULT_MODEL = "gpt-oss-120b"


# ---------------------------------------------------------------------------
# Rate limiter (sliding window, multi-rule for ARC fairshare)
# ---------------------------------------------------------------------------

class RateLimiter:
    def __init__(self, limits):
        self.limits = limits
        self.longest_window = max(w for _, w in limits) if limits else 0
        self.times: deque = deque()
        self.n_total = 0

    def acquire(self):
        if not self.limits:
            self.n_total += 1
            return
        while True:
            now = time.time()
            while self.times and now - self.times[0] > self.longest_window:
                self.times.popleft()
            sleep_for = 0.0
            offender = None
            for max_n, window in self.limits:
                in_win = [t for t in self.times if now - t < window]
                if len(in_win) >= max_n:
                    target = in_win[len(in_win) - max_n]
                    delta = target + window - now + 0.5
                    if delta > sleep_for:
                        sleep_for = delta
                        offender = (max_n, window, len(in_win))
            if sleep_for <= 0:
                self.times.append(time.time())
                self.n_total += 1
                return
            mx, win, cur = offender
            print(f"[rate-limit] {cur}/{mx} in last {win}s — sleeping "
                  f"{sleep_for:.1f}s  (calls so far: {self.n_total})")
            time.sleep(sleep_for)


# ---------------------------------------------------------------------------
# API call wrapper
# ---------------------------------------------------------------------------

def call_chat(client, model, user_text, max_tokens, limiter, max_retries=5):
    messages = [{"role": "user", "content": user_text}]
    last_err = None
    for attempt in range(max_retries):
        limiter.acquire()
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            status = getattr(e, "status_code", None)
            if status is not None and 400 <= status < 500 and status != 429:
                raise
            last_err = e
            backoff = min(60, 2 ** attempt)
            print(f"[retry {attempt+1}/{max_retries}] {type(e).__name__}: "
                  f"{str(e)[:200]} — sleeping {backoff}s")
            time.sleep(backoff)
    raise RuntimeError(f"Failed after {max_retries} retries; last error: {last_err}")


def call_scores_api(trace_text, client, model, max_tokens, limiter):
    prompt = SCORES_PROMPT_TEMPLATE.format(trace=trace_text)
    try:
        raw = call_chat(client, model, prompt, min(max_tokens, 1024), limiter)
        parsed = parse_json_output(raw)
        scores = _validate_scores_block(parsed)
        return scores, {}
    except Exception as e:
        return [], {"scores_error": str(e)}


# ---------------------------------------------------------------------------
# Pass-1 runners (W1 and W2) — API versions
# ---------------------------------------------------------------------------

def _run_w1_pass1_api(
    trace_str: str,
    client, model: str, max_tokens: int, limiter,
    graph_guidance: str, span_index: str,
) -> Tuple[Optional[dict], dict, str]:
    ordered_spans = get_ordered_step_spans(trace_str)
    task_desc     = extract_task_description(trace_str)
    valid_ids     = extract_span_ids(trace_str)
    trace_text    = format_trace_for_prompt(ordered_spans)

    user_text = W1_PASS1_TEMPLATE.format(
        taxonomy_block       = WW_TAXONOMY_BLOCK,
        graph_guidance_block = (graph_guidance + "\n") if graph_guidance else "",
        span_index_block     = (span_index + "\n\n") if span_index else "",
        task_description     = task_desc,
        trace                = trace_text,
    )

    raw = call_chat(client, model, user_text, max_tokens, limiter)
    parsed = parse_json_output(raw)
    if parsed is None:
        return None, {"error": "json_parse_failed", "raw": raw[:500]}, trace_text

    errors = parsed.get("errors", [])
    errors = [e for e in errors if (e.get("location") or "").strip() in valid_ids]
    scores, scores_meta = call_scores_api(trace_text, client, model, max_tokens, limiter)
    return (
        {"errors": errors, "scores": scores},
        {"n_raw_errors": len(parsed.get("errors", [])), **scores_meta},
        trace_text,
    )


def _run_w2_pass1_api(
    trace_str: str,
    client, model: str, max_tokens: int, limiter,
    graph_guidance: str,
) -> Tuple[Optional[dict], dict, str]:
    ordered_spans = get_ordered_step_spans(trace_str)
    task_desc     = extract_task_description(trace_str)
    valid_ids     = extract_span_ids(trace_str)

    full_trace_text = format_trace_for_prompt(ordered_spans)
    if not ordered_spans:
        return {"errors": [], "scores": []}, {"error": "no_step_spans"}, full_trace_text

    per_step_budget = min(4096, max(512, max_tokens // 8))
    graph_block     = (graph_guidance + "\n") if graph_guidance else ""
    all_errors: List[dict] = []
    seen_pairs: set = set()
    cumulative_text = ""
    meta = {"calls": 0, "error": None}

    for i, entry in enumerate(ordered_spans):
        step_num     = i + 1
        step_name    = entry["name"]
        span_id      = entry["span_id"]
        span_content = entry["content"]

        cumulative_text += (
            f"\n--- Step {step_num}: {step_name} (span_id: \"{span_id}\") ---\n"
            f"{span_content}\n"
        )

        user_text = W2_PASS1_STEP_TEMPLATE.format(
            taxonomy_block       = WW_TAXONOMY_BLOCK,
            graph_guidance_block = graph_block,
            task_description     = task_desc,
            step_num             = step_num,
            step_name            = step_name,
            span_id              = span_id,
            cumulative_spans     = cumulative_text.strip(),
        )

        try:
            raw = call_chat(client, model, user_text, per_step_budget, limiter)
        except Exception as e:
            meta["error"] = str(e)
            break

        meta["calls"] += 1
        parsed = parse_json_output(raw)
        if parsed is None or not parsed.get("has_error"):
            continue

        for err in parsed.get("errors", []):
            category = (err.get("category") or "").strip()
            pair_key = (category, span_id)
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            if span_id in valid_ids:
                all_errors.append({
                    "category":    category,
                    "location":    span_id,
                    "evidence":    err.get("evidence", ""),
                    "description": err.get("description",
                                           f"Error at step {step_num} ({step_name})."),
                    "impact":      err.get("impact", ""),
                })

    scores, scores_meta = call_scores_api(full_trace_text, client, model, max_tokens, limiter)
    meta["calls"] += 1
    meta.update(scores_meta)
    return {"errors": all_errors, "scores": scores}, meta, full_trace_text


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="W1/W2 +GI+SI via ARC LLM API (sequential, resumable)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--model",                 default=DEFAULT_MODEL,
                    help="ARC model name (gpt-oss-120b, gpt-oss-120b-thinking-high, ...).")
    ap.add_argument("--data_dir",              default=str(BENCH_DIR / "data"))
    ap.add_argument("--output_dir",            default=str(REPO / "baselines" / "who_and_when" / "causal" / "outputs"))
    ap.add_argument("--split",                 default="GAIA_dedup")
    ap.add_argument("--variant",               choices=["w1", "w2"], required=True)
    ap.add_argument("--max_tokens",            type=int, default=8000,
                    help="Auto-bumped to 24000 for gpt-oss / reasoning models.")
    ap.add_argument("--model_tag",             default=None)
    ap.add_argument("--causal_only",           action="store_true")
    ap.add_argument("--corr_threshold",        type=float, default=1.0)
    ap.add_argument("--edge_threshold",        type=float, default=0.20)
    ap.add_argument("--random_edges",          action="store_true")
    ap.add_argument("--random_seed",           type=int, default=42)
    ap.add_argument("--random_n",              type=int, default=12)
    ap.add_argument("--propagation_threshold", type=float, default=0.10)
    ap.add_argument("--span_index",            action="store_true")
    ap.add_argument("--validate_span_id",      action="store_true", default=True)
    ap.add_argument("--no_validate_span_id",   dest="validate_span_id", action="store_false")
    ap.add_argument("--causal_graph",          default=None)
    ap.add_argument("--suppes_graph",          default=None)
    ap.add_argument("--limit_traces",          type=int, default=None)
    ap.add_argument("--rpm",                   type=int, default=30,
                    help="Requests per minute  (ARC fairshare: 30).")
    ap.add_argument("--rph",                   type=int, default=1000,
                    help="Requests per hour    (ARC fairshare: 1000).")
    ap.add_argument("--rp3h",                  type=int, default=3000,
                    help="Requests per 3 hours (ARC fairshare: 3000).")
    ap.add_argument("--max_retries",           type=int, default=5)
    args = ap.parse_args()

    api_key = os.environ.get("ARC_LLM_API_KEY") or os.environ.get("API_KEY")
    if not api_key:
        print("ERROR: ARC_LLM_API_KEY (or API_KEY) not set.", file=sys.stderr)
        print("  set -a; source /data/wang/junh/.cache/keys/arc_llm_api.sh; set +a", file=sys.stderr)
        print('  export ARC_LLM_API_KEY="$API_KEY"', file=sys.stderr)
        sys.exit(1)

    is_reasoning = bool(re.search(r"(gpt-oss|qwenlong|-l1-|deepseek-r1|qwq)",
                                  args.model, re.IGNORECASE))
    if is_reasoning and args.max_tokens <= 8000:
        print(f"[INFO] reasoning model ({args.model}); bumping max_tokens 8000 → 24000")
        args.max_tokens = 24000

    causal_graph = Path(args.causal_graph) if args.causal_graph else DEFAULT_CAUSAL_GRAPH
    suppes_graph = Path(args.suppes_graph) if args.suppes_graph else DEFAULT_SUPPES_GRAPH
    edges = load_graph_edges(
        threshold      = args.edge_threshold,
        causal_only    = args.causal_only,
        corr_threshold = args.corr_threshold,
        causal_graph   = causal_graph,
        suppes_graph   = suppes_graph,
        random_edges   = args.random_edges,
        random_seed    = args.random_seed,
        random_n       = args.random_n,
    )
    if args.random_edges:
        graph_tag = f"random{args.random_n}_seed{args.random_seed}"
    elif args.causal_only:
        graph_tag = "causal_only"
    elif args.corr_threshold < 1.0:
        graph_tag = f"causal_corr{args.corr_threshold}"
    else:
        graph_tag = f"suppes_t{args.edge_threshold}"
    print(f"Loaded {len(edges)} edges ({graph_tag})")
    for src, dst, w in edges[:10]:
        print(f"    {src} → {dst}  ({w:.3f})")

    graph_guidance = format_graph_guidance(
        edges, causal_only=args.causal_only, random_edges=args.random_edges,
    )

    model_tag = args.model_tag if args.model_tag else args.model.replace("/", "-")
    span_tag  = "_span_index" if args.span_index else ""
    out_dir = os.path.join(
        args.output_dir,
        f"outputs_{model_tag}-{args.split}-who_and_when_{args.variant}_graph_inject_{graph_tag}{span_tag}",
    )
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output dir: {out_dir}")

    if glob.glob(os.path.join(args.data_dir, "*.json")):
        data_dir = args.data_dir
    else:
        data_dir = os.path.join(args.data_dir, args.split)
    file_paths = sorted(glob.glob(f"{data_dir}/*.json"))

    pending = [fp for fp in file_paths
               if not os.path.exists(
                   os.path.join(out_dir, os.path.splitext(os.path.basename(fp))[0] + ".json")
               )]
    print(f"Found {len(file_paths)} traces. Pending: {len(pending)} "
          f"(skipping {len(file_paths) - len(pending)} already done)")
    if args.limit_traces:
        pending = pending[:args.limit_traces]
        print(f"limit_traces={args.limit_traces}; processing {len(pending)}")
    if not pending:
        print("Nothing to do.")
        return

    if args.variant == "w1":
        est_p1 = len(pending)
    else:
        est_p1 = sum(
            len(get_ordered_step_spans(open(fp).read())) for fp in pending
        )
    print(f"Estimated Pass-1 calls: {est_p1}; Pass-2 adds up to {len(pending)} more.")

    client = OpenAI(base_url=ARC_BASE_URL, api_key=api_key)
    limits = []
    if args.rpm  > 0: limits.append((args.rpm,   60))
    if args.rph  > 0: limits.append((args.rph,   3600))
    if args.rp3h > 0: limits.append((args.rp3h,  10800))
    limiter = RateLimiter(limits)

    skipped = p2_triggered_total = 0

    for fp in tqdm(pending, desc=f"who_and_when_{args.variant}+GI+SI"):
        trace_id  = os.path.splitext(os.path.basename(fp))[0]
        out_file  = os.path.join(out_dir, f"{trace_id}.json")
        meta_file = os.path.join(out_dir, f"_meta_{trace_id}.json")

        with open(fp) as f:
            trace_str = f.read()

        span_index_text = build_span_index(trace_str) if args.span_index else ""

        meta: Dict = {
            "trace_id":            trace_id,
            "variant":             args.variant,
            "graph":               graph_tag,
            "api":                 "arc",
            "model":               args.model,
            "pass1_detected":      [],
            "pass2_triggered":     False,
            "pass2_filtered_edges": 0,
            "pass2_new_errors":    0,
            "pass2_parse_failed":  False,
            "p2_dropped":          0,
        }

        try:
            if args.variant == "w1":
                p1_output, p1_meta, trace_text = _run_w1_pass1_api(
                    trace_str, client, args.model, args.max_tokens, limiter,
                    graph_guidance, span_index_text,
                )
            else:
                p1_output, p1_meta, trace_text = _run_w2_pass1_api(
                    trace_str, client, args.model, args.max_tokens, limiter,
                    graph_guidance,
                )
        except Exception as e:
            print(f"[ERROR] {trace_id}: pass1 failed: {e}")
            with open(out_file, "w") as f:
                json.dump({"errors": [], "scores": [], "_error": str(e)}, f)
            with open(meta_file, "w") as f:
                json.dump({**meta, "error": str(e)}, f, indent=2)
            skipped += 1
            continue

        meta["pass1_meta"] = p1_meta
        if p1_output is None:
            with open(out_file, "w") as f:
                json.dump({"errors": [], "scores": [],
                           "_error": p1_meta.get("error", "pass1_failed")}, f)
            with open(meta_file, "w") as f:
                json.dump(meta, f, indent=2)
            skipped += 1
            continue

        p1_errors = p1_output.get("errors", [])
        p1_scores = p1_output.get("scores", [])
        detected_cats = list({e["category"] for e in p1_errors})
        meta["pass1_detected"] = detected_cats

        p2_errors: List[dict] = []
        if detected_cats and edges:
            filtered_edges = propagate_confidence(detected_cats, edges,
                                                  args.propagation_threshold)
            meta["pass2_filtered_edges"] = len(filtered_edges)
            if filtered_edges:
                meta["pass2_triggered"] = True
                p2_triggered_total += 1
                p2_user_text = _build_pass2_prompt(
                    trace_str, p1_errors, filtered_edges, span_index=span_index_text
                )
                try:
                    p2_raw = call_chat(client, args.model, p2_user_text,
                                       args.max_tokens, limiter)
                    p2_parsed = parse_json_output(p2_raw)
                    if p2_parsed is None:
                        meta["pass2_parse_failed"] = True
                    else:
                        p2_errors = p2_parsed.get("errors", [])
                        if args.validate_span_id:
                            valid_ids = extract_span_ids(trace_str)
                            p2_errors, p2_dropped = validate_locations(p2_errors, valid_ids)
                            meta["p2_dropped"] = p2_dropped
                        p1_cats = {e["category"] for e in p1_errors}
                        p2_errors = [e for e in p2_errors
                                     if e.get("category") not in p1_cats]
                        meta["pass2_new_errors"] = len(p2_errors)
                except Exception as e:
                    meta["pass2_error"] = str(e)

        merged = p1_errors + p2_errors
        with open(out_file, "w") as f:
            json.dump({"errors": merged, "scores": p1_scores}, f, indent=2)
        with open(meta_file, "w") as f:
            json.dump(meta, f, indent=2)

    n_done = len(pending) - skipped
    print(f"\nDone. {n_done} processed, {skipped} skipped.")
    print(f"Pass 2 triggered: {p2_triggered_total} traces.")
    print(f"Total API calls: {limiter.n_total}")
    print(f"Score: python eval/calculate_scores.py --results_dir {args.output_dir}")


if __name__ == "__main__":
    main()
