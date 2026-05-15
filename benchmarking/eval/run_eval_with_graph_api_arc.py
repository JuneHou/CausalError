"""
eval/run_eval_with_graph_api_arc.py

ARC LLM API variant of the one-pass +CG eval (in-prompt graph guidance,
no two-pass dynamic injection). Mirrors run_eval_with_graph_vllm.py in
prompt logic and output format; mirrors run_eval_graph_inject_api_arc.py
in auth, rate limiting, retries, and sweep-flag support.

ARC rate limits (fairshare, as of 2025-05):
  30 req/min, 1000 req/hr, 3000 req/3hr

Intended model: gpt-oss-120b (no openai/ prefix on ARC).

Auth:
    set -a; source /data/wang/junh/.cache/keys/arc_llm_api.sh; set +a
    export ARC_LLM_API_KEY="$API_KEY"

Usage (from benchmarking/):
    # causal_only (anchor point)
    python eval/run_eval_with_graph_api_arc.py \\
        --model gpt-oss-120b \\
        --split GAIA_dedup --causal_only \\
        --output_dir outputs_thres/t_causal_only

    # threshold sweep: τ = 0.35  (union of causal + Suppes geomean >= 0.35)
    python eval/run_eval_with_graph_api_arc.py \\
        --model gpt-oss-120b \\
        --split GAIA_dedup --corr_threshold 0.35 \\
        --output_dir outputs_thres/t0.35

    # threshold sweep: τ = 0.25
    python eval/run_eval_with_graph_api_arc.py \\
        --model gpt-oss-120b \\
        --split SWE_Bench_dedup --corr_threshold 0.25 \\
        --output_dir outputs_thres/t0.25

    # random-12 null baseline
    python eval/run_eval_with_graph_api_arc.py \\
        --model gpt-oss-120b \\
        --split GAIA_dedup --random_edges \\
        --output_dir outputs_thres/t_random12_seed42

    # legacy plain-Suppes threshold (no causal union)
    python eval/run_eval_with_graph_api_arc.py \\
        --model gpt-oss-120b \\
        --split GAIA_dedup --edge_threshold 0.20 \\
        --output_dir outputs_thres/t0.20
"""

import os
import re
import sys
import glob
import time
import argparse
from collections import deque
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

BENCH_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BENCH_DIR))
sys.path.insert(0, str(BENCH_DIR / "eval"))

# Graph utilities (with corr_threshold + random_edges support) come from
# the +GI runner, since the +CG vllm runner's load_graph_edges does not
# support the corr-union or random-edges modes used by the sweep.
from run_eval_graph_inject_vllm import (   # noqa: E402
    DEFAULT_CAUSAL_GRAPH,
    DEFAULT_SUPPES_GRAPH,
    build_span_index,
    format_graph_guidance,
    load_graph_edges,
)
# Single-pass +CG prompt comes from the +CG vllm runner.
from run_eval_with_graph_vllm import get_prompt   # noqa: E402


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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="One-pass +CG eval via ARC LLM API (sequential, resumable)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--model",          default=DEFAULT_MODEL,
                    help="ARC model name (gpt-oss-120b, gpt-oss-120b-thinking-high, ...).")
    ap.add_argument("--data_dir",       default="data")
    ap.add_argument("--output_dir",     default="outputs/zero_shot2")
    ap.add_argument("--split",          default="GAIA_dedup")
    ap.add_argument("--max_tokens",     type=int, default=8000,
                    help="Auto-bumped to 24000 for gpt-oss / reasoning models.")
    ap.add_argument("--model_tag",      default=None)
    ap.add_argument("--span_index",     action="store_true", default=False,
                    help="Prepend compact span_id index to each prompt")
    ap.add_argument("--causal_only",    action="store_true", default=False,
                    help="Use only the intervention-validated causal edges")
    ap.add_argument("--corr_threshold", type=float, default=1.0,
                    help="Include causal + Suppes edges with geomean >= this "
                         "(union semantics, matches the +GI sweep). "
                         "Ignored if --causal_only.")
    ap.add_argument("--edge_threshold", type=float, default=0.20,
                    help="Plain Suppes geomean threshold (NO causal union). "
                         "Used only when neither --causal_only nor "
                         "--corr_threshold<1.0 is set. Kept for back-compat "
                         "with existing outputs_thres/*/graph_t<τ>/ runs.")
    ap.add_argument("--random_edges",   action="store_true",
                    help="Random-12 null baseline (sampled from non-Suppes pairs).")
    ap.add_argument("--random_seed",    type=int, default=42)
    ap.add_argument("--random_n",       type=int, default=12)
    ap.add_argument("--causal_graph",   default=None)
    ap.add_argument("--suppes_graph",   default=None)
    ap.add_argument("--limit_traces",   type=int, default=None)
    ap.add_argument("--rpm",            type=int, default=30,
                    help="Requests per minute  (ARC fairshare: 30).")
    ap.add_argument("--rph",            type=int, default=1000,
                    help="Requests per hour    (ARC fairshare: 1000).")
    ap.add_argument("--rp3h",           type=int, default=3000,
                    help="Requests per 3 hours (ARC fairshare: 3000).")
    ap.add_argument("--max_retries",    type=int, default=5)
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

    # ------------------------------------------------------------------
    # Load graph (uses the +GI runner's loader for corr_threshold/random_edges)
    # ------------------------------------------------------------------
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
        print(f"  {len(edges)} random edges (seed={args.random_seed})")
    elif args.causal_only:
        graph_tag = "causal_only"
        print(f"  {len(edges)} edges (causal_only)")
    elif args.corr_threshold < 1.0:
        graph_tag = f"causal_corr{args.corr_threshold}"
        print(f"  {len(edges)} edges (causal + corr geomean>={args.corr_threshold})")
    else:
        graph_tag = f"t{args.edge_threshold}"
        print(f"  {len(edges)} edges (geomean>={args.edge_threshold})")
    for src, dst, w in edges[:10]:
        print(f"    {src} → {dst}  ({w:.3f})")
    if len(edges) > 10:
        print(f"    ... and {len(edges)-10} more")

    graph_guidance = format_graph_guidance(
        edges, causal_only=args.causal_only, random_edges=args.random_edges,
    )

    # ------------------------------------------------------------------
    # Output directory (mirrors run_eval_with_graph_vllm.py naming:
    # outputs_<model>-<split>-graph_<tag>[_span_index]/)
    # ------------------------------------------------------------------
    model_tag = args.model_tag if args.model_tag else args.model.replace("/", "-")
    span_tag  = "_span_index" if args.span_index else ""
    out_dir = os.path.join(
        args.output_dir,
        f"outputs_{model_tag}-{args.split}-graph_{graph_tag}{span_tag}",
    )
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output dir: {out_dir}")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    if glob.glob(os.path.join(args.data_dir, "*.json")):
        data_dir = args.data_dir
    else:
        data_dir = os.path.join(args.data_dir, args.split)
    file_paths = sorted(glob.glob(f"{data_dir}/*.json"))

    pending = [fp for fp in file_paths
               if not os.path.exists(os.path.join(out_dir, os.path.basename(fp)))]
    print(f"Found {len(file_paths)} traces. Pending: {len(pending)} "
          f"(skipping {len(file_paths) - len(pending)} already done)")
    if args.limit_traces:
        pending = pending[:args.limit_traces]
        print(f"limit_traces={args.limit_traces}; processing {len(pending)}")
    if not pending:
        print("Nothing to do.")
        return

    client = OpenAI(base_url=ARC_BASE_URL, api_key=api_key)
    limits = []
    if args.rpm  > 0: limits.append((args.rpm,   60))
    if args.rph  > 0: limits.append((args.rph,   3600))
    if args.rp3h > 0: limits.append((args.rp3h,  10800))
    limiter = RateLimiter(limits)
    skipped = 0

    for fp in tqdm(pending):
        out_file = os.path.join(out_dir, os.path.basename(fp))
        with open(fp) as f:
            trace_str = f.read()
        span_idx  = build_span_index(trace_str) if args.span_index else ""
        user_text = get_prompt(trace_str, span_index=span_idx,
                               graph_guidance=graph_guidance)
        try:
            response = call_chat(client, args.model, user_text,
                                 args.max_tokens, limiter,
                                 max_retries=args.max_retries)
        except Exception as e:
            print(f"\nError on {os.path.basename(fp)}: {e}")
            response = "Error processing file. No output generated."
            skipped += 1
        with open(out_file, "w") as f:
            f.write(response)

    n_done = len(pending) - skipped
    print(f"\nDone. {n_done} processed, {skipped} skipped.")
    print(f"Total API calls: {limiter.n_total}")
    print(f"Score: python eval/calculate_scores.py --results_dir {args.output_dir}")


if __name__ == "__main__":
    main()
