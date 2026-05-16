"""
baselines/who_and_when/causal/run_who_and_when_with_graph_api_deepinfra.py

DeepInfra-API variant of +CG (+span_index) (one-pass causal graph in prompt)
for Who&When W1 and W2.

Used primarily for W2+CG+SI: graph guidance injected into every per-step prompt
so the model sees causal priors while stepping through spans. W1+CG+SI is also
supported (one call over the full trace).

Unlike +GI+SI (run_who_and_when_graph_inject_api_deepinfra.py) there is NO
second pass; the causal guidance is baked directly into the prompt.

Models supported (no Qwen):
    openai/gpt-oss-20b          (reasoning, bumped to 24k output tokens)
    openai/gpt-oss-120b         (reasoning, bumped to 24k output tokens)
    mistralai/Mistral-Small-3.1-24B-Instruct-2503

Auth:
    set -a; source /data/wang/junh/.cache/keys/deepinfra.sh; set +a
    export DEEPINFRA_API_KEY="$API_KEY"

Usage (from baselines/who_and_when/causal/):
    # W2 + CG + SI  (primary use)
    python run_who_and_when_with_graph_api_deepinfra.py \\
        --variant w2 --causal_only --span_index \\
        --split GAIA_dedup --model openai/gpt-oss-120b

    # W1 + CG + SI
    python run_who_and_when_with_graph_api_deepinfra.py \\
        --variant w1 --causal_only --span_index \\
        --split SWE_Bench_dedup --model openai/gpt-oss-20b
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
from run_eval_with_graph_vllm import (        # noqa: E402
    DEFAULT_CAUSAL_GRAPH,
    DEFAULT_SUPPES_GRAPH,
    format_graph_guidance,
)
# load_graph_edges is imported from the +GI eval because that copy carries the
# full --corr_threshold / --random_edges surface that we added to this runner's
# flags; the +CG eval's copy still uses the legacy (causal_only, threshold)
# signature. format_graph_guidance stays in run_eval_with_graph_vllm — it's
# +CG-specific (one-pass prompt block), independent of which loader produced the edges.
from run_eval_graph_inject_vllm import (      # noqa: E402
    build_span_index,
    load_graph_edges,
)
from run_who_and_when_with_graph_vllm import (  # noqa: E402
    W1_PROMPT_TEMPLATE,
    W2_STEP_PROMPT_TEMPLATE,
)


DEEPINFRA_BASE_URL = "https://api.deepinfra.com/v1/openai"
DEFAULT_MODEL      = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"


# ---------------------------------------------------------------------------
# Rate limiter (sliding window)
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
            print(f"[rate-limit] {cur}/{mx} in last {win}s — sleeping {sleep_for:.1f}s")
            time.sleep(sleep_for)


# ---------------------------------------------------------------------------
# API call wrapper
# ---------------------------------------------------------------------------

# Module-level extras merged into every chat.completions.create call. Populated
# from CLI in main() — used for reasoning_effort='low' on gpt-oss-20b (without
# it, DeepInfra's default 'high' burns the 24k max_tokens on thinking and the
# JSON tail gets truncated → Pass-1 parse failures).
_API_EXTRA: Dict = {}

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
                **_API_EXTRA,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            status = getattr(e, "status_code", None)
            if status is not None and 400 <= status < 500 and status != 429:
                raise
            # DeepInfra wraps upstream 400s (context-length overflow in particular)
            # inside a 500 InternalServerError, so the status check above misses
            # them. Without this short-circuit we'd burn ~31s of backoff (1+2+4+8+16)
            # re-sending an oversized prompt that will never fit.
            err_str = str(e)
            if "maximum context length" in err_str or "BadRequestError" in err_str:
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
# W1 runner — one call over the full trace with graph in prompt
# ---------------------------------------------------------------------------

def run_w1_api(
    trace_str: str,
    client, model: str, max_tokens: int, limiter,
    graph_guidance: str, span_index: str,
) -> Tuple[Optional[dict], dict]:
    ordered_spans = get_ordered_step_spans(trace_str)
    task_desc     = extract_task_description(trace_str)
    valid_ids     = extract_span_ids(trace_str)
    trace_text    = format_trace_for_prompt(ordered_spans)

    user_text = W1_PROMPT_TEMPLATE.format(
        taxonomy_block       = WW_TAXONOMY_BLOCK,
        graph_guidance_block = (graph_guidance + "\n") if graph_guidance else "",
        span_index_block     = (span_index + "\n\n") if span_index else "",
        task_description     = task_desc,
        trace                = trace_text,
    )

    raw = call_chat(client, model, user_text, max_tokens, limiter)
    parsed = parse_json_output(raw)
    if parsed is None:
        return None, {"error": "json_parse_failed", "raw": raw[:500]}

    errors = parsed.get("errors", [])
    errors = [e for e in errors if (e.get("location") or "").strip() in valid_ids]
    scores, scores_meta = call_scores_api(trace_text, client, model, max_tokens, limiter)
    return (
        {"errors": errors, "scores": scores},
        {"n_raw_errors": len(parsed.get("errors", [])), **scores_meta},
    )


# ---------------------------------------------------------------------------
# W2 runner — one call per step, graph in every step prompt
# ---------------------------------------------------------------------------

def run_w2_api(
    trace_str: str,
    client, model: str, max_tokens: int, limiter,
    graph_guidance: str,
) -> Tuple[Optional[dict], dict]:
    ordered_spans = get_ordered_step_spans(trace_str)
    task_desc     = extract_task_description(trace_str)
    valid_ids     = extract_span_ids(trace_str)

    full_trace_text = format_trace_for_prompt(ordered_spans)
    if not ordered_spans:
        return {"errors": [], "scores": []}, {"error": "no_step_spans"}

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

        user_text = W2_STEP_PROMPT_TEMPLATE.format(
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
    return {"errors": all_errors, "scores": scores}, meta


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="W1/W2 +CG+SI (one-pass in-prompt) via DeepInfra API (sequential, resumable)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--model",         default=DEFAULT_MODEL,
                    help="DeepInfra model id.")
    ap.add_argument("--data_dir",      default=str(BENCH_DIR / "data"))
    ap.add_argument("--output_dir",    default=str(REPO / "baselines" / "who_and_when" / "causal" / "outputs"))
    ap.add_argument("--split",         default="GAIA_dedup")
    ap.add_argument("--variant",       choices=["w1", "w2"], required=True)
    ap.add_argument("--max_tokens",    type=int, default=8000,
                    help="Auto-bumped to 24000 for gpt-oss / reasoning models.")
    ap.add_argument("--model_tag",     default=None)
    ap.add_argument("--causal_only",   action="store_true")
    ap.add_argument("--corr_threshold", type=float, default=1.0,
                    help="Causal-union threshold: include intervention-validated causal edges UNION Suppes edges with geomean sqrt(precedence*PR_delta) >= this. Set e.g. 0.35 for the corr graph. Ignored if --causal_only.")
    ap.add_argument("--edge_threshold", type=float, default=0.20,
                    help="Pure-Suppes threshold (no causal union) using the same geomean sqrt(precedence*PR_delta) score. Only used when neither --causal_only nor --corr_threshold<1 is set.")
    ap.add_argument("--random_edges",  action="store_true",
                    help="Sample N random non-Suppes edges as a null-graph control.")
    ap.add_argument("--random_seed",   type=int, default=42)
    ap.add_argument("--random_n",      type=int, default=12)
    ap.add_argument("--causal_graph",  default=None)
    ap.add_argument("--suppes_graph",  default=None)
    ap.add_argument("--span_index",    action="store_true",
                    help="Inject span-id index into W1 prompt (W2 omits it by design).")
    ap.add_argument("--limit_traces",  type=int, default=None)
    ap.add_argument("--rpm",           type=int, default=600)
    ap.add_argument("--max_retries",   type=int, default=5)
    ap.add_argument("--reasoning_effort", default="auto",
                    choices=["auto", "low", "medium", "high", "none"],
                    help="DeepInfra reasoning_effort for gpt-oss models. "
                         "'auto' (default): set to 'low' for gpt-oss-20b "
                         "(without this, the default 'high' burns max_tokens "
                         "on thinking and truncates the JSON tail), otherwise "
                         "unset. 'none' explicitly omits the parameter.")
    args = ap.parse_args()

    api_key = os.environ.get("DEEPINFRA_API_KEY") or os.environ.get("API_KEY")
    if not api_key:
        print("ERROR: DEEPINFRA_API_KEY (or API_KEY) not set.", file=sys.stderr)
        print("  set -a; source /data/wang/junh/.cache/keys/deepinfra.sh; set +a", file=sys.stderr)
        print('  export DEEPINFRA_API_KEY="$API_KEY"', file=sys.stderr)
        sys.exit(1)

    is_reasoning = bool(re.search(r"(gpt-oss|qwenlong|-l1-|deepseek-r1|qwq)",
                                  args.model, re.IGNORECASE))
    if is_reasoning and args.max_tokens <= 8000:
        print(f"[INFO] reasoning model ({args.model}); bumping max_tokens 8000 → 24000")
        args.max_tokens = 24000

    # Resolve reasoning_effort and stash on the module-level _API_EXTRA so
    # every call_chat invocation picks it up. 'auto' → 'low' for gpt-oss-20b,
    # else unset.
    if args.reasoning_effort == "auto":
        if re.search(r"gpt-oss-20b", args.model, re.IGNORECASE):
            _API_EXTRA["reasoning_effort"] = "low"
            print(f"[INFO] gpt-oss-20b detected; setting reasoning_effort='low' "
                  f"(thinking would otherwise consume the {args.max_tokens}-token "
                  f"budget and truncate the JSON answer)")
    elif args.reasoning_effort != "none":
        _API_EXTRA["reasoning_effort"] = args.reasoning_effort
        print(f"[INFO] reasoning_effort='{args.reasoning_effort}' (user override)")

    causal_graph = Path(args.causal_graph) if args.causal_graph else DEFAULT_CAUSAL_GRAPH
    suppes_graph = Path(args.suppes_graph) if args.suppes_graph else DEFAULT_SUPPES_GRAPH
    edges = load_graph_edges(
        causal_only    = args.causal_only,
        threshold      = args.edge_threshold,
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
        graph_tag = f"t{args.edge_threshold}"
    print(f"Loaded {len(edges)} edges ({graph_tag})")
    for src, dst, w in edges[:10]:
        print(f"    {src} → {dst}  ({w:.3f})")

    graph_guidance = format_graph_guidance(edges, causal_only=args.causal_only)

    model_tag = args.model_tag if args.model_tag else args.model.replace("/", "-")
    span_tag  = "_span_index" if args.span_index else ""
    out_dir = os.path.join(
        args.output_dir,
        f"outputs_{model_tag}-{args.split}-who_and_when_{args.variant}_graph_{graph_tag}{span_tag}",
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

    client  = OpenAI(base_url=DEEPINFRA_BASE_URL, api_key=api_key)
    limiter = RateLimiter([(args.rpm, 60)] if args.rpm > 0 else [])
    skipped = 0

    for fp in tqdm(pending, desc=f"who_and_when_{args.variant}+CG{span_tag}"):
        trace_id  = os.path.splitext(os.path.basename(fp))[0]
        out_file  = os.path.join(out_dir, f"{trace_id}.json")
        meta_file = os.path.join(out_dir, f"_meta_{trace_id}.json")

        with open(fp) as f:
            trace_str = f.read()

        span_index_text = build_span_index(trace_str) if (args.span_index and args.variant == "w1") else ""

        meta: Dict = {
            "trace_id": trace_id,
            "variant":  args.variant,
            "graph":    graph_tag,
            "api":      "deepinfra",
            "model":    args.model,
        }

        try:
            if args.variant == "w1":
                output, run_meta = run_w1_api(
                    trace_str, client, args.model, args.max_tokens, limiter,
                    graph_guidance, span_index_text,
                )
            else:
                output, run_meta = run_w2_api(
                    trace_str, client, args.model, args.max_tokens, limiter,
                    graph_guidance,
                )
        except Exception as e:
            print(f"[ERROR] {trace_id}: {e}")
            with open(out_file, "w") as f:
                json.dump({"errors": [], "scores": [], "_error": str(e)}, f)
            with open(meta_file, "w") as f:
                json.dump({**meta, "error": str(e)}, f, indent=2)
            skipped += 1
            continue

        meta.update(run_meta)
        if output is None:
            output = {"errors": [], "scores": [],
                      "_error": run_meta.get("error", "unknown")}
            skipped += 1

        with open(out_file, "w") as f:
            json.dump(output, f, indent=2)
        with open(meta_file, "w") as f:
            json.dump(meta, f, indent=2)

    n_done = len(pending) - skipped
    print(f"\nDone. {n_done} processed, {skipped} skipped.")
    print(f"Total API calls: {limiter.n_total}")
    print(f"Score: python eval/calculate_scores.py --results_dir {args.output_dir}")


if __name__ == "__main__":
    main()
