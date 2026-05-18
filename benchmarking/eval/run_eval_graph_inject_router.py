"""
eval/run_eval_graph_inject_router.py

OpenRouter wrapper for the two-pass graph-inject eval.

This script intentionally does NOT modify the original runner
`run_eval_graph_inject.py`. Instead, it monkey-patches only the request
function so the existing graph logic / parsing / output layout remain unchanged.

Typical usage:
  export OPENROUTER_API_KEY=...
  python benchmarking/eval/run_eval_graph_inject_router.py \
    --model "openrouter/google/gemini-2.5-flash" \
    --split GAIA_dedup \
    --corr_threshold 0.35 --span_index \
    --output_dir benchmarking/outputs_thres/t0.35
"""

import argparse
import glob
import math
import os
import re
import shutil
import sys
import tempfile
import time
from typing import Dict, Optional

import litellm
from litellm import RateLimitError, completion

import run_eval_graph_inject as base


def _build_router_call(
    *,
    api_base: str,
    api_key: str,
    request_interval_sec: float,
    max_retries: int,
    extra_headers: Dict[str, str],
    default_max_completion_tokens: int,
    pass1_reasoning_effort: str,
    max_prompt_tokens: int,
):
    def _safe_token_count(model: str, text: str) -> int:
        messages = [{"role": "user", "content": text}]
        try:
            return int(litellm.token_counter(model=model, messages=messages))
        except Exception:  # noqa: BLE001
            # OpenRouter model ids often have a prefix (openrouter/...) that token_counter
            # may not recognize; retry without it.
            if model.startswith("openrouter/"):
                try:
                    bare = model.split("/", 1)[1]
                    return int(litellm.token_counter(model=bare, messages=messages))
                except Exception:  # noqa: BLE001
                    pass
            # Last-resort approximation.
            return max(1, math.ceil(len(text) / 4))

    def _trim_text_to_limit(model: str, text: str, limit: int) -> str:
        if _safe_token_count(model, text) <= limit:
            return text
        lo, hi = 0, len(text)
        best = ""
        while lo <= hi:
            mid = (lo + hi) // 2
            cand = text[:mid]
            if _safe_token_count(model, cand) <= limit:
                best = cand
                lo = mid + 1
            else:
                hi = mid - 1
        return best

    def _truncate_prompt_if_needed(prompt: str, model: str) -> str:
        if max_prompt_tokens <= 0:
            return prompt
        original_tokens = _safe_token_count(model, prompt)
        if original_tokens <= max_prompt_tokens:
            return prompt

        markers = ["The data to analyze:\n\n", "The trace to analyze:\n\n"]
        marker_pos = -1
        marker_len = 0
        for m in markers:
            pos = prompt.rfind(m)
            if pos > marker_pos:
                marker_pos = pos
                marker_len = len(m)

        if marker_pos >= 0:
            prefix = prompt[: marker_pos + marker_len]
            body = prompt[marker_pos + marker_len :]
            prefix_tokens = _safe_token_count(model, prefix)
            if prefix_tokens >= max_prompt_tokens:
                # Pathological case: even prompt header exceeds budget.
                trimmed = _trim_text_to_limit(model, prefix, max_prompt_tokens)
                print(
                    f"  [router-truncate] prompt header alone exceeded limit: "
                    f"{original_tokens} -> {_safe_token_count(model, trimmed)} tokens"
                )
                return trimmed

            # Binary-search largest body prefix that still fits.
            remaining = max_prompt_tokens - prefix_tokens
            body_trim = _trim_text_to_limit(model, body, remaining)
            truncated = prefix + body_trim
        else:
            # Fallback if marker is not found.
            truncated = _trim_text_to_limit(model, prompt, max_prompt_tokens)

        final_tokens = _safe_token_count(model, truncated)
        print(
            f"  [router-truncate] prompt truncated for token budget: "
            f"{original_tokens} -> {final_tokens} (limit={max_prompt_tokens})"
        )
        return truncated

    def _call_litellm_router(
        prompt: str,
        model: str,
        max_completion_tokens: Optional[int] = None,
        use_reasoning: bool = True,
    ) -> str:
        prompt = _truncate_prompt_if_needed(prompt, model)
        messages = [{"role": "user", "content": prompt}]
        is_reasoning_model = (
            "o1" in model
            or "o3" in model
            or "o4" in model
            or "anthropic" in model
            or "gemini-2.5" in model
            or "gpt-oss" in model
        )

        if is_reasoning_model and use_reasoning:
            params = {
                "messages": messages,
                "model": model,
                "reasoning_effort": pass1_reasoning_effort,
                "drop_params": True,
                "api_base": api_base,
                "api_key": api_key,
            }
        else:
            # Keep original behavior parity:
            # disable thinking for Gemini Flash pass-2 but not Pro.
            re_effort = "none" if ("gemini-2.5-flash" in model) else None
            params = {
                "messages": messages,
                "model": model,
                "temperature": 0.0,
                "top_p": 1,
                "reasoning_effort": re_effort,
                "drop_params": True,
                "api_base": api_base,
                "api_key": api_key,
            }

        if extra_headers:
            params["extra_headers"] = extra_headers
        effective_max_tokens = (
            max_completion_tokens
            if max_completion_tokens is not None
            else default_max_completion_tokens
        )
        params["max_completion_tokens"] = effective_max_tokens

        for attempt in range(max_retries):
            try:
                response = completion(**params)
                if request_interval_sec > 0:
                    time.sleep(request_interval_sec)
                content = response.choices[0].message["content"]
                if content is None:
                    finish_reason = response.choices[0].finish_reason
                    raise RuntimeError(
                        f"Model returned content=None (finish_reason={finish_reason!r}). "
                        f"Check model/provider compatibility for '{model}'."
                    )
                return content
            except RateLimitError:
                wait_sec = min(120, 5 * (2**attempt))
                print(
                    f"  Rate limit (attempt {attempt+1}/{max_retries}): "
                    f"sleeping {wait_sec}s..."
                )
                time.sleep(wait_sec)
            except Exception as exc:  # noqa: BLE001
                # OpenRouter can surface 429-like issues through provider-specific errors.
                err = str(exc).lower()
                is_rl = ("429" in err) or ("rate limit" in err)
                # Handle OpenRouter 402 token-affordability failures by shrinking
                # max_completion_tokens and retrying.
                # Example:
                # "You requested up to 16000 tokens, but can only afford 15542."
                m = re.search(
                    r"requested up to\s+(\d+)\s+tokens,\s+but can only afford\s+(\d+)",
                    err,
                )
                if m:
                    requested = int(m.group(1))
                    affordable = int(m.group(2))
                    # Keep a small safety margin to avoid exact-boundary failures.
                    new_cap = max(256, min(params["max_completion_tokens"], affordable - 32))
                    if new_cap < params["max_completion_tokens"]:
                        print(
                            "  [router-token-cap] OpenRouter affordability limit hit: "
                            f"requested={requested}, affordable={affordable}. "
                            f"Retrying with max_completion_tokens={new_cap}."
                        )
                        params["max_completion_tokens"] = new_cap
                        continue
                if is_rl and attempt < max_retries - 1:
                    wait_sec = min(120, 5 * (2**attempt))
                    print(
                        f"  Provider rate issue (attempt {attempt+1}/{max_retries}): "
                        f"sleeping {wait_sec}s..."
                    )
                    time.sleep(wait_sec)
                    continue
                raise
        raise RuntimeError(f"Exceeded {max_retries} retries for model {model}")

    return _call_litellm_router


def _pop_forwarded_arg(tokens, name: str):
    """Remove `--name value` from token list and return (new_tokens, value_or_none)."""
    out = []
    i = 0
    found = None
    while i < len(tokens):
        tok = tokens[i]
        if tok == name and i + 1 < len(tokens):
            found = tokens[i + 1]
            i += 2
            continue
        out.append(tok)
        i += 1
    return out, found


def _get_forwarded_arg(tokens, name: str, default: str):
    for i, tok in enumerate(tokens):
        if tok == name and i + 1 < len(tokens):
            return tokens[i + 1]
    return default


def _set_forwarded_arg(tokens, name: str, value: str):
    out = []
    i = 0
    replaced = False
    while i < len(tokens):
        tok = tokens[i]
        if tok == name and i + 1 < len(tokens):
            out.extend([name, value])
            i += 2
            replaced = True
            continue
        out.append(tok)
        i += 1
    if not replaced:
        out.extend([name, value])
    return out


def _resolve_split_dir(data_dir: str, split: str) -> str:
    """
    Resolve split directory robustly across common invocation locations.

    Supports running from:
      - benchmarking/                (data/GAIA_dedup)
      - repo root                    (benchmarking/data/GAIA_dedup)
    """
    candidates = [os.path.join(data_dir, split)]
    if not os.path.isabs(data_dir):
        candidates.append(os.path.join("benchmarking", data_dir, split))
        candidates.append(os.path.join(str(base.BENCH_DIR), data_dir, split))
    for c in candidates:
        if glob.glob(os.path.join(c, "*.json")):
            return c
    raise FileNotFoundError(
        f"No trace files found for split '{split}'. Tried: {candidates}. "
        "Set --data_dir/--split correctly."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OpenRouter wrapper for run_eval_graph_inject.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--openrouter_api_key_env",
        type=str,
        default="OPENROUTER_API_KEY",
        help="Environment variable holding the OpenRouter API key.",
    )
    parser.add_argument(
        "--openrouter_base_url",
        type=str,
        default=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        help="OpenRouter base URL.",
    )
    parser.add_argument(
        "--http_referer",
        type=str,
        default=os.getenv("OR_SITE_URL", ""),
        help="Optional HTTP-Referer header for OpenRouter analytics.",
    )
    parser.add_argument(
        "--x_title",
        type=str,
        default=os.getenv("OR_APP_NAME", "trail-benchmark"),
        help="Optional X-Title header for OpenRouter analytics.",
    )
    parser.add_argument(
        "--request_interval_sec",
        type=float,
        default=1.0,
        help="Delay between requests to avoid provider throttling.",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=5,
        help="Retry budget for rate-limit/transient failures.",
    )
    parser.add_argument(
        "--default_max_completion_tokens",
        type=int,
        default=15000,
        help=(
            "Applied when the base runner does not set max_completion_tokens. "
            "Keeps OpenRouter credit checks within budget."
        ),
    )
    parser.add_argument(
        "--pass1_reasoning_effort",
        type=str,
        choices=["none", "low", "medium", "high"],
        default="low",
        help=(
            "Reasoning effort for Pass-1 on reasoning models. "
            "Use 'low' for cheaper OpenRouter runs."
        ),
    )
    parser.add_argument(
        "--max_prompt_tokens",
        type=int,
        default=130000,
        help=(
            "Hard cap for prompt tokens before sending to model. "
            "If exceeded, trace content is truncated to fit."
        ),
    )
    parser.add_argument(
        "eval_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded verbatim to run_eval_graph_inject.py",
    )

    args = parser.parse_args()

    api_key = os.getenv(args.openrouter_api_key_env) or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OpenRouter API key missing. Set OPENROUTER_API_KEY (or override "
            "--openrouter_api_key_env)."
        )

    headers: Dict[str, str] = {}
    if args.http_referer:
        headers["HTTP-Referer"] = args.http_referer
    if args.x_title:
        headers["X-Title"] = args.x_title

    base._call_litellm = _build_router_call(
        api_base=args.openrouter_base_url,
        api_key=api_key,
        request_interval_sec=args.request_interval_sec,
        max_retries=args.max_retries,
        extra_headers=headers,
        default_max_completion_tokens=args.default_max_completion_tokens,
        pass1_reasoning_effort=args.pass1_reasoning_effort,
        max_prompt_tokens=args.max_prompt_tokens,
    )

    # Forward to the base runner's CLI parser.
    forwarded = args.eval_args
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]

    # Wrapper-level canary helper: allow --limit_traces in forwarded args even
    # though the base runner doesn't support it. We consume it here and build a
    # temporary reduced data_dir with only N traces.
    forwarded, limit_raw = _pop_forwarded_arg(forwarded, "--limit_traces")
    limit_traces = int(limit_raw) if limit_raw is not None else None
    if limit_traces is not None and limit_traces <= 0:
        raise ValueError("--limit_traces must be a positive integer")

    # Helpful guardrail: recommend explicit OpenRouter model prefix.
    model_idx = None
    for i, tok in enumerate(forwarded):
        if tok == "--model" and i + 1 < len(forwarded):
            model_idx = i + 1
            break
    if model_idx is not None:
        model_val = forwarded[model_idx]
        if not re.search(r"^(openrouter/|google/|gemini/)", model_val):
            print(
                "[warn] Model name may be invalid for OpenRouter. "
                "Expected something like 'openrouter/google/gemini-2.5-flash'."
            )

    if limit_traces is not None:
        split = _get_forwarded_arg(forwarded, "--split", "GAIA")
        data_dir = _get_forwarded_arg(forwarded, "--data_dir", "data")
        source_split_dir = _resolve_split_dir(data_dir, split)
        source_files = sorted(glob.glob(os.path.join(source_split_dir, "*.json")))
        selected = source_files[:limit_traces]
        print(
            f"[router-wrapper] canary mode: using {len(selected)} traces "
            f"(requested {limit_traces}) from {source_split_dir}"
        )
        with tempfile.TemporaryDirectory(prefix="router_canary_") as tmp_root:
            tmp_split_dir = os.path.join(tmp_root, split)
            os.makedirs(tmp_split_dir, exist_ok=True)
            for fp in selected:
                shutil.copy2(fp, os.path.join(tmp_split_dir, os.path.basename(fp)))
            forwarded = _set_forwarded_arg(forwarded, "--data_dir", tmp_root)
            litellm.drop_params = True
            sys.argv = [sys.argv[0], *forwarded]
            base.main()
        return

    litellm.drop_params = True
    sys.argv = [sys.argv[0], *forwarded]
    base.main()


if __name__ == "__main__":
    main()

