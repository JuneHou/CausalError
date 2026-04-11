"""
compress_traces.py — Remove noise keys from TRAIL trace JSONs.

Drops all OpenTelemetry/Patronus metadata fields that carry zero error-detection
signal (verified against all 20 TRAIL error taxonomy categories in
paper/trace_key_analysis.md). Conversation history (llm.input_messages.*) and
LLM responses (llm.output_messages.*) are preserved in full.

Special handling for llm.tools.N.json_schema:
  Tool schemas are identical across all LLM spans in a trace (tools don't change
  mid-run). We extract the unique set once and attach it at the trace root as
  "_tool_schemas", then drop it from every individual LLM span. This eliminates
  the largest single source of redundancy while keeping the information for
  Tool Definition Issues detection.

Special handling for input.value on LLM spans:
  For LLM spans, input.value is a JSON-string re-encoding of llm.input_messages.*,
  which is already present in structured form. Dropping input.value from LLM spans
  eliminates the largest remaining redundancy (~43 MB across GAIA, ~39% of total)
  without losing any information.

Optional --dedup mode (llm.input_messages deduplication):
  Each LLM call in a trace receives the full accumulated conversation history.
  Consecutive execution-step LLM spans share a growing prefix of messages (step N
  contains all of step N-1's messages plus 2 new ones). With --dedup, spans after
  the first share only their delta (new messages). The number of omitted prefix
  messages is recorded as _messages_prefix_count in span_attributes so the full
  conversation can always be reconstructed. Unrelated LLM calls (e.g. planning vs
  execution, or sub-agent calls) have no common prefix and are left unchanged.
  Output goes to <split>_dedup by default.

Usage:
    python benchmarking/compress_traces.py --split GAIA
    python benchmarking/compress_traces.py --split GAIA --dedup
    python benchmarking/compress_traces.py --split "SWE Bench"
    python benchmarking/compress_traces.py --split GAIA --input_dir path/to/data --output_dir path/to/out
    python benchmarking/compress_traces.py --split GAIA --dry_run   # stats only, no files written
"""

import json
import os
import glob
import argparse
from pathlib import Path

# ---------------------------------------------------------------------------
# Drop lists (justified in paper/trace_key_analysis.md)
# ---------------------------------------------------------------------------

# Top-level span fields to drop
TOP_LEVEL_DROP = {
    "trace_id",          # redundant in every span; kept at root {"trace_id": ..., "spans": [...]}
    "trace_state",       # OTel propagation header, no signal
    "span_kind",         # OTel kind (INTERNAL/CLIENT); redundant with openinference.span.kind
    "service_name",      # deployment metadata
    "resource_attributes",  # OTel SDK/runtime metadata
    "scope_name",        # OTel instrumentation scope
    "scope_version",     # OTel instrumentation version
    "links",             # OTel span links, empty in TRAIL
    "logs",              # function-call logs on UNKNOWN (Patronus wrapper) spans only; no agent-behavior signal
}

# span_attributes keys to drop (exact match)
ATTR_DROP_EXACT = {
    "input.mime_type",          # always "text/plain"/"application/json", no signal
    "output.mime_type",         # same
    "llm.invocation_parameters",# temperature/max_tokens config, not error signal
    "llm.model_name",           # model identifier, not error signal
    "pat.app",                  # Patronus app tag
    "pat.project.id",           # Patronus project UUID
    "pat.project.name",         # Patronus project name
    "smolagents.tools_names",   # subset of info already in tool spans
    "smolagents.max_steps",     # config param, ResAb evident from error pattern
}

# span_attributes key prefixes to drop
ATTR_DROP_PREFIX = (
    "llm.token_count.",         # completion/prompt/total counts, no error signal
)

# span_attributes key prefix to extract-once instead of per-span
ATTR_EXTRACT_ONCE_PREFIX = "llm.tools."   # tool JSON schemas, repeated every LLM span


def _should_drop_attr(key: str) -> bool:
    if key in ATTR_DROP_EXACT:
        return True
    for prefix in ATTR_DROP_PREFIX:
        if key.startswith(prefix):
            return True
    return False


def _is_extract_once(key: str) -> bool:
    return key.startswith(ATTR_EXTRACT_ONCE_PREFIX)


# ---------------------------------------------------------------------------
# llm.input_messages deduplication helpers
# ---------------------------------------------------------------------------

def _extract_messages(attrs: dict) -> list:
    """Parse llm.input_messages.N.message.{role,content} into an ordered list of dicts."""
    msgs: dict = {}
    for k, v in attrs.items():
        if not k.startswith("llm.input_messages."):
            continue
        parts = k.split(".")
        # expected: llm . input_messages . N . message . field
        if len(parts) < 5:
            continue
        try:
            idx = int(parts[2])
        except ValueError:
            continue
        field = parts[4]
        if idx not in msgs:
            msgs[idx] = {}
        msgs[idx][field] = v
    return [msgs[i] for i in sorted(msgs.keys())]


def _common_prefix_len(a: list, b: list) -> int:
    """Return the length of the longest common prefix between two message lists."""
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


# ---------------------------------------------------------------------------
# Core compression
# ---------------------------------------------------------------------------

def compress_span(span: dict, tool_schemas: dict, prev_llm_msgs: list) -> dict:
    """Return a compressed copy of span.

    tool_schemas  — mutable dict, accumulates hoisted tool JSON schemas.
    prev_llm_msgs — mutable list (length-1 wrapper) holding the message list
                    from the last LLM span seen in DFS order. Used for dedup.
                    Pass an empty list [] to disable deduplication.
    """
    out = {}

    for k, v in span.items():
        if k in TOP_LEVEL_DROP:
            continue
        if k == "span_attributes":
            continue   # handled below
        if k == "child_spans":
            continue   # handled recursively
        out[k] = v

    # Process span_attributes
    attrs = span.get("span_attributes", {})
    span_kind = attrs.get("openinference.span.kind", "")
    new_attrs = {}
    for k, v in attrs.items():
        if _should_drop_attr(k):
            continue
        if _is_extract_once(k):
            tool_schemas[k] = v   # hoist to trace root
            continue
        # input.value on LLM spans is a JSON-string re-encoding of llm.input_messages.*
        # (already present in structured form) — drop to eliminate the largest redundancy.
        if k == "input.value" and span_kind == "LLM":
            continue
        new_attrs[k] = v

    # Deduplicate llm.input_messages for LLM spans when dedup is enabled.
    # prev_llm_msgs is used as a mutable length-1 list: [] = disabled, [[...]] = last msgs.
    if span_kind == "LLM" and prev_llm_msgs:
        curr_msgs = _extract_messages(attrs)
        prefix_len = _common_prefix_len(prev_llm_msgs[0], curr_msgs)
        if prefix_len > 0:
            # Drop the shared-prefix keys from new_attrs.
            for k in list(new_attrs.keys()):
                if k.startswith("llm.input_messages."):
                    parts = k.split(".")
                    if len(parts) >= 3:
                        try:
                            if int(parts[2]) < prefix_len:
                                del new_attrs[k]
                        except ValueError:
                            pass
            # Record how many messages were omitted so reconstruction is possible.
            new_attrs["_messages_prefix_count"] = prefix_len
        # Update the shared state for the next LLM span in DFS order.
        prev_llm_msgs[0] = curr_msgs
    elif span_kind == "LLM" and prev_llm_msgs is not None and len(prev_llm_msgs) == 0:
        # dedup enabled but this is the first LLM span — seed the state
        curr_msgs = _extract_messages(attrs)
        prev_llm_msgs.append(curr_msgs)

    if new_attrs:
        out["span_attributes"] = new_attrs

    # Recurse into children (prev_llm_msgs is mutable, so DFS order is preserved).
    children = span.get("child_spans", [])
    if children:
        out["child_spans"] = [compress_span(c, tool_schemas, prev_llm_msgs) for c in children]

    return out


def compress_trace(trace_data: dict, dedup: bool = False) -> dict:
    """Return compressed trace. Tool schemas hoisted to root as _tool_schemas.

    dedup=True enables llm.input_messages prefix deduplication across LLM spans.
    """
    tool_schemas: dict = {}
    # prev_llm_msgs: [] means dedup disabled; starts empty and is seeded on first LLM span.
    prev_llm_msgs: list = [] if dedup else None
    compressed_spans = [
        compress_span(span, tool_schemas, prev_llm_msgs) for span in trace_data.get("spans", [])
    ]
    out = {
        "trace_id": trace_data["trace_id"],
        "spans": compressed_spans,
    }
    if tool_schemas:
        out["_tool_schemas"] = tool_schemas
    return out


# ---------------------------------------------------------------------------
# Stats helper
# ---------------------------------------------------------------------------

def byte_size(obj: dict) -> int:
    return len(json.dumps(obj, separators=(",", ":")))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compress TRAIL trace JSONs")
    parser.add_argument("--split", type=str, default="GAIA",
                        help="Dataset split name (GAIA or 'SWE Bench')")
    parser.add_argument("--input_dir", type=str, default=None,
                        help="Input directory (default: benchmarking/data/<split>)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: benchmarking/data/<split>_compressed or <split>_dedup with --dedup)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print stats only, do not write files")
    parser.add_argument("--dedup", action="store_true",
                        help="Also deduplicate llm.input_messages prefix across consecutive LLM spans. "
                             "Records omitted count in _messages_prefix_count. Output goes to <split>_dedup.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    bench_dir = repo_root / "benchmarking"

    input_dir = Path(args.input_dir) if args.input_dir else bench_dir / "data" / args.split
    default_suffix = "dedup" if args.dedup else "compressed"
    output_dir = Path(args.output_dir) if args.output_dir else bench_dir / "data" / f"{args.split}_{default_suffix}"

    files = sorted(glob.glob(str(input_dir / "*.json")))
    if not files:
        print(f"No JSON files found in {input_dir}")
        return

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    total_orig = 0
    total_comp = 0
    n_overflow_before = 0   # >500KB proxy for context overflow
    n_overflow_after  = 0

    print(f"Processing {len(files)} traces from {input_dir}")
    print(f"Output dir : {output_dir}")
    print()

    for path in files:
        with open(path) as f:
            trace_data = json.load(f)

        orig_bytes = os.path.getsize(path)
        compressed = compress_trace(trace_data, dedup=args.dedup)
        comp_bytes = byte_size(compressed)

        total_orig += orig_bytes
        total_comp += comp_bytes
        if orig_bytes > 500_000:
            n_overflow_before += 1
        if comp_bytes > 500_000:
            n_overflow_after += 1

        if not args.dry_run:
            out_path = output_dir / Path(path).name
            with open(out_path, "w") as f:
                json.dump(compressed, f, separators=(",", ":"))

    # Summary
    reduction = (1 - total_comp / total_orig) * 100
    print(f"{'Metric':<35} {'Before':>12} {'After':>12} {'Δ':>10}")
    print("-" * 72)
    print(f"{'Total size (MB)':<35} {total_orig/1e6:>12.2f} {total_comp/1e6:>12.2f} {(total_comp-total_orig)/1e6:>+10.2f}")
    print(f"{'Size reduction':<35} {'':>12} {'':>12} {-reduction:>+9.1f}%")
    print(f"{'Traces >500KB (proxy overflow)':<35} {n_overflow_before:>12} {n_overflow_after:>12} {n_overflow_after-n_overflow_before:>+10}")
    print(f"{'Avg size per trace (KB)':<35} {total_orig/len(files)/1e3:>12.1f} {total_comp/len(files)/1e3:>12.1f}")
    if not args.dry_run:
        print(f"\nWrote {len(files)} compressed traces to {output_dir}")
    else:
        print("\n[dry_run] No files written.")


if __name__ == "__main__":
    main()
