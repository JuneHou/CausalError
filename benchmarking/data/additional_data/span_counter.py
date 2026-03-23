#!/usr/bin/env python3
"""
Span Counter for TRAIL Benchmark

This script counts spans in OpenTelemetry trace JSON files.
It provides three counting modes:
1. Non-recursive: Counts main spans + immediate (level 1) child spans only
2. Recursive: Counts all spans including nested child/grandchild spans
3. Agent-steps: Counts agent runner spans (CodeAgent.run, ToolCallingAgent.run)
   plus CHAIN step spans (Step 1, Step 2, ...) at any depth — the level of
   abstraction that matches Table 5 for our nested multi-agent trace structure.

For the TRAIL paper (Table 5), we used NON-RECURSIVE counting.
However, traces in this dataset use a nested two-level agent structure where
the actual agent execution steps sit 3–5 levels deep:
  main → answer_single_question → CodeAgent.run → Step N → LLM/TOOL
The non-recursive method counts only the 2 level-1 children per trace (avg 3.0),
while the agent-steps method counts the meaningful execution steps (avg ~8.38),
which aligns with Table 5's reported value of 8.28 for the full 118-trace GAIA set.

Author: Varun Gangal (vgtomahawk@gmail.com)
"""

import os
import json
import argparse
from typing import Dict, List, Any


def count_spans_in_traces_withchildspans(input_dir: str, count_recursive_child_spans: bool = True) -> Dict[str, Any]:
    """
    Count spans (including child spans) in trace JSON files within a directory.

    Args:
        input_dir (str): Directory containing trace JSON files
        count_recursive_child_spans (bool): If True, count all nested children recursively.
                                           If False, count only level 1 children (default for Table 5).

    Returns:
        Dict with span count statistics including:
        - total_traces: Number of trace files processed
        - main_spans: Statistics for top-level spans
        - child_spans: Statistics for child spans (level 1 only if non-recursive)
        - total_spans: Combined statistics
    """
    main_span_counts = []
    child_span_counts = []
    total_spans_per_trace = []

    for filename in sorted(os.listdir(input_dir)):
        if filename.endswith('.json'):
            filepath = os.path.join(input_dir, filename)

            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    trace_data = json.load(f)

                # Count main (top-level) spans
                main_spans = trace_data.get('spans', [])
                main_span_count = len(main_spans)
                main_span_counts.append(main_span_count)

                # Count child spans
                def count_child_spans(spans: List[Dict], recursive: bool = count_recursive_child_spans) -> int:
                    """
                    Count child spans within a list of spans.

                    Args:
                        spans: List of span objects
                        recursive: If True, count nested children recursively

                    Returns:
                        Total count of child spans
                    """
                    total_child_spans = 0
                    for span in spans:
                        child_spans_list = span.get('child_spans', [])
                        # Count immediate children
                        total_child_spans += len(child_spans_list)

                        # Recursively count nested child spans if requested
                        if recursive and child_spans_list:
                            total_child_spans += count_child_spans(child_spans_list, recursive=True)

                    return total_child_spans

                child_span_count = count_child_spans(main_spans)
                child_span_counts.append(child_span_count)

                # Total spans per trace
                total_spans_per_trace.append(main_span_count + child_span_count)

            except json.JSONDecodeError:
                print(f"Error decoding JSON in {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    # Compute statistics
    if main_span_counts:
        return {
            'total_traces': len(main_span_counts),
            'main_spans': {
                'min': min(main_span_counts),
                'max': max(main_span_counts),
                'mean': sum(main_span_counts) / len(main_span_counts),
                'total': sum(main_span_counts)
            },
            'child_spans': {
                'min': min(child_span_counts) if child_span_counts else 0,
                'max': max(child_span_counts) if child_span_counts else 0,
                'mean': sum(child_span_counts) / len(child_span_counts) if child_span_counts else 0,
                'total': sum(child_span_counts)
            },
            'total_spans': {
                'min': min(total_spans_per_trace),
                'max': max(total_spans_per_trace),
                'mean': sum(total_spans_per_trace) / len(total_spans_per_trace),
                'total': sum(total_spans_per_trace)
            }
        }

    return {
        'total_traces': 0,
        'main_spans': {'min': 0, 'max': 0, 'mean': 0, 'total': 0},
        'child_spans': {'min': 0, 'max': 0, 'mean': 0, 'total': 0},
        'total_spans': {'min': 0, 'max': 0, 'mean': 0, 'total': 0}
    }


def count_agent_step_spans(input_dir: str) -> Dict[str, Any]:
    """
    Count agent-step spans: CodeAgent.run + ToolCallingAgent.run + "Step N" spans.

    These span types together represent the actual execution steps in the nested
    multi-agent trace structure used by this dataset:
      - CodeAgent.run / ToolCallingAgent.run: agent runner spans that contain steps
      - Step N (Step 1, Step 2, ...): individual reasoning + tool-use steps

    This counting method produces numbers close to Table 5's "Total Spans" figure
    for our 117-trace GAIA subset (980 vs paper's 977 for 118 traces).

    Args:
        input_dir: Directory containing trace JSON files.

    Returns:
        Dict with total_traces, agent_runner_spans, step_spans, total_spans stats.
    """
    def flatten_all(spans: List[Dict]) -> List[Dict]:
        result = []
        for s in spans:
            result.append(s)
            result.extend(flatten_all(s.get('child_spans', [])))
        return result

    runner_names = {'CodeAgent.run', 'ToolCallingAgent.run'}

    runner_counts = []
    step_counts   = []
    total_counts  = []

    for filename in sorted(os.listdir(input_dir)):
        if not filename.endswith('.json'):
            continue
        filepath = os.path.join(input_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                trace_data = json.load(f)
            all_spans = flatten_all(trace_data.get('spans', []))
            n_runners = sum(1 for s in all_spans if s.get('span_name', '') in runner_names)
            n_steps   = sum(1 for s in all_spans
                            if s.get('span_name', '').startswith('Step ')
                            and s.get('span_name', '').split(' ', 1)[-1].isdigit())
            runner_counts.append(n_runners)
            step_counts.append(n_steps)
            total_counts.append(n_runners + n_steps)
        except json.JSONDecodeError:
            print(f"Error decoding JSON in {filename}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    if not total_counts:
        zero = {'min': 0, 'max': 0, 'mean': 0.0, 'total': 0}
        return {'total_traces': 0, 'agent_runner_spans': zero,
                'step_spans': zero, 'total_spans': zero}

    def stats(vals):
        return {'min': min(vals), 'max': max(vals),
                'mean': sum(vals) / len(vals), 'total': sum(vals)}

    return {
        'total_traces': len(total_counts),
        'agent_runner_spans': stats(runner_counts),
        'step_spans':         stats(step_counts),
        'total_spans':        stats(total_counts),
    }


def print_agent_step_statistics(stats: Dict[str, Any], dataset_name: str) -> None:
    """Print agent-step span statistics."""
    print(f"\n{'='*60}")
    print(f"Span Statistics for {dataset_name}")
    print(f"Counting Mode: AGENT-STEPS (CodeAgent.run + ToolCallingAgent.run + Step N)")
    print(f"{'='*60}")
    print(f"Total Traces: {stats['total_traces']}")
    print(f"\nAgent Runner Spans (CodeAgent.run / ToolCallingAgent.run):")
    r = stats['agent_runner_spans']
    print(f"  Min: {r['min']}  Max: {r['max']}  Mean: {r['mean']:.2f}  Total: {r['total']}")
    print(f"\nStep Spans (Step 1, Step 2, ...):")
    s = stats['step_spans']
    print(f"  Min: {s['min']}  Max: {s['max']}  Mean: {s['mean']:.2f}  Total: {s['total']}")
    print(f"\nTotal Agent-Step Spans (Runner + Steps):")
    t = stats['total_spans']
    print(f"  Min: {t['min']}  Max: {t['max']}  Mean: {t['mean']:.2f}  Total: {t['total']}")
    print(f"\n  Table 5 target (GAIA 118 traces): Total=977  Mean=8.28")
    diff = t['total'] - 977
    sign = '+' if diff >= 0 else ''
    print(f"  Difference from Table 5:          {sign}{diff}  ({sign}{t['mean'] - 8.28:.2f} avg)")
    print(f"{'='*60}\n")


def print_statistics(stats: Dict[str, Any], dataset_name: str, recursive: bool):
    """Print span statistics in a readable format."""
    counting_mode = "RECURSIVE" if recursive else "NON-RECURSIVE (Table 5 method)"

    print(f"\n{'='*60}")
    print(f"Span Statistics for {dataset_name}")
    print(f"Counting Mode: {counting_mode}")
    print(f"{'='*60}")
    print(f"Total Traces: {stats['total_traces']}")
    print(f"\nMain Spans (Top-level):")
    print(f"  Min: {stats['main_spans']['min']}")
    print(f"  Max: {stats['main_spans']['max']}")
    print(f"  Mean: {stats['main_spans']['mean']:.2f}")
    print(f"  Total: {stats['main_spans']['total']}")
    print(f"\nChild Spans:")
    print(f"  Min: {stats['child_spans']['min']}")
    print(f"  Max: {stats['child_spans']['max']}")
    print(f"  Mean: {stats['child_spans']['mean']:.2f}")
    print(f"  Total: {stats['child_spans']['total']}")
    print(f"\nTotal Spans (Main + Children):")
    print(f"  Min: {stats['total_spans']['min']}")
    print(f"  Max: {stats['total_spans']['max']}")
    print(f"  Mean: {stats['total_spans']['mean']:.2f}")
    print(f"  Total: {stats['total_spans']['total']}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Count spans in OpenTelemetry trace files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Count spans non-recursively (as used in TRAIL paper Table 5)
  python span_counter.py --input-dir ./traces/gaia --non-recursive

  # Count spans recursively (includes all nested children)
  python span_counter.py --input-dir ./traces/gaia --recursive

  # Count agent-step spans (CodeAgent.run + ToolCallingAgent.run + Step N)
  python span_counter.py --input-dir ./traces/gaia --agent-steps

  # Compare all three methods
  python span_counter.py --input-dir ./traces/gaia --compare
        """
    )

    parser.add_argument(
        '--input-dir',
        required=True,
        help='Directory containing trace JSON files'
    )

    parser.add_argument(
        '--recursive',
        action='store_true',
        help='Count all nested child spans recursively'
    )

    parser.add_argument(
        '--non-recursive',
        action='store_true',
        help='Count only level 1 child spans (original Table 5 method)'
    )

    parser.add_argument(
        '--agent-steps',
        action='store_true',
        help='Count agent runner + Step N spans (matches Table 5 for nested multi-agent traces)'
    )

    parser.add_argument(
        '--compare',
        action='store_true',
        help='Show all three counting methods side by side'
    )

    parser.add_argument(
        '--dataset-name',
        default='Dataset',
        help='Name of the dataset for display purposes'
    )

    args = parser.parse_args()

    # Default to non-recursive if nothing specified
    if not args.recursive and not args.agent_steps and not args.compare:
        args.non_recursive = True

    # Verify directory exists
    if not os.path.exists(args.input_dir):
        print(f"Error: Directory '{args.input_dir}' does not exist")
        return 1

    # Count and display statistics
    if args.compare:
        print("\n" + "="*60)
        print("COMPARISON: All Three Counting Methods")
        print("="*60)

        non_recursive_stats = count_spans_in_traces_withchildspans(
            args.input_dir, count_recursive_child_spans=False)
        print_statistics(non_recursive_stats, args.dataset_name, recursive=False)

        recursive_stats = count_spans_in_traces_withchildspans(
            args.input_dir, count_recursive_child_spans=True)
        print_statistics(recursive_stats, args.dataset_name, recursive=True)

        agent_step_stats = count_agent_step_spans(args.input_dir)
        print_agent_step_statistics(agent_step_stats, args.dataset_name)

        nr  = non_recursive_stats['total_spans']['total']
        rec = recursive_stats['total_spans']['total']
        ast = agent_step_stats['total_spans']['total']
        print("Summary:")
        print(f"  Non-recursive total:  {nr:5d}  avg {non_recursive_stats['total_spans']['mean']:.2f}")
        print(f"  Recursive total:      {rec:5d}  avg {recursive_stats['total_spans']['mean']:.2f}")
        print(f"  Agent-steps total:    {ast:5d}  avg {agent_step_stats['total_spans']['mean']:.2f}")
        print(f"  Table 5 target:        977   avg 8.28  (118 traces)")
        print()

    elif args.agent_steps:
        stats = count_agent_step_spans(args.input_dir)
        print_agent_step_statistics(stats, args.dataset_name)

    else:
        recursive_mode = args.recursive
        stats = count_spans_in_traces_withchildspans(
            args.input_dir, count_recursive_child_spans=recursive_mode)
        print_statistics(stats, args.dataset_name, recursive=recursive_mode)

    return 0


if __name__ == '__main__':
    exit(main())
