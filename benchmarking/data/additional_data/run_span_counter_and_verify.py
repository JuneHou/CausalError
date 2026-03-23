#!/usr/bin/env python3
"""
Run the author's span_counter.py on the HuggingFace-downloaded GAIA dataset.
No additional output: invokes span_counter.py so only the author's script prints.
"""

import os
import sys
import argparse
import subprocess

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_BENCHMARKING_DIR = os.path.dirname(_THIS_DIR)


def main():
    parser = argparse.ArgumentParser(
        description="Run author's span_counter.py on HF GAIA (no extra output)"
    )
    parser.add_argument(
        "--input-dir",
        default="/data/wang/junh/githubs/trail-benchmark/benchmarking/data/GAIA",
        help="Directory of trace JSON files (default: data/GAIA)",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Use sample_trace/ (6 traces) instead of full GAIA",
    )
    parser.add_argument(
        "--dataset-name",
        default="GAIA",
        help="Label for span_counter output (default: GAIA)",
    )
    args = parser.parse_args()

    if args.sample:
        args.input_dir = "/data/wang/junh/githubs/trail-benchmark/benchmarking/data/additional_data/sample_trace"
        args.dataset_name = "Sample (6 traces)"

    if not os.path.isdir(args.input_dir):
        print(f"Error: Directory '{args.input_dir}' does not exist")
        return 1

    script = os.path.join(_THIS_DIR, "span_counter.py")
    cmd = [
        sys.executable,
        script,
        "--input-dir", args.input_dir,
        "--compare",
        "--dataset-name", args.dataset_name,
    ]
    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    sys.exit(main())
