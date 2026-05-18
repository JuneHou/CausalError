#!/usr/bin/env python3
"""
Build concise protocol/QC sentences for paper text from manifests.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


def load_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=Path("/data/wang/junh/githubs/trail-benchmark/analysis/workstream_c_error_taxonomy/results"),
    )
    args = parser.parse_args()

    r = args.results_dir
    sampling = load_json(r / "trail_sampling_manifest.json")
    pilot = load_json(r / "trail_gpt5_pilot_run_manifest.json")
    main_run = load_json(r / "trail_gpt5_main_run_manifest.json")
    qc = load_json(r / "trail_qc_agreement_manifest.json")
    adj = load_json(r / "trail_adjudication_manifest.json")
    bucket_rows = load_csv(r / "trail_mechanism_bucket_summary.csv")

    n_total = sampling.get("selected_counts", {}).get("n_selected_enriched", 0)
    split_counts = sampling.get("selected_counts", {}).get("by_split", {})
    qc_n = qc.get("n_qc_common", 0)
    mean_kappa = qc.get("mean_kappa", 0.0)

    top_bucket = bucket_rows[0]["bucket"] if bucket_rows else "N/A"
    top_bucket_pct = bucket_rows[0]["percent"] if bucket_rows else "N/A"

    protocol_sentence = (
        f"We audited {n_total} TRAIL instances (GAIA={split_counts.get('GAIA_dedup',0)}, SWE={split_counts.get('SWE_Bench_dedup',0)}) "
        f"under a frozen two-layer schema: benchmark taxonomy labels plus mechanism-level labels (5 buckets, 12 pattern tags). "
        f"Labeling used GPT-5.2 with a pilot ({pilot.get('rows_labeled',0)} rows) before full-pass annotation ({main_run.get('rows_labeled',0)} rows), followed by adjudication."
    )
    qc_sentence = (
        f"For quality control, we manually re-labeled {qc_n} sampled instances and compared against GPT labels "
        f"(mean Cohen's kappa={mean_kappa:.3f}); adjudicated labels were used for final analysis. "
        f"The largest mechanism bucket in the final set was {top_bucket} ({top_bucket_pct}%)."
    )

    payload = {
        "protocol_sentence": protocol_sentence,
        "qc_sentence": qc_sentence,
        "inputs_present": {
            "sampling_manifest": bool(sampling),
            "pilot_manifest": bool(pilot),
            "main_manifest": bool(main_run),
            "qc_manifest": bool(qc),
            "adjudication_manifest": bool(adj),
            "bucket_summary": bool(bucket_rows),
        },
    }
    with (r / "trail_protocol_summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    with (r / "trail_protocol_summary.md").open("w", encoding="utf-8") as f:
        f.write("## Workstream C protocol sentence\n")
        f.write(payload["protocol_sentence"] + "\n\n")
        f.write("## Workstream C QC sentence\n")
        f.write(payload["qc_sentence"] + "\n")

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
