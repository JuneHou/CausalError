# Sample GAIA Traces for Span Counting Verification

This directory contains 6 sample traces from the GAIA validation split used in the TRAIL benchmark paper.

## Trace Format

These are OpenTelemetry traces in JSON format with the following structure:

```json
{
  "trace_id": "...",
  "spans": [
    {
      "span_id": "...",
      "span_name": "...",
      "parent_span_id": null,
      "child_spans": [
        {
          "span_id": "...",
          "span_name": "...",
          "parent_span_id": "...",
          "child_spans": [...]
        }
      ],
      ...
    }
  ]
}
```

## Files

- `0035f455b3ff2295167a844f04d85d34.json` - Simple trace (fewer nested levels)
- `0140b3f657eddf76ca82f72c49ac8e58.json` - Complex trace (multiple nested levels)
- `01c5727165fc43899b3b594b9bef5f19.json` - Complex trace (multiple nested levels)
- `0242ca2533fac5b8b604a9060b3e15d6.json` - Medium complexity
- `041b7f9c8c76c2ca1a8e67c6769267c3.json` - Medium complexity
- `4ae16319f0de44a7d1e84595b41ae08d.json` - Example trace (used in documentation)

## Usage

**Author's script (`span_counter.py`)** — use as provided, with any directory of trace JSONs:

```bash
# From benchmarking/data/additional_data/
# Sample (6 traces in this repo)
python span_counter.py --input-dir sample_trace --non-recursive
python span_counter.py --input-dir sample_trace --compare --dataset-name "Sample"

# HuggingFace-downloaded full GAIA (benchmarking/data/GAIA)
python span_counter.py --input-dir ../GAIA --compare --dataset-name "GAIA"
```

**Runner with verification** — same counts using the author's logic, then compare to `span_counting_diagram.txt`:

```bash
# Full GAIA (default: ../GAIA), with verification vs expected_span_counts.json
python run_span_counter_and_verify.py

# Sample 6 traces only
python run_span_counter_and_verify.py --sample

# Custom trace dir
python run_span_counter_and_verify.py --input-dir /path/to/gaia/traces --no-verify
```

## Expected Results

For these 6 sample traces:

| Method | Total Spans | Avg per Trace |
|--------|-------------|---------------|
| Non-Recursive (Table 5) | 18 | 3.00 |
| Recursive (All nested) | 164 | 27.33 |

The recursive method yields **9.11x more spans** than the non-recursive method for this sample!

This dramatic difference (164 vs 18 spans) illustrates why you might be seeing "3K+ spans" when expecting much fewer.

## Notes

- These traces are from agent runs on the GAIA benchmark
- The hierarchical structure reflects nested function calls in agentic frameworks
- Deeper nesting levels often represent framework internals rather than semantic agent steps

## Author's script only

- **`span_counter.py`** is the author's (Varun) script unchanged. Do not edit it.
- **`run_span_counter_and_verify.py`** calls the author's counting logic (imports from `span_counter`) and adds verification against `expected_span_counts.json` (from `span_counting_diagram.txt`). Use this to run on the HuggingFace-downloaded GAIA dir and check Table 5 method.
- Sample (6 traces) verification should **PASS**. Full GAIA verification may **FAIL** if your HF download has a different trace set (e.g. 117 vs 118 traces) or different span structure than the author's reference.
