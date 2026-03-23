# Quick Start Guide - Verifying TRAIL Table 5 Span Counts

## What You Received

1. **`span_counter.py`** - Python script to count spans
2. **`sample_gaia_traces_for_jun/`** - 6 example GAIA traces
3. **`SPAN_COUNTING_EXPLANATION.md`** - Detailed methodology documentation

## Step 1: Verify the Script Works

Run the script on the sample data with the `--compare` flag to see both counting methods:

```bash
python span_counter.py --input-dir sample_gaia_traces_for_jun --compare --dataset-name "Sample"
```

**Expected Output:**
- Non-Recursive (Table 5): 18 total spans across 6 traces
- Recursive (All nested): 164 total spans across 6 traces
- Difference: 9.11x more spans with recursive counting!

## Step 2: Run on Your GAIA Data

Now run the same script on your full GAIA trace directory:

```bash
# First, try both methods to compare
python span_counter.py --input-dir /path/to/your/gaia/traces --compare --dataset-name "GAIA"
```

If you're seeing ~3K-4K spans total, you'll likely see:
- **Non-recursive**: ~1,000 spans (matches Table 5)
- **Recursive**: ~4,000-5,000 spans (what you're currently getting)

## Step 3: Understanding Your Trace Format

The script expects traces in this JSON format:

```json
{
  "trace_id": "some-id",
  "spans": [
    {
      "span_id": "...",
      "span_name": "...",
      "child_spans": [
        {
          "span_id": "...",
          "child_spans": [...]
        }
      ]
    }
  ]
}
```

If your traces have a different structure, you may need to adapt the script. The key parts are:
- Line 48: `main_spans = trace_data.get('spans', [])`
- Line 65: `child_spans_list = span.get('child_spans', [])`

## Common Issues

### "Getting fewer spans than expected"
- Make sure you're loading the original trace files, not annotation files
- Verify the JSON structure matches what the script expects

### "Getting way more spans than expected"
- You're likely counting recursively (all nested levels)
- Use `--non-recursive` flag or check that recursive=False in the code

### "Script won't run"
- Requires Python 3.7+
- Only uses standard library (json, os, argparse, typing)
- No pip install needed!

## For Your Paper/Documentation

To cite the methodology:

> Span counts in Table 5 were computed using non-recursive counting, which counts only top-level spans and their immediate (level 1) child spans. Deeply nested child spans (level 2+) representing framework internals were excluded to focus on semantic agent actions rather than implementation details.

## Questions?

If the counts still don't match, please share:
1. What counting method you were using (recursive vs non-recursive)
2. Your total span count
3. Number of traces you're analyzing
4. (Optional) A sample trace file for debugging

The TRAIL team can help debug any discrepancies!
