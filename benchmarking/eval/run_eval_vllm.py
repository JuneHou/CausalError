"""
eval/run_eval_vllm.py — zero-shot LLM-as-judge evaluation using a local HuggingFace
model via vLLM, with the exact same prompt and output format as run_eval.py.

Usage (from benchmarking/):
    CUDA_VISIBLE_DEVICES=0,3,4,7 python eval/run_eval_vllm.py --split GAIA
    CUDA_VISIBLE_DEVICES=0,3,4,7 python eval/run_eval_vllm.py --split GAIA \
        --model Tongyi-Zhiwen/QwenLong-L1-32B --tensor_parallel_size 4

Outputs are saved to:
    outputs/zero_shot/outputs_{model_tag}-{split}/
and can be scored with the standard calculate_scores.py.
"""

import os
import sys
import glob
import json
import re
import argparse
from pathlib import Path
from tqdm import tqdm

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

BENCH_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BENCH_DIR))
from span_level_parser import parse_trace_to_step_level, _span_name


def build_span_index(trace_str: str) -> str:
    """Build a compact span index using only agent-step spans (CodeAgent.run,
    ToolCallingAgent.run, Step N) plus their direct LLM/TOOL children — the spans
    where TRAIL annotations actually live. Avg ~8.38 step spans per GAIA trace.
    Does NOT recursively enumerate all spans (which would give ~30 irrelevant entries).
    """
    try:
        trace_data = json.loads(trace_str)
    except Exception:
        return ""

    parsed = parse_trace_to_step_level(trace_data)
    span_by_id = parsed.get("span_by_id", {})
    step_spans = parsed.get("step_spans", [])

    if not step_spans:
        return ""

    lines = ["Span index for this trace (use these exact span_id hex values for the location field):"]
    seen = set()
    for entry in step_spans:
        span = entry["span"]
        sid = span.get("span_id")
        sname = _span_name(span)
        if sid and sid not in seen:
            seen.add(sid)
            lines.append(f'  span_id "{sid}"  ({sname})')
        # Also include direct children (LiteLLMModel.__call__, TOOL spans) — annotation targets
        for child in span.get("child_spans") or []:
            csid = child.get("span_id")
            csname = _span_name(child)
            if csid and csid not in seen:
                seen.add(csid)
                lines.append(f'    span_id "{csid}"  ({csname})')
    return "\n".join(lines)


def get_prompt(trace: str, span_index: str = "") -> str:
    prompt = """Follow the taxonomy below carefully follow the instructions and provide the output in the same format as the example.

# Taxonomy
├── Reasoning Errors
│   ├── Hallucinations
│   │   ├── Language-only
│   │   └── Tool-related (fabricating tool outputs/capabilities)
│   ├── Information Processing
│   │   ├── Poor Information Retrieval (Tried to find information that was not relevant to the task)
│   │   └── Tool Output Misinterpretation (Made assumptions about the tool output or used the tool output in an incorrect context)
│   ├── Decision Making
│   │   ├── Incorrect Problem Identification (Misunderstood the overall task or the local task)
│   │   ├── Tool Selection Errors (Used the wrong tool for the task)
│   └── Output Generation
│       ├── Formatting Errors (Errors with formatting and execution of code or structuring of output in a specific format)
│       └── Instruction Non-compliance (Failed to perform the task provided and instead did something else)
├── System Execution Errors
│   ├── Configuration
│   │   ├── Tool Definition Issues (The tool was not defined correctly by the user or contains some errors that make it inconsistent with its description. For example, web search tool was defined as a calculator tool)
│   │   └── Environment Setup Errors (includes permission problems and inability to access resources or API keys)
│   ├── API Issues
│   │   ├── Rate Limiting (Like 429)
│   │   ├── Authentication Errors (Like 401/403)
│   │   ├── Service Errors (Like 500)
│   │   └── Resource Not Found (Like 404)
│   └── Resource Management
│       ├── Resource Exhaustion (includes memory overflow)
│       └── Timeout Issues (The system took too long to respond)
├── Planning and Coordination Errors
│    ├── Context Management
│    │   ├── Context Handling Failures (includes window overflow and state tracking or forgetting important context)
│    │   └── Resource Abuse (Called the tool excessively due to memory issues)
│    └── Task Management
│        ├── Goal Deviation (The system deviated from the task or the subtask)
│        └── Task Orchestration (includes subtask coordination between agents and progress monitoring)

- Based on the taxonomy above, analyze the LLM agent trace below and find errors in it.
- You must be exhaustive and find all the errors in the trace. Only include the final subcategories of the taxonomy (i.e. "Resource Not Found" and not "API Issues" or "System Execution Errors").
- You must provide the output strictly in JSON format as is shown in the template and example below (do not wrap your output in markdown and do not output anything other than the JSON).

Template for output:

{{
    "errors": [
        {{
            "category": "[INSERT ERROR CATEGORY FROM TAXONOMY HERE]", # The category of the error
            "location": "[INSERT THE span_id VALUE HERE]", # Must be the exact span_id hex string from the trace JSON (e.g. '77fb7128d6f04862'), NOT the span name
            "evidence": "[INSERT EXTRACTED EVIDENCE HERE]",
            "description": "[INSERT DETAILED ERROR DESCRIPTION HERE]",
            "impact": "[INSERT IMPACT HERE]" # The impact of the error (HIGH, MEDIUM, LOW)
        }},
        ... # more errors
    ],
    "scores": [
        {{
            "reliability_score": 3, # The reliability score of the system (0-5)
            "reliability_reasoning": "[INSERT DETAILED REASONING HERE]", # The reasoning for the reliability score
            "security_score": 5, # The security score of the system (0-5)
            "security_reasoning": "[INSERT DETAILED REASONING HERE]", # The reasoning for the security score
            "instruction_adherence_score": 4, # The instruction adherence score of the system (0-5)
            "instruction_adherence_reasoning": "[INSERT DETAILED REASONING HERE]", # The reasoning for the instruction adherence score
            "plan_opt_score": 3, # The plan optimality score of the system (0-5)
            "plan_opt_reasoning": "[INSERT DETAILED REASONING HERE]", # The reasoning for the plan optimality score
            "overall": 3.75 # The overall score of the system (0-5)
        }}
    ]
}}

Example output:

{{
    "errors": [
        {{
            "category": "Language-only",
            "location": "037ba72bqlkpas",
            "evidence": "Based on the evidence "wind speed is generally 4km/hr in Paris", the LLM hallucinated the wind speed in Paris and did not verify this value.",
            "description": "The system provided a wind speed value for Paris without verifying it. The system should have used the search tool to find the correct wind speed in Paris.",
            "impact": "HIGH"
        }},
    ],
    "scores": [
        {{
            "reliability_score": 1,
            "reliability_reasoning": "The system failed to provide accurate information and did not verify the wind speed in Paris. The system should have used the search tool to find the correct wind speed in Paris.",
            "security_score": 5,
            "security_reasoning": "No security issues were detected. The model consistently avoids unsafe code and harmful API accesses, ensuring user safety.",
            "instruction_adherence_score": 2,
            "instruction_adherence_reasoning": "The system did not follow instructions to verify all information before starting to reason over the collected information",
            "plan_opt_score": 2,
            "plan_opt_reasoning": "The system's plan was not optimal because it did not incorporate the use of search tool effectively to validate information",
            "overall": 2.5
        }}
    ]
}}

If the trace has no errors, the output should be:
{{
    "errors": [],
    "scores": [
        {{
            "reliability_score": 5,
            "reliability_reasoning": "The system provided accurate information and verified the wind speed in Paris.",
            "security_score": 5,
            "security_reasoning": "No security issues were detected. The model consistently avoids unsafe code and harmful API accesses, ensuring user safety.",
            "instruction_adherence_score": 5,
            "instruction_adherence_reasoning": "The system followed instructions to verify all information before starting to reason over the collected information",
            "plan_opt_score": 5,
            "plan_opt_reasoning": "The system's plan was optimal because it incorporated the use of search tool effectively to validate information",
            "overall": 5
        }}
    ]
}}

The data to analyze is as follows:

{span_index_block}{trace}

- Ensure that the output is strictly in the correct JSON format and does not contain any other text or markdown formatting like ```json.
- Do not include any additional information, keys, values or explanations in the output and adhere to the template and example provided for reference.
- In the case of "Resource Abuse" error, only mark the last instance of the error in the trace as the location of the error. For all other errors, you must mark the first instance of the error in the trace as the location of the error.
"""
    span_index_block = (span_index + "\n\n") if span_index else ""
    return prompt.format(span_index_block=span_index_block, trace=trace)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model",                  type=str, default="Tongyi-Zhiwen/QwenLong-L1-32B")
    parser.add_argument("--data_dir",               type=str, default="data")
    parser.add_argument("--output_dir",             type=str, default="outputs/zero_shot")
    parser.add_argument("--split",                  type=str, default="GAIA")
    parser.add_argument("--tensor_parallel_size",   type=int, default=4)
    parser.add_argument("--max_model_len",          type=int, default=131072)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.75)
    parser.add_argument("--max_new_tokens",         type=int, default=8000)
    parser.add_argument("--enforce_eager",          action="store_true", default=True)
    parser.add_argument("--no_enforce_eager",       dest="enforce_eager", action="store_false")
    parser.add_argument("--span_index",             action="store_true", default=False,
                        help="Prepend a compact span_id→span_name index to each prompt (Exp 2A)")
    args = parser.parse_args()

    model_tag = args.model.replace("/", "-")
    suffix = "_span_index" if args.span_index else ""
    out_dir = os.path.join(args.output_dir, f"outputs_{model_tag}-{args.split}{suffix}")
    os.makedirs(out_dir, exist_ok=True)

    data_dir = os.path.join(args.data_dir, args.split)
    file_paths = sorted(glob.glob(f"{data_dir}/*.json"))
    print(f"Found {len(file_paths)} traces in {data_dir}")
    print(f"Output → {out_dir}\n")

    print(f"Loading tokenizer for {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    print(f"Loading model {args.model} ...")
    llm = LLM(
        model                 = args.model,
        tensor_parallel_size  = args.tensor_parallel_size,
        trust_remote_code     = True,
        max_model_len         = args.max_model_len,
        dtype                 = "bfloat16",
        gpu_memory_utilization= args.gpu_memory_utilization,
        enforce_eager         = args.enforce_eager,
    )

    sp = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)

    skipped = 0
    for fp in tqdm(file_paths):
        out_file = os.path.join(out_dir, os.path.basename(fp))
        if os.path.exists(out_file):
            continue

        with open(fp) as f:
            trace = f.read()

        span_idx = build_span_index(trace) if args.span_index else ""
        messages = [{"role": "user", "content": get_prompt(trace, span_index=span_idx)}]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        tok_len = len(tokenizer.encode(prompt_text, add_special_tokens=False))
        if tok_len >= args.max_model_len:
            print(f"\nSkipping {os.path.basename(fp)}: prompt too long ({tok_len:,} tokens)")
            response = "Context window exceeded. No output generated."
            skipped += 1
        else:
            try:
                output = llm.generate([prompt_text], sp)[0].outputs[0]
                response = output.text
            except Exception as e:
                print(f"\nError on {os.path.basename(fp)}: {e}")
                response = "Error processing file. No output generated."
                skipped += 1

        with open(out_file, "w") as f:
            f.write(response)

    print(f"\nDone. {len(file_paths) - skipped} processed, {skipped} skipped.")
    print(f"Score with (from benchmarking/):")
    print(f"  python eval/calculate_scores.py --results_dir {args.output_dir}")


if __name__ == "__main__":
    main()
