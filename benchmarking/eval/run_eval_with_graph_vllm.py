"""
eval/run_eval_with_graph_vllm.py — zero-shot QwenLong evaluation with Suppes causal graph
guidance injected into the prompt. Mirrors run_eval_with_graph.py (Gemini) but uses vLLM.

Identical to run_eval_vllm.py except the prompt is augmented with a "Causal Error Patterns"
section derived from the Suppes graph, placed before the trace.

Usage (from benchmarking/):
    CUDA_VISIBLE_DEVICES=3,4,5,6 python eval/run_eval_with_graph_vllm.py --split GAIA --causal_only
    CUDA_VISIBLE_DEVICES=3,4,5,6 python eval/run_eval_with_graph_vllm.py --split GAIA --causal_only --span_index

Outputs are saved to:
    outputs/zero_shot/outputs_{model_tag}-{split}-graph_causal_only/
and can be scored with the standard calculate_scores.py.
"""

import os
import sys
import glob
import json
import argparse
from pathlib import Path
from tqdm import tqdm

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

BENCH_DIR = Path(__file__).resolve().parent.parent
GRAPH_DATA_DIR = BENCH_DIR.parent / "graph" / "data"
DEFAULT_GRAPH_INPUT = GRAPH_DATA_DIR / "graph_input.pt"

sys.path.insert(0, str(BENCH_DIR))
from span_level_parser import parse_trace_to_step_level, _span_name


# ---------------------------------------------------------------------------
# Span index (same as run_eval_vllm.py)
# ---------------------------------------------------------------------------

def build_span_index(trace_str: str) -> str:
    try:
        trace_data = json.loads(trace_str)
    except Exception:
        return ""
    parsed = parse_trace_to_step_level(trace_data)
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
        for child in span.get("child_spans") or []:
            csid = child.get("span_id")
            csname = _span_name(child)
            if csid and csid not in seen:
                seen.add(csid)
                lines.append(f'    span_id "{csid}"  ({csname})')
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Causal graph loading + formatting (same logic as run_eval_with_graph.py)
# ---------------------------------------------------------------------------

def load_suppes_edges(causal_only: bool = False, threshold: float = 0.10,
                      graph_input: Path = DEFAULT_GRAPH_INPUT) -> list:
    import torch
    if not graph_input.exists():
        raise FileNotFoundError(f"{graph_input} not found")
    gi = torch.load(graph_input, weights_only=False)
    node_names = gi["node_names"]
    edge_index = gi["edge_index"]
    edge_weight = gi["edge_weight"]
    edge_is_causal = gi.get("edge_is_causal")
    correct_idx = node_names.index("Correct") if "Correct" in node_names else len(node_names) - 1

    edges = []
    for i in range(edge_index.shape[1]):
        src = edge_index[0, i].item()
        dst = edge_index[1, i].item()
        w = edge_weight[i].item()
        if src == correct_idx or dst == correct_idx:
            continue
        if causal_only:
            is_causal = (edge_is_causal[i].item() == 1.0) if edge_is_causal is not None else (w == 1.0)
            if not is_causal:
                continue
        else:
            if w < threshold:
                continue
        edges.append((node_names[src], node_names[dst], w))

    edges.sort(key=lambda x: -x[2])
    return edges


def format_graph_guidance(edges: list) -> str:
    if not edges:
        return ""
    lines = [
        "# Causal Error Patterns (data-driven, from prior trace analysis)",
        "The following causal relationships between error types have been statistically observed.",
        "When you identify an error of type A in the trace, actively look for errors of type B",
        "in subsequent spans, as B has been found to causally follow A.",
        "Higher strength values indicate stronger causal association.",
        "",
        "Format: [Source Error] → [Consequent Error]  (strength: X.XX)",
        "",
    ]
    for src, dst, w in edges:
        lines.append(f"  {src} → {dst}  (strength: {w:.2f})")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompt (same as run_eval_vllm.py, with graph guidance block added)
# ---------------------------------------------------------------------------

def get_prompt(trace: str, span_index: str = "", graph_guidance: str = "") -> str:
    graph_block = (graph_guidance + "\n\n") if graph_guidance else ""
    span_index_block = (span_index + "\n\n") if span_index else ""
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

{graph_block}- Based on the taxonomy above, analyze the LLM agent trace below and find errors in it.
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
    return prompt.format(graph_block=graph_block, span_index_block=span_index_block, trace=trace)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model",                  type=str,   default="Tongyi-Zhiwen/QwenLong-L1-32B")
    parser.add_argument("--data_dir",               type=str,   default="data")
    parser.add_argument("--output_dir",             type=str,   default="outputs/zero_shot")
    parser.add_argument("--split",                  type=str,   default="GAIA")
    parser.add_argument("--tensor_parallel_size",   type=int,   default=4)
    parser.add_argument("--max_model_len",          type=int,   default=131072)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.75)
    parser.add_argument("--max_new_tokens",         type=int,   default=8000)
    parser.add_argument("--enforce_eager",          action="store_true", default=True)
    parser.add_argument("--no_enforce_eager",       dest="enforce_eager", action="store_false")
    parser.add_argument("--span_index",             action="store_true", default=False,
                        help="Prepend compact span_id index to each prompt")
    parser.add_argument("--causal_only",            action="store_true", default=False,
                        help="Use only the 11 bootstrap-validated causal edges (w=1.0)")
    parser.add_argument("--edge_threshold",         type=float, default=0.10,
                        help="Min edge weight to include when not using --causal_only")
    parser.add_argument("--graph_input",            type=str,   default=None)
    args = parser.parse_args()

    graph_input = Path(args.graph_input) if args.graph_input else DEFAULT_GRAPH_INPUT
    edges = load_suppes_edges(causal_only=args.causal_only, threshold=args.edge_threshold,
                              graph_input=graph_input)
    graph_guidance = format_graph_guidance(edges)
    print(f"Loaded {len(edges)} causal edges ({'causal_only' if args.causal_only else f'threshold={args.edge_threshold}'})")

    model_tag = args.model.replace("/", "-")
    graph_tag = "graph_causal_only" if args.causal_only else f"graph_t{args.edge_threshold}"
    span_tag  = "_span_index" if args.span_index else ""
    out_dir = os.path.join(args.output_dir, f"outputs_{model_tag}-{args.split}-{graph_tag}{span_tag}")
    os.makedirs(out_dir, exist_ok=True)

    data_dir = os.path.join(args.data_dir, args.split)
    file_paths = sorted(glob.glob(f"{data_dir}/*.json"))
    print(f"Found {len(file_paths)} traces in {data_dir}")
    print(f"Output → {out_dir}\n")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    is_qwen3_1m = "Qwen3-30B-A3B" in args.model
    if is_qwen3_1m:
        import os as _os
        _os.environ["VLLM_ATTENTION_BACKEND"] = "DUAL_CHUNK_FLASH_ATTN"
        _os.environ["VLLM_USE_V1"] = "0"
        print("Qwen3-30B-A3B detected: enabling DUAL_CHUNK_FLASH_ATTN, chunked prefill, max_num_seqs=1")
    llm = LLM(
        model                  = args.model,
        tensor_parallel_size   = args.tensor_parallel_size,
        trust_remote_code      = True,
        max_model_len          = args.max_model_len,
        dtype                  = "bfloat16",
        gpu_memory_utilization = args.gpu_memory_utilization,
        enforce_eager          = args.enforce_eager,
        **(dict(
            enable_chunked_prefill   = True,
            max_num_batched_tokens   = 131072,
            max_num_seqs             = 1,
        ) if is_qwen3_1m else {}),
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
        messages = [{"role": "user", "content": get_prompt(trace, span_index=span_idx,
                                                            graph_guidance=graph_guidance)}]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

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
