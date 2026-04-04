"""
eval/run_eval_with_graph.py — LLM-as-judge evaluation with Suppes causal graph guidance.

Identical to run_eval.py except the prompt is augmented with a "Causal Error Patterns"
section derived from the Suppes graph built by the graph/ pipeline.

The causal graph encodes statistically-derived co-occurrence relationships:
  A → B (weight w) means: in traces where A appears, B tends to follow causally.
Only edges with weight >= EDGE_THRESHOLD are included to keep the prompt focused.

Usage (from benchmarking/):
    python eval/run_eval_with_graph.py --split GAIA --model openai/gpt-4o
    python eval/run_eval_with_graph.py --split GAIA --model openai/gpt-4o --edge_threshold 0.15

Outputs are saved to:
    outputs/outputs_{model}-{split}-graph_guided/
and can be scored with the standard calculate_scores.py.
"""

import os
import glob
import json
import time
import argparse
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import litellm
from litellm import completion, ContextWindowExceededError, RateLimitError
from dotenv import load_dotenv, find_dotenv
from tqdm import tqdm

load_dotenv(find_dotenv())

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BENCH_DIR           = Path(__file__).resolve().parent.parent
GRAPH_DATA_DIR      = BENCH_DIR.parent / "graph" / "data"
DEFAULT_GRAPH_INPUT = GRAPH_DATA_DIR / "graph_input.pt"

# Default edge weight threshold — only edges at or above this are shown to the LLM.
DEFAULT_EDGE_THRESHOLD = 0.10


# ---------------------------------------------------------------------------
# Load and format Suppes graph
# ---------------------------------------------------------------------------

def load_suppes_edges(
    threshold: float,
    causal_only: bool = False,
    graph_input: Path = DEFAULT_GRAPH_INPUT,
) -> list[tuple[str, str, float]]:
    """
    Load edges from graph_input.pt.

    Two modes:
      causal_only=True  — return only edges where edge_is_causal==1 (bootstrap
                          stability=1.0, i.e. survived 100% of bootstrap rounds).
                          These are the fully validated causal edges (~11 edges).
      causal_only=False — return all Suppes edges with weight >= threshold.
                          edge_weight is bootstrap stability in [0,1]; threshold
                          filters by stability but does NOT imply causal validation.

    Note: ALL edges in the graph already passed the Suppes precedence + probability-
    raising tests. edge_weight encodes how often each edge survived bootstrap
    resampling. edge_is_causal marks the subset stable across 100% of samples.
    """
    try:
        import torch
    except ImportError:
        raise ImportError("torch is required to load the Suppes graph. "
                          "Run from the causal conda environment or install torch.")

    if not graph_input.exists():
        raise FileNotFoundError(
            f"{graph_input} not found — run graph/04_build_graph_input.py first."
        )

    gi             = torch.load(graph_input, weights_only=False)
    node_names     = gi["node_names"]           # list of 20 strings
    edge_index     = gi["edge_index"]           # (2, E)
    edge_weight    = gi["edge_weight"]          # (E,)  bootstrap stability in [0,1]
    edge_is_causal = gi.get("edge_is_causal")  # (E,)  1.0 = validated causal edge
    correct_idx    = node_names.index("Correct") if "Correct" in node_names else len(node_names) - 1

    edges = []
    for i in range(edge_index.shape[1]):
        src = edge_index[0, i].item()
        dst = edge_index[1, i].item()
        w   = edge_weight[i].item()

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


def format_graph_guidance(edges: list[tuple[str, str, float]]) -> str:
    """
    Format Suppes edges as a concise guidance block for the LLM prompt.
    """
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
# Prompt
# ---------------------------------------------------------------------------

def get_prompt(trace: str, graph_guidance: str) -> str:
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

{graph_guidance}
- Based on the taxonomy above, analyze the LLM agent trace below and find errors in it.
- You must be exhaustive and find all the errors in the trace. Only include the final subcategories of the taxonomy (i.e. "Resource Not Found" and not "API Issues" or "System Execution Errors").
- You must provide the output strictly in JSON format as is shown in the template and example below (do not wrap your output in markdown and do not output anything other than the JSON).

Template for output:

{{
    "errors": [
        {{
            "category": "[INSERT ERROR CATEGORY FROM TAXONOMY HERE]", # The category of the error
            "location": "[INSERT LOCATION OF ERROR HERE]", # The location of the error in the trace (span id)
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

{trace}

- Ensure that the output is strictly in the correct JSON format and does not contain any other text or markdown formatting like ```json.
- Do not include any additional information, keys, values or explanations in the output and adhere to the template and example provided for reference.
- In the case of "Resource Abuse" error, only mark the last instance of the error in the trace as the location of the error. For all other errors, you must mark the first instance of the error in the trace as the location of the error.
"""
    return prompt.format(trace=trace, graph_guidance=graph_guidance)


# ---------------------------------------------------------------------------
# LiteLLM call (identical to run_eval.py)
# ---------------------------------------------------------------------------

def call_litellm(trace: str, graph_guidance: str, model: str = "openai/gpt-4o", api_base: str = None) -> str:
    prompt = get_prompt(trace, graph_guidance)

    if (
        "o1" in model
        or "o3" in model
        or "o4" in model
        or "anthropic" in model
        or "gemini-2.5" in model
    ):
        params = {
            "messages": [{"role": "user", "content": prompt}],
            "model": model,
            "max_completion_tokens": 8000,
            "reasoning_effort": "high",
            "drop_params": True,
        }
    else:
        params = {
            "messages": [{"role": "user", "content": prompt}],
            "model": model,
            "temperature": 0.0,
            "top_p": 1,
            "max_completion_tokens": 8000,
            "reasoning_effort": None,
            "drop_params": True,
        }

    if api_base:
        params["api_base"] = api_base
        dmx_key = os.environ.get("DMX_API_KEY")
        if dmx_key:
            params["api_key"] = dmx_key

    for attempt in range(3):
        try:
            response = completion(**params)
            return response.choices[0].message["content"]
        except RateLimitError as e:
            print(f"Rate limit error (attempt {attempt+1}/3): sleeping 60s and retrying...")
            time.sleep(60)
    raise RateLimitError("Exceeded 3 retries due to rate limiting")


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------

def process_file(file_path: str, output_dir: str, model: str, graph_guidance: str, api_base: str = None) -> str:
    output_file = os.path.join(output_dir, os.path.basename(file_path))
    if os.path.exists(output_file):
        return file_path  # already done, skip

    with open(file_path, "r") as f:
        trace = f.read()

    try:
        response = call_litellm(trace=trace, graph_guidance=graph_guidance, model=model, api_base=api_base)
    except ContextWindowExceededError as e:
        print(f"Context window exceeded for {file_path}: {e}. Skipping.")
        response = "Context window exceeded. No output generated."
    except Exception as e:
        print(f"Error processing {file_path}: {e}. Skipping.")
        response = "Error processing file. No output generated."

    with open(output_file, "w") as f:
        f.write(response or "No output produced")

    return file_path


# ---------------------------------------------------------------------------
# Main eval loop
# ---------------------------------------------------------------------------

def run_eval(
    directory: str,
    output_dir: str,
    model: str,
    graph_guidance: str,
    max_workers: int = 1,
    api_base: str = None,
) -> None:
    file_paths = glob.glob(f"{directory}/*.json")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_file, fp, output_dir, model, graph_guidance, api_base)
            for fp in file_paths
        ]
        for future in tqdm(
            concurrent.futures.as_completed(futures), total=len(file_paths)
        ):
            future.result()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM-as-judge eval augmented with Suppes causal graph guidance",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model",          type=str,   default="openai/gpt-4o")
    parser.add_argument("--data_dir",       type=str,   default="data")
    parser.add_argument("--output_dir",     type=str,   default="outputs/zero_shot")
    parser.add_argument("--max_workers",    type=int,   default=5)
    parser.add_argument("--split",          type=str,   default="GAIA",
                        help="Dataset split: GAIA or SWE Bench")
    parser.add_argument("--edge_threshold", type=float, default=DEFAULT_EDGE_THRESHOLD,
                        help="Minimum bootstrap stability weight to include (ignored if --causal_only)")
    parser.add_argument("--causal_only",    action="store_true",
                        help="Use only the ~11 fully validated causal edges (edge_is_causal=1, "
                             "bootstrap stability=1.0) instead of all Suppes edges")
    parser.add_argument("--api_base",      type=str,   default=None,
                        help="Custom API base URL for OpenAI-compatible proxies "
                             "(e.g. https://www.DMXapi.com/v1). "
                             "Set OPENAI_API_KEY in .env to your proxy key.")
    parser.add_argument("--graph_input",   type=str,   default=None,
                        help="Path to graph_input.pt (default: graph/data/graph_input.pt). "
                             "Use graph/data_train/graph_input.pt for train-only leakage-free graph.")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Build graph guidance string once (shared across all traces)
    # ------------------------------------------------------------------
    graph_input_path = Path(args.graph_input) if args.graph_input else DEFAULT_GRAPH_INPUT
    print(f"Loading Suppes graph from {graph_input_path} ...")
    edges = load_suppes_edges(args.edge_threshold, causal_only=args.causal_only,
                              graph_input=graph_input_path)
    graph_guidance = format_graph_guidance(edges)
    if args.causal_only:
        print(f"  {len(edges)} edges included (causal_only=True, bootstrap stability=1.0)")
    else:
        print(f"  {len(edges)} edges included (bootstrap stability >= {args.edge_threshold})")
    print()
    print("--- Graph guidance preview (first 10 edges) ---")
    for line in graph_guidance.splitlines()[:15]:
        print(line)
    print("...")

    # ------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------
    model_tag  = args.model.replace("/", "-")
    graph_tag  = "causal_only" if args.causal_only else f"suppes_t{args.edge_threshold}"
    # Append _train suffix when using train-only graph
    if args.graph_input and "data_train" in args.graph_input:
        graph_tag += "_train"
    output_dir = os.path.join(
        args.output_dir,
        f"outputs_{model_tag}-{args.split}-graph_{graph_tag}",
    )
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Run evaluation
    # ------------------------------------------------------------------
    data_dir = os.path.join(args.data_dir, args.split)
    print(f"\nEvaluating {data_dir} → {output_dir}")
    run_eval(
        directory     = data_dir,
        output_dir    = output_dir,
        model         = args.model,
        graph_guidance = graph_guidance,
        max_workers   = args.max_workers,
        api_base      = args.api_base,
    )


if __name__ == "__main__":
    litellm.drop_params = True
    main()
