# Replicating the Labeled Errors (Original Run Config)

You want to **re-run the same agent setup** that produced each trace so you can reproduce the same failures. Here is what the dataset provides and what is missing.

---

## What is in the trace files

The **trace JSON files** themselves contain the original run configuration, but the **split format differs** (GAIA vs SWE Bench).

### SWE Bench traces (`benchmarking/data/SWE Bench/*.json`)

Each trace has **OpenInference-style LLM spans** with full prompts and model info. Look for spans with `"span_name": "LiteLLMModel.__call__"` and their `span_attributes`:

| What you need | Where in the trace | Example |
|---------------|--------------------|--------|
| **System prompt** | First LLM span → `span_attributes["llm.input_messages.0.message.content"]` | Full Thought/Code/Observation system prompt with examples, rules, allowed modules, “500 characters at a time”, etc. |
| **Input / task prompt** | Same span → `span_attributes["llm.input_messages.1.message.content"]` | “New task: …” + full issue text, `<issue>`, `<repo>`, `<base_commit>`, `<patch>`, and instructions (gitingest, regex, 500-char limit, etc.) |
| **Model** | Same span → `span_attributes["llm.model_name"]` | `"anthropic/claude-3-7-sonnet-latest"` (paper: backbone `claude-3-7-sonnet-20250219`) |
| **Temperature** | Same span → `span_attributes["llm.invocation_parameters"]` | `"{}"` in the traces — **not recorded**; use 0 for reproducibility or your framework default. |

So for SWE Bench: **system prompt, input prompt, and model are all in the trace**. Temperature is not; the paper does not report it.

### GAIA traces (`benchmarking/data/GAIA/*.json`)

GAIA uses a different structure (OpenDeepResearch-style hierarchy). Two places matter:

| What you need | Where in the trace |
|---------------|--------------------|
| **Task / input prompt** | In an early span: under `logs[].body["function.arguments"]` or `logs[].body["function.output"]` → look for `"question"` and optionally the `"example"` object (task_id, true_answer, annotator metadata). |
| **Model** | In LLM spans: `span_attributes["llm.model_name"]` → `"o3-mini"`. Paper: **`o3-mini-2025-01-31`** for manager and search agents. In some GAIA traces the model also appears as `function.arguments["model_id"]` (e.g. `"o3-mini"`). |
| **System prompt** | In GAIA, the **system prompt is inside nested LLM spans** (same as SWE Bench: `llm.input_messages.0.message.content` in any `LiteLLMModel.__call__` span). It may differ from SWE Bench (different agent: OpenDeepResearch vs CodeAct). |
| **Temperature** | Not present in the trace; paper does not specify. Use 0 or framework default for replication. |

So for GAIA: **input question, model, and (in LLM spans) system prompt are in the trace**. Temperature is not.

---

## Summary table

| Item | SWE Bench | GAIA | Note |
|------|-----------|------|------|
| **System prompt** | ✅ In first LLM span: `llm.input_messages.0.message.content` | ✅ In LLM spans: same key | Full text in trace |
| **Input / task prompt** | ✅ In first LLM span: `llm.input_messages.1.message.content` | ✅ In logs: `question` (and `example`) | Full task in trace |
| **Model** | ✅ `llm.model_name` → e.g. `anthropic/claude-3-7-sonnet-latest` | ✅ `llm.model_name` → `o3-mini`; paper: `o3-mini-2025-01-31` | Paper also names exact SWE model: `claude-3-7-sonnet-20250219` |
| **Temperature** | ❌ Not in trace (`llm.invocation_parameters` is `{}`) | ❌ Not in trace | Use 0 or default for replication |

---

## How to replicate the errors

1. **Extract** from each trace (per split):
   - System prompt  
   - Input/task prompt  
   - Model id (and map to exact version if needed, e.g. `claude-3-7-sonnet-20250219`, `o3-mini-2025-01-31`).
2. **Set** temperature (not in trace): use **0** for deterministic replication, or your framework’s default.
3. **Re-run** the same agent setup:
   - **SWE Bench**: CodeAct-style agent with gitingest, sandbox, Thought/Code/Observation and the instructions from the trace (500-char limit, etc.). Paper: “we use a CodeAct agent … provide it access to a sandboxed environment, a python interpreter and the gitingest library” and “instructional constraints such as output length limits”.
   - **GAIA**: OpenDeepResearch-style hierarchical agent (manager + search agents) with `o3-mini-2025-01-31` as in the paper.
4. **Compare** the new trace (or final answer) to the original trace and the TRAIL annotations to see if you get the same (or similar) errors.

**Extraction script:** From the `benchmarking/` directory you can run:

```bash
python extract_run_config.py --trace_file "data/SWE Bench/<trace_id>.json"
python extract_run_config.py --trace_file "data/GAIA/<trace_id>.json" --out_dir extracted_configs
```

With `--out_dir`, the script writes a JSON of the run config plus `*_system_prompt.txt` and `*_input_prompt.txt` for easy reuse.

---

## Paper references for setup

- **GAIA**: “We closely follow this hierarchical structure and adopt the **Hugging Face OpenDeepResearch agent** … We select the state-of-the-art **o3-mini-2025-01-31** … as the backbone model for the manager and search agents.”
- **SWE Bench**: “We use a **CodeAct agent** … provide it access to a sandboxed environment, a python interpreter and the **gitingest** library. We select **claude-3-7-sonnet-20250219** as the backbone model … we add **instructional constraints** such as output length limits and force exploration via prompts. The complete prompt is at § A.12.”

So for full replication you still need the **exact agent code** (OpenDeepResearch for GAIA, CodeAct + gitingest for SWE Bench) and, for SWE Bench, the **full prompt** from the paper appendix § A.12 if it differs from what’s in the trace. The trace gives you system prompt, input prompt, and model; temperature you must choose (e.g. 0).
