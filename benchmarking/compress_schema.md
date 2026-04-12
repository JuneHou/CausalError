# Trace Compression Schema

Documents every field removed or restructured by `compress_traces.py`, with justification and a note on what information is preserved. Use this to verify no error-detection signal is lost.

---

## Overview

### GAIA (117 traces) — compress only

| Stage | Total size | Avg per trace | Traces >500 KB |
|---|---|---|---|
| Original | 111.21 MB | 950.5 KB | 41 |
| After `--compress` | 51.86 MB | 443.2 KB | 29 |
| Reduction | **−53.4%** | | −12 |

### GAIA (117 traces) — compress + dedup (`--dedup`)

| Stage | Total size | Avg per trace | Traces >500 KB |
|---|---|---|---|
| Original | 111.21 MB | 950.5 KB | 41 |
| After `--dedup` | 36.70 MB | 313.7 KB | 26 |
| Reduction | **−67.0%** | | −15 |

### SWE Bench (31 traces) — compress + dedup (`--dedup`)

| Stage | Total size | Avg per trace | Traces >500 KB |
|---|---|---|---|
| Original | 74.37 MB | 2398.9 KB | 30 |
| After `--dedup` | 14.23 MB | 459.0 KB | 10 |
| Reduction | **−80.9%** | | −20 |

SWE Bench traces are much larger on average (2399 KB vs 951 KB for GAIA) due to longer
accumulated conversation histories, which makes the `--dedup` step proportionally more
effective (−80.9% vs −67.0%).

---

## Span structure

Each trace JSON has the shape:
```
{
  "trace_id": "...",
  "spans": [ <span>, ... ],
  "_tool_schemas": { ... }   ← added by compression (hoisted from LLM spans)
}
```

Each span has top-level fields and a `span_attributes` dict. The tree structure is encoded via `child_spans`.

---

## Top-level span fields

### Dropped

| Field | Reason |
|---|---|
| `trace_id` | Redundant — already present at the root of the trace JSON. |
| `trace_state` | OTel distributed-tracing propagation header. No content in TRAIL traces. |
| `span_kind` | OTel kind (INTERNAL/CLIENT/SERVER). Always redundant with `openinference.span.kind` in `span_attributes`. |
| `service_name` | Deployment identifier (`gaia-annotations/app:GAIA-Samples`). Same value in every span of every trace. |
| `resource_attributes` | OTel SDK metadata: `pat.account.id`, `service.name`, `telemetry.sdk.{language,name,version}`. No variation across traces. |
| `scope_name` | OTel instrumentation scope name. No signal. |
| `scope_version` | OTel instrumentation scope version. No signal. |
| `links` | OTel span links. Always `[]` in TRAIL. |
| `logs` | Low-level function call logs (`function.name`, `function.arguments`, `function.output`). Only populated on **UNKNOWN** spans (Patronus evaluation-harness wrapper spans, e.g. `main()` returning `<null>`). Not agent behavior — not relevant to the 20 TRAIL error categories. |

### Kept

| Field | Content | Why kept |
|---|---|---|
| `span_id` | Hex string (e.g. `"77fb7128d6f04862"`) | Primary key; used in the `location` field of judge output. |
| `span_name` | Human-readable span name | Context for the judge. |
| `parent_span_id` | Hex string | Needed to reconstruct the call tree. |
| `timestamp` | ISO datetime | Useful for ordering and for Timeout Issues detection. |
| `duration` | Float (seconds) | Direct signal for Timeout Issues. |
| `status_code` | `"OK"` / `"ERROR"` | Quick indicator of span failure. |
| `status_message` | Error message string | Signal for System Execution Errors (rate limits, auth, 404, etc.). |
| `events` | List of OTel events | **Contains exception events** (Name: `"exception"`) with `exception.message` and `exception.stacktrace`. Present on CHAIN (192 spans), TOOL (174 spans), and LLM (1 span) across all GAIA traces. Direct evidence for Tool Definition Issues, Environment Setup Errors, and API Issues. |
| `child_spans` | Nested span list | Structural field for the trace tree. Processed recursively. |

---

## `span_attributes` fields

### Dropped from all span kinds

| Key | Reason |
|---|---|
| `pat.app` | Patronus application tag. Same across all spans. No signal. |
| `pat.project.id` | Patronus project UUID. Same across all traces. No signal. |
| `pat.project.name` | Patronus project name. Same across all traces. No signal. |
| `input.mime_type` | Always `"text/plain"` or `"application/json"`. No variation. |
| `output.mime_type` | Same — always `"text/plain"` or `"application/json"`. |

### Dropped from specific span kinds

| Key | Span kind | Reason |
|---|---|---|
| `llm.invocation_parameters` | LLM | Temperature, max_tokens, stop sequences. Config metadata — not error signal. |
| `llm.model_name` | LLM | Model identifier string. Not error signal. |
| `llm.token_count.prompt` | LLM, AGENT | Token counts. No direct error signal (context overflow is detectable from skip behavior, not counts). |
| `llm.token_count.completion` | LLM, AGENT | Same. |
| `llm.token_count.total` | LLM, AGENT | Same. |
| `smolagents.max_steps` | AGENT | Configuration parameter. Resource Abuse errors are evident from step patterns in the trace, not from this value. |
| `smolagents.tools_names` | AGENT | Comma-separated tool name list. This is a strict subset of the information already available via TOOL spans (`tool.name`, `tool.description`, `tool.parameters`) and the hoisted `_tool_schemas`. |
| `input.value` | **LLM only** | **Primary compression win (−43 MB, −39% of original).** On LLM spans, `input.value` is a JSON-string re-encoding of `llm.input_messages.*`, which is already present in fully-structured form as `llm.input_messages.N.message.{content,role}`. Zero information loss. **Not dropped from AGENT, CHAIN, or TOOL spans** (see below). |

### Restructured (not dropped)

| Key | Action | Effect |
|---|---|---|
| `llm.tools.N.tool.json_schema` | **Hoisted** to trace root as `_tool_schemas` | Tool schemas are identical across all LLM spans in a trace (tools don't change mid-run). Extracting a single copy and attaching it at `{"_tool_schemas": {"llm.tools.0.tool.json_schema": ..., ...}}` eliminates per-span repetition while keeping the full information for Tool Definition Issues detection. |

---

## Kept fields by span kind

### AGENT spans

| Key | Content | Why kept |
|---|---|---|
| `openinference.span.kind` | `"AGENT"` | Span type classifier. |
| `smolagents.task` | The full task string given to the agent | Task context needed to evaluate Incorrect Problem Identification and Instruction Non-compliance. |
| `smolagents.managed_agents.N.name` | Sub-agent name | Agent orchestration context for Task Orchestration errors. |
| `smolagents.managed_agents.N.description` | Sub-agent description | Relevant to Tool Definition Issues when the description mismatches behavior. |
| `input.value` | JSON: `{"task": "..."}` | Agent-level input; task string. Kept (not LLM span). |
| `output.value` | Final agent output | The answer produced; needed to assess Instruction Non-compliance and Goal Deviation. |

### CHAIN spans

| Key | Content | Why kept |
|---|---|---|
| `openinference.span.kind` | `"CHAIN"` | Span type classifier. |
| `input.value` | Serialized `ActionStep` object | Step-level state: `step_number`, `error`, `tool_calls`, `observations`. Contains the error field directly — key signal for detecting step-level failures. |
| `output.value` | Step result | What the step produced. |

### LLM spans

| Key | Content | Why kept |
|---|---|---|
| `openinference.span.kind` | `"LLM"` | Span type classifier. |
| `llm.input_messages.N.message.role` | `"system"` / `"user"` / `"assistant"` / `"tool"` | Message role. |
| `llm.input_messages.N.message.content` | Full message content | The conversation history fed to the LLM at this step — the primary source of signal for Hallucinations, Tool Output Misinterpretation, Incorrect Problem Identification, Formatting Errors, and Instruction Non-compliance. |
| `llm.output_messages.0.message.role` | `"assistant"` | Role of the LLM's response. |
| `llm.output_messages.0.message.content` | LLM response text | The model's reasoning and code/action output at this step. |
| `llm.output_messages.0.message.tool_calls.*` | Tool call name, arguments, ID | Structured tool invocation details — needed for Tool Selection Errors, Tool Definition Issues, and Poor Information Retrieval. |
| `output.value` | LLM output (may overlap with `llm.output_messages`) | Kept as a fallback; some spans populate one but not the other. |

### TOOL spans

| Key | Content | Why kept |
|---|---|---|
| `openinference.span.kind` | `"TOOL"` | Span type classifier. |
| `tool.name` | Tool name string | Which tool was called. |
| `tool.description` | Tool description string | Cross-referenced against behavior for Tool Definition Issues. |
| `tool.parameters` | Tool parameter schema | Parameter definitions; relevant to Tool Definition Issues. |
| `input.value` | JSON: `{"args": [...], "kwargs": {...}}` | Actual arguments passed to the tool. |
| `output.value` | Tool return value | The tool's actual output — primary signal for Tool Output Misinterpretation, Resource Not Found, and Service Errors. |

### UNKNOWN spans

| Key | Content | Why kept |
|---|---|---|
| *(none beyond OTel fields)* | Patronus wrapper spans for the evaluation harness | All `span_attributes` on UNKNOWN spans are `pat.*` fields, which are dropped. UNKNOWN spans carry no agent-behavior signal and are effectively empty after compression. |

---

## What is NOT deduplicated (future work)

`llm.input_messages.*` on LLM spans is not deduplicated. Because each LLM call receives the full accumulated conversation history, consecutive LLM spans contain overlapping prefixes (step N's messages are a superset of step N−1's). This accounts for ~40 MB (36% of original). Deduplication (keeping only delta messages per step) would reduce average trace size further but requires a trace-level pass and changes how the judge reads the conversation.

---

## Error category coverage check

| TRAIL error category | Primary field(s) retained |
|---|---|
| Language-only Hallucination | `llm.input_messages.*`, `llm.output_messages.*` |
| Tool-related Hallucination | `llm.output_messages.*`, `tool.name`, `output.value` (TOOL) |
| Poor Information Retrieval | `llm.output_messages.*`, `output.value` (TOOL), `tool.name` |
| Tool Output Misinterpretation | `llm.input_messages.*`, `output.value` (TOOL) |
| Incorrect Problem Identification | `smolagents.task`, `llm.input_messages.*`, `llm.output_messages.*` |
| Tool Selection Errors | `llm.output_messages.*.tool_calls.*`, `tool.name`, `tool.description` |
| Formatting Errors | `llm.output_messages.*`, `output.value` (TOOL) |
| Instruction Non-compliance | `smolagents.task`, `llm.output_messages.*`, `output.value` (AGENT) |
| Tool Definition Issues | `tool.description`, `tool.parameters`, `_tool_schemas`, `smolagents.managed_agents.*` |
| Environment Setup Errors | `events` (exception.message), `status_code`, `status_message`, `output.value` (TOOL) |
| Rate Limiting (429) | `status_code`, `status_message`, `events` |
| Authentication Errors (401/403) | `status_code`, `status_message`, `events` |
| Service Errors (500) | `status_code`, `status_message`, `events` |
| Resource Not Found (404) | `status_code`, `status_message`, `events`, `output.value` (TOOL) |
| Resource Exhaustion | `status_code`, `status_message`, `events`, `output.value` (TOOL) |
| Timeout Issues | `duration`, `status_message`, `events` |
| Context Handling Failures | `llm.input_messages.*` (accumulating history), `output.value` (CHAIN) |
| Resource Abuse | `llm.output_messages.*` (repeated identical calls), TOOL span patterns |
| Goal Deviation | `smolagents.task`, `llm.output_messages.*`, `output.value` (AGENT) |
| Task Orchestration | `smolagents.managed_agents.*`, `output.value` (CHAIN/AGENT) |
