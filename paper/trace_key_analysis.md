# Trace Key Analysis: Which Keys Are Needed for Which Error Types

## Legend
- ✓ = directly provides evidence for this error type
- ~ = provides supporting/contextual signal (useful but not primary)
- ✗ = not needed / no error signal

## Error Type Abbreviations (columns)
| Abbrev | Full Name |
|--------|-----------|
| Lang | Language-only Hallucination |
| ToolH | Tool-related Hallucination |
| PoorR | Poor Information Retrieval |
| MisI | Tool Output Misinterpretation |
| WrongP | Incorrect Problem Identification |
| ToolSel | Tool Selection Errors |
| Fmt | Formatting Errors |
| InstrNC | Instruction Non-compliance |
| ToolDef | Tool Definition Issues |
| EnvS | Environment Setup Errors |
| RateL | Rate Limiting |
| Auth | Authentication Errors |
| SvcErr | Service Errors |
| NotFnd | Resource Not Found |
| ResEx | Resource Exhaustion |
| Tmout | Timeout Issues |
| CtxFail | Context Handling Failures |
| ResAb | Resource Abuse |
| GoalD | Goal Deviation |
| TaskO | Task Orchestration |

---

## Full Key × Error Type Table

| Key | Lang | ToolH | PoorR | MisI | WrongP | ToolSel | Fmt | InstrNC | ToolDef | EnvS | RateL | Auth | SvcErr | NotFnd | ResEx | Tmout | CtxFail | ResAb | GoalD | TaskO | **Verdict** |
|-----|------|-------|-------|------|--------|---------|-----|---------|---------|------|-------|------|--------|--------|-------|-------|---------|-------|-------|-------|-------------|
| **TOP-LEVEL SPAN FIELDS** | | | | | | | | | | | | | | | | | | | | | |
| `timestamp` | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **DROP** — duration covers timing |
| `trace_id` | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **DROP** — pure metadata |
| `span_id` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | **KEEP** — location target for all errors |
| `parent_span_id` | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ~ | ✗ | ✗ | ✓ | **KEEP** — needed for call-chain structure (TaskO, CtxFail) |
| `trace_state` | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **DROP** — OTel telemetry metadata |
| `span_name` | ~ | ~ | ~ | ~ | ~ | ✓ | ✓ | ~ | ✓ | ~ | ~ | ~ | ~ | ~ | ~ | ~ | ~ | ✓ | ~ | ✓ | **KEEP** — identifies operation type ("Step N", "page_down", etc.) |
| `span_kind` | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **DROP** — OTel internal; redundant with `openinference.span.kind` |
| `service_name` | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **DROP** — deployment metadata |
| `resource_attributes` | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **DROP** — OTel SDK/runtime metadata |
| `scope_name` | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **DROP** — OTel instrumentation metadata |
| `scope_version` | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **DROP** — OTel instrumentation metadata |
| `duration` | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ~ | ✓ | ✗ | ~ | ✗ | ✗ | **KEEP** — primary signal for Timeout; supports ResEx, ResAb |
| `status_code` | ✗ | ~ | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ~ | ✓ | ✗ | ✗ | **KEEP** — marks Error spans; primary for all system execution errors |
| `status_message` | ✗ | ~ | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | **KEEP** — exception message; primary for system + resource errors |
| `events` | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ~ | ~ | ✗ | ✗ | **KEEP** — full stack trace + exception details for system errors |
| `links` | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **DROP** — OTel span links, empty in TRAIL |
| `logs` | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **DROP** — empty in TRAIL traces |
| `child_spans` | ~ | ~ | ~ | ~ | ~ | ~ | ~ | ~ | ~ | ~ | ~ | ~ | ~ | ~ | ~ | ~ | ~ | ~ | ~ | ~ | **KEEP** — structural field to traverse tree (not error signal itself) |
| **SPAN ATTRIBUTES** | | | | | | | | | | | | | | | | | | | | | |
| `openinference.span.kind` | ✗ | ✓ | ~ | ~ | ✗ | ✓ | ✓ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ~ | ✗ | ✓ | **KEEP** — LLM/TOOL/AGENT/CHAIN; identifies span role |
| `input.value` | ✗ | ~ | ✓ | ~ | ✗ | ✓ | ✓ | ✗ | ✓ | ~ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ~ | ✗ | ✗ | **KEEP** — tool call arguments; primary for Fmt, ToolSel, PoorR |
| `input.mime_type` | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **DROP** — always "text/plain" or "application/json", no signal |
| `output.value` | ✗ | ✗ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **KEEP** — tool return value; primary for PoorR, MisI, API errors |
| `output.mime_type` | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **DROP** — type tag only, no signal |
| `llm.input_messages.*` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ~ | ✗ | ✓ | ✓ | ✓ | ✓ | **KEEP** — full conversation history; primary for all reasoning errors |
| `llm.output_messages.*` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ | ✓ | ✓ | **KEEP** — LLM decisions and tool calls; primary for all reasoning errors |
| `llm.invocation_parameters` | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **DROP** — temperature/model config; not error signal |
| `llm.model_name` | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **DROP** — model identifier; not error signal |
| `llm.token_count.*` | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ~ | ✗ | ~ | ✗ | ✗ | ✗ | **DROP** — usage stats; ResEx/CtxFail already evident from status_message |
| `llm.tools.N.json_schema` | ✗ | ~ | ✗ | ✗ | ✗ | ~ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **DROP*** — only ToolDef; repeated every LLM span; extract once if needed |
| `tool.name` | ✗ | ✓ | ~ | ~ | ✗ | ✓ | ✓ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | **KEEP** — identifies which tool was called; primary for tool errors |
| `tool.description` | ✗ | ~ | ✗ | ✗ | ✗ | ~ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **KEEP** — needed for ToolDef (mismatch between description and behavior) |
| `tool.parameters` | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **KEEP** — expected param schema; primary for Fmt (wrong args) and ToolDef |
| `pat.app` | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **DROP** — Patronus app tag, no error signal |
| `pat.project.id` | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **DROP** — project metadata |
| `pat.project.name` | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **DROP** — project metadata |
| `smolagents.task` | ~ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ | **KEEP** — original task spec; primary for WrongP, InstrNC, GoalD, TaskO |
| `smolagents.tools_names` | ✗ | ✗ | ✗ | ✗ | ✗ | ~ | ✗ | ✗ | ~ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **DROP** — subset of info already in llm.tools schemas |
| `smolagents.max_steps` | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ~ | ✗ | ✗ | **DROP** — config param; ResAb evident from repeated error pattern |
| `smolagents.managed_agents.N.*` | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ~ | ✓ | **KEEP** — sub-agent names/descriptions; primary for TaskO |

---

## Summary: KEEP vs DROP

### KEEP (13 keys)
| Key | Why |
|-----|-----|
| `span_id` | Location target for every error |
| `parent_span_id` | Call-chain structure for TaskO, CtxFail |
| `span_name` | Operation identity ("Step N", tool name) |
| `duration` | Primary signal for Timeout; supports ResEx |
| `status_code` | Marks Error spans for all system execution errors |
| `status_message` | Exception message for system + resource errors |
| `events` | Full stack trace for system errors |
| `child_spans` | Structural traversal field |
| `openinference.span.kind` | LLM/TOOL/AGENT/CHAIN classification |
| `input.value` | Tool call arguments for Fmt, ToolSel, PoorR |
| `output.value` | Tool return value for PoorR, MisI, API errors |
| `llm.input_messages.*` | Full conversation history; all reasoning errors |
| `llm.output_messages.*` | LLM decisions and tool calls; all reasoning errors |
| `llm.tools.N.json_schema` | *(conditional)* Extract once per trace for ToolDef only |
| `tool.name` | Tool identity for tool-related errors |
| `tool.description` | Tool definition for ToolDef |
| `tool.parameters` | Expected param schema for Fmt, ToolDef |
| `smolagents.task` | Original task spec for WrongP, InstrNC, GoalD, TaskO |
| `smolagents.managed_agents.N.*` | Sub-agent info for TaskO |

### DROP (14 keys — zero error signal across all 20 categories)
`timestamp`, `trace_id`, `trace_state`, `span_kind`, `service_name`,
`resource_attributes`, `scope_name`, `scope_version`, `links`, `logs`,
`input.mime_type`, `output.mime_type`, `llm.invocation_parameters`,
`llm.model_name`, `llm.token_count.*`, `pat.app`, `pat.project.id`,
`pat.project.name`, `smolagents.tools_names`, `smolagents.max_steps`

### Special Note on `llm.tools.N.json_schema`
This key appears in **every** LLM span (repeated for each of the N tools at every step),
making it a major contributor to trace size. For ToolDef detection, only one copy is
needed (tool schemas don't change within a trace). Strategy: extract once at the
agent/trace level and drop from individual LLM spans.
