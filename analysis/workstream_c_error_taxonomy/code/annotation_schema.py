#!/usr/bin/env python3
"""
Frozen two-layer annotation schema for Workstream C.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set


BENCHMARK_TAXONOMY = {
    "Reasoning Errors": {
        "Hallucinations": [
            "Language-only",
            "Tool-related",
        ],
        "Information Processing": [
            "Poor Information Retrieval",
            "Tool Output Misinterpretation",
        ],
        "Decision Making": [
            "Incorrect Problem Identification",
            "Tool Selection Errors",
        ],
        "Output Generation": [
            "Formatting Errors",
            "Instruction Non-compliance",
        ],
    },
    "System Execution Errors": {
        "Configuration": [
            "Tool Definition Issues",
            "Environment Setup Errors",
        ],
        "API Issues": [
            "Rate Limiting",
            "Authentication Errors",
            "Service Errors",
            "Resource Not Found",
        ],
        "Resource Management": [
            "Resource Exhaustion",
            "Timeout Issues",
        ],
    },
    "Planning and Coordination Errors": {
        "Context Management": [
            "Context Handling Failures",
            "Resource Abuse",
        ],
        "Task Management": [
            "Goal Deviation",
            "Task Orchestration",
        ],
    },
}

MECHANISM_BUCKETS = [
    "causal-backed-gain",
    "corr-added-gain",
    "causal-preserving-neutral",
    "corr-induced-harm",
    "shared-failure",
]

MECHANISM_BUCKET_DEFINITIONS = {
    "causal-backed-gain": {
        "definition": "Causal-only already improves over baseline, and corr-union does not reverse that gain.",
        "inclusion_rule": "delta_wf1_vs_base > 0 for causal-only; corr does not reduce below baseline-level recovery.",
        "exclusion_rule": "Do not use when corr causes clear regression against causal-only with harmful FP propagation.",
    },
    "corr-added-gain": {
        "definition": "Corr-union yields additional improvement beyond causal-only on the same trace.",
        "inclusion_rule": "delta_corr_vs_causal > 0 with evidence of added true-positive recovery or better joint/localization behavior.",
        "exclusion_rule": "Do not use when improvement comes only from noisy FP increases without meaningful TP recovery.",
    },
    "causal-preserving-neutral": {
        "definition": "Corr-union is approximately neutral relative to causal-only while preserving causal behavior.",
        "inclusion_rule": "Near-zero net change vs causal-only and no material new harmful error pattern.",
        "exclusion_rule": "Do not use if corr clearly helps or harms in a direction that changes mechanism interpretation.",
    },
    "corr-induced-harm": {
        "definition": "Corr-union degrades behavior relative to causal-only due to over-propagation or spurious triggering.",
        "inclusion_rule": "delta_corr_vs_causal < 0 with evidence of added FP chains, localization drift, or wrong-category spread.",
        "exclusion_rule": "Do not use for cases where all variants fail similarly without corr-specific harm.",
    },
    "shared-failure": {
        "definition": "Baseline, causal-only, and corr-union all fail on the key gold signal(s).",
        "inclusion_rule": "Core gold categories remain missed across all variants; no variant offers meaningful recovery.",
        "exclusion_rule": "Do not use if any variant has clear mechanism-consistent recovery.",
    },
}

PATTERN_TAGS = [
    "missing-context-recovery",
    "dependency-chain-recovery",
    "precision-preserving-correction",
    "late-stage-recovery",
    "causal-anchor-dominant",
    "weak-signal-augmentation",
    "over-propagation-fp-chain",
    "localization-drift",
    "spurious-correlation-trigger",
    "already-solved-no-graph-needed",
    "shared-upstream-miss",
    "ambiguous-gold-location",
]

PATTERN_TAG_DEFINITIONS = {
    "missing-context-recovery": "Model recovers a gold error after graph signal supplies context that was absent in baseline/causal pass.",
    "dependency-chain-recovery": "Recovery follows a plausible upstream->downstream dependency chain consistent with error propagation logic.",
    "precision-preserving-correction": "Gain occurs with minimal added FP categories; precision is largely preserved.",
    "late-stage-recovery": "Improvement appears at later reasoning/trace stages rather than initial detection stage.",
    "causal-anchor-dominant": "Observed behavior is mainly explained by intervention-validated causal edges rather than corr additions.",
    "weak-signal-augmentation": "Corr edges amplify weak but relevant cues to recover otherwise missed gold categories.",
    "over-propagation-fp-chain": "Corr edges trigger cascaded false positives across multiple non-gold categories.",
    "localization-drift": "Category may be partially right but location/span quality degrades.",
    "spurious-correlation-trigger": "Corr edge appears to activate unsupported category predictions lacking grounded evidence.",
    "already-solved-no-graph-needed": "Baseline already solves key gold categories; graph adds little value.",
    "shared-upstream-miss": "All variants miss the same upstream prerequisite signal, causing downstream failure.",
    "ambiguous-gold-location": "Gold location mapping itself is noisy/uncertain, limiting definitive localization conclusions.",
}

CORR_EDGE_ROLE_LABELS = [
    "beneficial",
    "neutral",
    "harmful",
    "unknown",
]

SEVERITY_LABELS = ["low", "medium", "high"]
CONFIDENCE_LABELS = ["high", "medium"]


def leaf_categories() -> List[str]:
    out: List[str] = []
    for level2 in BENCHMARK_TAXONOMY.values():
        for leaves in level2.values():
            out.extend(leaves)
    return out


LEAF_CATEGORIES = leaf_categories()
_CANON = {c.lower().replace(" ", "").replace("-", ""): c for c in LEAF_CATEGORIES}
_ALIASES = {
    "hallucinations": "Language-only",
    "toolrelatedhallucination": "Tool-related",
    "toolrelatedhallucinations": "Tool-related",
    "toolselectionerror": "Tool Selection Errors",
    "toolselectionerrors": "Tool Selection Errors",
    "instructionnoncompliance": "Instruction Non-compliance",
}


def normalize_category(category: str) -> Optional[str]:
    if not category:
        return None
    key = category.lower().strip().replace(" ", "").replace("-", "")
    if key in _CANON:
        return _CANON[key]
    if key in _ALIASES:
        return _ALIASES[key]
    for k, v in _CANON.items():
        if key in k or k in key:
            return v
    return None


def extract_leaf_categories(errors: Iterable[Dict]) -> List[str]:
    found: Set[str] = set()
    for err in errors:
        if not isinstance(err, dict):
            continue
        cat = normalize_category(str(err.get("category", "")))
        if cat:
            found.add(cat)
    return sorted(found)


def schema_payload() -> Dict:
    return {
        "benchmark_taxonomy": BENCHMARK_TAXONOMY,
        "mechanism_buckets": MECHANISM_BUCKETS,
        "mechanism_bucket_definitions": MECHANISM_BUCKET_DEFINITIONS,
        "pattern_tags": PATTERN_TAGS,
        "pattern_tag_definitions": PATTERN_TAG_DEFINITIONS,
        "corr_edge_role_labels": CORR_EDGE_ROLE_LABELS,
        "severity_labels": SEVERITY_LABELS,
        "confidence_labels": CONFIDENCE_LABELS,
    }


def write_schema_files(out_dir: Path) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "annotation_schema_freeze.json"
    md_path = out_dir / "annotation_schema_freeze.md"
    payload = schema_payload()

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    lines = [
        "# Workstream C Annotation Schema Freeze",
        "",
        "## Layer 1: Benchmark taxonomy (frozen)",
        "",
    ]
    for l1, l2 in BENCHMARK_TAXONOMY.items():
        lines.append(f"- {l1}")
        for l2_name, leaves in l2.items():
            lines.append(f"  - {l2_name}: {', '.join(leaves)}")
    lines.extend(
        [
            "",
            "## Layer 2: Mechanism annotation labels (frozen)",
            f"- mechanism_bucket: {', '.join(MECHANISM_BUCKETS)}",
            f"- pattern_tags: {', '.join(PATTERN_TAGS)}",
            f"- corr_edge_role: {', '.join(CORR_EDGE_ROLE_LABELS)}",
            f"- impact_severity: {', '.join(SEVERITY_LABELS)}",
            f"- confidence: {', '.join(CONFIDENCE_LABELS)}",
            "",
        ]
    )
    lines.extend(["## Mechanism bucket definitions (theory-based, frozen)", ""])
    for k, v in MECHANISM_BUCKET_DEFINITIONS.items():
        lines.append(f"- {k}")
        lines.append(f"  - definition: {v['definition']}")
        lines.append(f"  - inclusion_rule: {v['inclusion_rule']}")
        lines.append(f"  - exclusion_rule: {v['exclusion_rule']}")
    lines.extend(["", "## Pattern tag definitions (frozen)", ""])
    for k, v in PATTERN_TAG_DEFINITIONS.items():
        lines.append(f"- {k}: {v}")
    with md_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return {"schema_json": str(json_path), "schema_md": str(md_path)}
