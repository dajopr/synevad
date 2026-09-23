---
name: "standalone-defect-prompt-compiler"
description: "Compile a standalone severity-graded defect-taxonomy JSON (from `standalone-defect-taxonomy-derivation`) into ready-to-run single-shot image-editing prompts — one per defect mode per severity grade (minimal, slight, moderate, severe), each applied independently to the defect-free source image. Use when asked to build/compile/assemble standalone or non-chained defect-generation prompts, four-severity editing prompts, or FLUX/editor prompts from a standalone taxonomy. Prefer the chained `defect-prompt-compiler` instead when the taxonomy has `prompt_chain` transitions and the defect must grow over successive edits."
---

# Standalone Defect Prompt Compiler

Turn a per-category **standalone severity taxonomy** (the JSON emitted by `standalone-defect-taxonomy-derivation`, or by `chained-to-standalone-taxonomy`) into the prompt set the synthesis pipeline runs: **one self-contained edit prompt per `mode × grade`**, each applied to the *same* defect-free source image, with the fixed instruction block appended.

For four modes × four grades this yields **16 generation prompts**.

**Standalone semantics.** Unlike the chained compiler, no prompt here depends on the output of another. `applies_to` is `"defect_free_source"` for every prompt, and the compiled text must never contain incremental phrasing. The grades are independent samples along a severity axis, not frames of a growth trajectory.

**No verification prompts.** This skill emits generation prompts only. Generated images are gated by a separate general-purpose image-edit scorer that lives outside this pipeline, so anti-patterns must survive into the prompt negatives — the compiler is the last place they can be enforced.

This skill does **not** derive taxonomies, invent defect features, or touch benchmark defect references. It is a deterministic compile step: every output string traces back to the input taxonomy. If the input is missing or malformed, say so rather than inventing content.

## Input contract

A taxonomy JSON document per category matching the `standalone-defect-taxonomy-derivation` output schema. The fields this skill consumes:

| Field | Used for |
|---|---|
| `benchmark`, `category` | Labelling the compiled output |
| `grading_mode`, `severity_scale` | Confirming this is a standalone taxonomy with the expected four grades |
| `fixed_instruction_block` | Appended verbatim to every prompt (pins lighting/background/viewpoint/image character) |
| `modes[].name`, `modes[].mechanism` | Naming and framing each prompt |
| `modes[].signature.location` | Fallback `region_of_interest` if a prompt entry omits it |
| `modes[].prompt_set[]` (`stage`, `edit`, `negatives`, `region_of_interest`, `instance_count`) | Body of each generation prompt |
| `modes[].stages[].descriptor` | Cross-check only — the compiled prompt uses `edit`, not the descriptor |

Do not read or require `citations`, `conflicts`, `contamination_log`, `search_queries`, `derived_from` — those are provenance, not prompt inputs. If `prompt_set` is absent or incomplete for a mode, flag it in `warnings` and skip that mode rather than fabricating entries.

**Wrong-variant guard:** if the input has `prompt_chain` / `transition` keys or `grading_mode != "standalone"`, stop and tell the user this is a chained taxonomy — it belongs to `defect-prompt-compiler`, or to `chained-to-standalone-taxonomy` if they want it ported. Do not attempt to convert incremental edits into absolute ones yourself; that is a derivation decision, not a compile step.

## Procedure

The transformation is deterministic string assembly, so **prefer running the script** in the appendix: save it as `compile_standalone_prompts.py` and run `python compile_standalone_prompts.py <taxonomy.json> <out.json>`. It applies the template verbatim and is the reproducibility artifact. Only assemble by hand (using the template below) if a script run is not possible; the result must be identical.

### 1. Validate the input

Confirm the JSON parses, `grading_mode` is `"standalone"`, and each mode has a `prompt_set` covering exactly `minimal, slight, moderate, severe` in that order. Record any gap in `warnings`.

### 2. Compile one prompt per mode × grade

Assemble each prompt with the generation template below: an opening that names the mode, mechanism and severity grade; the absolute `edit` text; the instance-count clause; an explicit localisation clause using `region_of_interest` (fall back to `signature.location`); the `negatives` rendered as a single negative-constraint clause; then the `fixed_instruction_block` appended verbatim. Set `applies_to: "defect_free_source"` on every prompt.

Preserve the absolute phrasing already in the taxonomy — never rewrite it into an incremental instruction, and never merge two grades into one prompt.

### 3. Lint for chain leakage

Scan every compiled prompt for incremental markers: `deepen`, `extend`, `further`, `existing`, `previous`, `more than`, `already`, `continue`, and for references to another grade (`than at moderate severity`, `beyond slight`). Incidental adjectival use of a grade word — "slight separation of the fracture edges" — is legitimate and must not be flagged. Any real hit is a taxonomy bug, not something to paper over: record it in `warnings` with the mode, grade and offending token, and still emit the prompt so the user can trace it.

### 4. Self-check and emit

Emit one compiled JSON per category matching the output schema below, and run the self-check, recording it in `self_check`.

## Self-check (record in `self_check`)

- Every mode in the taxonomy has exactly 4 generation prompts (or is listed in `warnings` with a reason).
- Prompts cover `minimal, slight, moderate, severe` per mode, in that order.
- Every prompt ends with the exact `fixed_instruction_block` text and names its `region_of_interest`.
- Every `negatives` item from a `prompt_set` entry appears (in normalized form) as a negative clause in that prompt.
- Every prompt has `applies_to: "defect_free_source"`.
- Chain-leakage lint clean, or hits logged in `warnings`.
- No content originates outside the input taxonomy; nothing was invented to fill a gap.
- Output validates against the schema.

Emit only with `"passed": true`, or with the failing items listed explicitly.

---

## Generation prompt template

`{…}` are substituted from the taxonomy; everything else is literal. The script applies this verbatim — hand assembly must match byte-for-byte.

Placeholder rendering rules:

- `{mode_name_text}` = the mode's `name` with underscores replaced by spaces.
- `{mechanism}` = the mode's `mechanism`.
- `{stage}` = `minimal`, `slight`, `moderate`, or `severe`.
- `{edit}` = the `prompt_set` entry's `edit`, verbatim, with any trailing period stripped.
- `{instance_clause}` = `"Add exactly one such defect."` when `instance_count` is `"single"`, or `"Add a single cluster of this defect, not scattered separate ones."` when `"cluster"`. Default to `"single"` if missing.
- `{region}` = the entry's `region_of_interest`, or the mode's `signature.location` if omitted.
- `{negatives_clause}` = `"Constraints: " + "; ".join(negatives) + ". "` — if `negatives` is empty, omit the whole clause. **Normalize each item to prohibitive form first:** an item that does not already begin with `no / not / without / avoid / never` gets `no ` prefixed, and exact repeats are dropped. A taxonomy that names the forbidden feature positively ("wide gouge") would otherwise render inside the constraint clause as a *requirement*. The `negatives` echoed in the compiled output use the normalized text.
- `{fixed_instruction_block}` = the taxonomy's `fixed_instruction_block`, appended verbatim.

Semantics: each prompt is applied **independently to the defect-free source image**. There is no ordering between prompts and no prompt consumes another's output.

```
Edit the image to add a single {mode_name_text} defect at {stage} severity, caused by {mechanism}. The defect must appear as: {edit}. {instance_clause} Apply this change only within the {region} and leave every other part of the image unchanged. {negatives_clause}Keep the edit photorealistic and continuous with the surrounding surface. {fixed_instruction_block}
```

Notes:

- `{edit}` is copied verbatim from the taxonomy and is already phrased as an **absolute** description of the finished appearance at that grade. Do not convert it into an incremental instruction, and do not import wording from a neighbouring grade to "clarify" it — the grades are meant to be independently interpretable.
- Naming the severity grade in the opening is deliberate: it is the only cross-grade signal the editor receives, and it costs nothing.
- The localisation clause and the "leave every other part unchanged" wording are what keep the mask tight for downstream blending and pixel-level evaluation.
- The negatives clause carries the mode's anti-patterns. Since no verification prompt is compiled, this is the only place anti-patterns are enforced — never drop it to shorten a prompt.

---

## Compiled output schema

```json
{
  "benchmark": "mvtec_ad",
  "category": "hazelnut",
  "grading_mode": "standalone",
  "severity_scale": ["minimal", "slight", "moderate", "severe"],
  "generation_prompts": [
    {
      "mode": "shell_crack",
      "stage": "minimal",
      "applies_to": "defect_free_source",
      "region_of_interest": "high-curvature ridge",
      "instance_count": "single",
      "negatives": ["no material loss", "no discoloration halo"],
      "prompt": "Edit the image to add a single shell crack defect at minimal severity, caused by brittle fracture under compressive load at shell curvature. The defect must appear as: add one barely perceptible broken hairline on a short section of the shell ridge, almost matching the surrounding shell tone. Add exactly one such defect. Apply this change only within the high-curvature ridge and leave every other part of the image unchanged. Constraints: no material loss; no discoloration halo. Keep the edit photorealistic and continuous with the surrounding surface. keep the diffuse top-down lighting, plain gray background, centered object pose, and image sharpness unchanged"
    }
  ],
  "counts": {"modes": 1, "prompts": 4},
  "warnings": [],
  "self_check": {"passed": true, "failures": []}
}
```

### Field notes

- `generation_prompts` is ordered by mode (taxonomy order), then by `severity_scale` order within each mode.
- `applies_to` is always `"defect_free_source"`. There is no chaining, so no `input_stage` / `output_stage` pair as in the chained compiler's output.
- `negatives`, `region_of_interest` and `instance_count` are echoed alongside the assembled `prompt` so a runner can build masks or filters without re-parsing the prompt string.
- `counts.prompts` must equal `4 × counts.modes` unless a mode was skipped; skipped modes appear in `warnings`.
- `warnings` entries: `{"mode": "...", "stage": "...", "issue": "...", "action": "skipped" | "emitted"}`. Lint hits use `"action": "emitted"`.
- There is no `verification_prompts` field. Scoring is handled by a separate general-purpose image-edit scorer.

---

## Appendix: compile script

Save as `compile_standalone_prompts.py` and run `python compile_standalone_prompts.py <taxonomy.json> <out.json>`.

```python
#!/usr/bin/env python3
"""Compile a standalone severity-graded defect taxonomy into single-shot editing prompts."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

SEVERITY_SCALE = ["minimal", "slight", "moderate", "severe"]

TEMPLATE = (
    "Edit the image to add a single {mode_name_text} defect at {stage} severity, "
    "caused by {mechanism}. The defect must appear as: {edit}. {instance_clause} "
    "Apply this change only within the {region} and leave every other part of the "
    "image unchanged. {negatives_clause}Keep the edit photorealistic and continuous "
    "with the surrounding surface. {fixed_instruction_block}"
)

INSTANCE_CLAUSES = {
    "single": "Add exactly one such defect.",
    "cluster": "Add a single cluster of this defect, not scattered separate ones.",
}

CHAIN_MARKERS = [
    "deepen", "deepened", "extend", "extended", "further",
    "existing", "previous", "more than", "continue", "already",
]

# A bare grade name is often legitimate wording ("slight separation of the edges"),
# so only flag it when it is used to *refer to a grade*.
GRADE_REFERENCE_PATTERNS = [
    r"\b(?:than|from|beyond|versus|vs\.?|as at|compared to)\s+(?:the\s+)?{g}\b",
    r"\b{g}\s+(?:severity|stage|grade|level|version|state)\b",
    r"\bat\s+(?:the\s+)?{g}\b",
]

NEGATION_PREFIXES = ("no ", "not ", "without ", "avoid ", "never ")


def normalize_negatives(items: list[str]) -> list[str]:
    """Render constraints in consistent prohibitive form, dropping exact repeats.

    A taxonomy that names a forbidden feature positively ("wide gouge") would
    otherwise render inside the constraint clause as a requirement.
    """
    out: list[str] = []
    seen: set[str] = set()
    for raw in items:
        s = " ".join((raw or "").split())
        if not s:
            continue
        n = s if s.lower().startswith(NEGATION_PREFIXES) else f"no {s[0].lower()}{s[1:]}"
        if n.lower() not in seen:
            seen.add(n.lower())
            out.append(n)
    return out


def render_negatives(negatives: list[str]) -> str:
    if not negatives:
        return ""
    return "Constraints: " + "; ".join(negatives) + ". "


def lint_chain_leakage(prompt_body: str, stage: str) -> list[str]:
    low = prompt_body.lower()
    hits = [m for m in CHAIN_MARKERS if re.search(rf"\b{re.escape(m)}\b", low)]
    for grade in SEVERITY_SCALE:
        if grade == stage:
            continue
        for pat in GRADE_REFERENCE_PATTERNS:
            if re.search(pat.format(g=grade), low):
                hits.append(f"reference to '{grade}' grade")
                break
    return sorted(set(hits))


def compile_taxonomy(tax: dict[str, Any]) -> dict[str, Any]:
    warnings: list[dict[str, Any]] = []
    failures: list[str] = []

    grading_mode = tax.get("grading_mode")
    if grading_mode != "standalone":
        raise SystemExit(
            f"Input is not a standalone taxonomy (grading_mode={grading_mode!r}). "
            "Chained taxonomies belong to defect-prompt-compiler, or to "
            "chained-to-standalone-taxonomy if they should be ported."
        )

    scale = tax.get("severity_scale", SEVERITY_SCALE)
    if scale != SEVERITY_SCALE:
        failures.append(f"severity_scale is {scale}, expected {SEVERITY_SCALE}")

    fixed_block = tax.get("fixed_instruction_block", "").strip()
    if not fixed_block:
        failures.append("fixed_instruction_block missing or empty")

    prompts: list[dict[str, Any]] = []

    for mode in tax.get("modes", []):
        name = mode.get("name", "")
        mechanism = mode.get("mechanism", "")
        default_region = mode.get("signature", {}).get("location", "")
        by_stage = {e.get("stage"): e for e in (mode.get("prompt_set") or [])}

        missing = [s for s in SEVERITY_SCALE if s not in by_stage]
        if missing:
            warnings.append({
                "mode": name, "stage": ",".join(missing),
                "issue": f"prompt_set missing grade(s): {', '.join(missing)}",
                "action": "skipped",
            })
            continue

        for stage in SEVERITY_SCALE:
            entry = by_stage[stage]
            edit = (entry.get("edit") or "").strip().rstrip(".")
            region = entry.get("region_of_interest") or default_region
            negatives = normalize_negatives(entry.get("negatives") or [])
            instance_count = entry.get("instance_count") or "single"

            if not region:
                warnings.append({
                    "mode": name, "stage": stage,
                    "issue": "no region_of_interest and no signature.location fallback",
                    "action": "emitted",
                })
            if not negatives:
                warnings.append({
                    "mode": name, "stage": stage,
                    "issue": "no negatives; anti-patterns unenforced (no scorer downstream)",
                    "action": "emitted",
                })
            for hit in lint_chain_leakage(edit, stage):
                warnings.append({
                    "mode": name, "stage": stage,
                    "issue": f"incremental/cross-grade marker in edit text: {hit!r}",
                    "action": "emitted",
                })

            prompt = TEMPLATE.format(
                mode_name_text=name.replace("_", " "),
                stage=stage,
                mechanism=mechanism,
                edit=edit,
                instance_clause=INSTANCE_CLAUSES.get(instance_count, INSTANCE_CLAUSES["single"]),
                region=region,
                negatives_clause=render_negatives(negatives),
                fixed_instruction_block=fixed_block,
            )

            prompts.append({
                "mode": name, "stage": stage,
                "applies_to": "defect_free_source",
                "region_of_interest": region,
                "instance_count": instance_count,
                "negatives": negatives,
                "prompt": prompt,
            })

    compiled_modes = len({p["mode"] for p in prompts})
    if len(prompts) != 4 * compiled_modes:
        failures.append(
            f"expected 4 prompts per compiled mode, got {len(prompts)} for {compiled_modes} modes"
        )
    for p in prompts:
        if not p["prompt"].endswith(fixed_block):
            failures.append(f"{p['mode']}/{p['stage']}: prompt does not end with fixed block")
        for neg in p["negatives"]:
            if neg not in p["prompt"]:
                failures.append(f"{p['mode']}/{p['stage']}: negative {neg!r} missing from prompt")

    return {
        "benchmark": tax.get("benchmark"),
        "category": tax.get("category"),
        "grading_mode": "standalone",
        "severity_scale": SEVERITY_SCALE,
        "generation_prompts": prompts,
        "counts": {"modes": compiled_modes, "prompts": len(prompts)},
        "warnings": warnings,
        "self_check": {"passed": not failures, "failures": failures},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("taxonomy", type=Path)
    ap.add_argument("out", type=Path)
    args = ap.parse_args()

    tax = json.loads(args.taxonomy.read_text(encoding="utf-8"))
    compiled = compile_taxonomy(tax)
    args.out.write_text(json.dumps(compiled, indent=2, ensure_ascii=False), encoding="utf-8")

    sc = compiled["self_check"]
    print(
        f"{compiled['category']}: {compiled['counts']['prompts']} prompts, "
        f"{len(compiled['warnings'])} warnings, self_check passed={sc['passed']}"
    )
    for f in sc["failures"]:
        print(f"  FAIL: {f}", file=sys.stderr)
    return 0 if sc["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
```
