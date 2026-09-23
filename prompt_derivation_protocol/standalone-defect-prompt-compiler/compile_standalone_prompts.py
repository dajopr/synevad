#!/usr/bin/env python3
"""Compile a standalone severity-graded defect taxonomy into single-shot editing prompts."""
import argparse
import json
import re
import sys
from pathlib import Path

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

GRADE_REFERENCE_PATTERNS = [
    r"\b(?:than|from|beyond|versus|vs\.?|as at|compared to)\s+(?:the\s+)?{g}\b",
    r"\b{g}\s+(?:severity|stage|grade|level|version|state)\b",
    r"\bat\s+(?:the\s+)?{g}\b",
]

NEGATION_PREFIXES = ("no ", "not ", "without ", "avoid ", "never ")


def normalize_negatives(items):
    out = []
    seen = set()
    for raw in items:
        s = " ".join((raw or "").split())
        if not s:
            continue
        n = s if s.lower().startswith(NEGATION_PREFIXES) else "no " + s
        if n.lower() not in seen:
            seen.add(n.lower())
            out.append(n)
    return out


def render_negatives(negatives):
    if not negatives:
        return ""
    return "Constraints: " + "; ".join(negatives) + ". "
def lint_chain_leakage(prompt_body, stage):
    low = prompt_body.lower()
    hits = [m for m in CHAIN_MARKERS if re.search(r"\b" + re.escape(m) + r"\b", low)]
    for grade in SEVERITY_SCALE:
        if grade == stage:
            continue
        for pat in GRADE_REFERENCE_PATTERNS:
            if re.search(pat.format(g=grade), low):
                hits.append("reference to '%s' grade" % grade)
                break
    return sorted(set(hits))


def compile_taxonomy(tax):
    warnings = []
    failures = []

    grading_mode = tax.get("grading_mode")
    if grading_mode != "standalone":
        raise SystemExit(
            "Input is not a standalone taxonomy (grading_mode=%r). Chained taxonomies "
            "belong to defect-prompt-compiler, or to chained-to-standalone-taxonomy "
            "if they should be ported." % (grading_mode,)
        )

    scale = tax.get("severity_scale", SEVERITY_SCALE)
    if scale != SEVERITY_SCALE:
        failures.append("severity_scale is %r, expected %r" % (scale, SEVERITY_SCALE))

    fixed_block = (tax.get("fixed_instruction_block") or "").strip()
    if not fixed_block:
        failures.append("fixed_instruction_block missing or empty")

    prompts = []

    for mode in tax.get("modes", []):
        name = mode.get("name", "")
        mechanism = mode.get("mechanism", "")
        default_region = mode.get("signature", {}).get("location", "")
        by_stage = {e.get("stage"): e for e in (mode.get("prompt_set") or [])}

        missing = [s for s in SEVERITY_SCALE if s not in by_stage]
        if missing:
            warnings.append({
                "mode": name, "stage": ",".join(missing),
                "issue": "prompt_set missing grade(s): %s" % ", ".join(missing),
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
                    "issue": "incremental/cross-grade marker in edit text: %r" % (hit,),
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
            "expected 4 prompts per compiled mode, got %d for %d modes" % (len(prompts), compiled_modes)
        )
    for p in prompts:
        if not p["prompt"].endswith(fixed_block):
            failures.append("%s/%s: prompt does not end with fixed block" % (p["mode"], p["stage"]))
        for neg in p["negatives"]:
            if neg not in p["prompt"]:
                failures.append("%s/%s: negative %r missing from prompt" % (p["mode"], p["stage"], neg))

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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("taxonomy", type=Path)
    ap.add_argument("out", type=Path)
    args = ap.parse_args()

    tax = json.loads(args.taxonomy.read_text(encoding="utf-8"))
    compiled = compile_taxonomy(tax)
    args.out.write_text(json.dumps(compiled, indent=2, ensure_ascii=False), encoding="utf-8")

    sc = compiled["self_check"]
    print(
        "%s: %d prompts, %d warnings, self_check passed=%s"
        % (compiled["category"], compiled["counts"]["prompts"], len(compiled["warnings"]), sc["passed"])
    )
    for f in sc["failures"]:
        print("  FAIL: %s" % f, file=sys.stderr)
    return 0 if sc["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
