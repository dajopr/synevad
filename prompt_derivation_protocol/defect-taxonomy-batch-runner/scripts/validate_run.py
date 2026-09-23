#!/usr/bin/env python3
"""Validate a completed batch run and write a run report.

Checks every category listed in the manifest: that both artifacts exist, that
the taxonomy is standalone and structurally sound, that the compiled prompt
count matches the mode count, and that no chain-leakage or numeric-quantity
markers survived into the prompts.

Usage:
    python validate_run.py run_manifest.json --report run_report.md
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

SEVERITY_SCALE = ["minimal", "slight", "moderate", "severe"]
INCREMENTAL = ["deepen", "extend", "further", "existing", "previous",
               "more than", "already", "continue"]
NUMERIC = re.compile(r"\b\d+(\.\d+)?\s*(%|mm|cm|px|pixels?|percent)?\b")


def check_names(c: dict[str, Any], tax: dict[str, Any] | None) -> list[str]:
    """The taxonomy must be filed under the dataset name, not the description.

    `category` is the join key between the manifest, the output folder and every
    downstream consumer, so a taxonomy that came back filed under its prose
    description is broken even though its content may be fine.
    """
    fails: list[str] = []
    folder = c["category"]
    desc = c.get("category_description")
    if desc is not None and not str(desc).strip():
        fails.append("category_description is present but empty — omit the field instead")
    if tax is not None:
        got = (tax.get("category") or "").strip()
        if got and got != folder:
            hint = " (looks like the description was used as the name)" if got == desc else ""
            fails.append(f"taxonomy category is {got!r}, expected the dataset name "
                         f"{folder!r}{hint}")
        tax_desc = (tax.get("category_description") or "").strip()
        if tax_desc and desc and tax_desc != desc.strip():
            fails.append(f"taxonomy category_description is {tax_desc!r}, "
                         f"expected {desc!r}")
    return fails


def check_taxonomy(tax: dict[str, Any]) -> list[str]:
    fails: list[str] = []
    if tax.get("grading_mode") != "standalone":
        fails.append(f"grading_mode is {tax.get('grading_mode')!r}, expected 'standalone'")
    if tax.get("severity_scale") != SEVERITY_SCALE:
        fails.append(f"severity_scale is {tax.get('severity_scale')}, expected {SEVERITY_SCALE}")
    if not (tax.get("fixed_instruction_block") or "").strip():
        fails.append("fixed_instruction_block missing or empty")

    modes = tax.get("modes") or []
    if not 3 <= len(modes) <= 5:
        fails.append(f"{len(modes)} modes, expected 3-5")

    for m in modes:
        name = m.get("name", "<unnamed>")
        stages = m.get("stages") or {}
        missing = [s for s in SEVERITY_SCALE if s not in stages]
        if missing:
            fails.append(f"{name}: stages missing {', '.join(missing)}")
        for grade, req in (("moderate", "literature"), ("severe", "literature"),
                           ("slight", "softened_moderate"), ("minimal", "softened_slight")):
            got = (stages.get(grade) or {}).get("derivation")
            if got != req:
                fails.append(f"{name}/{grade}: derivation is {got!r}, expected {req!r}")
        for grade in ("minimal", "slight"):
            axes = (stages.get(grade) or {}).get("attenuation_axes") or []
            if len(axes) < 2:
                fails.append(f"{name}/{grade}: needs >=2 attenuation_axes, got {len(axes)}")

        pset = {e.get("stage"): e for e in (m.get("prompt_set") or [])}
        pmissing = [s for s in SEVERITY_SCALE if s not in pset]
        if pmissing:
            fails.append(f"{name}: prompt_set missing {', '.join(pmissing)}")
        anti = set(m.get("signature", {}).get("anti_patterns") or [])
        for stage, entry in pset.items():
            edit = (entry.get("edit") or "").lower()
            for marker in INCREMENTAL:
                if marker in edit:
                    fails.append(f"{name}/{stage}: incremental marker {marker!r} in edit")
            if NUMERIC.search(edit):
                fails.append(f"{name}/{stage}: numeric quantity in edit text")
            negs = set(entry.get("negatives") or [])
            for a in anti - negs:
                fails.append(f"{name}/{stage}: anti-pattern {a!r} missing from negatives")

    sc = tax.get("self_check") or {}
    if not sc.get("passed"):
        for f in sc.get("failures") or ["self_check.passed is false"]:
            fails.append(f"taxonomy self_check: {f}")
    if tax.get("contamination_log"):
        fails.append(f"{len(tax['contamination_log'])} contamination incident(s) logged — review manually")
    return fails


def check_prompts(comp: dict[str, Any], tax: dict[str, Any]) -> list[str]:
    fails: list[str] = []
    # The fixed instruction block is appended verbatim to every prompt and may
    # legitimately contain words on the incremental list ("keep the existing
    # lighting"). Strip it before linting so it cannot raise false positives.
    fixed = (tax.get("fixed_instruction_block") or "").strip().lower()
    n_modes = len(tax.get("modes") or [])
    prompts = comp.get("generation_prompts") or []
    if len(prompts) != 4 * n_modes:
        fails.append(f"{len(prompts)} prompts for {n_modes} modes, expected {4 * n_modes}")
    for p in prompts:
        if p.get("applies_to") != "defect_free_source":
            fails.append(f"{p.get('mode')}/{p.get('stage')}: applies_to is {p.get('applies_to')!r}")
        text = (p.get("prompt") or "").lower()
        if fixed:
            text = text.replace(fixed, " ")
        for marker in INCREMENTAL:
            if marker in text:
                fails.append(f"{p.get('mode')}/{p.get('stage')}: incremental marker {marker!r} in prompt")
    sc = comp.get("self_check") or {}
    if not sc.get("passed"):
        for f in sc.get("failures") or ["self_check.passed is false"]:
            fails.append(f"compiler self_check: {f}")
    return fails


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifest", type=Path)
    ap.add_argument("--report", type=Path, default=Path("run_report.md"))
    args = ap.parse_args()

    man = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows, details, total_prompts = [], [], 0

    for c in man.get("categories", []):
        cat = c["category"]
        tax_p, pr_p = Path(c["outputs"]["taxonomy"]), Path(c["outputs"]["prompts"])
        fails: list[str] = []
        n_modes = n_prompts = 0

        for sp in c.get("sample_images") or []:
            if not Path(sp).exists():
                fails.append(f"staged sample missing: {Path(sp).name}")
        n_want = (c.get("sample_selection") or {}).get("selected")
        if n_want is not None and len(c.get("sample_images") or []) != n_want:
            fails.append(f"{len(c.get('sample_images') or [])} sample paths listed, expected {n_want}")

        if not tax_p.exists():
            fails.append("taxonomy.json missing")
        if not pr_p.exists():
            fails.append("prompts.json missing")

        if tax_p.exists() and pr_p.exists():
            try:
                tax = json.loads(tax_p.read_text(encoding="utf-8"))
                comp = json.loads(pr_p.read_text(encoding="utf-8"))
            except json.JSONDecodeError as e:
                tax = comp = None
                fails.append(f"unparseable JSON: {e}")
            if tax is not None:
                n_modes = len(tax.get("modes") or [])
                n_prompts = len(comp.get("generation_prompts") or [])
                total_prompts += n_prompts
                fails += check_names(c, tax)
                fails += check_taxonomy(tax)
                fails += check_prompts(comp, tax)

        desc = c.get("category_description") or "—"
        rows.append((cat, desc, n_modes, n_prompts,
                     "PASS" if not fails else f"FAIL ({len(fails)})"))
        if fails:
            details.append((cat, fails))

    lines = [f"# Run report — {man.get('benchmark')}", "",
             f"Manifest: `{args.manifest}`  ",
             f"Categories: {len(rows)} · prompts compiled: {total_prompts} · "
             f"failing: {len(details)}", "",
             "| category | description | modes | prompts | result |",
             "|---|---|---|---|---|"]
    lines += [f"| {c} | {d} | {m} | {p} | {r} |" for c, d, m, p, r in rows]
    if details:
        lines += ["", "## Failures", ""]
        for cat, fails in details:
            lines.append(f"### {cat}")
            lines += [f"- {f}" for f in fails]
            lines.append("")
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"{len(rows)} categories, {total_prompts} prompts, {len(details)} failing -> {args.report}")
    for cat, fails in details:
        print(f"  FAIL {cat}: {fails[0]}" + (f" (+{len(fails)-1} more)" if len(fails) > 1 else ""),
              file=sys.stderr)
    return 0 if not details else 1


if __name__ == "__main__":
    raise SystemExit(main())
