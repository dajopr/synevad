#!/usr/bin/env python3
"""Attach a plain-language description to each category in a run manifest.

The dataset name is never replaced. `category` stays the directory name in the
manifest, in the dispatch prompt, in the output paths and in the taxonomy JSON,
so every artifact joins back to the dataset on one key. What this script adds is
`category_description`: a short phrase describing what the object actually is,
handed to `standalone-defect-taxonomy-derivation` alongside the name.

That matters because the derivation classifies material family from what it is
told and seeds its literature search with it. `grid` alone is a poor seed;
`grid` plus "anodized aluminium mesh screen" is a good one -- and the name is
still `grid` everywhere it needs to be.

Descriptions are validated, not trusted: one carrying a defect term would steer
the taxonomy toward a conclusion the derivation is supposed to reach on its own,
and one carrying a benchmark identifier breaks the contamination firewall.

Usage:
    python set_descriptions.py run_manifest.json \
        --describe grid="anodized aluminium mesh screen, rigid metal"
    python set_descriptions.py run_manifest.json --from descriptions.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

# A description that already names a failure mode pre-decides the taxonomy.
DEFECT_TERMS = {
    "crack", "cracked", "scratch", "scratched", "defect", "defective", "broken",
    "damage", "damaged", "anomaly", "anomalous", "fault", "faulty", "flaw",
    "contaminated", "contamination", "hole", "missing", "bent", "dent", "dented",
    "stain", "stained", "corroded", "corrosion", "chipped", "torn", "worn",
    "misprint", "smudge", "squeeze", "cut", "poke", "fray", "frayed", "melt",
}

BENCHMARK_TERMS = {"mvtec", "visa", "vis-a", "mpdd", "btad", "realiad", "real-iad",
                   "loco", "benchmark", "dataset", "train", "test"}

# A description may be a phrase, not just a label -- but it stays a description
# of the pristine object, not a substitute for the derivation's own analysis.
MAX_WORDS = 12
MAX_CHARS = 120


def validate(text: str) -> list[str]:
    problems: list[str] = []
    clean = text.strip()
    if not clean:
        return ["empty description"]
    if len(clean) > MAX_CHARS:
        problems.append(f"{len(clean)} chars, max {MAX_CHARS}")
    words = re.findall(r"[a-z0-9\-]+", clean.lower())
    if len(words) > MAX_WORDS:
        problems.append(f"{len(words)} words, max {MAX_WORDS} — describe what the object is, "
                        f"not how it is made or how it fails")
    hits = sorted(set(words) & DEFECT_TERMS)
    if hits:
        problems.append(f"defect term(s) {', '.join(hits)} — the description must not "
                        f"pre-decide a failure mode")
    bench = sorted(set(words) & BENCHMARK_TERMS)
    if bench:
        problems.append(f"benchmark/split term(s) {', '.join(bench)} — firewall violation")
    if re.search(r"\d", clean) and not re.search(r"[a-z]", clean.lower()):
        problems.append("numeric-only description")
    return problems


def apply_descriptions(man: dict[str, Any], mapping: dict[str, str],
                       strict: bool) -> tuple[int, list[str]]:
    by_cat = {c["category"]: c for c in man.get("categories", [])}
    unknown = sorted(set(mapping) - set(by_cat))
    errors = [f"no such category in manifest: {u}" for u in unknown]
    applied = 0

    for cat, text in mapping.items():
        entry = by_cat.get(cat)
        if entry is None:
            continue
        problems = validate(text)
        if problems:
            errors.append(f"{cat} -> {text!r}: " + "; ".join(problems))
            continue
        clean = text.strip()
        entry["category_description"] = clean
        # `category` is untouched -- in the manifest, in the call, in the paths.
        # The description rides alongside it as extra grounding.
        for call in entry.get("calls", []):
            if call.get("step") == 1:
                # Rebuild rather than assign, so the description sits next to the
                # name and `output` stays last -- consumers that read args
                # positionally otherwise pick up the description as the path.
                rebuilt: dict[str, Any] = {}
                for key, value in call["args"].items():
                    if key == "category_description":
                        continue
                    rebuilt[key] = value
                    if key == "category":
                        rebuilt["category"] = cat
                        rebuilt["category_description"] = clean
                if "category_description" not in rebuilt:
                    rebuilt["category_description"] = clean
                call["args"] = rebuilt
        applied += 1

    described = sum(1 for e in by_cat.values() if e.get("category_description"))
    man.setdefault("descriptions", {})
    man["descriptions"].update({
        "described": described,
        "total": len(by_cat),
        "policy": "category is always the dataset directory name; "
                  "category_description is optional extra grounding sent alongside it",
    })

    if errors and strict:
        return applied, errors
    return applied, errors


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifest", type=Path)
    ap.add_argument("--describe", action="append", default=[], metavar="CATEGORY=DESCRIPTION",
                    help="description for one category; repeatable")
    ap.add_argument("--from", dest="from_file", type=Path, default=None,
                    help='JSON file of {"category": "description", ...}')
    ap.add_argument("--clear", action="append", default=[], metavar="CATEGORY",
                    help="drop an existing description; repeatable")
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero and write nothing if any description fails validation")
    ap.add_argument("--plan-md", type=Path, default=None,
                    help="re-render the human-readable plan with the descriptions")
    args = ap.parse_args()

    mapping: dict[str, str] = {}
    if args.from_file:
        mapping.update(json.loads(args.from_file.read_text(encoding="utf-8")))
    for item in args.describe:
        if "=" not in item:
            print(f"error: --describe expects CATEGORY=DESCRIPTION, got {item!r}", file=sys.stderr)
            return 2
        cat, _, text = item.partition("=")
        mapping[cat.strip()] = text.strip()

    man = json.loads(args.manifest.read_text(encoding="utf-8"))
    for cat in args.clear:
        for entry in man.get("categories", []):
            if entry.get("category") == cat.strip():
                entry.pop("category_description", None)
                for call in entry.get("calls", []):
                    call.get("args", {}).pop("category_description", None)
    applied, errors = apply_descriptions(man, mapping, args.strict)

    if errors and args.strict:
        for e in errors:
            print(f"  REJECT {e}", file=sys.stderr)
        print(f"strict mode: nothing written ({len(errors)} rejected)", file=sys.stderr)
        return 1

    args.manifest.write_text(json.dumps(man, indent=2), encoding="utf-8")
    if args.plan_md:
        # The plan was rendered before naming; re-render so the table a human
        # reads matches the names the agents will actually receive.
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from scan_categories import render_plan_md  # noqa: PLC0415

        args.plan_md.parent.mkdir(parents=True, exist_ok=True)
        args.plan_md.write_text(render_plan_md(man), encoding="utf-8")
    n = man["descriptions"]
    print(f"{applied} description(s) applied; {n['described']}/{n['total']} categories "
          f"described -> {args.manifest}")
    for e in errors:
        print(f"  REJECT {e}", file=sys.stderr)
    for c in man["categories"]:
        if c.get("category_description"):
            print(f"  {c['category']}: {c['category_description']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
