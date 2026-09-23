#!/usr/bin/env python3
"""Scan a benchmark-layout image root and emit a batch run manifest.

Layout assumed: <root>/<category>/... , where each category directory holds a
defect-free split somewhere beneath it (train/good, Data/Images/Normal, ...).

The scanner is the contamination firewall for the batch pipeline: directories
whose path names encode defect labels (test/<defect>, Anomaly, ground_truth,
...) are excluded from the manifest **by count only** -- their names are never
written into the output, so a downstream agent reading the manifest cannot
learn a benchmark defect label from it.

Usage:
    python scan_categories.py <root> --out run_manifest.json [options]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".JPG", ".PNG"}

# Path components that mark a defect-free split.
NORMAL_TOKENS = {"good", "normal", "ok", "nominal", "pristine", "defect_free",
                 "defectfree", "non_defective", "nondefective", "clean"}

# Path components that mark defective images, masks, or label-bearing branches.
# Anything under one of these is excluded and never named in the manifest.
ANOMALY_TOKENS = {"anomaly", "anomalous", "defect", "defects", "defective", "bad",
                  "ng", "broken", "ground_truth", "groundtruth", "gt", "mask",
                  "masks", "label", "labels", "annotation", "annotations", "fault"}

# "test" is not itself a defect label; test/good is defect-free. Treated as a
# de-prioritised branch rather than an excluded one.
SPLIT_TOKENS = {"test", "val", "validation", "eval"}

SEVERITY_SCALE = ["minimal", "slight", "moderate", "severe"]

BENCHMARK_HINTS = {
    "mvtec_ad": ("mvtec_ad", "mvtecad", "mvtec-ad", "mvtec"),
    "mvtec_loco": ("mvtec_loco", "loco"),
    "visa": ("visa", "vis_a"),
    "mpdd": ("mpdd",),
    "btad": ("btad",),
    "real_iad": ("real_iad", "realiad"),
}


def tokens(rel: Path) -> set[str]:
    return {p.lower() for p in rel.parts}


def is_image(p: Path) -> bool:
    return p.suffix.lower() in {e.lower() for e in IMAGE_EXTS}


def infer_benchmark(root: Path, override: str | None) -> str:
    if override:
        return override
    hay = f"{root.name}/{root.parent.name}".lower()
    for name, hints in BENCHMARK_HINTS.items():
        if any(h in hay for h in hints):
            return name
    return "unknown"


def collect_image_dirs(cat_dir: Path) -> dict[Path, list[Path]]:
    """Map each directory containing images -> sorted list of image files."""
    by_dir: dict[Path, list[Path]] = {}
    for p in sorted(cat_dir.rglob("*")):
        if p.is_file() and is_image(p):
            by_dir.setdefault(p.parent, []).append(p)
    return {d: sorted(v) for d, v in by_dir.items()}


def classify_dirs(cat_dir: Path, by_dir: dict[Path, list[Path]]):
    """Split image dirs into (defect-free candidates, excluded)."""
    candidates: list[tuple[tuple[int, int, str], Path, list[Path]]] = []
    excluded_dirs = 0
    excluded_images = 0

    for d, imgs in by_dir.items():
        rel = d.relative_to(cat_dir)
        tk = tokens(rel)
        if tk & ANOMALY_TOKENS:
            excluded_dirs += 1
            excluded_images += len(imgs)
            continue
        has_normal = bool(tk & NORMAL_TOKENS)
        under_test = bool(tk & SPLIT_TOKENS)
        if not has_normal and under_test:
            # e.g. a bare test/ directory of mixed images -- cannot vouch for it.
            excluded_dirs += 1
            excluded_images += len(imgs)
            continue
        # rank: explicit normal token first, train before test, shallower first
        rank = (0 if has_normal else 1, 1 if under_test else 0, str(rel))
        candidates.append((rank, d, imgs))

    candidates.sort(key=lambda c: c[0])
    return candidates, excluded_dirs, excluded_images


def pick_samples(items: list[Path], k: int, strategy: str) -> list[Path]:
    """Choose k samples from a filename-sorted defect-free pool.

    `first` (default) takes the head of the sorted pool -- reproducible and
    trivially auditable. `spaced` samples at even intervals, which covers more
    of the pose/lighting variation in a long split.
    """
    if k >= len(items):
        return list(items)
    if strategy == "spaced":
        step = len(items) / k
        return [items[int(i * step)] for i in range(k)]
    return items[:k]


def stage_samples(picked: list[Path], dest: Path, restage: bool) -> tuple[list[Path], int]:
    """Copy the chosen samples next to the category's output files.

    Numbered prefixes preserve selection order regardless of source names, so
    the staged folder alone is enough to reproduce the run. Re-scanning is
    silent when the staged copy already matches its source; a staged file that
    has *diverged* from the source is kept (someone swapped a sample on
    purpose) and counted, so the caller can surface it. `restage` overwrites
    unconditionally.

    Returns the staged paths and the number kept despite differing from source.
    """
    dest.mkdir(parents=True, exist_ok=True)
    staged: list[Path] = []
    diverged = 0
    for i, src in enumerate(picked):
        target = dest / f"{i:02d}_{src.name}"
        if restage or not target.exists():
            shutil.copy2(src, target)
        elif target.stat().st_size != src.stat().st_size:
            diverged += 1
        staged.append(target)
    return staged, diverged


def image_stats(paths: list[Path]) -> dict[str, Any]:
    try:
        from PIL import Image  # type: ignore
    except ImportError:
        return {"probe": "unavailable", "note": "Pillow not installed"}
    sizes, modes = [], []
    for p in paths:
        try:
            with Image.open(p) as im:
                sizes.append(im.size)
                modes.append(im.mode)
        except Exception:  # noqa: BLE001 - unreadable sample is a warning, not a crash
            continue
    if not sizes:
        return {"probe": "failed"}
    return {
        "probe": "ok",
        "width": sizes[0][0],
        "height": sizes[0][1],
        "mode": modes[0],
        "uniform_size": len(set(sizes)) == 1,
        "uniform_mode": len(set(modes)) == 1,
    }


def build_calls(category: str, benchmark: str, samples: list[str],
                tax_path: str, prompt_path: str,
                description: str | None = None) -> list[dict[str, Any]]:
    """Step 1 always carries the dataset directory name as `category`.

    `category_description` is optional extra grounding added by
    `set_descriptions.py`; it is omitted entirely when absent, so the derivation
    never sees an empty field it might try to fill.
    """
    return [
        {
            "step": 1,
            "skill": "standalone-defect-taxonomy-derivation",
            "args": {
                "benchmark": benchmark,
                "category": category,
                **({"category_description": description} if description else {}),
                "defect_free_samples": samples,
                "grading_mode": "standalone",
                "severity_scale": SEVERITY_SCALE,
                "output": tax_path,
            },
        },
        {
            "step": 2,
            "skill": "standalone-defect-prompt-compiler",
            "args": {"taxonomy": tax_path, "output": prompt_path},
            "depends_on": 1,
        },
    ]


def load_prior(path: Path) -> dict[str, dict[str, Any]]:
    """Descriptions and notes a human added to a previous manifest at this path.

    A re-scan rewrites the manifest wholesale, so without this a
    `category_description` or a `category_note` would be silently lost the next
    time someone re-runs with --resume. Carried over by category key.
    """
    if not path.exists():
        return {}
    try:
        prior = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return {c["category"]: c for c in prior.get("categories", []) if "category" in c}


def scan(root: Path, out_root: Path, benchmark: str, samples: int,
         only: list[str] | None, resume: bool, strategy: str = "first",
         stage: bool = True, restage: bool = False,
         prior: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    prior = prior or {}
    carried = 0
    warnings: list[dict[str, str]] = []
    categories: list[dict[str, Any]] = []

    cat_dirs = sorted(d for d in root.iterdir() if d.is_dir() and not d.name.startswith("."))
    if only:
        wanted = {c.strip().lower() for c in only}
        missing = wanted - {d.name.lower() for d in cat_dirs}
        for m in sorted(missing):
            warnings.append({"category": m, "issue": "requested category not found under root"})
        cat_dirs = [d for d in cat_dirs if d.name.lower() in wanted]

    for cat_dir in cat_dirs:
        by_dir = collect_image_dirs(cat_dir)
        if not by_dir:
            warnings.append({"category": cat_dir.name, "issue": "no image files found", "action": "skipped"})
            continue

        cands, ex_dirs, ex_imgs = classify_dirs(cat_dir, by_dir)
        if not cands:
            warnings.append({
                "category": cat_dir.name,
                "issue": "no defect-free split could be identified",
                "action": "skipped",
            })
            continue

        _, src_dir, imgs = cands[0]
        pool = sorted(imgs)
        picked = pick_samples(pool, samples, strategy)
        if len(pool) < 3:
            warnings.append({
                "category": cat_dir.name,
                "issue": f"only {len(pool)} defect-free sample(s); derivation expects 3-10",
                "action": "emitted",
            })

        cat_out = out_root / cat_dir.name
        tax_path = str(cat_out / "taxonomy.json")
        prompt_path = str(cat_out / "prompts.json")

        if stage:
            staged, diverged = stage_samples(picked, cat_out / "samples", restage)
            if diverged:
                warnings.append({
                    "category": cat_dir.name,
                    "issue": f"{diverged} staged sample(s) differ from source; kept the staged copies "
                             f"(pass --restage to overwrite)",
                    "action": "emitted",
                })
        else:
            staged = picked

        status = "pending"
        if resume:
            has_tax = Path(tax_path).exists()
            has_prompts = Path(prompt_path).exists()
            status = "complete" if (has_tax and has_prompts) else ("taxonomy_done" if has_tax else "pending")

        sample_strs = [str(p) for p in staged]
        prev = prior.get(cat_dir.name, {})
        description = prev.get("category_description") or None
        if description:
            carried += 1
        entry: dict[str, Any] = {
            "category": cat_dir.name,
            **({"category_description": description} if description else {}),
            "category_dir": str(cat_dir),
            "sample_source": str(src_dir.relative_to(cat_dir)) or ".",
            "defect_free_image_count": len(pool),
            "sample_selection": {"strategy": strategy, "requested": samples, "selected": len(picked)},
            "samples_dir": str(cat_out / "samples") if stage else None,
            "sample_images": sample_strs,
            "source_images": [str(p) for p in picked],
            "image_stats": image_stats(picked[:3]),
            "excluded_branches": {"dirs": ex_dirs, "images": ex_imgs},
            "outputs": {"taxonomy": tax_path, "prompts": prompt_path},
            "status": status,
            "calls": build_calls(cat_dir.name, benchmark, sample_strs, tax_path,
                                 prompt_path, description),
        }
        if prev.get("category_note"):
            entry["category_note"] = prev["category_note"]
        categories.append(entry)

    todo = [c for c in categories if c["status"] != "complete"]
    remaining_calls = sum(1 if c["status"] == "taxonomy_done" else 2 for c in todo)
    return {
        "schema": "defect-taxonomy-batch-manifest/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "root": str(root),
        "benchmark": benchmark,
        "layout": "benchmark_category_subdirs",
        "grading_mode": "standalone",
        "severity_scale": SEVERITY_SCALE,
        "output_root": str(out_root),
        "descriptions": {
            "described": carried,
            "total": len(categories),
            "policy": "category is always the dataset directory name; "
                      "category_description is optional extra grounding sent alongside it",
        },
        "sampling": {
            "strategy": strategy,
            "per_category": samples,
            "staged": stage,
            "staged_dir_name": "samples",
        },
        "categories": categories,
        "plan": {
            "categories_found": len(categories),
            "categories_to_run": len(todo),
            "skill_calls": remaining_calls,
            "expected_prompts_range": [12 * len(todo), 20 * len(todo)],
            "note": "3-5 modes x 4 grades per category; exact count known after derivation",
        },
        "firewall": {
            "defect_labelled_branches_excluded": True,
            "names_omitted": "excluded directory names are counted, never recorded",
        },
        "warnings": warnings,
    }


def render_plan_md(man: dict[str, Any]) -> str:
    lines = [
        f"# Batch run plan — {man['benchmark']}",
        "",
        f"Root: `{man['root']}`  ",
        f"Output: `{man['output_root']}`  ",
        f"Samples: {man['sampling']['per_category']} per category "
        f"({man['sampling']['strategy']}), "
        + (f"staged to `<category>/{man['sampling']['staged_dir_name']}/`  "
           if man['sampling']['staged'] else "read in place from the source split  "),
        f"Categories: {man['plan']['categories_found']} found, "
        f"{man['plan']['categories_to_run']} to run — {man['plan']['skill_calls']} skill calls",
        "",
        "| # | category | description | samples | defect-free pool | status |",
        "|---|---|---|---|---|---|",
    ]
    for i, c in enumerate(man["categories"], 1):
        shown = c.get("category_description") or "—"
        lines.append(f"| {i} | {c['category']} | {shown} | {len(c['sample_images'])} | "
                     f"{c['defect_free_image_count']} | {c['status']} |")
    note = ("Each category's samples sit next to its outputs, so a run needs "
            "nothing further from the source tree."
            if man["sampling"]["staged"] else
            "Samples are referenced in place (`--no-stage`); the source tree must stay "
            "reachable for the whole run.")
    lines += ["", note, "", "## Call list", ""]
    for c in man["categories"]:
        if c["status"] == "complete":
            lines.append(f"- ~~{c['category']}~~ — already complete, skipped")
            continue
        lines.append(f"- **{c['category']}**")
        for call in c["calls"]:
            if c["status"] == "taxonomy_done" and call["step"] == 1:
                lines.append(f"  {call['step']}. ~~`{call['skill']}`~~ — taxonomy exists")
                continue
            # Look `output` up by key: a later --describe pass appends
            # category_description to args, so the last value is not the path.
            args = call["args"]
            target = args.get("output") or list(args.values())[-1]
            lines.append(f"  {call['step']}. `{call['skill']}` → `{target}`")
    if man["warnings"]:
        lines += ["", "## Warnings", ""]
        lines += [f"- **{w.get('category','-')}**: {w['issue']}" for w in man["warnings"]]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="benchmark root containing category subdirectories")
    ap.add_argument("--out", type=Path, default=Path("run_manifest.json"))
    ap.add_argument("--plan-md", type=Path, default=None, help="also write a human-readable plan")
    ap.add_argument("--output-root", type=Path, default=Path("taxonomies"),
                    help="where per-category taxonomy.json / prompts.json will be written")
    ap.add_argument("--benchmark", default=None, help="override inferred benchmark name")
    ap.add_argument("--samples", type=int, default=6, help="defect-free samples per category (3-10)")
    ap.add_argument("--spaced", action="store_true",
                    help="sample evenly across the split instead of taking the first N")
    ap.add_argument("--no-stage", action="store_true",
                    help="do not copy samples into <output-root>/<category>/samples/")
    ap.add_argument("--restage", action="store_true",
                    help="overwrite staged samples that already exist")
    ap.add_argument("--categories", default=None, help="comma-separated subset to include")
    ap.add_argument("--resume", action="store_true", help="mark categories with existing outputs complete")
    args = ap.parse_args()

    if not args.root.is_dir():
        print(f"error: {args.root} is not a directory", file=sys.stderr)
        return 2
    if not 3 <= args.samples <= 10:
        print("error: --samples must be between 3 and 10", file=sys.stderr)
        return 2

    benchmark = infer_benchmark(args.root, args.benchmark)
    prior = load_prior(args.out)
    man = scan(
        root=args.root.resolve(),
        out_root=args.output_root.resolve(),
        benchmark=benchmark,
        samples=args.samples,
        only=args.categories.split(",") if args.categories else None,
        resume=args.resume,
        strategy="spaced" if args.spaced else "first",
        stage=not args.no_stage,
        restage=args.restage,
        prior=prior,
    )
    for target in (args.out, args.plan_md):
        if target and target.parent:
            target.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(man, indent=2), encoding="utf-8")
    if args.plan_md:
        args.plan_md.write_text(render_plan_md(man), encoding="utf-8")

    p = man["plan"]
    if man["descriptions"]["described"]:
        print(f"  carried over {man['descriptions']['described']} category description(s) "
              f"from the previous manifest")
    staged_note = ("" if args.no_stage else
                   f", {sum(len(c['sample_images']) for c in man['categories'])} samples staged")
    print(f"{benchmark}: {p['categories_found']} categories, {p['categories_to_run']} to run, "
          f"{p['skill_calls']} skill calls{staged_note}, {len(man['warnings'])} warnings -> {args.out}")
    if benchmark == "unknown":
        print("  note: benchmark name not inferable from the path; pass --benchmark", file=sys.stderr)
    for w in man["warnings"]:
        print(f"  WARN {w.get('category','-')}: {w['issue']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
