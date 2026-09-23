# Run manifest schema — `defect-taxonomy-batch-manifest/v1`

Written by `scripts/scan_categories.py`, read by the dispatcher and by
`scripts/validate_run.py`. It is the run's only state file: statuses are
updated in place as categories complete, so an interrupted batch resumes from
it with `--resume`.

```json
{
  "schema": "defect-taxonomy-batch-manifest/v1",
  "generated_at": "2026-08-18T09:12:00+00:00",
  "root": "/data/mvtec_ad",
  "benchmark": "mvtec_ad",
  "layout": "benchmark_category_subdirs",
  "grading_mode": "standalone",
  "severity_scale": ["minimal", "slight", "moderate", "severe"],
  "output_root": "/work/taxonomies",
  "descriptions": {"described": 1, "total": 15, "policy": "..."},
  "sampling": {"strategy": "first", "per_category": 6, "staged": true,
               "staged_dir_name": "samples"},
  "categories": [
    {
      "category": "hazelnut",
      "category_description": "whole hazelnut in its woody shell",
      "category_dir": "/data/mvtec_ad/hazelnut",
      "sample_source": "train/good",
      "defect_free_image_count": 391,
      "sample_selection": {"strategy": "first", "requested": 6, "selected": 6},
      "samples_dir": "/work/taxonomies/hazelnut/samples",
      "sample_images": ["/work/taxonomies/hazelnut/samples/00_000.png", "..."],
      "source_images": ["/data/mvtec_ad/hazelnut/train/good/000.png", "..."],
      "image_stats": {"probe": "ok", "width": 1024, "height": 1024,
                      "mode": "RGB", "uniform_size": true, "uniform_mode": true},
      "excluded_branches": {"dirs": 5, "images": 110},
      "outputs": {
        "taxonomy": "/work/taxonomies/hazelnut/taxonomy.json",
        "prompts": "/work/taxonomies/hazelnut/prompts.json"
      },
      "status": "pending",
      "calls": [
        {"step": 1, "skill": "standalone-defect-taxonomy-derivation",
         "args": {"benchmark": "mvtec_ad", "category": "hazelnut",
                  "category_description": "whole hazelnut in its woody shell",
                  "defect_free_samples": ["/work/taxonomies/hazelnut/samples/00_000.png"],
                  "grading_mode": "standalone",
                  "severity_scale": ["minimal", "slight", "moderate", "severe"],
                  "output": "/work/taxonomies/hazelnut/taxonomy.json"}},
        {"step": 2, "skill": "standalone-defect-prompt-compiler",
         "args": {"taxonomy": "/work/taxonomies/hazelnut/taxonomy.json",
                  "output": "/work/taxonomies/hazelnut/prompts.json"},
         "depends_on": 1}
      ]
    }
  ],
  "plan": {"categories_found": 15, "categories_to_run": 15, "skill_calls": 30,
           "expected_prompts_range": [180, 300], "note": "..."},
  "firewall": {"defect_labelled_branches_excluded": true,
               "names_omitted": "excluded directory names are counted, never recorded"},
  "warnings": [{"category": "grid", "issue": "...", "action": "emitted"}]
}
```

## Field notes

- `category` is the dataset directory name and the only key that addresses a
  category: output paths, `--categories`, resume, status, the dispatch prompt and
  the taxonomy's own `category` field all use it. It is never rewritten.
- `category_description` is optional prose grounding — what the object actually
  is — passed to the derivation as `calls[0].args.category_description`. The
  field is **absent**, not null or empty, when a category has none, so an agent
  is never handed a blank to fill. Only `set_descriptions.py` writes it, and it
  validates before writing.
- `descriptions.described` counts categories carrying one. Zero means the
  directory names were judged self-explanatory, not that the step was skipped.
- `sample_source` is relative to `category_dir`, so the manifest never contains
  an absolute path into a split it did not select.
- `excluded_branches` is deliberately name-free: counts only. A non-zero `dirs`
  value means defect-labelled branches exist and were skipped — that is the
  firewall working, not a problem to investigate by opening them.
- `sample_images` point at the **staged copies** under `samples_dir`, not at the
  dataset. They are what `calls[0].args.defect_free_samples` carries, so a
  dispatched agent never needs a path into the source tree. `source_images` keeps
  the originals for provenance only — do not pass those to a child skill.
- Selection is deterministic: the defect-free pool is sorted by filename and the
  first `per_category` entries are taken (`strategy: "first"`), or evenly spaced
  ones under `--spaced`. A re-scan of the same root yields the same manifest.
- Staged filenames are prefixed `00_`, `01_`, … in selection order, so the
  `samples/` folder is self-describing even if the source names were not.
- `samples_dir` is `null` under `--no-stage`; `sample_images` then point into the
  source split and the run depends on that tree staying reachable.
- A staged file that differs in size from its source is **kept**, not
  overwritten, and raises a warning — assume it was swapped deliberately.
  `--restage` forces the copy.
- `image_stats` probes the first three samples with Pillow. `uniform_size` /
  `uniform_mode` false is a signal that the acquisition setup varies within the
  category, which weakens the `fixed_instruction_block` — worth a note in the plan.
- `status` ∈ `pending` → `taxonomy_done` → `complete`, plus `failed`. Only the
  dispatcher writes `complete`/`failed`; `--resume` derives the first three from
  which artifacts exist on disk.
- `category_note` (optional, added by the orchestrator in step 2 of the
  procedure) carries a visual correction — e.g. `"directory name 'pill' shows a
  pressed tablet, not a capsule"` — and is passed verbatim into the dispatch
  prompt.
- `calls[].args` mirrors the child skills' input contracts. `defect_free_samples`
  is the only image input either skill receives.
- `expected_prompts_range` is an estimate (3–5 modes × 4 grades). The true count
  comes from `run_report.md` after validation — report that number, not this one.
