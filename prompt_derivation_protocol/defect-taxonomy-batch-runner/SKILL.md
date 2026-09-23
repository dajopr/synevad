---
name: defect-taxonomy-batch-runner
description: "Scan a folder or dataset root of images, identify every object category in it, and emit an ordered list of `standalone-defect-taxonomy-derivation` + `standalone-defect-prompt-compiler` calls covering all of them — then optionally execute that plan autonomously, one category at a time, and validate the results. Use whenever the user points at a benchmark root, dataset directory, or image folder and wants taxonomies and/or generation prompts for *all* categories, *every* category, the *whole* dataset, or 'the categories in this folder' rather than one named category. Also use for phrases like batch/bulk/sweep taxonomy derivation, 'run the taxonomy pipeline over MVTec/VisA', 'plan the calls for each category', or resuming a partially finished batch. For a single already-named category, call `standalone-defect-taxonomy-derivation` directly instead."
---

# Batch Defect Taxonomy Runner

Turn a directory of images into a complete, resumable batch: for each category,
its defect-free sample images staged into the run directory, one standalone
severity taxonomy, and one compiled standalone prompt set.

This skill orchestrates; it derives nothing itself. All defect content comes
from the two child skills:

| Step | Skill | Produces |
|---|---|---|
| 0 | `scripts/scan_categories.py` | `<out>/<category>/samples/` + `run_manifest.json` |
| 0b | `scripts/set_descriptions.py` | optional `category_description` per category |
| 1 | `standalone-defect-taxonomy-derivation` | `<out>/<category>/taxonomy.json` |
| 2 | `standalone-defect-prompt-compiler` | `<out>/<category>/prompts.json` |

Never write taxonomy fields, defect modes, severity descriptors or edit prompts
yourself. If a category's derivation fails, record the failure and move on —
a fabricated taxonomy is worse than a missing one.

## Two modes, selected by flag

| User says | Mode | Behaviour |
|---|---|---|
| "plan", "list the calls", "dry run", "what would you run", `--plan-only` | **plan** | Scan, identify categories, write `run_manifest.json` + `run_plan.md`, stop. Present the table and the call list. |
| "run it", "autonomously", "do all of them", "go", `--run` | **run** | Plan, then dispatch every pending category, then validate and report. |
| anything ambiguous | **plan, then ask** | Produce the plan, show it, ask whether to execute. |

Additional flags the user may pass, all forwarded to `scan_categories.py`:
`--categories a,b,c` (subset), `--samples N` (3–10 per category, default 6),
`--output-root PATH`, `--benchmark NAME`, `--resume` (skip categories whose
artifacts already exist), `--spaced`, `--no-stage`, `--restage` (see *Sample
staging*), `--concurrency N` (dispatch fan-out, default 4).

In an unattended run (scheduled task, user away), default to **run** with
`--resume`, and state the assumption at the top of the output.

## Sample staging

The scan copies each category's **first 6 defect-free images** — sorted by
filename, head of the split — into `<out>/<category>/samples/`, numbered
`00_`, `01_`, … so selection order survives whatever the source names were.
The manifest then points the derivation call at those copies, not at the
source tree.

This is what makes a run self-contained: after the scan, every input the two
child skills need sits beside the files they will write, so a dispatched agent
never has to reach back into the dataset — and, per the firewall below, never
has a reason to.

- `--samples N` changes the count (3–10, the derivation skill's range).
- `--spaced` samples evenly across the split instead of taking the head. The
  head is the default because it is reproducible and trivially auditable;
  reach for `--spaced` when one category's split is long and visibly varies in
  pose or lighting, since the `fixed_instruction_block` is extracted from these
  images and should reflect the whole split.
- Re-scanning is idempotent: staged files that already match their source are
  left alone. A staged file that *differs* from its source is kept and
  reported as a warning — someone swapped a sample deliberately — and
  `--restage` overwrites it.
- `--no-stage` references the source images in place. Only use it when copying
  is impossible (read-only quota, very large images); it makes the run
  dependent on the source tree staying reachable, and re-checks the firewall's
  path discipline on every dispatch.

## Category names and descriptions

The dataset directory name is the category name, everywhere, always:

| Field | Value | Used in |
|---|---|---|
| `category` | the dataset directory name | output folder, manifest key, resume, the dispatch prompt, and the `category` field of the taxonomy the agent writes |
| `category_description` | optional prose you write from the staged samples | passed alongside the name as extra grounding; absent when not written |

Nothing is ever renamed. The reason a description exists at all is that the
derivation skill classifies material family from what it is told and seeds its
literature search with it — `grid` and `metal_nut` are thin on their own, but
`grid` plus "anodized aluminium mesh screen, rigid perforated metal" gives the
derivation something to work with while every artifact still joins back on
`grid`.

Add descriptions with the script, never by hand-editing the manifest:

```bash
python scripts/set_descriptions.py <out>/run_manifest.json --plan-md <out>/run_plan.md \
    --describe grid="anodized aluminium mesh screen, rigid perforated metal" \
    --describe metal_nut="hexagonal steel fastener with threaded bore"
```

It writes `category_description` on the entry and on `calls[0].args`, and
re-renders the plan table. `--clear CATEGORY` removes one. Rules it enforces —
a rejected description is skipped, and `--strict` makes any rejection abort
without writing:

- **No defect terms.** `shell with a crack` is rejected: naming a failure mode
  pre-decides what the taxonomy is supposed to derive independently.
- **No benchmark or split identifiers.** Firewall violation.
- **≤12 words, ≤120 chars.** Describe what the object *is* — material, form,
  finish. How it is manufactured or how it fails is the derivation's job.

Describe only where the directory name is genuinely thin. `bottle` and
`hazelnut` explain themselves, and prose invented from six images is a
hallucination surface at the one input that steers the literature search. When
the samples do not support a confident description, write none — the field is
optional by design, and an absent field is better than a vague one.

A re-scan writing to the same manifest path carries descriptions and
`category_note`s over, so `--resume` never silently drops them.

## Input contract

Benchmark layout — category names are directory names:

```
<root>/
  <category>/
    train/good/*.png        <- defect-free samples come from here
    test/<label>/*.png      <- EXCLUDED, see firewall
    ground_truth/...        <- EXCLUDED
```

VisA-style nesting (`<category>/Data/Images/Normal/…`) and flat
`<category>/*.png` are both handled by the same scanner. If `<root>` has no
category subdirectories — a single flat folder of mixed loose images — stop and
tell the user: this skill covers the benchmark layout only, and guessing
category boundaries from loose files would produce categories the derivation
step cannot ground.

## Contamination firewall

The derivation skill's firewall forbids any use of benchmark defect labels or
defect images. **In a batch run the folder itself is the leak vector**: in
MVTec-style roots the subdirectory names under `test/` *are* the ground-truth
defect labels, and reading them into the plan would contaminate every
downstream call.

Therefore:

1. Run the scanner rather than listing the tree by hand. It excludes every
   branch whose path contains a defect/mask/label token and records only
   **counts**, never names, in the manifest.
2. Do not `ls`, `find`, `tree` or otherwise enumerate anything below
   `<category>/test/`, `ground_truth/`, `Anomaly/`, `masks/`. If you have
   already seen such names in this session, do not pass them on and note it in
   the report.
3. Only images under `<out>/<category>/samples/` may be viewed. That folder is
   defect-free by construction, which is the second reason staging exists: the
   viewable set is a directory, not a rule someone has to remember.
4. Dispatch prompts carry the staged sample paths and the category name —
   nothing else, and no path into the source tree.

The per-category derivation applies its own firewall to literature search; that
is the child skill's job, not this one's.

## Procedure

### 1. Scan

```bash
python scripts/scan_categories.py <root> \
    --out <out>/run_manifest.json --plan-md <out>/run_plan.md \
    --output-root <out> --samples 6 [--resume] [--categories a,b] [--spaced]
```

This writes the manifest and plan **and** stages the samples, so the run
directory is ready to execute the moment the scan returns.

Read `references/manifest_schema.md` if you need the field meanings. Resolve
any warning the scanner emits before dispatching — a category with two
defect-free samples, an unidentifiable split, or a staged sample that diverged
from its source will produce a weak or irreproducible taxonomy.

### 2. Confirm the categories are real, and name them

Directory names are a claim, not a fact. View **one** staged sample per
category (`<out>/<category>/samples/00_*`) and check:

- the name matches what is in the picture (`bottle` really is a bottle);
- whether the directory name alone is a usable literature-search seed, or too
  generic to ground a material classification — if the latter, write a
  `category_description` with `set_descriptions.py` per the section above;
- no two categories are the same object under different names — if they are,
  flag it, since duplicate taxonomies waste a full derivation each;
- the material class is visually plausible, so the derivation starts from a
  sane classification;
- whether all categories share one acquisition setup (same background,
  lighting, framing). If they do, say so in the plan: the derivation skill
  builds the `fixed_instruction_block` once per benchmark in that case.
  Otherwise it is derived per category, which is the default.

Record anything the name cannot carry — "the object is pressed, not moulded",
"two objects per frame" — as a `category_note` on the manifest entry; the
dispatch prompt passes notes through verbatim.

### 3. Present the plan

Show the table from `run_plan.md`: category, description, sample count,
defect-free pool size, status, and the two calls per category. Call out every
description you wrote — prose you invented is the one input the user should
sanity-check before a long run. State the totals — categories,
skill calls, and the expected prompt count (3–5 modes × 4 grades × categories).
In plan mode, stop here.

### 4. Dispatch

One subagent per pending category, `--concurrency` at a time (default 4).
Use the prompt template in `references/dispatch.md` verbatim — it is what keeps
each cold-started agent inside the firewall. Each agent invokes the two skills
in order and writes both artifacts; it returns a one-line status, not the
taxonomy contents. Do not read the full taxonomy JSONs back into this session —
validation is scripted precisely so the orchestrator stays small.

Rules:

- Categories are independent. One failure never blocks another.
- Step 2 depends on step 1 *within* a category only.
- On failure, retry that category once. If it fails again, mark it `failed`
  with the reason and continue.
- After each category completes, update its `status` in `run_manifest.json`
  so an interrupted run resumes cleanly with `--resume`.

### 5. Validate and report

```bash
python scripts/validate_run.py run_manifest.json --report run_report.md
```

This re-checks each artifact independently of the agent that produced it:
standalone grading mode, 3–5 modes, four ordered grades, correct `derivation`
per grade, ≥2 attenuation axes on the attenuated grades, anti-patterns present
as negatives in all four prompts, no incremental phrasing, no numeric
quantities, prompt count equal to 4 × modes, and both `self_check` blocks
passing. Report failures per category and offer to re-run only those.

## Output layout

```
<out>/
  run_manifest.json      # machine state: categories, samples, calls, status
  run_plan.md            # human-readable call list
  run_report.md          # post-run validation
  <category>/            # always the dataset directory name
    samples/             # first 6 defect-free images, staged by the scan
      00_<name>.png
      ...
    taxonomy.json        # from standalone-defect-taxonomy-derivation
    prompts.json         # from standalone-defect-prompt-compiler
```

## Self-check before reporting done

- Every category in the manifest is `complete` or carries a recorded failure
  reason — none silently skipped.
- Every category has its staged samples on disk, in the count the manifest
  claims (`validate_run.py` checks this first).
- Each taxonomy's `category` field is the dataset directory name — never the
  description text (`validate_run.py` checks this and hints when the two were
  swapped).
- Any `category_description` echoed into a taxonomy matches what was sent.
- `run_report.md` exists and every listed category passed, or its failures are
  reproduced in the response.
- No defect label, `test/` subdirectory name, or defective image path appears
  anywhere in the manifest, plan, dispatch prompts, or your response.
- The manifest's `plan.skill_calls` equals the number of calls actually made
  (plus retries), and every artifact path in it exists on disk.
- Totals reported to the user are read from `run_report.md`, not estimated.
