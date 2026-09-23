# synevad — severity-graded synthetic anomalies as a model-selection proxy

Picking an anomaly detector for a new production line means choosing a backbone, a
resolution and a coreset ratio *before* you have defects to validate on. This repository
generates graded synthetic defects on the defect-free frames you do have, ranks candidate
detectors on them, and measures how much real performance that choice gives up.

The headline quantity is **choice regret**: the real test score of the model the synthetic
arm picked, against the real test score of the best model available. Regret 0 means the
proxy chose as well as an oracle with the real test set in hand.

Two synthetic arms are implemented and are scored over the same models:

| Arm | Defects from | Entry point |
|---|---|---|
| **FLUX** (ours) | instruction-edited by FLUX.2, four severity grades per defect mode, masks estimated from the edit residual | `scripts/generate_standalone_sweep.py` |
| **DRAEM** | Perlin noise × Describable-Textures overlays, frozen once rather than sampled per epoch | `scripts/generate_draem.py` |

The detector is [PatchCore](https://github.com/amazon-science/patchcore-inspection),
installed unmodified as a dependency, with DINOv2/DINOv3 patch tokens available as
backbones through an adapter (`synevad/backbones.py`) that leaves the upstream package
untouched.

## Install

```bash
uv sync
cp .env_template .env    # then edit: every path in it is yours
```

`uv sync` pulls PatchCore and EditReward from git at pinned commits. Python ≥ 3.12, and a
CUDA GPU for anything that generates or sweeps. The analysis stage is CPU-only and needs
nothing but pandas and an MLflow store.

## The two splits, and why there are two

A frame spent on synthetic generation must never also train the detector: if PatchCore
fits on the exact background a synthetic defect sits on, its memory bank has already seen
that frame and every synthetic score is optimistic. Both scripts below enforce that as a
filesystem fact — symlink trees — rather than as a filter some code path could skip.

```bash
# 1. the generation split: first N frames per category -> <cat>/source, the rest train
MVTEC_SRC=/path/to/MVTec bash scripts/make_split.sh

# 2. the pooled split the sweeps train on: a few frames per seed in train/good,
#    every other frame free to carry a defect, never the first 10 (step 1's sources)
python scripts/make_split_pools.py --sweep synevad/config/sweep_ablate_split.yaml
```

Both arms train on the **pooled** split, so their real arms score the same models and the
two regrets are directly comparable.

Which frames land in which pool depends on the order frames are ranked in, so a pooled
split is not guaranteed to reproduce one cut earlier under a different ranking. To
reproduce an existing split exactly, replay its record instead of re-deriving it:

```bash
python scripts/make_split_pools.py --pools <tree containing <cat>/train_pools.json>
```

## Does it work? — the end-to-end smoke

Before committing a GPU-week to a full sweep, run the whole chain small:

```bash
MVTEC_SRC=/path/to/MVTec bash scripts/smoke_e2e.sh
```

It cuts both splits, generates 64 defects, fits 16 PatchCore models over a 4-config x
2-seed grid on two categories, runs the proxy analysis, and then asserts the result is
usable. Same code paths and same models as a real run — only the grid is small. About ten
minutes on one GPU, most of it FLUX. `--preflight` validates the environment and writes
the generated configs without running anything heavy.

Everything lands under `.smoke/` (gitignored): its own split trees, corpus, MLflow store
and analysis output, so it never touches a real corpus or tracking store. Stages are
skippable (`SKIP_GENERATE=1`, ...) and each is logged to `.smoke/logs/`. `ARM=draem` runs
the DRAEM arm instead.

The last stage is [`scripts/check_pipeline.py`](scripts/check_pipeline.py), which is worth
running on its own against any analysis directory:

```bash
python scripts/check_pipeline.py outputs/analysis/v3_ablate
```

It exits non-zero unless every table is present and non-empty, both arms carry finite
scores, and **at least one population produced a finite `regret`** — which can only happen
if the sweep trained several models, the synthetic arm ranked them and the real arm scored
them, so one finite value means the whole chain ran. It also catches the two mis-sized
runs that otherwise fail silently: one category (no fixed-config baseline) and one seed
(no noise floor).

## The two experiments

### 1. Which query — the severity grades (`v1`)

The main sweep: 15 MVTec categories x 4 backbones x 2 layer sets x 2 resolutions x 2
embedding factors x 2 coreset ratios x 4 shot counts x 3 seeds, one PatchCore fit each,
with the FLUX corpus sliced by severity grade. It asks which slice of the synthetic set
you should rank models on — all of it, or only the `minimal` defects, or only the
`severe` ones. It trains on the `scripts/make_split.sh` tree.

```bash
python scripts/generate_standalone_sweep.py \
    --config synevad/synthesis/configs/standalone_sweep.yaml \
    --sources-root data/synevad/MVTec --source-subdir source \
    --prompts-dir prompts/standalone --out-root "$SYNTHETIC_BENCH"

python -m synevad --config synevad/config/mvtec.yaml --sweep synevad/config/sweep.yaml \
    --gpus 0,1,2,3
python scripts/analyze_proxy.py --version v1 --config synevad/config/mvtec.yaml \
    --select-by fixed image_auroc@minimal image_auroc@slight \
                image_auroc@moderate image_auroc@severe image_auroc@all oracle
```

`synevad/config/mvtec.yaml` defines the queries those rules name: `all` and the four
severity grades.

### 2. Which generator — FLUX against DRAEM (`v3_ablate`, `v3_draem`)

Both arms on the **pooled** split, capped at `max_defects: 60` each, so they are scored
over the same models and the same number of anomalies and their regrets are directly
comparable. `run_ablations.sh` is both arms end to end against one freshly cut split:

```bash
GPUS=0,1,2,3 bash run_ablations.sh
```

Expanded, per arm:

```bash
# FLUX — the v3 corpus
python scripts/generate_standalone_sweep.py \
    --config synevad/synthesis/configs/standalone_sweep.yaml \
    --sources-root data/synevad/MVTec --source-subdir source \
    --prompts-dir prompts/standalone --out-root "$FLUX_ABLATE_BENCH"
bash scripts/run_ablate_split.sh          # sweep + analyse, version v3_ablate

# DRAEM
DTD_PATH=... python scripts/generate_draem.py --out "$DRAEM_BENCH"
bash scripts/run_draem_split.sh           # sweep + analyse, version v3_draem
```

Each `run_*.sh` runs `python -m synevad` over the sweep grid and then
`scripts/analyze_proxy.py`. Multi-GPU sweeps need `MLFLOW_TRACKING_URI` pointing at a
tracking server; SQLite is refused, because parallel workers cannot share it.

Resume identity is `(tags, params)` and includes neither the synthetic root nor the split,
so each arm has a `version:` of its own in its sweep file — otherwise one arm's runs would
look complete against another's.

## Where the numbers are

Nothing here renders a table. `scripts/analyze_proxy.py` writes
`outputs/analysis/<version>/` and prints the selection summary to stdout; the frames are
Parquet, and these are the ones that carry the claims:

| File | One row per | Read for |
|---|---|---|
| `selection.parquet` | population | `regret`, `skill`, `regret_fixed`, `beats_fixed`, and the `decidable` flag |
| `selected.parquet` | population × selection rule | the real `image_auroc` / `aupro` / `pixel_auroc` of the model each rule picked |
| `selected_summary.parquet` | category × rule × metric | the same, averaged — the results table |
| `correlation.parquet` | population | whether the two arms co-vary at all (r / rho / tau) |
| `calibration.parquet` | category × severity × scorer | the signed gap `synth − real`, and whether its sign survives the spread across models |
| `paired.parquet` | model × set × query × metric | the real and synthetic scores everything above is derived from |

A **population** is one (metric, set, query, category, seed, num_train_samples) cell — a
selection made among models that saw the same training images. Read `decidable` first: it
is False when the real scores span less than one seed's worth of noise, and every other
number in the row is then an artefact, because there was no decision to get right.

The selection rules worth comparing are `fixed` (the best single configuration across
categories — what you get for free with no synthetic data), `image_auroc@<query>` (picked
on the synthetic arm), and `oracle` (the best model on the real test set). The module
docstring of `scripts/analyze_proxy.py` is the full manual.

## Cropped high-resolution frames

MVTec frames are ~1024 px square, so one edit covers the whole frame. Frames from a fixed
inspection camera are often several thousand pixels on a side, where resampling the whole
frame down to the editor's resolution shrinks every hairline crack and pit past the point
the editor can draw one.

`synevad/synthesis/configs/cropped_example.yaml` is the config for that case: fixed windows, one per
region of interest, each edited at native resolution with its own prompts, masks and
manifest rows. It carries the mask-extent floors and negative-control settings that a
cropped dark-metal corpus needs, with notes on which of them must be retuned first.
Window centres go in `synevad/synthesis/configs/category_overrides_cropped.yaml`.

## Optional: EditReward scoring

`scripts/score_manifests.py` scores each generated edit with EditReward and writes
`reward_score` / `scorer` into the manifests. Nothing in the two arms above depends on
it; set `EDITREWARD_CHECKPOINT` and `EDITREWARD_CONFIG` if you want it.

## Layout

Everything importable is one package; `scripts/` holds the entry points and no logic worth
reusing.

```
synevad/
  synthesis/   generation: FLUX editing, mask estimation from the edit residual,
    configs/   blending, and the inpainted negative controls — plus the YAML driving it
  data/        dataset adapters, the two splits, the DRAEM overlay corpus
  eval/        the PatchCore fit-and-score loop
  metrics/     per-run image / pixel / ordinal metrics
  analysis/    correlation, selection and calibration over a finished sweep
  config/      eval + sweep YAML, one pair per arm
  db/          MLflow read and write
scripts/       entry points: generate, sweep, analyse, check
prompts/standalone/   one JSON per category, 4 defect modes x 4 severity grades
```

The two halves meet at the generation manifest: `synevad.synthesis` writes it, `synevad.data`
reads it. Only two imports cross the other way, both lazy: `synevad.metrics.area`, which
measures each composite's mask as it is made, and `synevad.backbones.DINO_MODELS`, so the
mask estimator and the detector agree on what a DINO backbone name means.

## Further reading

The code is the documentation. The two places worth reading first:

- `synevad/synthesis/configs/standalone_sweep.yaml` — every knob of the `masks:` block with
  the reasoning and the measurement behind its value.
- `scripts/analyze_proxy.py`'s module docstring — the populations, what each output frame
  holds, and the order the columns of `combined.parquet` have to be read in.

## Licence

PatchCore is Apache-2.0 and is consumed as an upstream dependency, not vendored.
