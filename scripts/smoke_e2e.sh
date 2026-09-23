#!/usr/bin/env bash
# End-to-end smoke: split -> generate -> sweep -> analyse -> check, on two categories and
# a deliberately tiny grid. It runs the real code paths with real models; what makes it a
# smoke is the size of the grid, not a stub anywhere.
#
# About 10 minutes on one GPU at the defaults: ~9 for the 64 FLUX edits and their masks,
# ~80s for the 16 PatchCore fits, seconds for the analysis.
#
#   bash scripts/smoke_e2e.sh                     # everything
#   bash scripts/smoke_e2e.sh --preflight         # validate env + configs, run nothing
#   ARM=draem bash scripts/smoke_e2e.sh           # DRAEM instead of FLUX
#   SKIP_GENERATE=1 bash scripts/smoke_e2e.sh     # reuse the corpus from a previous run
#
# Everything it writes lands under $SMOKE_ROOT (default .smoke/, gitignored): its own
# split trees, its own corpus, its own MLflow store, its own analysis output. It never
# touches $MVTEC_PATH, $MVTEC_SPLIT_PATH, $MLFLOW_DIR or any real corpus, so it is safe to
# run beside real work.
#
# Stages, each skippable and each logged to $SMOKE_ROOT/logs/<stage>.log:
#   1 split      scripts/make_split.sh + scripts/make_split_pools.py   SKIP_SPLIT
#   2 generate   the arm's generator into $SMOKE_ROOT/bench            SKIP_GENERATE
#   3 sweep      python -m synevad over a 4-config x 2-seed grid        SKIP_SWEEP
#   4 analyse    scripts/analyze_proxy.py                              SKIP_ANALYSE
#   5 check      assert the analysis carries a usable result           (always)
#
# Settings:
#   ARM          flux | draem                                          [flux]
#   CATEGORIES   two or more; the fixed-config baseline holds one out  [bottle,screw]
#   BACKBONES    PatchCore backbones; a dinov3_* here also exercises
#                the adapter in synevad/backbones.py                    [resnet18,wide_resnet50_2]
#   SEEDS        >=2, or the seed noise floor cannot be measured       [42,6800]
#   SHOTS        data.num_train_samples                                [2]
#   PER_CELL     generated rows per (mode x severity) cell             [2]
#   LIMIT        source frames per category                            [2]
#   GPUS         generation GPUs; the sweep runs single-process        [0]
#   MVTEC_SRC    official MVTec root                                   [$MVTEC_SRC]
#   DTD_PATH     Describable Textures root, ARM=draem only             [$DTD_PATH]
#   SMOKE_ROOT   where everything lands                                [.smoke]
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
if [[ -f .env ]]; then set -a; source .env; set +a; fi

PREFLIGHT_ONLY=0
[[ "${1:-}" == "--preflight" ]] && PREFLIGHT_ONLY=1

ARM="${ARM:-flux}"
CATEGORIES="${CATEGORIES:-bottle,screw}"
BACKBONES="${BACKBONES:-resnet18,wide_resnet50_2}"
SEEDS="${SEEDS:-42,6800}"
SHOTS="${SHOTS:-2}"
PER_CELL="${PER_CELL:-2}"
LIMIT="${LIMIT:-2}"
GPUS="${GPUS:-0}"
SMOKE_ROOT="${SMOKE_ROOT:-$REPO/.smoke}"
PYTHON="${PYTHON:-uv run python}"
VERSION=smoke
EXPERIMENT=smoke

# The smoke's own trees. Exported, because the configs read them through ${oc.env:...} —
# this is what keeps the run off the real corpora and out of the real tracking store.
export MVTEC_PATH="$SMOKE_ROOT/split/MVTec"
export MVTEC_SPLIT_PATH="$SMOKE_ROOT/split/MVTec_pools"
export MLFLOW_DIR="$SMOKE_ROOT/mlflow"
export FLUX_ABLATE_BENCH="$SMOKE_ROOT/bench/flux"
export DRAEM_BENCH="$SMOKE_ROOT/bench/draem"
# Point the store at the smoke's own SQLite file. **Set, never unset**: `synevad.db.logging`
# calls `load_dotenv()` at import, which leaves an existing variable alone but repopulates
# one a caller deliberately cleared (the comment at synevad/db/logging.py:13 says so), so
# `unset MLFLOW_TRACKING_URI` here would silently hand the smoke the tracking server from
# .env and write its runs into the real store. A server also buys nothing: it exists to
# serialize the concurrent writers of a multi-GPU sweep, and this one is single-process.
mkdir -p "$MLFLOW_DIR"
export MLFLOW_TRACKING_URI="sqlite:///$MLFLOW_DIR/mlflow.db"
# Only meaningful when reading a server's artifacts off its own filesystem. Empty is not a
# directory, so `synevad.db.read.artifact_path` treats it as unset — and an *exported* empty
# value is what stops load_dotenv() putting the real one back.
export MLFLOW_LOCAL_ARTIFACT_ROOT=""

LOGS="$SMOKE_ROOT/logs"
CONFIG_OUT="$SMOKE_ROOT/config"
ANALYSIS_OUT="$SMOKE_ROOT/analysis"

case "$ARM" in
    flux)  BASE_CONFIG=synevad/config/ablate_split.yaml; BENCH="$FLUX_ABLATE_BENCH" ;;
    draem) BASE_CONFIG=synevad/config/draem_split.yaml;  BENCH="$DRAEM_BENCH" ;;
    *) echo "ARM must be flux or draem, got: $ARM" >&2; exit 2 ;;
esac
SMOKE_CONFIG="$CONFIG_OUT/${ARM}_smoke.yaml"
SMOKE_SWEEP="$CONFIG_OUT/${ARM}_smoke_sweep.yaml"

step() { printf '\n==== [%s] %s ====\n' "$(date +%H:%M:%S)" "$*"; }
die()  { echo "smoke: $*" >&2; exit 1; }
run_logged() {  # run_logged <logname> <cmd...>
    local log="$LOGS/$1.log"; shift
    echo "[cmd] $*" | tee "$log"
    "$@" 2>&1 | tee -a "$log"
    return "${PIPESTATUS[0]}"
}

# --- 0 preflight ----------------------------------------------------------------------
step "0 preflight"

n_cats=$(awk -F, '{print NF}' <<< "$CATEGORIES")
(( n_cats >= 2 )) || die "CATEGORIES needs at least 2 (the fixed-config baseline holds one
  out and the leave-one-out query choice needs another); got: $CATEGORIES"
n_seeds=$(awk -F, '{print NF}' <<< "$SEEDS")
(( n_seeds >= 2 )) || die "SEEDS needs at least 2, or replicate_noise has nothing to
  measure and every population comes back with decidable unknown; got: $SEEDS"
# analyze_proxy's --min-n is 3: a population with fewer models gets no statistic. The grid
# below varies backbone x coreset ratio within a (category, seed, shots) cell.
n_models=$(( $(awk -F, '{print NF}' <<< "$BACKBONES") * 2 ))
(( n_models >= 3 )) || die "BACKBONES x 2 coreset ratios must be >= 3 models per
  population (analyze_proxy --min-n); got $n_models from BACKBONES=$BACKBONES"

[[ -n "${MVTEC_SRC:-}" ]] || die "set MVTEC_SRC to the official MVTec root"
[[ -d "$MVTEC_SRC" ]]     || die "MVTEC_SRC is not a directory: $MVTEC_SRC"
for cat in ${CATEGORIES//,/ }; do
    [[ -d "$MVTEC_SRC/$cat/train/good" ]] || die "no $cat/train/good under $MVTEC_SRC"
done
if [[ "$ARM" == flux ]]; then
    for cat in ${CATEGORIES//,/ }; do
        [[ -f "prompts/standalone/$cat.json" ]] || die "no prompts/standalone/$cat.json"
    done
elif [[ "$ARM" == draem ]]; then
    [[ -n "${DTD_PATH:-}" && -d "${DTD_PATH:-}/images" ]] \
        || die "ARM=draem needs DTD_PATH pointing at a root containing images/<class>/*.jpg"
fi

mkdir -p "$LOGS" "$CONFIG_OUT"

# The eval config is derived from the real one rather than duplicated, so the smoke cannot
# drift from what the arms actually run. Exactly one gate is relaxed, and the run prints
# which, so nothing is quietly different.
$PYTHON - "$BASE_CONFIG" "$SMOKE_CONFIG" "$PER_CELL" <<'CFGEOF'
import sys
from omegaconf import OmegaConf

base, out, per_cell = sys.argv[1], sys.argv[2], int(sys.argv[3])
cfg = OmegaConf.load(base)

# min_num_defects marks a run validity=error below its threshold. The real arms see 60
# defects; the smoke sees per_cell rows per (mode x severity) cell, so a severity-sliced
# query holds per_cell of them. `max_defects` is deliberately left alone: it caps at 60,
# the smoke generates fewer, and `synevad.data.synevad` returns every row below the cap —
# while `params.max_defects` interpolates it, so removing it breaks the config.
cfg.min_num_defects = max(1, min(per_cell, 4))

OmegaConf.save(cfg, out, resolve=False)
print(f"wrote {out}  (min_num_defects={cfg.min_num_defects}, everything else as {base})")
CFGEOF

# 2 coreset ratios x the backbones gives >= 3 models per population; seeds and shots are
# population keys, not competitors, so they multiply runs without widening the choice.
cat > "$SMOKE_SWEEP" <<EOF
category: [$(sed 's/,/, /g' <<< "$CATEGORIES")]
version: [$VERSION]
model.backbone: [$(sed 's/,/, /g' <<< "$BACKBONES")]
model.layers: [default]
data.image_size: [[256, 256]]
model.embed_dimension_factor: [1.0]
model.coreset_sampling_ratio: [0.1, 0.25]
data.num_train_samples: [$SHOTS]
seed: [$(sed 's/,/, /g' <<< "$SEEDS")]
EOF
echo "wrote $SMOKE_SWEEP"

n_runs=$(( n_cats * n_models * n_seeds ))
echo
echo "arm=$ARM  categories=$CATEGORIES  models/population=$n_models  seeds=$n_seeds"
echo "runs=$n_runs  corpus=$BENCH  analysis=$ANALYSIS_OUT/$VERSION"

# Resolved the way synevad resolves it, i.e. after load_dotenv() has had its say. If this
# prints anything but the smoke's own sqlite file, the run would write to a real store.
resolved=$($PYTHON -c 'from synevad.db import tracking_uri_from_env; print(tracking_uri_from_env())')
echo "store=$resolved"
case "$resolved" in
    "sqlite:///$MLFLOW_DIR/mlflow.db") ;;
    *) die "the tracking store resolved to $resolved, not the smoke's own sqlite file.
  Refusing to run: this would write into a real MLflow store." ;;
esac

if (( PREFLIGHT_ONLY )); then
    echo
    echo "preflight only — nothing run. Drop --preflight for the real thing."
    exit 0
fi

# --- 1 split --------------------------------------------------------------------------
if [[ "${SKIP_SPLIT:-0}" != 1 ]]; then
    step "1 split -> $MVTEC_PATH and $MVTEC_SPLIT_PATH"
    run_logged split_gen env SRC="$MVTEC_SRC" SPLIT_ROOT="$MVTEC_PATH" N=10 \
        bash scripts/make_split.sh
    run_logged split_pools $PYTHON scripts/make_split_pools.py \
        --mvtec-src "$MVTEC_SRC" --mvtec "$MVTEC_SPLIT_PATH" \
        --categories "$CATEGORIES" --sweep "$SMOKE_SWEEP"
fi

# --- 2 generate -----------------------------------------------------------------------
if [[ "${SKIP_GENERATE:-0}" != 1 ]]; then
    step "2 generate ($ARM) -> $BENCH"
    if [[ "$ARM" == flux ]]; then
        # --skip-smoke: the generator's own pre-flight edits one image per category
        # before the real run, to catch a bad config early. That is what stage 0 and this
        # whole script already are, so here it would just double the FLUX time.
        run_logged generate $PYTHON scripts/generate_standalone_sweep.py \
            --config synevad/synthesis/configs/standalone_sweep.yaml \
            --sources-root "$MVTEC_PATH" --source-subdir source \
            --prompts-dir prompts/standalone --skip-smoke \
            --out-root "$BENCH" \
            --categories "$CATEGORIES" --limit "$LIMIT" --per-cell "$PER_CELL" --waves 1 \
            --run-tag "$VERSION" --gpus "$GPUS" --mask-gpu "${GPUS%%,*}"
    else
        run_logged generate $PYTHON scripts/generate_draem.py \
            --mvtec "$MVTEC_PATH" --dtd "$DTD_PATH" --out "$BENCH" \
            --categories "$CATEGORIES" --limit "$LIMIT" --n-per-source 8
    fi
fi

for cat in ${CATEGORIES//,/ }; do
    [[ -d "$BENCH/$cat/images" ]] || die "stage 2 produced no $BENCH/$cat/images"
done

# --- 3 sweep --------------------------------------------------------------------------
if [[ "${SKIP_SWEEP:-0}" != 1 ]]; then
    step "3 sweep ($n_runs PatchCore fits, single process) -> $MLFLOW_DIR"
    run_logged sweep $PYTHON -m synevad \
        --config "$SMOKE_CONFIG" --sweep "$SMOKE_SWEEP" --experiment "$EXPERIMENT"
fi

# --- 4 analyse ------------------------------------------------------------------------
if [[ "${SKIP_ANALYSE:-0}" != 1 ]]; then
    step "4 analyse -> $ANALYSIS_OUT/$VERSION"
    run_logged analyse $PYTHON scripts/analyze_proxy.py \
        --version "$VERSION" --experiment "$EXPERIMENT" --config "$SMOKE_CONFIG" \
        --out "$ANALYSIS_OUT" \
        --select-by fixed image_auroc@all image_auroc@minimal oracle
fi

# --- 5 check --------------------------------------------------------------------------
step "5 check"
$PYTHON scripts/check_pipeline.py "$ANALYSIS_OUT/$VERSION" --expect-runs "$n_runs"

echo
echo "smoke passed. Output under $SMOKE_ROOT; remove it with: rm -rf $SMOKE_ROOT"
