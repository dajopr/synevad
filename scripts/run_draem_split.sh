#!/usr/bin/env bash
# The DRAEM arm (60 defects per category) run on the pooled split, so DRAEM and FLUX
# (run_ablate_split.sh) train on the same frames and their real arms score the same models.
# The split's pools skip make_split.sh's first 10 frames (make_split_pools.py
# --reserve-first), which are DRAEM's bases, so the overlays pass the contamination check
# on it.
#
#   GPUS=1,2,4,5 bash scripts/run_draem_split.sh
#
# Run it against the split tree the arms trained on. This script never builds the split:
# scripts/make_split_pools.py does, and a rebuild with other seeds or shots lists other
# train/good frames.
#
# VERSION only names what analyze_proxy reads; the tag the runs log comes from the sweep
# file's `version:` list. Change both together.
#
# Do not export CUDA_VISIBLE_DEVICES; --gpus uses physical device ids.
set -euo pipefail

# shellcheck source=scripts/synevad_run_common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/synevad_run_common.sh"
synevad_run_setup
synevad_run_trap run_draem_split
synevad_require_gpus

EXPERIMENT="${EXPERIMENT:-synevad}"
VERSION="${VERSION:-v3_draem}"
CONFIG="${CONFIG:-synevad/config/draem_split.yaml}"
SWEEP="${SWEEP:-synevad/config/sweep_draem_split.yaml}"
# Exported so the config's ${oc.env:...} reads the same tree checked here.
export MVTEC_SPLIT_PATH="${MVTEC_SPLIT_PATH:-data/synevad/MVTec_pools}"
export DRAEM_BENCH="${DRAEM_BENCH:-data/bench/draem}"  # read by synevad/config/draem_split.yaml

if [[ ! -d "$MVTEC_SPLIT_PATH/bottle/train/good" ]]; then
    echo "[run_draem_split] no split at $MVTEC_SPLIT_PATH; build it with scripts/make_split_pools.py" >&2
    exit 1
fi
if [[ ! -d "$DRAEM_BENCH/bottle/images" ]]; then
    echo "[run_draem_split] no DRAEM corpus at $DRAEM_BENCH (scripts/generate_draem.py)" >&2
    exit 1
fi

synevad_sweep_and_analyze "$CONFIG" "$SWEEP" "$VERSION" "$EXPERIMENT"
echo "[run_draem_split] done  experiment=$EXPERIMENT  analysis=$ANALYZE_OUT/$VERSION"
