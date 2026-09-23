#!/usr/bin/env bash
# Both synthetic arms end to end: build the shared split once, then sweep + analyse each.
#
#   bash run_ablations.sh
#   GPUS=1,2,4,5 bash run_ablations.sh
#   SKIP_SPLIT=1 bash run_ablations.sh          # the split is already cut
#
# The split is cut once here and both arms train on it, so their real arms score the same
# models and the two regrets are comparable. Cutting it again with other seeds or shot
# counts lists other train/good frames — to match a split cut earlier, replay its record:
#
#   MVTEC_POOLS_RECORD=<tree with <cat>/train_pools.json> bash run_ablations.sh
#
# Do not export CUDA_VISIBLE_DEVICES; --gpus uses physical device ids.
set -euo pipefail

# shellcheck source=scripts/synevad_run_common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scripts/synevad_run_common.sh"
synevad_run_setup

export GPUS="${GPUS:-1,2,4,5}"

if [[ "${SKIP_SPLIT:-0}" != 1 ]]; then
    # Sized from one arm's grid; both sweeps share seeds and num_train_samples.
    synevad_build_split synevad/config/sweep_ablate_split.yaml
fi

bash scripts/run_draem_split.sh
bash scripts/run_ablate_split.sh
