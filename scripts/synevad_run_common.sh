#!/usr/bin/env bash
# Shared setup for the synevad run scripts. Sourced, not executed.
#
# GPU hygiene: synevad workers use physical cuda:{id}. Do not export
# CUDA_VISIBLE_DEVICES alongside --gpus or later ids will fail the availability check.
# Multi-GPU requires MLFLOW_TRACKING_URI (a tracking server; SQLite is refused).

synevad_run_setup() {
    REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    cd "$REPO"
    export PYTHONPATH="$REPO"
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
    unset CUDA_VISIBLE_DEVICES || true

    if [[ -f "$REPO/.env" ]]; then
        set -a
        # shellcheck disable=SC1091
        source "$REPO/.env"
        set +a
    fi

    PYTHON=(uv run python)
    GPUS="${GPUS:-0,1,2,3}"
    ANALYZE_OUT="${ANALYZE_OUT:-outputs/analysis}"
    SELECT_BY=(fixed image_auroc@all oracle)
}

synevad_build_split() {
    # PatchCore trains on a few frames per seed here; every other official frame is free
    # to carry a synthetic defect. That tree is separate from the FLUX/DRAEM split in
    # .env's MVTEC_PATH. The sweep file sizes the pools, so the split cannot drift from
    # the seeds and shot counts the grid asks for.
    local sweep=${1:-${SWEEP:-}}
    MVTEC_POOLS="${MVTEC_SPLIT_PATH:-$REPO/data/synevad/MVTec_pools}"
    export MVTEC_PATH="$MVTEC_POOLS"
    if [[ -z "${MVTEC_SRC:-}" ]]; then
        echo "MVTEC_SRC is required to build the pooled train split" >&2
        exit 1
    fi
    local split_args=(
        --mvtec-src "$MVTEC_SRC"
        --mvtec "$MVTEC_PATH"
    )
    if [[ -n "$sweep" ]]; then
        split_args+=(--sweep "$sweep")
    fi
    # A split that must match one cut earlier is replayed, not re-derived:
    # MVTEC_POOLS_RECORD=<tree> passes --pools. See scripts/make_split_pools.py.
    if [[ -n "${MVTEC_POOLS_RECORD:-}" ]]; then
        split_args+=(--pools "$MVTEC_POOLS_RECORD")
    fi
    echo "[split] $MVTEC_SRC -> $MVTEC_PATH (per-seed train pools, not the FLUX tree)"
    "${PYTHON[@]}" scripts/make_split_pools.py "${split_args[@]}"
}

synevad_run_trap() {
    local label=$1
    _synevad_cleanup() {
        echo "[$label] interrupted — stopping children" >&2
        trap - INT TERM
        kill -- -$$ 2>/dev/null || true
        exit 130
    }
    trap _synevad_cleanup INT TERM
}

synevad_require_gpus() {
    if [[ -z "${GPUS:-}" ]]; then
        echo "GPUS is empty" >&2
        exit 1
    fi
    case "$GPUS" in
        *,*)
            if [[ -z "${MLFLOW_TRACKING_URI:-}" ]]; then
                echo "multi-GPU sweeps need MLFLOW_TRACKING_URI (MLflow tracking server)" >&2
                exit 1
            fi
            ;;
    esac
}

synevad_sweep_and_analyze() {
    local config=$1
    local sweep=$2
    local version=$3
    local experiment=$4

    echo "[synevad] experiment=$experiment version=$version gpus=$GPUS"
    "${PYTHON[@]}" -m synevad \
        --config "$config" \
        --sweep "$sweep" \
        --gpus "$GPUS" \
        --experiment "$experiment"

    echo "[analyze] experiment=$experiment version=$version -> $ANALYZE_OUT"
    "${PYTHON[@]}" scripts/analyze_proxy.py \
        --version "$version" \
        --experiment "$experiment" \
        --config "$config" \
        --out "$ANALYZE_OUT" \
        --select-by "${SELECT_BY[@]}"
}
