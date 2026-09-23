#!/usr/bin/env bash
# Build the train/gen split as symlink trees, so a frame spent on synthetic generation can
# never also train the detector. Synthetic anomalies are edited onto clean *train* frames;
# if PatchCore also fits on those frames its memory bank has already seen the exact
# background each synthetic defect sits on and every synthetic score is optimistic.
#
#   data/synevad/MVTec/<cat>/source/              -> the first N train/good frames (generation sources)
#   data/synevad/MVTec/<cat>/train/good/          -> every OTHER train frame (what the detector fits on)
#   data/synevad/MVTec/<cat>/{test,ground_truth}  -> the real dirs, unfiltered
#
# The split is thus a filesystem fact, not a filter in the data pipeline: MVTEC_PATH points at
# data/synevad/MVTec (.env) and generation reads <cat>/source (scripts/generate_standalone_sweep.py), so no code path
# can reach a contaminated frame. source/ sits beside train/ rather than inside it because the
# dataset adapter only ever iterates <cat>/train and <cat>/test.
#
# Re-runnable — each category's trees are rebuilt from scratch:
#
#   bash scripts/make_split.sh          # N=10 sources per category, from $MVTEC_SRC
#   N=20 bash scripts/make_split.sh
#
# Raising N is always safe; LOWERING it after generating puts already-used frames back into
# training, so re-check with the contamination test (synevad.data.assert_no_train_contamination,
# which every synthetic dataloader runs anyway).
#
# Any benchmark laid out as <root>/<cat>/{train/good,test,ground_truth} splits the same way —
# override SRC and SPLIT_ROOT:
#
#   SRC=/path/to/MPDD SPLIT_ROOT=$PWD/data/synevad/MPDD bash scripts/make_split.sh
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SRC=${SRC:-${MVTEC_SRC:?set MVTEC_SRC (or pass SRC=) to the official MVTec root}}
SPLIT_ROOT=${SPLIT_ROOT:-$REPO/data/synevad/MVTec}
N=${N:-10}

if [ ! -d "$SRC" ]; then
    echo "source root is not a directory: $SRC" >&2
    exit 1
fi

for cat_dir in "$SRC"/*/; do
    cat_dir=${cat_dir%/}
    cat=$(basename "$cat_dir")
    [ -d "$cat_dir/train/good" ] || continue

    gen_dir=$SPLIT_ROOT/$cat/source
    train_dir=$SPLIT_ROOT/$cat/train/good
    rm -rf "${SPLIT_ROOT:?}/$cat"
    mkdir -p "$gen_dir" "$train_dir"

    # Sorted (glob order): the first N frames are spent on generation, the rest train.
    i=0
    for img in "$cat_dir"/train/good/*; do
        if [ "$i" -lt "$N" ]; then
            ln -s "$img" "$gen_dir/$(basename "$img")"
        else
            ln -s "$img" "$train_dir/$(basename "$img")"
        fi
        i=$((i + 1))
    done

    # Nothing is held out of test/ or ground_truth/ — link the directories whole.
    for sub in test ground_truth; do
        if [ -d "$cat_dir/$sub" ]; then
            ln -s "$cat_dir/$sub" "$SPLIT_ROOT/$cat/$sub"
        fi
    done

    echo "$cat | gen=$(ls "$gen_dir" | wc -l) train=$(ls "$train_dir" | wc -l) total=$i"
done
