"""Cut a train/eval split by what the sweep trains on, not by what the generator edited.

``scripts/make_split.sh`` puts the first N official frames in ``source/`` and everything
else in ``train/good``. That is the right shape when only a handful of frames carry a
synthetic defect. It is the wrong shape when nearly every official frame is a candidate
base: holding every base out leaves nothing to train on, so the eval set collapses to
whatever the generator happened not to touch.

This module writes the split the other way round. ``--train-frames`` frames per swept seed
go to ``train/good``; every *other* official frame goes to ``source/`` and is available as
a synthetic base. The pools are disjoint per seed and recorded in ``train_pools.json``, so
the swept seeds still train on different images and :func:`train_pool_for_seed` can hand a
run exactly its own block.

Two properties make the arms comparable:

* The first ``reserved`` frames in sorted order are never pooled. Those are
  ``scripts/make_split.sh``'s generation sources, so a corpus edited from them still
  passes the contamination check on this tree.
* ``counts`` lets a caller rank frames by what spending one costs — a frame carrying many
  edits is expensive to train on, because every edit onto it has to leave the eval set.
  With no counts every non-reserved frame is equally good and the order is the filename's.
  **The ordering is part of the split**: two callers who disagree about it cut different
  pools, so a split that must match a previous one is replayed from its recorded
  ``train_pools.json`` (``--pools``) rather than re-derived.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}

# Frames per seed in ``train/good``, and the seeds they are cut for. 8 is the largest
# `data.num_train_samples` the sweeps ask for, so a bigger pool would spend frames no run
# can train on; the seed list has to match the sweep's or `train_pool_for_seed` refuses
# the run.
DEFAULT_TRAIN_FRAMES = 8
DEFAULT_SEEDS: tuple[int, ...] = (42, 6800, 9999)
# The FLUX/DRAEM corpora are edited from the first N sorted frames (``scripts/make_split.sh``,
# N=10). Keeping those out of the pools lets those corpora pass the contamination check on
# this split too, so every arm can be scored against the same models.
DEFAULT_RESERVED_FRAMES = 10
TRAIN_POOLS_FILENAME = "train_pools.json"
# This split and the tree from make_split.sh have the same shape
# (``<cat>/{source,train/good}`` symlinks), and rebuilding a category wipes it first, so
# the root carries a marker rather than trusting whatever path was passed in.
SPLIT_MARKER_FILENAME = ".synevad_split"


@dataclass(frozen=True)
class SplitInfo:
    """What :func:`write_split` laid down for one category."""

    category: str
    n_source: int
    n_train: int
    # seed -> the frames that seed trains on (disjoint between seeds)
    pools: dict[int, tuple[str, ...]]
    # Edits that have to leave the eval set because their base frame ended up in a
    # training pool. Zero unless the caller passed ``counts``.
    spent_edits: int


def choose_train_pools(
    src_train: Path,
    *,
    counts: Mapping[str, tuple[int, int]] | None = None,
    n_frames: int = DEFAULT_TRAIN_FRAMES,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    reserved: int = DEFAULT_RESERVED_FRAMES,
) -> dict[int, tuple[str, ...]]:
    """One disjoint block of ``n_frames`` official frames per seed, cheapest first.

    ``counts`` maps a frame stem to ``(edits worth keeping, edits in total)``. Spending a
    frame on training costs the edits made onto it, so frames are ranked by how many they
    carry — the kept count first, then the total, then the filename for a stable order —
    and dealt round-robin, which keeps the pools equally cheap instead of loading the
    whole cost onto the last seed. With no counts every frame costs ``(0, 0)`` and the
    order is the filename's. Disjoint blocks are what keep the swept seeds from collapsing
    onto one training set once the pool is this small.

    The first ``reserved`` frames in sorted order are never pooled: they are
    ``scripts/make_split.sh``'s generation sources, and training on them would fail the
    FLUX/DRAEM corpora's contamination check.
    """
    if n_frames < 1:
        raise ValueError(f"n_frames must be >= 1, got {n_frames}")
    if reserved < 0:
        raise ValueError(f"reserved must be >= 0, got {reserved}")
    ordered_seeds = list(dict.fromkeys(int(seed) for seed in seeds))
    if not ordered_seeds:
        raise ValueError("no seeds to build training pools for")
    frames = sorted(
        p
        for p in Path(src_train).iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )[reserved:]
    needed = n_frames * len(ordered_seeds)
    if len(frames) < needed:
        raise ValueError(
            f"{src_train} has {len(frames)} frames after reserving the first {reserved}, "
            f"need {needed} ({n_frames} per seed x {len(ordered_seeds)} seeds)"
        )
    counts = {} if counts is None else counts
    cheapest = sorted(frames, key=lambda p: (*counts.get(p.stem, (0, 0)), p.name))
    pools: dict[int, list[str]] = {seed: [] for seed in ordered_seeds}
    for i, path in enumerate(cheapest[:needed]):
        pools[ordered_seeds[i % len(ordered_seeds)]].append(path.name)
    return {seed: tuple(sorted(names)) for seed, names in pools.items()}


def claim_split_root(split_root: Path) -> Path:
    """Mark ``split_root`` as ours, refusing someone else's tree.

    Rebuilding a category deletes it first, and the FLUX/DRAEM split from
    ``scripts/make_split.sh`` looks exactly like this one from the outside, so an unmarked
    non-empty root is treated as not ours. Silently rebuilding ``$MVTEC_PATH`` would
    re-cut the split every other arm trains on.
    """
    split_root = Path(split_root)
    marker = split_root / SPLIT_MARKER_FILENAME
    if marker.is_file():
        return marker
    if split_root.exists() and any(split_root.iterdir()):
        raise ValueError(
            f"{split_root} has no {SPLIT_MARKER_FILENAME} marker, so it is not a tree "
            "this writer laid down — refusing to rebuild it. The FLUX/DRAEM split from "
            "scripts/make_split.sh must not be rebuilt as a pooled split; point --mvtec "
            f"at $MVTEC_SPLIT_PATH instead. If this really is the pooled tree, "
            f"`touch {marker}`."
        )
    split_root.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        "Pooled train/eval split (scripts/make_split_pools.py). Symlinks only; "
        "rebuilt in place.\n",
        encoding="utf-8",
    )
    return marker


def write_split(
    *,
    src_root: Path,
    split_root: Path,
    category: str,
    pools: Mapping[int, Sequence[str]],
    counts: Mapping[str, tuple[int, int]] | None = None,
) -> SplitInfo:
    """Rebuild ``split_root/<cat>`` with only the training pools in ``train/good``.

    The inverse of the FLUX split. ``train/good`` is just the union of the per-seed pools
    — 8 frames a seed — and *every other* official frame goes to ``source/``, where it is
    available as a synthetic base.

    ``train_pools.json`` beside them records which frames belong to which seed, so
    ``make_mvtecad_dataset`` draws that seed's block and never another's. ``test/`` and
    ``ground_truth/`` are directory symlinks into ``src_root``; the FLUX/DRAEM tree from
    ``scripts/make_split.sh`` is untouched.
    """
    src_root = Path(src_root)
    split_root = Path(split_root)
    src_cat = src_root / category
    src_train = src_cat / "train" / "good"
    if not src_train.is_dir():
        raise FileNotFoundError(f"no official train/good at {src_train}")

    claim_split_root(split_root)

    pool_names = {name for names in pools.values() for name in names}
    if not pool_names:
        raise ValueError(f"{category}: empty training pools")
    if sum(len(names) for names in pools.values()) != len(pool_names):
        raise ValueError(f"{category}: training pools overlap: {pools}")

    cat_root = split_root / category
    if cat_root.is_symlink() or cat_root.is_file():
        cat_root.unlink()
    elif cat_root.is_dir():
        shutil.rmtree(cat_root)

    gen_dir = cat_root / "source"
    train_dir = cat_root / "train" / "good"
    gen_dir.mkdir(parents=True)
    train_dir.mkdir(parents=True)

    n_source = n_train = 0
    unseen = set(pool_names)
    for img in sorted(src_train.iterdir()):
        if not img.is_file() or img.suffix.lower() not in IMAGE_EXTS:
            continue
        if img.name in pool_names:
            unseen.discard(img.name)
            dest_dir = train_dir
            n_train += 1
        else:
            dest_dir = gen_dir
            n_source += 1
        (dest_dir / img.name).symlink_to(img.resolve())
    if unseen:
        raise ValueError(
            f"{category}: pool frames are not in {src_train}: {sorted(unseen)[:10]}"
        )

    for sub in ("test", "ground_truth"):
        src_sub = src_cat / sub
        if src_sub.exists():
            (cat_root / sub).symlink_to(src_sub.resolve())

    write_train_pools(cat_root, pools)
    counts = {} if counts is None else counts
    spent = sum(counts.get(Path(name).stem, (0, 0))[0] for name in pool_names)
    return SplitInfo(
        category=category,
        n_source=n_source,
        n_train=n_train,
        pools={int(seed): tuple(names) for seed, names in pools.items()},
        spent_edits=spent,
    )


def write_train_pools(cat_root: Path, pools: Mapping[int, Sequence[str]]) -> Path:
    """Record the per-seed training pools next to the split's ``train/`` tree."""
    path = Path(cat_root) / TRAIN_POOLS_FILENAME
    payload = {
        "n_train": min(len(names) for names in pools.values()),
        "pools": {str(int(seed)): list(names) for seed, names in sorted(pools.items())},
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def load_train_pools(cat_root: Path) -> dict[int, tuple[str, ...]] | None:
    """The per-seed training pools written by :func:`write_split`, or None.

    None means an ordinary MVTec tree (the FLUX/DRAEM split), where ``train/good`` is
    the pool and the seed only picks the draw.
    """
    path = Path(cat_root) / TRAIN_POOLS_FILENAME
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        int(seed): tuple(names) for seed, names in (payload.get("pools") or {}).items()
    }


def train_pool_for_seed(cat_root: Path, seed: int | None) -> tuple[str, ...] | None:
    """The frames seed ``seed`` may train on, or None outside a pooled split.

    An unknown seed is an error rather than a fallback: silently handing it another
    seed's pool would make two swept seeds train on the same images, which is the one
    thing the disjoint pools exist to prevent.
    """
    pools = load_train_pools(cat_root)
    if pools is None:
        return None
    if seed is not None and int(seed) in pools:
        return pools[int(seed)]
    wanted = sorted(pools) if seed is None else sorted({*pools, int(seed)})
    raise ValueError(
        f"{Path(cat_root) / TRAIN_POOLS_FILENAME} has no training pool for seed {seed} "
        f"(pools: {sorted(pools)}). Rebuild the split for the seeds you sweep: "
        "python scripts/make_split_pools.py --seeds "
        + ",".join(str(s) for s in wanted)
    )
