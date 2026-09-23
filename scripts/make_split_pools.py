"""Cut the pooled train/eval split: a few frames per seed train, every other frame is a base.

``scripts/make_split.sh`` puts the first N official frames in ``source/`` and everything
else in ``train/good``. That is right when only a handful of frames carry a synthetic
defect, and wrong when nearly every official frame is a candidate base — holding every
base out leaves nothing to train on.

This writes a *separate* symlink tree (``data/synevad/MVTec_pools`` by default) the other
way round: ``--train-frames`` frames per swept seed go to ``train/good`` (the sweep never
draws more than 8), every other official frame goes to ``source/`` and can carry a
synthetic defect. The pools are disjoint per seed and recorded in ``train_pools.json``,
so the swept seeds still train on different images. The FLUX/DRAEM split from
``make_split.sh`` is not modified.

The pools never take the first ``--reserve-first`` frames (default 10), the ones
``scripts/make_split.sh`` spends on FLUX/DRAEM generation. Those corpora then pass the
contamination check here too, so they can be scored on the same models.

    python scripts/make_split_pools.py
    python scripts/make_split_pools.py --sweep synevad/config/sweep_ablate_split.yaml
    python scripts/make_split_pools.py --train-frames 8 --seeds 42,6800,9999

**Reproducing an existing split.** Which frames land in which pool depends on the order
frames are ranked in, so a re-derived split is not guaranteed to match one cut earlier
under a different ranking. ``--pools DIR`` replays a recorded split instead of deriving
one: point it at a tree (or a directory of ``<category>/train_pools.json`` files) and the
frames are taken from the record, exactly.

    python scripts/make_split_pools.py --pools data/synevad/MVTec_pools
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from omegaconf import OmegaConf

from synevad.data.split_pools import (
    DEFAULT_RESERVED_FRAMES,
    DEFAULT_SEEDS,
    DEFAULT_TRAIN_FRAMES,
    TRAIN_POOLS_FILENAME,
    SplitInfo,
    choose_train_pools,
    load_train_pools,
    write_split,
)

REPO = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT = os.environ.get(
    "MVTEC_SPLIT_PATH", str(REPO / "data/synevad/MVTec_pools")
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--mvtec-src",
        default=os.environ.get("MVTEC_SRC"),
        help="official MVTec root (default: $MVTEC_SRC)",
    )
    ap.add_argument(
        "--mvtec",
        default=DEFAULT_SPLIT,
        help="where to write the split "
        f"(default: $MVTEC_SPLIT_PATH or {DEFAULT_SPLIT})",
    )
    ap.add_argument(
        "--pools",
        default=None,
        help="replay a recorded split: a directory holding <category>/"
        f"{TRAIN_POOLS_FILENAME}. Frames are taken from the record rather than "
        "re-derived, so the result matches the recorded split exactly. Ignores "
        "--train-frames / --seeds / --reserve-first.",
    )
    ap.add_argument("--categories", default=None, help="comma-separated category filter")
    add_pool_args(ap)
    return ap.parse_args(argv)


def add_pool_args(ap: argparse.ArgumentParser) -> None:
    """The knobs that size ``train/good``."""
    ap.add_argument(
        "--train-frames",
        type=int,
        default=None,
        help="official frames per seed in train/good (default: --sweep's largest "
        f"data.num_train_samples, else {DEFAULT_TRAIN_FRAMES}). Every other frame "
        "becomes a synthetic base.",
    )
    ap.add_argument(
        "--seeds",
        default=None,
        help="comma-separated seeds to cut disjoint pools for; must cover the seeds you "
        "sweep or the run fails (default: --sweep's seeds, else "
        f"{','.join(str(s) for s in DEFAULT_SEEDS)})",
    )
    ap.add_argument(
        "--reserve-first",
        type=int,
        default=DEFAULT_RESERVED_FRAMES,
        help="leading official frames never put in train/good, because make_split.sh "
        "spends them on FLUX/DRAEM generation; match its N "
        f"(default: {DEFAULT_RESERVED_FRAMES})",
    )
    ap.add_argument(
        "--sweep",
        default=None,
        help="sweep config to read `seed` and `data.num_train_samples` from, so the "
        "split cannot drift from the grid it is cut for "
        "(e.g. synevad/config/sweep_ablate_split.yaml)",
    )


def pool_plan(args: argparse.Namespace) -> tuple[int, tuple[int, ...]]:
    """``(frames per seed, seeds)`` — explicit flags win over ``--sweep``."""
    n_frames, seeds = None, None
    if args.sweep:
        sweep = OmegaConf.to_container(OmegaConf.load(args.sweep), resolve=False) or {}
        assert isinstance(sweep, dict)
        shots = sweep.get("data.num_train_samples")
        if shots:
            n_frames = max(int(shot) for shot in shots)
        swept_seeds = sweep.get("seed")
        if swept_seeds:
            seeds = tuple(int(seed) for seed in swept_seeds)
    if args.train_frames is not None:
        n_frames = int(args.train_frames)
    if args.seeds:
        seeds = tuple(int(part) for part in args.seeds.split(",") if part.strip())
    return n_frames or DEFAULT_TRAIN_FRAMES, seeds or DEFAULT_SEEDS


def discover_categories(src_root: Path, wanted: set[str] | None) -> list[str]:
    """Categories laid out as ``<cat>/train/good`` under the official root."""
    found = sorted(
        p.name for p in src_root.iterdir() if (p / "train" / "good").is_dir()
    )
    if wanted is None:
        return found
    missing = sorted(wanted - set(found))
    if missing:
        raise SystemExit(f"no train/good under {src_root} for: {', '.join(missing)}")
    return [cat for cat in found if cat in wanted]


def build_split(
    *,
    src_root: Path,
    split_root: Path,
    category: str,
    n_frames: int,
    seeds: tuple[int, ...],
    reserved: int,
    recorded: dict[int, tuple[str, ...]] | None = None,
) -> SplitInfo:
    """Pick this category's per-seed pools and lay down its split tree."""
    pools = recorded
    if pools is None:
        pools = choose_train_pools(
            src_root / category / "train" / "good",
            n_frames=n_frames,
            seeds=seeds,
            reserved=reserved,
        )
    return write_split(
        src_root=src_root,
        split_root=split_root,
        category=category,
        pools=pools,
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.mvtec_src:
        raise SystemExit("pass --mvtec-src or set $MVTEC_SRC")
    src_root = Path(args.mvtec_src).expanduser()
    if not src_root.is_dir():
        raise SystemExit(f"official MVTec root is not a directory: {src_root}")
    wanted = (
        {part.strip() for part in args.categories.split(",") if part.strip()}
        if args.categories
        else None
    )
    categories = discover_categories(src_root, wanted)
    if not categories:
        raise SystemExit(f"no <cat>/train/good under {src_root}")

    record_root = Path(args.pools).expanduser() if args.pools else None
    split_root = Path(args.mvtec).expanduser()
    n_frames, seeds = pool_plan(args)
    if record_root is None:
        print(
            f"pools: {n_frames} frames x seeds {','.join(str(s) for s in seeds)} "
            f"-> {n_frames * len(seeds)} frames per category in train/good"
        )
    else:
        print(f"pools: replayed from {record_root}/<category>/{TRAIN_POOLS_FILENAME}")

    for category in categories:
        recorded = None
        if record_root is not None:
            recorded = load_train_pools(record_root / category)
            if recorded is None:
                raise SystemExit(
                    f"{record_root / category / TRAIN_POOLS_FILENAME} is missing — "
                    "--pools needs a record for every category being built"
                )
        try:
            split = build_split(
                src_root=src_root,
                split_root=split_root,
                category=category,
                n_frames=n_frames,
                seeds=seeds,
                reserved=args.reserve_first,
                recorded=recorded,
            )
        except (FileNotFoundError, ValueError) as exc:
            raise SystemExit(str(exc)) from exc
        print(f"{category} | source={split.n_source} train={split.n_train}")


if __name__ == "__main__":
    main()
