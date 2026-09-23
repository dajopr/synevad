"""Persist a DRAEM-style synthetic eval corpus in Synevad's generation-manifest form.

Overlays Perlin-masked DTD textures onto the gen-split sources (``<cat>/source``,
built by ``scripts/make_split.sh``), measures each native composite's mask, and writes
``generation_manifest_draem.jsonl`` plus ``composited/`` / ``mask_full/`` so
``make_synevad_dataset`` can load the set without a FLUX path.

The 50% no-anomaly skip in official DRAEM is training noise and is not copied: empty
Perlin is retried a few times, then kept (the eval gate drops an empty mask). Train
images are never sources.

DTD (Describable Textures Dataset)::

    wget https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz
    tar xf dtd-r1.0.1.tar.gz
    # DTD_PATH should contain images/<class>/*.jpg

Example::

    DTD_PATH=/path/to/dtd \\
      python scripts/generate_draem.py --out data/bench/draem

    python -m synevad --config synevad/config/draem_split.yaml --sweep synevad/config/sweep_draem_split.yaml

    python scripts/analyze_proxy.py --version v3_draem \\
      --config synevad/config/draem_split.yaml \\
      --select-by fixed image_auroc@all oracle
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from synevad.synthesis.inputs import load_sources
from synevad.synthesis.manifest import write_manifest
from synevad.data.draem import (
    CLASS_NAME,
    SEVERITY,
    augment_nonzero,
    list_dtd_textures,
    overlay_rng,
)
from synevad.metrics.area import area_fields

DEFAULT_OUT = os.environ.get("DRAEM_BENCH", "data/bench/draem")
DEFAULT_N_PER_SOURCE = 16
DTD_URL = "https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--mvtec",
        default=os.environ.get("MVTEC_PATH", "data/synevad/MVTec"),
        help="gen-split root (default: $MVTEC_PATH or data/synevad/MVTec)",
    )
    ap.add_argument(
        "--dtd",
        default=os.environ.get("DTD_PATH"),
        help="DTD root containing images/<class>/*.jpg (default: $DTD_PATH)",
    )
    ap.add_argument("--out", default=DEFAULT_OUT, help=f"corpus root (default: {DEFAULT_OUT})")
    ap.add_argument(
        "--n-per-source",
        type=int,
        default=DEFAULT_N_PER_SOURCE,
        help=f"overlays per gen-split source (default: {DEFAULT_N_PER_SOURCE})",
    )
    ap.add_argument("--seed", type=int, default=0, help="corpus seed (default: 0)")
    ap.add_argument("--categories", default=None, help="comma-separated category filter")
    ap.add_argument("--limit", type=int, default=None, help="cap sources per category")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="print the job count without writing images",
    )
    return ap.parse_args(argv)


def discover_categories(root: Path, wanted: set[str] | None) -> list[str]:
    names = sorted(
        p.name
        for p in root.iterdir()
        if p.is_dir() and (p / "source").is_dir()
    )
    if wanted is not None:
        missing = sorted(wanted - set(names))
        if missing:
            raise SystemExit(f"no source/ tree for: {', '.join(missing)}")
        names = [n for n in names if n in wanted]
    if not names:
        raise SystemExit(f"no <cat>/source directories under {root}")
    return names


def generate_one(
    source: Path,
    *,
    category: str,
    overlay_i: int,
    seed: int,
    dtd_paths: list[Path],
    out_images: Path,
) -> dict:
    """One overlay: persist RGB + mask, return a Synevad manifest row."""
    source_id = source.stem
    image_id = f"{source_id}__draem{overlay_i:02d}"
    rng = overlay_rng(seed, category, source_id, overlay_i)
    with Image.open(source) as im:
        image = np.asarray(im.convert("RGB"))
    blended, binary, meta = augment_nonzero(image, rng=rng, dtd_paths=dtd_paths)

    composited_dir = out_images / "composited"
    mask_dir = out_images / "mask_full"
    composited_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    blended_path = composited_dir / f"{image_id}.png"
    mask_path = mask_dir / f"{image_id}.png"
    Image.fromarray(blended).save(blended_path)
    Image.fromarray(binary, mode="L").save(mask_path)

    mask_ok = meta.n_mask > 0
    row = {
        "image_id": image_id,
        "source_image_id": source_id,
        "roi_id": source_id,
        "generation_id": "draem",
        "category": category,
        "class": CLASS_NAME,
        "severity": SEVERITY,
        "prompt_id": CLASS_NAME,
        "prompt": "draem perlin+dtd overlay",
        "seed": int(seed),
        "overlay_index": int(overlay_i),
        "attempt_index": int(meta.attempts),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "scorer": None,
        "reward_score": None,
        "reward_edit_score": None,
        "reward_noop_score": None,
        "mask_status": "ok" if mask_ok else "no_defect",
        "mask_coverage": float(meta.n_mask) / float(binary.size),
        "draem_texture": meta.texture_path,
        "draem_perlin_scalex": meta.perlin_scalex,
        "draem_perlin_scaley": meta.perlin_scaley,
        "draem_beta": round(meta.beta, 6),
        **area_fields(binary),
        "full_blended_path": str(blended_path.resolve()),
        "mask_full_path": str(mask_path.resolve()),
    }
    return row


def generate_category(
    *,
    category: str,
    sources: list[Path],
    dtd_paths: list[Path],
    out_root: Path,
    n_per_source: int,
    seed: int,
    dry_run: bool = False,
) -> list[dict]:
    out_images = out_root / category / "images"
    rows: list[dict] = []
    for source in sources:
        for overlay_i in range(n_per_source):
            if dry_run:
                rows.append(
                    {
                        "image_id": f"{source.stem}__draem{overlay_i:02d}",
                        "source_image_id": source.stem,
                        "category": category,
                    }
                )
                continue
            rows.append(
                generate_one(
                    source,
                    category=category,
                    overlay_i=overlay_i,
                    seed=seed,
                    dtd_paths=dtd_paths,
                    out_images=out_images,
                )
            )
    if not dry_run:
        out_images.mkdir(parents=True, exist_ok=True)
        write_manifest(out_images / "generation_manifest_draem.jsonl", rows)
    return rows


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.n_per_source < 1:
        raise SystemExit("--n-per-source must be at least 1")
    if not args.dtd:
        raise SystemExit(
            "DTD path is required (--dtd or $DTD_PATH). Download with\n"
            f"  wget {DTD_URL}\n"
            "and point DTD_PATH at the extracted tree (the directory that contains "
            "images/<class>/*.jpg)."
        )

    mvtec = Path(args.mvtec).expanduser()
    wanted = (
        {part.strip() for part in args.categories.split(",") if part.strip()}
        if args.categories
        else None
    )
    categories = discover_categories(mvtec, wanted)
    dtd_paths = list_dtd_textures(args.dtd)
    out_root = Path(args.out).expanduser()

    total = 0
    for category in categories:
        sources = load_sources(str(mvtec / category / "source"), limit=args.limit)
        print(
            f"{category}: {len(sources)} sources × {args.n_per_source} "
            f"= {len(sources) * args.n_per_source} overlays"
        )
        rows = generate_category(
            category=category,
            sources=sources,
            dtd_paths=dtd_paths,
            out_root=out_root,
            n_per_source=args.n_per_source,
            seed=args.seed,
            dry_run=args.dry_run,
        )
        total += len(rows)
    suffix = " (dry-run)" if args.dry_run else ""
    print(f"{total} rows{suffix}")


if __name__ == "__main__":
    main()
