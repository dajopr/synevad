"""Assert that an `analyze_proxy.py` output directory holds a usable result.

`scripts/smoke_e2e.sh` runs this as its last stage, but it is useful on any analysis
directory: it is the difference between "the pipeline ran without raising" and "the
pipeline produced numbers someone could act on".

Each check prints a line and the script exits non-zero if any of them failed, so it works
as a test without a test runner::

    python scripts/check_pipeline.py .smoke/analysis/smoke --expect-runs 16
    python scripts/check_pipeline.py outputs/analysis/v3_ablate

What is checked, and why each one is worth a line of its own:

``frames``
    Every table `analyze_proxy` writes is present and non-empty. A stage that silently
    produced nothing shows up here rather than three steps downstream.
``paired``
    Both arms carry finite scores on the same rows. A `synth` column of all-NaN is the
    signature of a corpus the loader could not match to the runs, which every later table
    would report as "no populations" rather than as an error.
``selection``
    At least one population produced a finite `regret`. This is the end-to-end assertion:
    a regret exists only if the sweep trained several models, the synthetic arm ranked
    them, and the real arm scored them — so one finite value means the whole chain ran.
``fixed baseline``
    `regret_fixed` is finite somewhere, which needs two or more categories: the baseline
    config is chosen on the *other* categories. Catches a run configured with one.
``noise floor``
    `decidable` was computable, which needs two or more seeds. Without it a small regret
    cannot be told from an undecidable comparison, so a run that lost its replicates is
    worth flagging even though nothing raised.
``selectors``
    Every rule asked for appears in `selected.parquet`, so a typo in `--select-by` (which
    is otherwise silently dropped) fails here.

Exit codes: 0 all passed, 1 at least one check failed, 2 the directory does not exist.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Written unconditionally by `analyze_proxy.main`.
REQUIRED_FRAMES: tuple[str, ...] = (
    "paired",
    "correlation",
    "calibration",
    "selection",
    "fixed_baseline",
    "leave_one_out",
    "combined",
    "selected",
    "selected_summary",
)


class Checks:
    """Collects pass/fail lines so every check runs before anything exits."""

    def __init__(self) -> None:
        self.failed = 0

    def __call__(self, name: str, ok: bool, detail: str = "") -> bool:
        mark = "PASS" if ok else "FAIL"
        self.failed += not ok
        print(f"  [{mark}] {name}{f' — {detail}' if detail else ''}")
        return ok


def load(directory: Path, name: str) -> pd.DataFrame | None:
    path = directory / f"{name}.parquet"
    if not path.is_file():
        return None
    return pd.read_parquet(path)


def finite(frame: pd.DataFrame, column: str) -> int:
    """How many rows carry a finite value in `column` (0 if the column is absent)."""
    if column not in frame.columns:
        return 0
    return int(pd.to_numeric(frame[column], errors="coerce").notna().sum())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("directory", type=Path, help="an analyze_proxy.py --out/<version> directory")
    ap.add_argument(
        "--expect-runs",
        type=int,
        default=None,
        help="how many models the sweep should have produced; checked against the "
        "distinct run ids in paired.parquet",
    )
    ap.add_argument(
        "--selector",
        nargs="+",
        default=["fixed", "oracle"],
        help="rules that must appear in selected.parquet (default: fixed oracle)",
    )
    args = ap.parse_args(argv)

    directory: Path = args.directory
    if not directory.is_dir():
        print(f"not a directory: {directory}", file=sys.stderr)
        return 2

    print(f"checking {directory}")
    check = Checks()

    # --- frames present and non-empty ---
    frames: dict[str, pd.DataFrame] = {}
    for name in REQUIRED_FRAMES:
        frame = load(directory, name)
        if check(f"{name}.parquet", frame is not None and not frame.empty,
                 "missing" if frame is None else (f"{len(frame)} rows" if not frame.empty else "empty")):
            frames[name] = frame  # type: ignore[assignment]
    check("run_config.json", (directory / "run_config.json").is_file())

    if "paired" not in frames or "selection" not in frames:
        print("\nthe core frames are missing; later checks cannot run")
        return 1

    paired, selection = frames["paired"], frames["selection"]

    # --- both arms carry numbers ---
    n_real, n_synth = finite(paired, "real"), finite(paired, "synth")
    check("paired: real arm scored", n_real > 0, f"{n_real} finite rows")
    check("paired: synthetic arm scored", n_synth > 0, f"{n_synth} finite rows")

    if args.expect_runs is not None:
        # `model_id` is one sweep grid point; the two run-id columns have the same
        # cardinality (one run per point, both arms in it) and are the fallback.
        column = next(
            (c for c in ("model_id", "synth_run_id", "real_run_id") if c in paired.columns),
            None,
        )
        n_models = paired[column].nunique() if column else 0
        check(
            "sweep produced the expected models",
            n_models == args.expect_runs,
            f"{n_models} distinct {column or 'run id'}, expected {args.expect_runs}",
        )

    # --- the end-to-end assertion ---
    n_regret = finite(selection, "regret")
    check("selection: a population produced a finite regret", n_regret > 0,
          f"{n_regret}/{len(selection)} populations")

    # --- the two things a mis-sized run loses silently ---
    n_fixed = finite(frames.get("fixed_baseline", selection), "regret_fixed")
    if n_fixed == 0 and "combined" in frames:
        n_fixed = finite(frames["combined"], "regret_fixed")
    check("fixed-config baseline computed (needs >= 2 categories)", n_fixed > 0,
          f"{n_fixed} populations")

    combined = frames.get("combined", selection)
    decidable = combined["decidable"].notna().sum() if "decidable" in combined.columns else 0
    check("seed noise floor computed (needs >= 2 seeds)", int(decidable) > 0,
          f"{int(decidable)} populations carry a decidable flag")

    # --- the rules that were asked for came back ---
    if "selected" in frames:
        present = set(frames["selected"].get("selector", pd.Series(dtype=str)))
        missing = [rule for rule in args.selector if rule not in present]
        check("every requested selector is in selected.parquet", not missing,
              f"missing: {', '.join(missing)}" if missing else f"{len(present)} rules")

    print()
    if check.failed:
        print(f"{check.failed} check(s) failed")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
