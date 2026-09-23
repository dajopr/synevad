"""Run both proxy analyses over a sweep and dump the raw values as Parquet.

`synevad.analysis.correlation` asks whether the synthetic and real arms co-vary;
`synevad.analysis.selection` asks whether selecting on the synthetic arm actually picks a
good model. They answer different questions and only the second one supports a claim like
"synthetic data helps pick the best model", so this script always computes both, over the
*same* populations, and writes them side by side::

    python scripts/analyze_proxy.py --version sweep_v-1 --out outputs/analysis

    # re-analyse a dumped frame without touching the tracking store
    python scripts/analyze_proxy.py --paired outputs/analysis/sweep_v-1/paired.parquet

    # bootstrap intervals on regret/skill (slow-ish: n_boot resamples per group)
    python scripts/analyze_proxy.py --version sweep_v-1 --n-boot 2000

    # add the severity_* metrics, rebuilt from the runs' logged predictions
    python scripts/analyze_proxy.py --version sweep_v-1 --ordinal \
        --config synevad/config/mvtec.yaml

    # the same, with one grade taken out of the ladder
    python scripts/analyze_proxy.py --version sweep_v-1 --exclude-severity minimal

Nothing is rounded — these are the raw values, to be aggregated or plotted downstream.

Populations
-----------
One population per (metric, set, query, category, seed, num_train_samples): a selection
is made **among models trained on the same images**. The seed fixes which images were
drawn and `num_train_samples` how many, so pooling either would ask the proxy to spot
the luckier draw and the larger shot budget — neither of which is a knob anyone turns at
selection time, and the shot axis carries most of the real spread, so a proxy that had
learned nothing but "more shots is better" would report near-perfect skill. Populations
are correspondingly small (one per cell of the sweep grid, not one per category), so
watch `n` and `min_n`. `--pool-training-sets` restores the pooled grouping, with
`aggregate_replicates` averaging the seeds as before.

Outputs
-------
::

    <out>/<version>/
        paired.parquet          one row per (model, set, query, metric): the real and
                                synthetic scores everything else is derived from
        correlation.parquet     `correlate` per group: r / rho / tau, CIs, per-arm spread
        calibration.parquet     `evaluate_bias` per (category, severity, scorer): the
                                signed gap `synth - real`, its spread across models and
                                whether the sign survives it
        selection.parquet       `evaluate_selection` per group: regret, skill, hit@k,
                                plus the seed noise floor and a `decidable` flag
        noise.parquet           `replicate_noise`: within-config spread across seeds
        fixed_baseline.parquet  `fixed_config_baseline`: regret of using no synthetic data
        leave_one_out.parquet   `leave_one_out_query`: query chosen on the other
                                categories
        selected.parquet        `selected_metrics`: per population and selection rule,
                                the real image_auroc / aupro / pixel_auroc of the model
                                that rule picked, and its gap to the best available
        selected_summary.parquet  the same averaged to one row per (category, rule,
                                metric) — the results table, also printed to stdout
        selected_summary.tex    that table as a `booktabs` float: the best non-oracle
                                value of each row in bold, the next distinct one
                                underlined
        combined.parquet        correlation + selection + baseline joined on the group
                                keys
        run_config.json         the arguments these files were produced with

    with --ordinal, additionally:

        metrics_augmented.parquet  the stored metrics frame plus the severity_* columns
        severity_check.parquet     per-run recomputed-vs-stored image_auroc deltas

Parquet rather than CSV because the dtypes carry meaning a CSV round trip destroys:
`decidable` and `beats_fixed` are nullable booleans where blank means "no comparison was
possible", not `False`, and the `param_*` columns are MLflow strings that `read_csv`
would silently re-infer as numbers. The readers still accept a `.csv`.

Reading `combined.parquet`
--------------------------
The columns that carry the claim, in the order they have to be checked:

    decidable       False means the real scores span less than one seed's worth of noise.
                    Every other number in the row is then an artefact — there was no
                    model-selection decision to get right. Check this first.
    real_range      the same warning from the correlation side: a real metric at ceiling
                    has nothing left for the synthetic one to track.
    skill           0 perfect, 0 no better than a random pick, negative worse than random.
    regret          real score given up by taking the synthetic argmax, in metric units.
    regret_fixed    what a practitioner gets for free by reusing one config chosen on the
                    other categories. `beats_fixed` is regret < regret_fixed.
    kendall_tau     rank agreement, for contrast: it can be high while `hit` is -1, since
                    it weights the (worst, second-worst) pair like (best, second-best).

Severity rows (`--ordinal`)
---------------------------
`metric` also takes the ordinal values `severity_cindex`, `severity_cindex_defects`,
`severity_tau_b`, `severity_spearman` and `severity_adjacent_auc`
(`synevad.metrics.ordinal`). On those rows the synthetic arm is the ordinal statistic and
the **real arm is still `image_auroc`** — real defects carry no severity grade, so there
is nothing else to score them against (see `synevad.analysis.data.PROXY_ONLY_METRICS`).
Everything downstream is unchanged: regret is denominated in the real metric whatever the
proxy was.

Sweeps that ran before the metric existed did not log those columns, so `--ordinal`
rebuilds them from each run's `predictions/scores.csv` (`synevad.analysis.severity`) and
pairs against the augmented frame. It needs `--config` to be the config the sweep was run
with — the `queries` block is keyed by `set_name`, and a run whose set name is not in it
is skipped with a note. The pixel metrics are *not* recomputed: `anomaly_maps.npz` is not
logged by default, so `pixel_auroc`, `pixel_aupr` and `aupro` are carried over from the
stored parquet as they are. `--ordinal` also writes two extra files:
`metrics_augmented.parquet`, the stored metrics plus the new columns, and
`severity_check.parquet`, the recomputed-vs-stored `image_auroc` deltas. The second is the
guard on the first — the severity statistics are computed over the same selected rows as
that AUROC, so a disagreement above `--check-tolerance` aborts the run rather than
analysing columns that cannot be trusted.

`--exclude-severity` drops named grades from the ordinal statistics before they are
computed: `--exclude-severity minimal` asks whether the ladder still orders once the
grade the generator is least reliable at is out of it, and `--exclude-severity none`
drops the defect-free images so `severity_cindex` carries the ordering alone (which is
what `severity_cindex_defects` already reports, but with `none` gone every level pair,
`tau_b` and `adjacent_auc` included, is defect-only too). Surviving grades keep their
ladder positions, so the filtered numbers sit on the same scale as the unfiltered ones
and the two runs are directly comparable — dump them to different `--out` directories.
The reproduction check is unaffected: `image_auroc` is always recomputed over the query's
full row set, so it still has to match what the sweep stored. The exclusion is recorded in
`run_config.json`, which is the only thing distinguishing two otherwise identical dumps.

The comparison worth reading off is `skill` on `severity_cindex_defects` against `skill`
on `image_auroc`, within one `set_name` and query. The first strips the good-vs-defect
signal out of the proxy, so if it does not beat the second, the severity ladder added
nothing that AUROC did not already carry. Rows where `n_severity_levels < 2` are the
metric not applying — a single-grade query, or the real arm — not a proxy that failed.

Reading `calibration.parquet`
-----------------------------
Correlation and selection are both invariant to a shift — add 0.1 to every synthetic
score and `r`, `skill` and `regret` are unchanged — so neither says whether the number the
synthetic set *reports* is the model's real quality. `bias` is that: the mean of
`synth - real` over the models of a cell, positive where the generated images flatter
them. Cells are (category, severity, scorer): `analyze_proxy` splits the `query` axis
into the severity grade and (if the name carries one) the scorer decision, so a
hand-added `minimal_accept` contributes to `severity == 'minimal'` and
`scorer == 'accept'` rather than to a third query that pools models already counted
elsewhere. Default configs only name the four grades and `all`.

Read `bias` next to `bias_sd`, never on its own. A constant offset costs selection
nothing — every model is flattered equally and the ranking is untouched — so a large
`bias` with a small `bias_sd` means "usable for ranking, not quotable as an estimate",
while a small `bias` with a large `bias_sd` is unbiased on average and wrong per model,
which is the worse case and the one a mean alone hides. `overestimates` is the headline:
true / false when the whole interval sits on one side of zero, null when it straddles.
`over_share` is the same read without the normality assumption.

Only metrics whose two arms are the same quantity get a row. The `severity_*` proxies are
scored against a broadcast real `image_auroc` (see `synevad.analysis.data`), so their gap
would subtract an AUROC from a concordance; those rows are dropped rather than reported.

`leave_one_out.parquet` is the honest version of "which query works best" — reading the
maximum over queries out of `selection.parquet` selects the proxy on the data it is then
scored on, and with a few dozen queries some will look excellent by chance.

Reading `selected_summary.parquet`
----------------------------------
Everything above is a *difference*: regret against the population's best model, skill
against a random pick. Neither can be quoted beside a published number, and neither says
whether a category sits at 0.99 or at 0.75. This table reports the levels instead — the
real `image_auroc`, `aupro` and `pixel_auroc` of the model each rule would have picked —
one row per (category, rule, metric), and prints them to stdout as three fixed-width
blocks. `--select-by` is the list of rules: `fixed` is one config chosen on the other
categories with no synthetic data, `oracle` is the real argmax, and anything else is
`<metric>@<query>`, the argmax of that synthetic column.

`selected_summary.tex` is the same numbers as a float ready to paste: within one row and
one metric the best value is `\\textbf` and the next distinct one `\\underline`, ranked on
the *printed* precision so two cells showing the same number are marked alike. The oracle
is printed but held out of that ranking — it reads the answer, so leaving it in would put
the bold in the same column of every row. The last row is the unweighted mean over the
categories, and the metric groups are headed I-AUC / P-AUC / PRO rather than by the parquet
keys. `--latex-caption` and `--latex-label` set the float's two strings.

`selected_summary.parquet` is the same numbers averaged to one row per (category, rule,
metric) — the results table. Default `--select-by` is `fixed`, `image_auroc@all`, `oracle`.

`value` is the mean over the category's training-set populations — every seed and shot
count the sweep ran — so it is what the rule gets on average across the grid, not the
score of one run; `value_sd` is the spread that average hides, and is mostly the shot
count. `regret` is the mean gap to each population's own ceiling, which is the column that
compares two rules; the oracle's is 0 by construction. Note the oracle is resolved **per
metric**, so it is the best achievable in each column separately rather than one config,
while every other rule picks one config and is then read on all three.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from synevad.analysis import (
    DEFAULT_BY,
    DEFAULT_REPORT_METRICS,
    DEFAULT_SELECTORS,
    TRAINING_SET_KEYS,
    aggregate_replicates,
    augment_metrics,
    build_paired_metrics,
    check_image_auroc,
    conditioned_by,
    correlate,
    evaluate_bias,
    evaluate_selection,
    fixed_config_baseline,
    leave_one_out_query,
    format_table,
    latex_table,
    recompute_severity,
    replicate_noise,
    selected_metrics,
    summarise_selected,
    with_noise_floor,
)
from synevad.analysis.calibration import BIAS_FIELDS, worst_cells
from synevad.analysis.selected import DEFAULT_RANK_METRIC
from synevad.metrics import DEFAULT_MIN_LEVEL_N, SEVERITY_NAMES, normalise_exclusions

# Correlation fields that `evaluate_selection` also reports, identically, because both run
# on the same population; keeping one copy each keeps `combined.parquet` readable.
SHARED_FIELDS = ("n", "n_dropped", "alpha")


def load_paired(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame]:
    """The `build_paired_metrics` frame, whatever `--ordinal` produced, and the runs.

    The second element is written out next to the analysis tables; it is empty unless
    `--ordinal` ran, in which case it carries the augmented metrics frame and the
    reproduction check. The third is the run metadata, empty under `--paired`, where
    there is no store to have read it from.
    """
    if args.paired is not None:
        path = Path(args.paired)
        frame = (
            pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        )
        print(f"read {len(frame)} paired rows from {path}", file=sys.stderr)
        return frame, {}, pd.DataFrame()

    # Imported here so `--paired` works with an unmigrated or absent tracking store.
    import mlflow

    from synevad.db import get_metadata_by_version, tracking_uri_from_env

    mlflow.set_tracking_uri(tracking_uri_from_env())
    runs = get_metadata_by_version(args.version, experiment_names=args.experiment)
    if runs.empty:
        raise SystemExit(
            f"no finished runs tagged version={args.version!r} in {list(args.experiment)}"
        )
    print(f"found {len(runs)} runs for version={args.version}", file=sys.stderr)

    if not args.ordinal:
        return build_paired_metrics(runs), {}, runs
    paired, extras = build_ordinal_paired(runs, args)
    return paired, extras, runs


def build_ordinal_paired(
    runs: pd.DataFrame, args: argparse.Namespace
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Pair against the stored metrics *augmented* with the severity columns.

    The severity statistics are rebuilt from each run's `predictions/scores.csv`
    (`synevad.analysis.severity`), so a sweep that ran before the metric existed can be
    analysed without re-running it. The pixel metrics cannot be rebuilt — `anomaly_maps.npz`
    is not logged by default — so they are carried over from the stored parquet unchanged.

    Aborts on a failed reproduction check rather than analysing numbers the check says are
    untrustworthy: if the recomputed `image_auroc` disagrees with the stored one, the two
    frames were written from different row sets and the severity columns are wrong too.
    """
    from omegaconf import OmegaConf

    from synevad.db import get_ad_metrics

    config = OmegaConf.load(args.config)
    if args.exclude_severity:
        print(
            f"excluding severity grade(s) {', '.join(args.exclude_severity)} from the "
            "ordinal statistics (image_auroc is still recomputed over every row)",
            file=sys.stderr,
        )
    recomputed = recompute_severity(
        runs,
        config,
        min_level_n=args.min_level_n,
        exclude=args.exclude_severity,
    )
    if recomputed.empty:
        raise SystemExit(
            f"--ordinal: no run yielded a readable scores.csv (is {args.config} the "
            "config this sweep was run with? its `queries` block is keyed by set_name)"
        )

    stored = get_ad_metrics(runs)
    check = check_image_auroc(stored, recomputed)
    worst = float(check["delta"].max()) if len(check) else float("nan")
    disagree = int((check["delta"] > args.check_tolerance).sum()) if len(check) else 0
    print(
        f"image_auroc reproduces to {worst:.2e} over {len(check)} rows "
        f"({disagree} above --check-tolerance {args.check_tolerance:g})",
        file=sys.stderr,
    )
    # The guard the severity columns rest on, and the one that keeps a selection rule
    # reading the stored `image_auroc` out of the same table as one reading a recomputed
    # column. Only `--image-metrics stored` mixes the two, so only it can fail here:
    # `recomputed` takes every image metric from the rows everything else was recomputed
    # over, which is the documented way out rather than a way around.
    if disagree and args.image_metrics == "stored":
        raise SystemExit(
            f"--image-metrics stored: the stored image_auroc disagrees with the one "
            f"rebuilt from scores.csv on {disagree}/{len(check)} (run, query) rows, worst "
            f"{worst:.3e}. The two describe different row sets — a query whose expression "
            f"changed since the sweep, or a gate naming a column the file did not carry at "
            f"eval time — so the severity columns cannot be read beside the stored "
            f"ones. Re-run with --image-metrics recomputed, point --config at the config "
            f"this sweep ran with, or raise --check-tolerance if you accept the gap."
        )

    augmented = augment_metrics(
        stored, recomputed, prefer_recomputed=args.image_metrics == "recomputed"
    )
    if args.image_metrics == "recomputed" and disagree:
        print(
            f"image metrics taken from scores.csv, not from the runs: {disagree} rows "
            "would otherwise have been compared across two different row sets",
            file=sys.stderr,
        )
    graded = int((augmented["n_severity_levels"].fillna(0) >= 2).sum())
    print(
        f"severity columns on {len(augmented)} metric rows, {graded} of them with two or "
        "more grades to order",
        file=sys.stderr,
    )
    if not graded:
        # Not an empty recompute — the real arm alone yields rows, with no grades to
        # order — so this would otherwise "succeed" into an analysis whose severity rows
        # are all NaN. The usual cause is the set_name/config mismatch named here.
        seen = sorted(
            {str(name) for name in runs.get("tags.set_name", pd.Series(dtype=str))}
        )
        excluded = (
            f" You also excluded {', '.join(args.exclude_severity)}, which may have left "
            "fewer than two grades standing; try it without --exclude-severity first."
            if args.exclude_severity
            else ""
        )
        raise SystemExit(
            f"--ordinal: no run had two or more severity grades to order. The runs are "
            f"tagged set_name {seen}, and {args.config} keys its `queries` block by "
            f"{sorted(config.queries.keys())} — a run whose set name is not in that block "
            "is skipped, and the real arm carries no grades. Point --config at the config "
            f"this sweep was run with.{excluded}"
        )
    paired = build_paired_metrics(runs, metrics=augmented)
    return paired, {"metrics_augmented": augmented, "severity_check": check}


def collapse_replicates(paired: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """Average seed replicates, unless asked not to or the sweep has no seed column.

    On by default: `default_group_keys` folds the seed into the model identity, so a
    sweep of 7 configs x 3 seeds arrives as 24 "models". Selecting over those measures
    partly whether the synthetic set tracks seed noise, and `real_best` becomes the
    luckiest seed rather than the best config.

    Off whenever the seed is a *grouping* key, which is what `--pool-training-sets`
    switches off: there the seeds are separate populations rather than replicates of one
    model, and averaging would delete the column the grouping needs.
    """
    if args.replicate_key in args.by:
        print(
            f"grouping by {args.replicate_key!r}; keeping replicates as populations",
            file=sys.stderr,
        )
        return paired
    if args.keep_replicates:
        return paired
    if args.replicate_key not in paired.columns:
        print(
            f"no {args.replicate_key!r} column; analysing rows as given",
            file=sys.stderr,
        )
        return paired

    collapsed = aggregate_replicates(paired, replicate_key=args.replicate_key)
    print(
        f"collapsed {len(paired)} rows to {len(collapsed)} configs "
        f"(max {int(collapsed['n_replicates'].max())} replicates)",
        file=sys.stderr,
    )
    return collapsed


def build_combined(
    correlation: pd.DataFrame,
    selection: pd.DataFrame,
    baseline: pd.DataFrame,
    by: list[str],
) -> pd.DataFrame:
    """Join the three per-group frames on whatever keys they share.

    Correlation and selection share `by` exactly. The baseline is real-only, so it is
    keyed by metric and the held-out column alone and fans out across the sets and
    queries that share them.
    """
    lean = correlation.drop(columns=[c for c in SHARED_FIELDS if c in correlation])
    combined = selection.merge(lean, on=by, how="outer", validate="one_to_one")

    baseline_keys = [key for key in baseline.columns if key in combined.columns]
    if baseline_keys and not baseline.empty:
        combined = combined.merge(baseline, on=baseline_keys, how="left")
        # Nullable, so a fold with nothing to choose the fixed config on (one category,
        # say) reads as blank rather than as a synthetic arm that lost the comparison.
        combined["beats_fixed"] = (
            (combined["regret"] < combined["regret_fixed"])
            .astype("boolean")
            .mask(combined["regret_fixed"].isna() | combined["regret"].isna())
        )

    # Decision first, evidence second, correlation last: the order the pathcstring reads in.
    lead = [
        *by,
        "n",
        "decidable",
        "skill",
        "regret",
        "regret_fixed",
        "beats_fixed",
        "hit",
        "hit_at_k",
        "regret_in_noise_sd",
    ]
    ordered = [col for col in lead if col in combined.columns]
    return combined[[*ordered, *[c for c in combined.columns if c not in ordered]]]


def run_analyses(
    paired: pd.DataFrame, noise: pd.DataFrame, args: argparse.Namespace
) -> dict[str, pd.DataFrame]:
    """Both analyses over the same populations, plus the baselines.

    `noise` is computed by the caller off the *un*collapsed frame — the seed spread is
    precisely what `aggregate_replicates` averages away, so it cannot be recovered from
    `paired` here.
    """
    by = list(args.by)
    shared = dict(min_defect_samples=args.min_defect_samples, min_n=args.min_n)

    correlation = correlate(paired, by=by, alpha=args.alpha, **shared)
    # Its own grouping rather than `by`: `bias_keys` swaps `query` for the severity and
    # scorer axes it is the product of, so the gap is reported *per severity* instead of
    # averaged over the axis the question names.
    calibration = evaluate_bias(paired, by=by, alpha=args.alpha, **shared)
    selection = evaluate_selection(
        paired,
        by=by,
        k=args.k,
        alpha=args.alpha,
        n_boot=args.n_boot,
        rng=args.seed,
        **shared,
    )
    if not noise.empty:
        selection = with_noise_floor(selection, noise)

    # The baseline is conditioned on whatever the selection was conditioned on, or
    # `beats_fixed` would compare a regret measured within one training set against a
    # config chosen across all of them.
    baseline_by = [key for key in ("metric", *TRAINING_SET_KEYS) if key in by]
    baseline = pd.DataFrame()
    if args.held_out in paired.columns and any(
        col.startswith("param_")
        and col != args.replicate_key
        and col not in baseline_by
        for col in paired.columns
    ):
        baseline = fixed_config_baseline(
            paired,
            by=baseline_by,
            held_out=args.held_out,
            replicate_key=args.replicate_key,
        )

    held_out = pd.DataFrame()
    loo_by = [key for key in by if key not in ("query", args.held_out)]
    if args.held_out in paired.columns and paired[args.held_out].nunique() > 0:
        held_out = leave_one_out_query(
            paired,
            by=loo_by,
            held_out=args.held_out,
            choose_by=args.choose_by,
            k=args.k,
            alpha=args.alpha,
            **shared,
        )

    frames = {
        "paired": paired,
        "correlation": correlation,
        "calibration": calibration,
        "selection": selection,
        "noise": noise,
        "fixed_baseline": baseline,
        "leave_one_out": held_out,
        "combined": build_combined(correlation, selection, baseline, by),
    }
    frames.update(run_selected(paired, args))
    return frames


def run_selected(
    paired: pd.DataFrame, args: argparse.Namespace
) -> dict[str, pd.DataFrame]:
    """The achieved-metrics table: what each selection rule picked, and what it scored.

    The rest of the analysis reports differences — regret against the population's best,
    skill against a random pick — and a difference cannot be quoted next to a published
    number. This reports the levels: the real `image_auroc`, `aupro` and `pixel_auroc` of
    the model each rule would have chosen, per category.

    On the same frame and the same populations `evaluate_selection` used, so a row here
    and its row in `selection.parquet` are the same comparison read two ways.
    """
    try:
        selected = selected_metrics(
            paired,
            by=list(args.by),
            selectors=args.select_by,
            metrics=args.report_metric,
            held_out=args.held_out,
            rank_metric=args.fixed_rank_metric,
            replicate_key=args.replicate_key,
            min_n=args.min_n,
        )
    except (KeyError, ValueError) as error:
        print(f"skipped the achieved-metrics table: {error}", file=sys.stderr)
        return {}

    if selected.empty:
        print(
            f"skipped the achieved-metrics table: no population had {args.min_n} or "
            "more configs to choose between",
            file=sys.stderr,
        )
        return {}

    summary_by = [key for key in ("set_name", args.held_out) if key in selected.columns]
    return {
        "selected": selected,
        "selected_summary": summarise_selected(selected, by=summary_by),
    }


def report_bias(calibration: pd.DataFrame, *, file) -> None:
    """Which (category, severity) cells the synthetic arm flatters, on stderr.

    Only the cells whose interval clears zero are named: the scan produces one row per
    cell and a list of every sign, most of them undecided, is not a summary. The spread
    line beside it is the one that matters for selection — a bias every model shares is a
    rescaling, a bias that varies by model is a ranking error (`synevad.analysis.calibration`).
    """
    if calibration.empty or "overestimates" not in calibration.columns:
        print("no calibration cells (nothing paired on a shared metric)", file=file)
        return

    decided = calibration["overestimates"].notna()
    over = int((calibration["overestimates"] == True).sum())  # noqa: E712 — nullable
    under = int((calibration["overestimates"] == False).sum())  # noqa: E712
    print(
        f"{int(decided.sum())}/{len(calibration)} severity x category cells have a "
        f"quality gap whose sign clears the model spread: {over} overestimate the model, "
        f"{under} underestimate it",
        file=file,
    )

    if not decided.any():
        return
    # Every group key, not just the three the question names: with the training-set keys
    # in the grouping there is one cell per seed, and printing only category/severity
    # would list the same label several times with different numbers beside it.
    keys = [
        key for key in calibration.columns if key not in BIAS_FIELDS and key != "metric"
    ]
    for _, row in worst_cells(calibration).iterrows():
        # `param_*` values are bare numbers (a seed, a shot count) and mean nothing
        # without their key; the named axes speak for themselves.
        where = " / ".join(
            f"{key.removeprefix('param_')}={row[key]}"
            if key.startswith("param_")
            else str(row[key])
            for key in keys
        )
        print(
            f"  {row['bias']:+.3f} {row.get('metric', '?')} on {where}"
            f"  (spread {row['bias_sd']:.3f} across {int(row['n'])} models)",
            file=file,
        )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--version", default=None, help="sweep version tag to analyse")
    source.add_argument(
        "--paired",
        default=None,
        help="re-analyse a dumped paired frame (.parquet/.csv) instead of the store",
    )
    ap.add_argument(
        "--experiment",
        nargs="+",
        default=["synevad"],
        help="MLflow experiment names to search (default: synevad)",
    )
    ap.add_argument("--out", default="outputs/analysis", help="output directory root")
    ordinal = ap.add_argument_group(
        "severity metrics",
        "rebuild the ordinal severity columns from each run's predictions/scores.csv, "
        "for sweeps that ran before the metric existed. Reads the store, so --version only",
    )
    ordinal.add_argument(
        "--ordinal",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="add the severity_* metrics to the analysis (default: on with --version, "
        "off with --paired, which has no artifacts to rebuild them from)",
    )
    ordinal.add_argument(
        "--config",
        default="synevad/config/mvtec.yaml",
        help="config carrying the `queries` block the sweep was run with",
    )
    ordinal.add_argument(
        "--min-level-n",
        type=int,
        default=DEFAULT_MIN_LEVEL_N,
        help="drop severity levels with fewer images than this before ordering",
    )
    ordinal.add_argument(
        "--exclude-severity",
        nargs="+",
        default=[],
        metavar="GRADE",
        help=(
            "severity grades to drop from the ordinal statistics, e.g. `minimal`, or "
            "`none` to drop the defect-free images and keep the ordering alone. "
            f"One or more of {sorted(SEVERITY_NAMES)}. Surviving grades keep their ladder "
            "positions, so the numbers stay comparable with an unfiltered run"
        ),
    )
    ordinal.add_argument(
        "--check-tolerance",
        type=float,
        default=1e-9,
        help="how far the recomputed image_auroc may sit from the stored one before the "
        "two count as different row sets (only enforced under --image-metrics stored)",
    )
    ordinal.add_argument(
        "--image-metrics",
        choices=["recomputed", "stored"],
        default="recomputed",
        help="where the image_auroc/image_aupr of each run come from: `recomputed` (the "
        "default) rebuilds them from scores.csv, so every image-level column in the "
        "analysis is measured over the same rows as the severity ones beside it; "
        "`stored` keeps what the runs logged and aborts if the two disagree",
    )
    ap.add_argument(
        "--by",
        nargs="+",
        default=list(DEFAULT_BY),
        help=(
            "group keys defining one population of models "
            f"(default: {' '.join(DEFAULT_BY)}, plus the training-set keys "
            f"{' '.join(TRAINING_SET_KEYS)} unless --pool-training-sets). Dropping "
            "`category` pools categories, which measures which category is easiest, not "
            "which model is best"
        ),
    )
    ap.add_argument(
        "--pool-training-sets",
        action="store_true",
        help="put every seed and shot count in one population, so selection is made "
        "across models trained on different images (inflates skill: the shot count "
        "dominates the real spread and any proxy gets it right)",
    )
    ap.add_argument(
        "--k", type=int, default=2, help="shortlist size for hit@k / regret@k"
    )
    ap.add_argument(
        "--n-boot",
        type=int,
        default=-1,
        help="bootstrap resamples for regret/skill intervals (-1 = off)",
    )
    ap.add_argument("--seed", type=int, default=-1, help="bootstrap seed")
    ap.add_argument(
        "--alpha", type=float, default=0.05, help="two-sided interval level"
    )
    ap.add_argument(
        "--min-n", type=int, default=3, help="smallest population worth a statistic"
    )
    ap.add_argument(
        "--min-defect-samples",
        type=int,
        default=None,
        help="drop queries that selected fewer defects than this (cf. config.min_num_defects)",
    )
    ap.add_argument(
        "--keep-replicates",
        action="store_true",
        help="with --pool-training-sets, analyse seeds as separate models instead of "
        "averaging them (inflates n and makes real_best the luckiest seed); ignored "
        "otherwise, since the seeds are already separate populations",
    )
    ap.add_argument(
        "--replicate-key", default="param_seed", help="the replicate column"
    )
    ap.add_argument(
        "--held-out",
        default="category",
        help="column held out by the baseline and the query choice",
    )
    ap.add_argument(
        "--choose-by",
        default="skill",
        help="selection field the leave-one-out query choice maximises",
    )
    table = ap.add_argument_group(
        "achieved metrics",
        "the per-category results table: what each way of picking a model actually "
        "scored, in metric units rather than in regret (synevad.analysis.selected)",
    )
    table.add_argument(
        "--select-by",
        nargs="+",
        default=list(DEFAULT_SELECTORS),
        metavar="RULE",
        help="the selection rules to compare, each `<metric>@<query>` or one of `fixed` "
        "(one config chosen on the other categories, no synthetic data) and `oracle` "
        "(the real argmax, i.e. the ceiling). A bare metric means `<metric>@all` "
        f"(default: {' '.join(DEFAULT_SELECTORS)})",
    )
    table.add_argument(
        "--report-metric",
        nargs="+",
        default=list(DEFAULT_REPORT_METRICS),
        metavar="METRIC",
        help="the real metrics each rule's pick is reported on "
        f"(default: {' '.join(DEFAULT_REPORT_METRICS)})",
    )
    table.add_argument(
        "--latex-caption",
        default="Real score of the model each selection rule picked, per category.",
        help="caption of the `selected_summary.tex` float written beside the tables",
    )
    table.add_argument(
        "--latex-label",
        default="tab:selected",
        help="\\label of that float",
    )
    table.add_argument(
        "--fixed-rank-metric",
        default=DEFAULT_RANK_METRIC,
        help="the real metric the `fixed` rule ranks configs on; one config is then "
        f"read for every --report-metric (default: {DEFAULT_RANK_METRIC})",
    )
    args = ap.parse_args()

    # `--ordinal` rebuilds its inputs from the runs' artifacts, which `--paired` does
    # not have. Left unset it simply switches off there instead of making `--paired`
    # unusable; asked for explicitly it is an error, because silently dropping what was
    # requested is worse than refusing it.
    from_store = args.paired is None
    for name in ("ordinal",):
        requested = getattr(args, name)
        if requested and not from_store:
            ap.error(
                f"--{name} rebuilds its inputs from the runs' artifacts, which --paired "
                f"does not have access to; use --version, or dump a paired frame with "
                f"--{name} once and re-analyse that"
            )
        setattr(args, name, from_store if requested is None else requested)

    if args.exclude_severity:
        # Normalised in place so `run_config.json` records the grades that were actually
        # dropped, and a typo fails here rather than after the store has been read.
        try:
            args.exclude_severity = sorted(normalise_exclusions(args.exclude_severity))
        except ValueError as error:
            ap.error(f"--exclude-severity: {error}")
    return args


def main() -> None:
    args = parse_args()

    paired, extras, runs = load_paired(args)
    if paired.empty:
        raise SystemExit("paired frame is empty; nothing to analyse")

    # Resolved once and written back, so `run_config.json` records the grouping the
    # numbers were actually produced at rather than the one that was typed.
    if not args.pool_training_sets:
        args.by = conditioned_by(paired, args.by)
    conditioned = [key for key in TRAINING_SET_KEYS if key in args.by]
    print(
        f"one population per {' x '.join(args.by)}"
        + (
            f" — selection is made within a training set ({', '.join(conditioned)})"
            if conditioned
            else " — pooled across training sets"
        ),
        file=sys.stderr,
    )

    # Before collapsing: the seed spread is exactly what `aggregate_replicates` averages
    # away, and it is the floor every regret has to clear to mean anything.
    noise = pd.DataFrame()
    if args.replicate_key in paired.columns:
        noise = replicate_noise(
            paired, by=list(args.by), replicate_key=args.replicate_key
        )

    frames = run_analyses(collapse_replicates(paired, args), noise, args)

    label = args.version or Path(args.paired).stem
    out = Path(args.out) / label
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in {**extras, **frames}.items():
        path = out / f"{name}.parquet"
        frame.to_parquet(path, index=False)
        print(f"wrote {len(frame):>5} rows  {path}", file=sys.stderr)

    if "selected_summary" in frames:
        # Beside the parquet rather than instead of it: the float is for pasting into a
        # paper, and every number in it is in the table next to it for re-reading.
        path = out / "selected_summary.tex"
        path.write_text(
            latex_table(
                frames["selected_summary"],
                metrics=args.report_metric,
                caption=args.latex_caption,
                label=args.latex_label,
            )
        )
        print(f"wrote        {path}", file=sys.stderr)

    (out / "run_config.json").write_text(json.dumps(vars(args), indent=1, default=str))

    # The two headline counts, so a run that produced nothing usable says so on stderr
    # rather than only in a column somebody has to know to look at.
    combined = frames["combined"]
    print(file=sys.stderr)
    if "decidable" in combined.columns:
        decidable = int(combined["decidable"].sum())
        print(
            f"{decidable}/{len(combined)} populations are decidable "
            "(real spread exceeds the seed noise floor)",
            file=sys.stderr,
        )
    else:
        print(
            "no seed replicates: the noise floor is unknown, so a small regret cannot "
            "be told apart from an undecidable comparison",
            file=sys.stderr,
        )

    if "beats_fixed" in combined.columns:
        scored = combined["beats_fixed"].notna().sum()
        print(
            f"{int(combined['beats_fixed'].sum())}/{scored} beat the fixed-config "
            "baseline that uses no synthetic data",
            file=sys.stderr,
        )
        if scored < len(combined):
            print(
                f"({len(combined) - scored} population(s) had no other {args.held_out} "
                "to choose a fixed config on)",
                file=sys.stderr,
            )

    report_bias(frames["calibration"], file=sys.stderr)

    if "selected_summary" in frames:
        # stdout, not stderr: this is the results table, not a progress note, so
        # `> table.txt` captures it without the diagnostics above.
        print(
            f"\nreal score of the model each rule picked, averaged over the "
            f"{args.held_out}'s training-set populations\n",
        )
        print(format_table(frames["selected_summary"], metrics=args.report_metric))



if __name__ == "__main__":
    main()
