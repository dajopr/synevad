"""What you actually get, in metric units, for choosing a model each way.

`selection.py` reports the *cost* of a choice — regret against the best model in the
population, and skill against a random pick. Both are differences, and a difference
cannot be quoted in a results table: "regret 0.004 AUROC" says nothing about whether the
category sits at 0.99 or at 0.75, and it cannot be read against a number from a paper.
This module reports the level instead. For each way of picking a model it names the
config that rule would have picked and reads that config's **real** `image_auroc`,
`aupro` and `pixel_auroc` straight off the paired frame, so the output is a table of the
same quantities every anomaly-detection benchmark prints, one row per category.

The two costs come along beside the level rather than instead of it — `regret` against
the population's ceiling, and `skill`, that regret rescaled by the regret of a random
pick. `selection.py` computes those for one proxy against one real metric; here they are
computed the same way for *every* rule, the fixed baseline and the oracle included, so
"what did this rule score", "what did it leave on the table" and "how much of the
available spread did it capture" are three columns of one frame rather than three passes
over the sweep. Regret is the number to quote in metric units; skill is the one that can
be averaged across categories whose models sit at different spreads.

The rules being compared (`DEFAULT_SELECTORS`), each a `metric@query` or a keyword:

    fixed                 no synthetic data at all: one config, chosen on the *other*
                          categories by mean rank (`fixed_config_choice`). What a
                          practitioner gets for free, and the bar worth clearing.
    image_auroc@all       AUROC over every generated image
    image_auroc@<grade>   the same on one severity slice (minimal/slight/moderate/severe),
                          where the config's `queries` block defines the slice
    oracle                the real argmax — not a selection rule, since it reads the
                          answer, but the ceiling the others are trying to reach

Two things to keep in mind when reading the output:

* **The oracle is per metric.** The config with the best real `image_auroc` need not be
  the one with the best real `aupro`, so `regret` on the aupro row is measured against
  the aupro ceiling. That is the standard definition and the only one under which regret
  0 means "nothing was left on the table for this metric", but it does mean the oracle
  row is not one config — it is the best achievable in each column separately.
* **Every other selector *is* one config per population**, so its three metrics are read
  off the same model. A rule that picks well for AUROC and badly for AUPRO shows up as
  exactly that, which is the comparison the table exists to make.

`selected_metrics` works at the population grain the rest of the analysis uses — one per
(category, seed, shot count) — and `summarise_selected` averages those into the
per-category table. That average is over the whole sweep grid, shot counts included, so a
category's number is "what this rule gets you on average across the training budgets
swept", not the score of any one run; `value_sd` and `n_populations` are what it was taken
over.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .selection import DEFAULT_BY, config_dimensions, fixed_config_choice

# The metrics the table reports, read off the *real* arm for whichever config each rule
# picked. Every one is a real-arm quantity `build_paired_metrics` already carries, so this
# is a display choice and not a computation.
DEFAULT_REPORT_METRICS: tuple[str, ...] = ("image_auroc", "aupro", "pixel_auroc")

# Keyword selectors: the two rules that are not "argmax of a synthetic column".
FIXED = "fixed"
ORACLE = "oracle"

DEFAULT_SELECTORS: tuple[str, ...] = (
    FIXED,
    "image_auroc@all",
    ORACLE,
)

# The metric whose real scores the fixed config is ranked on. The baseline is a choice
# made with no synthetic data, so it cannot depend on a query, and ranking on the headline
# image metric is what a practitioner carrying one config between categories would do.
DEFAULT_RANK_METRIC = "image_auroc"

# Fixed key set, so the frame has a stable schema whichever branch each population took.
SELECTED_FIELDS: tuple[str, ...] = (
    "value",
    "best",
    "chance",
    "regret",
    "skill",
    "n_models",
    "n_tied",
    "config",
)

SUMMARY_FIELDS: tuple[str, ...] = (
    "value",
    "value_sd",
    "best",
    "chance",
    "regret",
    "regret_sd",
    "skill",
    "skill_sd",
    "n_populations",
    "n_scored",
    "n_skilled",
)

# Prefix for the synthetic score columns joined onto the real table, kept out of the way
# of the metric names that sit beside them in the same frame.
_SYNTH = "synth::"

_NAN = float("nan")


@dataclass(frozen=True)
class Selector:
    """One way of picking a model, and how to evaluate it.

    `label` is the spec it was parsed from, so it round-trips into `--select-by` and keeps
    the table in the order the caller listed rather than in alphabetical order.
    """

    label: str
    kind: str  # "synth", "fixed" or "oracle"
    metric: str | None = None
    query: str | None = None


def parse_selector(spec: str) -> Selector:
    """`"fixed"`, `"oracle"` or `"<metric>@<query>"` into a `Selector`.

    A bare metric is read as `<metric>@all`: `all` is the query every set defines, and a
    selector naming no query most likely means the unfiltered one.
    """
    spec = spec.strip()
    if spec in (FIXED, ORACLE):
        return Selector(label=spec, kind=spec)
    metric, _, query = spec.partition("@")
    if not metric:
        raise ValueError(
            f"selector {spec!r} names no metric; expected `<metric>@<query>`, "
            f"{FIXED!r} or {ORACLE!r}"
        )
    return Selector(label=spec, kind="synth", metric=metric, query=query or "all")


def population_keys(
    by: Sequence[str] = DEFAULT_BY, *, drop: Sequence[str] = ("metric", "query")
) -> list[str]:
    """`by` with the two axes a selector fixes taken out.

    A population here is the models that could have been chosen between — the grouping
    `evaluate_selection` uses, minus `metric` and `query`, which stop being grouping keys
    once a selector names them. `set_name` stays: two synthetic sets are two different
    proxies, and their picks are two different rows.
    """
    return [key for key in by if key not in set(drop)]


def real_by_config(
    paired: pd.DataFrame,
    *,
    by: Sequence[str],
    configs: Sequence[str],
    metrics: Sequence[str] = DEFAULT_REPORT_METRICS,
) -> pd.DataFrame:
    """One row per (population, config): the real score of each reported metric.

    `real` is broadcast across queries and sets by `build_paired_metrics`, so the mean over
    a (population, config, metric) cell collapses those duplicates back to the single value
    the real run logged. A metric the sweep never produced comes back as an all-NaN column
    rather than a missing one, so the schema does not depend on the run.
    """
    metrics = list(metrics)
    wanted = paired[paired["metric"].isin(metrics)]
    table = (
        wanted.groupby([*by, *configs, "metric"], dropna=False, sort=True)["real"]
        .mean()
        .unstack("metric")
    )
    return table.reindex(columns=metrics)


def synth_by_config(
    paired: pd.DataFrame,
    selector: Selector,
    *,
    by: Sequence[str],
    configs: Sequence[str],
) -> pd.Series:
    """The synthetic column a `synth` selector takes its argmax of, per (population, config).

    Empty when the frame carries no such (metric, query) — a sweep predating the metric,
    or a query the config never defined. The caller reports that as NaN rather than as a
    rule that chose badly.
    """
    rows = paired[
        (paired["metric"] == selector.metric) & (paired["query"] == selector.query)
    ]
    if rows.empty:
        return pd.Series(dtype=float)
    return rows.groupby([*by, *configs], dropna=False, sort=True)["synth"].mean()


def _config_key(values: Sequence, configs: Sequence[str]):
    """A config as it is keyed in the table's index: a tuple, or a scalar for one column."""
    return tuple(values) if len(configs) > 1 else values[0]


def _fixed_choices(
    paired: pd.DataFrame,
    *,
    by: Sequence[str],
    configs: Sequence[str],
    held_out: str,
    rank_metric: str,
) -> dict[tuple, object]:
    """`fixed_config_choice` as a lookup from population to the config it carried over.

    Ranked on one metric's real scores: the fixed config is chosen with no synthetic data,
    so it cannot depend on a query, and the *same* config is then read for all three
    reported metrics — which is the whole point of the baseline. Folds it could not decide
    (one category, or a winner never run on the held-out one) are simply absent.
    """
    if held_out not in by:
        raise KeyError(
            f"the {FIXED!r} selector holds out {held_out!r}, which is not a population "
            f"key ({list(by)}); there is nothing to choose the config on"
        )

    rows = paired[paired["metric"] == rank_metric]
    if rows.empty:
        return {}

    outer = [key for key in by if key != held_out]
    choice = fixed_config_choice(rows, outer, held_out=held_out, configs=configs)
    choice = choice[choice["fixed_config"].notna()]

    order = [[*outer, held_out].index(key) for key in by]
    return {
        tuple(np.asarray(keys, dtype=object)[order]): _config_key(values, configs)
        for keys, values in zip(
            choice[[*outer, held_out]].itertuples(index=False, name=None),
            choice[list(configs)].itertuples(index=False, name=None),
        )
    }


def _picked(block: pd.DataFrame, column: str) -> pd.Index:
    """The rows of `block` tied at the top of `column`, empty if it is all NaN.

    Ties are kept rather than broken, matching `selection_regret`: the caller averages
    them, which is the expected value of breaking them uniformly and the honest reading of
    a proxy that cannot separate two models.
    """
    values = block[column]
    top = values.max(skipna=True)
    return block.index[:0] if not np.isfinite(top) else block.index[values == top]


def _label_config(chosen: pd.Index, configs: Sequence[str]) -> str | None:
    """The printable form of a pick, or `None` where nothing was picked.

    A tie is named by its size rather than by listing every config: the value beside it is
    already their average, and a cell holding six comma-separated configs is unreadable.
    """
    if not len(chosen):
        return None
    if len(chosen) > 1:
        return f"{len(chosen)} tied"
    values = chosen[0] if isinstance(chosen[0], tuple) else (chosen[0],)
    return ", ".join(f"{col}={val}" for col, val in zip(configs, values))


def selected_metrics(
    paired: pd.DataFrame,
    *,
    by: Sequence[str] = DEFAULT_BY,
    selectors: Sequence[str] = DEFAULT_SELECTORS,
    metrics: Sequence[str] = DEFAULT_REPORT_METRICS,
    held_out: str = "category",
    rank_metric: str = DEFAULT_RANK_METRIC,
    replicate_key: str = "param_seed",
    min_n: int = 3,
) -> pd.DataFrame:
    """What each selection rule picked, per population, and what that model really scored.

    One row per (population, selector, metric): `value` is the real score of the picked
    config, `best` the real score of the best config in that population *for that metric*,
    and `regret` their difference. `config` names the pick, `n_tied` how many configs the
    rule could not separate at the top (their values are averaged), and `n_models` the size
    of the population the pick was made in.

    `chance` is the mean real score over that same population — what picking uniformly at
    random is worth — and `skill` is `regret` rescaled by the regret of that random pick
    (`1 - regret / (best - chance)`), which is `selection.py`'s definition applied to every
    rule rather than to one proxy. 1 is a perfect pick, 0 is chance, below 0 is worse than
    chance; it is NaN where the real arm is constant, since there every pick is the best
    one and the ratio has no denominator. Regret is in metric units and so cannot be
    compared between two categories whose models sit at different spreads; skill can, at
    the cost of being a ratio — a population whose models are nearly tied has a tiny
    denominator and lands far from 0 on a difference that does not matter.

    Pass the same frame and the same `by` `evaluate_selection` was given, or the
    populations will not line up with the rest of the analysis. Populations with fewer than
    `min_n` configs are skipped rather than reported as a choice between two models.

    A selector naming a metric or a query the frame does not carry yields NaN rows rather
    than raising: a sweep predating a metric should still produce the table for the
    rules it can evaluate.
    """
    parsed = [parse_selector(spec) for spec in selectors]
    metrics = list(metrics)
    keys = population_keys(by)
    missing = [key for key in keys if key not in paired.columns]
    if missing:
        raise KeyError(
            f"group keys {missing} not in paired frame; available: {list(paired.columns)}"
        )

    configs = config_dimensions(paired, keys, replicate_key=replicate_key)
    if not configs:
        raise KeyError("no `param_*` columns to identify a config")

    # One table indexed by (population, config), the real metrics beside the synthetic
    # columns the rules choose on, so every lookup below is a column of the same block.
    table = real_by_config(paired, by=keys, configs=configs, metrics=metrics)
    for selector in parsed:
        if selector.kind == "synth":
            table[_SYNTH + selector.label] = synth_by_config(
                paired, selector, by=keys, configs=configs
            )

    fixed = (
        _fixed_choices(
            paired,
            by=keys,
            configs=configs,
            held_out=held_out,
            rank_metric=rank_metric,
        )
        if any(selector.kind == FIXED for selector in parsed)
        else {}
    )

    rows: list[dict] = []
    for population, group in table.groupby(level=keys, dropna=False, sort=True):
        population = population if isinstance(population, tuple) else (population,)
        # The population levels are constant inside the group; dropping them leaves the
        # config as the index, which is what every pick below is keyed by.
        block = group.droplevel(keys)
        if len(block) < min_n:
            continue
        labelled = dict(zip(keys, population))

        for selector in parsed:
            chosen = _chosen(selector, block, fixed.get(population))
            for metric in metrics:
                # The oracle is the argmax of the metric being reported, so unlike every
                # other rule it is resolved per column rather than once per population.
                picked = _picked(block, metric) if selector.kind == ORACLE else chosen
                best = block[metric].max(skipna=True)
                # `selection.py`'s `real_mean`: the expected score of picking uniformly
                # at random in this population, over the same models `best` is the max
                # of and skipping the same non-finite ones.
                chance = block[metric].mean(skipna=True)
                value = float(block.loc[picked, metric].mean()) if len(picked) else _NAN
                regret = (
                    float(best) - value
                    if np.isfinite(best) and np.isfinite(value)
                    else _NAN
                )
                # `regret_random`, the scale skill is regret in. Not finite and positive
                # exactly where the real arm is constant, which is where no regret number
                # means anything and skill is NaN rather than 0.
                spread = float(best) - float(chance) if np.isfinite(chance) else _NAN
                rows.append(
                    {
                        **labelled,
                        "selector": selector.label,
                        "metric": metric,
                        "value": value,
                        "best": float(best) if np.isfinite(best) else _NAN,
                        "chance": float(chance) if np.isfinite(chance) else _NAN,
                        "regret": regret,
                        "skill": 1.0 - regret / spread
                        if np.isfinite(regret) and spread > 0.0
                        else _NAN,
                        "n_models": len(block),
                        "n_tied": len(picked),
                        "config": _label_config(picked, configs),
                    }
                )

    return pd.DataFrame(rows, columns=[*keys, "selector", "metric", *SELECTED_FIELDS])


def _chosen(selector: Selector, block: pd.DataFrame, fixed) -> pd.Index:
    """The config(s) `selector` picks in one population, as an index into `block`.

    Empty where the rule could not be applied — a synthetic column the sweep never
    produced, a fixed config never run on this category. That becomes a NaN row, which is
    not the same as a rule that picked badly.
    """
    if selector.kind == ORACLE:
        return block.index[:0]  # resolved per metric by the caller
    if selector.kind == FIXED:
        return (
            block.index[block.index == fixed] if fixed is not None else block.index[:0]
        )
    column = _SYNTH + selector.label
    return _picked(block, column) if column in block.columns else block.index[:0]


def summarise_selected(
    selected: pd.DataFrame, by: Sequence[str] = ("set_name", "category")
) -> pd.DataFrame:
    """The per-category table: each rule's mean achieved score over its populations.

    Averaged over the training-set populations of a category — every seed and shot count
    the sweep ran — so `value` is what the rule gets on average across the grid rather than
    on any one run. `value_sd` is the spread over those populations, which is mostly the
    shot count and is why the mean alone is not a model's score; `regret` is the mean gap
    to each population's own ceiling, which is the number that compares two rules.

    `skill` is the mean of the per-population skills, not `1 - mean regret / mean spread`:
    each population is one decision and they weigh the same. It is averaged over the
    populations where it is defined — `n_skilled`, which is below `n_scored` wherever a
    population's real arm was constant.

    Aggregates whichever of `SUMMARY_FIELDS`' sources the frame carries, so a
    `selected.parquet` written before `chance` and `skill` existed still summarises into
    the columns it can fill.
    """
    keys = [key for key in by if key in selected.columns] + ["selector", "metric"]
    if selected.empty:
        return pd.DataFrame(columns=[*keys, *SUMMARY_FIELDS])

    aggregates = {
        "value": ("value", "mean"),
        "value_sd": ("value", "std"),
        "best": ("best", "mean"),
        "chance": ("chance", "mean"),
        "regret": ("regret", "mean"),
        "regret_sd": ("regret", "std"),
        "skill": ("skill", "mean"),
        "skill_sd": ("skill", "std"),
        "n_populations": ("value", "size"),
        "n_scored": ("value", "count"),
        "n_skilled": ("skill", "count"),
    }
    return (
        selected.groupby(keys, dropna=False, sort=False)
        .agg(
            **{
                name: spec
                for name, spec in aggregates.items()
                if spec[0] in selected.columns
            }
        )
        .reset_index()
    )


def selector_order(frame: pd.DataFrame) -> list[str]:
    """Selector labels in the order they were evaluated, not alphabetically.

    `DEFAULT_SELECTORS` is ordered as an argument — the free baseline, then the rules that
    cost something, then the ceiling — and sorting would put `image_auroc@all` first and
    `oracle` in the middle.
    """
    return list(dict.fromkeys(frame["selector"]))


def wide_table(
    summary: pd.DataFrame,
    *,
    value: str = "value",
    index: str = "category",
    metrics: Sequence[str] = DEFAULT_REPORT_METRICS,
) -> pd.DataFrame:
    """`summarise_selected` pivoted to the printable shape: categories down, rules across.

    Columns are a (metric, selector) MultiIndex in the order both were given, so the
    metrics stay in blocks and the selectors keep their argument order inside each.

    `dropna=False`, so a rule that could not be applied anywhere keeps its column of NaN
    instead of vanishing from the table — "we tried this and it never applied" and "we
    never tried it" are different results, and the figure hatches the first.

    Anything the summary is keyed by beyond `index` is averaged into the cell, so filter
    to one `set_name` first if the directory holds more than one synthetic set.
    """
    if summary.empty:
        return pd.DataFrame()

    grid = summary.pivot_table(
        index=index,
        columns=["metric", "selector"],
        values=value,
        sort=False,
        dropna=False,
    )
    wanted = [
        (metric, selector)
        for metric in metrics
        for selector in selector_order(summary)
        if (metric, selector) in grid.columns
    ]
    if not wanted:
        return pd.DataFrame()
    return grid.reindex(columns=pd.MultiIndex.from_tuples(wanted)).sort_index()


def _number(value: float, digits: int) -> str:
    """One cell at printing precision, with negative zero normalised away.

    `regret` is `best - value`, and the oracle's is zero by construction — but the
    subtraction lands on -1.1e-16 often enough that the column prints `-0.000`. A
    negative regret is impossible, so that reads as a result rather than as the float
    artefact it is. Anything that rounds to zero prints as zero.
    """
    return f"{round(value, digits) + 0.0:.{digits}f}"


def format_table(
    summary: pd.DataFrame,
    *,
    value: str = "value",
    index: str = "category",
    metrics: Sequence[str] = DEFAULT_REPORT_METRICS,
    digits: int = 3,
) -> str:
    """`wide_table` as fixed-width text, one block per metric, for stdout.

    One block per metric rather than one grid: fifteen categories by five rules by three
    metrics is 225 numbers, and a table that wide wraps in any terminal.
    """
    grid = wide_table(summary, value=value, index=index, metrics=metrics)
    if grid.empty:
        return "(no selections to report)"

    width = max(len(str(name)) for name in grid.index)
    blocks = []
    for metric in dict.fromkeys(level for level, _ in grid.columns):
        block = grid[metric]
        columns = [str(col) for col in block.columns]
        widths = [max(len(col), digits + 2) for col in columns]
        header = "  ".join(col.rjust(w) for col, w in zip(columns, widths))
        lines = [metric, f"{'':{width}}  {header}"]
        for name, row in block.iterrows():
            cells = "  ".join(
                (_number(v, digits) if np.isfinite(v) else "—").rjust(w)
                for v, w in zip(row, widths)
            )
            lines.append(f"{str(name):{width}}  {cells}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


# The mean row's label. Not a category, so it is separated by a rule and left out of the
# ranking's input — it is an average of the rows above it, not a sixteenth measurement.
MEAN_ROW = "mean"

# What each metric is called in the table's column heads. The parquet keys are the
# pipeline's names; a results table is read next to the anomaly-detection literature,
# which prints these three as I-AUC, P-AUC and PRO. Display only — nothing keys off it,
# and a metric with no entry keeps its own name.
METRIC_LABELS: dict[str, str] = {
    "image_auroc": "I-AUC",
    "pixel_auroc": "P-AUC",
    "aupro": "PRO",
}

# LaTeX-special characters that appear in a metric, selector or category name. `@` and
# `-` are safe in text mode; `_` is the one this table actually hits (`metal_nut`,
# `image_auroc`), and the rest are here so a renamed column cannot silently emit
# uncompilable source.
_LATEX_ESCAPES = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def escape_latex(text: object) -> str:
    """One name as LaTeX text: `metal_nut` into `metal\\_nut`.

    Public because a caller building the float's caption around a rule's name needs the
    same escaping the table's own cells get — a bare `image_auroc@all` in a caption is
    a compile error, and a silently different one from the header it describes.
    """
    return "".join(_LATEX_ESCAPES.get(char, char) for char in str(text))


def _metric_head(metric: str) -> str:
    """The column head for one metric: its paper name where it has one, else its key."""
    return escape_latex(METRIC_LABELS.get(metric, metric))


def _marks(
    row: pd.Series,
    exclude: set[str],
    digits: int,
    *,
    higher_is_better: bool = True,
) -> dict[object, str]:
    """`{selector: "best" | "second"}` for one row of one metric.

    Ranked on the **rounded** values, so two cells that print the same number are marked
    the same: `image_auroc@minimal` and `image_auroc@all` can be the identical column on a
    corpus where every grade ranked the same, and bolding one of them
    over a difference in the fifteenth decimal would be an artefact.

    `exclude` keeps the oracle out of the comparison — it reads the answer, so it wins by
    construction and would take the bold off whichever rule actually did best.

    `higher_is_better=False` is the regret table: the same float means the opposite there,
    and marking the largest gap as the winner would invert the result.

    Fewer than two comparable cells are left unmarked. A bold on the only column that
    could be filled is not a comparison, and it reads as one — the achieved-level table
    against the oracle alone is exactly that case.
    """
    ranked = row.drop(
        labels=[key for key in row.index if key in exclude], errors="ignore"
    )
    ranked = ranked[np.isfinite(ranked.astype(float))].round(digits)
    if len(ranked) < 2:
        return {}

    distinct = sorted(set(ranked), reverse=higher_is_better)
    places = dict(zip(distinct[:2], ("best", "second")))
    return {key: places[val] for key, val in ranked.items() if val in places}


def _latex_cells(
    block: pd.Series, exclude: set[str], digits: int, *, higher_is_better: bool
) -> list[str]:
    """One row of one ranked block as LaTeX cells: bold, underline, plain or `--`."""
    marked = {"best": r"\textbf{%s}", "second": r"\underline{%s}"}
    places = _marks(block, exclude, digits, higher_is_better=higher_is_better)
    return [
        marked.get(places.get(selector), "%s") % _number(cell, digits)
        if np.isfinite(cell)
        else "--"
        for selector, cell in block.items()
    ]


def _latex_head(selector: str, selector_labels: Mapping[str, str] | None) -> str:
    """A column head: the caller's label as written, else the escaped selector name."""
    named = (selector_labels or {}).get(selector)
    if named is not None:
        return named
    head = escape_latex(selector)
    # Two lines only where there is a `@` to break at: `image_auroc@all` is wide
    # enough to set the column width on its own, `fixed` is not.
    return r"\shortstack{%s}" % head.replace("@", r"\\@") if "@" in head else head


def latex_table(
    summary: pd.DataFrame,
    *,
    value: str = "value",
    index: str = "category",
    metrics: Sequence[str] = DEFAULT_REPORT_METRICS,
    digits: int = 3,
    exclude_from_ranking: Sequence[str] = (ORACLE,),
    selector_labels: Mapping[str, str] | None = None,
    higher_is_better: bool = True,
    mean_row: bool = True,
    caption: str | None = None,
    label: str | None = None,
) -> str:
    """The per-category table as a `booktabs` float: best in bold, runner-up underlined.

    Rows are categories with the metrics as `\\multicolumn` groups, which is the shape an
    anomaly-detection results table is usually read in; the group heads print the names
    that literature uses (`METRIC_LABELS`: I-AUC, P-AUC, PRO), not the parquet keys.
    Within one row and one metric the highest value is `\\textbf` and the next distinct
    one `\\underline`, ranked on the printed precision (see `_marks`) and with the oracle
    held out of the comparison — it is the ceiling, not a rule anyone can run, so leaving
    it in would put the bold on the same column in every row and say nothing.

    The `mean` row is the unweighted mean over the categories, which is the average every
    MVTec table quotes; it is ranked like any other row but takes no part in ranking them.
    It skips missing cells, so a rule with a gap is averaged over the categories it *did*
    cover — read it next to the `--`s in its column. A cell no rule could fill prints `--`.

    `higher_is_better=False` inverts the marking, for a `value="regret"` table where the
    small number is the good one. Everything else is unchanged: the same column is still
    held out of the ranking, and the mean is still the mean.

    With one selector per metric the group heads collapse into a single header row of
    metric names — a `\\multicolumn{1}` group over its own only column, with the rule's
    name repeated underneath it, is noise, and the caption is where that one rule belongs.

    `selector_labels` renames the column heads: `{"fixed": "F"}` heads that column `F`
    instead of `\\texttt{fixed}`, for a paper that refers to the rules by symbol. Values are
    LaTeX the caller wrote and go through **unescaped**, so `IAC$_w$` sets a subscript
    rather than printing four literal characters; a selector with no entry keeps its own
    name, escaped and broken across two lines as before. Whatever a column is headed, the
    rule it stands for belongs in the caption — a table of symbols is unreadable without
    the key.

    `caption` and `label` are written through unescaped: they are LaTeX the caller wrote,
    and escaping them would break a caption that meant its markup.

    Needs `\\usepackage{booktabs}` and nothing else — the two-line column heads are
    `\\shortstack`, which is plain LaTeX.
    """
    grid = wide_table(summary, value=value, index=index, metrics=metrics)
    if grid.empty:
        return "% no selections to report\n"

    metrics = list(dict.fromkeys(level for level, _ in grid.columns))
    if mean_row:
        grid = pd.concat([grid, grid.mean(axis=0).to_frame(MEAN_ROW).T])

    excluded = set(exclude_from_ranking)
    body = []
    for position, (name, row) in enumerate(grid.iterrows()):
        cells = []
        for metric in metrics:
            cells += _latex_cells(
                row[metric], excluded, digits, higher_is_better=higher_is_better
            )
        body.append(f"{escape_latex(name)} & " + " & ".join(cells) + r" \\")
        # By position, not by label: the mean is the last row, and a category that
        # happened to share its neighbour's name would otherwise draw the rule twice.
        if mean_row and position == len(grid) - 2:
            body.append(r"\midrule")

    per_metric = len(grid[metrics[0]].columns)
    groups, rules, heads = [], [], []
    for position, metric in enumerate(metrics):
        first = 2 + position * per_metric
        groups.append(rf"\multicolumn{{{per_metric}}}{{c}}{{{_metric_head(metric)}}}")
        rules.append(rf"\cmidrule(lr){{{first}-{first + per_metric - 1}}}")
        heads += [_latex_head(name, selector_labels) for name in grid[metric].columns]

    # One column per metric needs one header row, headed by the metrics themselves.
    head_rows = (
        [
            f"{escape_latex(index)} & "
            + " & ".join(_metric_head(m) for m in metrics)
            + r" \\"
        ]
        if per_metric == 1
        else [
            f"& {' & '.join(groups)} " + r"\\",
            "".join(rules),
            f"{escape_latex(index)} & " + " & ".join(heads) + r" \\",
        ]
    )

    lines = [
        r"% requires \usepackage{booktabs}",
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{caption}}}" if caption else None,
        rf"\label{{{label}}}" if label else None,
        r"\small",
        rf"\begin{{tabular}}{{l{'r' * (len(metrics) * per_metric)}}}",
        r"\toprule",
        *head_rows,
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(line for line in lines if line is not None) + "\n"


def _row_head(name: object, index_labels: Mapping[object, str] | None) -> str:
    """A row's (or a transposed column's) label: the caller's LaTeX, else the escaped name.

    For a category too wide for its column — `toothbrush` sets the width of a narrow
    float on its own — so the paper can print `toothb.` without renaming the frame.
    """
    named = (index_labels or {}).get(name)
    return named if named is not None else escape_latex(name)


def _begin_tabular(spec: str, position: str | None) -> str:
    """`\\begin{tabular}`, set to sit beside another one wherever `position` is given.

    Beside another tabular it is aligned by `position` (`t`: the top rules line up), and
    it loses its outer padding (`@{}`). The padding would only add to `latex_float`'s
    `gap`, and the outer edges then sit flush with the column. In a single column of the
    paper, those four paddings are worth 8pt of width.
    """
    if not position:
        return rf"\begin{{tabular}}{{{spec}}}"
    return rf"\begin{{tabular}}[{position}]{{@{{}}{spec}@{{}}}}"


def latex_float(
    tabulars: Sequence[Sequence[str]],
    *,
    caption: str | None = None,
    label: str | None = None,
    column_sep: str | None = None,
    gap: str = r"\hspace{1em}",
    vertical: bool = False,
) -> str:
    """One float around `tabulars` — the lines of each `*_tabular` — side by side.

    Side by side because the lines are joined with `%` and `gap`, never with a blank line,
    which would be a paragraph break and stack them. `gap` is preceded by `\\nobreak`:
    it is glue, and TeX would otherwise break the line there as soon as the tabulars
    together ran a fraction of a point past the column, stacking them. What is wanted in
    that case is an overfull-box warning, not a different table. Tabulars meant to share
    a line should be built with `position="t"`, so their `\\toprule`s align however tall
    each one runs. That also drops their outer padding, so `gap` is the whole distance
    between them, both between their rules and between their text.

    `vertical` stacks them instead, one centred paragraph each with `gap` (a vertical
    length then, e.g. `\\medskip`) between: for tabulars that are each narrower than the
    column but not together.

    `column_sep` is `\\tabcolsep` set inside the float, so it narrows this table alone;
    `None` keeps the document's.
    """
    body = []
    for position, tabular in enumerate(tabulars):
        if position and vertical:
            body.append(rf"\par{gap}")
        elif position:
            body[-1] += "%"
            body.append(rf"\nobreak{gap}%")
        body.extend(tabular)
    lines = [
        r"% requires \usepackage{booktabs}",
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{caption}}}" if caption else None,
        rf"\label{{{label}}}" if label else None,
        r"\small",
        rf"\setlength{{\tabcolsep}}{{{column_sep}}}" if column_sep else None,
        *body,
        r"\end{table}",
    ]
    return "\n".join(line for line in lines if line is not None) + "\n"


def stacked_tabular(
    summary: pd.DataFrame,
    *,
    value: str = "value",
    index: str = "n",
    metrics: Sequence[str] = DEFAULT_REPORT_METRICS,
    digits: int = 3,
    exclude_from_ranking: Sequence[str] = (ORACLE,),
    selector_labels: Mapping[str, str] | None = None,
    higher_is_better: bool = True,
    mean_row: bool = True,
    position: str | None = None,
) -> list[str]:
    """`latex_table`'s metric groups stacked down the rows instead of across the columns.

    One block per metric, in `metrics` order. Each block is the metric's paper name in the
    row-label column, then one row per `index` value below it, then the block's own `mean`
    row. The rules are the columns, once: three rules over three metrics is three columns
    rather than nine, which is what lets the levels table sit beside another one in a
    single column of the paper.

    The first metric's name is the head of the row-label column, on the row with the rule
    heads, where `latex_table` would print the index name. That name is not needed: the
    row labels of a table by training-set size are obviously training-set sizes. Every
    later metric's name gets a row of its own, in the same column. So each block costs
    one row beyond its values, rather than a group head and a rule.

    Ranked per row as in `latex_table`, with the oracle held out; a block's mean row is
    the mean of that block's rows.

    Returns the tabular's lines, for `latex_float`.
    """
    grid = wide_table(summary, value=value, index=index, metrics=metrics)
    if grid.empty:
        raise ValueError(f"no {value} to report for {', '.join(metrics)}")

    metrics = list(dict.fromkeys(level for level, _ in grid.columns))
    selectors = [s for s in selector_order(summary) if (metrics[0], s) in grid.columns]
    width = len(selectors)
    excluded = set(exclude_from_ranking)

    heads = [_latex_head(name, selector_labels) for name in selectors]
    # A `\multicolumn` replaces the spec of the columns it spans, so it has to repeat the
    # `@{}` a tabular set beside another one has, or its name sits indented by a padding.
    label_align = "@{}l" if position else "l"
    lines = [_begin_tabular(f"l{'r' * width}", position), r"\toprule"]
    for block_position, metric in enumerate(metrics):
        if block_position:
            name = _metric_head(metric)
            lines += [
                r"\midrule",
                rf"\multicolumn{{{1 + width}}}{{{label_align}}}{{{name}}} \\",
            ]
        else:
            lines += [
                f"{_metric_head(metric)} & " + " & ".join(heads) + r" \\",
                r"\midrule",
            ]

        block = grid[metric].reindex(columns=selectors)
        if mean_row:
            block = pd.concat([block, block.mean(axis=0).to_frame(MEAN_ROW).T])
        for name, row in block.iterrows():
            cells = _latex_cells(
                row, excluded, digits, higher_is_better=higher_is_better
            )
            lines.append(f"{escape_latex(name)} & " + " & ".join(cells) + r" \\")

    return [*lines, r"\bottomrule", r"\end{tabular}"]


def grid_tabular(
    grid: pd.DataFrame,
    *,
    digits: int = 3,
    corner: str = "",
    title: str | None = None,
    column_labels: Mapping[object, str] | None = None,
    row_labels: Mapping[object, str] | None = None,
    rank: str | None = "rows",
    higher_is_better: bool = True,
    exclude_from_ranking: Sequence[str] = (ORACLE,),
    rules_before: Sequence[int] = (),
    last_column_apart: bool = False,
    position: str | None = None,
) -> list[str]:
    """A grid the caller has already aggregated, as a tabular: its index down, columns across.

    For tables whose rows are not one axis of `summarise_selected`, such as a query's
    regret by training-set size followed by its median and max over categories. The
    caller builds the numbers; this sets them the way the other tables here are set.

    `rank="rows"` marks each row as `latex_table` does: best in bold and runner-up
    underlined, on the printed precision, with `exclude_from_ranking` held out.
    `rank="columns"` does the same down each column, for a grid whose competitors are its
    rows, such as queries against summary statistics. `rank=None` marks nothing, for a
    grid with no competitors, such as modes against severities. `rules_before` puts a
    `\\midrule` before each of those row positions. `last_column_apart` puts a vertical
    rule before the last column, for an `all` or `mean` rollup. `title` heads the value
    columns (unescaped LaTeX) on a row of its own above the column heads, and `corner`
    is the cell left of those heads.

    `column_labels` and `row_labels` are LaTeX, written unescaped. Unlabelled columns get
    `latex_table`'s head and unlabelled rows their escaped name. A cell with no value
    prints `--`.

    Returns the tabular's lines, for `latex_float`.
    """
    if grid.empty:
        raise ValueError("an empty grid has nothing to report")

    width = len(grid.columns)
    spec = "l" + "r" * (width - 1) + ("|r" if last_column_apart else "r")
    lines = [_begin_tabular(spec, position), r"\toprule"]
    if title:
        lines += [
            rf"& \multicolumn{{{width}}}{{c}}{{{title}}} \\",
            rf"\cmidrule(lr){{2-{1 + width}}}",
        ]
    heads = [
        (column_labels or {}).get(name, _latex_head(str(name), None))
        for name in grid.columns
    ]
    lines += [f"{corner} & " + " & ".join(heads) + r" \\", r"\midrule"]

    excluded = set(exclude_from_ranking)
    if rank == "rows":
        cells = [
            _latex_cells(row, excluded, digits, higher_is_better=higher_is_better)
            for _, row in grid.iterrows()
        ]
    elif rank == "columns":
        # Ranked down each column, then read back across: the same marks, turned.
        by_column = [
            _latex_cells(column, excluded, digits, higher_is_better=higher_is_better)
            for _, column in grid.items()
        ]
        cells = [list(row) for row in zip(*by_column)]
    elif rank is None:
        cells = [
            [_number(v, digits) if np.isfinite(v) else "--" for v in row]
            for _, row in grid.iterrows()
        ]
    else:
        raise ValueError(f"rank {rank!r} is not 'rows', 'columns' or None")

    breaks = set(rules_before)
    for row_position, (name, row_cells) in enumerate(zip(grid.index, cells)):
        if row_position in breaks:
            lines.append(r"\midrule")
        lines.append(
            f"{_row_head(name, row_labels)} & " + " & ".join(row_cells) + r" \\"
        )
    return [*lines, r"\bottomrule", r"\end{tabular}"]


def panels_tabular(
    panels: Sequence[tuple[str, pd.DataFrame]],
    *,
    value: str = "value",
    index: str = "category",
    metric: str = DEFAULT_RANK_METRIC,
    digits: int = 3,
    exclude_from_ranking: Sequence[str] = (ORACLE,),
    selector_labels: Mapping[str, str] | None = None,
    index_labels: Mapping[object, str] | None = None,
    higher_is_better: bool = True,
    mean_row: bool = True,
    position: str | None = None,
) -> list[str]:
    """`latex_panels`' tabular without its float, for `latex_float` to set beside another.

    `index_labels` renames the row labels (unescaped LaTeX); every other argument is
    `latex_panels`'.
    """
    grids = []
    for title, summary in panels:
        grid = wide_table(summary, value=value, index=index, metrics=[metric])
        if grid.empty:
            raise ValueError(f"panel {title!r} has no {metric} selections to report")
        grids.append(grid[metric])
    if not grids:
        raise ValueError("no panels to report")

    joined = pd.concat(grids, axis=1, keys=range(len(grids))).sort_index()
    if mean_row:
        joined = pd.concat([joined, joined.mean(axis=0).to_frame(MEAN_ROW).T])

    excluded = set(exclude_from_ranking)
    body = []
    for row_position, (name, row) in enumerate(joined.iterrows()):
        cells = []
        for panel in range(len(grids)):
            cells += _latex_cells(
                row[panel], excluded, digits, higher_is_better=higher_is_better
            )
        body.append(f"{_row_head(name, index_labels)} & " + " & ".join(cells) + r" \\")
        if mean_row and row_position == len(joined) - 2:
            body.append(r"\midrule")

    groups, rules, heads, spec = [], [], [], ["l"]
    first = 2
    for panel, ((title, _), grid) in enumerate(zip(panels, grids)):
        width = len(grid.columns)
        # A `\multicolumn` replaces the column's own spec, `|` included, so the group
        # head has to carry the rule itself or the line breaks in this row.
        align = "c|" if panel < len(grids) - 1 else "c"
        groups.append(rf"\multicolumn{{{width}}}{{{align}}}{{{title}}}")
        rules.append(rf"\cmidrule(lr){{{first}-{first + width - 1}}}")
        heads += [_latex_head(name, selector_labels) for name in grid.columns]
        spec.append("r" * width)
        first += width

    return [
        _begin_tabular(f"{spec[0]}{'|'.join(spec[1:])}", position),
        r"\toprule",
        f"& {' & '.join(groups)} " + r"\\",
        "".join(rules),
        f"{escape_latex(index)} & " + " & ".join(heads) + r" \\",
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
    ]


def latex_panels(
    panels: Sequence[tuple[str, pd.DataFrame]],
    *,
    value: str = "value",
    index: str = "category",
    metric: str = DEFAULT_RANK_METRIC,
    digits: int = 3,
    exclude_from_ranking: Sequence[str] = (ORACLE,),
    selector_labels: Mapping[str, str] | None = None,
    higher_is_better: bool = True,
    mean_row: bool = True,
    caption: str | None = None,
    label: str | None = None,
    column_sep: str | None = None,
) -> str:
    """Several one-metric `latex_table`s side by side, sharing rows, caption and label.

    `panels` is `(title, summary)` in column order: each summary becomes a group of columns
    headed by its title (LaTeX, written through unescaped like `selector_labels`), and a
    vertical rule separates one group from the next. Rows are the union of the panels'
    `index` values, sorted, so a category one panel lacks prints `--` there rather than
    dropping out of the other.

    Each panel is ranked on its own. Two panels are two comparisons — typically measured
    over different populations — and a bold that crossed the rule would claim one rule beat
    another it was never compared with. The `mean` row is each panel's mean over the rows
    it fills, as in `latex_table`.

    The vertical rule is plain `|`, so booktabs' padding leaves small gaps where it meets
    `\\toprule`, `\\midrule` and `\\bottomrule`; nothing beyond `booktabs` is needed.

    `column_sep` is a LaTeX length (`3pt`) set as `\\tabcolsep` inside the float, so it
    narrows the padding either side of every column — the rule's included — in this table
    alone. `None` leaves the document's default, 6pt unless the preamble changed it.
    """
    if not panels:
        return "% no selections to report\n"
    tabular = panels_tabular(
        panels,
        value=value,
        index=index,
        metric=metric,
        digits=digits,
        exclude_from_ranking=exclude_from_ranking,
        selector_labels=selector_labels,
        higher_is_better=higher_is_better,
        mean_row=mean_row,
    )
    return latex_float([tabular], caption=caption, label=label, column_sep=column_sep)
