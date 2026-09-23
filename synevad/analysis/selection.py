"""Whether picking a model by its synthetic score actually gets you a good model.

`correlation.py` answers a different question — whether the two arms co-vary across a
population. That is necessary but not sufficient for a model-selection claim: Kendall's
tau weights the (worst, second-worst) pair exactly like (best, second-best), so a high
tau is compatible with always choosing the wrong model, and a mediocre one with always
choosing the right one. Selection is a property of the top of the ranking and needs its
own statistic.

`selection_regret` is the single-population statistic: how much real score is given up by
taking the synthetic arm's argmax instead of the real arm's. `evaluate_selection` applies
it per group, mirroring `correlate`.

Regret on its own still does not support "synthetic data helps pick the best model" —
"helps" is comparative, and a regret of 0.003 AUROC may be either impressive or vacuous.
The rest of the module supplies what turns it into a claim:

* `skill` normalises regret by that of a uniformly random pick: 0 means the synthetic
  ranking is worth no more than a coin flip, 1 means it is perfect, negative means it is
  actively misleading.
* `replicate_noise` estimates how far the real score moves across seeds alone. A regret
  inside that band means the decision was never decidable, whatever the correlation said.
* `fixed_config_baseline` is the competitor that matters in practice: always use one
  config, chosen on the other categories, with no synthetic data at all.
* `leave_one_out_query` keeps the choice of *which* synthetic query to trust out of
  sample, since scanning queries and reporting the best is proxy selection on the very
  data it is then evaluated on.
* `aggregate_replicates` collapses seed replicates before any of the above, so that
  "best model" does not quietly mean "luckiest seed".
* `conditioned_by` puts the training-set columns into the grouping instead, so that a
  population is the models that saw the *same* images. Selecting across training sets
  asks the proxy to spot which draw was luckier and how many shots it had — questions
  about the data, not about the model — and the shot axis dominates the real spread, so
  a proxy that only ever learned "more shots is better" scores near-perfect skill.
  `evaluate_per_training_set` is that grouping plus the noise floor in one call.

Like `correlation.py`, nothing here warns: degenerate populations return NaN fields so a
scan over a few hundred groups stays readable.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy import stats

# `set_name` is in the default grouping and is *not* optional: `pair_runs` emits one row
# per synthetic set by design, so pooling sets would enter the same model several times
# into one population and inflate n.
DEFAULT_BY: tuple[str, ...] = ("metric", "set_name", "query", "category")

# The columns that say *which images a model was trained on*: `make_mvtecad_dataset` draws
# the subset with `np.random.choice` under the global seed, so the draw is fixed by the
# seed and its size by `num_train_samples`. Both are grouping keys, not model identity —
# see `conditioned_by`. Names follow `build_paired_metrics`, which renames `params.<k>` to
# `param_<k>` and `apply_overrides`, which flattens the dotted key to `data_num_train_samples`.
TRAINING_SET_KEYS: tuple[str, ...] = ("param_seed", "param_data_num_train_samples")

# Fixed key set of `selection_regret`, so `pd.DataFrame.from_records` gets a stable
# schema no matter which branch each group took.
SELECTION_FIELDS: tuple[str, ...] = (
    "n",
    "n_dropped",
    "k",
    "alpha",
    "regret",
    "regret_at_k",
    "hit",
    "hit_at_k",
    "skill",
    "skill_at_k",
    "real_best",
    "real_chosen",
    "real_mean",
    "real_worst",
    "regret_random",
    "regret_worst",
    "synth_rank_of_best",
    "n_tied_at_top",
    "regret_ci_lo",
    "regret_ci_hi",
    "skill_ci_lo",
    "skill_ci_hi",
)

_NAN = float("nan")


def _bootstrap_ci(
    real: np.ndarray,
    synth: np.ndarray,
    *,
    n_boot: int,
    alpha: float,
    rng: np.random.Generator | int | None,
) -> dict[str, float]:
    """Percentile intervals for `regret` and `skill`, resampling models with replacement.

    Resamples *rows*, so a frame that still holds seed replicates would be resampled at
    the replicate level and report an interval that is too narrow — run
    `aggregate_replicates` first. `real_best` is recomputed inside each resample, which is
    the point: with eight configs the identity of the best model is itself uncertain.
    """
    n = real.size
    generator = np.random.default_rng(rng)
    idx = generator.integers(0, n, size=(n_boot, n))
    boot_real, boot_synth = real[idx], synth[idx]

    best = boot_real.max(axis=1)
    mean = boot_real.mean(axis=1)
    top = boot_synth == boot_synth.max(axis=1, keepdims=True)
    chosen = (boot_real * top).sum(axis=1) / top.sum(axis=1)

    regret = best - chosen
    random = best - mean
    # A resample that drew one model repeatedly has nothing to choose between; leaving
    # those draws out of the skill interval beats reporting an infinite ratio.
    skill = np.where(
        random > 0.0, 1.0 - regret / np.where(random > 0.0, random, 1.0), _NAN
    )

    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {
        "regret_ci_lo": float(np.percentile(regret, lo)),
        "regret_ci_hi": float(np.percentile(regret, hi)),
        "skill_ci_lo": float(np.nanpercentile(skill, lo))
        if np.isfinite(skill).any()
        else _NAN,
        "skill_ci_hi": float(np.nanpercentile(skill, hi))
        if np.isfinite(skill).any()
        else _NAN,
    }


def selection_regret(
    real: ArrayLike,
    synth: ArrayLike,
    *,
    k: int = 1,
    alpha: float = 0.05,
    min_n: int = 4,
    n_boot: int = 0,
    rng: np.random.Generator | int | None = None,
) -> dict[str, float]:
    """Real score forgone by selecting on the synthetic arm instead of the real one.

    `regret` is `real[argmax real] - real[argmax synth]`, the headline number: zero means
    the synthetic set picked a model that is real-optimal. Ties on the synthetic arm are
    resolved by *averaging* the real scores of the tied models, i.e. the expected regret
    of breaking ties uniformly. That makes a constant synthetic arm score exactly as well
    as a random pick rather than undefined, which is the right reading of it.

    `regret_at_k` is the same for the workflow that actually gets used — shortlist the top
    k by synthetic score, then validate those k for real. Its ties are broken
    *adversarially* (among equal synthetic scores the lower real score is taken first), so
    it is a worst case where `regret` is an average case, and `regret_at_k >= regret` at
    k=1. `k` is clamped to the population size.

    `skill` rescales regret against `regret_random = real_best - real_mean`, the expected
    regret of choosing uniformly at random. 1 is perfect selection, 0 is no better than
    chance, below 0 is worse than chance. It is NaN exactly when the real arm is constant
    — there the decision is vacuous and no regret number means anything, which is the
    range-restriction failure `proxy_quality` reports as `real_range`.

    `synth_rank_of_best` is where the real-best model sits in the synthetic ranking (1 is
    best), using the pessimistic rank for ties and the most favourable of the real-best
    models when the real arm itself ties. It answers "how deep a shortlist would have been
    needed", where `hit_at_k` only answers yes or no for one k.

    Non-finite pairs are dropped pairwise and counted in `n_dropped`. Below `min_n`
    models every field is NaN. Set `n_boot` to get percentile intervals for `regret` and
    `skill` at k=1; off by default so a scan stays cheap and the caller decides.
    """
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")

    real = np.asarray(real, dtype=float).ravel()
    synth = np.asarray(synth, dtype=float).ravel()
    if real.shape != synth.shape:
        raise ValueError(
            f"real and synth differ in length: {real.size} vs {synth.size}"
        )

    keep = np.isfinite(real) & np.isfinite(synth)
    n_dropped = int(keep.size - keep.sum())
    real, synth = real[keep], synth[keep]
    n = int(real.size)

    result: dict[str, float] = {field: _NAN for field in SELECTION_FIELDS}
    result.update({"n": n, "n_dropped": n_dropped, "k": k, "alpha": alpha})
    if n < min_n:
        return result

    real_best, real_mean, real_worst = (
        float(real.max()),
        float(real.mean()),
        float(real.min()),
    )
    regret_random = real_best - real_mean

    # Expected case: average the real scores of everything tied at the synthetic top.
    top = synth == synth.max()
    real_chosen = float(real[top].mean())
    regret = real_best - real_chosen

    # Worst case: -synth is the primary key, real the tie-breaker, so among equal
    # synthetic scores the weakest real model is shortlisted first.
    order = np.lexsort((real, -synth))
    picked = order[: min(k, n)]
    real_at_k = float(real[picked].max())
    regret_at_k = real_best - real_at_k

    # method="max" is the pessimistic rank (a 3-way tie for first ranks 3, not 1); taking
    # the minimum over the real-best models keeps a tie on the real arm from being
    # counted against the proxy, since either model would be a correct pick.
    synth_ranks = stats.rankdata(-synth, method="max")
    synth_rank_of_best = float(synth_ranks[real == real_best].min())

    result.update(
        {
            "regret": regret,
            "regret_at_k": regret_at_k,
            "hit": float((real[top] == real_best).mean()),
            "hit_at_k": float(real_at_k == real_best),
            "skill": 1.0 - regret / regret_random if regret_random > 0.0 else _NAN,
            "skill_at_k": (
                1.0 - regret_at_k / regret_random if regret_random > 0.0 else _NAN
            ),
            "real_best": real_best,
            "real_chosen": real_chosen,
            "real_mean": real_mean,
            "real_worst": real_worst,
            "regret_random": regret_random,
            "regret_worst": real_best - real_worst,
            "synth_rank_of_best": synth_rank_of_best,
            "n_tied_at_top": float(top.sum()),
        }
    )

    if n_boot > 0:
        result.update(_bootstrap_ci(real, synth, n_boot=n_boot, alpha=alpha, rng=rng))
    return result


def evaluate_selection(
    paired: pd.DataFrame,
    by: Sequence[str] = DEFAULT_BY,
    *,
    k: int = 1,
    alpha: float = 0.05,
    min_n: int = 4,
    n_boot: int = 0,
    rng: np.random.Generator | int | None = None,
    min_defect_samples: int | None = None,
) -> pd.DataFrame:
    """`selection_regret` per group of `by`, one row per group.

    The `correlate` analogue, and the same warning applies to widening or narrowing `by`,
    with one addition: dropping `category` pools categories into a single population, and
    a selection made across categories is not a selection anyone would make — you do not
    choose between "the best config for screw" and "the best config for cable". Pooled
    regret mostly measures which category is easiest.

    The same argument runs the other way for the training set: `DEFAULT_BY` is too *narrow*
    when the sweep varies the seed or the shot count, since neither is a knob anyone turns
    at selection time. Pass `conditioned_by(paired)` — or call `evaluate_per_training_set`
    — to keep those populations apart.

    `min_defect_samples` screens out queries that selected too few defects to score,
    mirroring `config.min_num_defects`; off by default so the screening stays visible in
    the caller.
    """
    by = list(by)
    missing = [key for key in by if key not in paired.columns]
    if missing:
        raise KeyError(
            f"group keys {missing} not in paired frame; available: {list(paired.columns)}"
        )

    if min_defect_samples is not None:
        paired = paired[paired["n_defect_samples"] >= min_defect_samples]

    if paired.empty:
        return pd.DataFrame(columns=[*by, *SELECTION_FIELDS])

    rows = []
    # dropna=False for the same reason as `correlate`: a null group key is a legitimate
    # population from an older run, not a row to delete.
    for keys, group in paired.groupby(by, dropna=False, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        rows.append(
            dict(
                zip(by, keys),
                **selection_regret(
                    group["real"],
                    group["synth"],
                    k=k,
                    alpha=alpha,
                    min_n=min_n,
                    n_boot=n_boot,
                    rng=rng,
                ),
            )
        )

    return pd.DataFrame.from_records(rows).astype(
        {"n": "int64", "n_dropped": "int64", "k": "int64"}
    )


def config_keys(
    paired: pd.DataFrame, *, replicate_key: str = "param_seed"
) -> list[str]:
    """The `param_*` columns that identify a config, i.e. every one but the seed."""
    return [
        col
        for col in paired.columns
        if col.startswith("param_") and col != replicate_key
    ]


def training_set_keys(
    paired: pd.DataFrame, *, keys: Sequence[str] = TRAINING_SET_KEYS
) -> list[str]:
    """Those of `keys` the frame actually carries, in order.

    Missing ones are not an error: a sweep that never varied the shot count, or an older
    run with no `params.seed`, simply has one training set per category and nothing to
    condition on.
    """
    return [key for key in keys if key in paired.columns]


def conditioned_by(
    paired: pd.DataFrame,
    by: Sequence[str] = DEFAULT_BY,
    *,
    keys: Sequence[str] = TRAINING_SET_KEYS,
) -> list[str]:
    """`by` widened with the training-set columns: one population per training set.

    The grouping to hand `evaluate_selection` when the claim is about model selection
    rather than about data. Left as it is, `by` pools every seed and every shot count into
    one population, so `real_best` is whichever config got the most images and the
    luckiest draw of them, and the regret being reported is mostly the cost of not knowing
    that 16 shots beat 1 — which the proxy is not being asked to advise on. Conditioning
    puts each training set in its own population, where the only thing separating the
    models is the config.

    The price is population size: with a sweep of 3 x 2 model configs x 5 shot counts x 3
    seeds, pooling gives one population of 90 and conditioning gives 15 of 6. `min_n` is
    the guard; below it every field is NaN rather than a statistic over four points.

    Keys the frame does not have are skipped, so this is a no-op on a sweep with a single
    training set and safe to apply unconditionally.
    """
    by = list(by)
    return by + [key for key in training_set_keys(paired, keys=keys) if key not in by]


def aggregate_replicates(
    paired: pd.DataFrame,
    *,
    replicate_key: str = "param_seed",
    keep: Sequence[str] = ("category", "version", "set_name", "query", "metric"),
) -> pd.DataFrame:
    """Collapse seed replicates to one row per config, averaging `real` and `synth`.

    `default_group_keys` folds every swept param into the model identity, seed included,
    so a sweep of 8 configs x 3 seeds reaches the analysis as 24 "models". Selecting over
    those 24 measures partly whether the synthetic set tracks seed noise, and `real_best`
    becomes the luckiest seed of the luckiest config rather than the best config — a
    winner's curse that inflates every regret in the frame. Averaging first makes the
    population the 8 configs it really is.

    Run this before `evaluate_selection` whenever the sweep has replicates *and* the
    populations pool them; use `replicate_noise` on the *un*aggregated frame to find out
    how much was averaged away. It is the wrong move under `conditioned_by`, which keeps
    the seeds apart as populations rather than folding them into the model identity —
    averaging first would delete the column that grouping needs.
    """
    if replicate_key not in paired.columns:
        raise KeyError(
            f"no replicate column {replicate_key!r}; available: {list(paired.columns)}"
        )

    keys = [col for col in keep if col in paired.columns] + config_keys(
        paired, replicate_key=replicate_key
    )
    aggregations = {
        "real": ("real", "mean"),
        "synth": ("synth", "mean"),
        "n_replicates": ("real", "size"),
    }
    # Worst case across replicates, so a config is screened out by `min_defect_samples`
    # if any of its seeds would have been.
    for column in ("n_samples", "n_defect_samples"):
        if column in paired.columns:
            aggregations[column] = (column, "min")

    aggregated = (
        paired.groupby(keys, dropna=False, sort=True).agg(**aggregations).reset_index()
    )
    aggregated.insert(0, "model_id", range(len(aggregated)))
    return aggregated


def replicate_noise(
    paired: pd.DataFrame,
    by: Sequence[str] = DEFAULT_BY,
    *,
    replicate_key: str = "param_seed",
) -> pd.DataFrame:
    """Within-config spread of each arm across seeds, pooled over configs.

    The noise floor of the whole exercise. If the real scores of two configs differ by
    less than `real_noise_sd`, no proxy could reliably order them and no amount of
    correlation would mean the synthetic set "picked the best model" — it picked a
    coin flip. Pairs with `with_noise_floor`.

    Pooled as the root of the mean within-config variance, over configs with at least two
    replicates; NaN when the sweep has none.

    `by` may be the conditioned grouping of the selection frame it will be joined to: the
    replicate key is dropped from it here, since a population grouped by seed holds one
    replicate per config and could only ever report NaN. The floor is then estimated at
    the next grain up and fans out over the seeds in `with_noise_floor`, which is the
    honest reading anyway — "how far would this score move on another draw" is a question
    about the draws collectively, not about the one in hand.
    """
    if replicate_key not in paired.columns:
        raise KeyError(
            f"no replicate column {replicate_key!r}; available: {list(paired.columns)}"
        )

    by = [key for key in by if key != replicate_key]
    missing = [key for key in by if key not in paired.columns]
    if missing:
        raise KeyError(f"group keys {missing} not in paired frame")

    if paired.empty:
        return pd.DataFrame(
            columns=[
                *by,
                "real_noise_sd",
                "synth_noise_sd",
                "n_configs",
                "n_configs_with_replicates",
                "n_replicates_max",
            ]
        )

    # A conditioned `by` already holds `param_data_num_train_samples`, which `config_keys`
    # also returns; pandas would group on the duplicate twice.
    within = (
        paired.groupby(
            by
            + [
                key
                for key in config_keys(paired, replicate_key=replicate_key)
                if key not in by
            ],
            dropna=False,
            sort=True,
        )
        .agg(
            real_var=("real", "var"),  # ddof=1, so NaN for a lone replicate
            synth_var=("synth", "var"),
            n_replicates=("real", "size"),
        )
        .reset_index()
    )

    pooled = (
        within.groupby(by, dropna=False, sort=True)
        .agg(
            real_var=("real_var", "mean"),  # skipna, so lone replicates abstain
            synth_var=("synth_var", "mean"),
            n_configs=("n_replicates", "size"),
            n_configs_with_replicates=("n_replicates", lambda s: int((s > 1).sum())),
            n_replicates_max=("n_replicates", "max"),
        )
        .reset_index()
    )
    pooled["real_noise_sd"] = np.sqrt(pooled.pop("real_var"))
    pooled["synth_noise_sd"] = np.sqrt(pooled.pop("synth_var"))
    return pooled


def with_noise_floor(
    selection: pd.DataFrame, noise: pd.DataFrame, *, on: Sequence[str] | None = None
) -> pd.DataFrame:
    """Join `evaluate_selection` to `replicate_noise` and read regret in noise units.

    Adds `regret_in_noise_sd` and `decidable`. `decidable` is the gate to apply *before*
    reading any correlation or regret: it is False when the real scores of the whole
    population span less than one seed's worth of noise, i.e. when there was no real
    difference between the models to detect. A skill of 1.0 on an undecidable population
    says nothing about synthetic data.
    """
    on = (
        list(on)
        if on is not None
        else [col for col in selection.columns if col in noise.columns]
    )
    if not on:
        raise KeyError("selection and noise frames share no columns to join on")

    merged = selection.merge(noise, on=on, how="left")
    sd = merged["real_noise_sd"]
    merged["regret_in_noise_sd"] = merged["regret"] / sd.where(sd > 0.0)
    merged["decidable"] = merged["regret_random"] > sd
    return merged


def evaluate_per_training_set(
    paired: pd.DataFrame,
    by: Sequence[str] = DEFAULT_BY,
    *,
    replicate_key: str = "param_seed",
    keys: Sequence[str] = TRAINING_SET_KEYS,
    **kwargs,
) -> pd.DataFrame:
    """`evaluate_selection` over `conditioned_by(paired, by)`, with the noise floor joined.

    The one-call form of the grouping this module argues for: each population is the
    models trained on one training set, so regret is the real score given up by taking the
    synthetic argmax *among models that saw the same images*. Pooled regret answers a
    different question, in which the winner is mostly whichever run got the most images.

    Takes the un-aggregated frame — the seeds are populations here, not replicates, so do
    not run `aggregate_replicates` first. The floor still comes from the seed spread,
    estimated one grain up by `replicate_noise` and fanned back out, so `decidable` reads
    as "the configs differ by more than a redraw of the training images would move them",
    which is the right bar for a ranking that is supposed to generalise.

    `kwargs` go to `evaluate_selection`.
    """
    grouped = conditioned_by(paired, by, keys=keys)
    selection = evaluate_selection(paired, by=grouped, **kwargs)
    if selection.empty or replicate_key not in paired.columns:
        return selection

    noise = replicate_noise(paired, by=grouped, replicate_key=replicate_key)
    return selection if noise.empty else with_noise_floor(selection, noise)


def config_dimensions(
    paired: pd.DataFrame,
    by: Sequence[str] = (),
    *,
    held_out: str | None = None,
    replicate_key: str = "param_seed",
) -> list[str]:
    """The `param_*` columns that actually distinguish one config from another.

    `config_keys` returns every swept parameter; this drops the ones that cannot be part
    of a config *here*. Grouping keys are excluded, because a key is either a population
    boundary or a config dimension and never both, and so are the columns that do not
    vary within a group — `apply_overrides` logs each swept key twice (dotted and
    aliased) and tags the category and version as params too, so `param_category` and the
    alias of a grouping key would otherwise enter the config identity and split every
    config into a population of one.
    """
    keys = [*by, *([] if held_out is None else [held_out])]
    return [
        key
        for key in config_keys(paired, replicate_key=replicate_key)
        if key not in keys
        and (not keys or paired.groupby(keys, dropna=False)[key].nunique().max() > 1)
    ]


def fixed_config_choice(
    paired: pd.DataFrame,
    by: Sequence[str] = ("metric",),
    *,
    held_out: str = "category",
    replicate_key: str = "param_seed",
    configs: Sequence[str] | None = None,
) -> pd.DataFrame:
    """The config a practitioner would carry over, per held-out value, and what it got.

    One row per (`by`, held-out value), with the config columns themselves alongside the
    printable `fixed_config` label, so a caller can look the choice up in a frame of its
    own — `analysis/selected.py` reads three real metrics off it — rather than parsing
    the label back apart. `real_fixed` is what the carried-over config scored on the
    held-out value and `real_best` what the best config there did; `fixed_config_baseline`
    is their difference.

    Configs are aggregated across the other held-out values by **mean rank**, not mean
    score. Raw scores are not comparable across categories — image AUROC sits near 0.99 on
    bottle and near 0.75 on screw — so a mean would let whichever category has the widest
    spread choose the config on its own.

    Reads only the `real` column, so the result does not depend on `set_name` or `query`.
    A fold with no other held-out value to choose on, or whose winner was never run on the
    held-out value itself, gets a null `fixed_config` and a NaN `real_fixed` rather than a
    row silently chosen on the value it is scored on.
    """
    by = list(by)
    missing = [key for key in [*by, held_out] if key not in paired.columns]
    if missing:
        raise KeyError(f"keys {missing} not in paired frame")

    configs = list(
        configs
        if configs is not None
        else config_dimensions(
            paired, by, held_out=held_out, replicate_key=replicate_key
        )
    )
    if not configs:
        raise KeyError("no `param_*` columns to identify a config")

    # One real score per (group, held-out value, config); `real` repeats across queries
    # and sets, and mean collapses those duplicates back to the single scored value.
    scores = (
        paired.groupby([*by, held_out, *configs], dropna=False, sort=True)["real"]
        .mean()
        .reset_index()
    )
    scores["_config"] = list(map(tuple, scores[configs].itertuples(index=False)))

    rows = []
    for keys, group in scores.groupby(by, dropna=False, sort=True):
        for value in group[held_out].unique():
            inside = group[group[held_out] == value]
            outside = group[group[held_out] != value]
            row = dict(zip(by, _as_tuple(keys)), **{held_out: value})
            unchosen = {
                **row,
                **dict.fromkeys(configs),
                "fixed_config": None,
                "real_fixed": _NAN,
                "real_best": float(inside["real"].max()),
            }

            if outside.empty:
                rows.append(unchosen)
                continue

            ranked = (
                outside.assign(
                    _rank=outside.groupby(held_out, dropna=False)["real"].rank(
                        ascending=False, method="average"
                    )
                )
                .groupby("_config", dropna=False)["_rank"]
                .mean()
                .sort_values(
                    kind="stable"
                )  # ties resolved by config order, not by luck
            )
            # Only configs that were also run on the held-out value can be scored there;
            # without one, there is no baseline to report.
            available = set(inside["_config"])
            chosen = next((key for key in ranked.index if key in available), None)
            if chosen is None:
                rows.append(unchosen)
                continue

            rows.append(
                {
                    **unchosen,
                    **dict(zip(configs, chosen)),
                    "fixed_config": ", ".join(
                        f"{col}={val}" for col, val in zip(configs, chosen)
                    ),
                    "real_fixed": float(
                        inside.loc[inside["_config"] == chosen, "real"].max()
                    ),
                }
            )

    return pd.DataFrame(
        rows,
        columns=[*by, held_out, *configs, "fixed_config", "real_fixed", "real_best"],
    )


def fixed_config_baseline(
    paired: pd.DataFrame,
    by: Sequence[str] = ("metric",),
    *,
    held_out: str = "category",
    replicate_key: str = "param_seed",
) -> pd.DataFrame:
    """Regret of using no synthetic data at all: one fixed config, chosen elsewhere.

    The honest competitor. For each held-out category, the config that ranks best *on the
    other categories* is carried over and its regret measured on the held-out one.
    Beating a random pick is a low bar; beating this is the claim worth making, because
    this baseline is what a practitioner does for free.

    Configs are aggregated across the other categories by **mean rank**, not mean score.
    Raw scores are not comparable across categories — image AUROC sits near 0.99 on
    bottle and near 0.75 on screw — so a mean would let whichever category has the widest
    spread choose the config on its own. Deliberately the stronger baseline: a weak
    comparator here would flatter the synthetic arm.

    Reads only the `real` column, so the result does not depend on `set_name` or `query` —
    the real arm is broadcast across queries by `build_paired_metrics`. Group by `metric`
    alone and merge onto a selection frame on `("metric", held_out)`. Pass the same frame
    (same filtering, same `aggregate_replicates` treatment) used for `evaluate_selection`,
    or the two populations will not match.

    Under `conditioned_by`, put the training-set keys in `by` as well — a baseline chosen
    across shot counts is not the comparator for a regret measured within one, and
    `beats_fixed` would be comparing two different questions. Keys in `by` are dropped
    from the config identity, so a key can be a population boundary or a config dimension
    but never both.

    The choice itself is `fixed_config_choice`, which reports the config rather than only
    its cost; this is the difference of the two scores it returns.
    """
    choice = fixed_config_choice(
        paired, by, held_out=held_out, replicate_key=replicate_key
    )
    if choice.empty:
        return pd.DataFrame(columns=[*by, held_out, "regret_fixed", "fixed_config"])

    baseline = choice[[*by, held_out]].copy()
    baseline["regret_fixed"] = choice["real_best"] - choice["real_fixed"]
    baseline["fixed_config"] = choice["fixed_config"]
    return baseline


def _as_tuple(key) -> tuple:
    """`groupby` on a single column yields scalars, on several a tuple."""
    return key if isinstance(key, tuple) else (key,)


def leave_one_out_query(
    paired: pd.DataFrame,
    by: Sequence[str] = ("metric", "set_name"),
    *,
    held_out: str = "category",
    choose_by: str = "skill",
    **kwargs,
) -> pd.DataFrame:
    """Pick the synthetic query on other categories, then score it on the held-out one.

    `evaluate_selection` over the full `by` grid produces a row per query, and reading off
    the best one is proxy selection on the data it is then evaluated on — with a few
    dozen queries, some will look excellent by chance alone. This chooses the query by
    mean `choose_by` across the *other* categories and reports what it then achieved on
    the held-out category. That number generalises; the in-sample maximum does not.

    `kwargs` go to `evaluate_selection`, so `k`, `min_n` and `min_defect_samples` behave
    as they do there.
    """
    by = list(by)
    empty = pd.DataFrame(columns=[*by, held_out, "query", *SELECTION_FIELDS])
    selection = evaluate_selection(paired, by=[*by, "query", held_out], **kwargs)
    if selection.empty:
        return empty

    rows = []
    for keys, group in selection.groupby(by, dropna=False, sort=True):
        keys = _as_tuple(keys)
        for value in group[held_out].unique():
            outside = group[group[held_out] != value]
            inside = group[group[held_out] == value]

            ranked = (
                outside.groupby("query", dropna=False)[choose_by]
                .mean()
                .sort_values(ascending=False)
            )
            available = set(inside["query"])
            query = next((q for q in ranked.index if q in available), None)
            if query is None:
                continue

            picked = inside[inside["query"] == query].iloc[0].to_dict()
            rows.append({**dict(zip(by, keys)), **picked, "query": query})

    # A single held-out value leaves every fold with nothing to choose on, so no row
    # survives; return the schema rather than a column-less frame, which would write a
    # zero-byte CSV that `read_csv` then refuses to parse.
    return pd.DataFrame(rows) if rows else empty
