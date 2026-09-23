import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from pandas.errors import UndefinedVariableError
from torch import Tensor
from sklearn.metrics import roc_auc_score, average_precision_score
from synevad.data import Batch
from synevad.metrics.ordinal import empty_severity, severity_concordance
from synevad.metrics.pixel import (
    DEFAULT_FPR_LIMIT,
    DEFAULT_N_BINS,
    BinStrategy,
    PixelHistograms,
    build_pixel_histograms,
    pixel_metrics,
)


# Severity levels thinner than this are dropped before the ordinal statistic — a grade
# represented by a single image adds pair count but no information. 3 is the smallest
# level that can carry a rank correlation of its own.
DEFAULT_MIN_LEVEL_N: int = 3


@dataclass
class Query:
    """A named `pandas.DataFrame.query` expression over the columns of `Predictions`.

    `expr` is None to select everything; otherwise any pandas expression, e.g.
    `severity in ['moderate', 'severe']`. Defect-free samples are added back unless
    `keep_good` is False, so an expression only has to describe the defects to keep.
    """

    name: str
    expr: str | None
    keep_good: bool = True


@dataclass
class Predictions:
    """Per-sample table mirroring the logged `scores.csv`, plus the pixel-level maps.

    `frame` is positionally aligned with `mask` and `anomaly_map`, so filtering rows
    of the frame selects the matching maps.
    """

    frame: pd.DataFrame
    mask: np.ndarray
    anomaly_map: np.ndarray

    def select(self, expr: str | None, keep_good: bool = True) -> np.ndarray:
        """Positional row indices the query keeps, aligned with `PixelHistograms` rows.

        Indices rather than a sliced `Predictions`: the pixel metrics read per-image
        histograms, so materialising the query's share of the maps — 130 MB per query at
        256px, and nine queries per synthetic set — bought nothing.
        """
        return select_rows(self.frame, expr, keep_good)


def select_rows(
    frame: pd.DataFrame, expr: str | None, keep_good: bool = True
) -> np.ndarray:
    """The query semantics of `Predictions.select`, over a bare `scores.csv` frame.

    Split out so a caller holding only the logged per-image table —
    `synevad.analysis.severity` recomputing a metric from artifacts, with no pixel maps in
    reach — selects exactly the rows the sweep did, rather than a re-implementation that
    drifts from it.
    """
    if expr is None:
        return np.arange(len(frame), dtype=np.intp)

    if keep_good:
        expr = f"({expr}) or label == 0"
    # The python engine keeps `.str` accessors and other method calls usable.
    return frame.query(expr, engine="python").index.to_numpy(dtype=np.intp)


@dataclass
class ADMetrics:
    image_auroc: float
    pixel_auroc: float
    image_aupr: float
    pixel_aupr: float
    aupro: float

    n_samples: float
    n_good_samples: int
    n_defect_samples: int
    positive_pixel_rate: float

    # Ordinal severity statistics (`synevad.metrics.ordinal`). NaN on the real arm and on
    # any query spanning a single grade — real defects carry no severity, and one grade
    # has nothing to order. Defaulted so every existing construction site still type-checks.
    severity_cindex: float = float("nan")
    severity_cindex_defects: float = float("nan")
    severity_tau_b: float = float("nan")
    severity_spearman: float = float("nan")
    severity_adjacent_auc: float = float("nan")
    n_severity_levels: float = 0.0
    n_severity_pairs: float = 0.0

    # Spare columns, spliced into `as_dict` as columns of their own. A dict rather than
    # named fields so a caller can add a metric without editing this dataclass.
    extra: dict[str, float] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"""Metrics:{self.image_auroc=:.3f} {self.pixel_auroc=:.3f} """
            f"""{self.image_aupr=:.3f} {self.pixel_aupr=:.3f}"""
        )

    def as_dict(self) -> dict[str, float]:
        values = asdict(self)
        # Popped before the splice, not inside it: `{**values, **values.pop("extra")}`
        # expands `values` first and leaves an `extra` column holding the mapping itself.
        extra = values.pop("extra")
        return {**values, **extra}


@dataclass
class ADMetricsCollection:
    names: list[str] = field(default_factory=list)
    metrics: list[ADMetrics] = field(default_factory=list)

    def add(self, metrics: ADMetrics, name: str):
        self.names.append(name)
        self.metrics.append(metrics)

    def as_frame(self) -> pd.DataFrame:
        frame = pd.DataFrame([m.as_dict() for m in self.metrics])
        frame.insert(0, "query", self.names)
        return frame


def _to_numpy(values: list[Tensor]) -> np.ndarray:
    return torch.cat(values).detach().cpu().numpy()


def _sample_id(index: int, image_path: str) -> str:
    # The parent dir disambiguates identical stems across defect classes.
    path = Path(image_path)
    return f"{index:05d}_{path.parent.name}_{path.stem}"


def build_predictions(batches: list[Batch]) -> Predictions:
    rows = []
    for batch in batches:
        if batch.score is None or batch.anomaly_map is None:
            raise ValueError("batch has no predictions, call add_predictions() first")

        # Key order defines the column order of the logged scores.csv.
        for idx, image_path in enumerate(batch.image_paths):
            rows.append(
                {
                    "sample_id": _sample_id(len(rows), image_path),
                    "image_path": image_path,
                    "class_name": batch.class_name[idx],
                    "label": int(batch.label[idx]),
                    # Empty for defect-free samples, which have no mask on disk.
                    "mask_path": batch.mask_paths[idx] or "",
                    "score": float(batch.score[idx]),
                    "scorer": batch.scorer_result[idx],
                    "severity": batch.severity[idx],
                    # The mask estimator's verdict from the generation manifest, so the
                    # gate can tell a defect with a usable mask from one whose mask was
                    # rejected. "real" on the real arm, "none" for defect-free frames,
                    # "unknown" for manifests written before the column existed.
                    "mask_status": batch.mask_status[idx],
                    # EditReward margin from the generation manifest. Empty in the CSV
                    # for real and defect-free images, and for rows generated before the
                    # judge ran — an unscored row is not a zero-margin one.
                    "reward_score": batch.reward_score[idx],
                    # Share of the frame the mask covers, from the generation
                    # manifest, which the synthetic arms gate on
                    # (`defect_area_frac != 0`). Empty for real and defect-free
                    # images. Zero is a real measurement (an empty mask), not a
                    # missing one — see `synevad.metrics.area`.
                    "defect_area_frac": batch.defect_area_frac[idx],
                }
            )

    frame = pd.DataFrame(rows)
    return Predictions(
        frame=frame,
        mask=_to_numpy([batch.mask for batch in batches]),
        anomaly_map=_to_numpy([batch.anomaly_map for batch in batches]),  # type: ignore
    )


def make_queries(config: DictConfig) -> list[Query]:
    """Builds queries from config entries that are null, an expression string, or a
    mapping of `{expr: ..., keep_good: ...}`."""
    queries = []
    for key, entry in config.items():
        if isinstance(entry, DictConfig):
            entry = OmegaConf.to_container(entry, resolve=True)
            assert isinstance(entry, dict)
            query = Query(
                str(key), str(entry["expr"]), bool(entry.get("keep_good", True))
            )
        elif entry is None or isinstance(entry, str):
            query = Query(str(key), entry)
        else:
            raise TypeError(f"query {key!r}: expected null, a string or a mapping")
        queries.append(query)
    return queries


def gate_queries(
    queries: list[Query], gate: str | None, frame: pd.DataFrame | None = None
) -> list[Query]:
    """`queries` with `gate` ANDed into every expression.

    A gate is the restriction that holds whatever else a query says — on the synthetic
    arm, a composite whose mask has no pixels (`defect_area_frac != 0`). Folding it into
    `Query.expr` rather than filtering the frame keeps `keep_good` meaning what it always
    did: `select_rows` still adds the defect-free rows back, and they are the negatives
    every image metric needs. It also means every caller that re-derives a selection from
    the config — the certainty and severity recomputes — gets the same rows for free.

    With `frame`, a gate naming a column the frame does not carry is dropped, with a note
    on stderr, instead of raising. That is for the analysis scans, which read artifacts
    written before the gate's columns existed: an ungated recompute of an old run is worth
    more than a scan that stops on it. The eval path passes no frame — it just built those
    columns, so a gate that will not resolve there is a typo and should fail loudly.
    """
    if gate is None:
        return queries

    if frame is not None:
        try:
            frame.head(0).query(gate, engine="python")
        except UndefinedVariableError as error:
            print(
                f"gate {gate!r} names a column this frame does not carry ({error}); "
                "scoring it ungated",
                file=sys.stderr,
            )
            return queries

    return [
        replace(query, expr=gate if query.expr is None else f"({query.expr}) and ({gate})")
        for query in queries
    ]


def resolvable_queries(queries: list[Query], frame: pd.DataFrame | None) -> list[Query]:
    """`queries` minus the ones naming a column `frame` does not carry, with a note.

    The same tolerance `gate_queries` gives the gate, for the same reason and the same
    callers: the analysis scans read artifacts written before a column existed, and one
    unresolvable query should cost that query rather than the scan. A gate can degrade to
    "ungated" and still describe a population; a query cannot degrade at all — dropping
    `severity == 'minimal' and mask_status == 'ok'` to its first half would silently
    report a *different* selection under the same name — so the query goes, and the
    metric for it is simply absent.

    `frame` is None on the eval path, which has just built every column: an expression
    that will not resolve there is a typo and should fail loudly, so nothing is checked.
    """
    if frame is None:
        return queries

    head = frame.head(0)
    kept = []
    for query in queries:
        if query.expr is not None:
            try:
                head.query(query.expr, engine="python")
            except UndefinedVariableError as error:
                print(
                    f"query {query.name!r} ({query.expr!r}) names a column this frame "
                    f"does not carry ({error}); skipping it",
                    file=sys.stderr,
                )
                continue
        kept.append(query)
    return kept


def queries_for(
    config: DictConfig, set_name: str, frame: pd.DataFrame | None = None
) -> list[Query]:
    """The eval set's queries, each restricted by the set's gate.

    `config.queries[set_name]` is the query block; the optional `config.gate` block is
    keyed by the same set names and holds one expression per set (null, or absent, for a
    set that is scored on everything it was given). The real arm belongs in that second
    group: its defects were photographed, so they carry no estimated mask, and a gate
    mentioning one would drop every one of them.

    `frame` is passed through to both tolerances — `resolvable_queries` for the queries
    and `gate_queries` for the gate. Applied in that order, so a query dropped for its own
    missing column is never first widened by a gate that is also being dropped.
    """
    queries = resolvable_queries(make_queries(config.queries[set_name]), frame)
    gate = config.gate.get(set_name) if "gate" in config else None
    return gate_queries(queries, None if gate is None else str(gate), frame)


def compute_metrics_from_queries(
    data: Predictions,
    queries: list[Query],
    *,
    n_bins: int = DEFAULT_N_BINS,
    strategy: BinStrategy = "quantile",
    fpr_limit: float = DEFAULT_FPR_LIMIT,
    min_level_n: int = DEFAULT_MIN_LEVEL_N,
) -> pd.DataFrame:
    """One row per query, over per-image histograms built once for the whole set.

    The bin grid comes from every pixel of `data` and is never rebuilt per query, which
    is what lets a query be a sum of rows: two images binned against different edges
    could not be added, because bin k would span a different score range in each.
    """
    histograms = build_pixel_histograms(
        data.mask, data.anomaly_map, n_bins=n_bins, strategy=strategy
    )

    all_metrics = ADMetricsCollection()
    for query in queries:
        rows = data.select(query.expr, query.keep_good)
        all_metrics.add(
            _metrics(
                data.frame,
                histograms,
                rows,
                fpr_limit=fpr_limit,
                min_level_n=min_level_n,
            ),
            query.name,
        )

    return all_metrics.as_frame()


def compute_metrics(
    data: Predictions,
    *,
    n_bins: int = DEFAULT_N_BINS,
    strategy: BinStrategy = "quantile",
    fpr_limit: float = DEFAULT_FPR_LIMIT,
    min_level_n: int = DEFAULT_MIN_LEVEL_N,
) -> ADMetrics:
    """Whole-set metrics, building histograms of their own. For callers with one set."""
    histograms = build_pixel_histograms(
        data.mask, data.anomaly_map, n_bins=n_bins, strategy=strategy
    )
    rows = np.arange(len(data.frame), dtype=np.intp)
    return _metrics(
        data.frame,
        histograms,
        rows,
        fpr_limit=fpr_limit,
        min_level_n=min_level_n,
    )


def _metrics(
    frame: pd.DataFrame,
    histograms: PixelHistograms,
    rows: np.ndarray,
    *,
    fpr_limit: float,
    min_level_n: int = DEFAULT_MIN_LEVEL_N,
) -> ADMetrics:
    selected = frame.iloc[rows]
    label = selected["label"].to_numpy()
    score = selected["score"].to_numpy()
    # At most a few hundred scalars, so the exact sklearn implementations stay.
    pixel = pixel_metrics(histograms, rows, fpr_limit=fpr_limit)

    # Costs one `rankdata` per severity-level pair — at most ten pairs, against the
    # per-image histogram pass above. Absent `severity` means a caller built a frame by
    # hand; the fields stay NaN rather than making the column mandatory.
    severity = (
        severity_concordance(
            label, score, selected["severity"].tolist(), min_level_n=min_level_n
        )
        if "severity" in selected.columns
        else empty_severity()
    )

    return ADMetrics(
        image_auroc=float(roc_auc_score(label, score)),
        pixel_auroc=pixel.auroc,
        image_aupr=float(average_precision_score(label, score)),
        pixel_aupr=pixel.aupr,
        aupro=pixel.aupro,
        n_samples=len(label),
        n_good_samples=(label == 0).sum(),
        n_defect_samples=(label != 0).sum(),
        positive_pixel_rate=pixel.positive_pixel_rate,
        **severity,
    )
