"""Pixel metrics as re-aggregations of per-image score histograms.

Every pixel score this repo reports is a functional of two counting measures over a
shared threshold grid: how many defect-free pixels sit at or above t, and how many defect
pixels do. Computing those once per image turns the nine queries of `mvtec.yaml` from
nine full passes over ~39M pixels into nine sums over an `(n_images, n_bins)` array.

AUPRO looks per-region but is not. Writing R for the number of regions and n_r for the
size of the region a pixel belongs to,

    PRO(t) = (1/R) * sum_r |{p in r : s_p >= t}| / n_r
           = sum over defect pixels p of  1 / (R * n_r(p))  *  1[s_p >= t]

so it is a *weighted* count over defect pixels, and one weight histogram per image plus a
region count carries it. Regions never span images, so both are exactly additive over the
rows a query selects — which is the whole reason a query costs a sum rather than a pass.

The false positive rate depends only on defect-free pixels, so it is shared by every
region rather than recomputed per region. That is what removes the O(regions x pixels)
term that made the predecessor here cost 82s per config.

Scores are binned raw. The min-max rescale the torchmetrics-based predecessor needed —
to stop float32 sigmoid saturating on PatchCore's unbounded distances — has no analogue
in a histogram, which never sees a sigmoid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, NamedTuple

import numpy as np
from scipy import ndimage

# Measured against sklearn on a real 22M-pixel bottle run: binned AUROC lands within
# 3e-8 and average precision within 5e-5, roughly 20x tighter than the 1e-3 that matters
# here, while the histograms stay under 200 MB for a 673-image eval set.
DEFAULT_N_BINS: int = 16384

# Bergmann et al.'s convention, and the limit every published MVTec AUPRO is quoted at.
# Deliberately not a config key: sweeping it would make one `aupro` column mean two
# things across a set of runs that otherwise look comparable.
DEFAULT_FPR_LIMIT: float = 0.3

# 4-connected, matching MVTec AD's official `pro_curve_util.py`, so `aupro` is comparable
# with published numbers. The `kornia.contrib.connected_components` predecessor was a 3x3
# max-pool (8-connected) run for a fixed 1000 iterations, which at 384px had not
# converged — so it matched neither convention and over-segmented.
REGION_STRUCTURE: np.ndarray | None = None

# Quantile edges are estimated from a strided sample; exact quantiles of ~39M values cost
# more than every metric downstream of them.
_QUANTILE_SAMPLE: int = 2_000_000

BinStrategy = Literal["quantile", "uniform"]


@dataclass(frozen=True)
class ScoreBins:
    """A monotone quantisation of anomaly scores onto one grid shared by every image.

    `edges` holds the n_bins - 1 interior edges, so `searchsorted(edges, s, "right")`
    lands in [0, n_bins). Quantile edges are the default: uniform-width bins put their
    resolution where the score *range* is, so one hot pixel costs an order of magnitude
    of average-precision accuracy, while equal-mass bins put it where the pixels are and
    bound the error by the bin count alone.

    The grid is built from every pixel of an eval set and never rebuilt per query. Two
    images binned against different edges could not be summed: bin k would mean a
    different score range in each.
    """

    edges: np.ndarray

    @property
    def n_bins(self) -> int:
        return len(self.edges) + 1

    @classmethod
    def from_scores(
        cls,
        scores: np.ndarray,
        *,
        n_bins: int = DEFAULT_N_BINS,
        strategy: BinStrategy = "quantile",
    ) -> "ScoreBins":
        flat = np.asarray(scores, dtype=np.float64).ravel()
        low, high = float(flat.min()), float(flat.max())
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            # A constant map is a real case (an empty memory bank, a saturated backbone);
            # one interior edge keeps every pixel in a single well-defined bin.
            return cls(edges=np.array([low], dtype=np.float64))

        if strategy == "uniform":
            edges = np.linspace(low, high, n_bins + 1)[1:-1]
        else:
            sample = flat if flat.size <= _QUANTILE_SAMPLE else flat[:: flat.size // _QUANTILE_SAMPLE]
            edges = np.quantile(sample, np.linspace(0.0, 1.0, n_bins + 1)[1:-1])
            # Ties collapse quantiles onto each other; duplicate edges would create empty
            # bins that contribute nothing but still cost memory.
            edges = np.unique(edges)
        return cls(edges=np.ascontiguousarray(edges, dtype=np.float64))

    def index(self, scores: np.ndarray) -> np.ndarray:
        """Bin index per element, flattened, as int32."""
        return np.searchsorted(
            self.edges, np.asarray(scores, dtype=np.float64).ravel(), side="right"
        ).astype(np.int32)


@dataclass(frozen=True)
class PixelHistograms:
    """Per-image score histograms; every pixel metric is a sum of rows of these.

    Rows are positionally aligned with `Predictions.frame`, so a query is a set of row
    indices followed by a sum along axis 0.
    """

    bins: ScoreBins
    negative: np.ndarray  # (n_images, n_bins) int64 — ground-truth defect-free pixels
    positive: np.ndarray  # (n_images, n_bins) int64 — ground-truth defect pixels
    region_weight: np.ndarray  # (n_images, n_bins) float64 — sum of 1 / region_size
    n_regions: np.ndarray  # (n_images,) int64

    def __len__(self) -> int:
        return len(self.n_regions)


class PixelMetrics(NamedTuple):
    auroc: float
    aupr: float
    aupro: float
    positive_pixel_rate: float


def _as_image_stack(array: np.ndarray) -> np.ndarray:
    """Drop the channel axis `collate_fn` leaves on masks, accepting maps without one."""
    array = np.asarray(array)
    if array.ndim == 4:
        if array.shape[1] != 1:
            raise ValueError(f"expected a single channel, got shape {array.shape}")
        return array[:, 0]
    if array.ndim != 3:
        raise ValueError(f"expected (N, H, W) or (N, 1, H, W), got shape {array.shape}")
    return array


def build_pixel_histograms(
    masks: np.ndarray,
    anomaly_maps: np.ndarray,
    *,
    n_bins: int = DEFAULT_N_BINS,
    strategy: BinStrategy = "quantile",
    structure: np.ndarray | None = REGION_STRUCTURE,
) -> PixelHistograms:
    """One pass over every image, building the statistics all queries re-aggregate.

    `masks` may be (N, H, W) or (N, 1, H, W); `collate_fn` produces the latter because
    `ImageDataset` unsqueezes a channel onto each mask.
    """
    masks = _as_image_stack(masks)
    anomaly_maps = _as_image_stack(anomaly_maps)
    if masks.shape != anomaly_maps.shape:
        raise ValueError(
            f"masks {masks.shape} and anomaly maps {anomaly_maps.shape} must align"
        )

    bins = ScoreBins.from_scores(anomaly_maps, n_bins=n_bins, strategy=strategy)
    width = bins.n_bins
    n = len(masks)

    negative = np.zeros((n, width), dtype=np.int64)
    positive = np.zeros((n, width), dtype=np.int64)
    region_weight = np.zeros((n, width), dtype=np.float64)
    n_regions = np.zeros(n, dtype=np.int64)

    for i, (mask, amap) in enumerate(zip(masks, anomaly_maps)):
        index = bins.index(amap)
        defect = np.asarray(mask).ravel() > 0

        positive[i] = np.bincount(index[defect], minlength=width)
        negative[i] = np.bincount(index[~defect], minlength=width)

        labels, count = ndimage.label(np.asarray(mask) > 0, structure=structure)
        n_regions[i] = count
        if count:
            labels = labels.ravel()
            sizes = np.bincount(labels)
            hit = labels > 0
            region_weight[i] = np.bincount(
                index[hit], weights=1.0 / sizes[labels[hit]], minlength=width
            )

    return PixelHistograms(
        bins=bins,
        negative=negative,
        positive=positive,
        region_weight=region_weight,
        n_regions=n_regions,
    )


def _at_or_above(counts: np.ndarray) -> np.ndarray:
    """Counts at or above each of the n_bins + 1 thresholds, highest score first.

    Leads with a zero so the curves start at the origin, which is what makes the
    trapezoid below an area rather than an area minus its first step.
    """
    return np.concatenate(([0.0], np.cumsum(counts[::-1])))


def _truncate_at(x: np.ndarray, y: np.ndarray, limit: float) -> tuple[np.ndarray, np.ndarray]:
    """`x`/`y` up to `limit`, interpolating one point so the area ends exactly there."""
    keep = x <= limit
    x_cut, y_cut = x[keep], y[keep]
    nxt = int(keep.sum())
    if nxt < len(x) and x_cut[-1] < limit:
        span = x[nxt] - x_cut[-1]
        frac = (limit - x_cut[-1]) / span if span > 0 else 0.0
        x_cut = np.append(x_cut, limit)
        y_cut = np.append(y_cut, y_cut[-1] + frac * (y[nxt] - y_cut[-1]))
    return x_cut, y_cut


def pixel_metrics(
    histograms: PixelHistograms,
    rows: np.ndarray | None = None,
    *,
    fpr_limit: float = DEFAULT_FPR_LIMIT,
) -> PixelMetrics:
    """The four pixel scores for the images at `rows` (all of them when None).

    Degenerate selections return NaN rather than raising: a query whose rows hold no
    defect pixels has no AUROC, and one with no regions has no AUPRO. The predecessor
    raised out of sklearn on the first.
    """
    if rows is None:
        positive, negative, weight = (
            histograms.positive.sum(axis=0),
            histograms.negative.sum(axis=0),
            histograms.region_weight.sum(axis=0),
        )
        n_regions = int(histograms.n_regions.sum())
    else:
        rows = np.asarray(rows, dtype=np.intp)
        positive, negative, weight = (
            histograms.positive[rows].sum(axis=0),
            histograms.negative[rows].sum(axis=0),
            histograms.region_weight[rows].sum(axis=0),
        )
        n_regions = int(histograms.n_regions[rows].sum())

    n_positive, n_negative = positive.sum(), negative.sum()
    total = n_positive + n_negative
    positive_rate = float(n_positive / total) if total else float("nan")

    if n_positive == 0 or n_negative == 0:
        return PixelMetrics(float("nan"), float("nan"), float("nan"), positive_rate)

    tp, fp = _at_or_above(positive), _at_or_above(negative)
    tpr, fpr = tp / n_positive, fp / n_negative

    # Trapezoidal ROC AUC: within a bin the two classes are interleaved arbitrarily, and
    # the trapezoid is the expected area over those orderings — the same convention
    # sklearn's `roc_auc_score` applies to tied scores.
    auroc = float(np.trapezoid(tpr, fpr))

    # Step-sum average precision, matching `average_precision_score`: sum of the
    # precision at each threshold weighted by the recall it gained.
    predicted = tp + fp
    precision = np.divide(tp, predicted, out=np.ones_like(tp), where=predicted > 0)
    aupr = float(np.sum(np.diff(tpr) * precision[1:]))

    if n_regions == 0:
        aupro = float("nan")
    else:
        pro = _at_or_above(weight) / n_regions
        fpr_cut, pro_cut = _truncate_at(fpr, pro, fpr_limit)
        aupro = float(np.trapezoid(pro_cut, fpr_cut) / fpr_limit)

    return PixelMetrics(auroc, aupr, aupro, positive_rate)
