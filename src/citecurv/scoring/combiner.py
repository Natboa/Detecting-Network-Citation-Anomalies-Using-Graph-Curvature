"""Score combiner: frozen z-statistics, simplex weights, calibration (task S3.2, L13).

total = sum_c w_c * z_c over the four named components, with

    z_c = (x_c - center_c) / scale_c

where (center_c, scale_c) are the mean and population std (ddof=0) of
component c on TRAIN rows, or the median and MAD * 1.4826 when robust=True;
scale_c is floored at std_floor so a near-constant component cannot blow up.
The statistics are fitted once (ZStats.fit), frozen into the run artifact
(to_dict / from_dict) and applied unchanged to val and test rows.

Missing components: a NaN component value contributes 0 to total (the event is
scored from the components it has) and n_components_used records how many
positive-weight components contributed, so a total built from fewer parts is
visible in the breakdown (NFR7). A component with no finite training value has
NaN statistics and therefore always contributes 0. Infinite values are not
treated as missing: they propagate to total and make the ranking metrics raise.

Weights live on the simplex (non-negative, summing to 1) and are per-run
artifacts: calibrate_weights picks them on validation rows by maximising
average precision (or AUROC) over a step lattice of the simplex; the chosen
vector goes to the run's config copy, never to configs/default.yaml.

LearnedCombiner is the small learned alternative (a logistic head over the
same z-features); it is used only for the extra L13 ablation cell.

Leak argument. ZStats read component values of training rows only (rows of
the given split marked "train"; cold-start rows are excluded by default since
they are never scored), so no val/test value can shift the normalisation that
is applied to val/test events. calibrate_weights and LearnedCombiner read
labels, which is supervision on pre-cutoff rows (val by default), never test
rows; the result is a fixed set of numbers applied to every test event, so no
test label or test component distribution reaches a test score. Each component
value itself obeys the ts <= t rule of its own module.
"""

from dataclasses import dataclass

import numpy as np

COMPONENTS = ("s_embed", "s_geom", "s_temp", "s_curvdyn")
MAD_TO_STD = 1.4826
METRICS = ("ap", "auroc")
_WEIGHT_TOL = 1e-6
_METRIC_TIE_TOL = 1e-12


def _nan_to_none(x):
    return None if x is None or (isinstance(x, float) and x != x) else float(x)


def _none_to_nan(x):
    return float("nan") if x is None else float(x)


def _cold_mask(df):
    if "cold_start" in df.columns:
        return df["cold_start"].astype(bool).to_numpy()
    return np.zeros(len(df), dtype=bool)


def _labels_for(df, labels):
    """0/1 label per row of df: by event_id join (DataFrame) or by position."""
    import pandas as pd

    if isinstance(labels, pd.DataFrame):
        if labels["event_id"].duplicated().any():
            raise ValueError("labels has duplicate event_id values")
        lab = pd.Series(labels["label"].to_numpy(), index=labels["event_id"].to_numpy())
        y = lab.reindex(df["event_id"].to_numpy())
        if y.isna().any():
            raise ValueError(f"{int(y.isna().sum())} rows have no label")
        y = y.to_numpy()
    else:
        if isinstance(labels, pd.Series) and not labels.index.equals(df.index):
            raise ValueError("labels Series index differs from the frame index")
        y = np.asarray(labels).ravel()
        if y.size != len(df):
            raise ValueError(f"length mismatch: {len(df)} rows vs {y.size} labels")
    if y.size and not np.isin(y, (0, 1)).all():
        raise ValueError("labels must be 0/1 (or bool)")
    return y.astype(np.int64)


@dataclass(frozen=True)
class ZStats:
    """Frozen per-component normalisation statistics (fit on TRAIN rows only)."""

    components: tuple
    center: dict
    scale: dict
    n_fit: dict
    robust: bool = False
    std_floor: float = 1e-6

    @classmethod
    def fit(
        cls,
        df,
        components=COMPONENTS,
        split=None,
        train_label="train",
        robust=False,
        std_floor=1e-6,
        exclude_cold_start=True,
    ):
        """Fit center/scale per component.

        split: None (every row of df is a training row; the caller has already
        selected them), the name of a column of df, or an array/Series of split
        labels aligned to df by position (a Series must share df's index).
        Only rows equal to train_label are used. NaN values are ignored.
        exclude_cold_start: drop rows with cold_start == True (never scored).
        """
        import pandas as pd

        rows = np.ones(len(df), dtype=bool)
        if split is not None:
            if isinstance(split, str):
                split = df[split]
            elif isinstance(split, pd.Series) and not split.index.equals(df.index):
                raise ValueError("split Series index differs from the frame index")
            s = np.asarray(split)
            if s.size != len(df):
                raise ValueError(f"length mismatch: {len(df)} rows vs {s.size} split labels")
            rows &= s == train_label
            if not rows.any():
                raise ValueError(f"no {train_label!r} rows to fit z-statistics on")
        if exclude_cold_start:
            rows &= ~_cold_mask(df)
        center, scale, n_fit = {}, {}, {}
        for c in components:
            x = df[c].to_numpy(dtype=float)[rows]
            x = x[~np.isnan(x)]
            n_fit[c] = int(x.size)
            if x.size == 0:
                center[c], scale[c] = float("nan"), float("nan")
                continue
            if robust:
                mid = float(np.median(x))
                spread = MAD_TO_STD * float(np.median(np.abs(x - mid)))
            else:
                mid = float(np.mean(x))
                spread = float(np.std(x))
            center[c], scale[c] = mid, max(spread, std_floor)
        return cls(tuple(components), center, scale, n_fit, bool(robust), float(std_floor))

    def z_matrix(self, df):
        """(n, len(components)) array of z-scores; NaN where the input is NaN."""
        x = np.column_stack([df[c].to_numpy(dtype=float) for c in self.components])
        center = np.array([self.center[c] for c in self.components])
        scale = np.array([self.scale[c] for c in self.components])
        return (x - center) / scale

    def to_dict(self) -> dict:
        """JSON-safe dict (NaN statistics stored as null) for the run artifact."""
        return {
            "components": list(self.components),
            "center": {c: _nan_to_none(self.center[c]) for c in self.components},
            "scale": {c: _nan_to_none(self.scale[c]) for c in self.components},
            "n_fit": {c: int(self.n_fit[c]) for c in self.components},
            "robust": self.robust,
            "std_floor": self.std_floor,
        }

    @classmethod
    def from_dict(cls, d):
        comps = tuple(d["components"])
        return cls(
            comps,
            {c: _none_to_nan(d["center"][c]) for c in comps},
            {c: _none_to_nan(d["scale"][c]) for c in comps},
            {c: int(d["n_fit"][c]) for c in comps},
            bool(d["robust"]),
            float(d["std_floor"]),
        )


def weight_vector(weights, components) -> np.ndarray:
    """Validate weights (dict or sequence) and align them to components.

    Dict keys must be components; missing ones get weight 0. Weights must be
    non-negative and sum to 1 (tolerance 1e-6).
    """
    if isinstance(weights, dict):
        unknown = set(weights) - set(components)
        if unknown:
            raise ValueError(f"unknown weight keys: {sorted(unknown)}")
        w = np.array([float(weights.get(c, 0.0)) for c in components])
    else:
        w = np.asarray(weights, dtype=float).ravel()
        if w.size != len(components):
            raise ValueError(f"weights length {w.size} != {len(components)} components")
    if (w < 0).any():
        raise ValueError("weights must be non-negative")
    if abs(w.sum() - 1.0) > _WEIGHT_TOL:
        raise ValueError(f"weights must sum to 1 (got {w.sum():.6g})")
    return w


def _total(z, w):
    present = ~np.isnan(z)
    return np.where(present, z, 0.0) @ w, (present & (w > 0)).sum(axis=1)


def combine(components_df, weights, zstats):
    """Copy of components_df plus z_<component>, total and n_components_used."""
    w = weight_vector(weights, zstats.components)
    z = zstats.z_matrix(components_df)
    total, used = _total(z, w)
    out = components_df.copy()
    for i, c in enumerate(zstats.components):
        out[f"z_{c}"] = z[:, i]
    out["total"] = total
    out["n_components_used"] = used.astype(np.int64)
    return out


def _lattice(n, step):
    m = int(round(1.0 / step))
    if m < 1 or abs(m * step - 1.0) > 1e-9:
        raise ValueError(f"step must divide 1 exactly (got {step})")
    if n < 1:
        raise ValueError("n must be >= 1")

    def compositions(total, parts):
        if parts == 1:
            yield (total,)
            return
        for first in range(total + 1):
            for rest in compositions(total - first, parts - 1):
                yield (first, *rest)

    return np.array(list(compositions(m, n)), dtype=np.int64), m


def simplex_grid(n=4, step=0.1) -> np.ndarray:
    """All weight vectors with entries in {0, step, ..., 1} summing to 1.

    Rows are in ascending lexicographic order; shape (C(m+n-1, n-1), n) with
    m = 1/step (286 rows for the default n=4, step=0.1).
    """
    parts, m = _lattice(n, step)
    return parts / m


def calibrate_weights(val_df, val_labels, zstats, step=0.1, metric="ap", exclude_cold_start=True):
    """Pick simplex weights maximising `metric` on validation rows.

    val_df: component frame of VAL rows (event_id + components [+ cold_start]).
    val_labels: DataFrame (event_id, label) joined on event_id, or 0/1 values
    aligned to val_df by position. Cold-start rows are dropped first.
    metric: "ap" (default, plan S3.2) or "auroc".
    Ties (within 1e-12) go to the vector closest to uniform (Euclidean), then
    to the lexicographically smallest one, so the choice is deterministic.
    Returns (weights: dict component -> float, best_metric: float,
    table: DataFrame with w_<component>, <metric>, chosen per grid point).
    """
    import pandas as pd

    from citecurv.eval.metrics import auroc, average_precision

    if metric not in METRICS:
        raise ValueError(f"metric must be one of {METRICS}, got {metric!r}")
    score_fn = average_precision if metric == "ap" else auroc
    y = _labels_for(val_df, val_labels)
    keep = ~_cold_mask(val_df) if exclude_cold_start else np.ones(len(val_df), dtype=bool)
    y = y[keep]
    if y.sum() == 0:
        raise ValueError("validation rows contain no positive label; cannot calibrate")
    z = zstats.z_matrix(val_df)[keep]

    parts, m = _lattice(len(zstats.components), step)
    grid = parts / m
    values = np.array([score_fn(_total(z, w)[0], y) for w in grid])
    if np.isnan(values).all():
        raise ValueError(f"{metric} is undefined on these validation rows")
    best = np.nanmax(values)
    n = len(zstats.components)
    dist = ((n * parts - m) ** 2).sum(axis=1)  # n^2 * squared distance to uniform, exact
    tied = np.flatnonzero(values >= best - _METRIC_TIE_TOL)
    pick = tied[np.argmin(dist[tied])]  # argmin returns the first, i.e. lexicographically smallest

    table = pd.DataFrame({f"w_{c}": grid[:, i] for i, c in enumerate(zstats.components)})
    table[metric] = values
    table["chosen"] = np.arange(len(grid)) == pick
    weights = {c: float(grid[pick, i]) for i, c in enumerate(zstats.components)}
    return weights, float(values[pick]), table


class LearnedCombiner:
    """Logistic head over the z-features (NaN component -> 0), for the L13 cell.

    predict_score returns the decision function (log-odds): monotone in the
    predicted probability and free of saturation ties. Fit on validation rows,
    the same supervision calibrate_weights uses, so the ablation compares the
    two combiners on equal information; never fit on test rows.
    """

    def __init__(self, zstats, C=1.0, class_weight="balanced", random_state=0, max_iter=1000):
        self.zstats = zstats
        self.C = float(C)
        self.class_weight = class_weight
        self.random_state = int(random_state)
        self.max_iter = int(max_iter)
        self.coef = None
        self.intercept = None

    def features(self, df) -> np.ndarray:
        z = self.zstats.z_matrix(df)
        return np.where(np.isnan(z), 0.0, z)

    def fit(self, df, labels, exclude_cold_start=True):
        from sklearn.linear_model import LogisticRegression

        y = _labels_for(df, labels)
        keep = ~_cold_mask(df) if exclude_cold_start else np.ones(len(df), dtype=bool)
        if len(np.unique(y[keep])) < 2:
            raise ValueError("LearnedCombiner needs both classes in the fit rows")
        model = LogisticRegression(
            C=self.C,
            class_weight=self.class_weight,
            random_state=self.random_state,
            max_iter=self.max_iter,
        )
        model.fit(self.features(df)[keep], y[keep])
        self.coef = {c: float(v) for c, v in zip(self.zstats.components, model.coef_[0])}
        self.intercept = float(model.intercept_[0])
        return self

    def predict_score(self, df) -> np.ndarray:
        if self.coef is None:
            raise RuntimeError("LearnedCombiner is not fitted")
        w = np.array([self.coef[c] for c in self.zstats.components])
        return self.features(df) @ w + self.intercept

    def to_dict(self) -> dict:
        return {
            "zstats": self.zstats.to_dict(),
            "C": self.C,
            "class_weight": self.class_weight,
            "random_state": self.random_state,
            "max_iter": self.max_iter,
            "coef": self.coef,
            "intercept": self.intercept,
        }

    @classmethod
    def from_dict(cls, d):
        obj = cls(
            ZStats.from_dict(d["zstats"]),
            C=d["C"],
            class_weight=d["class_weight"],
            random_state=d["random_state"],
            max_iter=d["max_iter"],
        )
        obj.coef = None if d["coef"] is None else {c: float(v) for c, v in d["coef"].items()}
        obj.intercept = None if d["intercept"] is None else float(d["intercept"])
        return obj
