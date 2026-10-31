"""Time-respecting train/val/test split (task S4.1, NFR3).

The split is by event time only. Given a cutoff c (days since 1970-01-01):

- test  = events with t > c;
- val   = pre-cutoff events whose t lies in the last val_frac of the pre-cutoff
          time span, i.e. val_start < t <= c with
          val_start = c - val_frac * (c - t_min);
- train = the remaining pre-cutoff events (t <= val_start).

All events that share a timestamp (all references of one citing paper) land in
the same split, and every event gets exactly one label.

choose_cutoff picks c as a deterministic function of the frozen events and
labels: the latest event time such that the test period holds at least
min_test_pos positives and at least min_test_share of all events. Only event
times are candidates (any value in [t_i, t_{i+1}) gives the same partition, so
returning the event time t_i makes the value canonical); the largest event time
is excluded because it leaves the test set empty, and the smallest one is the
earliest candidate, so train is never empty.

Leak argument. The cutoff search reads labels, and that is allowed: labels are
outcome data used to size the evaluation set, never inputs to a model. A
retraction date may post-date the cutoff (a paper published and cited before
the cutoff can be retracted years later); this is fine for the same reason,
because labels are joined to finished scores only at evaluation time. Features
never read labels or retraction dates; they read graph state with ts <= t
(the scored event excluded), and the split only decides which events are used
for fitting (train), calibration and early stopping (val), and reporting
(test). Choosing c from label counts changes which events are evaluated, not
what any feature of any event can see.
"""

import datetime as _dt
from pathlib import Path

import numpy as np

EPOCH = _dt.date(1970, 1, 1)
SPLITS = ("train", "val", "test")


def _label_vector(event_ids, event_labels):
    """0/1 label per event id; events missing from event_labels count as 0.

    event_labels: DataFrame with event_id, label (it may list positives only),
    or a Series indexed by event_id.
    """
    import pandas as pd

    if isinstance(event_labels, pd.Series):
        lab = event_labels
    else:
        if event_labels["event_id"].duplicated().any():
            raise ValueError("event_labels has duplicate event_id values")
        lab = pd.Series(event_labels["label"].to_numpy(), index=event_labels["event_id"].to_numpy())
    y = lab.reindex(np.asarray(event_ids)).fillna(0).to_numpy()
    if y.size and not np.isin(y, (0, 1)).all():
        raise ValueError("labels must be 0/1 (or bool)")
    return y.astype(np.int64)


def _as_papers(value):
    if isinstance(value, str):
        return [value]
    if value is None or (isinstance(value, float) and value != value):  # None or NaN
        return []
    return [str(p) for p in value]


def _last_positive_time_per_paper(event_ids, t, y, paper_of_positive):
    """For each positive paper, the latest t among its positive events."""
    if paper_of_positive is None:
        raise ValueError('positive_unit="paper" needs paper_of_positive (event_id -> paper id(s))')
    mapping = paper_of_positive.to_dict() if hasattr(paper_of_positive, "to_dict") else paper_of_positive
    last = {}
    missing = 0
    for e, ts in zip(event_ids[y == 1], t[y == 1]):
        papers = _as_papers(mapping.get(e))
        if not papers:
            missing += 1
            continue
        for p in papers:
            last[p] = max(last.get(p, ts), ts)
    if missing:
        raise ValueError(f"{missing} positive events have no paper in paper_of_positive")
    return np.sort(np.fromiter(last.values(), dtype=np.int64, count=len(last)))


def choose_cutoff(
    events,
    event_labels,
    min_test_pos=100,
    min_test_share=0.15,
    positive_unit="event",
    paper_of_positive=None,
) -> int:
    """Latest event time c such that the test set {t > c} satisfies both bounds.

    events: DataFrame with event_id, t. event_labels: see _label_vector.
    min_test_pos: minimum number of positives in the test period, counted as
    positive events (positive_unit="event") or as distinct positive papers
    (positive_unit="paper"; paper_of_positive maps each positive event_id to
    the retracted paper id or ids that made it positive, dict or Series).
    min_test_share: minimum fraction of all events in the test period
    (inclusive bound).
    Raises ValueError when no candidate qualifies.
    """
    if positive_unit not in ("event", "paper"):
        raise ValueError(f'positive_unit must be "event" or "paper", got {positive_unit!r}')
    if not 0.0 <= min_test_share <= 1.0:
        raise ValueError("min_test_share must lie in [0, 1]")
    event_ids = events["event_id"].to_numpy()
    t = events["t"].to_numpy(dtype=np.int64)
    y = _label_vector(event_ids, event_labels)
    n = t.size
    candidates = np.unique(t)[:-1]

    n_test = n - np.searchsorted(np.sort(t), candidates, side="right")
    if positive_unit == "event":
        pos_t = np.sort(t[y == 1])
    else:
        pos_t = _last_positive_time_per_paper(event_ids, t, y, paper_of_positive)
    n_pos_test = pos_t.size - np.searchsorted(pos_t, candidates, side="right")

    ok = (n_pos_test >= min_test_pos) & (n_test / max(n, 1) >= min_test_share)
    if not ok.any():
        best_pos = int(n_pos_test.max()) if candidates.size else 0
        best_share = float(n_test.max() / n) if candidates.size else 0.0
        raise ValueError(
            f"no cutoff satisfies min_test_pos={min_test_pos} ({positive_unit}s) and "
            f"min_test_share={min_test_share}: the earliest cutoff gives at most "
            f"{best_pos} positives and a {best_share:.3f} test share"
        )
    return int(candidates[ok].max())


def split_events(events, cutoff, val_frac=0.10):
    """'train' / 'val' / 'test' per event, as a Series aligned to events.index.

    See the module docstring for the rule. Raises ValueError when no event
    lies at or before the cutoff (the pre-cutoff span would be undefined).
    """
    import pandas as pd

    if not 0.0 <= val_frac < 1.0:
        raise ValueError("val_frac must lie in [0, 1)")
    t = events["t"].to_numpy(dtype=np.int64)
    pre = t <= cutoff
    if not pre.any():
        raise ValueError(f"no events at or before cutoff {cutoff}")
    t_min = int(t[pre].min())
    val_start = cutoff - val_frac * (cutoff - t_min)
    labels = np.where(~pre, "test", np.where(t > val_start, "val", "train"))
    return pd.Series(labels, index=events.index, name="split")


def split_counts(events, split, event_labels=None) -> dict:
    """Event counts per split, plus positive-event counts when labels are given."""
    s = np.asarray(split)
    out = {"n_events": int(s.size)}
    for name in SPLITS:
        out[f"n_{name}"] = int((s == name).sum())
    if event_labels is not None:
        y = _label_vector(events["event_id"].to_numpy(), event_labels)
        for name in SPLITS:
            out[f"n_pos_{name}"] = int(y[s == name].sum())
    return out


def _plain(value):
    """numpy scalars/containers -> plain Python for yaml.safe_dump."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_split_yaml(path, cutoff, counts, params=None):
    """Freeze the split: cutoff_t (days since epoch), cutoff_date, counts, params.

    params: optional dict of the parameters used (min_test_pos, val_frac, ...),
    recorded for traceability. Returns the path written.
    """
    import yaml

    cutoff = int(cutoff)
    doc = {
        "cutoff_t": cutoff,
        "cutoff_date": (EPOCH + _dt.timedelta(days=cutoff)).isoformat(),
        "test_rule": "t > cutoff_t",
        "counts": _plain(dict(counts)),
    }
    if params is not None:
        doc["params"] = _plain(dict(params))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(doc, f, sort_keys=False)
    return path
