"""S4.1: time split.

Spec under test (IMPLEMENTATION_PLAN.md S4.1, CONTRACTS.md "Split"):
- choose_cutoff returns the LATEST cutoff (an event time; test = events with
  t > cutoff) whose test period holds >= min_test_pos positives AND
  >= min_test_share of all events; positives are counted per event or, with
  positive_unit="paper", as distinct positive papers via a mapping
  event_id -> paper id(s). No qualifying cutoff -> ValueError.
- split_events labels every event train/val/test by time only; val = pre-cutoff
  events in the last val_frac of the pre-cutoff span (t_min .. cutoff).
- No event in two splits; same input -> same output, whatever the row order.
"""

import datetime as dt

import numpy as np
import pandas as pd
import pytest
import yaml

from citecurv.eval.split import (
    choose_cutoff,
    split_counts,
    split_events,
    write_split_yaml,
)

# event_id, t, label. Candidate cutoffs and their test sets (t > cutoff):
#   c=1: 8 events (80%), 3 pos | c=2: 7 (70%), 3 pos | c=3: 5 (50%), 2 pos
#   c=4: 4 (40%), 2 pos        | c=5: 2 (20%), 1 pos | c=6: 1 (10%), 1 pos
HAND = [
    (0, 1, 0), (1, 1, 1), (2, 2, 0), (3, 3, 1), (4, 3, 0),
    (5, 4, 0), (6, 5, 1), (7, 5, 0), (8, 6, 0), (9, 7, 1),
]
# positive event -> retracted paper(s); last positive time: P1 7, P2 5, P3 7
PAPERS = {1: "P1", 3: "P2", 6: "P2", 9: ["P3", "P1"]}


def hand_frames():
    events = pd.DataFrame({
        "event_id": [e for e, _, _ in HAND],
        "src_id": [f"s{e}" for e, _, _ in HAND],
        "dst_id": [f"d{e}" for e, _, _ in HAND],
        "t": [t for _, t, _ in HAND],
    })
    labels = pd.DataFrame({"event_id": [e for e, _, _ in HAND], "label": [y for _, _, y in HAND]})
    return events, labels


@pytest.mark.parametrize(
    "min_pos, min_share, expected",
    [
        (2, 0.15, 4),   # c=5 has a single positive
        (1, 0.15, 5),   # c=6 holds only 10% of events
        (1, 0.5, 3),    # share bound is inclusive: 5/10 == 0.5
        (3, 0.0, 2),
        (0, 0.0, 6),    # latest cutoff that still leaves a non-empty test set
    ],
)
def test_hand_cutoff_event_unit(min_pos, min_share, expected):
    events, labels = hand_frames()
    assert choose_cutoff(events, labels, min_test_pos=min_pos, min_test_share=min_share) == expected


@pytest.mark.parametrize("min_pos, expected", [(2, 5), (3, 4), (1, 5)])
def test_hand_cutoff_paper_unit(min_pos, expected):
    events, labels = hand_frames()
    got = choose_cutoff(
        events, labels, min_test_pos=min_pos, min_test_share=0.15,
        positive_unit="paper", paper_of_positive=PAPERS,
    )
    assert got == expected


def test_no_qualifying_cutoff_raises():
    events, labels = hand_frames()
    with pytest.raises(ValueError, match="no cutoff"):
        choose_cutoff(events, labels, min_test_pos=4, min_test_share=0.15)
    with pytest.raises(ValueError, match="no cutoff"):
        choose_cutoff(events, labels, min_test_pos=0, min_test_share=0.9)  # train would be empty


def test_paper_unit_needs_a_mapping_for_every_positive():
    events, labels = hand_frames()
    with pytest.raises(ValueError, match="paper_of_positive"):
        choose_cutoff(events, labels, positive_unit="paper")
    with pytest.raises(ValueError, match="no paper"):
        choose_cutoff(
            events, labels, min_test_pos=1, positive_unit="paper",
            paper_of_positive={1: "P1", 3: "P2"},
        )
    with pytest.raises(ValueError, match="positive_unit"):
        choose_cutoff(events, labels, positive_unit="author")


def test_labels_listing_only_positives_and_series_mapping():
    events, labels = hand_frames()
    only_pos = labels[labels["label"] == 1]
    assert choose_cutoff(events, only_pos, min_test_pos=2, min_test_share=0.15) == 4
    mapping = pd.Series({1: ["P1"], 3: ["P2"], 6: ("P2",), 9: np.array(["P3", "P1"])})
    got = choose_cutoff(events, labels, min_test_pos=2, min_test_share=0.15,
                        positive_unit="paper", paper_of_positive=mapping)
    assert got == 5


def _random_frames(seed, n=400):
    rng = np.random.default_rng(seed)
    events = pd.DataFrame({
        "event_id": rng.permutation(np.arange(1000, 1000 + n)),
        "src_id": [f"s{i}" for i in range(n)],
        "dst_id": [f"d{i}" for i in range(n)],
        "t": rng.integers(0, 60, size=n),
    })
    labels = pd.DataFrame({"event_id": events["event_id"], "label": (rng.random(n) < 0.1).astype(int)})
    papers = {e: [f"P{rng.integers(0, 15)}"] for e in labels.loc[labels.label == 1, "event_id"]}
    return events, labels, papers


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("unit", ["event", "paper"])
def test_brute_force_cutoff(seed, unit):
    events, labels, papers = _random_frames(seed)
    min_pos, share = 8, 0.2
    pos = dict(zip(labels.event_id, labels.label))
    best = None
    for c in sorted(set(events.t))[:-1]:
        test = events[events.t > c]
        pos_ids = [e for e in test.event_id if pos[e] == 1]
        n_pos = len(pos_ids) if unit == "event" else len({p for e in pos_ids for p in papers[e]})
        if n_pos >= min_pos and len(test) / len(events) >= share:
            best = c
    got = choose_cutoff(events, labels, min_test_pos=min_pos, min_test_share=share,
                        positive_unit=unit, paper_of_positive=papers)
    assert got == best


def test_cutoff_is_deterministic_under_row_order():
    events, labels, papers = _random_frames(3)
    a = choose_cutoff(events, labels, min_test_pos=8, positive_unit="paper", paper_of_positive=papers)
    perm = np.random.default_rng(0).permutation(len(events))
    b = choose_cutoff(events.iloc[perm], labels.iloc[perm[::-1]], min_test_pos=8,
                      positive_unit="paper", paper_of_positive=papers)
    assert a == b


def test_split_events_hand_case():
    events = pd.DataFrame({
        "event_id": [10, 11, 12, 13, 14, 15, 16],
        "t": [0, 10, 72, 73, 80, 81, 100],
    }, index=[7, 3, 5, 1, 0, 2, 9])
    # pre-cutoff span 0..80, val = last 10% -> 72 < t <= 80
    split = split_events(events, cutoff=80, val_frac=0.10)
    assert split.index.equals(events.index)
    assert list(split) == ["train", "train", "train", "val", "val", "test", "test"]


def test_split_partitions_and_respects_time_order():
    events, _, _ = _random_frames(1)
    cutoff = 45
    split = split_events(events, cutoff=cutoff)
    assert set(split.unique()) == {"train", "val", "test"}
    assert len(split) == len(events)  # one label per event: no event in two splits
    t = events["t"]
    assert t[split == "train"].max() < t[split == "val"].min()
    assert t[split == "val"].max() <= cutoff < t[split == "test"].min()
    # all events sharing a timestamp land in the same split
    assert (events.assign(s=split).groupby("t")["s"].nunique() == 1).all()


def test_split_is_deterministic_under_row_order():
    events, _, _ = _random_frames(2)
    a = split_events(events, cutoff=40)
    shuffled = events.sample(frac=1.0, random_state=1)
    b = split_events(shuffled, cutoff=40)
    by_id_a = dict(zip(events.event_id, a))
    by_id_b = dict(zip(shuffled.event_id, b))
    assert by_id_a == by_id_b


def test_val_frac_zero_and_bad_cutoff():
    events, _ = hand_frames()
    split = split_events(events, cutoff=4, val_frac=0.0)
    assert (split != "val").all()
    with pytest.raises(ValueError, match="before"):
        split_events(events, cutoff=0)


def test_split_counts_and_yaml_round_trip(tmp_path):
    events, labels = hand_frames()
    split = split_events(events, cutoff=4, val_frac=0.5)  # span 1..4: val = 2.5 < t <= 4
    counts = split_counts(events, split, labels)
    assert counts == {
        "n_events": 10,
        "n_train": 3, "n_val": 3, "n_test": 4,
        "n_pos_train": 1, "n_pos_val": 1, "n_pos_test": 2,
    }
    path = write_split_yaml(tmp_path / "split.yaml", cutoff=np.int64(365), counts=counts,
                            params={"min_test_pos": 100, "val_frac": 0.1})
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert loaded["cutoff_t"] == 365
    assert loaded["cutoff_date"] == (dt.date(1970, 1, 1) + dt.timedelta(days=365)).isoformat()
    assert loaded["cutoff_date"] == "1971-01-01"
    assert loaded["counts"] == counts
    assert loaded["params"] == {"min_test_pos": 100, "val_frac": 0.1}
