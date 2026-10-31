"""S4.5: synthetic cartel injection (CIDRE donor/recipient pattern) and the clique variant.

Spec under test (citecurv.eval.inject):
- inject_cartel adds citation events among member papers of one field, unlabeled, published
  inside the cartel window, with pub date >= t_min. It only adds references to existing
  member papers: an injected src->dst event has t = pub date of src and pub date of dst < t.
- New rows get fresh event_ids continuing after the max; truth_event_ids equals exactly the
  set of new rows; existing rows are unchanged; no duplicate (src, dst) pairs; the returned
  frame is sorted by (t, event_id).
- Roles: along-flow pairs (donor cites recipient) at `internal_density`, every other
  time-feasible member pair at `internal_density * reciprocity`; so donors give more than
  they receive and recipients receive more than they give whenever reciprocity < 1.
- OBVIOUSNESS presets are ordered: realized density blatant > moderate > subtle.
- Reproducible under seed and different under another seed.
- inject_clique: time-consistent analogue of PyGOD gen_structural_outlier(directed=True):
  one edge per member pair with distinct dates, later paper cites earlier paper.
"""

import numpy as np
import pandas as pd
import pytest

from citecurv.data.synthetic import generate_citation_stream
from citecurv.eval.inject import OBVIOUSNESS, InjectionResult, inject_cartel, inject_clique

BASE = dict(n_works=3000, n_fields=4, years=(2012, 2018), refs_mean=8, emb_dim=8, seed=1)
T_MIN = int((pd.Timestamp("2016-01-01") - pd.Timestamp("1970-01-01")).days)


@pytest.fixture(scope="module")
def base():
    return generate_citation_stream(**BASE)


def pub_days(works):
    t = (pd.to_datetime(works["pub_date"]) - pd.Timestamp("1970-01-01")).dt.days
    return dict(zip(works["work_id"], t.astype("int64")))


def new_rows(res, events):
    return res.events[~res.events["event_id"].isin(events["event_id"])]


def run(base, **kw):
    args = dict(n_cartels=4, seed=0, t_min=T_MIN)
    args.update(kw)
    return inject_cartel(base["events"], base["works"], **args)


@pytest.fixture(scope="module")
def default_run(base):
    return run(base, authorships=base["authorships"])


def test_result_shape(default_run, base):
    res = default_run
    assert isinstance(res, InjectionResult)
    assert list(res.events.columns) == list(base["events"].columns)
    assert res.truth_event_ids.dtype == np.int64
    assert list(res.cartels.columns)[:3] == ["cartel_id", "work_id", "role"]
    assert set(res.cartels["role"]) == {"donor", "recipient"}
    assert res.cartels["cartel_id"].nunique() == 4
    assert not res.cartels["work_id"].duplicated().any()  # cartels are disjoint


def test_truth_ids_are_exactly_the_new_rows(default_run, base):
    res, ev = default_run, base["events"]
    added = new_rows(res, ev)
    assert len(added) > 0
    assert sorted(added["event_id"]) == sorted(res.truth_event_ids.tolist())
    assert added["event_id"].min() == ev["event_id"].max() + 1
    assert res.events["event_id"].is_unique
    old = res.events[res.events["event_id"].isin(ev["event_id"])]
    pd.testing.assert_frame_equal(
        old.sort_values("event_id").reset_index(drop=True),
        ev.sort_values("event_id").reset_index(drop=True),
    )


def test_sorted_and_no_duplicate_pairs(default_run):
    ev = default_run.events
    assert (ev.sort_values(["t", "event_id"]).index == ev.index).all()
    assert (ev.index == pd.RangeIndex(len(ev))).all()
    assert not ev.duplicated(["src_id", "dst_id"]).any()


def test_time_consistency(default_run, base):
    pub = pub_days(base["works"])
    added = new_rows(default_run, base["events"])
    assert (added["t"].to_numpy() == added["src_id"].map(pub).to_numpy()).all()
    assert (added["dst_id"].map(pub).to_numpy() < added["t"].to_numpy()).all()


def test_inside_window_member_set_and_t_min(default_run, base):
    res = default_run
    pub = pub_days(base["works"])
    added = new_rows(res, base["events"])
    cartel_of = dict(zip(res.cartels["work_id"], res.cartels["cartel_id"]))
    win = res.cartels.groupby("cartel_id")[["t_window_start", "t_window_end"]].first()
    src_c = added["src_id"].map(cartel_of)
    dst_c = added["dst_id"].map(cartel_of)
    assert src_c.notna().all() and (src_c == dst_c).all()  # both ends in the same cartel
    lo = src_c.map(win["t_window_start"]).to_numpy()
    hi = src_c.map(win["t_window_end"]).to_numpy()
    t = added["t"].to_numpy()
    assert ((t >= lo) & (t <= hi)).all()
    assert (t >= T_MIN).all()
    member_t = res.cartels["work_id"].map(pub)
    assert (member_t >= res.cartels["t_window_start"]).all()
    assert (member_t <= res.cartels["t_window_end"]).all()
    assert (member_t >= T_MIN).all()
    assert ((win["t_window_end"] - win["t_window_start"]) == 365).all()
    assert (win["t_window_end"] <= max(pub.values())).all()  # whole window inside the data


def test_one_field_per_cartel_and_unlabeled(base):
    works = base["works"].copy()
    rng = np.random.default_rng(0)
    flagged = set(works["work_id"].iloc[rng.choice(len(works), 300, replace=False)])
    works["is_retracted_oa"] = works["work_id"].isin(flagged)
    excluded = set(works["work_id"].iloc[2000:2300])
    res = inject_cartel(base["events"], works, n_cartels=3, seed=2, t_min=T_MIN,
                        exclude_ids=excluded)
    field = dict(zip(works["work_id"], works["field"]))
    assert (res.cartels.groupby("cartel_id")["work_id"]
            .agg(lambda s: s.map(field).nunique()) == 1).all()
    assert (res.cartels["field"] == res.cartels["work_id"].map(field)).all()
    members = set(res.cartels["work_id"])
    assert not members & flagged
    assert not members & excluded


def test_fixed_field(base):
    res = run(base, field="F02", n_cartels=2)
    assert set(res.cartels["field"]) == {"F02"}


def test_disjoint_authors_when_authorships_given(default_run, base):
    auth = base["authorships"]
    for _, grp in default_run.cartels.groupby("cartel_id"):
        a = auth[auth["work_id"].isin(grp["work_id"])]
        assert not a["author_id"].duplicated().any()  # no author on two member papers


def test_reproducible_and_seed_sensitive(base, default_run):
    again = run(base, authorships=base["authorships"])
    pd.testing.assert_frame_equal(again.events, default_run.events)
    pd.testing.assert_frame_equal(again.cartels, default_run.cartels)
    assert np.array_equal(again.truth_event_ids, default_run.truth_event_ids)
    other = run(base, authorships=base["authorships"], seed=1)
    assert set(other.cartels["work_id"]) != set(default_run.cartels["work_id"])


def _role_degrees(res, events):
    added = new_rows(res, events)
    role = dict(zip(res.cartels["work_id"], res.cartels["role"]))
    out_deg = added.groupby("src_id").size()
    in_deg = added.groupby("dst_id").size()
    deg = res.cartels.assign(
        out=res.cartels["work_id"].map(out_deg).fillna(0),
        inn=res.cartels["work_id"].map(in_deg).fillna(0),
    )
    return deg.groupby("role")[["out", "inn"]].sum(), added, role


def test_donor_recipient_asymmetry(base):
    res = run(base, n_cartels=8, reciprocity=0.3, internal_density=0.5, size_range=(12, 20))
    sums, _, _ = _role_degrees(res, base["events"])
    assert sums.loc["donor", "out"] > 1.5 * sums.loc["donor", "inn"]
    assert sums.loc["recipient", "inn"] > 1.5 * sums.loc["recipient", "out"]


def test_reciprocity_knob_is_the_return_rate(base):
    """Realized (recipient->donor rate) / (donor->recipient rate) tracks `reciprocity`."""
    pub = pub_days(base["works"])
    for rho in (0.2, 0.8):
        res = run(base, n_cartels=8, reciprocity=rho, internal_density=0.6, size_range=(15, 25))
        added = new_rows(res, base["events"])
        role = dict(zip(res.cartels["work_id"], res.cartels["role"]))
        n_edges = {("donor", "recipient"): 0, ("recipient", "donor"): 0}
        for s, d in zip(added["src_id"], added["dst_id"]):
            if (role[s], role[d]) in n_edges:
                n_edges[(role[s], role[d])] += 1
        feas = {("donor", "recipient"): 0, ("recipient", "donor"): 0}
        for _, grp in res.cartels.groupby("cartel_id"):
            ids = list(grp["work_id"])
            for a in ids:
                for b in ids:
                    if pub[a] > pub[b] and (role[a], role[b]) in feas:
                        feas[(role[a], role[b])] += 1
        rate = {k: n_edges[k] / feas[k] for k in feas}
        assert rate[("donor", "recipient")] == pytest.approx(0.6, abs=0.12)
        realized = rate[("recipient", "donor")] / rate[("donor", "recipient")]
        assert realized == pytest.approx(rho, abs=0.15)


def _realized_density(res, works):
    pub = pub_days(works)
    edges = len(res.truth_event_ids)
    feasible = 0
    for _, grp in res.cartels.groupby("cartel_id"):
        t = np.array([pub[w] for w in grp["work_id"]])
        feasible += int((t[:, None] > t[None, :]).sum())
    return edges / feasible


def test_presets_ordered(base):
    dens = {
        name: _realized_density(run(base, n_cartels=4, **OBVIOUSNESS[name]), base["works"])
        for name in ("blatant", "moderate", "subtle")
    }
    assert dens["blatant"] > dens["moderate"] > dens["subtle"] > 0
    keys = ("internal_density", "reciprocity")
    for k in keys:
        assert OBVIOUSNESS["blatant"][k] > OBVIOUSNESS["moderate"][k] > OBVIOUSNESS["subtle"][k]


def test_skips_pairs_that_already_exist(base):
    """Pre-existing member->member citations are not duplicated nor counted as injected."""
    first = run(base, n_cartels=1, internal_density=1.0, reciprocity=1.0, size_range=(10, 10))
    members = list(first.cartels["work_id"])
    pub = pub_days(base["works"])
    ev = base["events"]
    existing = set(zip(ev["src_id"], ev["dst_id"]))
    later, earlier = next(
        (a, b) for a in members for b in members if pub[a] > pub[b] and (a, b) not in existing
    )
    planted = pd.DataFrame({"event_id": [ev["event_id"].max() + 1], "src_id": [later],
                            "dst_id": [earlier], "t": [pub[later]]})
    ev2 = pd.concat([ev, planted], ignore_index=True).sort_values(["t", "event_id"])
    ev2 = ev2.reset_index(drop=True)
    res = inject_cartel(ev2, base["works"], n_cartels=1, seed=0, t_min=T_MIN,
                        internal_density=1.0, reciprocity=1.0, size_range=(10, 10))
    assert set(res.cartels["work_id"]) == set(members)  # same draw: the planted edge is internal
    assert not res.events.duplicated(["src_id", "dst_id"]).any()
    assert len(res.truth_event_ids) == len(first.truth_event_ids) - 1
    assert planted["event_id"].iloc[0] not in set(res.truth_event_ids)


def test_raises_when_no_window_fits(base):
    with pytest.raises(ValueError):
        run(base, size_range=(500, 500), window_days=30)


def test_event_labels_helper(default_run):
    lab = default_run.event_labels()
    assert list(lab.columns) == ["event_id", "label"]
    assert (lab["event_id"].to_numpy() == default_run.events["event_id"].to_numpy()).all()
    assert set(lab.loc[lab["label"] == 1, "event_id"]) == set(default_run.truth_event_ids)


# --- clique variant -------------------------------------------------------------------

def test_clique_full_time_feasible_pairs(base):
    res = inject_clique(base["events"], base["works"], n_cliques=3, clique_size=10, seed=0,
                        t_min=T_MIN)
    pub = pub_days(base["works"])
    assert len(res.cartels) == 30 and set(res.cartels["role"]) == {"clique"}
    assert (res.cartels["work_id"].map(pub) >= T_MIN).all()
    added = new_rows(res, base["events"])
    assert sorted(added["event_id"]) == sorted(res.truth_event_ids.tolist())
    assert (added["dst_id"].map(pub).to_numpy() < added["t"].to_numpy()).all()
    assert (added["t"].to_numpy() == added["src_id"].map(pub).to_numpy()).all()
    existing = set(zip(base["events"]["src_id"], base["events"]["dst_id"]))
    expected = set()
    for _, grp in res.cartels.groupby("cartel_id"):
        for a in grp["work_id"]:
            for b in grp["work_id"]:
                if pub[a] > pub[b] and (a, b) not in existing:
                    expected.add((a, b))
    assert set(zip(added["src_id"], added["dst_id"])) == expected


def test_clique_drop_prob_and_seed(base):
    full = inject_clique(base["events"], base["works"], 2, 12, seed=3, t_min=T_MIN)
    dropped = inject_clique(base["events"], base["works"], 2, 12, seed=3, t_min=T_MIN,
                            drop_prob=0.5)
    assert len(dropped.truth_event_ids) < len(full.truth_event_ids)
    # PyGOD rule: keep int((1 - p) * n) of all new candidate edges, pooled over cliques
    pub = pub_days(base["works"])
    n_cand = sum(int((t[:, None] > t[None, :]).sum()) for t in (
        np.array([pub[w] for w in g["work_id"]]) for _, g in dropped.cartels.groupby("cartel_id")))
    existing = set(zip(base["events"]["src_id"], base["events"]["dst_id"]))
    members = set(dropped.cartels["work_id"])
    n_existing = sum(1 for s_, d_ in existing if s_ in members and d_ in members)
    kept = int(0.5 * n_cand)
    assert kept - n_existing <= len(dropped.truth_event_ids) <= kept
    again = inject_clique(base["events"], base["works"], 2, 12, seed=3, t_min=T_MIN)
    pd.testing.assert_frame_equal(again.events, full.events)
