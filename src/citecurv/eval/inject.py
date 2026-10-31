"""Synthetic cartel injection (task S4.5, decision L18; FR10).

Two injectors give controlled ground truth on top of a natural stream (real or synthetic):

- inject_cartel: a time-consistent version of the donor/recipient pattern that CIDRE finds
  in journal citation networks (Kojaku, Livan, & Masuda, 2021): a group with excess
  internal citations, in which some members mostly give citations (donors) and others
  mostly receive them (recipients).
- inject_clique: a time-consistent analogue of PyGOD's gen_structural_outlier(directed=True)
  (pygod 1.1.0), the crude comparison variant. PyGOD links every pair of each random clique
  with one directed edge whose direction follows a random order and drops a share p of
  them. Here the direction is forced by time (the later paper cites the earlier one), and
  pairs published on the same day are skipped. The real PyGOD call happens in the baseline
  environment.

Design decisions (shared by both injectors):

1. References are added to EXISTING member papers; no new papers are created. An injected
   event src->dst gets t = pub date of src (all references of a paper share its date) and
   requires pub date of dst < pub date of src, the same strict rule as the generator. The
   anomaly therefore lives only in edges: works, embeddings and authorships are unchanged,
   so the clean and the injected run share every node-level input, and no injected event
   can precede the paper that makes it. This mirrors coerced or "sneaked" references being
   added to real reference lists.
2. Members are papers of one field (works["field"], else the first concept), unlabeled
   (is_retracted_oa False and not in exclude_ids), published inside the cartel window
   [w0, w0 + window_days] with w0 >= t_min. Hence every injected event has t >= t_min and
   an evaluation with cutoff t_min can assert that all of them are post-cutoff. Windows
   that end before the last observed publication day are preferred (a window cut short by
   the data end would squeeze the members together). Cartels are
   disjoint. When `authorships` is given, members of one cartel have pairwise disjoint
   author sets, so no injected edge is a self-citation (a cartel is a group of distinct
   teams, not one team citing itself).
3. Reciprocity. A citation points back in time, so two papers can never cite each other:
   pairwise reciprocity is identically 0 at the paper level (in real data as well). The
   knob is therefore defined on the CIDRE roles, collapsed to two blocks. Roles are drawn
   at random among members (a share recipient_share are recipients, at least one of each).
   Every time-feasible ordered member pair (u later than v) becomes an edge u->v with
   probability internal_density when u is a donor and v a recipient (the donation flow),
   and with probability internal_density * reciprocity for every other role combination.
   So reciprocity = 1 gives a homogeneous dense group, reciprocity = 0 a pure one-way
   donation; the realized rate of recipient->donor edges over the donor->recipient rate
   estimates `reciprocity`. For reciprocity < 1 donors give more than they receive and
   recipients receive more than they give.
4. A sampled pair that already exists in `events` is skipped (no duplicate (src, dst)).
   New rows get event_ids max(event_id) + 1, + 2, ... in (t, src_id, dst_id) order and the
   returned frame is re-sorted by (t, event_id), so inside one timestamp the injected rows
   come after the natural ones.

5. Age-matched variant (inject_cartel(age_matched=True)). In the default construction both ends
   of every cartel edge lie in the cartel window, so the cited paper is at most window_days old
   and cited-paper age alone separates the cartel (docs/rehearsal_report.md section 5). With
   age_matched=True the cartel is split into a citing core and a cited set. The donors are drawn
   in the window as before (they make every injected citation, so t >= w0 >= t_min still holds).
   The recipients are drawn from older papers of the same field: for each recipient an age a is
   drawn from the field's natural cited-age distribution (t - pub date of dst over the events of
   `events` with t < w0 whose citing paper is in the field; all fields when the field has fewer
   than 30 such events), and the unused eligible paper whose publication day is closest to
   (mean donor publication day - a) is taken (same eligibility and author-disjointness rules,
   but no t_min: recipients may predate t_min; they only receive). Edges: every donor -> recipient
   pair with pub(recipient) < pub(donor) becomes an edge with probability internal_density.
   Donor -> donor edges are not planted (their cited age is bounded by window_days, the signal the
   variant removes) and recipients make no citations (a citation made by an old recipient would
   precede t_min), so `reciprocity` is ignored: the group is a pure one-way donation flow, the
   reciprocity = 0 end of the CIDRE pattern. The distribution is measured on the events passed in;
   pass the pre-injection stream to keep earlier injections out of it. No labels are read.

Leak argument (NFR3). The injector only uses publication dates already present in `works`
and draws no information from events later than an injected event's t: each injected
event is attached to its source paper's own publication date, like the paper's natural
references. Ground truth (truth_event_ids, cartels) is outcome data and must never be fed
to a scorer. event_id is a key, not a feature: injected rows carry the largest ids, so a
model that consumed event_id as an input would see them as a separable block.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Obviousness ladder (book 3.3): density and reciprocity stepped down from blatant to subtle;
# smaller groups spread over longer windows are harder to see. Synthetic design values.
# Window length matters because both ends of a cartel edge lie in the window, so cited-paper
# age is at most window_days. On the default synthetic stream (10 cartels, events after
# 2021-01-01) cited-paper recency alone gives AUROC 0.99 / 0.98 / 0.94 but AP 0.33 / 0.05 /
# 0.004 for blatant / moderate / subtle (results/synthetic_profile.json, injection_check).
OBVIOUSNESS = {
    "blatant": {"size_range": (15, 25), "internal_density": 0.6, "reciprocity": 0.9,
                "window_days": 365},
    "moderate": {"size_range": (10, 20), "internal_density": 0.35, "reciprocity": 0.6,
                 "window_days": 730},
    "subtle": {"size_range": (8, 15), "internal_density": 0.2, "reciprocity": 0.3,
               "window_days": 1460},
}

CARTEL_COLUMNS = ["cartel_id", "work_id", "role", "field", "t_window_start", "t_window_end"]


@dataclass
class InjectionResult:
    events: pd.DataFrame          # input events + injected rows, sorted by (t, event_id)
    truth_event_ids: np.ndarray   # int64, sorted: exactly the injected rows
    cartels: pd.DataFrame         # CARTEL_COLUMNS, one row per member paper

    def event_labels(self) -> pd.DataFrame:
        """Synthetic label rule (CONTRACTS.md): an injected event is positive."""
        ids = self.events["event_id"].to_numpy(np.int64)
        return pd.DataFrame({"event_id": ids,
                             "label": np.isin(ids, self.truth_event_ids).astype(np.int64)})


def _canonical_cartels(df):
    return pd.DataFrame({
        "cartel_id": df["cartel_id"].astype(np.int64),
        "work_id": df["work_id"].astype(str),
        "role": df["role"].astype(str),
        "field": df["field"].astype(str),
        "t_window_start": df["t_window_start"].astype(np.int64),
        "t_window_end": df["t_window_end"].astype(np.int64),
    }).reset_index(drop=True)


def _first_concept(c):
    if isinstance(c, str):
        return c or None
    try:
        return c[0] if len(c) else None
    except TypeError:
        return None


def _pub_days(works):
    """pub_date -> days since 1970-01-01 (float, NaN when missing or unparsable)."""
    return (pd.to_datetime(works["pub_date"], errors="coerce") - pd.Timestamp("1970-01-01")).dt.days


def _eligible_works(works, t_min, exclude_ids):
    """work_id, t, field of unlabeled dated works with t >= t_min, sorted by (t, work_id)."""
    t = _pub_days(works)
    field = works["field"] if "field" in works else works["concepts"].map(_first_concept)
    retracted = (works["is_retracted_oa"].fillna(False).astype(bool)
                 if "is_retracted_oa" in works else pd.Series(False, index=works.index))
    info = pd.DataFrame({"work_id": works["work_id"].astype(str), "t": t, "field": field})
    keep = info["t"].notna() & info["field"].notna() & ~retracted
    if exclude_ids is not None:
        keep &= ~info["work_id"].isin(set(exclude_ids))
    info = info[keep].astype({"t": np.int64, "field": str})
    if t_min is not None:
        info = info[info["t"] >= int(t_min)]
    return info.sort_values(["t", "work_id"], kind="stable").reset_index(drop=True)


def _author_sets(authorships, work_ids):
    if authorships is None:
        return None
    a = authorships[authorships["work_id"].isin(set(work_ids))]
    return {w: frozenset(g) for w, g in a.groupby("work_id")["author_id"]}


def _pick_members(rng, cand, size, window_days, author_sets, t_last):
    """Choose a window start and `size` members from cand (one field, sorted by t).
    Starts whose whole window ends by t_last (last observed pub day) are preferred, so that
    members spread over the full window instead of a window cut short by the data end.
    Returns (w0, member work_ids) or None when no window holds enough papers."""
    t = cand["t"].to_numpy()
    ids = cand["work_id"].to_numpy()
    starts = np.unique(t)
    count = np.searchsorted(t, starts + window_days, "right") - np.searchsorted(t, starts, "left")
    valid = starts[count >= size]
    inside = valid[valid + window_days <= t_last]
    if len(inside):
        valid = inside
    for w0 in rng.permutation(valid):
        lo, hi = np.searchsorted(t, w0, "left"), np.searchsorted(t, w0 + window_days, "right")
        order = rng.permutation(hi - lo) + lo
        if author_sets is None:
            return int(w0), list(ids[order[:size]])
        picked, seen = [], set()
        for p in order:
            authors = author_sets.get(ids[p], frozenset())
            if authors & seen:
                continue
            picked.append(ids[p])
            seen |= authors
            if len(picked) == size:
                return int(w0), picked
    return None


def _natural_ages(events, works):
    """(t, citing field, cited age in days) of the given events with a dated cited paper, sorted by t."""
    pub = pd.Series(_pub_days(works).to_numpy(), index=works["work_id"].astype(str).to_numpy())
    fld = works["field"] if "field" in works else works["concepts"].map(_first_concept)
    fld = pd.Series(fld.to_numpy(), index=works["work_id"].astype(str).to_numpy())
    t = events["t"].to_numpy(np.int64)
    age = t - pub.reindex(events["dst_id"].astype(str).to_numpy()).to_numpy()
    keep = np.isfinite(age) & (age > 0)
    out = pd.DataFrame({"t": t[keep], "field": fld.reindex(events["src_id"].astype(str).to_numpy()).to_numpy()[keep],
                        "age": age[keep]})
    return out.sort_values("t", kind="stable").reset_index(drop=True)


def _field_ages(nat, f, w0, min_events=30):
    """Natural cited ages of field f from events with t < w0 (all fields when fewer than min_events)."""
    before = nat[nat["t"] < w0]
    ages = before.loc[before["field"] == f, "age"].to_numpy()
    if len(ages) < min_events:
        ages = before["age"].to_numpy()
    if not len(ages):
        raise ValueError(f"age_matched: no natural citation before window start {w0}")
    return ages


def _pick_recipients(rng, cand, n, anchor, ages, author_sets, seen):
    """n recipients from cand (one field): for each, draw an age from `ages` and take the unused,
    author-disjoint paper whose pub day is closest to anchor - age. Returns (ids, pub days)."""
    cand = cand.sort_values(["t", "work_id"], kind="stable")
    t, ids = cand["t"].to_numpy(), cand["work_id"].to_numpy()
    taken = np.zeros(len(t), dtype=bool)
    seen = set(seen)
    out_id, out_t = [], []
    for a in rng.choice(ages, size=n):
        pos = int(np.searchsorted(t, anchor - a))
        lo, hi = pos - 1, pos
        while lo >= 0 or hi < len(t):  # walk outward from the target day
            if hi >= len(t) or (lo >= 0 and (anchor - a) - t[lo] <= t[hi] - (anchor - a)):
                k, lo = lo, lo - 1
            else:
                k, hi = hi, hi + 1
            if taken[k]:
                continue
            authors = author_sets.get(ids[k], frozenset()) if author_sets is not None else frozenset()
            if authors & seen:
                continue
            taken[k] = True
            seen |= authors
            out_id.append(ids[k])
            out_t.append(int(t[k]))
            break
        else:
            raise ValueError("age_matched: not enough eligible older papers for the recipients")
    return np.array(out_id, dtype=object), np.array(out_t, dtype=np.int64)


def _feasible_pairs(t):
    """Ordered member pairs (a, b), a published strictly after b: a may cite b."""
    a, b = np.nonzero(t[:, None] > t[None, :])
    return a, b


def _existing_pairs(events, members):
    m = set(members)
    sub = events[events["src_id"].isin(m) & events["dst_id"].isin(m)]
    return set(zip(sub["src_id"], sub["dst_id"]))


def _append(events, src, dst, t):
    """Append injected rows with fresh ids; return (sorted events, truth ids)."""
    new = pd.DataFrame({"src_id": pd.Series(src, dtype=events["src_id"].dtype),
                        "dst_id": pd.Series(dst, dtype=events["dst_id"].dtype),
                        "t": np.asarray(t, dtype=np.int64)})
    new = new.sort_values(["t", "src_id", "dst_id"], kind="stable").reset_index(drop=True)
    start = int(events["event_id"].max()) + 1 if len(events) else 0
    new.insert(0, "event_id", np.arange(start, start + len(new), dtype=np.int64))
    if "doi_pair_key" in events:
        new["doi_pair_key"] = new["src_id"] + "|" + new["dst_id"]
    out = pd.concat([events, new[[c for c in events.columns if c in new]]], ignore_index=True)
    out = out.sort_values(["t", "event_id"], kind="stable").reset_index(drop=True)
    return out, new["event_id"].to_numpy(np.int64)


def inject_cartel(events, works, n_cartels, size_range=(8, 25), reciprocity=0.7,
                  internal_density=0.5, window_days=365, seed=0, t_min=None, field=None,
                  authorships=None, *, recipient_share=0.4, exclude_ids=None,
                  age_matched=False, recipient_pool=None) -> InjectionResult:
    """Plant n_cartels citation cartels on the CIDRE donor/recipient pattern.

    events: CONTRACTS.md events frame (event_id, src_id, dst_id, t [, extra columns]).
    works: works frame with work_id, pub_date and field (or concepts); is_retracted_oa if
        present marks labeled papers, which are never members; exclude_ids adds more.
    size_range: inclusive range of member papers per cartel (drawn uniformly).
    internal_density: edge probability along the donation flow (donor cites recipient).
    reciprocity: multiplier for every other time-feasible member pair (module docstring 3).
    window_days: members are published in [w0, w0 + window_days]; w0 >= t_min.
    field: restrict every cartel to this field (default: a random field with a valid window).
    authorships: optional (work_id, author_id); members then have disjoint author sets.
    recipient_share: share of members given the recipient role.
    age_matched: draw recipients among older papers so that cited ages follow the field's natural
        cited-age distribution (module docstring 5); reciprocity is then ignored.
    recipient_pool: works frame the age-matched recipients are drawn from (default: works); lets a
        caller restrict donors to one period's works while recipients come from the whole stream.
    Returns InjectionResult(events, truth_event_ids, cartels); see the module docstring for
    time consistency, id assignment and the leak argument. Raises ValueError when a cartel
    cannot be placed.
    """
    lo, hi = int(size_range[0]), int(size_range[1])
    if not 2 <= lo <= hi:
        raise ValueError("size_range must satisfy 2 <= low <= high")
    for name, val in (("internal_density", internal_density), ("reciprocity", reciprocity),
                      ("recipient_share", recipient_share)):
        if not 0.0 <= val <= 1.0:
            raise ValueError(f"{name} must lie in [0, 1]")

    rng = np.random.default_rng(seed)
    info = _eligible_works(works, t_min, exclude_ids)
    t_last = _pub_days(works).max()  # last observed publication day
    old = (_eligible_works(works if recipient_pool is None else recipient_pool, None, exclude_ids)
           if age_matched else None)
    a_sets = _author_sets(authorships, pd.concat([info["work_id"], old["work_id"]]) if age_matched
                          else info["work_id"])
    nat = _natural_ages(events, works if recipient_pool is None else recipient_pool) if age_matched else None
    fields = [field] if field is not None else sorted(info["field"].unique())
    used: set = set()
    src_all, dst_all, t_all, rows = [], [], [], []

    for c in range(n_cartels):
        size = int(rng.integers(lo, hi + 1))
        n_rec = min(max(int(round(recipient_share * size)), 1), size - 1)
        placed = None
        for f in rng.permutation(np.array(fields, dtype=object)):
            cand = info[(info["field"] == f) & ~info["work_id"].isin(used)]
            placed = _pick_members(rng, cand, size - n_rec if age_matched else size, window_days, a_sets, t_last)
            if placed is not None:
                break
        if placed is None:
            raise ValueError(f"cartel {c}: no window of {window_days} days holds {size} eligible "  # donors only when age_matched
                             f"papers of one field (t_min={t_min}, field={field})")
        w0, members = placed
        if age_matched:
            donors = members  # the citing core; the recipients are drawn below
            mem = info[info["work_id"].isin(donors)]
            ids, t = mem["work_id"].to_numpy(), mem["t"].to_numpy()
            used.update(donors)
            rec_ids, rec_t = _pick_recipients(rng, old[(old["field"] == f) & ~old["work_id"].isin(used)],
                                              n_rec, float(t.mean()), _field_ages(nat, f, w0), a_sets,
                                              set().union(*[a_sets.get(d, frozenset()) for d in donors])
                                              if a_sets is not None else set())
            used.update(rec_ids)
            existing = _existing_pairs(events, list(ids) + list(rec_ids))
            for i in range(len(ids)):
                ok = rec_t < t[i]
                hit = ok & (rng.random(len(rec_ids)) < internal_density)
                for j in np.flatnonzero(hit):
                    if (ids[i], rec_ids[j]) not in existing:
                        src_all.append(ids[i])
                        dst_all.append(rec_ids[j])
                        t_all.append(t[i])
            rows.extend((c, w, "donor", f, w0, w0 + window_days) for w in ids)
            rows.extend((c, w, "recipient", f, w0, w0 + window_days) for w in rec_ids)
            continue
        used.update(members)
        mem = info[info["work_id"].isin(members)]  # keeps (t, work_id) order
        ids, t = mem["work_id"].to_numpy(), mem["t"].to_numpy()

        recipient = np.zeros(size, dtype=bool)
        recipient[rng.choice(size, size=n_rec, replace=False)] = True

        a, b = _feasible_pairs(t)
        along = ~recipient[a] & recipient[b]
        p = np.where(along, internal_density, internal_density * reciprocity)
        keep = rng.random(len(a)) < p
        existing = _existing_pairs(events, ids)
        for i, j in zip(a[keep], b[keep]):
            if (ids[i], ids[j]) not in existing:
                src_all.append(ids[i])
                dst_all.append(ids[j])
                t_all.append(t[i])
        for k in range(size):
            rows.append((c, ids[k], "recipient" if recipient[k] else "donor", f, w0,
                         w0 + window_days))

    out, truth = _append(events, src_all, dst_all, t_all)
    cartels = _canonical_cartels(pd.DataFrame(rows, columns=CARTEL_COLUMNS))
    return InjectionResult(out, np.sort(truth), cartels)


def inject_clique(events, works, n_cliques, clique_size, seed=0, t_min=None, *,
                  drop_prob=0.0, exclude_ids=None) -> InjectionResult:
    """Time-consistent analogue of PyGOD gen_structural_outlier(m=clique_size, n=n_cliques,
    p=drop_prob, directed=True).

    Members: n_cliques * clique_size distinct unlabeled works with pub date >= t_min, drawn
    uniformly at random from all fields and dates (as PyGOD draws nodes uniformly). Each
    member pair published on different days gets one edge, later paper cites earlier paper;
    same-day pairs have no time-consistent direction and are skipped. As in PyGOD, a share
    drop_prob of the new edges (pooled over all cliques) is dropped: int((1 - p) * n) are
    kept. Pairs already in `events` are skipped afterwards. cartels rows carry role "clique", the member's own
    field, and the clique's first and last publication day as its window.
    """
    if clique_size < 2 or n_cliques < 1:
        raise ValueError("need n_cliques >= 1 and clique_size >= 2")
    if not 0.0 <= drop_prob < 1.0:
        raise ValueError("drop_prob must lie in [0, 1)")
    rng = np.random.default_rng(seed)
    info = _eligible_works(works, t_min, exclude_ids).sort_values("work_id", kind="stable")
    need = n_cliques * clique_size
    if need > len(info):
        raise ValueError(f"need {need} eligible works, have {len(info)}")
    pick = info.iloc[rng.choice(len(info), size=need, replace=False)]
    cand, rows = [], []  # candidate edges (src, dst, t) over all cliques, in clique order
    for c in range(n_cliques):
        mem = pick.iloc[c * clique_size:(c + 1) * clique_size].sort_values(["t", "work_id"])
        ids, t, fld = mem["work_id"].to_numpy(), mem["t"].to_numpy(), mem["field"].to_numpy()
        a, b = _feasible_pairs(t)
        cand.extend(zip(ids[a], ids[b], t[a]))
        rows.extend((c, ids[k], "clique", fld[k], int(t.min()), int(t.max()))
                    for k in range(clique_size))
    if drop_prob > 0:  # PyGOD keeps int((1 - p) * n) of all new edges, across cliques
        kept = np.sort(rng.permutation(len(cand))[: int((1 - drop_prob) * len(cand))])
        cand = [cand[i] for i in kept]
    existing = _existing_pairs(events, pick["work_id"])
    cand = [e for e in cand if (e[0], e[1]) not in existing]
    out, truth = _append(events, [e[0] for e in cand], [e[1] for e in cand], [e[2] for e in cand])
    cartels = _canonical_cartels(pd.DataFrame(rows, columns=CARTEL_COLUMNS))
    return InjectionResult(out, np.sort(truth), cartels)
