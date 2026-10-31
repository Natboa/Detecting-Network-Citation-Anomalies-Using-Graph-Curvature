"""Synthetic citation stream: stand-in data for rehearsing the pipeline before S4.0.

Everything this module produces is synthetic and must be labelled as such wherever it
appears (tables, figures, notes). Cartels are planted afterwards by
citecurv.eval.inject; this module generates the natural background stream.

Generative model. Works are created in publication order; each knob below is a keyword
argument of generate_citation_stream (defaults in DEFAULT_KNOBS). The defaults are
stand-in values chosen to give a plausible shape, not values calibrated on the real slice;
recalibrate them against the S4.0 dataset profile.

1. Output growth: the number of works in year y is proportional to (1 + growth_rate)^y;
   publication days are uniform within each year (`years` is inclusive).
2. Fields: each work belongs to one of n_fields fields, with shares drawn from a symmetric
   Dirichlet(field_concentration); F00 is the largest field. Each field has
   venues_per_field venues with Zipf(venue_zipf) popularity; a venue belongs to one field.
   Each work also gets one of n_subtopics subtopics of its field.
3. Authors and careers: a team has 1 + Poisson(authors_mean - 1) authors. Each slot is a
   new author with probability new_author_prob (career length ~ Exponential with mean
   career_mean_years, activity level ~ LogNormal(0, author_activity_sigma)), otherwise an
   active author of the same field chosen with weight equal to its activity level
   (heterogeneous productivity without runaway accumulation).
4. Reference-list length: lognormal with sigma refs_sigma; a share review_rate of works
   are review-like with review_factor times longer lists. Lengths are scaled so that the
   nominal mean is refs_mean. The realized mean is lower, because a work can only cite
   works published before it inside the stream (the first years have few predecessors).
5. Reference targets. Each reference slot is, independently:
   - a self-citation (self_cite_rate): uniform among earlier works sharing an author;
   - a noise citation (noise_rate): uniform among all earlier works, any field, ignoring
     popularity and age (natural odd edges, so that not every odd edge is a cartel);
   - a cross-field citation (cross_field_rate): another field chosen by field share,
     then a popularity-and-age draw inside it;
   - otherwise an own-field citation: a popularity-and-age draw inside the own field,
     with weight multiplied by subtopic_affinity for works of the same subtopic.
   Popularity-and-age weight of candidate j cited at time t:
       (in_degree_j + pa_offset) ** pa_exponent * (1 - exp(-a / ageing_rise_years))
       * exp(-a / ageing_decay_years),            a = age of j at t in years,
   i.e. preferential attachment with a rise-then-decay ageing kernel. Slots that cannot be
   filled (no eligible earlier work) fall back to the own field; if that is exhausted too,
   the reference is dropped. Sampling is without replacement, so no duplicate pairs.
6. Embeddings: global offset + field centroid + subtopic offset + slow drift of the field
   centroid along a fixed direction (linear in time over the year span) + isotropic paper
   noise. Component norms are emb_global, emb_field, emb_subtopic, emb_drift (reached at
   the end of the span) and emb_noise (expected norm). Cosine similarity therefore carries
   field signal and, through subtopic affinity, mild citation signal.

Time convention: a work cites only works published strictly earlier,
pub_date(dst) < pub_date(src). Same-day citations exist in real data but are not
generated; this keeps the stream acyclic and every event time-consistent with
t = pub_date(src) (CONTRACTS.md). A consequence, true of real paper-level data as well:
no pair of works can cite each other, so paper-level reciprocity is 0.

Leak note: the generator is data, not a feature, so NFR3 does not constrain it; it hands
the pipeline a stream that already satisfies the contract every leak argument relies on
(event time = pub date of the citing work, references point strictly backwards).
Determinism (NFR4): every random draw comes from numpy Generators spawned from `seed`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from citecurv.config import REPO_ROOT

_EPOCH = np.datetime64("1970-01-01", "D")
_DAYS_PER_YEAR = 365.25

DEFAULT_KNOBS = {
    "growth_rate": 0.04,          # yearly growth of output
    "field_concentration": 4.0,   # Dirichlet concentration of field shares (lower = more uneven)
    "cross_field_rate": 0.10,     # share of reference slots aimed at another field
    "noise_rate": 0.02,           # share of uniform random citations (any field, any age)
    "self_cite_rate": 0.08,       # share of slots aimed at earlier works of a co-author
    "pa_offset": 1.0,             # attractiveness = (in_degree + offset) ** exponent
    "pa_exponent": 1.0,           # 0 switches preferential attachment off
    "ageing_rise_years": 0.5,     # rise time of the ageing kernel
    "ageing_decay_years": 6.0,    # decay time of the ageing kernel
    "refs_sigma": 0.6,            # lognormal sigma of reference-list length
    "review_rate": 0.02,          # share of review-like works
    "review_factor": 4.0,         # review lists are this many times longer
    "authors_mean": 3.0,          # mean team size (1 + Poisson)
    "new_author_prob": 0.3,       # chance a team slot is a career starter
    "career_mean_years": 8.0,     # mean career length (exponential)
    "author_activity_sigma": 1.0, # lognormal sigma of per-author activity (productivity spread)
    "venues_per_field": 6,
    "venue_zipf": 1.0,            # Zipf exponent of venue popularity within a field
    "n_subtopics": 8,             # subtopics per field
    "subtopic_affinity": 5.0,     # own-field weight multiplier for same-subtopic candidates
    "emb_global": 0.5,            # norm of the offset shared by all works
    "emb_field": 1.0,             # norm of each field centroid
    "emb_subtopic": 0.8,          # norm of each subtopic offset
    "emb_drift": 0.5,             # drift of the field centroid over the full year span
    "emb_noise": 1.0,             # expected norm of the per-work noise
}

MINI_FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "mini_stream"
MINI_GENERATOR = {"n_works": 60, "n_fields": 2, "years": [2019, 2022], "refs_mean": 3.5,
                  "emb_dim": 16, "seed": 7}
MINI_INJECTION = {"n_cartels": 1, "size_range": [6, 6], "internal_density": 0.8,
                  "reciprocity": 0.8, "window_days": 365, "seed": 0}
MINI_T_MIN_DATE = "2021-01-01"


def to_days(pub_dates) -> np.ndarray:
    """ISO dates (YYYY-MM-DD) -> int64 days since 1970-01-01."""
    arr = np.asarray(pub_dates, dtype="datetime64[D]")
    return (arr - _EPOCH).astype(np.int64)


def _to_iso(days: np.ndarray) -> np.ndarray:
    return np.datetime_as_string(_EPOCH + days.astype("timedelta64[D]"), unit="D")


def _unit(rng, n, d):
    v = rng.standard_normal((n, d))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


class _AuthorPool:
    """Authors of one field: global id, career end year, activity level (grown in place)."""

    def __init__(self):
        self.ids = np.empty(64, dtype=np.int64)
        self.end = np.empty(64, dtype=np.int64)
        self.act = np.empty(64, dtype=np.float64)
        self.n = 0

    def add(self, aid, end, act):
        if self.n == len(self.ids):
            for name in ("ids", "end", "act"):
                arr = getattr(self, name)
                setattr(self, name, np.concatenate([arr, np.empty_like(arr)]))
        self.ids[self.n], self.end[self.n], self.act[self.n] = aid, end, act
        self.n += 1
        return self.n - 1


def _publication_days(rng, n_works, years, growth_rate):
    y0, y1 = years
    year_list = np.arange(y0, y1 + 1)
    w = (1.0 + growth_rate) ** np.arange(len(year_list))
    exact = n_works * w / w.sum()
    counts = np.floor(exact).astype(int)
    short = n_works - counts.sum()
    counts[np.argsort(-(exact - counts), kind="stable")[:short]] += 1  # largest remainder
    parts = []
    for y, c in zip(year_list, counts):
        lo = to_days([f"{y}-01-01"])[0]
        hi = to_days([f"{y + 1}-01-01"])[0]
        parts.append(rng.integers(lo, hi, size=c))
    t = np.concatenate(parts)
    order = np.lexsort((rng.random(n_works), t))
    return t[order]


def _draw_team(rng, pool, year, k, next_id, new_prob, career_mean, activity_sigma):
    """Global author ids of a k-author team; creates new authors as needed."""
    team = []
    for _ in range(k):
        pos = None
        if pool.n > 0 and rng.random() >= new_prob:
            w = pool.act[: pool.n] * (pool.end[: pool.n] >= year)
            if team:
                w[team] = 0.0
            cum = np.cumsum(w)
            if cum[-1] > 0:
                pos = int(np.searchsorted(cum, rng.random() * cum[-1], side="right"))
        if pos is None:
            pos = pool.add(next_id[0], year + int(rng.exponential(career_mean)),
                           rng.lognormal(0.0, activity_sigma))
            next_id[0] += 1
        team.append(pos)
    return [int(pool.ids[p]) for p in team]


def generate_citation_stream(n_works=20000, n_fields=8, years=(2005, 2025), refs_mean=25,
                             emb_dim=64, seed=0, **knobs) -> dict:
    """Generate a synthetic citation stream (see the module docstring for the model).

    Returns a dict:
      works: DataFrame per CONTRACTS.md (work_id, doi_norm=None, title, abstract=None,
             pub_date, concepts=[field], venue_id, is_retracted_oa=False, field),
             rows in publication order.
      events: DataFrame event_id, src_id, dst_id, t (int64 days; t = pub date of src),
              canonical order (t, event_id), event_id = 0..M-1.
      authorships: DataFrame work_id, author_id (team order).
      embeddings: tuple (ids (N,) str, vectors (N, emb_dim) float32), rows aligned with works.
      params: the resolved generator parameters (for traceability, NFR10).
    Unknown knobs raise TypeError.
    """
    unknown = set(knobs) - set(DEFAULT_KNOBS)
    if unknown:
        raise TypeError(f"unknown knob(s): {sorted(unknown)}")
    k = {**DEFAULT_KNOBS, **knobs}
    shares_slots = np.array([k["self_cite_rate"], k["noise_rate"], k["cross_field_rate"]])
    if shares_slots.min() < 0 or shares_slots.sum() > 1:
        raise ValueError("self_cite_rate + noise_rate + cross_field_rate must lie in [0, 1]")
    if n_fields < 1 or n_works < 1:
        raise ValueError("n_works and n_fields must be positive")

    r_dates, r_fields, r_auth, r_refs, r_emb = (
        np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(5)
    )
    N = int(n_works)
    pub_t = _publication_days(r_dates, N, years, k["growth_rate"])
    year = pub_t.astype("datetime64[D]").astype("datetime64[Y]").astype(int) + 1970

    # fields, subtopics, venues
    field_share = np.sort(r_fields.dirichlet(np.full(n_fields, k["field_concentration"])))[::-1]
    field = r_fields.choice(n_fields, size=N, p=field_share)
    sub = r_fields.integers(0, k["n_subtopics"], size=N)
    n_ven = int(k["venues_per_field"])
    ven_p = 1.0 / np.arange(1, n_ven + 1) ** k["venue_zipf"]
    venue = r_fields.choice(n_ven, size=N, p=ven_p / ven_p.sum())

    # reference-list lengths, scaled so the nominal mean is refs_mean
    m_reg = refs_mean / (1.0 - k["review_rate"] + k["review_rate"] * k["review_factor"])
    mu = np.log(m_reg) - k["refs_sigma"] ** 2 / 2
    lengths = r_refs.lognormal(mu, k["refs_sigma"], size=N)
    lengths[r_refs.random(N) < k["review_rate"]] *= k["review_factor"]
    ref_len = np.maximum(1, np.rint(lengths)).astype(int)

    slot_p = np.append(shares_slots, 1.0 - shares_slots.sum())  # self, noise, cross, own
    if n_fields == 1:
        slot_p = np.array([slot_p[0], slot_p[1], 0.0, slot_p[2] + slot_p[3]])
    other_fields = [np.array([g for g in range(n_fields) if g != f]) for f in range(n_fields)]
    other_p = [field_share[o] / field_share[o].sum() if len(o) else o for o in other_fields]

    f_idx = [np.flatnonzero(field == f) for f in range(n_fields)]
    f_t = [pub_t[ix] for ix in f_idx]
    f_sub = [sub[ix] for ix in f_idx]

    indeg = np.zeros(N)
    taken = np.zeros(N, dtype=bool)
    rise, decay = k["ageing_rise_years"], k["ageing_decay_years"]

    def pa_sample(g, n_take, t_i, own_sub):
        if n_take <= 0:
            return np.empty(0, dtype=np.int64)
        n = int(np.searchsorted(f_t[g], t_i, side="left"))  # strictly earlier works
        if n == 0:
            return np.empty(0, dtype=np.int64)
        cand = f_idx[g][:n]
        age = (t_i - f_t[g][:n]) / _DAYS_PER_YEAR
        w = np.exp(-age / decay)
        if rise > 0:
            w *= -np.expm1(-age / rise)
        if k["pa_exponent"] != 0:
            w *= (indeg[cand] + k["pa_offset"]) ** k["pa_exponent"]
        if own_sub is not None and k["subtopic_affinity"] != 1.0:
            w *= np.where(f_sub[g][:n] == own_sub, k["subtopic_affinity"], 1.0)
        w[taken[cand]] = 0.0
        n_take = min(n_take, int(np.count_nonzero(w > 0)))
        if n_take == 0:
            return np.empty(0, dtype=np.int64)
        # Efraimidis-Spirakis: weighted sampling without replacement via exponential keys
        with np.errstate(divide="ignore"):
            keys = r_refs.exponential(size=n) / w
        return cand[np.argpartition(keys, n_take - 1)[:n_take]]

    pools = [_AuthorPool() for _ in range(n_fields)]
    next_author = [0]
    author_works: dict[int, list[int]] = {}
    auth_rows_w, auth_rows_a = [], []
    src_parts, dst_parts = [], []

    for i in range(N):
        f, t_i = int(field[i]), int(pub_t[i])
        team_size = 1 + int(r_auth.poisson(k["authors_mean"] - 1.0))
        team = _draw_team(r_auth, pools[f], int(year[i]), team_size, next_author,
                          k["new_author_prob"], k["career_mean_years"],
                          k["author_activity_sigma"])
        auth_rows_w.extend([i] * len(team))
        auth_rows_a.extend(team)

        n_prev = int(np.searchsorted(pub_t, t_i, side="left"))
        r = min(int(ref_len[i]), n_prev)
        if r > 0:
            k_self, k_noise, k_cross, k_own = (int(x) for x in r_refs.multinomial(r, slot_p))
            chosen: list[int] = []
            if k_self:
                own = sorted({j for a in team for j in author_works.get(a, ()) if pub_t[j] < t_i})
                take = min(k_self, len(own))
                if take:
                    chosen.extend(int(j) for j in r_refs.choice(own, size=take, replace=False))
                k_own += k_self - take
            if k_noise:
                picks = r_refs.choice(n_prev, size=min(k_noise, n_prev), replace=False)
                seen = set(chosen)
                fresh = [int(j) for j in picks if j not in seen]
                chosen.extend(fresh)
                k_own += k_noise - len(fresh)
            taken[chosen] = True
            if k_cross:
                targets = r_refs.choice(other_fields[f], size=k_cross, p=other_p[f])
                for g in np.unique(targets):
                    want = int((targets == g).sum())
                    got = pa_sample(int(g), want, t_i, None)
                    taken[got] = True
                    chosen.extend(got.tolist())
                    k_own += want - len(got)
            got = pa_sample(f, k_own, t_i, int(sub[i]))
            chosen.extend(got.tolist())
            refs = np.asarray(chosen, dtype=np.int64)
            taken[refs] = False
            indeg[refs] += 1.0
            src_parts.append(np.full(len(refs), i, dtype=np.int64))
            dst_parts.append(refs)
        for a in team:
            author_works.setdefault(a, []).append(i)

    # embeddings
    span = max(int(pub_t[-1] - pub_t[0]), 1)
    frac = (pub_t - pub_t[0]) / span
    g_off = _unit(r_emb, 1, emb_dim)[0] * k["emb_global"]
    centroid = _unit(r_emb, n_fields, emb_dim) * k["emb_field"]
    drift = _unit(r_emb, n_fields, emb_dim) * k["emb_drift"]
    sub_off = _unit(r_emb, n_fields * int(k["n_subtopics"]), emb_dim) * k["emb_subtopic"]
    noise = r_emb.standard_normal((N, emb_dim)) * (k["emb_noise"] / np.sqrt(emb_dim))
    vectors = (g_off + centroid[field] + frac[:, None] * drift[field]
               + sub_off[field * int(k["n_subtopics"]) + sub] + noise).astype(np.float32)

    # frames
    width = max(6, len(str(N - 1)))
    work_ids = np.array([f"SW{i:0{width}d}" for i in range(N)])
    field_names = np.array([f"F{f:02d}" for f in range(n_fields)])
    fname = field_names[field]
    works = _canonical_works(pd.DataFrame({
        "work_id": work_ids,
        "doi_norm": None,
        "title": [f"Synthetic work {w}" for w in work_ids],
        "abstract": None,
        "pub_date": _to_iso(pub_t),
        "concepts": [[x] for x in fname],
        "venue_id": [f"SV{f:02d}-{v:02d}" for f, v in zip(field, venue)],
        "is_retracted_oa": False,
        "field": fname,
    }))

    src = np.concatenate(src_parts) if src_parts else np.empty(0, dtype=np.int64)
    dst = np.concatenate(dst_parts) if dst_parts else np.empty(0, dtype=np.int64)
    order = np.lexsort((dst, src))  # src index is already in time order
    src, dst = src[order], dst[order]
    events = _canonical_events(pd.DataFrame({
        "event_id": np.arange(len(src), dtype=np.int64),
        "src_id": work_ids[src],
        "dst_id": work_ids[dst],
        "t": pub_t[src],
    }))

    a_width = max(6, len(str(next_author[0] - 1)))
    authorships = _canonical_authorships(pd.DataFrame({
        "work_id": work_ids[np.asarray(auth_rows_w, dtype=np.int64)],
        "author_id": [f"SA{a:0{a_width}d}" for a in auth_rows_a],
    }))

    params = {"n_works": N, "n_fields": int(n_fields), "years": [int(years[0]), int(years[1])],
              "refs_mean": refs_mean, "emb_dim": int(emb_dim), "seed": int(seed), **k}
    return {"works": works, "events": events, "authorships": authorships,
            "embeddings": (work_ids.astype(str), vectors), "params": params}


# --- canonical dtypes (shared by the generator and the fixture loader) ----------------

def _str_col(s):
    return s.astype(str)


def _none_col(s):
    return pd.Series([None if pd.isna(x) else x for x in s], index=s.index, dtype=object)


def _canonical_works(df):
    out = pd.DataFrame({
        "work_id": _str_col(df["work_id"]),
        "doi_norm": _none_col(df["doi_norm"]),
        "title": _str_col(df["title"]),
        "abstract": _none_col(df["abstract"]),
        "pub_date": _str_col(df["pub_date"]),
        "concepts": pd.Series([list(c) for c in df["concepts"]], index=df.index, dtype=object),
        "venue_id": _str_col(df["venue_id"]),
        "is_retracted_oa": df["is_retracted_oa"].astype(bool),
        "field": _str_col(df["field"]),
    })
    return out.reset_index(drop=True)


def _canonical_events(df):
    out = pd.DataFrame({
        "event_id": df["event_id"].astype(np.int64),
        "src_id": _str_col(df["src_id"]),
        "dst_id": _str_col(df["dst_id"]),
        "t": df["t"].astype(np.int64),
    })
    return out.reset_index(drop=True)


def _canonical_authorships(df):
    return pd.DataFrame({"work_id": _str_col(df["work_id"]),
                         "author_id": _str_col(df["author_id"])}).reset_index(drop=True)


# --- profile ---------------------------------------------------------------------------

def _r(x, nd=4):
    return round(float(x), nd)


def _summary(x):
    x = np.asarray(x, dtype=float)
    return {"mean": _r(x.mean()), "median": _r(np.median(x)), "p90": _r(np.quantile(x, 0.9)),
            "p99": _r(np.quantile(x, 0.99)), "max": _r(x.max())}


def _gini(x):
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1) @ x / (n * x.sum()))


def _cosine_rows(vec, i, j):
    a, b = vec[i].astype(np.float64), vec[j].astype(np.float64)
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))


def profile(stream, n_pairs=20000) -> dict:
    """Shape summary of a stream (synthetic or real): degrees, field mix, reciprocity,
    self-citation, citation age, growth, authorship and embedding similarity.

    reciprocity_paper is the share of events u->v whose reverse v->u also exists (0 by
    construction when references point strictly backwards); reciprocity_author is the
    same share over distinct directed author pairs (a cites b, a != b) induced by events.
    Embedding cosines use n_pairs random work pairs drawn with a fixed seed.
    """
    works, ev, auth = stream["works"], stream["events"], stream.get("authorships")
    ids, vec = stream["embeddings"]
    N, M = len(works), len(ev)
    index = pd.Index(works["work_id"])
    s = index.get_indexer(ev["src_id"])
    d = index.get_indexer(ev["dst_id"])
    pub_t = to_days(works["pub_date"].to_numpy())
    field = works["field"].to_numpy() if "field" in works else np.zeros(N, dtype=int)
    out_deg = np.bincount(s, minlength=N)
    in_deg = np.bincount(d, minlength=N)

    top = max(1, N // 100)
    in_sorted = np.sort(in_deg)[::-1]
    key = s.astype(np.int64) * N + d
    rev = d.astype(np.int64) * N + s
    years = pd.Series(pub_t.astype("datetime64[D]").astype("datetime64[Y]").astype(int) + 1970)
    per_year = years.value_counts().sort_index()
    age = (ev["t"].to_numpy() - pub_t[d]) / _DAYS_PER_YEAR

    out = {
        "synthetic": bool(stream.get("params") is not None),
        "n_works": int(N), "n_events": int(M),
        "refs_per_paper": {**_summary(out_deg), "share_zero": _r((out_deg == 0).mean()),
                           "mean_if_citing": _r(out_deg[out_deg > 0].mean() if M else 0.0)},
        "in_degree": {**_summary(in_deg), "share_uncited": _r((in_deg == 0).mean()),
                      "gini": _r(_gini(in_deg)),
                      "top1pct_share": _r(in_sorted[:top].sum() / max(M, 1))},
        "field_mix": {
            "works_share": {str(f): _r(c / N) for f, c in
                            pd.Series(field).value_counts().sort_index().items()},
            "within_field_event_share": _r((field[s] == field[d]).mean() if M else 0.0),
        },
        "reciprocity_paper": _r(np.isin(rev, key).mean() if M else 0.0),
        "citation_age_years": {"median": _r(np.median(age)), "p10": _r(np.quantile(age, 0.1)),
                               "p90": _r(np.quantile(age, 0.9)), "mean": _r(age.mean())},
        "works_per_year": {str(y): int(c) for y, c in per_year.items()},
        "growth_cagr": _r((per_year.iloc[-1] / per_year.iloc[0]) ** (1 / max(len(per_year) - 1, 1))
                          - 1),
    }

    if auth is not None and len(auth):
        a_codes, a_uniques = pd.factorize(auth["author_id"])
        aw = pd.DataFrame({"w": index.get_indexer(auth["work_id"]), "a": a_codes})
        pairs = (pd.DataFrame({"e": np.arange(M), "w": s}).merge(aw, on="w")
                 .rename(columns={"a": "a_s"})[["e", "a_s"]])
        pairs = pairs.merge(pd.DataFrame({"e": np.arange(M), "w": d}).merge(aw, on="w")
                            .rename(columns={"a": "a_d"})[["e", "a_d"]], on="e")
        shared = pairs.loc[pairs["a_s"] == pairs["a_d"], "e"].nunique()
        cross = pairs[pairs["a_s"] != pairs["a_d"]]
        A = len(a_uniques)
        akey = np.unique(cross["a_s"].to_numpy(np.int64) * A + cross["a_d"].to_numpy(np.int64))
        arev = (akey % A) * A + akey // A
        team = auth.groupby("work_id").size()
        papers = auth.groupby("author_id").size()
        out["self_citation_rate"] = _r(shared / max(M, 1))
        out["reciprocity_author"] = _r(np.isin(arev, akey).mean() if len(akey) else 0.0)
        out["authors"] = {"n_authors": int(A), "team_size_mean": _r(team.mean()),
                          "papers_per_author": _summary(papers)}
    if "venue_id" in works:
        out["n_venues"] = int(works["venue_id"].nunique())

    rng = np.random.default_rng(0)
    i, j = rng.integers(0, N, size=(2, n_pairs))
    keep = i != j
    i, j = i[keep], j[keep]
    cos = _cosine_rows(vec, i, j)
    same = field[i] == field[j]
    e_s = rng.choice(M, size=min(M, n_pairs), replace=False) if M else np.empty(0, int)
    cited = _cosine_rows(vec, s[e_s], d[e_s]) if M else np.zeros(1)
    cited_same = field[s[e_s]] == field[d[e_s]] if M else np.zeros(1, dtype=bool)
    out["embedding_cosine"] = {
        "intra_field": _r(cos[same].mean() if same.any() else np.nan),
        "inter_field": _r(cos[~same].mean() if (~same).any() else np.nan),
        "cited_pairs": _r(cited.mean()),
        "cited_pairs_same_field": _r(cited[cited_same].mean() if cited_same.any() else np.nan),
    }
    if stream.get("params") is not None:
        out["params"] = stream["params"]
    return out


# --- integration fixture (plan 9.4) ------------------------------------------------------

def make_mini_fixture() -> dict:
    """The ~200-event integration fixture: a small two-field stream plus one planted cartel.

    All cartel members (and therefore every positive event) are published on or after
    meta["t_min"], so an integration test can use t_min as the split cutoff and assert that
    every injected event is post-cutoff. Returned keys: works, events (with the cartel),
    authorships, embeddings, event_labels (event_id, label), cartels, meta.
    """
    from citecurv.eval.inject import inject_cartel

    gen = dict(MINI_GENERATOR)
    gen["years"] = tuple(gen["years"])
    stream = generate_citation_stream(**gen)
    t_min = int(to_days([MINI_T_MIN_DATE])[0])
    inj = dict(MINI_INJECTION)
    inj["size_range"] = tuple(inj["size_range"])
    res = inject_cartel(stream["events"], stream["works"], t_min=t_min,
                        authorships=stream["authorships"], **inj)
    meta = {"synthetic": True, "generator": MINI_GENERATOR, "injection": MINI_INJECTION,
            "t_min": t_min, "t_min_date": MINI_T_MIN_DATE,
            "n_events": int(len(res.events)), "n_positive": int(len(res.truth_event_ids))}
    return {"works": stream["works"], "events": res.events,
            "authorships": stream["authorships"], "embeddings": stream["embeddings"],
            "event_labels": res.event_labels(), "cartels": res.cartels, "meta": meta}


def write_fixture(fx, out_dir) -> None:
    """Write a fixture dict as small CSV files + meta.json (lists joined with ';')."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    opts = {"index": False, "lineterminator": "\n"}
    works = fx["works"].copy()
    works["concepts"] = works["concepts"].map(";".join)
    works.to_csv(out / "works.csv", **opts)
    fx["events"].to_csv(out / "events.csv", **opts)
    fx["authorships"].to_csv(out / "authorships.csv", **opts)
    ids, vec = fx["embeddings"]
    emb = pd.DataFrame(vec.astype(np.float64), columns=[f"e{c:02d}" for c in range(vec.shape[1])])
    emb.insert(0, "work_id", ids)
    emb.to_csv(out / "embeddings.csv", float_format="%.9g", **opts)
    fx["event_labels"].to_csv(out / "event_labels.csv", **opts)
    fx["cartels"].to_csv(out / "cartels.csv", **opts)
    (out / "meta.json").write_text(json.dumps(fx["meta"], indent=2, sort_keys=True) + "\n",
                                   encoding="utf-8")


def load_fixture(in_dir=MINI_FIXTURE_DIR) -> dict:
    """Inverse of write_fixture, with the generator's canonical dtypes."""
    from citecurv.eval.inject import _canonical_cartels

    d = Path(in_dir)
    works = pd.read_csv(d / "works.csv", keep_default_na=False, na_values={"doi_norm": [""],
                                                                           "abstract": [""]})
    works["concepts"] = [c.split(";") if c else [] for c in works["concepts"]]
    emb = pd.read_csv(d / "embeddings.csv")
    labels = pd.read_csv(d / "event_labels.csv").astype(np.int64)
    return {
        "works": _canonical_works(works),
        "events": _canonical_events(pd.read_csv(d / "events.csv")),
        "authorships": _canonical_authorships(pd.read_csv(d / "authorships.csv")),
        "embeddings": (emb["work_id"].to_numpy().astype(str),
                       emb.drop(columns="work_id").to_numpy().astype(np.float32)),
        "event_labels": labels,
        "cartels": _canonical_cartels(pd.read_csv(d / "cartels.csv")),
        "meta": json.loads((d / "meta.json").read_text(encoding="utf-8")),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="Synthetic citation stream utilities (synthetic data).")
    ap.add_argument("--profile", metavar="JSON", help="profile the default stream into JSON")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--write-mini-fixture", nargs="?", const=str(MINI_FIXTURE_DIR), metavar="DIR",
                    help=f"write the integration fixture (default {MINI_FIXTURE_DIR})")
    args = ap.parse_args(argv)
    if args.write_mini_fixture:
        write_fixture(make_mini_fixture(), args.write_mini_fixture)
        print(f"wrote {args.write_mini_fixture}")
    if args.profile:
        stream = generate_citation_stream(seed=args.seed)
        prof = profile(stream)
        prof["injection_check"] = _injection_check(stream, seed=args.seed)
        Path(args.profile).write_text(json.dumps(prof, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({k: prof[k] for k in ("n_works", "n_events", "self_citation_rate")}))


def _auroc(score, label):
    ranks = pd.Series(score).rank(method="average").to_numpy()
    pos = label.astype(bool)
    n_pos, n_neg = pos.sum(), (~pos).sum()
    return (ranks[pos].mean() - (n_pos + 1) / 2) / n_neg


def _average_precision(score, label):
    """Step-wise AP over distinct thresholds (tied scores enter together)."""
    order = np.argsort(-np.asarray(score, dtype=float), kind="stable")
    s, y = np.asarray(score, dtype=float)[order], label.astype(bool)[order]
    last = np.r_[np.flatnonzero(np.diff(s) != 0), len(s) - 1]  # end of each tie group
    tp = np.cumsum(y)[last]
    precision = tp / (last + 1)
    recall_step = np.diff(np.r_[0, tp]) / max(y.sum(), 1)
    return float((recall_step * precision).sum())


def _injection_check(stream, seed=0, n_cartels=10, t_min_date="2021-01-01"):
    """Per obviousness preset: injected volume, realized density, and the AUROC of single
    trivial features on post-t_min events (higher = more anomalous). A value near 1 would
    mean the planted cartels are trivially separable. Synthetic numbers only."""
    from citecurv.eval.inject import OBVIOUSNESS, inject_cartel

    t_min = int(to_days([t_min_date])[0])
    ids, vec = stream["embeddings"]
    row = {w: r for r, w in enumerate(ids)}
    pub = dict(zip(stream["works"]["work_id"], to_days(stream["works"]["pub_date"].to_numpy())))
    field = dict(zip(stream["works"]["work_id"], stream["works"]["field"]))
    out = {"note": "synthetic; single-feature AUROC / AP on events with t >= t_min",
           "t_min_date": t_min_date, "n_cartels": n_cartels}
    for name, preset in OBVIOUSNESS.items():
        res = inject_cartel(stream["events"], stream["works"], n_cartels=n_cartels, seed=seed,
                            t_min=t_min, authorships=stream["authorships"], **preset)
        ev = res.events[res.events["t"] >= t_min]
        label = ev["event_id"].isin(res.truth_event_ids).to_numpy()
        s = ev["src_id"].map(row).to_numpy()
        d = ev["dst_id"].map(row).to_numpy()
        recency = -(ev["t"].to_numpy() - ev["dst_id"].map(pub).to_numpy())
        dissim = 1 - _cosine_rows(vec, s, d)
        same_field = (ev["src_id"].map(field) == ev["dst_id"].map(field)).to_numpy().astype(float)
        feasible = 0
        for _, grp in res.cartels.groupby("cartel_id"):
            t = np.array([pub[w] for w in grp["work_id"]])
            feasible += int((t[:, None] > t[None, :]).sum())
        out[name] = {
            "preset": {k: list(v) if isinstance(v, tuple) else v for k, v in preset.items()},
            "n_injected": int(len(res.truth_event_ids)),
            "positive_share_post_t_min": _r(label.mean()),
            "realized_density": _r(len(res.truth_event_ids) / max(feasible, 1)),
            "auroc_recency": _r(_auroc(recency, label)),
            "ap_recency": _r(_average_precision(recency, label)),
            "auroc_embed_dissimilarity": _r(_auroc(dissim, label)),
            "ap_embed_dissimilarity": _r(_average_precision(dissim, label)),
            "auroc_same_field": _r(_auroc(same_field, label)),
        }
    return out


if __name__ == "__main__":
    main()
