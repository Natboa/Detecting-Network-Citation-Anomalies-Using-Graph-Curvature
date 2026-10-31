"""Synthetic demo inputs for the explorer: contract-conforming (CONTRACTS.md) but entirely simulated.

Use them to develop and preview the page, never as results. The citation stream is a seeded preferential-attachment
process with field homophily; cartels follow the CIDRE donor/recipient pattern at three obviousness levels; the
Forman screen values and cold-start flags are computed from that stream with the contract formulas, but the four
component values are SIMULATED noise plus a level-dependent shift on injected events. They do not implement the S3.1
definitions. Metrics and tables are computed from these simulated scores, so they describe the simulation only.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from citecurv.viz.graphctx import TemporalIndex
from citecurv.viz.precompute import COMPONENTS, num, precision_at, rank_events, ranking_curves, recall_by_level

FIELDS = ("Biology", "Chemistry", "Computer science", "Medicine", "Physics", "Psychology")
TOPICS = ("graph", "protein", "network", "cohort", "lattice", "memory", "catalyst", "sensor", "survey", "trial")
# (level, internal density of donor -> recipient edges, component shift on injected events)
LEVELS = (("blatant", 0.8, 3.0), ("moderate", 0.5, 1.6), ("subtle", 0.25, 0.7))
WEIGHTS = {"s_embed": 0.35, "s_geom": 0.2, "s_temp": 0.3, "s_curvdyn": 0.15}
COLD_START_K = 3


def make_demo_data(seed: int = 0, n_papers: int = 1500, refs_mean: float = 8.0, year0: int = 2010,
                   n_years: int = 12, cartels_per_level: int = 2) -> dict:
    """Keyword arguments for `build_explorer` (everything except out_html), all synthetic."""
    rng = np.random.default_rng(seed)
    epoch = pd.Timestamp("1970-01-01")
    d0 = (pd.Timestamp(f"{year0}-01-01") - epoch).days
    d1 = (pd.Timestamp(f"{year0 + n_years}-01-01") - epoch).days

    # ---- papers
    ids = np.array([f"SYN{i:05d}" for i in range(n_papers)], dtype=object)
    pub = np.sort(rng.integers(d0, d1, size=n_papers))
    field_idx = rng.choice(len(FIELDS), size=n_papers, p=[0.24, 0.16, 0.2, 0.2, 0.12, 0.08])
    works = pd.DataFrame({
        "work_id": ids,
        "doi_norm": [None] * n_papers,
        "title": [f"Synthetic paper {i:05d}: {FIELDS[f].lower()} {TOPICS[rng.integers(len(TOPICS))]} study"
                  for i, f in enumerate(field_idx)],
        "abstract": [None] * n_papers,
        "pub_date": [(epoch + pd.Timedelta(days=int(p))).strftime("%Y-%m-%d") for p in pub],
        "concepts": [[FIELDS[f]] for f in field_idx],
        "venue_id": [f"SYN-V{f}{rng.integers(3)}" for f in field_idx],
        "is_retracted_oa": [False] * n_papers,
        "field": [FIELDS[f] for f in field_idx],
    })

    # ---- authors (field-local pools)
    n_auth = max(10, n_papers // 3)
    auth_field = rng.integers(len(FIELDS), size=n_auth)
    pools = [np.flatnonzero(auth_field == f) for f in range(len(FIELDS))]
    au_rows = []
    for i, f in enumerate(field_idx):
        pool = pools[f] if len(pools[f]) else np.arange(n_auth)
        for a in rng.choice(pool, size=min(len(pool), int(rng.integers(1, 5))), replace=False):
            au_rows.append((ids[i], f"SYN-A{int(a):04d}"))
    authorships = pd.DataFrame(au_rows, columns=["work_id", "author_id"])

    # ---- organic citations: preferential attachment, 3x weight inside the field
    in_count = np.zeros(n_papers)
    rows = []
    for i in range(20, n_papers):
        earlier = np.flatnonzero(pub[:i] < pub[i])
        if len(earlier) == 0:
            continue
        w = (in_count[earlier] + 1.0) * np.where(field_idx[earlier] == field_idx[i], 3.0, 1.0)
        m = min(len(earlier), max(1, int(rng.poisson(refs_mean))))
        for j in rng.choice(earlier, size=m, replace=False, p=w / w.sum()):
            rows.append((ids[i], ids[j], int(pub[i]), "organic", None))
            in_count[j] += 1

    # ---- cartels (CIDRE donor/recipient pattern) in one field: new donor papers inside the last three years cite
    # established recipients (published earlier, already cited), so injected events are mostly not cold-start
    window_start = d1 - 3 * 365
    members, cartel_no = [], 0
    used: set = set()
    for level, density, _ in LEVELS:
        for _ in range(cartels_per_level):
            f = int(rng.integers(len(FIELDS)))
            free = np.array([i not in used for i in range(n_papers)])
            rec_pool = np.flatnonzero((field_idx == f) & free & (pub < window_start) & (pub >= window_start - 4 * 365)
                                      & (in_count >= COLD_START_K))
            don_pool = np.flatnonzero((field_idx == f) & free & (pub >= window_start))
            if len(rec_pool) < 2 or len(don_pool) < 4:
                continue
            size = int(rng.integers(8, 15))
            n_rec = int(min(len(rec_pool), max(2, round(0.4 * size))))
            n_don = int(min(len(don_pool), size - n_rec))
            recipients = sorted(rng.choice(rec_pool, size=n_rec, replace=False), key=lambda k: pub[k])
            donors = sorted(rng.choice(don_pool, size=n_don, replace=False), key=lambda k: pub[k])
            used.update(recipients)
            used.update(donors)
            cid = f"C{cartel_no:02d}"
            cartel_no += 1
            members += [(cid, ids[k], "recipient", level) for k in recipients]
            members += [(cid, ids[k], "donor", level) for k in donors]
            for dn in donors:
                for rc in recipients:
                    if pub[rc] < pub[dn] and rng.random() < density:
                        rows.append((ids[dn], ids[rc], int(pub[dn]), "injected", level))

    ev = pd.DataFrame(rows, columns=["src_id", "dst_id", "t", "kind", "level"])
    ev = ev.drop_duplicates(["src_id", "dst_id"], keep="last").reset_index(drop=True)
    ev["event_id"] = (rng.permutation(len(ev)) + 100_000).astype(np.int64)  # stable keys, not positions
    ev = ev.sort_values(["t", "event_id"], ignore_index=True)
    events = ev[["event_id", "src_id", "dst_id", "t"]].copy()

    # ---- contract quantities from the stream: Forman screen at t, cold start (k prior citations of dst)
    idx = TemporalIndex(events)
    forman, cold = np.empty(len(ev)), np.empty(len(ev), dtype=bool)
    for n, r in enumerate(events.itertuples(index=False)):
        in_src, _ = idx.degree_upto(idx.code[r.src_id], r.t, exclude_eid=r.event_id)
        _, out_dst = idx.degree_upto(idx.code[r.dst_id], r.t, exclude_eid=r.event_id)
        forman[n] = 2 - in_src - out_dst
        prior = idx.in_upto(idx.code[r.dst_id], r.t - 1)[1]
        cold[n] = len(prior) < COLD_START_K

    # ---- simulated components (NOT the S3.1 definitions)
    shift = ev["level"].map({lv: s for lv, _, s in LEVELS}).fillna(0.0).to_numpy()
    jit = lambda: rng.uniform(0.6, 1.4, size=len(ev))  # noqa: E731
    comp = pd.DataFrame({
        "s_embed": rng.gamma(2.0, 0.8, size=len(ev)) + shift * jit(),
        "s_geom": np.abs(rng.normal(size=len(ev))) + 0.6 * shift * jit(),
        "s_temp": rng.gamma(1.5, 0.7, size=len(ev)) + 0.9 * shift * jit(),
        "s_curvdyn": 0.8 * np.abs(rng.normal(size=len(ev))) + np.abs(forman) / 40 + 0.4 * shift * jit(),
    })
    train = ev["t"] < np.quantile(ev["t"], 0.7)
    zstats = {c: {"mean": float(comp.loc[train, c].mean()), "std": float(comp.loc[train, c].std(ddof=0))}
              for c in COMPONENTS}
    total = sum(WEIGHTS[c] * (comp[c] - zstats[c]["mean"]) / zstats[c]["std"] for c in COMPONENTS)
    cfg_hash = hashlib.sha1(f"demo|{seed}|{n_papers}|{refs_mean}".encode()).hexdigest()[:12]

    base = events.copy()
    scores = base.assign(**{c: comp[c].to_numpy() for c in COMPONENTS}, total=total.to_numpy(), cold_start=cold,
                         model_variant="demo-full", seed=seed, config_hash=cfg_hash)
    components = base.assign(**{c: comp[c].to_numpy() for c in COMPONENTS}, cold_start=cold,
                             p_link=np.exp(-comp["s_embed"].to_numpy()))
    esc = forman <= np.quantile(forman, 0.03)
    orc = np.where(esc, np.clip(np.tanh(forman / 25.0) + rng.normal(0, 0.1, size=len(ev)), -1, 1), np.nan)
    curvature = base.assign(forman=forman, orc=orc, escalated=esc, radius=np.where(esc, 1.0, np.nan),
                            computed_at_stage="demo")
    labels = pd.DataFrame({"event_id": ev["event_id"], "label": (ev["kind"] == "injected").astype(int)})

    # ---- metrics and tables computed from the simulated scores (they describe the simulation only)
    ranked = rank_events(scores).merge(labels, on="event_id")
    sc = ranked[ranked["scorable"]]
    tot = ranking_curves(sc["total"], sc["label"])
    pa = precision_at(sc["label"], (100, 1000))
    metrics = {"n": tot["n"], "n_pos": tot["n_pos"], "auroc": num(tot["auroc"]), "ap": num(tot["ap"])}
    for k, v in pa.items():
        metrics[f"p_at_{k}"], metrics[f"found_at_{k}"] = v["p"], v["found"]
    inj = ranked[ranked["label"] == 1].merge(ev[["event_id", "level"]], on="event_id")
    metrics["recall_by_level"] = recall_by_level(inj["level"], inj["rank"], (100, 1000))

    comp_rows = []
    for name in ("total", *COMPONENTS):
        c = ranking_curves(sc[name], sc["label"])
        order = sc.assign(_k=-sc[name]).sort_values(["_k", "t", "event_id"], kind="mergesort")
        p = precision_at(order["label"], (100,))
        comp_rows.append({"scorer": name, "AUROC": num(c["auroc"], 4), "AP": num(c["ap"], 4),
                          "P@100": p.get("100", {}).get("p")})
    lvl = inj.groupby("level").agg(injected=("event_id", "size"), cold_start=("cold_start", "sum")).reset_index()
    tables = {
        "Single components vs combined total (synthetic demo)": pd.DataFrame(comp_rows),
        "Injected events per obviousness level (synthetic demo)": lvl,
    }
    cartels = {"members": pd.DataFrame(members, columns=["cartel_id", "work_id", "role", "level"]),
               "event_ids": ev.loc[ev["kind"] == "injected", "event_id"].tolist()}
    return {
        "scores": scores, "components": components, "curvature": curvature, "events": events, "works": works,
        "authorships": authorships, "labels": labels, "metrics": metrics, "tables": tables, "cartels": cartels,
        "weights": dict(WEIGHTS), "zstats": zstats,
    }
