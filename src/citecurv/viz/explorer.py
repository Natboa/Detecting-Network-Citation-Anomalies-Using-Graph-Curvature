"""Results explorer: one self-contained, offline HTML page over the pipeline outputs (FR7-FR9, NFR7, NFR10, D4).

`build_explorer` reads the in-memory artifacts of CONTRACTS.md, precomputes every number the page shows (rank order,
z-scores and contributions, reference histograms, PR/ROC curves, the D4 flag-share heatmap, rollups, per-event
subgraphs with seeded layouts) and writes one HTML file holding a JSON blob plus inline CSS and vanilla JavaScript
that only renders. The page makes no network request, opens from disk in any browser, and shows inside a notebook
cell through `iframe_html` / `show`.

Leak argument: the page displays finished scores and never feeds anything back into scoring. The graph context of an
event (subgraph, degrees for the edge colours, timeline) reads only events with ts <= t (graphctx.py).

Determinism (NFR4): the output is a pure function of the inputs and `seed` (spring layouts are seeded, JSON keys are
sorted, floats are rounded to fixed significant digits, no timestamps are written), so equal inputs give equal bytes.
"""

from __future__ import annotations

import hashlib
import html as html_lib
import json
import math
from bisect import bisect_right
from collections.abc import Mapping
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from citecurv.viz.component_text import COMBINER_TEXT, COMPONENT_TEXT, STATUS
from citecurv.viz.graphctx import TemporalIndex, local_subgraph, received_timeline
from citecurv.viz.precompute import (
    COMPONENTS,
    author_rollup,
    coerce_cartels,
    coerce_weights,
    coerce_zstats,
    fallback_zstats,
    flag_count,
    flag_heatmap,
    histogram,
    incident_long,
    injected_events,
    num,
    paper_rollup,
    percentile_of,
    precision_at,
    rank_events,
    ranking_curves,
    recall_by_level,
    zscores,
)

ASSET_DIR = Path(__file__).resolve().parent / "assets"
REQUIRED = ("event_id", "src_id", "dst_id", "t", "total")
TITLE_MAX = 160
MISSING_TEXT = {
    "labels": "No labels given: PR and ROC curves, precision@k, label markers and the positives filter are hidden.",
    "events": "No event stream given: the local subgraph at time t and the citation timeline are hidden.",
    "works": "No works table given: titles, fields and venues are unavailable, so the D4 field x year heatmap and "
             "the field and venue filters are hidden.",
    "authorships": "No authorships given: the author rollup is hidden (FR8-author depends on the S4.0 pull).",
    "cartels": "No cartel table given: this view exists only for the synthetic injection track (S4.5).",
    "curvature": "No curvature cache given: the scored edge shows the Forman screen recomputed from degrees at t, "
                 "and ORC is unavailable.",
    "tables": "No comparison or ablation tables given.",
    "metrics": "No run metrics given.",
    "components": "No components table given: raw component values come from the scores file.",
}


# ---------------------------------------------------------------- small helpers


def _jsonable(v, sig: int = 8):
    if v is None or v is pd.NA:
        return None
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return num(v, sig)
    if isinstance(v, (list, tuple, np.ndarray)):
        return [_jsonable(x, sig) for x in list(v)[:20]]
    s = str(v)
    return s if len(s) <= 200 else s[:199] + "…"


def _first_concept(c):
    if c is None or (isinstance(c, float) and math.isnan(c)):
        return None
    if isinstance(c, str):
        return c.split(";")[0].split("|")[0].strip() or None
    try:
        items = list(c)
    except TypeError:
        return None
    return str(items[0]) if items else None


def _date(t) -> str:
    return (pd.Timestamp("1970-01-01") + pd.Timedelta(days=int(t))).strftime("%Y-%m-%d")


class _Nodes:
    """Paper table shared by every view: index -> id, title, field, venue."""

    def __init__(self, info: dict):
        self.info = info
        self.index: dict[str, int] = {}
        self.fields: list[str] = []
        self.venues: list[str] = []
        self._fi: dict[str, int] = {}
        self._vi: dict[str, int] = {}
        self.cols = {"id": [], "title": [], "field": [], "venue": [], "date": []}

    def _enum(self, value, table, lookup):
        if value is None:
            return None
        if value not in lookup:
            lookup[value] = len(table)
            table.append(value)
        return lookup[value]

    def __call__(self, pid) -> int:
        pid = str(pid)
        i = self.index.get(pid)
        if i is None:
            i = self.index[pid] = len(self.cols["id"])
            title, field, venue, date = self.info.get(pid, (None, None, None, None))
            self.cols["id"].append(pid)
            self.cols["title"].append(title)
            self.cols["field"].append(self._enum(field, self.fields, self._fi))
            self.cols["venue"].append(self._enum(venue, self.venues, self._vi))
            self.cols["date"].append(date)
        return i


def _works_info(works: pd.DataFrame | None) -> dict:
    if works is None:
        return {}
    w = works
    has_field = "field" in w.columns
    info = {}
    for r in w.itertuples(index=False):
        title = getattr(r, "title", None)
        if title is not None and not (isinstance(title, float) and math.isnan(title)):
            title = str(title)
            title = title if len(title) <= TITLE_MAX else title[: TITLE_MAX - 1] + "…"
        else:
            title = None
        field = getattr(r, "field", None) if has_field else None
        if field is None or (isinstance(field, float) and math.isnan(field)):
            field = _first_concept(getattr(r, "concepts", None))
        venue = getattr(r, "venue_id", None)
        venue = None if venue is None or (isinstance(venue, float) and math.isnan(venue)) else str(venue)
        date = getattr(r, "pub_date", None)
        date = None if date is None or (isinstance(date, float) and math.isnan(date)) else str(date)[:10]
        info[str(r.work_id)] = (title, None if field is None else str(field), venue, date)
    return info


def _layout(n_nodes: int, edges, seed: int) -> list[tuple[float, float]]:
    if n_nodes == 0:
        return []
    if n_nodes == 1:
        return [(0.5, 0.5)]
    g = nx.Graph()
    g.add_nodes_from(range(n_nodes))
    g.add_edges_from((a, b) for a, b in edges if a != b)
    pos = nx.spring_layout(g, seed=seed, iterations=80)
    xy = np.array([pos[i] for i in range(n_nodes)], dtype=float)
    xy -= xy.min(axis=0)
    span = float(xy.max()) or 1.0
    xy = xy / span
    xy += (1.0 - xy.max(axis=0)) / 2.0  # centre the shorter axis
    return [(round(float(x), 3), round(float(y), 3)) for x, y in xy]


def _curv_breaks(abs_forman: list) -> dict:
    """Class breaks [neutral, b1, b2] on |curvature| for the diverging edge colours.

    Forman: |F| <= 2 is neutral (F = 2 - in - out is at most 2), b1/b2 = median and 90th percentile of the larger
    |F| over all drawn edges, so the classes spread over the edges actually shown. ORC lies in [-2, 1]: fixed breaks.
    """
    big = np.array([a for a in abs_forman if a > 2], dtype=float)
    b1 = float(np.median(big)) if len(big) else 4.0
    b2 = float(np.percentile(big, 90)) if len(big) else 8.0
    b1 = max(b1, 3.0)
    b2 = max(b2, b1 + 1.0)
    return {"forman": [2.0, num(b1, 4), num(b2, 4)], "orc": [0.05, 0.25, 0.5]}


def _flatten(d, prefix=""):
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, Mapping):
            yield from _flatten(v, key + ".")
        else:
            yield key, v


def _table_payload(name: str, df: pd.DataFrame) -> dict:
    d = df.copy()
    if not isinstance(d.index, pd.RangeIndex) or d.index.name is not None:
        d = d.reset_index()
    cols = [str(c) for c in d.columns]
    rows = [[_jsonable(v, 6) for v in r] for r in d.itertuples(index=False, name=None)]
    return {"name": str(name), "columns": cols, "rows": rows}


def _single_run(scores: pd.DataFrame) -> dict:
    run = {}
    for c in ("model_variant", "seed", "config_hash"):
        if c in scores.columns:
            vals = scores[c].dropna().unique()
            if c in ("model_variant", "seed") and len(vals) > 1:
                raise ValueError(f"scores holds several runs ({c} has {len(vals)} values); pass one run at a time")
            run[c] = _jsonable(vals[0]) if len(vals) else None
    return run


# ---------------------------------------------------------------- payload


def explorer_payload(
    scores: pd.DataFrame,
    *,
    components=None, curvature=None, events=None, works=None, authorships=None, labels=None, metrics=None,
    tables=None, cartels=None, weights=None, zstats=None, top_n: int = 500, radius: int = 1,
    title: str = "Citation anomaly explorer", data_note: str = "synthetic demo data", seed: int = 0,
    max_nodes: int = 40, flag_share: float = 0.01, n_rollup: int = 200,
) -> dict:
    """The JSON blob behind the page (see build_explorer for the arguments)."""
    missing_cols = [c for c in REQUIRED if c not in scores.columns]
    if missing_cols:
        raise ValueError(f"scores is missing required columns {missing_cols} (CONTRACTS.md scores file)")
    run = _single_run(scores)
    notes: list[str] = []
    missing = {k: "" for k in MISSING_TEXT}

    s = scores.copy()
    s["src_id"], s["dst_id"] = s["src_id"].astype(str), s["dst_id"].astype(str)
    s["event_id"], s["t"] = s["event_id"].astype(np.int64), s["t"].astype(np.int64)
    comp_extra: pd.DataFrame | None = None
    if components is not None:
        c = components.copy()
        c["event_id"] = c["event_id"].astype(np.int64)
        fill = [k for k in (*COMPONENTS, "cold_start") if k in c.columns and k not in s.columns]
        if fill:
            s = s.merge(c[["event_id", *fill]], on="event_id", how="left")
        extra = [k for k in c.columns if k not in s.columns]  # input columns the scores file does not carry
        comp_extra = c.set_index("event_id")[extra] if extra else None
    else:
        missing["components"] = MISSING_TEXT["components"]
    for comp in COMPONENTS:
        if comp not in s.columns:
            s[comp] = np.nan
            notes.append(f"{comp} is absent from the scores (ablation or missing input); it is shown as n/a.")

    ranked = rank_events(s)
    scorable = ranked[ranked["scorable"]]
    n_scored = len(scorable)

    # ---- frozen z statistics and weights (fallbacks disclosed on the page)
    stats = coerce_zstats(zstats)
    zsrc = "given"
    if not stats:
        stats, zsrc = fallback_zstats(ranked), "fallback"
        notes.append("Frozen z statistics were not given: z-scores use the mean and standard deviation of all scored "
                     "events in this file, not the training period, so contributions are approximate.")
    w = coerce_weights(weights)
    wsrc = "given"
    if not w:
        w, wsrc = {c: 1.0 / len(COMPONENTS) for c in COMPONENTS}, "fallback"
        notes.append("Combiner weights were not given: equal weights (0.25 each) are assumed for the contribution "
                     "bars; the stored total stays authoritative.")
    z = zscores(ranked, stats)
    contrib = pd.DataFrame({c: z[c] * w.get(c, 0.0) for c in COMPONENTS})
    gap = ranked["total"] - contrib.fillna(0.0).sum(axis=1)

    nodes = _Nodes(_works_info(works))
    if works is None:
        missing["works"] = MISSING_TEXT["works"]
    labels_map = None
    if labels is not None:
        labels_map = dict(zip(labels["event_id"].astype(np.int64), labels["label"].astype(int)))
    else:
        missing["labels"] = MISSING_TEXT["labels"]

    # ---- rows of the ranked list: the top_n ranked events plus cold-start events in the same score range
    top = scorable.head(top_n)
    cut = float(top["total"].iloc[-1]) if len(top) else math.inf
    cold = ranked[ranked["cold_start"] & np.isfinite(ranked["total"]) & (ranked["total"] >= cut)].head(top_n)
    sel = ranked.loc[sorted(set(top.index) | set(cold.index))]
    rows = {"event_id": [], "src": [], "dst": [], "t": [], "total": [], "rank": [], "cold": [], "label": [],
            "z": {c: [] for c in COMPONENTS}, "raw": {c: [] for c in COMPONENTS}}
    for i, r in sel.iterrows():
        rows["event_id"].append(int(r.event_id))
        rows["src"].append(nodes(r.src_id))
        rows["dst"].append(nodes(r.dst_id))
        rows["t"].append(int(r.t))
        rows["total"].append(num(r.total))
        rows["rank"].append(None if pd.isna(r["rank"]) else int(r["rank"]))
        rows["cold"].append(int(bool(r.cold_start)))
        rows["label"].append(None if labels_map is None else labels_map.get(int(r.event_id)))
        for c in COMPONENTS:
            rows["z"][c].append(num(z.at[i, c], 4))
            rows["raw"][c].append(num(r[c], 6))

    # ---- reference distributions
    hist = {"total": histogram(scorable["total"], bins=50), "components": {}}
    ref_sorted = {}
    for c in COMPONENTS:
        v = pd.to_numeric(scorable[c], errors="coerce").to_numpy(dtype=float)
        ref_sorted[c] = np.sort(v[np.isfinite(v)])
        hist["components"][c] = histogram(v)
    n_flags = flag_count(n_scored, flag_share)
    threshold = float(scorable["total"].iloc[n_flags - 1]) if n_flags else None

    # ---- ranking quality (labels)
    curves = None
    if labels_map is not None:
        lab = scorable.assign(label=scorable["event_id"].map(labels_map))
        lab = lab[lab["label"].notna()]
        y = lab["label"].astype(int).to_numpy()
        if len(lab) and 0 < y.sum() < len(y):
            curves = {"total": ranking_curves(lab["total"], y), "components": {},
                      "at": precision_at(y, (10, 100, 1000)), "n": int(len(lab)), "n_pos": int(y.sum()),
                      "unlabelled": int(n_scored - len(lab))}
            for c in COMPONENTS:
                curves["components"][c] = ranking_curves(pd.to_numeric(lab[c], errors="coerce"), y, max_points=150)
            for key in ("auroc", "ap"):
                curves["total"][key] = num(curves["total"][key])
                for c in COMPONENTS:
                    curves["components"][c][key] = num(curves["components"][c][key])
        else:
            missing["curves"] = "Labels hold no positives (or no negatives) among the scored events."

    # ---- D4 heatmap: field of the citing paper (t is its publication date) x year of t
    heat = None
    if works is not None and n_scored:
        fld = scorable["src_id"].map(lambda p: nodes.info.get(p, (None, None))[1])
        if fld.notna().any():
            heat = flag_heatmap(scorable.assign(field=fld)[["event_id", "t", "total", "field"]], flag_share)
        else:
            missing["heatmap"] = "The works table has neither a field column nor concepts."

    # ---- rollups (FR8)
    papers = authors = None
    if n_scored:
        long = incident_long(ranked)
        rank_of = dict(zip(ranked["event_id"], ranked["rank"]))
        ends = ranked.set_index("event_id")[["src_id", "dst_id"]]
        proll_all = paper_rollup(ranked, threshold, long=long)
        proll = proll_all.head(n_rollup)
        top_long = long[long["paper"].isin(set(proll["paper_id"])) & (long["k"] < 10)]
        by_paper = {p: g for p, g in top_long.groupby("paper", sort=False)}
        papers = []
        for r in proll.itertuples(index=False):
            evs = []
            for e in by_paper[r.paper_id].itertuples(index=False):
                src, dst = ends.loc[e.event_id]
                other = dst if e.role == "src" else src
                rk = rank_of.get(e.event_id)
                evs.append([int(e.event_id), nodes(other), 1 if e.role == "src" else 0, num(e.total),
                            None if rk is None or pd.isna(rk) else int(rk), int(e.t)])
            papers.append({"p": nodes(r.paper_id), "score": num(r.score), "n_above": int(r.n_above),
                           "n_events": int(r.n_events), "n_src": int(r.n_as_src), "n_dst": int(r.n_as_dst), "ev": evs})
        if authorships is not None:
            aroll = author_rollup(long, authorships, threshold).head(n_rollup)
            au = authorships[["work_id", "author_id"]].astype(str)
            au = au[au["author_id"].isin(set(aroll["author_id"]))]
            pscore = proll_all.set_index("paper_id")["score"]
            small = long[(long.k < 10) | (long.total >= threshold)]
            j = small.merge(au, left_on="paper", right_on="work_id").drop_duplicates(["author_id", "event_id"])
            j = j.assign(_k=-j["total"]).sort_values(["author_id", "_k", "t", "event_id"], kind="mergesort")
            by_author = {a: g.head(10) for a, g in j.groupby("author_id", sort=False)}
            au = au.assign(ps=au["work_id"].map(pscore)).sort_values(["author_id", "ps", "work_id"],
                                                                     ascending=[True, False, True], kind="mergesort")
            papers_of = {a: g["work_id"].head(5).tolist() for a, g in au.groupby("author_id", sort=False)}
            authors = []
            for r in aroll.itertuples(index=False):
                evs = []
                for e in by_author.get(r.author_id, pd.DataFrame()).itertuples(index=False):
                    src, dst = ends.loc[e.event_id]
                    rk = rank_of.get(e.event_id)
                    evs.append([int(e.event_id), nodes(src), nodes(dst), num(e.total),
                                None if rk is None or pd.isna(rk) else int(rk), int(e.t)])
                authors.append({"a": str(r.author_id), "score": num(r.score), "n_above": int(r.n_above),
                                "n_papers": int(r.n_papers), "papers": [nodes(p) for p in papers_of.get(r.author_id, [])],
                                "ev": evs})
    if authorships is None:
        missing["authorships"] = MISSING_TEXT["authorships"]

    # ---- event stream, curvature cache
    idx = None
    if events is not None:
        ev = events.copy()
        if "src_id" not in ev.columns and "src" in ev.columns:
            ev = ev.rename(columns={"src": "src_id", "dst": "dst_id"})
        idx = TemporalIndex(ev)
    else:
        missing["events"] = MISSING_TEXT["events"]
    curv_row, orc_hist = {}, {}
    if curvature is not None:
        cv = curvature.copy()
        cv["event_id"] = cv["event_id"].astype(np.int64)
        # rows are needed only for the detail events; escalated rows (a small share) feed the ORC edge colours
        curv_row = cv[cv["event_id"].isin(set(top["event_id"]))].set_index("event_id").to_dict("index")
        if "orc" in cv.columns:
            esc = cv[cv["orc"].notna()].sort_values(["t", "event_id"], kind="mergesort")
            for (a, b), g in esc.groupby([esc["src_id"].astype(str), esc["dst_id"].astype(str)], sort=False):
                orc_hist[(a, b)] = (g["t"].tolist(), g["orc"].tolist())
    else:
        missing["curvature"] = MISSING_TEXT["curvature"]

    # ---- per-event details for the top_n ranked events
    details = {}
    abs_f = []
    skip = {"scorable", "rank"}
    for i, r in top.iterrows():
        eid = int(r.event_id)
        d = {
            "z": {c: num(z.at[i, c], 6) for c in COMPONENTS},
            "contrib": {c: num(contrib.at[i, c], 6) for c in COMPONENTS},
            "raw": {c: num(r[c], 6) for c in COMPONENTS},
            "pct": {c: percentile_of(ref_sorted[c], r[c]) for c in COMPONENTS},
            "sum_gap": num(gap.at[i], 6),
        }
        inputs = [[k, _jsonable(r[k])] for k in ranked.columns if k not in skip]
        if comp_extra is not None and eid in comp_extra.index:
            inputs += [[f"components.{k}", _jsonable(v)] for k, v in comp_extra.loc[eid].items()]
        cr = curv_row.get(eid)
        if cr is not None:
            inputs += [[f"curvature.{k}", _jsonable(v)] for k, v in cr.items()]
            d["curv"] = {"forman": num(cr.get("forman")), "orc": num(cr.get("orc")),
                         "escalated": bool(cr.get("escalated")) if cr.get("escalated") is not None else None}
        else:
            d["curv"] = None
        d["inputs"] = inputs
        d["graph"] = d["timeline"] = None
        if idx is not None:
            g = local_subgraph(idx, eid, r.src_id, r.dst_id, int(r.t), radius=radius, max_nodes=max_nodes)
            loc = {n["id"]: k for k, n in enumerate(g["nodes"])}
            pos = _layout(len(g["nodes"]), [(loc[e["a"]], loc[e["b"]]) for e in g["edges"]], seed)
            gn = [[nodes(n["id"]), x, y, n["hop"], n["in_deg"], n["out_deg"]] for n, (x, y) in zip(g["nodes"], pos)]
            ge = []
            for e in g["edges"]:
                orc = None
                if e["scored"]:
                    forman = d["curv"]["forman"] if d["curv"] and d["curv"]["forman"] is not None else e["forman_t"]
                    orc = d["curv"]["orc"] if d["curv"] else None
                else:
                    forman = e["forman_t"]
                    hist_ab = orc_hist.get((e["a"], e["b"]))
                    if hist_ab:
                        k = bisect_right(hist_ab[0], int(r.t))
                        orc = num(hist_ab[1][k - 1]) if k else None
                abs_f.append(abs(forman))
                ge.append([loc[e["a"]], loc[e["b"]], e["n"], e["first_t"], e["last_t"], num(forman), orc,
                           1 if e["scored"] else 0])
            d["graph"] = {"nodes": gn, "edges": ge, "ring1_total": g["ring1_total"], "ring1_shown": g["ring1_shown"],
                          "ring2_total": g["ring2_total"], "ring2_shown": g["ring2_shown"]}
            d["timeline"] = received_timeline(idx, r.dst_id, int(r.t))
        details[str(eid)] = d
    big_gaps = [abs(v["sum_gap"]) for v in details.values() if v["sum_gap"] is not None and abs(v["sum_gap"]) > 1e-6]
    if big_gaps:
        notes.append(f"For {len(big_gaps)} of the top {len(details)} events the contributions w_c*z_c do not add up "
                     f"to the stored total (largest gap {max(big_gaps):.3g}); the stored total is authoritative.")

    # ---- cartel injection track (synthetic)
    cart = None
    if cartels is not None:
        members, event_ids = coerce_cartels(cartels)
        stream = events if events is not None else s
        inj = injected_events(members, event_ids, stream, labels_map)
        rank_of = dict(zip(ranked["event_id"], ranked["rank"]))
        tot_of = dict(zip(ranked["event_id"], ranked["total"]))
        ks = sorted({k for k in (10, 100, 1000, top_n) if k <= max(n_scored, 1)}) or [max(n_scored, 1)]
        inj = inj.assign(rank=inj["event_id"].map(lambda e: rank_of.get(e)))
        level_of = members.drop_duplicates("cartel_id").set_index("cartel_id")["level"]
        inj = inj.assign(level=inj["cartel_id"].map(level_of))
        cart = {"ks": ks, "recall": recall_by_level(inj["level"], inj["rank"], ks), "cartels": [],
                "n_injected": int(len(inj)), "metrics_levels": []}
        if isinstance(metrics, Mapping):
            for k, v in _flatten(metrics):
                if "level" in k.lower() or "obvious" in k.lower():
                    cart["metrics_levels"].append([k, _jsonable(v)])
        stream_s = stream[["src_id", "dst_id"]].astype(str)
        for cid, mg in list(members.groupby("cartel_id", sort=True))[:60]:
            mids = mg["work_id"].tolist()
            loc = {m: k for k, m in enumerate(mids)}
            inside = stream_s[stream_s["src_id"].isin(loc) & stream_s["dst_id"].isin(loc)]
            pair_n = inside.groupby(["src_id", "dst_id"]).size()
            cevents = []
            for e in inj[inj["cartel_id"] == cid].itertuples(index=False):
                rk = e.rank
                cevents.append({"e": int(e.event_id), "a": loc[str(e.src_id)], "b": loc[str(e.dst_id)],
                                "rank": None if rk is None or pd.isna(rk) else int(rk),
                                "total": num(tot_of.get(e.event_id)), "t": int(e.t)})
            edges = [[loc[a], loc[b], int(n)] for (a, b), n in pair_n.items()]
            pos = _layout(len(mids), [(a, b) for a, b, _ in edges], seed)
            cart["cartels"].append({
                "id": str(cid), "level": str(mg["level"].iloc[0]),
                "members": [[nodes(m), role, x, y] for m, role, (x, y) in zip(mids, mg["role"], pos)],
                "edges": edges, "events": cevents,
            })
    else:
        missing["cartels"] = MISSING_TEXT["cartels"]

    if not tables:
        missing["tables"] = MISSING_TEXT["tables"]
    if not metrics:
        missing["metrics"] = MISSING_TEXT["metrics"]
    if heat is not None:
        notes.append("Field of an event = field of the citing paper (its publication date is t).")

    t_src = events if events is not None else s
    t_min, t_max = int(t_src["t"].min()), int(t_src["t"].max())
    papers_all = pd.unique(pd.concat([t_src["src_id"].astype(str), t_src["dst_id"].astype(str)]))
    synthetic = "synthetic" in (data_note or "").lower()
    note = data_note or ""
    if cartels is not None and not synthetic:
        note = (note + "; " if note else "") + "contains synthetic cartel injection"
        synthetic = True
    meta = {
        "title": title, "data_note": note, "synthetic": synthetic,
        "n_events": int(len(t_src)), "n_papers": int(len(papers_all)),
        "events_source": "events" if events is not None else "scores",
        "n_works": None if works is None else int(len(works)),
        "n_scores": int(len(ranked)), "n_scored": int(n_scored), "n_cold": int(ranked["cold_start"].sum()),
        "cold_share": num(ranked["cold_start"].mean(), 4) if len(ranked) else None,
        "t_min": _date(t_min), "t_max": _date(t_max), "top_n": int(top_n), "radius": int(radius),
        "max_nodes": int(max_nodes), "seed": int(seed), "run": run,
        "threshold": num(threshold), "flag_share": flag_share, "n_flags": int(n_flags),
        "components": list(COMPONENTS), "weights": {c: num(w.get(c)) for c in COMPONENTS}, "weights_source": wsrc,
        "zstats": {c: [num(stats[c][0]), num(stats[c][1])] if c in stats else None for c in COMPONENTS},
        "zstats_source": zsrc, "notes": notes, "component_status": STATUS, "combiner_text": COMBINER_TEXT,
        "curv_scale": _curv_breaks(abs_f),
    }
    return {
        "meta": meta, "missing": missing, "component_text": COMPONENT_TEXT,
        "nodes": nodes.cols, "fields": nodes.fields, "venues": nodes.venues,
        "rows": rows, "details": details, "hist": hist, "curves": curves, "heatmap": heat,
        "tables": [_table_payload(k, v) for k, v in (tables or {}).items()],
        "metrics": [[k, _jsonable(v)] for k, v in _flatten(metrics)] if isinstance(metrics, Mapping) else [],
        "papers": papers, "authors": authors, "cartels": cart,
    }


# ---------------------------------------------------------------- HTML


def _asset(name: str) -> str:
    path = ASSET_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"explorer asset {path} is missing; when installing from a wheel, the package data "
            "citecurv/viz/assets/* must be included (pyproject [tool.setuptools.package-data])")
    return path.read_text(encoding="utf-8")


def render_html(payload: dict) -> str:
    """Assemble the page: template + inline CSS + inline JS + the JSON blob (all '<' escaped inside the JSON)."""
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    blob = blob.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    root = "cx-" + hashlib.sha1(blob.encode("utf-8")).hexdigest()[:10]
    meta = payload["meta"]
    note = html_lib.escape(meta["data_note"] or "")
    if meta["data_note"] and meta["synthetic"]:
        banner = (f'<div class="cx-banner cx-banner--synthetic" role="note"><span class="cx-banner__icon" '
                  f'aria-hidden="true">!</span><div><strong>Synthetic data: {note}.</strong> Every number on this '
                  'page comes from simulated inputs. It is not a measurement of real citations.</div></div>')
    elif meta["data_note"]:
        banner = f'<div class="cx-banner" role="note"><div>{note}</div></div>'
    else:
        banner = ""
    page = _asset("template.html")
    for marker, value in (("%%CSS%%", _asset("explorer.css")), ("%%JS%%", _asset("explorer.js")),
                          ("%%TITLE%%", html_lib.escape(meta["title"])), ("%%BANNER%%", banner),
                          ("%%ROOT%%", root)):
        page = page.replace(marker, value)
    head, tail = page.split("%%DATA%%")
    return head + blob + tail


def build_explorer(
    out_html, scores: pd.DataFrame, *, components=None, curvature=None, events=None, works=None, authorships=None,
    labels=None, metrics=None, tables=None, cartels=None, weights=None, zstats=None, top_n: int = 500,
    radius: int = 1, title: str = "Citation anomaly explorer", data_note: str = "synthetic demo data",
    seed: int = 0, max_nodes: int = 40, flag_share: float = 0.01, n_rollup: int = 200,
) -> Path:
    """Write the explorer page and return its path.

    scores: CONTRACTS.md scores file for ONE run (event_id, src_id, dst_id, t, s_*, total, cold_start, model_variant,
        seed, config_hash). Optional, each enabling a panel: components (extra input columns for NFR10), curvature
        cache, events (graph context), works (titles, field, venue), authorships (author rollup), labels (event_id,
        label), metrics (dict, e.g. ranking_metrics output), tables (name -> DataFrame), cartels (members DataFrame
        cartel_id/work_id/role/level, or {"members": df, "event_ids": [...]}), weights (component -> w), zstats
        (frozen z statistics; see precompute.coerce_zstats for accepted shapes).
    top_n: events with a precomputed detail view. radius: 1 or 2 hops of graph context; max_nodes caps the drawing.
    flag_share: share of ranked events counted as flags (D4 heatmap, rollup threshold). n_rollup: papers/authors kept.
    data_note: shown in the banner; any note containing "synthetic" gets the prominent synthetic-data banner.
    seed: layout seed; same inputs + seed -> byte-identical file.
    """
    payload = explorer_payload(
        scores, components=components, curvature=curvature, events=events, works=works, authorships=authorships,
        labels=labels, metrics=metrics, tables=tables, cartels=cartels, weights=weights, zstats=zstats, top_n=top_n,
        radius=radius, title=title, data_note=data_note, seed=seed, max_nodes=max_nodes, flag_share=flag_share,
        n_rollup=n_rollup,
    )
    out = Path(out_html)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_html(payload), encoding="utf-8", newline="\n")
    return out


def iframe_html(page, height: int = 900) -> str:
    """An <iframe srcdoc> wrapping the page, for notebook cells (Colab, Jupyter): CSS and ids stay isolated."""
    text = Path(page).read_text(encoding="utf-8") if isinstance(page, Path) or (
        isinstance(page, str) and not page.lstrip().startswith("<")) else page
    return (f'<iframe title="Citation anomaly explorer" height="{int(height)}" '
            f'style="width:100%;height:{int(height)}px;border:0" '
            f'sandbox="allow-scripts allow-downloads allow-popups" srcdoc="{html_lib.escape(text, quote=True)}">'
            "</iframe>")


def show(page, height: int = 900):
    """Display the page inside a notebook cell (imports IPython lazily)."""
    from IPython.display import HTML, display

    display(HTML(iframe_html(page, height)))
